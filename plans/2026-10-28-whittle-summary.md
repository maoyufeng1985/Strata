# Whittle-Qwen-3.8-45B-A3B 在 Strata 上：全流程记录（2026-10-28）

用户要求：把 `logic65/Whittle-Qwen-3.8-45B-GGUF` 的 Q4_K_M 用 **Flash-Next 的框架**跑起来，
套用 Q36 做过的优化方向（MTP、PREFILL、DECODE、多卡），无人值守做到最优速度。

**结果：跑通了，decode 43.9 tok/s、prefill 215 tok/s（275 token 提示）/ 467 tok/s（1799 token 提示），
输出连贯。** 下面按"做了什么 / 量到了什么 / 没做到什么"分开写。

> **续（第二轮，同日晚些时候）：P100 做主卡、三张 P104 做 helper、engram（PLE 表）放内存之后，
> 又做了一轮 decode 优化。** 最大的两笔：把 decode 换成 serve 模式并打开 `--remote-expert-opt`
> （helper 在自己卡上把专家输出归约成一个向量再回传），**42.1 → 47.2 tok/s（+12%）**；
> `--expert-cache-remote-placement layer` 再 +1.2%。详见第 9 节。

---

## 1. 移植（几何）

引擎的每个内核都按一个编译期形状编译（`include/strata/core/geometry.hpp` 的
`kCompiledGeometry`），模型文件的元数据必须与它一致，否则 `geometry_unsupported()` 在加载时拒绝。
Whittle 与 Flash-Next 的形状不同，所以**移植不是改个开关，是把整条引擎参数化**。

| 维度 | Flash-Next | Whittle-45B |
|---|---|---|
| n_embd | 2560 | **2048** |
| n_layers | 48 | **40** |
| n_head | 24 | **16** |
| n_ff / n_ff_shexp | 640 | **512** |
| ssm_v_heads | 48 | **32** |
| ssm_conv_channels | 10240 | **8192** |
| ssm_value_dim | 6144 | **4096** |
| n_expert / n_expert_used | 512 / 10 | **256 / 8** |
| head_dim, idx_key_dim, idx_q_heads, hc, hc_lr, ssm_state_size, ssm_k_heads, idx_block, n_rot | 256, 128, 4, 4, 320, 128, 16, 4, 64 | 不变 |

三个维度在两个模型里数值巧合相同（10240 = 4×2560 **也** = ssm_conv_channels；6144 = 16×256×1.5；
640 = n_ff **也** = 2×head_dim×1.25），所以照着"看起来一样的数字"改会悄悄改错。

改到的文件（`plans/2026-10-28-whittle-geom-port.md` 有逐文件表）：
`geometry.hpp`、`gguf_reader.hpp`、`prefill.cpp/kernels.cu/moe_fused_iq.cu`、`fused_gr.cu`、
`iq_kernels.cu`、`qsa.hpp`、`plan.hpp`、`remote_experts.hpp`、`ple.cu`、`ngram.hpp/cpp`、
`generate.cpp`、`verify.cpp`、`layer.cpp` 和 5 个 `tests/core/*`。

**PLE 是单独的一块**：表 39,040,000 行、每行一个 256 值的 **Q4_K** 块（144 字节），
哈希乘数 32473892291597 / 22010589867355 / 6403636407469，词表 8 × 4,880,000。
`include/strata/kernels/ngram.hpp` 的 `PleFormat` 增加了 `Q4_K` 与 `dequant_q4_k`。

## 2. 跑通路上修掉的三个真 bug

这三个都不是"配置不对"，是**按 Flash-Next 的形状写死、Whittle 一进来就越界或拒绝**：

1. **`prefill/kernels.cu` 的 `moe_combine` 把 top-k 写死成 10**（`for k < 10`、`slot[t*10+k]`）。
   Whittle 是 top-8，于是 `slot[]` 只分配了 `T*8`，内核读到 `t*10+8`、`t*10+9` ——
   `CUDA_LAUNCH_BLOCKING=1` 下报 `prefill moe_combine: an illegal memory access was encountered`。
   同样写死的有 `route_kernel` 的 `rank < 10`、两个 peer 求和内核（含 `peer_scatter_add` 的 `p / 10`）、
   `swiglu_pair`（`n * 640`）、`swiglu_il`（`n * 640` / `r * 1280`）。全部改成几何常量。
   其中 `swiglu_pair` **不越界却会少算**：`n * 640` 对 n_ff 512 是说 640 列，多读 128 列、
   而且布局按 640 解；`swiglu_il` 的 `r * 1280` 同理。这两处是 Whittle 上"出字但质量不对"的来源之一。
2. **`verify.cpp` 拒绝 `ss.k != 10`**（`verify: geometry differs from the artifact's`），
   而且 native router 的三个分支都钉死 `NE == 512 && K == 10`。
   改成从 `kCompiledGeometry` 取，并接到树里已有的 `native_router_topk(…, 256, 8)` 上。
3. **Pascal 上的 BF16 转换缓冲没有空间**：`--expert-cache auto` 会把显存吃到只剩几百 MiB，
   而 Pascal 走 `bf16_path() == 2`（FP32 转换）需要 `2*N*K` 字节。
   报错信息原来只说"没显存"，现在把**试到的最小 tile、cudaMemGetInfo 的真实返回**都打出来
   ——正是这条信息定位到了真正的故障（见下一节）。

## 3. 一个不是 bug 的"bug"：单行窗口

`generate.cpp` 的 spec 循环里，没有 drafter 时 `T = S_mtp = o.spec`。native pack 又强制
`--spec ≥ 2`，于是**每轮都白跑一行**。用 `STRATA_SPEC_T_CYCLE` 量出来：

| 窗口行数 | ms/轮 |
|---|---|
| T1 | 22.58 |
| T2 | 29.13 |
| T3 | 34.30 |
| T4 | 41.67 |

每多一行 5.2–6.5 ms，而 drafts 接受 **0/32**。改动：没有 MTP、没有 DFlash2、没有后缀/lookup
drafter、也没有 `--spec-oracle`/`--spec-follow` 时 `T = 1`。**32.4 → 40.3 tok/s（+24%）**，
这是本次最大的一笔。

## 4. 实测（P100，275 token 提示）

| 配置 | decode tok/s | prefill tok/s |
|---|---|---|
| 首版（交错填充的 profile）| 18.41 | 179.4 |
| 路由轨迹建的 profile | 29.84 | 213.6 |
| ＋单行窗口 | 36.28 | 214.2 |
| ＋三张 P104 专家缓存 | 39.57 | 204.3 |
| 同上，256 token | **43.94** | 215.0 |
| 修完 SwiGLU 的宽度后，256 token | 43.67 | 208.2 |
| 单卡，256 token | 41.29 | 214.8 |

（最后一行与"修完"那行差 0.6%，在噪声内；修正本身不改变速度，改的是正确性。）

prefill 随长度上升：275 → 215、699 → 340、1799 → 467 tok/s。
**四卡对 prefill 零收益**（340.5 vs 340.1、470.9 vs 472.3）：PCIe 只有 6.1 GB/s。

## 5. 没做到 / 做不到的

- **MTP**：Whittle 的 GGUF 里没有 MTP 块（只有 `blk.0..blk.39`，元数据没有 `nextn_predict_layers`）。
  只能靠无权重 drafter：`--suffix-draft` 在它上面接受 0 个 draft。⇒ **没有可用的投机解码**。
- **Quest**：模型的 `attention.indexer.top_k = 262144`，等于它自己的 context_length ⇒ 注意力恒为
  dense，选择核无对象。也正因为这个，`--max-context` 被限制在 32768（选择核单块的上限）。
- **DFlash2**：那是 Flash-Next 专用的小模型（hidden 2560、48 层），对 Whittle 无效。
- **`--peer-device`**：能用（33.8 tok/s），但比三张 helper 缓存慢，且两者互斥。
- **FP16 累加 GEMM / fused FP16 K-quant**：都能进（后者确认 `on the CPU 0`、无主机同步），
  但在这张 P100 上都是噪声内。

## 6. 复现

`docs/WHITTLE45B.md` 是给用户的一页；`plans/2026-10-28-whittle-howto.md` 是给工程师的一页
（构建命令、同步方式、每条参数为什么在那里）。

`.run/` 下的脚本（不进仓库）：`R-flash-rebuild.sh`（增量构建）、`R-flash-bench.sh`（配置矩阵）、
`R-flash-tokens.sh`（出字检查）、`R-flash-profile.sh`（`STRATA_VERIFY_PROFILE` 的逐阶段剖析）、
`R-mktok.py`（把文本切成 `--tokens-file`）、`R-flash-decode.py`（把 `output :` 行解回文本）。

## 7. 测试

盒子上能建的单元测试（`STRATA_BUILD_TESTS=OFF`，只有默认目标）：

| 测试 | 结果 |
|---|---|
| `file_expert_source_test` | PASS（修完 `H`/`FF` 的 fixture 之后）|
| `expert_profile_version_test` | OK |
| `pin_budget_test` | OK |
| `ple_reader_test` | "nothing to do"（没有输入文件，rc=2 不是失败）|

`expert_layout_test`、`gguf_geometry_test`、`model_family_test`、`native_dense_ple_key_test` 在这个
构建里**没有目标**（`STRATA_BUILD_TESTS=OFF`），改了源码但没编译过 —— 这是本次**未验证**的部分。

## 8. 测量纪律（踩到的坑）

- **测量前必须 `pgrep -x strata` 为 0 且 `loadavg < 4`。** 有一次一个 32K 上下文的进程没退干净，
  load average 顶到 55，之后那一臂的 prefill 只有 103.6 tok/s，decode 也偏低——差点被当成配置差异。
- **一臂一条命令、串行跑。** 31 GiB 机器上两个引擎会 OOM；而且 OOM 之后 shell 可能连 SSH 都卡住。
- **profile 必须用真路由轨迹**，且 `make_profile.py` 原来把 `48×512` 写死，已加 `--n-layer`/`--n-expert`。
- **不要在盒子上跑 `cmake --build . --target clean`。** 本次误跑一次，141 MB 的二进制和全部对象文件被删，
  分不清"测试为什么失败"就是因为这个（测试跑的还是旧源码、拿到的是新库）。全量重建 20 分钟。
  增量改动用 `R-flash-rebuild.sh`（只 `touch` 推过的文件）。
- **推送源码后要 `touch`**，否则 make 认为对象比源新；而且**测试源码也要推** —— 本次
  `file_expert_source_test.cpp` 在盒子上一直是旧的 `2560/640` fixture，报了一个看起来像引擎 bug 的错。
- **bash 的 `case` 每个分支都要写 `;;`**（这台机器的 bash 5.2 不接受靠换行结束分支；
  第一次写的扫描脚本在盒子上直接语法错，一臂都没跑）。
- **不要同时开两条链式任务**：一次 `final` 链和一次 `sweep3` 链撞车，7+9 个臂被"机器不空闲"跳过，
  而且 `pgrep -f 'R-flash-serve3'` 会匹配到**自己那条远端命令**，等待循环永远不退出 ——
  要用 `[R]-flash-serve3` 这种写法，并且用一个串行驱动器跑完所有臂。
- **serve 模式下引擎不一定在 QUIT 后立刻退出**：一次 `coldov3` 之后留下一个引擎进程，
  下一个臂被空闲门槛挡掉。空闲门槛要按"没有引擎进程"判断，loadavg 衰减太慢（放宽到 10）。

## 9. 第二轮优化（serve 模式 + `--remote-expert-opt`）

用户要求：「P100 做主卡，P104 做 helper，engram 放内存，再优化」。

### 9.1 前两条已经就位

- **P100 主卡 + 三张 P104 helper**：本来就是这个配置。注意 `--gpu` 收的是 **nvidia-smi 下标**，
  这台机器 P100 = 2、三张 P104 = 1/0/3，所以是 `--gpu 2,1,0,3`（不是 `--gpu 0`）。
- **engram 放内存**：这个模型的"记忆表"就是 **PLE 的 n-gram 表**（3904 万行 Q4_K，5.62 GiB），
  `--ple-io ram` 在启动时把它 **锁进内存**（日志：`PLE table locked in RAM (--ple-io ram) in 0.5 s`），
  每个 token 约 0.04 ms（提示的 276 个 token 一共 11.4 ms PLE），**不在关键路径上**。
  这一条本来就成立，本轮复核过。

### 9.2 最大的一笔：`--remote-expert-opt`（+12%）

`docs/SECOND_GPU.md` 里记着这个旗标：helper 在**自己卡上**把专家输出加权归约成**每 token 一个向量**
再回传，而不是把每个专家的 2048 个 float 都搬回来。旗标只在 `--serve` 下允许
（`generate.cpp:3371`："--remote-expert-opt requires --serve"），而 serve 模式就是桌面 app 和
Terminal 走的那条路（引擎读 stdin 的 `GEN <max_new> <ids>`，写 stdout 的 `T <id>`）。

276 token 提示、128 token 生成，P100 + 三张 P104：

| 配置 | decode |
|---|---|
| 不用 `--remote-expert-opt` | 42.1 tok/s |
| `--remote-expert-opt` | **47.2 tok/s（+12.3%）** |
| ＋`--expert-cache-remote-placement layer` | **47.8 tok/s（+13.6%）** |
| `--remote-expert-opt` 但**不要** helper 缓存 | 38.2 tok/s（**helper 值 +23%**）|

256 token 生成（固定开销摊薄，同一套提示）：

| 配置 | decode |
|---|---|
| 不用 `--remote-expert-opt` | 44.2 tok/s |
| ＋`--remote-expert-opt` ＋`layer` 放置 | **47.7 tok/s（+7.8%）** |
| 再加 `STRATA_GDN_CHUNKED=1` | **48.2 tok/s（+9.1%）** |

生成越长，`--remote-expert-opt` 的相对收益越小（128 token 时 +13.6%，256 token 时 +7.8%）：
它省的是**每层**的回传，而主卡专家缓存的选择（"主卡不收 helper 已经有的专家"）在长生成里
命中率会掉一点（提示阶段 reopt 臂 streamed 2529 个专家，ctl 臂 1218 个）。

机制（引擎自己打的日志）：

| | 不用 | 用 |
|---|---|---|
| helper 回传的字节（每个，128 token）| 72.8 / 73.2 / 77.8 MiB | **10.4 / 10.9 / 11.4 MiB** |
| 若不归约要回的字节 | — | 131.5 / 147.2 / 140.2 MiB |
| 主卡等 helper 的时间（128 token）| 72 + 75 + 76 = 223 ms | **25 + 44 + 60 = 129 ms** |
| 主卡专家命中率 | 99.1% | 100.0% |

输出连贯：三臂的文本都正常（`<think>` 起头、Markdown 表格、分点结构），但**token 流在第 10 个
就分叉** —— 这是旗标自己的性质（"It changes the floating-point summation order, so output is not
claimed to be bitwise identical"），和 helper 缓存的 rounding 差异同一类。

### 9.3 扫描过的旋钮（serve 模式，同一个基准 47.2）

| 旋钮 | 结果 | 结论 |
|---|---|---|
| `STRATA_HC_PERSIST=1` | 47.50 | **CUDA 侧是空实现**（`fused_gr.cu:3042` 的 `fused_gr_persist_init(){}`），只有 HIP 有这个核 |
| `STRATA_GDN_CHUNKED=1` | 47.82 | 噪声内（+1.3%） |
| `STRATA_GDN_CHUNKED=2` | 47.68 | 噪声内 |
| `STRATA_TSUM=1` | 47.67 | 噪声内 |
| `STRATA_ADAPT_OVERLAP=1` | 46.39 | 引擎自己说 "adaptive overlap: off: not with --remote-expert-opt" |
| `STRATA_COLD_OVERLAP=3` | 47.31 | 噪声内 |
| `--expert-cache 6000` / `3000` | 47.43 / 47.37 | 噪声内（10240 个专家本来就在四张卡上） |
| `STRATA_HC_UPMIX=1` / `STRATA_HCD_EXACT=1` | — | 只在 prefill 路径上 |
| `STRATA_HC_REQ8=1` | — | 只省显存；多卡下会退回 bf16 |

### 9.4 剩下的热点（下一步该动的代码）

serve + `--remote-expert-opt` 的一轮是 21.2 ms，按引擎自己的计数器拆：

| 项 | ms/轮 | 出处 |
|---|---|---|
| GPU（40 层的密集部分 + 专家 + 头）| ~15 | `verify.cpp:3100` 的 `ms_wait` |
| helper 的**主机侧 staging+launching** | 1.9 | `remote_experts.cpp:422` `begin()`：每层每个 helper 3 次 memcpy + 2 个 kernel |
| helper 的**等待** | 1.0 | `remote_experts.cpp:611` 每层每个 helper 一次阻塞 sync |
| 其余主机时间 | ~3.3 | 取样、argmax、图的发射 |

三处可以动，都**不是**旋钮能解决的：

1. **helper 的 staging**（1.9 ms/轮）：`begin()` 里 input、metadata、output 是三次独立的
   `cudaMemcpyAsync`，每个 (层, helper) 一次。合成一次（把 input 和 metadata 放在同一个 pinned 缓冲里）
   能省 600 次发射的一半。更大的做法是把 helper 的这一段捕获成 CUDA graph。
2. **CPU 池在主机线程上**（generate 模式下 3.45 ms/轮）：`verify.cpp:3059-3063` 里主机线程
   调 `pool(...)`，而 GPU 要等这一层的 plan 发布完才能跑 —— 主机在这里是串在关键路径上的。
   Q36 的 `STRATA_Q36_HELPER_ASYNC=1` 就是为这件事写的，但那是 Q36 专用引擎
   （`src/qwen35/q36_engine.cu`），Flash 这条树里没有。
3. **输出头**（2.23 ms/轮）：417 MB 的 Q5_K 头，每轮读全量。P100 的 HBM2 有 730 GB/s，
   而这个核只跑出 187 GB/s —— 瓶颈是逐块反量化的 ALU，不是带宽。一个 sm_60 的 packed Q5_K
   头核（照着 `q8_sm60.cuh` 的做法）大概能把 2.23 ms 压到 1.1–1.5 ms（省 3–5%）。

**没有多卡头的路**：树里没有分片头、没有 peer logits 归约、没有 `--head-device`；
helper 只算专家行，头只在主卡上（`native_head.cpp:92`）。


## 10. 第三轮：arena 内存、KV 流式、ROUTE_PRIOR（用户后续提问）

### 10.1 专家 arena 18.16 GiB 能不能精简 —— **能，而且几乎不花代价**

`--resident-experts` / `--mmap-experts` 把专家改成从 pack 的 blob 就地读（走文件缓存），不再拷
18.16 GiB 进常驻内存。实测（276 token 提示、128 token 生成、其余配置相同）：

| 配置 | decode | 提示 | 说明 |
|---|---|---|---|
| 默认（常驻 arena）| 47.29 tok/s | 1360 ms | 180 GiB 常驻 |
| `--resident-experts` | **47.19 tok/s** | 3109 ms | 内存降到 1 GiB 级 |
| `--mmap-experts` | 46.69 tok/s | 2886 ms | 最省内存的形态 |
| `--resident-budget-gib 8` | 47.12 tok/s | 2892 ms | 只留 8 GiB 的热专家 |
| `--resident-budget-gib 4` | 47.16 tok/s | 2886 ms | 只留 4 GiB |

**decode 掉不到 0.2%（噪声内），代价全在提示的第一次读（1.36 s → 2.9–3.1 s，+115%）。**
原因是四张卡的专家缓存已经装了全部 10240 个（layer, expert）对，decode 时几乎不读主机内存，
所以那 18.16 GiB 在 decode 上是白占的；而提示要一次性读大量专家，文件缓存路径就吃亏了。

⇒ **单引擎、短提示：保持默认。要在同一台机器上跑别的东西、或提示不重要：加 `--resident-budget-gib 8`。**

### 10.2 KV 放内存流式传输（`--kv-resident`）—— **这台机器上不划算**

KV 的账（40 层里 10 个 QSA 层，2 个 KV 头，head_dim 256，int8）：
`2*256*2 + 2*(256/64)*2*2 = 1056 B` 每 token 每层 ⇒ **10.3 KiB 每 token**。

| 上下文 | KV 总量（int8）| 占 8.5 GiB 专家缓存的比例 |
|---|---|---|
| 4096 | 41 MiB | 0.5% |
| 32768（本模型上限）| 330 MiB | 3.9% |
| 262144（模型原生）| 2.58 GiB | — |

`--kv-resident` 的下限是 **20480 格**，所以 4096 上下文根本触发不了它。在 32768 上实测：

| 配置 | decode |
|---|---|
| 32K 全部驻显存 | **48.71 tok/s** |
| 32K，`--kv-resident 20480` | 47.22 tok/s（**-3.1%**）|
| 32K，`--kv-resident 32768` | 48.59 tok/s（等于不流式）|

省下的 330 MiB 只够多放约 180 个专家槽（总数 4778），而每次 attention 要多走一趟主机内存。
**结论：上下文 ≤32K 时不要开；只有上下文上百 K、专家缓存真被 KV 挤掉时才值得。**

### 10.3 `STRATA_ROUTE_PRIOR`（v0.2.7 的新默认）—— 在我们这套配置上是**负收益**

v0.2.7 让"路由偏向已在卡上的专家"成为 12 GB NVIDIA 卡的默认（RTX 3060 上 +15%/+37%）。
这棵树的 `route_prior.cu` 早就存在，但 `verify.cpp` 把它钉死成 `NE == 512 && K == 10`，
所以 256/8 的模型从来吃不到。**本轮把门槛去掉、把 NE/K 改成启动参数**（`route_prior_topk`），
然后实测：

| `STRATA_ROUTE_PRIOR` | decode | 跨卡/PCIe 的专家（占 40960 次路由）|
|---|---|---|
| 关（基准）| **47.28 tok/s** | 8593（21.0%）|
| 0.5 | 45.82 | 3339（8.2%）|
| 1.0 | 45.59 | 1142（2.8%）|
| 1.5（v0.2.7 默认）| 45.20 | 334（0.8%）|
| 2.0 | 45.41 | 87（0.2%）|

它把跨卡专家从 21% 压到 0.8%，速度却掉了 4.4% —— **说明我们的瓶颈不是跨卡搬运**（那部分已经被
三张 helper 缓存和 `--remote-expert-opt` 消掉了），而是被偏置挤走的热专家反而落到慢路径上。
**在这套 4 卡 + helper 缓存的配置上，`STRATA_ROUTE_PRIOR` 保持关闭。**

这个否定结论有独立价值：它排除了"再往路由偏置上做文章"这一整条线。

### 10.4 本轮之后的最终配置（256 token 生成）

| 配置 | decode |
|---|---|
| serve + `--remote-expert-opt` + `layer` 放置 | 47.76 tok/s |
| ＋P100 的 7 个 sm_60 精确开关 | **48.56 tok/s** |
| ＋`STRATA_GDN_CHUNKED=1` | 48.57 tok/s（噪声内）|

那 7 个开关（`nv_defaults.cpp`，cc 6.0）在**纯 P100** 上值 +17.8%，但这台机器是 P100(6.0) +
三张 P104(6.1)，`generate.cpp:2850` 要求"所有可见卡算力相同"才套用，于是整张表被静默跳过。
手设即可，值 +1.7%。
