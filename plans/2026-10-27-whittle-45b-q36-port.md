# Whittle-Qwen-3.8-45B-A3B 上 Strata：口径与路线（2026-10-27）

## 0. 一句话

用户要求「用 Flash-Next 框架跑 Whittle-45B，套用 Q36 已做过的优化方向（MTP / prefill / decode / 多卡），做到最优速度」。
**先决问题不是速度，是它能不能跑起来**：这个文件是 `qwen4exp` 架构，但它的几何与 Strata 编译期的
`kCompiledGeometry` 有多处不符。下面是核对结果、硬阻塞和第一轮该做什么。

## 1. 模型是什么（读 HF 卡片，非实测）

- `logic65/Whittle-Qwen-3.8-45B-A3B-GGUF`，Apache-2.0，2026-10-04 更新，2282 次下载。
- **未训练的拼接体**：Whittle-3.8-35B（Phase-2 step 32010，180 experts/layer + 10B n-gram 记忆）
  + 从 Qwen3.6-35B-A3B 捡回被剪掉的 76 experts/layer ⇒ 每层 256 routed experts，**没有 heal**。
- 总 45.1B（35.1B 本体 + 10.0B n-gram 记忆表），激活 ~3B（8/256 + shared）。
- 架构 `qwen4exp`（= Qwen3.8-Flash-Next 格式），**stock llama.cpp 可跑**（作者只在 llama.cpp 上验证）。
- 卡片实测：HumanEval 75.6、MBPP 77.8、tool-use 96/96、GSM-50 **38（比 35B 的 44 差）**、
  MATH-60 44、PopQA 长尾名 111/252。作者明确写：算术应用题弱于 35B，要算术就用 35B。
- 量化：Q3_K_M 19.85 GiB / **Q4_K_M 25.24 GiB（卡片全部数字测的这个）** / Q5_K_M 29.75 / Q8_0 44.72。
- n-gram 表占 Q4_K_M 的约 5.6 GiB，卡片建议放系统 RAM（`-ot per_layer_token_embd=CPU`），
  GPU 只放 256-expert 本体（21.5 GiB）。

## 2. 盒子现状（实测，2026-10-27 22:30）

| | |
|---|---|
| GPU | **P100-12GB 现在枚举为 CUDA index 2**；P104-100 8GB × 3 为 index 0/1/3（顺序又变了，用前必查） |
| 显存合计 | 12 + 8×3 = 36 GiB |
| 主机内存 | 31 GiB 总 / 30 GiB 可用 |
| 磁盘 | /home 152 GiB，已用 34 GiB，**剩 118 GiB** |
| 已有模型 | Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf（22.85 GB） |

⇒ 25.24 GiB 的模型 + 5.6 GiB 的表 + KV + 引擎缓冲，**在 31 GiB 内存上很紧**，但比显存宽裕。

## 3. 硬阻塞：几何不符（**已用 HTTP Range 读 GGUF 头实测**，未跑推理）

只取文件头 16 MiB（`Range: bytes=0-16777215`）就解析出全部 55 条 kv + 1022 条张量信息，
不必等 25 GiB 下完。结果：**Whittle-45B 的本体就是 Qwen3.6-35B-A3B 的形状**，
只是套了 Flash-Next 的 gated residual（HC）+ PLE + indexer 张量。

`kCompiledGeometry`（`include/strata/core/geometry.hpp`）逐项比对：

| key | 编译期（Flash-Next） | **Whittle-45B 实测** | |
|---|---:|---:|---|
| `block_count` | 48 | **40** | 不符 |
| `embedding_length` | 2560 | **2048** | 不符 |
| `expert_feed_forward_length` | 640 | **512** | 不符 |
| `attention.head_count` | 24 | **16** | 不符 |
| `ssm.time_step_rank` | 48 | **32** | 不符 |
| `ssm.inner_size` | 6144 | **4096** | 不符 |
| `attention.indexer.top_k` | 2048 | **262144** | 不符 |
| `attention.compress_ratios` | [4,0,0,0]×12 | **40 项全 0** | 走占位符分支，OK |
| `expert_count` | 512 | 256 | **允许**（运行时自由）|
| `expert_used_count` | 10 | 8 | **允许**（运行时自由）|
| `full_attention_interval` | 4 | 4 | OK |
| `ssm.state_size / group_count / conv_kernel` | 128/16/4 | 128/16/4 | OK |
| `head_count_kv / key_length / rope.dimension_count` | 2/256/64 | 2/256/64 | OK |
| `indexer.head_count / key_length` | 4/128 | 4/128 | OK |
| `hyper_connection.count / low_rank` | 4/320 | 4/320 | OK（`hc_dim = 8192`，与实测张量吻合）|
| `rms_eps` | 1e-6 | 9.99999997e-7 | 在位精确容差内（1e-6 相对）|

张量清单（1022 条）确认三件事：
1. **HC 在**：`hc_attn_{down,up,inject,norm}` + `hc_ffn_*` × 40 层，形状 `[8192,320]` / `[320,8192]` / `[8192,4]` / `[8192]`。
2. **PLE 在**：`per_layer_token_embd.weight [256, 39040000]`（≈10.0 B 参数，Q4_K_M 里约 5.6 GiB）
   + `blk.1.ple_{key,value,conv1d,norm_*}`（只挂在第 1 层，`qwen4exp.ple.layers` 也只有一个）。
3. **indexer 在**：10 个 full-attention 层各 `indexer.{q,k}_{proj,norm}`；`q_proj [2048,512]` 正是 `4 heads × 128`。
4. 专家：`ffn_{gate,up}_exps [2048,512,256]`、`ffn_down_exps [512,2048,256]` ⇒ 256 experts、FF 512、top-8。

⇒ **它不是"另一个架构"，是 Qwen3.6-35B-A3B 的宽度 + Flash-Next 的机制。**
`qwen35moe` 分支会把 `hc`/`idx_*` 清零（`family_layout.hpp:71-74`），所以不能当 qwen35moe 读。

## 3b. 移植面的量级（读代码，未实测）

`\b2560\b` 在 `src/` + `include/` 的 **110 个文件**里出现。内核用 `static_assert` 把形状钉在
`kCompiledGeometry` 上，所以这不是改一个常数：
- 改成 `2048/40/16/512/32`（= Whittle 的 qwen4exp 形状）要重编整套 Flash-Next 内核。
- 或者把 Whittle 重标成 qwen35moe 并加回 HC/PLE/indexer —— 但 q36 引擎（`src/qwen35/`）根本没有
  HC/PLE/indexer 三条路径（它是按 `qwen35moe` 写的，`n_layer=40` 那套已经跑通）。

**两条路都要动内核，不是改参数。** 具体哪条更小，取决于 `src/qwen35/` 与 Flash-Next 内核的复用程度。

### 复用度（读代码实测）

| | |
|---|---|
| `src/qwen35/q36_engine.cu` | 9,111 行；`#include` 里有 `core/family_layout`、`kernels/fused_gdn`、`kernels/native_mmvq`、`prefill/moe_*` —— **复用底层内核**，但自己写层循环 |
| q36 里的 HC / PLE / indexer | **0 处**（`rg 'hc_\|ple_\|indexer'` 只命中一个叫 `hc_b` 的局部变量）|
| `src/core/layer.cpp`（Flash-Next 的层） | 1,638 行，**HC/PLE/indexer 都在这里** |
| `src/program/generate.cpp` | 17,091 行，Flash-Next 的主程序（MTP、PLE I/O、graph、batch 都在这）|
| `src/prefill/prefill.cpp` | 5,166 行，Flash-Next 的 prompt 路径（q36 另有自己的一套）|

⇒ **q36 引擎 = 一套精简的、只针对 qwen35moe 的层循环 + 复用的底层内核。**
它没有 HC/PLE/indexer 三条路径，而 Whittle 三者都有。
**Flash-Next 的 `generate.cpp`（17k 行）才有这三条路径**，但它钉死在 2560/48/24/640 上。

## 4. 两条路（必须先选）

### 路 A：重编 Flash-Next 内核到 Whittle 宽度（**推荐先试**）
把 `kCompiledGeometry` 改成 `{40, 2048, 16, 512, 32, 4096, ...}`，重编 `strata` 二进制。
- **优点**：HC/PLE/indexer/MTP/graph/batch 全部现成，`generate.cpp` 那 17k 行的机制直接用；
  Whittle 本来就是按 Flash-Next 格式导出的，张量名一一对应。
- **代价**：110 个文件里的 `2560` 硬编码要逐个核对（`static_assert` 会在编译时拦住漏网的），
  `ple_key [2048,8192]` 等 PLE 宽度也要跟着改。
- **风险**：`hc_dim = hc × n_embd = 4 × 2048 = 8192` 自动跟随，`[8192,320]` 的张量实测吻合，
  说明 HC 是**宽度无关**的。PLE 的 `ple_key [2048,8192]` 也随 `n_embd` 走。这是好信号。

### 路 B：把 Whittle 重标成 qwen35moe，在 q36 引擎里补 HC/PLE/indexer
- **优点**：q36 的 MTP / prefill peers / helper 多卡 / GQA 共享读 / KV int8 **全部现成**（这正是用户要的）。
- **代价**：要在 q36 的层循环里**新写** HC（gated residual）、PLE（n-gram 记忆表）、indexer（稀疏选块）
  三条路径。HC 是每层都有的残差结构，P*E 是每 token 查表，indexer 是 QSA 的选块——都不是小改。
- **风险**：q36 的 attention 是 dense（`k_attn_part`），加 indexer 等于把 Quest 那条线重做一遍。

### 判断

**先走路 A**：它把"能不能跑"变成"改一个 geometry struct + 修 110 处宽度硬编码"，
编译器会强制你改全；而且 HC/PLE/indexer 的实现已经存在，不用重写。
路 B 的 HC/PLE/indexer 是**新写**，工作量与风险都更大，且正好落在我们刚关掉的 Quest 那条线上。

**但路 A 有一个必须先确认的点**：`strata` 二进制的多卡/helper 路径（`--expert-cache-device1..3`、
`--peer-device`）在 Flash-Next 上是否可用。q36 的 helper 优化是在 `q36_engine.cu` 里写的，
`generate.cpp` 有自己的 `--expert-cache-device*` 与 `--peer-device`。**这一步要实测，不能按名字假设。**

## 5. 与 Q36 优化方向的对应

用户要求"套用 Q36 做过的优化方向（MTP / prefill / decode / 多卡）"。两条路各自能拿到什么：

| Q36 的优化 | 路 A（Flash-Next 重编） | 路 B（q36 引擎 + 补路径） |
|---|---|---|
| MTP | `generate.cpp` 的 `--mtp` / `--spec` / `--mtp-hnorm` / `--mtp-q4` 现成 | q36 的 `--mtp N` + `q_ev` 修复现成 |
| prefill | `--prefill CHUNK` + `--prefill-pipe` + helpers 的 prompt 路径 | q36 的 `--chunk` + `STRATA_Q36_PREFILL_PEERS` 现成 |
| decode | `--expert-cache N` / `--peer-device` / `--kv int8` / `--kv-resident` | q36 的 helper-parallel decode + GQA 共享读现成 |
| 多卡 | `--expert-cache-device{1,2,3}` + `--peer-device` + `--layer-split` | q36 的 1 主 + 3 helper 实测最优配置现成 |
| n-gram 记忆表 | `--ple-gguf` / `--ple-io ram` / `--ple-row-cache` **现成** | **要新写** |

⇒ **路 A 在多卡和 PLE 上明显更省事**；路 B 在"已经在 P100+3×P104 上调优过"上更省事。

### 移植可控性的关键证据（读代码实测）

内核不是散落硬编码，而是**统一对着 `kCompiledGeometry` 做 `static_assert`**：

```
src/kernels/cuda/qsa_decode_attn.cu:25  static_assert(HD == kCompiledGeometry.head_dim && G == ...n_head/n_head_kv)
src/kernels/cuda/fused_gr.cu:39         static_assert(N == ...n_embd && HC == ...hc && LR == ...hc_lr)
src/kernels/cuda/s2_expert_grouped.cu:42 static_assert(H == ...n_embd && FF == ...n_ff)
src/kernels/cuda/native_ple_postops.cu:37 static_assert(N == ...n_embd && H == ...hc)
```

⇒ **把 `geometry.hpp` 里那几个默认值改成 `{40, 2048, 16, 512, 32, 4096, ...}`，编译器会把每一处必须
跟着改的 kernel 逐条报出来。** 这是"改一处 + 跟着编译错误走"，不是 110 个文件地毯式搜索。

只有一处是例外，写死了字面量而不是 `kCompiledGeometry`：
`src/kernels/cuda/ple.cu:238  static_assert(NG_N_EMBD == 2560 && NG_HC_DIM == 10240, "native PLE key geometry changed")`
—— 这里 `NG_HC_DIM = 10240 = 4 × 2560`，改宽度时这里要手动改成 `4 × 2048 = 8192`。

**这就是为什么先走路 A：它有编译器兜底。**

### 第一轮实际改了什么（MEASURED：编译真的跑过）

按"改一处 + 跟着编译错误走"，实际只改了 **7 个文件、9 个常数**：

| 文件 | 改动 |
|---|---|
| `include/strata/core/geometry.hpp` | 编译期默认值改成 Whittle 的 8 项（2048/40/32/8192/4096/16/512/512）|
| `include/strata/kernels/cpu/expert.hpp` | `H 2560→2048`、`FF 640→512`、`NE 512→256`（它自带 Q2_0 blob 布局，独立写死）|
| `src/kernels/cuda/s2_expert_grouped.cu` | `H 2560→2048`、`FF 640→512` |
| `src/kernels/cuda/qsa_decode_attn.cu` | `G 12→8`（query heads per KV head = 16/2）|
| `src/kernels/cuda/qsa_prompt_attn.cu` | 同上 |
| `src/kernels/cuda/native_ple_postops.cu` | `N 2560→2048` |
| `src/kernels/cuda/fused_gr.cu` | `N 2560→2048` |

**第二轮编译的 5 个错误正好对应这 5 个 kernel 文件**，没有出现别的架构性问题。
`head_dim 256`、`hc 4`、`hc_lr 320`、`indexer 4×128`、`rope 64`、`qsa_interval 4`
**全部不用改** —— 与第 3 节实测的元数据吻合。

⇒ 到目前为止这是**一个 9 行常数级的移植**，不是重写。剩下的风险在运行期，不在编译期。

### 编译结果（MEASURED 2026-10-27 22:55）

四轮编译，错误从 5 → 2 → 1 → **0**：

| 轮 | 错误数 | 内容 |
|---|---:|---|
| 1 | 5 | 5 个 kernel 各自的 `static_assert`（宽度写死）|
| 2 | 2 | `qsa_decode_attn.cu:431` 的 12 路 splat 初始化；`fused_gr.cu:810` 的 `H_TILE=1280` 不整除 2048 |
| 3 | 1 | `remote_expert_opt.cu:16` 的「8 行窗口 + 2560 宽」联合断言（纯文档性，下面两个是恒等式）|
| 4 | **0** | `[100%] Built target strata` |

**全部改动（9 个文件、12 个常数）：**

```
include/strata/core/geometry.hpp            n_embd 2560→2048, n_layers 48→40, ssm_v_heads 48→32,
                                            ssm_value_dim 6144→4096, n_head 24→16, n_ff 640→512,
                                            n_ff_shexp 640→512, ssm_conv_channels 10240→8192
include/strata/kernels/cpu/expert.hpp       H 2560→2048, FF 640→512, NE 512→256
src/kernels/cuda/s2_expert_grouped.cu       H 2560→2048, FF 640→512
src/kernels/cuda/qsa_decode_attn.cu         G 12→8, splat 由 3×float4 改为 2×float4
src/kernels/cuda/qsa_prompt_attn.cu         G 12→8
src/kernels/cuda/native_ple_postops.cu      N 2560→2048
src/kernels/cuda/fused_gr.cu                N 2560→2048, H_TILE 1280→1024
src/core/remote_expert_opt.cu               H 2560→2048 的联合断言
```

**没有改**：`head_dim 256`、`hc 4`、`hc_lr 320`、`indexer 4×128`、`rope 64`、
`qsa_interval 4`、`expert_used_count 8`、`n_vocab 248320`。
这些与第 3 节从 GGUF 头读到的元数据**逐项吻合**，所以一个都不用动。

⇒ **编译期不再是阻塞。** 下一步是运行期：这个二进制能不能真的把这个文件读进来并出 token。

## 7. 下一步（等模型下完）

1. `strata generate --tokens <ids> --max-new 8` 直接试加载。**先不建 pack**：
   `strata` 有"每个张量按 GGUF 原样读"的路径，加载会慢但能立刻验证几何。
2. 若加载报错，按报错改（这是运行期几何，与编译期不同）。
3. 加载通了，再跑 `tools/iq_pack.py` 生成 pack（让专家走优化路径）。
4. 最后才是速度：MTP / prefill / decode / 多卡。

**下载进度**：25.24 GiB，截至 22:58 已下 7.32 GiB（约 24 分钟，~5 MB/s）。

## 8. 一个把路 A 定死的事实（读代码实测）

`src/program/generate.cpp:2398-2415` 是**家族切换**：`strata` 二进制的 `main` 会先读
`--model`/`--native` 那个文件的 `general.architecture`：

```cpp
qwen35 = strata::model_family(meta) == strata::ModelFamily::Qwen35Moe;
if (qwen35) return strata::qwen35::run(argc, argv);   // ← 切到 q36 引擎
```

Whittle 的 `general.architecture = qwen4exp`（第 3 节实测），所以它**走不到 q36 引擎**，
只能走 Flash-Next 路径 —— 而那条路径刚刚被我们改成 2048 宽。

⇒ **两个选择被数据定死了**：
- 想用 q36 引擎（MTP/helper 多卡/GQA 现成）⇒ 必须**改写 GGUF 的架构字符串**为 `qwen35moe`，
  但那样 `check_architecture_qwen35moe` 会要求一整套 qwen35moe 键，且 q36 引擎没有 HC/PLE/indexer。
- 想保留 HC/PLE/indexer ⇒ **必须走 qwen4exp**，也就是路 A。

**Whittle 的 HC/PLE/indexer 是它的核心**（10B 记忆表 + gated residual），丢掉就不是这个模型了。
所以 **路 A 是唯一正确解**，不是"先试试"。

`strata` 二进制本身**同时链接了 q36 引擎**（`CMakeLists.txt:797`），所以将来若要两族都能跑，
是同一个二进制按文件头分流——但 Whittle 永远走 qwen4exp 那半边。

## 9. 第一次运行的容量账（算出来的，未实测）

| | GiB |
|---|---:|
| Whittle Q4_K_M 文件 | **25.24**（27,096,837,024 B，HTTP Content-Length 实测）|
| — 其中 256-expert 本体 | ~21.5（HF 卡片）|
| — 其中 PLE / n-gram 表 | ~5.6（HF 卡片；`per_layer_token_embd [256, 39040000]`）|
| 主机内存 | 31 |
| 显存合计 | 36（P100-12 + P104-8 × 3）|

**结论：**
- 文件本身装得进 31 GiB 内存（还剩 ~5 GiB），但**装不下"文件 + 引擎缓冲 + KV"**。
- 21.5 GiB 的本体**远超单卡 12 GiB**，必须走 expert cache（显存）+ CPU pool（内存）。
- PLE 表 5.6 GiB 按卡片建议**放内存**（`--ple-io ram`），不要再占显存。

**首次运行参数（计划）：**
```
strata generate --pack <pack> --tokens <ids> --max-new 8 --max-context 2048 \
  --gpu 2 --kv int8 --expert-cache <N> --ple-io ram
```
`--gpu 2` 是当前的 P100（序号又变了，跑前必须 `nvidia-smi` 复核）。
若内存不够，退到 `--mmap-experts`（让 OS 页缓存管专家，不常驻）。

## 10. 第二个硬阻塞：CPU 太老（MEASURED）

第一次真正启动时被拒绝（`whittle-20261027/load.err`）：

```
strata generate: this CPU (Intel(R) Xeon(R) CPU E5-2650 v2 @ 2.60GHz) does not support
AVX2 with FMA and F16C, which this engine's CPU expert kernels need; ...
Older CPUs are EXPERIMENTAL and slow: setup compiles an engine for them on this PC
(STRATA_ISA_FLOOR, see "Older CPUs" in docs/INSTALL.md)
```

`lscpu` 实测：这颗 Ivy Bridge（2013）有 **`avx`、`sse4_2`、`f16c`，没有 `avx2`、`fma`**。
检查在 `src/program/generate.cpp:3432`，条件 `!cpu_avx2_ok() && isa_floor[0]=='\0'` ⇒ 只在
**AVX2 floor 构建**上拒绝。

**所以问题不是"跑不了"，是"这份二进制是按 AVX2 编的"。**
`CMakeLists.txt:39` 有 `STRATA_ISA_FLOOR`：空（默认 AVX2）/ `avx`（SSE4.2+AVX）/ `none`（SSE4.2）。
这颗 CPU 支持 `avx` ⇒ 用 `-DSTRATA_ISA_FLOOR=avx` 重编一份即可。

**代价**：CPU 专家池（`--expert-cache` 之外的 CPU 路径）会慢，而 21.5 GiB 的本体在 36 GiB 显存上
**必然有一部分专家落在 CPU**。所以这条路能跑，但"最优速度"要重新评估：
CPU 侧吞吐会成为短板，可能会推翻第 9 节"CPU pool 兜底"的假设。

**做法**：编到独立目录 `/home/Strata/build-avx`，**不动**现有的 `/home/Strata/build/strata`
（那是 AVX2 版，将来在支持 AVX2 的机器上仍然有用）。

### 10b. 第一次配置漏了 CUDA（MEASURED）

`/home/Strata/build-avx` 第一次配置出来**没有 `strata` 目标**：

```
BUILD_RC=2
gmake: *** 没有规则可制作目标"strata"。 停止。
```

原因不是代码：`STRATA_ENABLE_CUDA` 的默认值是 **OFF**（`CMakeLists.txt:69`），第一次只给了
`-DSTRATA_ISA_FLOOR=avx`，于是只生成了纯 CPU 的目标（`strata-gguf`、`strata-plan`、`ggml*`）。
对照主构建的 `CMakeCache.txt` 补齐参数后重配：

```
cmake -S /home/Strata/src -B /home/Strata/build-avx -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ISA_FLOOR=avx -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_ARCHITECTURES="60;61"
```

⇒ `-- Strata: CUDA enabled, arch 60;61`，`strata` 目标出现，**711 s / RC=0**，
产出 `/home/Strata/build-avx/strata`（138,807,656 B，比 AVX2 版 184 MB 小，因为 ggml 只编了 avx 一份）。

启动后的前两行确认两件事都生效了：

```
strata generate: EXPERIMENTAL older-CPU build (ggml-cpu for avx): this CPU has no AVX2 ...
strata: sm_60 (Pascal): 7 exact speed switches on by default (...)
```

### 10c. 第一个运行期阻塞：20 层 down 是 Q4_K（MEASURED）

第二次启动，CPU、GPU、几何三关全过，停在专家格式：

```
strata generate: layer 5's experts are Q4_K/Q4_K (ggml types 12/12), which this engine has no GPU kernels for
```

`native_experts.txt` 实测（40 行）：

| 类型对 | 层数 |
|---|---:|
| gate/up Q4_K + down Q6_K（12/14） | 20 |
| gate/up Q4_K + down Q4_K（**12/12**） | **20** |

`iq_kernels.cu:773` 的 `STRATA_D_FMTS` **没有 12**：注释自己写着 "Q5_K / Q6_K downs: ... Flash-Next's FF 640
never selects them"。Flash-Next 的 `n_ff=640` 与 256 不整除，K-quant down 在那套模型上根本选不到，所以当初
只加过 13/14。Whittle 的 `n_ff=512`（= 2×256 块）把这条路打开了。

**这不是缺内核**：down 是 `Fmt<TY>` 模板，`Fmt<12>` 早就存在（gate/up 用的同一个
`vec_dot_q4_K_q8_1`），且 `kSplit<12>=true` 但 `split_off<12>()` 为真（`STRATA_KQ_SPLIT` 默认关），
所以加进表里走的是通用 `native_down_kernel<12>`，**零新内核**。

改动：`STRATA_D_FMTS` 两处各加一个 `X(12)`（`STRATA_Q6K_EXPERTS` 开/关两支）。

prompt 路径不用改：MMQ 的 `supported(Q4_K)` 在 `STRATA_MMQ_KQUANTS` 下走 `kq_ok()`（默认关），
退回 FP16 路径的 `dq_dispatch` 已有 `case 12: dq_q4_k`；fused 路径的 `d_covered` 不含 12，
于是 `fused_ring()` 对这份 pack 返回 false，**保留 MMQ 的缓冲区尺寸**，不会溢出。


## 4. 两条路（必须先选）

### 路 A：port 到 Strata（"用 Flash-Next 的框架跑"）
要先把 Whittle GGUF 转成 pack（`tools/strata_pack.py build --gguf ... --out ...`），
再让 `kCompiledGeometry` 与它的形状一致。工作量取决于第 3 节那几项差多少：
- 若只有层数差（48→40）：改 `n_layers` 一处 + 重编，但 48 层的图/表很多地方硬编码，需逐个查。
- 若宽度/head 也差：等于换一套内核形状，**是周级别**，不是小时级。

### 路 B：用 llama.cpp（作者验证过的路径）
`llama-server -m ... -ngl 99 -c 16384 --jinja -fa on -ot per_layer_token_embd=CPU`。
但这**不是 "Flash-Next 框架"**，也没有 Q36 那套 MTP/prefill/decode/多卡优化。
用户说的是「用 flash-next 的框架」，所以路 B 只在路 A 被证明不可行时才提。

## 5. 第一轮该做的（不猜）

1. **等模型下完**，用 `strata-gguf` / `strata_inspect.py` 读它的真实几何（第 3 节那几项），
   把「待读」全部填上。
2. 用 `strata-plan`（若支持 qwen4exp）或 `strata generate --tokens` 直接试加载，
   拿**真实的拒绝信息**（哪一项不符），而不是靠读代码推断。
3. 按真实差值决定：一项（层数）还是多项（宽度/head）。
4. **只有第 3 步给出「可实现」时**才动内核；否则把结论写回来，不硬改。

## 6. 不做的事

- 不把 Qwen3.6 的 MTP 接受率 / 逐位衰减 / 校准结论套到 45B 上（记忆里明确：换模型必须重测）。
- 不在报出速度前就声称"最优"。作者自己说这模型算术弱、是实验体。
- 下载没完成前不启动任何引擎（一台盒子只能一个引擎）。
