# Whittle-45B：把 Q36 的优化方向搬过来（2026-10-28）

## 0. 前提

Q36（Qwen3.6-35B-A3B）的 105 个提交里，大部分在 `src/qwen35/q36_engine.cu`（Q36 专用引擎），
但**共享内核**里的那几项，Flash-Next 路径直接受益。逐条对照：

| Q36 的方向 | 在共享树里？ | 对 Whittle 的意义 |
|---|---|---|
| GQA 共享的 decode attention（3x 注意力，+50% decode @32K） | `src/kernels/cuda/qsa_decode_attn.cu` ✓ | **直接可用**（n_head 16 / n_head_kv 2 ⇒ G=8）|
| Quest（训练无关的 decode 稀疏选择） | `qsa_select.cu` + 引擎开关 | Whittle 的 indexer 预算 = 上下文（262144）⇒ **恒为 dense，Quest 无对象**。除非改成真稀疏 |
| FP16 累加的 dense GEMM（Pascal） | `src/prefill/gemm.cu` | 盒子上 3 张 P104 是 sm_61 ⇒ **Pascal 路径** |
| fused FP16 K-quant 专家（Q4_K gate/up） | `src/prefill/moe_f16_kq*.cu` | **Whittle 的 gate/up 全是 Q4_K（type 12）** ⇒ 命中 |
| Q5_K/Q6_K down 的 fused FP16 | 同上 | Whittle 有 20 层 down = Q6_K（type 14）⇒ 命中 |
| peer prompt 路径（多卡 prefill） | `src/prefill/prefill.cpp` 的 `PeerPrefill` ✓ | 盒子上 4 张卡 ⇒ 可用 |
| int8 KV / --kv | 共享 ✓ | 已有 |
| MTP | **Whittle 没有 in-file MTP 块**（实测：GGUF 只有 blk.0..blk.39，无 blk.40；元数据里没有 `nextn_predict_layers`）| 只能靠外部 drafter：`--suffix-draft N`（无权重、纯文本，与模型无关）或 DFlash2（Flash-Next 专用的小模型，对 Whittle 无效）|

## 1. 已经白捡的两项

1. **native router**：树里已有 `native_router_topk<NE,K>`，注释写明是给
   "Qwen3.6-35B-A3B's 256 / 8" 准备的 —— **正是 Whittle 的形状**。
   但 `src/core/layer.cpp:420` 仍在用旧的钉死路径 `g.n_expert == 512 && k == 10`。
   ⇒ 把 decode 的路由换成 `native_router_topk(logits, ids, weights, g.n_expert, k, 1, stream)`。
   泛型 `router_top10` 要对 256 个专家做全排序，fused 的是一遍 warp 归约。
2. **`qsa_real_shapes()`** 原来钉死 n_head=24，已改成从 `kCompiledGeometry` 取。

## 2. 待办（按收益/风险排序）

| # | 事项 | 文件 | 备注 |
|---|---|---|---|
| A | 接上 `native_router_topk`（256x8） | `src/core/layer.cpp:420,446` | 小改，先测后上 |
| B | flash-next 路径是否走 fused FP16 K-quant | `src/prefill/prefill.cpp` | Whittle gate/up 是 Q4_K |
| C | cuBLAS 调优表 + split-K 启发式全是 Flash-Next 的形状 | `src/prefill/gemm.cu:858-880,934-936` | 重新测 |
| D | S26 专用专家核（`n_embd==2560 && n_ff==640` 才走） | `src/kernels/cuda/iq_kernels.cu:5158` | Whittle 走通用核 |
| E | v2k 路径（`d_type==7/8 && n_embd==2560`） | 同上 | 同上 |
| F | 多卡 prefill（peer）实测 | `src/prefill/prefill.cpp` | 4 卡，PCIe gen1 x4 |
| G | FP16 累加 GEMM 在 P104 上开 | `src/prefill/gemm.cu` | Pascal |
| H | Quest：Whittle 的 indexer 预算=上下文，恒 dense | `qsa_select.cu` | 要真稀疏得改模型预算 |
| I | **prompt attention 的 16 行 MMA 有 2 倍空转**：内核按 16 行（`gid` 与 `gid+8`）做 m16n8k16，G=12 时 12 行真 4 行补零；**G=8 时 8 行真 8 行补零** ⇒ 一半的 MMA 白做 | `src/kernels/cuda/qsa_prompt_attn.cu` | Whittle 的 n_head_kv=2、G=8 ⇒ 一个 block 正好能装下两个 KV 头的 16 行（`grid.y` 从 n_head_kv 改成 1）|

## 3. 验收口径

每个改动都要：**改前/改后同一天、同机器、同提示**的数；跑不动的（缺 nvcc/AVX2）不写进结论。

## 4. 实测（2026-10-28，P100 单卡，275 token 提示，32 token 生成）

引擎已能出字（`<think>` 起头、连贯）。下面的数都是 `R-flash-bench.sh` 的 `decode` 行。

| 配置 | decode tok/s | prefill tok/s | 说明 |
|---|---|---|---|
| 首版（profile 是交错填充）| 18.41 | 179.4 | CPU 每层 3.25 个专家 |
| `--expert-cache 512` | 10.22 | 144.5 | 槽太少，CPU 分掉更多 |
| `--spec 3` | 14.94 | 180.8 | 窗口变宽但没有 draft 被接受，纯亏 |
| `--spec 2 --suffix-draft 0`（旧 profile）| 20.13 | 177.0 | 后缀 drafter 全落空 |
| **路由轨迹建的 profile** | **29.84** | 213.6 | CPU 每层 1.36 个专家；**+62%** |
| 同上 + `--suffix-draft 0` | 31.12 | 203.1 | |
| 同上 + 三张辅助卡专家缓存 | 31.19 | 202.8 | CPU 专家 1.36 → 0.05，但 decode 只 +4.5% |

**这一条最重要**：profile 必须是真路由轨迹建的。交错填充的 profile 命中率低得离谱
（CPU 每层 3.25 个未驻留专家 vs 1.36），一换就是 +62%。

三张辅助卡专家缓存把 CPU 的活几乎清零（0.05 专家/层），但 decode 只涨 4.5% ——
说明瓶颈已经从 CPU 池转到了 PCIe/主卡本身。

**MTP 事实**：`--spec T` 的窗口里 drafts 接受率 0/32（32 轮全零）。Whittle 没有 in-file MTP 头，
`suffix drafter` 在这里也全落空（提示是重复段落，但模型输出与提示不重合）。
⇒ 目前这条路径上**没有可用的投机解码**，`--suffix-draft 0` 就是最优。

## 5. 单行窗口（本次最大的一笔）

`generate.cpp` 里，没有 drafter 时 `T = S_mtp = o.spec`：native pack 强制 `--spec ≥ 2`，
于是每轮都白跑 `T-1` 行——多行的 argmax 全被丢掉，只留第一行。用 `STRATA_SPEC_T_CYCLE` 量出的单价
（P100，三辅助卡缓存，同一个提示）：

| 窗口行数 | ms/轮 |
|---|---|
| T1 | 22.58 |
| T2 | 29.13 |
| T3 | 34.30 |
| T4 | 41.67 |

⇒ 每多一行 **5.2–6.5 ms**，而 drafts 接受 0 个。改动（`generate.cpp` 的 spec 循环）：
没有 MTP、没有 DFlash2、没有后缀/lookup drafter、也没有 `--spec-oracle`/`--spec-follow` 时，`T = 1`。

| 配置 | decode tok/s |
|---|---|
| 三辅助卡 + `--suffix-draft 0`（前）| 32.38 |
| **同上 + 单行窗口** | **40.28** |
| 单卡 + `--suffix-draft 0`（前）| 31.12 |
| 单卡 + 单行窗口 | 36.83 |

**+24%**。`--spec` 仍然只决定捕获的图宽和缓冲区大小。

## 6. 收尾的配置矩阵（干净机器，无残留进程）

32 token 生成，275 token 提示：

| 配置 | decode tok/s | prefill tok/s |
|---|---|---|
| P100 单卡 + 单行窗口 | 36.28 | 214.2 |
| P100 + 三 P104 专家缓存 + 单行窗口（`best4`）| **39.57** | 204.3 |

256 token 生成（长一点，固定开销摊薄）：

| 配置 | decode tok/s |
|---|---|
| P100 单卡 | 41.29 |
| **P100 + 三 P104**（`bestlong`）| **43.94**（+6.4%）|

长生成比短生成快 8–11%（图的捕获、profile 预填、PCIe 预热的固定开销）。

### 辅助卡专家缓存的收益随生成长度变化

同一提示（700 token 文档），64 token 生成：

| 卡 | decode tok/s | CPU 未驻留专家/层 |
|---|---|---|
| P100 单卡（4795 槽）| 32.47 | 1.07 |
| P100 + 三 P104（4778 + 3×1820 槽）| **37.96**（+17%）| 0.05 |

256 token 时只有 +6.4% —— 短生成里预填/捕获的固定开销被摊得更薄，缓存的差距显得更大。

### 输出质量

4 卡与单卡的 token 在第 9–17 个分歧（辅助卡与主卡的 rounding 不同，这是已知的、
文档里写过的 `bench/results/2026-09-27-cache-parity` 行为），两边都连贯：
在 700 token 的 DETAILS.md 提示下都正确接了 Markdown 表格
（`| Model | Speed | Notes |` / `| Qwen 2.5 7B | 100+ tokens/s |`）。

## 7. 查过的、没用的

| 项 | 结果 |
|---|---|
| `STRATA_PF_F16_CUDA=1`（fused FP16 K-quant）| 能进（`on the CPU 0`、无主机同步），但 prefill 214 vs 191 是噪声；decode 36.75 vs 36.51 |
| `STRATA_DENSE_F16_ACC=1`（Pascal FP16 累加 GEMM）| decode 36.95 vs 36.87、prefill 213 vs 201 —— 噪声内 |
| `--gpu` 四卡跑 prefill | 699 token：340.5 vs 单卡 340.1；1799 token：470.9 vs 472.3 ⇒ **无收益** |
| `--peer-device 1` | 33.80 tok/s（单卡 36.28 更慢? 实际是 peer 与缓存层的组合），与 `--expert-cache-device*` 互斥 |
| `--kv k8v4` / `--kv int8` | 40.32 / 39.78，打平；用 int8（KV 更小）|
| `STRATA_SPEC_COST=1`（成本感知窗口）| 反而更慢（22.7），因为它仍取 T4 并信那张空表 |
| `--spec 1` | 引擎拒绝（native pack 要求 ≥2）；已由单行窗口改动覆盖 |

## 8. 剩余的热点（T1 的 GPU 剖析，22.58 ms/轮）

| 项 | ms | 说明 |
|---|---|---|
| VRAM hits（驻留专家）| 3.13 (GDN) + 1.02 (QSA) | 24 层 × 8 专家 |
| head（Q5_K 词表头，417 MB）| 2.23 | 只有 1 行，纯带宽 |
| qkv gemv | 1.99 | |
| hc0/hc1 的 down+up（8 条）| ~3.5 | 每条都是小 GEMM |
| q+q-idx | 0.72 | |
| waitA（等辅助卡环）| 0.63 | |
| shared+quant | 1.59 | |

## 9. 第二轮：serve 模式 + `--remote-expert-opt`（同日晚些时候）

第 2 节的清单在 generate 模式上扫完了；这一轮换成 serve 模式（`--remote-expert-opt` 只在 serve 下允许），
并按第 8 节的热点重新排了优先级。

### 9.1 Q36 的"多卡"方向在这里对应哪个旗标

Q36 的 1 主 + 3 helper 是 `src/qwen35/q36_engine.cu` 里**专用引擎**的实现，它的
`STRATA_Q36_HELPER_ASYNC=1`（helper 异步）、`STRATA_Q36_HELPER_PARALLEL=1`（helper 并行）、
`STRATA_Q36_HELPER_RING=2`（环形 + 设备侧合并）在 Flash 这条树里**不存在**
（`STRATA_Q36_HELPER_RING` 是共享代码里的环形深度，但和 reduce 模式互斥，
`remote_expert_opt.cu:78` 直接拒绝）。

共享代码里对应的东西是 **`--remote-expert-opt`**：helper 在自己卡上把专家输出加权归约成
每 token 一个向量再回传。这是本轮最大的一笔：

| 配置（276 token 提示、128 token 生成、P100 + 三 P104）| decode |
|---|---|
| 不用 `--remote-expert-opt` | 42.1 tok/s |
| `--remote-expert-opt` | **47.2 tok/s（+12.3%）** |
| ＋`--expert-cache-remote-placement layer` | **47.8 tok/s（+13.6%）** |
| `--remote-expert-opt` 但**不要** helper 缓存 | 38.2 tok/s |

⇒ **helper 缓存值 +23%，`--remote-expert-opt` 值 +12%。** 两件事都要。

### 9.2 本轮扫过但无效的（serve 模式，基准 47.2）

| 旋钮 | 结果 | 为什么 |
|---|---|---|
| `STRATA_HC_PERSIST=1` | 47.50 | CUDA 侧是空实现（`fused_gr.cu:3042`），HIP 专用 |
| `STRATA_GDN_CHUNKED=1` / `2` | 47.82 / 47.68 | 噪声内 |
| `STRATA_TSUM=1` | 47.67 | 噪声内 |
| `STRATA_ADAPT_OVERLAP=1` | 46.39 | 引擎拒绝：`adaptive overlap: off: not with --remote-expert-opt` |
| `STRATA_COLD_OVERLAP=3` | 47.31 | 噪声内 |
| `--expert-cache 6000` / `3000` | 47.43 / 47.37 | 噪声内（10240 个专家本来就在四张卡上） |

### 9.3 剩下的三处代码级机会（都不是旋钮）

| # | 位置 | 预估 | 做法 |
|---|---|---|---|
| 1 | `remote_experts.cpp:422` `begin()`：每层每个 helper 3 次 memcpy + 2 个 kernel | 1.9 ms/轮 | 把 input 与 metadata 合并进同一个 pinned 缓冲，一次 memcpy；或把这一段捕获成 CUDA graph |
| 2 | `verify.cpp:3059-3063`：主机线程调 `pool(...)`，GPU 要等 plan 发布 | generate 模式下 3.45 ms/轮 | 把 CPU 池挪到工作线程（Q36 的 `STRATA_Q36_HELPER_ASYNC` 就是这个意思，Flash 树里没有） |
| 3 | `iq_kernels.cu` 的 Q5_K 头核：417 MB 只跑出 187 GB/s（P100 HBM2 有 730）| 2.23 → 1.1–1.5 ms | 一个 sm_60 的 packed Q5_K 头核（照 `q8_sm60.cuh` 的做法） |

`--remote-expert-opt` 之外的多卡路都查过了：**树里没有分片头、没有 peer logits 归约、
没有 `--head-device`**，helper 只算专家行，头永远在主卡上。

