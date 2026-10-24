# q36：prefill 的关键路径是 helper 的「下腿」，不是主卡的算力（2026-10-24）

对象：用户问题「prefill 阶段把一部分专家从内存流式发给主卡计算，如何充分利用主卡算力」
+「prefill 也能用 helper 的 DP4A 吗？主卡 FP16 18.7 TF」+「主卡 H2D 为什么吃不满链路」。
只读复核 + 远端实测。硬件账 + 代码路径，把「能 / 不能 / 待测」钉死。

**一句话结论：这台盒子上 prefill 的墙是 helper 的 PCIe 槽（Gen1 x4，0.81 GB/s），
不是主卡的算力。主卡的 18.7 TFLOP/s FP16 吃不上，因为专家权重是量化的（Q4_K/Q5_K），
根本不走 FP16 GEMM；就算走，也比 PCIe 便宜 240 倍。**

---

## 0. 三条实测数据（本轮新量）

| 量 | 值 | 怎么量的 |
|---|---|---|
| 主卡 P100 H2D 持续吞吐 | **6.05–6.13 GB/s**（pinned 源 + `cudaMemcpyAsync`） | `/tmp/h2d2` 两条路径对照，4–256 MiB |
| helper P104 H2D 持续吞吐 | **0.79–0.81 GB/s**（pinned+async 与 pageable+sync **同速**） | 同上，四张卡各跑一遍 |
| prefill 每层时间分解 | **forward 26.7 ms/层**，其中 `plan` 6.79 ms + `helper_wait` 7.65 ms | `STRATA_Q36_PROMPT_TIMING=1`，bench 0-2 聚合，40 层 × 3 pass |

## 1. 硬件事实（先核，再算）

| 卡 | PCI | 链路（`lspci` LnkSta） | 理论单方向 | H2D 实测 | FP16 | int8/DP4A |
|---|---|---|---|---|---|---|
| 主卡 P100-12G | 03:00 | `Speed 8GT/s (ok), Width x8 (downgraded)` → **Gen3 x8** | 8 GB/s | **6.05–6.13 GB/s = 链路的 76%** | **18.7 TF（峰值，无 int8 tensor）** | **无真 DP4A，int8 走 4×vmad 回退** |
| helper P104 ×3 | 01/02/05:00 | `Speed 2.5GT/s (ok), Width x4 (ok)` → **Gen1 x4** | 1.25 GB/s | **0.79–0.81 GB/s = 链路的 65%** | 3.3 TF | **真 `__dp4a`（sm_61）** |

**关键纠正（相对上一版）**：
- **「P100 的 H2D 只有 3 GB/s」是探针方法错**，不是硬件上限。旧探针用 pageable 源 + 同步
  `cudaMemcpy`，量的是「驱动逐页拷进 staging 再 DMA」的慢路径。换 pinned 源 +
  `cudaMemcpyAsync` 侧流后实测 **6.05–6.13 GB/s**。
- **helper 的 0.81 GB/s 不是低报，是链路上限**：pinned+async 与 pageable+sync **完全同速**
  （0.81 vs 0.81），换源类型不动分毫 → DMA 引擎已经贴住 Gen1 x4 链路。**helper 下腿没有软件解。**
- **P100 在 03:00 是 x8 槽被降级成 x8 宽度**（`LnkCap: Width x16` 但 `LnkSta: Width x8
  (downgraded)`），所以换槽前后带宽一样（第一版实测 4 MiB 3.77→3.65、16 MiB 4.10→3.93，
  ±5% 噪声内）。

## 2. 专家权重是量化的 —— 「主卡走 FP16 GEMM」不成立

从 gguf 读出的张量类型（`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`）：

```
blk.0.ffn_gate_exps.weight   type 12  dims [2048, 512, 256]   <- Q4_K
blk.0.ffn_up_exps.weight     type 12  dims [2048, 512, 256]   <- Q4_K
blk.0.ffn_down_exps.weight   type 13  dims [ 512, 2048, 256]  <- Q5_K
```

`q36_engine.cu:1468-1473` 的 `native_expert_layout` 把这两个类型算成每专家
`L.bytes`（gate+up+down 三个量化矩阵），`L.blobs` 是 `uint8_t*`（`:1244`）。
**主卡算专家走 `moe_mmq.cu` 的 int8 MMQ**（`q36_engine.cu:2375-2382` 的 `use_mmq`），
helper 走 `remote_experts.cpp` 的 native grouped kernel —— 两条路都是**边读边反量化**
（on-the-fly dequant），不是「先反量化成 FP16 再 GEMM」。

所以：**主卡的 18.7 TFLOP/s FP16 算力在这条专家路径上根本没被用到，
而且用不上**——权重是 4-bit/5-bit 量化数据，要吃 FP16 得先反量化，那比直接算还贵。
`q36_engine.cu:3122` 的 `K::mm` FP16 分支是给 dense 投影（attention / shared expert）用的。

模型形状（gguf kv）：`n_embd=2048`、`expert_ff=512`、256 专家/层、41 层、`top_k=8`。

## 3. 「prefill 用 helper 的 DP4A」—— 已经在用

`STRATA_Q36_PREFILL_PEERS` 就是干这个的：3 张 P104 按 `e % 3` 拿 3440/3400/3400 专家/层，
helper 侧 MMQ 跑在 sm_61 的真 `__dp4a`。prefill 的 `helper_wait` 13.3 ms/层
**就是 helper DP4A 在干活的账单**。所以这一项 = 现状，不是待做。

## 4. 关键路径实测（`STRATA_Q36_PROMPT_TIMING=1`，2026-10-24 远端）

两臂，同一台机同一天。P1 = `bin/strata-q36.krd`（keep-resident 默认开），
P2 = `bin/strata-q36.own`（编码修复基线）。**每层 ms，bench 0-2 聚合（40 层 × 3 pass）**：

| span | P1（keep 开） | P2（keep 关） | 是什么 |
|---|---|---|---|
| mixer | 7.04 | 7.02 | GDN / attention 投影 |
| router_route | 0.26 | 0.23 | router + route 核 |
| route_d2h | 0.05 | 0.05 | 路由 D2H + 阻塞同步 |
| **plan** | **6.79** | **6.82** | **peer 计划的 host 侧工作（GPU 空转）** |
| primary_work | 4.68 | 0.76 | 主卡自己的 MMQ + shared expert |
| **helper_wait** | **7.65** | **13.29** | **等 helper 的行 + 上腿 H2D** |
| combine | 0.20 | 0.19 | moe_end |
| **forward** | **26.67** | **28.37** | **整层** |
| HOST: prep_sync | 5.11 | 5.14 | `prepare_end()` 等行计划 |
| HOST: peer_plan | 6.77 | 6.70 | 计划总 host 墙 |
| HOST: sync3022 | 18.66 | 20.55 | 路由 D2H 的阻塞同步 |

**host 墙里最刺眼的一个数**（`the plan's host wall per layer`）：

```
wait=5.583 (helper CUDA1 4.947, CUDA2 0.007, CUDA3 0.164)
```

**CUDA1 一个人占 5.0 ms，CUDA2/CUDA3 几乎为 0。** 这不是「CUDA3 慢 3.2×」那个
收割顺序效应（`plans/2026-10-24-q36-helper-imbalance-review.md` 复核过），
是**第一张被等的 helper 把全部等待吃掉了**。

## 5. 为什么 CUDA1 独占 5 ms：下腿的窗口拷贝

`PeerPrefill36::prepare_begin`（`q36_peer_prefill.cu:799-847`）在 host 路线下做：
- 主卡 D2H 一次：`n_tok × n_embd × 4` = 512×2048×4 = **4 MiB** 激活 + 16 KiB 路由
  （`q36_engine.cu:4130`，**只拷一次，三个 helper 共用**，这一版已经优化过）。
- 然后**每个 helper 各自** H2D 一份同样的 4 MiB（`q36_peer_prefill.cu:841`，
  `cudaMemcpyAsync(m.d_x, m.h_x_use, ...)`）。
- `prepare_end()`（`:898`）阻塞等这张卡把拷贝 + 4 个行计划核做完。

**算账**（helper 链路实测 0.81 GB/s）：

```
4 MiB ÷ 0.81 GB/s = 5.18 ms   <- 与实测 CUDA1 wait 4.95-5.02 ms 吻合
```

**这个 5 ms 就是「第一张被等的 helper 的下腿传输时间」**，CUDA2/CUDA3 的 0.007/0.164
是因为它们的 `prepare_end` 在 CUDA1 之后才轮到，窗口早已上传完（**收割顺序效应**）。

**三个 helper 的下腿是并行的**，不是串行：`prepare_begin` 对三张卡依次入队
（`q36_peer_prefill.cu:4141-4148`），三张卡各走各的 PCIe；`prepare_end` 才按 r 序收割。
所以**下腿的墙是 5.2 ms/层**（不是 12 MiB 串行 = 15.5 ms），但它**占住整条关键路径**——
host 在这 5.2 ms 里干等着，GPU 空转。

**而且 P2P 走不了**：远端日志明写
`q36 peer prefill: CUDA1's prompt rows cross through host staging (no peer link with CUDA0:
peer access is not supported between these two devices)`。所以下腿是
**主卡 D2H → host → helper H2D 两段**，两段都吃 PCIe，且 helper 那一端只有 0.81 GB/s。

### 5b. 上腿与下腿的字节账（fp32 下腿 vs fp16 上腿，不对称）

| 腿 | 方向 | 每个 helper 每层 | 格式 | 传输时间 @0.81 GB/s | 在哪 |
|---|---|---|---|---|---|
| **下腿** | 主卡 → helper（激活 + 路由） | 4.00 MiB + 16 KiB | **fp32** | **5.18 ms** | host 墙 `wait`（`plan` 段） |
| **上腿** | helper → 主卡（算好的行） | 5.33 MiB（1365 行 × 2048） | **fp16** | 6.90 ms | GPU 段 `helper_wait` 7.65 ms |

**上腿已经是 fp16**（`STRATA_Q36_PEER_F16=1`，`q36_peer_prefill.cu:1005-1016`），
**下腿还是 fp32** —— 不对称。上腿 6.90 ms ≈ 实测 `helper_wait` 7.65 ms ✓。

**所以两段 PCIe 各占约 5–7 ms，加起来 12 ms/层，占 forward 26.7 ms 的 45%。**
下腿那 4 MiB 是 fp32，而 helper 收到后**立刻量化成 q8_1**（`q36_peer_prefill.cu:945`
的 `MMQ::quantize`）——**fp32 的那一半精度在传输路上就白扔了**。


## 6. 那「主卡多算专家」这条路的账

主卡算 X 个专家/层要 H2D `1.83 MiB × X`（一个专家的量化权重：gate+up 各
2048×512 的 Q4_K + down 512×2048 的 Q5_K，合计 ≈ 1.83 MiB）。

- **可行性上界**（H2D 不超过 helper_wait 13.3 ms）：`1.83N / 6000 ≤ 13.3` → **N ≤ 43**。
- **盈利上界**：主卡每多算 1 个专家，H2D 成本 = 1.83 MiB / 6.07 GB/s = **316 µs**；
  而 helper 算 1 个专家省下的时间 = 13.3 ms / 10240 专家 ≈ **1.3 µs**。
  **差 240 倍。** H2D 字节按专家数、不按行数，主卡每多算 1 个专家就白搬 1.83 MiB。

**结论：净收益 ≈ 0 或负。** keep-resident 的 39 个（D2D，0 PCIe）已是主卡能吃的上限。

## 7. 真正能动的地方：下腿的字节

下腿传的是 **fp32 激活**。这是纯浪费——helper 侧立刻就把它量化成 q8_1 算
（`q36_peer_prefill.cu:945` 的 `MMQ::quantize`）。fp32 → fp16 或 q8 都能省一半到四分之三：

| 激活格式 | 每 helper 每层 | 传输时间 @0.81 GB/s | 占 forward 26.7 ms |
|---|---|---|---|
| **fp32（现状）** | 4.00 MiB | **5.18 ms** | **19%** |
| fp16 | 2.00 MiB | 2.59 ms | 10% |
| q8_1 | 1.00 MiB | 1.29 ms | 5% |

**代价的诚实账**：主卡 D2H 那 4 MiB 也要跟着缩（4 MiB → 1 MiB），
主卡那一端 6.07 GB/s 只省 0.66 ms；真正的大头是 helper 那一端（省 2.6 / 3.9 ms/层）。
**但下腿的传输和上腿是串行的**（上腿要等下腿传完才能算），
所以下腿省下的 2.6 ms **不完全落到 forward 上**——要看 GPU 那边有没有别的事可做。
按 span 算，forward 26.7 ms 里 `plan` 6.79 + `helper_wait` 7.65 = 14.4 ms 是 PCIe 串行段，
其余 12.3 ms 是 GPU 算的（`mixer` 7.04 + `primary_work` 4.68 + router/combine 0.5）。
**下腿省 2.6 ms 只能落到 forward 2.6 ms（≈10%），前提是上腿不成为新瓶颈。**

**数值上**：激活转 fp16 的相对误差 < 1e-3（fp16 尾数 10 bit），
而 helper 侧本来就要量化成 q8_1（8 bit）——**fp16 这一步在数值上几乎免费**，
q8_1 的量化误差远大于 fp16 的截断误差。但**这会改数值路径，logits 不再逐字节相同，必须实测。**

**下腿没有现成旗标**：`STRATA_Q36_PEER_F16` 管的是**上腿**（helper → 主卡的行用 fp16，
`q36_engine.cu:1956-1962`），**下腿的激活仍是 fp32**。要做要新加一个。

## 8. 判断（按实测重算）

- **prefill 用 helper DP4A：已在用**（`helper_wait` 13.3 ms 就是它）。
- **主卡 FP16 18.7 TF：吃不上，也不该吃**（§2：专家权重是 Q4_K/Q5_K，走量化 MMQ，
  不走 FP16 GEMM；而且每专家 316 µs 的 H2D 只换来 1.3 µs 的算力节省）。
- **prefill 的真墙：helper 的两条 PCIe 腿**。下腿 4 MiB fp32 激活 = 5.18 ms/层，
  上腿 5.33 MiB fp16 行 = 6.90 ms/层，加起来 ≈ 12 ms/层，占 forward 26.7 ms 的 **45%**。
  这不是代码写得不好，是 **Gen1 x4 插槽**。软件能做的只有**少传字节**
  （下腿 fp32 → fp16/q8，能省 2.6–3.9 ms/层），**换槽位/换板**能做的更多。
- **P100 在 03:00 被降级成 x8 宽度**（`LnkCap x16` 但 `LnkSta: x8 downgraded`），
  主卡那一端还有一倍没吃满 —— 但主卡不是瓶颈，这不重要。

## 9. 下一步（按性价比排）

1. **下腿激活 fp16 或 q8**（新旗标，opt-in）：每个 helper 省 2.6 / 3.9 ms/层
   （三卡并行，所以是 2.6 / 3.9 不是 3 倍）。forward 26.7 → 约 24.1 / 22.8 ms/层，
   提速 10% / 15%。**数值路径会变，logits 要逐行对照**（AGENTS.md：改了默认必须实测）。
2. **把 `prepare_end` 的等待从「按 r 序」改成「按传输完成序」**：现在 CUDA2/CUDA3 的
   0.007 ms 说明它们的窗口早就传完了，但 host 仍按 r 序等 CUDA1 第一个。
   改成「谁先好谁先收」不能省时间（总传输量不变），但能让 GPU 提前开始算——
   **这条要配合「launch 能否在 wait 之前发」的检查**。
3. **换硬件**：把三张 P104 换到 x16 槽（主卡 03:00 那个是 Gen3 x16 槽，现在协商成 x8，
   主板在降级）。helper 换到宽槽，下腿能快 6–10 倍。
4. **不再做**：主卡多算专家（§6 净收益负）、主卡 FP16 专家 GEMM（§2 前提不成立）。

#2（主卡 FP16 小 GEMM 实际吞吐）**已不需要测**——§2 说明专家不走 FP16，
主卡 FP16 算力在这条路上用不上。
