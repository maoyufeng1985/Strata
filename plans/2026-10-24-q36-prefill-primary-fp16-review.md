# q36：prefill 把一部分专家流式给主卡算，怎么把主卡算力吃满 —— 复核（2026-10-24）

对象：用户问题「prefill 阶段把一部分专家从内存流式发给主卡计算，如何充分利用主卡算力」
+「prefill 也能用 helper 的 DP4A 吗？主卡 FP16 18.7 TF」。
只读：不改代码。硬件账 + 代码路径，把「能 / 不能 / 待测」钉死。

## 1. 硬件事实（先核，再算）

| 卡 | PCI | 链路 | 理论单方向 | H2D 引擎实测 | FP16 | int8/DP4A |
|---|---|---|---|---|---|---|
| 主卡 P100-12G | **03:00** | **Gen3 x8 @ 8.0 GT/s**（槽是 x16，`LnkSta: Width x8 (downgraded)`） | **8 GB/s** | **6.05–6.13 GB/s（2026-10-24 实测，pinned 源 + `cudaMemcpyAsync`，4–256 MiB）= 链路的 76%** | **18.7 TF（峰值，无 int8 tensor）** | **无真 DP4A，int8 走 4×vmad 回退（慢 2.2×）** |
| helper CUDA1 | 01:00 | **Gen1 x4 @ 2.5 GT/s** | 1.25 GB/s | **0.81 GB/s（pinned+async 与 pageable+sync 同速 = 链路上限）** | 3.3 TF | **真 `__dp4a`（sm_61）** |
| helper CUDA2 | 02:00 | 同上 | 1.25 GB/s | 0.81（同上） | 3.3 TF | 真 DP4A |
| helper CUDA3 | 05:00 | 同上 | 1.25 GB/s | 0.79（同上） | 3.3 TF | 真 DP4A |

**关键纠正**（2026-10-24 换槽后重测，三条）：
- **PCI 槽位变了**：P100 换到 **03:00**（`LnkCap: Port #2, Speed 8GT/s, Width x16`，但
  `LnkSta: Speed 8GT/s (ok), Width x8 (downgraded)` —— **协商成 x8**，理论单向 8 GB/s）；
  三张 P104 在 **01/02/05:00**（`LnkSta: Speed 2.5GT/s (ok), Width x4 (ok)`，
  Gen1 x4，理论单向 1.25 GB/s）。**helper 的链路只有主卡的 1/6**——这才是 helper 下腿慢的
  结构原因。
- **「P100 的 H2D 只有 3 GB/s」是探针方法错**，不是硬件上限。旧探针用 pageable 源 + 同步
  `cudaMemcpy`，量的是「驱动逐页拷进 staging 再 DMA」的慢路径。改用 pinned 源 +
  `cudaMemcpyAsync` 侧流后实测 **6.05–6.13 GB/s = Gen3 x8 链路的 76%**。详见 §4。
- **helper 的 0.81 GB/s 不是引擎上限，是 PCIe 链路上限**：pinned+async 与 pageable+sync
  **同速**（0.81 vs 0.81），换源类型不动分毫 → DMA 引擎已经贴住 1.25 GB/s 的 Gen1 x4 链路
  （吃到 65%）。**helper 下腿的墙是插槽，不是代码。**
- P100（GP100）**没有 int8 tensor core**（Pascal 里只有 GP104 有真 `__dp4a`），
  所以主卡 int8 走 4×vmad 回退。主卡的优势只在 **FP16（18.7 TF，helper 的 2×）**。

## 2. 三个子问题逐一答

### 2.1 「prefill 用 helper 的 DP4A」—— 已经在用，不是待做

`STRATA_Q36_PREFILL_PEERS`（编码修复后）就是干这个的：3 张 P104 按 `e % 3` 拿
3440/3400/3400 专家/层，helper 侧 MMQ 跑在 sm_61 真 `__dp4a`。prefill 的 `helper_wait`
13.5 ms/层 **就是 helper DP4A 在干活的账单**。所以「prefill 用 helper DP4A」= 现状，
不是要做的优化。

### 2.2 「主卡 FP16 18.7 TF 闲置吗」—— 是的，且现在吃不到

prefill 主卡现在只算 shared expert（`primary_work = 0.745 ms/层`），专家全在 helper。
主卡 18.7 TF 的 FP16 **完全闲置**。要吃上，得把一部分专家挪回主卡、且走 **FP16 GEMM
（`K::mm`，`q36_engine.cu:2786` 的 `T > kGemmT` 分支）**，而不是现在 keep-resident 的
MMQ int8 回退（那吃的是主卡的劣势，不是优势）。

### 2.3 「流式多少专家给主卡」—— 受两条硬上限

- **H2D 引擎**：主卡算 X 个专家/层要 H2D `1.83 MiB × X`。按实测 6.07 GB/s：
  X=44 → 80.5 MiB / 6.07 GB/s = **13.3 ms/层**（与 helper_wait 13.5 ms 刚好持平）。
  若主卡引擎只吃到 4 GB/s（pageable 那种走法），X=44 → 20 ms/层（比整层还长，不可行）。
  **主卡引擎吃多少 = 全案成立与否的前提。**
- **主卡 sm_60 回退**：主卡 int8 MMQ 比 helper sm_61 DP4A 慢 2.2×。若主卡走 **FP16**
  （18.7 TF），则反过来**比 helper 快 2×**（helper FP16 合计 9.9 TF）——所以主卡那部分
  **必须走 FP16，不能走 int8 回退**，否则把活送进最慢的路径。

## 3. 落点（最小、opt-in、不碰默认数值路径）

**形状**：主卡算 X 个专家（按热度挑**最冷**的 X，因为主卡 FP16 优势最大但 H2D 也最贵，
冷专家行数少、H2D 占的权重字节按专家数不按行数，选冷专家让 H2D 字节不落在热点上），
helper 少算 X 个。主卡那 X 个走 **FP16 GEMM**（`K::mm`），helper 那部分不变（DP4A）。

**改动点**（都 opt-in，旗标 `STRATA_Q36_PRIMARY_FP16=N`，N = 每层主卡算的专家数，0 = 关）：
1. `open_helpers` 的 `filter_resident`（`q36_engine.cu:6320`）：现在只滤主卡 `slot_of≥0` 的
   39 个。改成按**热度排序取最冷的 N 个**（`H.score` 在 `place_experts` 后是 0，得用启动时
   的静态热度 = 前 N 个冷 id；或保持 39 个常驻槽不动，N 只在这 39 个里选）。
2. `layer()` 的 MMQ 分支（`q36_engine.cu:3247` 的 `gu`/`dn` 两段 `mctx->run`）：主卡那
   `n_local = N` 个专家，`N > 0` 时改发射 `mm`（FP16）而不是 MMQ int8。`n_local = 0`
   （N=0，现状）时不发射（`Context::run` 的 `max_rows > 0` 短路）。
3. `stream_kick`（`q36_engine.cu:3563`）：主卡那 N 个走 H2D（host 内存）还是 D2D
   （`L.blobs` 39 个常驻）取决于 N 与 39 的关系——N ≤ 39 走 D2D（不花 PCIe），
   N > 39 多出的走 H2D。

## 4. 待测 → 已量（#1 主卡 H2D，2026-10-24，远端 192.168.1.98）

### 4.1 第一版探针：方法错，数字偏低

`/tmp/h2d` 用 **pageable 源 + 同步 `cudaMemcpy`**。这条路径驱动要把源先逐页拷进 staging
缓冲再 DMA，比引擎实际走的路径慢。换槽前后各量一次（`cuDeviceGetName`：dev 0=P100，
dev 1/2/3=P104）：

| 卡 | 位置 | 4 MiB | 16 MiB | 64 MiB |
|---|---|---|---|---|
| **主卡 P100（换槽前）** | 05:00, dev 0 | **3.77** | **4.10** | 2.99 |
| **主卡 P100（换槽后）** | 03:00, dev 0 | 3.65 | 3.93 | 3.00 |
| helper P104 ×3 | 05/01/02:00, dev 1/2/3 | ~0.78 | 0.79–0.80 | — |

旧的大小扫描（dev 0）：1 MiB 2.88 / 4 MiB 3.64 / 8 MiB 3.84 / **16 MiB 4.06（峰值）** /
32 MiB 3.01 / 64 MiB 2.93 / 128 MiB 2.99。

**我曾据此写下「这条路径的持续吞吐就是 ~3 GB/s，是 P100 H2D 引擎的硬件上限」——这句是错的。**
错在把 pageable+sync 的低效路径当成了硬件上限。链路本身是 Gen3 x8 @ 8.0 GT/s
（理论单向 8 GB/s），实测只吃到 46–51%，那是**方法**的问题，不是卡的问题。

### 4.2 第二版探针：pinned 源 + `cudaMemcpyAsync`（引擎实际走的路）

`/tmp/h2d2`，两条路径同机对照（cudaEvent 计时，warm 3 次 + 正式迭代）：

| 大小 | pageable+sync | **pinned+async** | 提升 |
|---|---|---|---|
| 4 MiB | 3.82 | **6.13** | +60% |
| 16 MiB | 4.04 | **6.05** | +50% |
| 64 MiB | 4.18 | **6.07** | +45% |
| 128 MiB | 4.18 | **6.07** | +45% |
| 256 MiB | 4.20 | **6.06** | +44% |

**MEASURED：P100 的 H2D 持续吞吐 = 6.05–6.13 GB/s**（pinned 源 + `cudaMemcpyAsync` 侧流），
= Gen3 x8 理论 8 GB/s 的 **76%**。4 MiB 以上基本是一条平线（6.05–6.13），不再随大小掉。

**引擎走的就是这条**：`L.hblobs` 是 `cudaHostAlloc` 的 pinned host 内存，拷贝是
`cudaMemcpyAsync` on `cst` 独立流（`q36_engine.cu` 的 `stream_kick`）。所以 6.07 就是引擎
能拿到的数，不是探针的数字游戏。

**换槽（05:00 → 03:00）对 P100 的 H2D 带宽没有影响**（第一版数据 4 MiB 3.77→3.65、
16 MiB 4.10→3.93，都在 ±5% 噪声内）。换槽前后的差别是**槽位型号**（x8 槽 → x16 槽），
但**协商宽度被主板降级成 x8**（`LnkSta: Width x8 (downgraded)`）——所以两边的链路宽度其实一样，
带宽自然一样。

### 4.3 helper 的 H2D：pinned+async 重测（0.81 是链路上限，不是低报）

用同一个 `/tmp/h2d2`（64 MiB × 30）跑四张卡：

| CUDA dev | 卡 | PCI | pageable+sync | **pinned+async** |
|---|---|---|---|---|
| dev 0 | P100 | 03:00（Gen3 **x8**） | 4.17 | **6.07** |
| dev 1 | P104 | 01:00（Gen1 x4） | 0.81 | **0.81** |
| dev 2 | P104 | 02:00（Gen1 x4） | 0.81 | **0.81** |
| dev 3 | P104 | 05:00（Gen1 x4） | 0.79 | **0.79** |

**MEASURED：三张 P104 的 pinned+async 与 pageable+sync 完全同速**（0.81 vs 0.81）。
这和 P100 的行为**相反**，说明差别在硬件层：

- P100：换源类型就快 45% → 它原先被 pageable 的逐页 staging 拖住，换成 pinned 就贴住链路上限。
- P104：**换源类型一点不变** → 它**已经在贴 PCIe 链路上限**了。Gen1 x4 理论 1.25 GB/s，
  实测 0.81 = 链路的 65%。

**所以我先前的怀疑（0.85 是 pageable 测的、可能低报）不成立**——pinned+async 一点没涨。
helper 下腿的墙**是插槽（Gen1 x4），不是代码，也不是 DMA 引擎**。这条对「怎么提升 prefill」
影响很大：helper 的下腿 H2D 没有软件解法，**只有换槽/换板**能改。

### 4.4 三条更正（对 q36 的影响）

1. **主卡可用 H2D = 6.07 GB/s**（不是 3、不是 4.1、不是 8）。= Gen3 x8 链路的 76%。
2. **N 的可行上界**：`1.83 MiB × N / 6000 ≤ 13.5 ms` → **N ≤ 44**（旧数按 3.0 反推是 22）。
   但这只是「H2D 不超过 helper_wait」的**可行性**上界，不是**盈利**上界——见 §5。
3. **helper 的 0.81 GB/s 是链路上限**（Gen1 x4 的 65%），**没有软件解**。

#2（主卡 FP16 小 GEMM 实际吞吐）**仍未量**，是下一个前提。

## 5. 判断（按 #1 的实测重算）

- **prefill 用 helper DP4A：已在用**（`helper_wait` 13.5 ms 就是它）。
- **主卡 FP16 18.7 TF：现在闲置**，要吃上得「主卡多算 X 个专家 + 走 FP16 不走 int8 回退」。
- **主卡 H2D 实测 6.07 GB/s**（pinned+async；不是 3、不是 4.1、不是 8、不是 0.85）。重算 N：
  - **可行性上界**（H2D 不超过 helper_wait 13.5 ms/层）：
    `1.83 MiB × N / 6.07 GB/s ≤ 13.5 ms` → **N ≤ 44**。
  - **盈利上界**（H2D 成本要小于 helper 算那些专家省下的时间）：主卡每多算 1 个专家，
    H2D 成本 = 1.83 MiB / 6.07 GB/s = **316 µs**；而 helper 算 1 个专家省下的时间
    = 13.5 ms / 10240 专家 ≈ **1.3 µs**。**两者差 240 倍** —— H2D 字节按专家数、不按行数，
    主卡每多算 1 个专家就白搬 1.83 MiB。
  - 所以 **N ≤ 44 是「不拖慢」的极限，N = 0 才是「不亏」的极限**。即使 N 提到 44（比
    keep-resident 的 39 还多 5 个），净收益也是**大幅负数**。
- **结论**：在这台盒子（主卡 H2D 6.07 GB/s、helper DP4A 快、helper_wait 13.5 ms 是关键路径），
  **「流式一部分专家给主卡」的净收益 ≈ 0 或负**——主卡的 H2D 成本（每专家 316 µs）远超
  helper 算那 1 个专家的省幅（1.3 µs）。**keep-resident 的 39 个（D2D，0 PCIe）已是主卡能吃
  的上限**（39 个常驻槽的权重已经在 VRAM，不花 H2D），再多就撞 H2D 墙。
- **真正能提升 prefill 的不是「主卡多算专家」**，而是**把 helper_wait 13.5 ms 压下来**。
  helper 下腿的 H2D 实测 **0.81 GB/s = Gen1 x4 链路的 65%**，而且 pinned+async 与
  pageable+sync **完全同速**（§4.3）——**这是 PCIe 插槽的墙，不是软件的墙**。§max-rows §3b
  证过它藏不掉；现在多知道一条：**也压不动**（源类型不是杠杆）。
  唯一能动的是**硬件**：把 P104 换到宽槽（x16 / Gen3），helper 的下腿能快 6–10 倍。
  或者让主卡那 6.07 GB/s 的 H2D **和 helper 的链路并行吃**（主卡 P100 在 03:00、helper 在
  01/02/05:00，`nvidia-smi topo -m` 显示四卡两两 **PHB**——各自独立 Host Bridge，物理上能并行，
  但各自的 DMA 引擎吞吐独立）。

## 6. 与 keep-resident 的关系

keep-resident（`79a1ebca`，已默认开）是这条路的第一步：主卡算自己 39 个常驻槽的专家
（D2D，0 PCIe），decode 掉 12.5%、prompt 持平。本方案（N>39 的流式扩展）按 §5 的 H2D 墙
**不可行**：可行性上界 N ≤ 44 只比 39 多 5 个，而盈利上界是 0。**主卡 39 个 D2D 专家已是
这台盒子的最优解**，prefill 的天花板在 helper 的下腿 H2D（0.81 GB/s = Gen1 x4 链路的 65%，
**硬件墙**），不在主卡。

## 7. 一句话给下一个人

想在这台盒子提升 prefill，**软件层面已经到顶**：

- 主卡的 18.7 TF FP16 吃不上，因为每搬 1 个专家要 316 µs 的 PCIe，而 helper 算它只省 1.3 µs。
- helper 的下腿 0.81 GB/s 是 Gen1 x4 插槽的物理上限，软件压不动。
- **唯一的下一步是换硬件**：把三张 P104 换到 x16 槽（主卡在 03:00 占的那个是 Gen3 x16 槽，
  现在协商成 x8 —— 换过去的主卡可能也能恢复满 x16）。换槽后重测这三条数，再谈优化。
