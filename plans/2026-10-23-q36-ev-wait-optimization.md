# q36：优化 helper 的 ev-wait（2026-10-23）

配套：[2026-10-23-q36-p4-prompt-optimization.md](2026-10-23-q36-p4-prompt-optimization.md)（方案与排名）、
[2026-10-21-q36-peer-ima-root-cause.md](2026-10-21-q36-peer-ima-root-cause.md) §13.4–13.6。

状态：**设计定稿，待实施**（排在尺子 `STRATA_Q36_PROMPT_TIMING=1` 之后，在它那个干净的 diff 上做）。

## 0. 实测更新（2026-10-23，尺子出数）——本节优先于下面所有估算

> ⚠️ **本节的构成表已被第二版尺子推翻**（见 §0.1）。第一版尺子把 21.02 ms/层当成"helper 等待"，
> 第二版（8 个边界）把它拆开后：真正的 helper 等待 + 上腿 H2D 是 **12.900 ms/层**，
> 而 **18.765 ms/层是 host 串行等 3 个 helper 的 `prepare()`**（不在 ev-wait 这条链上）。
> **先做 §0.1 的候选 A，它比本文的 fp16 大得多。**

`STRATA_Q36_PROMPT_TIMING=1` 在 P4 配置上量到（warm bench 1，40 层，512-token prompt，
二进制 `b82eca97`）：**47.53 ms/层**，GPU 时间线**完整闭合**（四段之和 × 40 = forward 总时间）。

| 段 | 每层 ms | 占比 |
|---|---|---|
| `attn+route+plan`（P1−P0） | **26.26** | **55%** |
| **`primary_mmq`（P2−P1）** | **21.02** | **44%** ← **helper 等待在这里** |
| `combine` | 0.225 | 0.5% |
| `post_combine` | 0.017 | — |

**所以"ev-wait 有 ~21 ms/层"是 MEASURED，不是估算**（下面 §1 的 13 ms 是**算出来的**，被实测修正）。

### 21.02 ms 的已知构成（算+测混合）

| 成分 | 时间 | 依据 |
|---|---|---|
| 上腿 D2H（11.2 MiB 过 helper 的 Gen1 x4） | ~9.0 ms | DERIVED（字节数实测，Gen1 x4 带宽实测 ~1.25 GB/s） |
| helper 的 MMQ 算（4.29 GMAC @ P104 int8） | ~0.9 ms | ESTIMATE |
| 主卡的 eparts H2D（11.2 MiB @ Gen3 x8） | ~1.4 ms | DERIVED |
| 主卡自己的 shared expert + 小核（`n_local=0`，不算专家） | ~1 ms | DERIVED（§13.5 的 2.74 GMAC/层） |
| **未解释** | **~8 ms** | **新边界（步 1b）要量的** |

### fp16 的收益据此重算（比 §3.1 的估计小）

- 上腿 D2H 11.2 → 5.6 MiB：~9.0 → ~4.5 ms（**省 ~4.5**）
- 主卡 H2D 11.2 → 5.6 MiB：~1.4 → ~0.7 ms（**省 ~0.7**）
- **合计省 ~5.2 ms/层**（原估 6.5，高估了 1.3）→ 47.53 → ~42.3 ms/层
  → 268.5 × 47.53/42.3 ≈ **302 tok/s（+12%）**（原估 +16%）。
- **前提**：那 8 ms 未解释的部分**不是**上腿字节造成的。**步 1b 的细分边界要先量出它是什么**
  ——如果它是主卡自己的活（GDN/attention 之外的部分），fp16 省不到它，收益还要再降。

### 0.1 第二版尺子的实测（2026-10-23 晚，`bin/strata-q36.prompt-timing2` = `c3ffcce5`）

8 个边界，warm bench 1，47.734 ms/层，`gpu_residual = 0.000`：

| 段 | 每层 ms | 占 |
|---|---|---|
| **plan**（host 串行等 3 个 helper 的 `prepare()`） | **18.765** | **39%** |
| **helper_wait**（等 helper + 上腿 H2D） | **12.900** | **27%** |
| **primary_work**（主卡自己的活） | **8.125** | **17%** |
| mixer（GDN/attention） | 7.096 | 15% |
| 其余（router/route/combine/post） | 0.847 | 2% |

**所以"helper 等待 = 21.02 ms/层"是错的**，真正的 `helper_wait`（P2−P1b）是 **12.900 ms/层**。
那 21.02 里混着 `primary_work` 的 8.125（第一版把 P1b 记在 `peer_prefill_wait()` **之后**，
CUDA 的等待节点会把事件时间戳推后 → helper 的等待被算进了"主卡自己的活"）。

### 0.2 候选 A：把 3 个 helper 的 `prepare()` 并行（**比 fp16 大得多**）

`peer_plan`（`q36_engine.cu:3967`）：

```cpp
for (int r = 0; r < 3; ++r) {
    peers[r]->prepare(...);   // ← 阻塞：prepare() 尾部有 cudaStreamSynchronize
    peers[r]->launch(...);
}
```

`prepare()` 的阻塞点 = `q36_peer_prefill.cu:737` 的 `cudaStreamSynchronize(m.stream)`，
等的是**这个 helper 自己**的下腿 H2D + 4 个 row-plan 核。
**3 个 helper 串行 → 三次下腿传输不重叠。**

**下腿的字节**：每个 helper 收**全部 512 token** 的激活（`n_tok=512`，4 MiB），
3 个 helper = **12 MiB/层过 Gen1 x4（~1.25 GB/s）→ 理论 ~9.6 ms**，
实测 `prep_sync` = **15.690 ms/层**（每次 prepare 平均 5.23 ms）。

**并行后**：三次下腿重叠 → ~5.2 ms/层（下腿传输与 row-plan 同时进行）
→ **省 ~10 ms/层**（47.7 → ~37.7）→ 267.7 × 47.734/37.7 ≈ **339 tok/s（+27%）**。

**改法**：拆成两轮 —— 先对 3 个 helper 都 `prepare()`（各自的阻塞 sync 互相独立，可并发），
再对 3 个 helper 都 `launch()`。**风险**：`launch()` 依赖 `prepare()` 产出的行数
（`m.n_owned` / `m.max_rows`），所以第二轮必须在第一轮全部完成之后；
`peer_plan` 里现有的两个 witness（所有权交叉核验、重复行检查）必须原样保留。

**代价**：`prepare()` 的下腿 D2H 是**阻塞的 `cudaMemcpy`（走主卡）**，三个一起做会抢主卡的
PCIe 通道；且 `m.h_x`（pinned）是每个 helper 自己的，互不冲突 ✅。

**fp16 的收益（候选 B/C）相对变小了**：上腿 fp16 省 ~2.8 ms/层（12.900 里的 11.2 MiB），
下腿 fp16 省 ~4.8 ms/层。**做 A 之后**再算 fp16，基线已经变了。

## 1. ev-wait 是什么，等什么

`PeerPrefill36::wait()`（`src/qwen35/q36_peer_prefill.cu:817-818`）：

```cpp
cudaEventRecord(m.ev, m.stream);              // 记在 HELPER 的 stream 上
cudaStreamWaitEvent(st_primary, m.ev, 0);     // 让主卡的 stream 等它
```

它等的是 **helper stream 上从 `prepare()` 起排队的全部东西**：

| 序 | 活 | 字节 | 走哪条链路 | 时间 |
|---|---|---|---|---|
| 1 | 下腿 H2D（激活 + 路由） | 4 MiB + 16 KiB | **helper Gen1 x4** | ~3.4 ms |
| 2 | 4 个 row-plan 核 | — | — | ~0 |
| 3 | `MMQ::quantize` + gate/up + swiglu + quantize + down | — | helper VRAM | ~0.9 ms |
| 4 | **上腿 D2H（结果行）** | **n×2048×4 ≈ 11.2 MiB** | **helper Gen1 x4** | **~9 ms** |
| 5 | （`wait()` 里）主卡 H2D h_out → eparts | 11.2 MiB | 主卡 Gen3 x8 | ~4 ms |

**主卡在 `peer_prefill_wait()`（`q36_engine.cu:3099`）处等的就是这个**，然后 `moe_end` 才能读 `eparts`。
所以 ev-wait 这条链在关键路径上贡献 **~13 ms/层**（4 + 9），对 47.7 ms/层 是**大头级**。

**这台机的链路（MEASURED，`nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current`）：**
3× P104 = **Gen 1 x4（~1.25 GB/s）**；P100 = Gen 3 **x8**（max x16，~8 GB/s）。
**Gen1 x4 是硬约束**——所以唯一能动的就是**字节数**。

## 2. 已排除的手段（各有硬理由，不要再试）

| 想法 | 为什么不行 |
|---|---|
| **主卡发已量化的 q8_1 激活**（省掉 helper 的量化 + 4 MiB→0.5 MiB） | helper 量化的是 `m.d_x` 按 `d_src` gather 出来的行（`q36_peer_prefill.cu:776`），**行序是 helper 自己那份**（1365 个 entry 的行序）；主卡的 `mq_x` 是**按专家排序的主卡 entries**，行序不同。要主卡先做 helper 的 gather 就得先知道 `d_src`，而 `d_src` 是 helper 侧 row-plan 核的产物（在 `prepare()` 之后）→ **循环依赖**。 |
| **上腿分段 D2H，与 MMQ 重叠** | MMQ 只要 ~0.9 ms，D2H 要 ~9 ms → **重叠最多省 0.9 ms**（算力比传输快 10 倍）。收益不成比例。 |
| **把结果 D2H 提前发**（不等 down product 全完） | 结果行由 `dn.dst = m.d_out`（:819）一次写全，**要等 down product 全部完成**才有可发的数据。 |
| **上腿 H2D 移出主卡计算 stream** | H2D 的目的地 `eparts` 被**本层** `moe_end` 读（`q36_engine.cu:3102`），而 `moe_end` 与下一层 attention/router 同一 stream FIFO → 挪走必须插 `cudaStreamWaitEvent` → 4 ms 回到关键路径，**省 0**（对抗验证已判 refuted）。 |
| **走 P2P** | 这台机 12 个有序对的 `CanAccessPeer` 全 0（MEASURED，2026-10-20），P2P 完全不可用。 |

## 3. 定案：上腿 fp16（+ 下腿 fp16，同一个旗标）

### 3.1 上腿 fp16（主收益）

`launch()` 的上腿 D2H（:837）传 `n × n_embd × 4` 字节 fp32；`wait()` 的主卡 H2D（:832-834）传同样多。

**改成 fp16**：
- **字节减半**：helper D2H 11.2 → 5.6 MiB（~9 → ~4.5 ms）；主卡 H2D 11.2 → 5.6 MiB（~4 → ~2 ms）。
  **合计省 ~6.5 ms/层**（DERIVED，Gen1 x4 计价）→ 47.7 → ~41.2 ms/层
  → 268.5 × 47.7/41.2 ≈ **311 tok/s（+16%）**。
- **改动点**：
  1. `launch()`：在 D2H 前把 `m.d_out` 的 fp32 打包成 fp16（新增一个 `d_out16` 缓冲 + 一个打包核），
     D2H 改为传 `n × n_embd × 2` 字节。**行数、行序、每行的语义都不变**（只是每行的元素宽度变了）。
  2. `wait()`：H2D 传 fp16 到主卡侧的一个新缓冲（`h_out16`），主卡侧**解回 fp32**再写 `eparts`
     ——或者更省：让 `k_moe_out`（`q36_engine.cu:2743`）读 fp16。但后者动 combine，风险高；
     **先做前者**（多一次主卡侧的解包核，~11.2 MiB 的写+读，可接受）。
  3. `p2p` 分支**不动**（这台机不走；P2P 机器上行为完全不变）。
- **旗标**：`STRATA_Q36_PEER_F16=1`（新 opt-in，默认关）。
- **数值风险**：结果行从 fp32 变 fp16 → **数值变了**。必须：
  - 定容差基线（flag on/off 的 logits 对照），容差怎么定要看 P1/P3 的 fp32-vs-fp32 噪声；
  - 若容差不过，退回 fp16 只用于**传输**、解回 fp32 后**在主卡上重新加权**（不行，那还是 fp16 精度）；
    真正的退路是 **bf16**（P100/sm_60 不支持 bf16）或 **fp16 + 误差补偿**。**先量容差再定。**

### 3.2 下腿 fp16（次收益，同一个旗标）

下腿 H2D（:674-677）传 `n_tok × n_embd × 4` fp32 = 4 MiB/helper/层 → 3.4 ms。
fp16 → 2 MiB → ~1.7 ms，**省 ~1.7 ms/层**。
- helper 侧收到 fp16 后**照旧自己量化 q8_1**（`MMQ::quantize` 的输入从 fp32 变 fp16——**注意这改变了
  量化的输入精度**，数值路径变了，比上腿更敏感）。
- 改动点：`prepare()` 的 D2H 改成 fp16（主卡侧打包）+ helper 侧 `m.d_x` 存 fp16 + 量化前先转回 fp32
  （或者 `MMQ::quantize` 支持 fp16 输入——要查 MMQ 的接口）。

### 3.3 上腿 SUMS（高风险，备选）

flash-next 的 `STRATA_PF_PEER_SUMS`：helper 不回传每行，改为**每 token 一个加权和**（用路由权重加权）。
- **行数**：1365 → 512（每层每 helper）→ 字节 11.2 → 4 MiB（fp32）或 2 MiB（fp16）→ ~3.4 ms / ~1.7 ms。
  **省 ~5.6 ms/层（比纯 fp16 多 ~1 ms）**。
- **代价**：helper 要多收路由权重（down leg 多传 T×8 float = 16 KiB，可忽略）+ 一个 scatter-add 核
  + **主卡 `moe_end` 的 combine 必须改成读每 token 的和**（`q36_engine.cu:2743` 的 `k_moe_out`）
  → **combine 在关键路径上，改错 = 静默错 logits**。
- **旗标**：`STRATA_Q36_PEER_SUMS=1`（默认关）。
- **顺序**：**在 3.1 之后**（先拿便宜的字节减半，再动 combine）。

## 4. 实施顺序与验收

| 步 | 内容 | 旗标 | 预期（DERIVED） | 前置 |
|---|---|---|---|---|
| 0 | 尺子：prompt 每层四段 + host 动作 | `STRATA_Q36_PROMPT_TIMING=1` | 0 | — |
| 1 | **上腿 fp16** | `STRATA_Q36_PEER_F16=1` | **~311（+16%）** | 尺子确认那 ~13 ms |
| 2 | 下腿 fp16 | 同上（同一旗标的另一半） | ~+4% | 1 |
| 3 | 上腿 SUMS | `STRATA_Q36_PEER_SUMS=1` | ~+2%（叠在 1 上） | 1 + 容差基线 |

**验收**：同一台机器、同一模型、P4 旗标串、`--bench-pp 512 --bench-tg 0`，warm = bench 1/2 均值，
对比 268.5；rc=0、IMA=0、nan=0；**且 flag on/off 的 logits 对照在容差内**（fp16 改了数值路径，
这一步不能省）。
**停止条件**：实测收益 < 噪声（pp512 的 bench 1/2 差可达 6%）→ 回退；或 logits 漂移超容差 → 回退。

## 5. 不变量（改这些文件时必须守住）

1. **`eparts` 行契约**：`peer_prow2`（`q36_engine.cu:3700-3745`）是主卡↔3 卡之间**唯一**的行映射表。
   fp16 **不改行数、不改行序、不改每行的语义**（只改元素宽度）→ 契约不变，但**必须核**：
   `k_moe_out` 读 `eparts` 时看到的必须是解回 fp32 后的值。
2. **两个 witness 检查不能绕过**：
   - 所有权交叉核验（`q36_engine.cu:3666-3697`）——静默损坏变明确失败；
   - 重复行检查（:3726-3733）——两 entry 落同一行 = 静默错 logits。
3. **p2p 分支不变**：P2P 机器上的行为必须逐字节不变（本机不走，但别的机器走）。
4. **默认路径逐字节不变**（AGENTS.md）：旗标关时，`launch()`/`wait()` 的字节数、核的发射、行的内容
   与今天完全一致。
5. **旗标解析**：和其它 q36 旗标一样，`getenv` 读一次、只认精确的 `"1"`
   （见 `q36_engine.cu:1911` 的 `helper_timing()` 的写法）。

## 6. 尺子（步 0）为什么是前置

47.7 ms/层里**已算清的只有主卡的 ~1 ms**；`peer_plan` 的 host 循环、3 次阻塞 D2H、每 helper 的
`cudaStreamSynchronize`（:706）、`:3022` 的 sync、helper 的 ev-wait —— **这些都还没被量过**。
上面 §1 的 13 ms 是**算出来的**（11.2 MiB × Gen1 x4），不是测出来的。
**尺子跑完才知道：这 13 ms 是不是真的、有多少落在 ev-wait 上、还剩多少在别处。**
