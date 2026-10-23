# q36 P4 prompt 的优化候选（排名版，2026-10-23）

配套：[2026-10-21-q36-peer-ima-root-cause.md](2026-10-21-q36-peer-ima-root-cause.md) §13.4–13.5。
状态：**候选排名稿**。所有数字标 MEASURED / DERIVED / ESTIMATE。每条 opt-in，默认路径逐字节不变。

## 0. 本稿的归因与边界

**本稿依据的归因**（不同于旧稿的"墙在主卡"）：P4 的 47.7 ms/层墙是 **host-bridge 路径**（Gen1 x4 helper 链路）的
**串行 + 并行复合体**，不是主卡 GPU 算力、也不是卡间总字节数。
证据链：1-helper = 205 tok/s，3-helper = 268.5 tok/s，prompt **只 +31%**（decode 却 ~3x）。这同时排除
"墙 = 主卡算力"（1↔3 helper 主卡活完全一样，应 0%）和"墙 = 固定总桥字节"（总流量与 helper 数无关，应 0%）。
+31% 是**随 helper 数线性增长的串行段**的特征：peer_plan 的 host 循环对每张 helper 调 `prepare()`，每张各做一次
**阻塞 4 MiB D2H 同一份激活**（q36_peer_prefill.cu:652，:706；q36_engine.cu:3754-3758）——3 次冗余串行拷贝。
这段下行腿限制了并行增益。墙的余量是 helper **上行腿 D2H**（每 11.2 MiB fp32 @ Gen1 x4 ~9 ms，3 张并行取 max）
+ 主卡侧**串行 H2D**（:832，~4 ms @ Gen3 x8）——这一段**已经随并行缩放**（就是实测 +31% 的来源）。
主卡自身 GPU 算力仅 ~15/47.7 ms（~32%，ESTIMATE）。未解释缺口 ~4-10 ms（见 §13.4/§13.5）。

**已实测、不再讨论的方向（任务明确关闭，不要重提）：**
| 方向 | 结论 | 依据 |
|---|---|---|
| 砍桥字节（flash-next F16 / SUMS，按传输降字节） | ✗ 不改 plateau | 任务规则 (a)：prompt 不随 helper 数 scaling，总桥字节与 helper 数无关 |
| CPU 池进 P4 的 prompt（MMQ 分支内） | ✗ 无目标 | 主卡在 peer-served 层专家算为 0（`n_local=0`），CPU 池无活可接（任务规则 b） |
| 靠 `--hybrid-rows` 让 CPU 上场 | ✗ 死路 | 切掉 MMQ 分支 → peer 出局 → 268.5→195.0（MEASURED） |
| 主卡另开 stream 算专家 | ✗ 无活 | P4 主卡 `n_local=0`，没有专家可算；主卡算力闲置但无专家可搬 |

> 注意：**"共享一次 D2H（消冗余 3× 拷贝）"≠ "砍桥字节（F16/SUMS）"**。前者是消除*同一份数据的重复传输*，
> 直接打掉"随 helper 数线性增长的串行段"；后者是降低*每次传输的字节数*，任务已判定为二阶。两者机制不同，本稿只取前者。

**没有任何 flag-only 候选能直接回答"墙在哪"**——缺的是测量数据，必须加代码埋点（候选 #0）。
因此 #0 按任务规则 4（先量尺子）排第一，而非因其是 flag-only。

## 1. 排名总表

| 排名 | 候选 | 旗标（opt-in） | 预期上限 tok/s | 依据 | 风险 | upside 性质 |
|---|---|---|---|---|---|---|
| **#0** | 给 prompt 路径埋每层时间线 | `STRATA_Q36_PROMPT_TIMING=1` | 0（前置） | 基线 268.5 | 低 | — |
| **#1′** | **上腿 fp16 + SUMS 减字节**（所有 verify 指向的那 9 ms） | `STRATA_Q36_PEER_F16=1` / `STRATA_Q36_PEER_SUMS=1` | 待 #0 量出那 9 ms 再算 | Gen1 x4 × 11.2 MiB | 中 / 高 | DERIVED |
| **#2′** | 下行腿在 helper stream 上异步/流水化 | `STRATA_Q36_PEER_ASYNC_DOWN=1` | ~285（+5–10%） | DERIVED | 中-高 | ESTIMATE |
| **#3′** | ~~上行腿 H2D 移出主卡计算 stream~~ | — | **作废** | 依赖不成立（见 §1.1） | — | — |
| **#4′** | ~~共享一次激活 D2H~~ | `STRATA_Q36_PEER_SHARED_DOWN=1` | **~274（+2%）**（原 +16% 已修正） | DERIVED @ 主卡链路 | 低-中 | DERIVED |
| **#5′** | ~~per-helper prepare() 并发~~ | `STRATA_Q36_PEER_PARALLEL_PREPARE=1` | **~285（+7%）**（原 +15~25% 已修正） | DERIVED（波前偏斜） | 中 | ESTIMATE |
| **#6** | 核查 router f32 SGEMM（主卡侧，受 32% 上限封顶） | `STRATA_Q36_ROUTER_CUBLAS=1`（待定） | ~280（+0–10%，不确定） | ESTIMATE（最大不确定项） | 低 | ESTIMATE |

> **排序已按 §1.1 的对抗验证更正过**：原 #1（共享 D2H）和 #2（prepare 并发）的 upside 各被打了
> ~7 倍和 ~2 倍，#3 作废。**新的头号桥侧靶是 #1′（减字节）**——因为所有 verify 都指向 helper 的
> ev-wait 那次 D2H（11.2 MiB fp32 @ Gen1 x4 ≈ 9 ms）。

> 所有 tok/s upside 都是 ESTIMATE/DERIVED：由归因的墙模型（各段 ms 本身多为 ESTIMATE/DERIVED）推算，
> 非实测。换算：`新 tok/s = 268.5 × 47.7 / (47.7 − 省下 ms/层)`。

### 1.1 ⚠️ 对抗验证后的更正（2026-10-23，4 个独立 skeptic，**4/4 全部 refuted**）

上表的排名和 upside 是**归因模型推出来的**，随后一次对抗验证把它们逐条查了一遍。**结论：#1 / #2 / #3
三条的 upside 都不成立或大幅缩水，#0 的规格要改一处。** 下面是被打掉的部分，本稿其余部分按此更正读。

| 候选 | 裁决 | 硬理由 |
|---|---|---|
| **#1 共享一次激活 D2H** | **refuted（收益缩水 ~7×）** | 被删的拷贝**走错链路**：`:652` 那个 D2H 的源是**主卡的设备内存**（引擎在 `q36_engine.cu:3020` 自己 D2H 下来、`:3032` 传进去的），走**主卡链路**（Gen3 x8），**不是** helper 的 Gen1 x4。真正的 Gen1 x4 流量是**另一条**拷贝（`:674` 的 H2D `h_x → helper d_x`），而 #1 的改法**不动它**。按主卡链路 4–8 GB/s 计价 → 省 **~1–2 ms/层**（不是 6.8）→ 上限 313 → **~274（+2%）**。**基本不值得做。** |
| **#2 prepare() 并发** | **refuted（前提被实测反证）** | 1→3 helper 是**变快**（205→268.5，+31%）。若那个串行循环是 ~10 ms 的墙，**3 个 helper 会让它变慢**（3.4→10.2 ms）而不是变快 → 循环是**从属**的。临界路径上只有**波前偏斜 ~3.4 ms**（helper0 在 t≈0、helper2 在 t≈7.4 ms）→ **~+7%**，不是 +15–25%。 |
| **#3 上腿 H2D 移出主卡 stream** | **refuted（依赖不成立）** | H2D 的目的地 `eparts` 被**本层** `moe_end` 读（`:3102`），而 `moe_end` 与下一层 attention/router 在**同一 stream 上 FIFO** → 挪到 copy stream 必须在 `moe_end` 前插 `cudaStreamWaitEvent` → 那 4 ms 回到关键路径，**省 0**。真正的墙在它**前面**的 ev-wait。 |
| **#0 尺子** | **规格要改一处** | 不能用"4 个共享 event + 扩 gp_drain 允许 n>1"——那正是 `q36_engine.cu:2059-2061` 注释里写明的 bug（40 层共用 8 个 event 会被最后一层覆盖）。必须 **per-layer event ring**（如 `ev_prompt[40][4]`）+ per-layer drain。**实施已按这个规格做。** 另：GPU event **抓不到 host 阻塞**（helper 的 stream 上 host 卡住时主卡 stream 空闲）→ **host 墙钟那一半才是 host-bridge 串行段的主信号**。 |

**所有 verify 都指向同一处 —— 那就是现在唯一该先打的靶：**

> **helper 的 ev-wait（`q36_peer_prefill.cu:817-818`）等的那次 D2H：每 helper 每层 11.2 MiB fp32
> 走 Gen1 x4 ≈ 9 ms。**

砍它只有两条路：**减字节**（fp16 → 5.6 MiB；SUMS → 行数 2.7×）或**提链路**（Gen1 x4 是硬件墙）。
注意：砍字节**不改变 plateau**（§13.4：总流量与 helper 数无关），只**降低绝对时间**——两者不冲突，
但必须先量出那 9 ms 是不是真的（**#0**）。

**据此，本稿的 #1 / #2 降级，#3 作废，#4（下行腿流水化）升为与 F16 同级。**

## 2. 候选详情

### #0 先量尺子：prompt 每层时间线  ★前置，必须最先做

- **问题**：47.7 ms/层里已算清的只有 ~1 ms（主卡非专家 GPU 活），**剩 40+ ms 无解释**；
  现有 probe 测不了 prompt——`gp_drain()` 丢弃所有 `n>1` 的 tile（q36_engine.cu:2061-2063）。
- **文件/行**：q36_engine.cu:2839-3102（layer 主体）、:3032（peer_plan 返回）、:3099（peer_prefill_wait）、:3102（moe_end）、
  :2061-2063（gp_drain 的 n>1 跳过）；q36_peer_prefill.cu:652、:706（prepare 内阻塞 D2H + sync）。
- **旗标**：`STRATA_Q36_PROMPT_TIMING=1`（新 opt-in）。
- **改动（默认路径不变）**：旗标开时，在 prompt 的 MMQ 分支上建一条**每层**时间线，四个边界各一个 timing `cudaEvent`：
  1. `P0` 层入口（attention 前） 2. `P1` peer_plan 返回后（主卡该发的都发了） 3. `P2` `peer_prefill_wait()` 后（helper 结果已合进 eparts） 4. `P3` `moe_end` 后。
  另量 host 侧墙钟：3 次阻塞 D2H（:652）各耗时、每 helper `cudaStreamSynchronize`（:706）耗时、peer_plan host 循环耗时。
  事件只在旗标开时创建；`gp_drain` 的 n>1 跳过要扩成"也记录 prompt tile"。
- **预期上限**：0（不改变速度）。**但它是 #1–#5 全部的前提**——没有它，其余都是猜；未解释缺口 ~4-10 ms 也靠它定位。
- **风险**：低。纯测量，不改任何数值路径；旗标关时行为与今天逐字节一致。
- **验收**：同二进制、`STRATA_Q36_PROMPT_TIMING=1` + P4 旗标串、`--bench-pp 512 --bench-tg 0`，打印每层 P0–P3 四段 ms + host 动作累计 ms。
- **停止条件**：无（前置步骤，永远值得做）。

### #1 共享一次激活 D2H（消 3× 冗余拷贝）  ★最高期望

- **问题**：peer_plan host 循环（q36_engine.cu:3754-3758）对每张 helper 调 `prepare(li,hdev,...)` 传**同一** `hdev`；
  而 `prepare()` 在 host route 下**每张卡各做一次阻塞 D2H**（q36_peer_prefill.cu:652-655）写进各自 pinned `h_x`（:452 各自 `cudaHostAlloc`）。
  同一份 4 MiB 激活被拷了 3 遍——这就是"随 helper 数线性增长的串行段"。
- **文件/行**：q36_engine.cu:3754-3758、q36_peer_prefill.cu:652-655、:452。
- **旗标**：`STRATA_Q36_PEER_SHARED_DOWN=1`。
- **改动**：在 peer_plan 循环**之前**对激活窗口做**一次** D2H 到共享 pinned 缓冲；`prepare()` 改为接收已填好的 host 指针、跳过自身拷贝。
  顺序保证：该共享缓冲在下一层 D2H 前不被覆盖（现在每卡各一份天然安全；共享后需显式 barrier 或双缓冲）。
- **预期上限**：**~274 tok/s（+2%）——原估 +16% 是错的，已修正。** 依据：被删的两次拷贝走**主卡链路**
  （源是主卡设备内存，`q36_engine.cu:3020` D2H 下来、`:3032` 传进 `prepare`），按 4–8 GB/s 计价
  → 省 **~1–2 ms/层**（不是按 Gen1 x4 算的 6.8 ms）→ 墙 47.7→46 左右。**（DERIVED，修正后）**
- **⚠️ 对抗验证的硬理由**：真正的 Gen1 x4 流量是**另一条**拷贝（`q36_peer_prefill.cu:674` 的 H2D
  `h_x → helper d_x`），而本改法**不动它**。所以"消掉 3 次冗余"消掉的是**快链路上的**冗余。
- **风险**：低-中。改 `prepare()` 签名与 `h_x` 所有权；需保证共享缓冲的读写顺序不被下一层覆盖。无数值路径变化。
- **验收**：`STRATA_Q36_PEER_SHARED_DOWN=1` + P4 旗标串，同模型、`--bench-pp 512 --bench-tg 0`，warm=bench 1/2 均值，对比 268.5。
- **停止条件**：**收益 +2% 量级在噪声边缘（pp512 的 bench1/2 差可达 6%）→ 大概率直接放弃**；
  或任何 IMA/nan（拷贝源/时序变了，需 IMA=0 验证）。**优先级已降到 F16/SUMS 之后。**

### #2 per-helper prepare() host 循环并发（3 线程）  ★与 #1 互补

- **问题**：即使共享了 D2H，peer_plan 循环仍是**串行**：helper1 prepare（等 ~3.4 ms）→ helper2 → helper3 ≈ **~10 ms/层纯串行**
  （每 prepare 仍有 H2D + 4 个 row-plan 核 + 8 B 回读 + sync，q36_peer_prefill.cu:706）。
- **文件/行**：q36_engine.cu:3754-3758、q36_peer_prefill.cu:652、:706。
- **旗标**：`STRATA_Q36_PEER_PARALLEL_PREPARE=1`。
- **改动**：3 个 host 线程各调一个 `prepare()` 并发；各自 sync 仍做一次但**互相不等**；`peer_plan` 继续前需 join barrier
  （launch 需要 `m.n_owned`/`m.max_rows` 定网格，必须所有 prepare 完成）。
- **预期上限**：**~285 tok/s（~+7%）——原估 +15~25% 是错的，已修正。** 依据：1→3 helper **变快**
  （205→268.5，+31%）证明这个串行循环**不是墙**（若是墙，3 卡会因串行段 3.4→10.2 ms 而变慢）。
  临界路径上只有**波前偏斜 ~3.4 ms**（helper0 在 t≈0 拿到活、helper2 在 t≈7.4 ms）。**（ESTIMATE）**
- **风险**：中。`prepare()` host 状态每卡独立（安全）；需 barrier 保证 launch 前完成。
- **验收**：`STRATA_Q36_PEER_PARALLEL_PREPARE=1` + P4 旗标串，同 bench。
- **停止条件**：**收益 ~+7% 落在噪声边缘 → 很可能不值得做**；墙降 < 噪声，或线程竞态（IMA/nan）→ 回退。

### #3 上行腿 H2D 移出主卡计算 stream（异步重叠）  ✗ **已作废（对抗验证：依赖不成立）**

- **原想法**：host route 下结果行上腿在主卡侧是**串行 H2D**（q36_peer_prefill.cu:832-834，~32 MiB/层，~4 ms @ Gen3 x8），
  跑在主卡计算 stream 上，挡住下一层 attention。native 引擎用 `m.copy` 专门做这种重叠（prefill.cpp:772,2843-2867），q36 没有。
- **✗ 作废理由（对抗验证，硬）**：H2D 的目的地 `eparts` 被**本层** `moe_end` 读（`q36_engine.cu:3102`），
  而 `moe_end` 与下一层 attention/router 在**同一 stream 上 FIFO**。挪到 copy stream 必须在 `moe_end`
  前插一条 `cudaStreamWaitEvent` → 那 4 ms 立刻回到关键路径 → **省 0**。本层内又没有别的活可重叠
  （只剩 H2D 自身）。
- **真正的墙在它前面**：helper 的 ev-wait（`q36_peer_prefill.cu:817-818`）等的那次 D2H（~9 ms），
  那才是该动的（→ 减字节，见 #4/F16/SUMS）。
- **旗标**（若日后重开）：`STRATA_Q36_PEER_ASYNC_UP=1`。**当前不实施。**

### #4 下行腿在 helper stream 上异步/流水化  ★低置信

- **问题**：下行腿 D2H（激活）现在是 host 线程里的阻塞拷贝，串行占用。可在各 helper stream 上 `cudaMemcpyAsync` 提前发起，
  与 3 张 helper 各自的 H2D 及主卡 attention 重叠。
- **文件/行**：q36_peer_prefill.cu:652-655、q36_engine.cu:3032（peer_plan 发起点）。
- **旗标**：`STRATA_Q36_PEER_ASYNC_DOWN=1`。
- **改动**：下行腿改为异步、在 peer_plan 前一次性发起，重叠进主卡/helper 计算。需重排：激活量化（`MMQ::quantize` 在 :3041，
  晚于 peer_plan :3032）若要走"发已量化激活"路径则需顺序重排或加一次量化。
- **预期上限**：~285 tok/s（**+5–10%**，DERIVED）：下行腿 ~10 ms 若与 ~15 ms 主卡算力完全重叠则省上界 ~10 ms，
  但受主卡忙度限制，实际更小。**（ESTIMATE，低置信）**
- **风险**：中-高。跨 stream 时序/顺序重排易错；若涉及发量化激活则动数值路径（需 fp16/q8_1 对照）。
- **验收**：`STRATA_Q36_PEER_ASYNC_DOWN=1` + P4 旗标串。
- **停止条件**：墙降 < 噪声或 IMA/nan → 回退。

### #5 核查 router f32 SGEMM（主卡侧，受 32% 上限封顶）  ★不确定，依赖 #0

- **问题**：归因把 router `f32mm`（q36_engine.cu:2943，537M FLOP，自定义 `k_sgemm_nt`）列为**最大不确定项**（2–72 ms ESTIMATE 区间）。
  若其实很大，主卡侧仍有可榨空间。
- **文件/行**：q36_engine.cu:2943（`k_sgemm_nt<4>`，grid (4,8,4)=128 blocks）。
- **旗标**：`STRATA_Q36_ROUTER_CUBLAS=1`（待 #0 定，可能只是改 `f32mm` 分支用 cuBLAS）。
- **改动**：在旗标开时让 router 走 cuBLAS SGEMM（若其在该小矩阵更快），或反之确认自定义核已最优。
- **预期上限**：~280 tok/s（**+0–10%**，ESTIMATE 且不确定）：主卡算力仅占 ~32%（~15 ms），即便 router 占其中大半、
  减半也只省 ~7 ms；且若 router 其实很快（自定义 split-K 已优），则几乎无收益。**受 32% 上限封顶。**
- **风险**：低（仅换 GEMM 后端）；但若数值不一致需容差对照。
- **验收**：`STRATA_Q36_ROUTER_CUBLAS=1` + P4 旗标串，off/on logits 对照。
- **停止条件**：#0 显示 router 实测 < 5 ms（已最优）→ 整条废弃；或 cuBLAS 更慢 → 回退。

## 3. 执行顺序与验收

| 步 | 内容 | 类型 | 前置 |
|---|---|---|---|
| **#0** | prompt 每层时间线 + host 动作 | 改代码（opt-in） | — |
| **#1** | 共享一次 D2H | 改代码（opt-in） | #0 |
| **#2** | prepare 并发 | 改代码（opt-in） | #0（叠 #1 最佳） |
| **#3** | 上行腿异步 | 改代码（opt-in） | #0 |
| **#4** | 下行腿异步 | 改代码（opt-in） | #0 |
| **#5** | router SGEMM 核查 | 改代码（opt-in） | #0（看 #0 结果再决定） |

**每步验收（机器 192.168.1.98，一次一台引擎）**：
```
STRATA_REMOTE_ZEROCOPY=0 STRATA_Q36_HELPER_ASYNC=1 STRATA_Q36_HELPER_PARALLEL=1 \
STRATA_Q8_SM60=1 STRATA_Q36_PREFILL_PEERS=1 STRATA_Q36_PREFILL_PEERS_HOST=1 \
<本步旗标=1> \
strata launch --model <Q4_K_XL> --max-context 2048 --no-warmup --bench-pp 512 --bench-tg 0
```
warm = bench 1/2 均值；rc=0、IMA=0、nan=0；与 P4 基线 **268.5 tok/s** 比。

**每步停止条件**：实测收益 < 噪声（pp512 bench1/2 差可达 6%）→ 回退该步。

**默认路径逐字节不变**：每个旗标关时，`q36_engine.cu`/`q36_peer_prefill.cu` 行为与今天完全一致
（新增事件/拷贝/stream 仅在旗标开时发生；`gp_drain` 的 n>1 跳过扩展仅在 `STRATA_Q36_PROMPT_TIMING=1` 时生效）。

## 4. 风险清单（跨所有步）

- **静默错 logits**：任何动 H2D 时序/stream 归属的方案（#1–#4）改的是**传输时序**而非数值，但时序偏移可能让
  combine 读到未就绪数据 → 必须有 flag on/off 的 logits 对照（IMA=0、nan=0 不够，需容差基线）。
- **行契约**：`eparts` 行分配（`peer_prow2`，q36_engine.cu:3700-3745）是主卡↔3 helper 的唯一映射；任何改"谁算哪些行"必须同步改它，
  保留两个重复行检查（:3726-3733）。
- **helper 侧 live cache 一致性**：`peer_plan` 所有权交叉核验（:3666-3697）把静默损坏变明确失败——**任何改动不得绕过**。
- **旗标关时逐字节一致**：项目规矩（AGENTS.md）——新功能 opt-in、默认路径与上一版逐字节相同。
