# q36：让 helper 不接主卡常驻的 39 个专家，主卡在 prefill 接回它们

日期：2026-10-24。目标机：192.168.1.98（1× Tesla P100-12GB 主卡 + 3× P104-100 8GB helper，16 核）。
前置：`plans/2026-10-24-q36-peer-owner-encoding.md` 的编码修复已落地并实测（prompt 371.4 → 462.0 tok/s）。

## 0. 一句话

修好编码后，主卡在 prefill 的 peer 层上**一个专家都不算**（`n_local = 0`），主卡 GPU 每层只剩
`primary_work = 0.745 ms`（shared expert），而 helper 的 `helper_wait` 涨到 13.5 ms/层（成了关键路径）。
主卡 VRAM 里白占着 39/256 专家（2.86 GiB 的 `L.blobs`）。**让 helper 不接这 39 个，主卡就变成
「prefill 算 39/256」，且权重 D2D 拷（不花 PCIe），主卡终于有活干。**

## 1. 一个必须想清楚的点：改 helper 缓存 = decode 也跟着变

helper 的专家缓存是**启动时**用 `RemoteExperts::open()` 建的（`q36_engine.cu:6311`），
**prefill 和 decode 共用这一份**：

- prefill 的 `peer_owner` 表（归属表）=「helper 声称它持有 (layer, e)」—— 由 `PeerPrefill36::owns()` 读
  helper 的 `slot_ptr()` 建成。
- decode 的 `moe_helpers()` / `RemoteExperts::begin()` 里，helper 拿哪条 entry，看它自己的
  `cache_.slot_of(layer, e)`（`remote_experts.cpp:459`）—— **同一张表**。

所以「helper 不接 39 个专家」**不是只动 prefill**，是动 `RemoteExperts::open` 的建表逻辑，
**decode 那 39 个也回到主卡**。这正是用户上一条问的：「decode 阶段 helper 也不算这 39 个吗？」
答案是——**改完之后，两条路一致：都不算**（现在 decode 本就不算，见 §13.9；prefill 改完也不算）。

## 2. 改法（最小，旗标 opt-in）

`open_helpers()` 给 `RemoteExperts::open()` 传的是 `empty_primary`（0 槽）—— 故意让
`primary.slot_of(...) < 0` 判据**恒真**（`remote_experts.cpp:262`），于是 `ranked` 里每个
(layer, e) 都被 helper 认领。这就是「全部 10240 = 40×256 专家全归 helper，`expert % 3 == k`」的来源。

要主卡接回 39 个，把「primary 声称持有这 39 个」喂给 open()：

- **新增一个 `Hy` 的 slot_of 视图**：主卡现在每层 `L.n_slots = 39`（`expert-vram` 默认放 39 个最热专家，
  `L.blobs` 2.86 GiB）。`slot_of[x] = x`（前 39 个 id 常驻）。把这 39 个 (layer, e) 做成
  `primary.slot_of(l, e) >= 0` 的 ExpertCache，传给 open() 替代 `empty_primary`。
- open() 的判据 `primary.slot_of(l, e) < 0 && !claimed && !picked` 就会**跳过这 39 个**：
  它们不进任何 helper 的 ranked → 3 张 helper 各少 13 个专家 → `peer_owner` 表里这 39 个 = 0（主卡的）。
- 结果：
  - prefill 主卡 MMQ 的 `n_local = 39/256 × n_ent`，`rmax > 0`，MMQ 发射算这 39 个；
  - `stream_kick` 对这 39 个：`H.slot_of[x] >= 0` → **D2D**（`L.blobs` → `stage[b]`），不花 PCIe；
  - decode 主卡 local grouped kernel 补这 39 个（helper 没接 → `helper_mask` 里它们是 0 → 走 local 路）。

**旗标**：`STRATA_Q36_PEER_KEEP_RESIDENT=1`（只认 `"1"`，函数局部静态）。开 = open() 传真实 39 个 slot 的
primary cache；关 = 维持 `empty_primary`（现状逐字节不变）。prefill 的 peer 路径本就该开着
`STRATA_Q36_PREFILL_PEERS`，这个旗标是它的精修，不单独生效。

## 3. 三个必须核对的点（§5 原文，现在更具体）

1. **这 39 个专家有多热** —— `slot_of[x] = x` 按 id 前 39 个，但路由**不均匀**。热专家给主卡
   （sm_60，MMQ 走 2.2× 慢回退）= 把最热的活放最慢的卡上。**先量**：跑一次现在的配置，
   看 `hy[li].score`（`peer_plan` 已有 `H.score` 累加）或 `route` 的 top-39 id 分布，确认「前 39 个 id
   最热」。若分布是「id 0..38 恰好最热」→ 给主卡是对的；若最热的是别的 id（rebalance 换过）→ 主卡接到的是
   冷专家，白给。
2. **`rebalance()` 会换主卡 VRAM 里的专家** —— 归属表 `peer_owner` 是启动时建一次、静态的；
   `rebalance()`（decode 每 32 token，`kRebalanceEvery`）按 `H.score` 把 39 个槽换进换出。
   换出来的专家（主卡已有 slot）：`peer_owner` 仍说主卡的 → 主卡算（权重 D2D，**对**）；
   换进来的专家（helper 有）：`peer_owner` 说 helper 的 → helper 算（**对**）。
   **正确性不依赖 rebalance 关掉**（每张 (layer,e) 归且只归一方，rebalance 只是改主卡槽位，不改 helper 表），
   但**性能会漂**：主卡算的 39 个随 rebalance 换血，热度跟着变。
3. **主卡 sm_60 的 MMQ 走 2.2× 回退** —— helper sm_61 是真 `__dp4a`。主卡多算 39/256（15%）的行，
   **未必省 15% 的时间**，甚至可能反慢（把 15% 的行从最快的 helper 挪到最慢的主卡）。
   要实测：`primary_work` 从 0.745 涨到多少 vs `helper_wait` 从 13.5 掉到多少，净 forward 是升还是降。

## 3b. 实测结果（2026-10-24，192.168.1.98，重启后）

二进制 `bin/strata-q36.keep8`（md5 `d426ed5b`，改法：helper 的 `ranked` 过滤掉主卡 `slot_of ≥ 0` 的专家；
对照 `bin/strata-q36.own` = 编码修复基线）。`STRATA_Q36_PEER_KEEP_RESIDENT=1` 开 = K2。

| | K1（关，编码修复基线） | K2（开，主卡接回 39 个） |
|---|---|---|
| helper 专家数 | 3440/3400/3400 | **2920/2880/2880**（各 −13/−12/−12，共 37 个给主卡） |
| pp512 bench 2 | 461.6 | **461.8**（持平） |
| pp512 bench 0 | 437.8 | **474.5** |
| tg64 bench 2 | 51.8 | **45.3**（**−12.5%**） |
| IMA / nan / Xid | 0/0/0 | 0/0/0 |
| logits（37 行强制定 token 流） | — | **argmax 37/37 同，max\|diff\| = 0.764**（不同：这 37 个专家从 helper 挪到主卡算，数值路径变了） |

**结论：不合并。**（性能负 + 数值也不再逐字节）
- **prompt 持平**（461.6 → 461.8）：主卡接回这 39 个专家后，prompt 没变快。主卡 MMQ（sm_60，2.2× 回退）
  算这 39 个的时间，和被省下的 `helper_wait` 抵消。
- **decode 掉 12.5%**（51.8 → 45.3）：这 39 个最热专家改由主卡 local grouped / CPU pool 补，
  decode 路径变慢。正是 §3 点 3 担忧的实锤：把最热专家从最快卡（helper sm_61 真 dp4a）挪到最慢卡
  （主卡 sm_60），净收益为负。
- **正确性没破**（IMA/nan/Xid 全 0），但**性能是负的**，所以 `STRATA_Q36_PEER_KEEP_RESIDENT` 保持
  opt-in 旗标，**默认关**（= 现状逐字节不变）。这个旗标本身留在代码里作参考，但「让主卡接回 39 个」
  这条路**不走**——主卡留 0 专家（编码修复后的现状）是这台盒子的最优解。

**改法落地时踩的两个坑**（记录在案）：
1. 第一版用 `open_sized` 伪造 primary 的 15360-slot ExpertCache，崩溃——`n_layers × n_expert` 的驻留表
   2.76 GiB，主卡只剩 0.55 GiB。放弃，改成**直接过滤 helper 的 `ranked` 列表**（不分配任何东西）。
2. 热度探针（§4 的 keep6）实测：稳态下主卡 39 个常驻槽位占窗口 57.2% 的路由行（是热的），但主卡
   sm_60 慢 2.2×，热点挪到主卡反而拖慢 decode——热度数字**不能单独作为「接回是对的」的依据**，
   要看净 forward / tg。

## 4. 待做 / 验证

- [x] 热度探针（`STRATA_Q36_PEER_KEEP_RESIDENT` 累计 + bench 末行）—— **做了但打不出来，放弃**：
      2026-10-24 实测，累计点 `if (lhy(li))`（`q36_engine.cu:3221`）内的 `pp_kr_tot` 始终为 0，
      bench 末打印块 `if (E.pp_kr_tot > 0)` 从不成立（远端 5 个二进制 keep/keep2/keep3/keep5 全部：
      字符串在二进制里、代码在源里、`lhy` 返回 true（39<256）、累计点可达，但计数为 0）。
      根因已定位（`bin/strata-q36.keep6` 修好）：累计点原来放在 prompt MMQ 分支的
      `if (lhy(li))` 块里，但 bench 跑时该块确实执行，问题在**读的是 `h_cnt`（rcnt 的 D2H 回读），
      而 `rcnt` 已被 route 核按 `peer_owner` 跳过了 helper 的专家** —— 修好编码后 `rcnt` 只数主卡
      自己的专家，`h_cnt` 在 helper 的专家上是 0，`pp_kr_tot` 累的是「主卡自己的行」而不是「窗口总行」。
      修法：把累计挪到 `moe_hybrid_cpu` 的 `ev_route` 自旋之后（那里 host 手里有完整的 `h_ids` 路由，
      prompt chunk 和 decode step 都经过），`bin/strata-q36.keep6`（md5 `786d16b6`）实测：

      | bench pass | 主卡常驻槽位占窗口行的份额 |
      |---|---|
      | bench 0（热机首遍） | **15.8%**（3233/20480） |
      | bench 1 | 46.9%（19199/40960） |
      | bench 2（稳态） | **57.2%**（35165/61440） |

      **解读**：稳态下主卡那 39 个常驻槽位吃掉了窗口 57% 的路由行。这 39 个 id 是**热的**（路由不均匀，
      前 39 个 id 恰好是最常被路由的 —— 符合 `rebalance()` 把它们排进去的设计）。
      但最热的一层只有 8/512 行（1.6%），最冷的一层 0/512 —— 热度**逐层差异极大**：
      有些层的 39 个常驻专家几乎不被路由（冷），有些层吃到 8 行。

      **对 §5 决策的意义**：主卡接回这 39 个**不是白给**（57% 的行是它们的），
      但 sm_60 的 MMQ 走 2.2× 回退，把这 39 个（= 窗口 57% 的行）从最快的 helper（sm_61 真 dp4a）
      挪到最慢的主卡，**净 forward 大概率不降反升** —— 主卡那 57% 的行算得比 helper 慢 2.2 倍。
      这正是点 3 的担忧实锤。所以 §5 的改法（helper 不接 39 个）**预期是负收益**，
      实测前不应盲目合并。先看下面的实测，再决定合并还是回退。
- [x] 改法落地（`bin/strata-q36.keep8`，§3b）：helper 的 `ranked` 过滤主卡常驻专家，旗标 `STRATA_Q36_PEER_KEEP_RESIDENT=1`
- [x] 远端实测：pp512 持平（461.6→461.8），**tg64 −12.5%（51.8→45.3）**，对照 `.own` 基线
- [x] IMA=0、nan=0、rc=0、Xid=0
- [x] 点 3 反效（decode 掉）→ **不合并**，旗标保持默认关（现状逐字节不变），记录在 §3b

## 5. 风险与不变量

- **旗标 opt-in**：`STRATA_Q36_PEER_KEEP_RESIDENT=1`，默认 `empty_primary`（现状逐字节不变）。
- **改的是 open() 的入参，不是数值路径**：每个 expert 还是同一个算子，只是归属从 helper 挪回主卡。
  行契约、`eparts` 的行序、每个数的数值路径、三个 witness 都不动。
- **旧二进制保留**：`bin/strata-q36`（`b7d53f977`）不动；对照组用 `bin/strata-q36.own`（`5bdddd29`，
  编码修好的，作为「主卡 0 专家」基线）和 `bin/strata-q36.keep`（本步）。
