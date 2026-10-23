# q36：让 CPU 池在 peer 分工里也吃 prompt（设计稿）

状态：**设计稿，未实现**。2026-10-23。
目标：在 P4（1 主卡 + 3 P104 helper）形状下，让**CPU 池也参与 prompt 的专家计算**——
现在 CPU 池在 P4 里完全不参与（见下"事实 2"）。

## 已钉死的事实（MEASURED / 代码核到）

1. **分支互斥**：`q36_engine.cu:2946`
   `hy_rows = lhy(li) && (T <= hybrid_t || !main_hybrid)`。
   - `hy_rows` 真 → hybrid 批次：`moe_hybrid_gpu`（主卡常驻专家，:2972）+
     `moe_hybrid_cpu`（非常驻专家交 CPU 池，:3091）+ `moe_helpers`（若 T<=MAXT）。
   - `hy_rows` 假 且 `T > kGemmT && use_mmq` → **MMQ 分支**（:2974）：主卡 MMQ 算自己的，
     helper 走 peer 计划（`peer_plan_or_exit`，:3032）。**这条分支没有 `moe_hybrid_cpu`。**
   - P4 = MMQ 分支（T=512 > kGemmT，use_mmq 真）。P1（`--hybrid-rows 512`）= hybrid 批次。

2. **P4 里 CPU 池不参与 prompt（推翻我 2026-10-23 那条口头说法）**：
   `moe_hybrid_cpu` 只在 `if (hy_rows)` 块内被调（:3091）。P4 走 MMQ 分支 → `hy_rows` 假 →
   CPU 池对 prompt 一行都不算。（`moe_hybrid_cpu` 内部 :3951 那句 "The helper path no longer skips
   the pool" 说的是 **hybrid 批次里的 helper 路径**，不是 peer/MMQ 分支。）

3. **peer 的行契约**（`q36_peer_prefill.cuh`，`q36_engine.cu:3700-3745`）：
   - 每个 entry 按 `own[expert]` 分桶：`kPeerNone` → 主卡，其余 → 对应 helper。
   - 行分配：主卡的 entry 先占 `[0, n_local)`，然后 helper 0/1/2 依次接，
     每 helper 内按 entry 序。`base_j = n_local + sum_{i<j} rows_i`。
   - `peer_prow2[e]` = entry e 的 eparts 行；`moe_end(T, prow, ...)` 用它把行映射回 entry。

4. **CPU 池的合并机制现成**：`k_merge_cpu`（:752 定义，:4007 调用）按 `d_miss` mask
   把 `cpu_out` 的行合进 `eparts`。CPU 池算完的 `cpu_out[e]` 对应 entry e 的行。

5. **CPU 池的入口条件**：`moe_hybrid_cpu` 需要
   - `H.slot_of[x] >= 0` 的专家（主卡常驻槽）→ 主卡 GPU 算；
   - `slot_of[x] < 0`（非常驻）→ CPU 池算；
   - 路由权重 `H.score` 用于 rebalance，不影响正确性。
   它读 `h_ids`（entry→expert id）和 `h_h`（激活）——**这两个在 MMQ 分支里没有 D2H 到 host**。

6. **`use_mmq` 时的权重来源**：MMQ 分支从 `L.blobs`（主卡 VRAM 里的 int8 专家）读权重，
   CPU 池算非常驻专家时需要 **fp16/q8_1 的 host 侧 blob**（`L.hblobs`）——hybrid 路径正是这么做的
   （`j.blob = L.hblobs + x * L.stride`，:3990）。

## 设计：三层桶 + 一个新分支

把"谁算哪个 entry"从现在的两桶（主卡 / helper）扩成三桶（主卡 / helper / CPU 池），
**在 MMQ 分支里加 CPU 池那一桶**，而不是去动 `hy_rows`（动了会把 MMQ 分支整个切掉，helper 出局）。

### 新增状态

- `peer_cpu_mask`：每层一个 `[n_ent]` 的 uint8（entry → 谁算），但只需要在 CPU 桶非空时存在。
- `peer_n_cpu`：CPU 桶的 entry 数。
- CPU 池的 entry 落在 eparts 的**哪一行**：沿用主卡那段的行空间（`[0, n_local)`），
  在主卡行之后、helper 行之前（或最后）。**建议放最后**（helper 行之后），这样主卡/helper 的
  现有行号完全不变 —— peer executor 的 `base_j` 计算要跟着改，但 `peer_prow2` 仍是唯一的行映射表，
  改一处即可。

### 新增分支：MMQ 分支里调用 CPU 池

在 `q36_engine.cu:3032`（`peer_plan_or_exit`）之后、:3041（`MMQ::quantize`）之前，
插一段「CPU 桶的规划 + 发起」：

```
if (peer_here && peer_cpu_share) {
    // 1) 把 entry→expert 的路由 id 拿到 host（MMQ 分支目前只有 h_peer_ids 在 peer 层拿过，
    //    但那是 peer_plan 内部用的；CPU 池需要 h_ids 全量 + 激活 h_h）
    //    → 在 :3020 那条 D2H 里一并取（ids 已经在取！h_peer_ids = ids 的 D2H，:3020）
    //    → 激活需要新增一条 D2H：T*N*4 字节（fp32）
    // 2) 选桶：entry 的专家既不在主卡常驻槽、也不在任何 helper 手里 → CPU 桶
    //    （这正是 P1 里 CPU 池吃的那批）
    // 3) 规划 CPU 桶的 job（同 moe_hybrid_cpu 的 job 打包，:3980-4000）
    // 4) 发起 pool->run_split_multi_native（CPU 池在独立线程上跑，与主卡 MMQ 重叠）
    // 5) 把结果回传：cpu_out → ecpu (D2H 已被 :4005 那条覆盖)
    // 6) 合并：k_merge_cpu(eparts, ecpu, d_miss_cpu, N) —— 复用现成 kernel
}
```

### 关键约束（必须核，实现前逐条）

- **行空间**：CPU 桶的 entry 必须在 `eparts` 有行。行契约在 `q36_peer_prefill.cuh` 里写死
  （"主卡先，helper 按序"），要加 CPU 段就得改契约 + `base_j` 计算。**这是本设计最脆的一环。**
- **激活来源**：CPU 池要 fp32 激活（`h_h`）。MMQ 分支里 `h` 在 device 上，量化成 `mq_x` 之后
  就没有 fp32 了。要在量化前 D2H 一份 `T*N*4`（512×2048×4 = 4 MiB/层，可接受）。
- **权重来源**：CPU 池算的是**非常驻**专家，要 `L.hblobs`（host 侧 fp16/q8_1 blob）。
  hybrid 路径就是这么用的（:3990）。q36 的 MMQ 分支里 `L.hblobs` 是否已加载？**待核**。
- **CPU 池格式**：`H.fmt` 是按层算好的（hybrid 路径设置）。MMQ 分支的层是否也有 `H.fmt`？**待核**。
- **线程数**：这台只有 16 线程，引擎自己推荐 7（§13.3 P2 实测拉满反而回退）。CPU 池和
  3 helper 同时抢 CPU，线程数可能要比独跑时更保守。
- **`moe_hybrid_cpu` 的 spin**：`while (cudaEventQuery(ev_route) == cudaErrorNotReady)` —— CPU 池
  要等主卡的路由 D2H。新分支里必须先 `cudaEventRecord(ev_route, st)` 再发起。

## 为什么可能有用（假设，待验）

P1（主卡+CPU 池，纯 hybrid）191.1 vs P3（主卡独跑）158.5 vs P4（主卡+3helper）268.5。
如果 CPU 池在 P4 里能吃掉一部分"helper 也没接住"的残余（helper 只接 `own[expert]` 非空的专家，
而 CPU 池能接**任何**非常驻专家，包括 helper 手里那 2/3 之外的部分），理论上有增量。
但增量大小**完全未知**：P4 的 268.5 已经包含 helper 的 +70%，CPU 池能加多少取决于残余量。

## 三条可能的结果（都要能接受）

1. **有增量**（比如 268.5 → 290+）：值得实现。
2. **没增量**（268.5 ± 几 %）：说明 P4 已经是墙（主卡 attention + 1/3 专家 + 每层 sync），
   CPU 加进来只是把已经重叠的计算换了个地方。**这本身就是结论**，写进 §13.4。
3. **变慢**：CPU 池和 3 helper 抢 16 个核，或 CPU 池的 D2H（4 MiB/层）拖累主卡。

## 待核清单核完（2026-10-23，代码核到）

- [x] **`L.hblobs` 在 MMQ 分支的层上已加载** ✓ —— `place_experts` :4820-4823 对**每一个**层
      （`mtp_only` 为假时）把全部专家 blob 写进 pinned host arena。P4 的 `mtp_only` 为**假**
      （:4724 判 `total_main + 64MiB <= budget2` 不成立；banner "40 of 256 per layer in VRAM,
      all 18.75 GiB in pinned host memory" 印证）。CPU 池算非常驻专家的权重**现成**。
- [x] **`H.fmt` 已设置** ✓ —— :4840 `C::native_fmt(...)` 对每个非-`mtp_only` 的层都做，
      且 :4841 校验 CPU/GPU 布局一致（不一致直接报错退出）。CPU 池的 `run_split_multi_native(H.fmt, ...)`
      需要的 fmt **现成**。
- [x] **`eparts` 行空间够** ✓（结论反转，见下）
- [x] **`moe_end` 的 `prow` 映射成立** ✓ —— `moe_end` → `k_moe_out(x, eparts, prow, ewts, ...)`：
      `prow` 是**按 entry 索引 eparts 行**的表，`peer_prow2[e]` 就是 entry e 的行。
      CPU 桶的 entry 只要在 `peer_prow2` 里有行，combine 就自动算它。
- [x] **`k_merge_cpu` 可复用** ✓ —— :752-756 `parts[e] = cpu[e] where miss[e]`，按 **entry** 索引
      `cpu_out`。语义正是"CPU 算的 entry 的行合进 eparts"。
- [x] **CPU 线程数** —— 这台 16 核、引擎推荐 7（§13.3 P2 实测拉满回退）。与 3 helper 并行时
      更保守，**具体值待跑**（不是 blocker）。

### 行空间：结论反转——不用加行，但主卡要"少算几行"

`eparts = dalloc(T * Kp * N)`（:2194），即每层每 window **固定 `T*Kp = n_ent` 行**。
`peer_plan`（:3709-3712）已经把**全部 n_ent 行**分配完：主卡 `[0, n_local)` + 各 helper 段。
所以 **CPU 桶不需要新行**——设计稿原先担心的"行空间不够"不成立。

**但真正的改动点更硬**：主卡 MMQ 现在算的是 `n_local` 行
（`gu.total_rows = n_local`、`dn.total_rows = n_local`，:3046/:3054；`bounds = rbounds`、
`ids = lid`、`n_local = peer_n`，:3035-3036）。CPU 要抢主卡的活，就得把
**`n_local` 拆成 `n_local_gpu`（主卡算）+ `n_local_cpu`（CPU 算）两段**：
- 主卡的 `gu`/`dn` 发射参数（`total_rows` / `bounds` 起点 / `ids`）都要按新段改；
- `k_peer_lid`（:3745）产的 id 表要分段；
- CPU 那段的 entry 的 `peer_prow2[e]` 行由 CPU 的 `k_merge_cpu` 填，主卡不再写那行。

**这是本设计真正脆的一环**（不是行空间）：主卡的 `bounds`/`ids`/`total_rows` 三件套是
MMQ 的发射契约，改错了 = 静默 logits 损坏。

### 设计 v2（按核完的结果改）

CPU 桶 = **主卡 `n_local` 里的一部分**（不是"helper 之外的残余"）：
选哪些 entry 给 CPU，用的是 `moe_hybrid_cpu` 的同一判据
（`H.slot_of[expert] < 0` = 非常驻；`slot_of >= 0` = 主卡常驻槽，主卡 GPU 算），
但**还要排除 helper 持有的专家**（`own[expert] != kPeerNone`）——那些归 helper，不进 CPU 桶。

即：
- entry 的专家有主卡常驻槽 → 主卡 GPU（现状）
- entry 的专家在某个 helper 手里 → 那张 helper（现状）
- entry 的专家**非常驻**且**不在任何 helper 手里** → **CPU 池**（新增第三桶）

**注意这条桶定义与 P1 的差别**：P1（hybrid 批次）里 CPU 池吃的是"非常驻"（`slot_of < 0`），
helper 传统路径要求 `T <= MAXT=8` 所以 prompt 阶段不参与。在 P4 里 helper 吃的是
`own[expert] != kPeerNone` 的专家（peer 计划），CPU 桶吃剩下的"非常驻且非 helper 的"。

## 终局结论：设计前提被推翻，本方向应关闭（2026-10-23）

读码核到一条决定性事实（§13.4 同步修正）：

**P4 里主卡在 served 层上���个专家都不算（`n_local = 0`）。**
- 3 张 helper 按 `expert % 3` 持走了**每层每一个**专家（banner 3440+3400+3400 = 10240 = 40×256）。
- `open_helpers()` 传 **empty_primary** cache（0 槽）→ `RemoteExperts::open()` 的
  `primary.slot_of(...) < 0` 判据**恒真** → helper 可以持有主卡也常驻的专家
  （`q36_engine.cu:3344-3350`）。
- `peer_owner` 由 `peers[r]->owns(l, e)` 构建（:3855-3860）→ 每个专家 `own != kPeerNone`
  → `peer_plan` pass 1（:3700-3705）数出 `n_local = 0` → 主卡 MMQ 无行可算。
- `stream_kick` 注释印证（:3286-3291）："the primary's MMQ on a served layer only computes the
  entries of the experts **nobody else holds**"。

**因此：**
- ❌ **原设计（CPU 池接主卡那 1/3）不成立**——主卡的专家计算是 0，**没有可抢的活**，
  原设计里"主卡 MMQ 发射契约三件套（`bounds`/`ids`/`total_rows`）拆分"这个最脆的点**根本不存在**。
- ⚠️ **CPU 池要参与，只能从 helper 那 10240 个专家里抢一份**给 CPU 算。但：
  - helper 走 **host-route**：fp32 激活 → helper 自己量化 q8_1 → q8_1 算（`q36_peer_prefill.cu:741`）；
  - CPU 池走 **host blob + CPU 反量化**（`H.fmt`，`q36_engine.cu:3991`、:4003）。
  - 两条路**输入格式与量化方式都不同**，"抢"过来不是搬一行代码，是新增一条第三种数值路径。
  - 且被抢的专家要**不出现在 helper 的 `owns()` 里**——即要改 `peer_owner` 表的构建
    （:3855-3860）让它对某些专家答"没人持有"，那 helper 的 cache 里就有个洞，
    **必须由主卡或 CPU 精确补上**，否则 = 静默 logits 损坏。

**结论：本方向（CPU 池与 helper 同层分工）在 P4 形状下性价比极低，建议关闭。**
理由三条：
1. **没有便宜的落点**——不是"主卡让出一点"，而是要动 `peer_owner` 的构建语义 +
   新增一条数值路径 + 改主卡的发射契约（`n_local` 从 0 变非 0 时，`bounds`/`ids`/`total_rows`
   全部要按新段改）。
2. **量级上限低**——CPU 池单独吃 prompt 是 191.1（§13.3 P1），而 P4 是 268.5；
   CPU 能加的量级大致是"替代一部分 helper 的 1/3"，但 helper 侧的量已经摊到 3 张卡上。
3. **风险高**——`peer_owner` 一旦与 helper 的 live cache 不一致，`peer_plan` 的所有权检查
   （:3666-3697）会**结束运行**（这是好设计），但任何漏算就是静默错 logits。

**替代方向（更便宜、同样能解"主卡闲着"）：**
既然主卡在 served 层上专家计算为 0，而主卡 VRAM 里还**白占** 40/256 的常驻槽
（2.93 GiB，`stream_kick` 跳过 owned 专家的上传 → 那 40 个槽的权重在 P4 里没被用），
那么**让 helper 不接主卡常驻的那 40 个专家**（即 `empty_primary` 改为传主卡的真实槽表），
主卡就能算这 40 个专家的 entry —— **主卡从"0 专家"变成"算 40/256"**，不用新增数值路径
（主卡还是 MMQ int8）、不用改 `peer_owner` 的语义（helper 只是少接一些），
代价是 helper 少接 40/256 的量（3.1%）。
**这比 CPU-share 便宜得多、风险低得多**，且方向正好补上"P4 的主卡闲着"这个实测事实。

**未验证**：上面这条替代方向我只是从代码推出来的，**没有实测**。要做的话先跑一条对照：
把 `empty_primary` 换成主卡真实槽表（一次代码改动，opt-in），量 P4 的 pp512 变化。
在实测前，它与 CPU-share 一样是 ESTIMATE。

---

## 以下是第 1~3 步的原始记录（保留，供复核）

跑 **P4 + `STRATA_Q36_HELPER_TIMING=1`**（`--bench-pp 512 --bench-tg 128`，同二进制 `b7d53f977`），
rc=0。pp512 warm（bench 1/2）= 252.0，tg128 = 51.4。

**报告本身**（`strata-q36 PHASE3 AGG GPU primary`）：
`layers=2688 dropped=0 | attn_ssm=0 router=0.507 helper_dispatch=0 local_grouped=0
local_shared=0 cpu_pool=0 helper_merge=0 | tile=0.507 segsum=0.507 resid=0`

**这张表量的是 decode 的单 token，不是 prompt** —— probe 自己的注释写明（`q36_engine.cu:2061-2063`）：
> "Only a T==1 forward records a tile: with 40 layers in one forward the 8 shared events would be
> overwritten by the last layer before they are read, so n>1 could not be measured with them at all.
> That is fine -- the number being explained is the single-token decode step."

prompt 的 forward 是 n=512 行 → `gseg_ring` 里的 tile 数 n>1 → `gp_drain` 直接 clear + dropped，
**不产生任何段**。所以 **prompt 的分段分解用现有工具测不了**（要测得改代码：给 prompt 路径按层
分配独立事件或改用 per-layer 事件数组，= 一次代码改动，不是旗标）。

**但这一轮有两条硬收获（都是 decode 侧，MEASURED）：**

1. **主卡每层每 token 一次 host 同步**：`prebegin_sync=2415.3 ms（6.290 ms/tok, 15360 calls）`
   —— 15360 = 40 层 × 384 token，即**每层每 token 一次 `cudaStreamSynchronize`**。
   采集点 :3407-3410（`moe_helpers` 的 drain，decode 路径），单次 ~0.41 µs。
2. **三张 helper 的等待不均**：`ms_wait` = CUDA1 0.233 / CUDA2 0.281 / **CUDA3 0.642** ms/tok
   —— CUDA3 等得最久（2.3× CUDA1）。`route_spin=0.171 ms/tok`（CPU 池等路由）。

**另一条关键事实（本轮读码核到）**：P4 的 `local_grouped = 0.000` 与下面的专家数一致 ——
**主卡在 P4 里可能一个专家都没算**（3 张 helper 持 3440+3400+3400 = **10240 = 40 层 × 256 全部**，
`own[e] != kPeerNone` 对每个专家成立 → `peer_plan` 的 `n_local = 0`，主卡无活可算）。
**这条必须先钉死**（见下"第 4 步"）——它决定 CPU-share 有没有可抢的活。

### 探针的读法修正（重要，勿再犯）

- `STRATA_Q36_PREFILL_PEERS: N helper(s) serve prompt chunks` = **启动时打一次**的持有关系
  （`peer_open` :3884），**不代表本次 prompt 走了 peer 分支**。
- `PHASE3 AGG GPU primary` 的段 = **decode 单 token**（T==1）的分解，**不是 prompt**。
- 判 prompt 走哪条分支：看 `hybrid experts: ... the rest of a batch of <= N rows on 8 CPU threads`
  （`<= 128 rows` = MMQ 分支 = P4；`<= 2048 rows` = hybrid 批次 = 探针）。

## 第 2 步：旗标实验——死路证实（2026-10-23，MEASURED）

探针 = **P4 + `--hybrid-rows 2048`**（纯旗标，其余同 P4）。二进制 `b7d53f977`，rc=0、IMA=0、nan=0。

| 配置 | 分支 | warm pp512 (bench 1/2) |
|---|---|---|
| P4（`--hybrid-rows` 默认 128） | MMQ 分支（主卡 MMQ + peer 计划 + 3 helper） | **268.5**（268.0/269.0） |
| 探针（`--hybrid-rows 2048`） | **hybrid 批次**（主卡 hybrid + CPU 池；**peer 出局**） | **195.0**（177.4/192.5） |

**死路证实（MEASURED）：靠调 `--hybrid-rows` 让 CPU 池参与 prompt，必然把 3 张 helper 踢掉。**
机制（代码核到）：`hybrid_t=2048` → prompt 的 `T=512 <= hybrid_t` → `hy_rows` 真（:2946）→
`mmq_rows = !hy_rows && ...` **假**（:2947）→ prompt 走 hybrid 批次（`moe_hybrid_gpu` +
`moe_hybrid_cpu`），**MMQ 分支不进 → `peer_plan_or_exit` 不执行 → helper 在 prompt 里出局**。
hybrid 批次里 helper 也不上场（`moe_helpers` 要求 `T <= MAXT=8`，prompt T=512）。

实测 -27%（268.5 → 195.0），正是"helper 出局、CPU 池顶一部分"的形状。

**注意 banner 的读法**：`STRATA_Q36_PREFILL_PEERS: 3 helper(s) serve prompt chunks (...)` 是
**启动时打印一次**的持有关系（`peer_open` :3884，紧跟 `peer_any` 构建），**不代表本次 prompt 走了
peer 分支**。判分支要看 `hybrid experts: ... the rest of a batch of <= N rows on 8 CPU threads`
（探针里 `<= 2048 rows` = hybrid 批次；P4 里 `<= 128 rows` = MMQ 分支）。

**结论**：CPU 池与 helper 要同层分工，**只能改代码**（在 MMQ 分支里加第三桶，见设计 v2），
没有任何现成旗标能做到。这与设计稿的前提一致。

### 待核（剩余，实施前）

- [ ] CPU 桶的**选桶判据**要用 host 侧的 `H.slot_of` + `peer_owner`：两者都在 host
      （`peer_owner` 是 host vector，:3620 读 `peer_owner.data()`），可在 `peer_plan` 里一起算。
- [ ] CPU 池的**激活输入**：`moe_hybrid_cpu` 用 `h_h`（fp32 激活，:2963 那条 D2H）。
      MMQ 分支目前**没有**这条 D2H（只有 `h_peer_ids` 的 ids，:3020）。新增一条 `T*N*4`（4 MiB/层）。
- [ ] CPU 池要等路由 D2H：`moe_hybrid_cpu` 的 spin 等 `ev_route`；新分支要先
      `cudaEventRecord(ev_route, st)`。
- [ ] CPU 池的 `h_miss` mask 要新增一份（`d_miss` 现成但语义是 hybrid 路径的），
      或复用并改语义。
- [ ] **数值一致性**：CPU 池算的专家用 fp16/q8_1 host blob + CPU 反量化，主卡 MMQ 用 int8 GPU
      ——两边的数值路径不同，混算的 logits 与纯 MMQ 会有差异。P1 里纯 CPU 池 vs P3 纯 MMQ
      也是两条路径，**混算的正确性基线**（容差）要定，且要有一致性对照（flag on/off 的 logits 比）。
- [ ] **量级参照**：P4（268.5）vs CPU 池单独吃 prompt（P1，191.1）——CPU 池单独能顶 191，
      探针里 hybrid 批次 + 3 个挂着的 helper = 195。**CPU 池对 prompt 的增量空间大致在这个量级
      （+20~35%），但 P4 里主卡/.helper 已经摊走 2/3，CPU 桶能吃到的残余比 P1 少**——
      这是本设计的主要不确定性，实测前不作结论。

