# q36：helper 的 `ms_wait` CUDA3 是 CUDA1 的 3.2 倍 —— 复核（2026-10-24）

复核对象：`plans/2026-10-24-q36-decode-and-primary-idle.md` §4 的待做项 3
（「helper 的负载不均（CUDA3 慢 3.2×）：`expert % 3` 的分法是否让某个 helper 拿到更多行」）。
只读：不改代码、不跑实验；带 path:line 的判断。

## 1. 那个 0.751 是什么量

decode 的 helper 计时（`STRATA_Q36_HELPER_TIMING=1`，warm bench 1/2 的 ms/token）：

| helper | `ms_begin` | `ms_wait` |
|---|---|---|
| CUDA1（P104 01:00，helper r=0，`e % 3 == 0`） | 1.568 | 0.233 |
| CUDA2（P104 02:00，r=1，`e % 3 == 1`） | 1.476 | 0.285 |
| CUDA3（P104 03:00，r=2，`e % 3 == 2`） | 1.504 | **0.751** |

两个计时器（`src/core/remote_experts.cpp`）：
- `ms_begin`：`begin_slot()` 整个主卡 host 侧过程（`remote_experts.cpp:437` 的 `Timer`）——
  staging 的下腿 H2D、4 个行计划核（`k_peer_rank/count/scan/scatter`）、launch 发射。**排队，不等。**
- `ms_wait`：`finish()` 里那次阻塞（`remote_experts.cpp:634`，`cudaStreamSynchronize(stream_)` 或
  HIP 的 spin，`w0` 到返回，`remote_experts.cpp:636`）。**只计那次阻塞调用本身**，不含它前面的
  `DeviceScope` 三个 runtime 调用（那些归 `ms_sync_overhead`，`:592`）。

「3.2×」= 0.751 / 0.233。它说的**不是**「CUDA3 算得慢」，而是：
**CUDA1 的 `finish()` 被主卡调用时，CUDA3 的流早已干完活、`cudaStreamSynchronize` 立刻返回；
轮到 CUDA3 自己被 `finish()` 时（最后收割），它还在等自己的流。**

## 2. 根因：收割顺序，不是负载

`moe_helpers()` 的收割是**按 r 顺序串行**的（`q36_engine.cu:3720` 起的注释 + 循环：
default 路径 `nwin == 1`，PARALLEL 路径 `for w in 0..nwin` 依次 `begin` 全部、再 `finish` 全部，
`remote_experts.cpp` 的 `finish` 阻塞同步那张卡的流）。三张卡各自独立流并行跑，但主卡
**按 r=0,1,2 的顺序**去 `finish`：

1. `finish(CUDA1)`：CUDA1 的流此刻还没干完（它是 `ms_begin` 最大那张，1.568）→ 主卡等它
   约 0.233 ms 才到（大部分已经并行跑完了，只剩尾巴）。
2. `finish(CUDA2)`：轮到时 CUDA2 多半也干完了 → 0.285。
3. `finish(CUDA3)`：最后收割。此时 CUDA1/CUDA2 的尾巴早被主卡等掉，**主卡整条关键路径被
   CUDA3 的最后一段算术 + 上腿 H2D 拖住** → 0.751。

所以 0.751 里装的是「**最后被收割的那张卡的剩余工作**」，不是「CUDA3 比别人多算多少」：

- **专家数**（open 时定死，`STRATA_Q36_HELPER_PARALLEL` 的 `e % 3`）：3440 / 3400 / 3400
  （CUDA1 多 40 个，是 `40 × 256 / 3` 除不尽的余数，`open_helpers` 的 `row_split` 循环
  `q36_engine.cu:6311-6323`：`for e = k; e < ne; e += n`，k=0 拿 86 个/层，k=1,2 拿 85/86 交错）。
  差异 < 1.2%，**不足以解释 3.2×**。
- **路由行数**（每层实际命中）：三张卡的行数是 `n_ent × P(路由到 e%3==k)`，取决于 token 的
  路由分布，**不取决于专家数**。`expert % 3` 的分法保证三张卡拿到的**专家 id 集合**大小接近，
  但**热专家集中在哪张卡**完全由路由决定。

结论：「`expert % 3` 让某个 helper 拿到更多行」**不成立**——它分的是专家数（3440/3400/3400，
近均匀），而 3.2× 的 `ms_wait` 差**是收割顺序造成的观测效应**，不是负载差。

## 3. 怎么复核才能分辨「收割顺序」vs「真的负载不均」

只读能确认的（已确认）：
- 收割是串行、按 r 序（`moe_helpers` 的 `for w` 循环 + `finish` 阻塞，path 见 §1/§2）。
- 专家数近均匀（3440/3400/3400，余数给 r=0，`q36_engine.cu:6311-6323`）。

**分辨需要跑**（不在本次只读范围，列出来供后续决定）：
1. 把三张卡的 `finish()` 改**乱序 / 最晚收割最忙的卡**，看 `ms_wait` 是否重新均摊到每张
   （如果是收割顺序效应，打乱顺序后 0.751 会摊薄；如果某张真的多算，它会一直最大）。
2. 打一次**每层每 helper 的行数直方图**（decode 的 `h_ids` 已经在 host，`moe_hybrid_cpu` 自旋后
   就有，和 keep-resident 的热度探针同一位置）：数三张卡各吃多少行。这是唯一的负载直接证据。
3. 关掉 `STRATA_Q36_HELPER_PARALLEL`（回到层条带 `e % 3` 按层，不并行）再比 `ms_wait`——
   条带模式下每张卡每层只算一层，收割顺序效应会更明显（一张卡等另外两张整层）。

## 4. 判断（给 §4 待做项 3 的收口建议）

- **0.751 vs 0.233 不是「CUDA3 负载重」**，是「CUDA3 是最后被收割的那张」。
- 真正的负载问题要**每层每 helper 的行数直方图**回答，专家数（3440/3400/3400）太均匀，
  背不动 3.2×。
- 收割顺序是可动的（把 `finish` 从「按 r 序」改成「按 `ms_begin` 降序 / 最忙的放最后等」），
  但那是**改行为**，不在只读复核范围；先量清楚再决定。

**给 §4 待做项 3 的建议写法**（待用户拍板是否执行）：
> 把 3.2× 归因为「收割顺序（`moe_helpers` 按 r 串行 `finish`，最后的卡背全部剩余）」，
> 不是「`expert % 3` 给 CUDA3 多分了行」。验证要两步：(a) 每层每 helper 行数直方图
> （decode 的 `h_ids` 已在 host，一处 O(n_ent) 遍），(b) 把 `finish` 顺序打乱重测。
> 在 (a) 出来之前，不动收割顺序。
