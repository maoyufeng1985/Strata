# q36：`STRATA_Q36_PROMPT_TIMING=1` — prompt 路径的尺子（2026-10-23）

配套：[2026-10-23-q36-p4-prompt-optimization.md](2026-10-23-q36-p4-prompt-optimization.md)（方案）、
[2026-10-23-q36-ev-wait-optimization.md](2026-10-23-q36-ev-wait-optimization.md)（ev-wait 优化）。

状态：**已实现，待上机取数**。

## 1. 为什么需要它

P4 的 prompt 是 268.5 tok/s、512 token、40 层 → **47.7 ms/层**。
已算清的部分只有主卡的 ~1 ms（attention + 共享专家 + router）。**剩下 40+ ms/层没有解释。**

现有的 `STRATA_Q36_HELPER_TIMING=1` probe **测不了 prompt**：`gp_drain()` 只记 T==1 的 tile
（`q36_engine.cu:2061-2063`：*"Only a T==1 forward records a tile … n>1 could not be measured with them
at all"*），因为它只有 **8 个共享事件**，40 层会互相覆盖。

所以这把尺子是所有优化的前提。

## 2. 它量什么

**旗标**：`STRATA_Q36_PROMPT_TIMING=1`（新 opt-in；只认精确的 `"1"`，读一次，函数局部静态——
与 `helper_timing()`（:1911）同写法）。

**只测 prompt 的 MMQ 分支**（`T > kGemmT && use_mmq`）；decode / hybrid / MTP / graph 路径**不碰**。

### GPU 侧（主卡 stream `st` 上，**每层一组事件**）

事件数组 `pp_ev[2 + 8*n_layers]`（:5093）：索引 0 = forward 起点，1 = forward 终点，
`2 + 8*l + {0..7}` = 第 l 个 MMQ 层的 P0..P3（8 个/层）。
**per-layer event ring**（不是共享的 8 个）——这是对抗验证特别要求的点。

| 边界 | 位置 | 行 |
|---|---|---|
| P0 | 层入口（attention 前） | :2895 |
| P0b | GDN / attention 分支结束 | :3043 |
| P0c | router + route 核之后、route D2H 之前 | :3130 |
| P0d | route D2H 的阻塞 sync 之后、`peer_plan_or_exit` 之前 | :3150 |
| P1 | `peer_plan_or_exit()` 返回后 | :3174 |
| P1b | `peer_prefill_wait()` **之前** | :3240 |
| P2 | `moe_end()` 之前 | :3268 |
| P3 | `moe_end()` 之后 | :3281 |

**八个跨度**（跨所有层与所有 chunk 求和，报告给每层均值）：

| 跨度 | 定义 | 含义 |
|---|---|---|
| `mixer` | P0b−P0 | GDN 或 attention 的本体（投影、conv、SSM / attention） |
| `router_route` | P0c−P0b | router + route 三个核 |
| `route_d2h` | P0d−P0c | route D2H + 它等的那次阻塞 sync |
| `plan` | P1−P0d | `peer_plan_or_exit` 的 host 活（在 GPU 时间线上） |
| `primary_work` | P1b−P1 | **主卡自己的活**（MMQ 发射 + 共享专家） |
| `helper_wait` | P2−P1b | **等 helper 的行落地 + 上腿 H2D** |
| `combine` | P3−P2 | `moe_end` |
| `post_combine` | 下一层 P0 − P3（最后一层用 forward 终点） | 层间 |

### P1b 的位置：**必须在 `peer_prefill_wait()` 之前**（一个已踩过的坑）

`peer_prefill_wait()` 里做的是 `cudaStreamWaitEvent(st, m.ev)`（`q36_peer_prefill.cu:854`）——
它是**插进 `st` 队列的一个等待点**。CUDA 的流语义：排在这个等待点**之后**记录的事件，
时间戳取在等待满足**之后**。

所以边界位置决定这一段量的是什么：

| P1b 记在哪 | `primary_work` = P1b−P1 | `helper_wait` = P2−P1b |
|---|---|---|
| **`peer_prefill_wait()` 之前** | 主卡自己的活（干净） | **helper 停顿 + 上腿 H2D**（要的量） |
| `peer_prefill_wait()` 之后 | max(主卡活, helper 剩余) + 上腿 H2D | ≈ 0（中间只有 no-op） |

**第二行是错的**：helper 的等待被算进了"主卡自己的活"。而 `n_local = 0`（主卡在 served 层上
一个专家都不算，§13.4 已实测）→ 那 21 ms/层会顶着"主卡自己的活"报出来，**方向正好反**。
第一版尺子（4 个边界）就有这个毛病，第二版必须改对。

> 中间那两行中间只有 `ev_mark(kSegMerge)`，它在 prompt 探针下是 no-op
> （`ev_mark` 只在 decode 探针的 `gpu_timing` 下记录，:2131-2133）→ 所以"之后"那种放法的
> `helper_wait` 会**恒为 0**，一眼就能看出标签反了。

### Host 侧（墙钟）

| 名 | 包住什么 | 行 |
|---|---|---|
| `prep_d2h` | `prepare()` 里 host route 的两个阻塞 D2H（激活 + 路由） | `q36_peer_prefill.cu:666-677` |
| `prep_sync` | `prepare()` 里 row-plan 的 `cudaStreamSynchronize` | `q36_peer_prefill.cu:736-739` |
| `peer_plan` | `peer_plan_or_exit()` 整段（host 循环 + 所有权交叉核验） | `q36_engine.cu:3124-3131` |
| `sync3022` | 每层 route D2H 之后那次 `cudaStreamSynchronize(st)` | `q36_engine.cu:3112-3114` |
| `ppwait` | `peer_prefill_wait()` 的 host 墙钟（**本身不阻塞 host**，预期很小） | `q36_engine.cu:3205-3211` |

**未归因余量** = forward 总时间 − 四个 GPU 跨度之和 − 五个 host 段之和。
**显式打出来**，不靠猜。

### 同步点

**整个 forward 只加一次 `cudaEventSynchronize`**（`pp_drain()` 里等 forward 终点，:2096）。
不在层间加。

## 3. 实施与复核中发现并修掉的问题

### 3.1 第一版（4 个边界，`bin/strata-q36.prompt-timing` = `b82eca97`）

1. **Blocker**：probe 会在 **graph 捕获的 decode batch** 上武装 —— `run_rows` → `forward(T, …,
   use_graph=true)`，`pp_drain()` 的 `cudaStreamSynchronize` 会落在 `cudaStreamBeginCapture` 里
   → `cudaErrorStreamCaptureImplicit`，且给 decode 路径加了一个 sync。
   **修法**：`forward()` 的开/关门加 `&& !graph_mode`（prompt 路径 `process()` 本来传 `graph_mode=false`
   → probe 仍能测 prompt）。
2. **Major**：跨度标签错位 —— `peer_prefill_wait()` 之后才记 P2，导致"helper 等待"没有单独成段。
   **修法**：P2 挪到 `moe_end` 之前，跨度改名 `helper_wait`→`combine`、`combine`→`post_combine`。
3. **Minor**：`mmq_here` 原来在 `layer()` 顶部无条件计算（flag-off 路径多一条语句）→ 移进三个受门控的块。
4. **Minor**：`pp_drain` 的 `pp_nlayers == 0` 早退是死代码 → 改成 `!pp_open || pp_nlayers == 0`。
5. **Minor**：`prepare()` 的 `n_pos == 0` 早退路径加了 D2H 计时却没加 `prep_calls` → 补上，
   使每调用均值自洽。

### 3.2 第二版（8 个边界，`bin/strata-q36.prompt-timing2` = `c3ffcce5`）

把 26.26 ms/层拆成 mixer / router+route / route-D2H / plan，把 21.02 ms/层拆成
primary_work / helper_wait。**两个 review agent 都判"无阻塞"，但都漏了下面这个真缺陷**——
它俩继承了规格里的同一个错误前提。

6. **Major（真缺陷，两个 review 都漏了）**：**P1b 记在 `peer_prefill_wait()` 之后**。
   `peer_prefill_wait()` 做的是 `cudaStreamWaitEvent(st, m.ev)`（`q36_peer_prefill.cu:854`）——
   那是**插进 `st` 队列的一个等待节点**。CUDA 的流语义：**排在等待节点之后记录的事件，
   时间戳取在等待满足之后**（记录本身要等前面的工作做完）。
   所以记在 wait 之后的 P1b 自己就带上了 helper 的等待时间：
   `primary_work = P1b−P1` = max(主卡活, helper 剩余)，`helper_wait = P2−P1b ≈ 0`。
   而 `n_local = 0`（主卡在 served 层上一个专家都不算）→ 那 21 ms/层会顶着
   **"主卡自己的活"**报出来，**方向正好反**。
   **修法**：P1b 挪到 `peer_prefill_wait()` **之前**（队列变成
   `[主卡MMQ][shared][P1b][wait(m.ev)][上腿H2D][P2][moe_end]`），
   `primary_work` = 主卡自己的活，`helper_wait` = 等 helper + 上腿 H2D。
   挪位只动探针的记录语句，`peer_prefill_wait()` 相对 `moe_end` 的位置没变 → GPU 行为不变。

7. **Major**：**报告一行里混了两种量纲** —— 八个跨度按每层均值打印，而 `forward` /
   `gpu_spansum` / `gpu_residual` 和 HOST WALL 五段是**总和**。第一版也有这毛病，
   当时靠"×40"脑补才读懂。
   **修法**：全部除以层数，一行里所有数都是 ms/层；原始总和 = 打印值 × 层数。

**为什么两个 review 都漏了第 6 条**：规格里写的是"P1b — right AFTER peer_prefill_wait()
returns … Then: primary_work = P1b - P1 (the primary's own MMQ/shared-expert launches) …
helper_wait = P2 - P1b (the wait for the helper's rows)"。这段**前提就是错的**
（"wait 之后那段间隙 = 等待"），两个 review 都照抄了这个前提，
没有回到 CUDA 的流队列序去核。**规格里的语义断言要自己验，不能继承。**

**已知未修（低优先）**：`pp_ev` 在引擎退出时**没有 delete**（:5093 `new cudaEvent_t[n]`）——
opt-in 路径上进程退出即回收，不影响测量。

## 4. 旗标关时的保证

- `prompt_timing()` 只读一次 env，精确 `"1"`；关时所有新语句都不执行。
- 事件**只在旗标开时创建**（:5088-5097），关时一个都不建。
- `pp_reset()` 无条件跑，但只清零成员变量。
- 不改任何算术、发射配置、行契约、所有权检查；不加第二个同步点。

## 5. 上机取数

**二进制**：
- `bin/strata-q36` = `b7d53f977` —— **对照**，§13 所有基线数的来源，**永不覆盖**
- `bin/strata-q36.prompt-timing` = `b82eca97` —— 第一版（4 个边界）
- `bin/strata-q36.prompt-timing2` = `c3ffcce5` —— 第二版（8 个边界，P1b 已挪位）

源码备份在 `/home/Strata/src.pristine/`（= 本地 `git show HEAD:`，去 CR 后 md5 `02286eb5`）。
远端 build 的源目录是 `/home/Strata/src`，文件在 `/home/Strata/src/src/qwen35/`。

**命令**：
```sh
cd /home/Strata && STRATA_REMOTE_ZEROCOPY=0 STRATA_Q36_HELPER_ASYNC=1 \
  STRATA_Q36_HELPER_PARALLEL=1 STRATA_Q8_SM60=1 STRATA_Q36_PREFILL_PEERS=1 \
  STRATA_Q36_PREFILL_PEERS_HOST=1 STRATA_Q36_PROMPT_TIMING=1 \
  ./bin/strata-q36.prompt-timing2 --model /home/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --max-context 2048 --no-warmup --bench-pp 512 --bench-tg 0 \
  --expert-cache-device1 auto --expert-cache-device2 auto --expert-cache-device3 auto
```

**输出**：stderr 上的 `strata-q36 PHASE3 bench %d prompt: ...`（每 pass 一行）+ 一行 AGG。
**单位**：一行里所有数都是 **ms/层**（第二版起统一）。

## 6. 实测结果（2026-10-23，`bin/strata-q36.prompt-timing2` = `c3ffcce5`）

192.168.1.98，P4 旗标，`--bench-pp 512 --bench-tg 0`，warm = bench 1。
**rc=0，IMA=0，nan=0**。bench：0=173.5 / 1=267.7 / 2=267.9 tok/s
（对照：`bin/strata-q36.prompt-timing` 那次是 175.6 / 268.9 / 268.5 → 尺子本身无影响）。

**bench 1（warm），一行里全是 ms/层**：

```
mixer=7.096 router_route=0.537 route_d2h=0.060 plan=18.765 primary_work=8.125
helper_wait=12.900 combine=0.232 post_combine=0.018 | forward=47.734 gpu_spansum=47.734 gpu_residual=0.000
HOST WALL: prep_d2h=2.440 prep_sync=15.690 peer_plan=18.758 sync3022=27.472 ppwait=0.047
```

**GPU 时间线完整闭合**：`gpu_residual = 0.000`（bench 1、bench 2 都是 0.000）。

### 6.1 每层时间去哪了

| 段 | ms/层 | 占 | 是什么 |
|---|---|---|---|
| **plan** | **18.765** | **39%** | **host 串行等 3 个 helper 的 `prepare()`**（不是 GPU 活） |
| **helper_wait** | **12.900** | **27%** | 等 helper 的行落地 + 上腿 H2D |
| **primary_work** | **8.125** | **17%** | 主卡自己的活：MMQ（`n_local=0`，空）+ 共享专家 |
| mixer | 7.096 | 15% | GDN/attention（30 层 GDN + 10 层 attention） |
| router_route + route_d2h | 0.597 | 1% | router + route 核 + route D2H |
| combine + post | 0.250 | 1% | `moe_end` + 层间 |

### 6.2 第一版尺子的结论被推翻了

第一版（4 个边界，`b82eca97`）说：`attn+route+plan = 26.263`、`primary_mmq = 21.022`，
并把 26.26 归给"attention/GDN"、21.02 归给"helper 等待 + 主卡 MMQ"。
**第二版把这个 26.26 拆开后，里面 18.765 ms 是 host 在等 helper 准备，不是 GPU 在算 attention。**

原来"未解释的 ~8 ms"也不存在 —— 它一直在这 26.26 里，是 `plan`。

**上一版尺子的错在哪**：它把 `P1−P0`（层入口 → plan 返回）当成"GPU 的 attention 时间"，
但那一段里 **host 阻塞了 18.7 ms**（GPU 空闲等 3 个 helper 准备），
而 GPU 事件测的是**墙上时间**，host 的阻塞也算在里面。

### 6.3 `plan` 的结构（新的头号目标）

`plan` = 18.765 ms/层 ≈ host 的 `peer_plan` 墙钟 18.758 ms/层（差 0.007 ms，互证）。
拆开看（host 墙钟，3 个 helper × 40 层 = 120 次 prepare）：

| host 段 | ms/层 | 说明 |
|---|---|---|
| `prep_sync` | **15.690** | `prepare()` 里的 `cudaStreamSynchronize`（`q36_peer_prefill.cu:737`），**每次 prepare 平均 5.23 ms** |
| `prep_d2h` | 2.440 | 两次阻塞 D2H（激活 + 路由），走**主卡**链路 |
| `ppwait` | 0.047 | `peer_prefill_wait` 的 host 墙钟（不阻塞 host）✅ 与设计一致 |

**`prepare()` 的阻塞点在 `peer_plan` 的 helper 循环里**（`q36_engine.cu:3967`）：

```cpp
for (int r = 0; r < 3; ++r) {
    peers[r]->prepare(...);   // ← 阻塞：等这个 helper 的下腿 H2D + row-plan 核
    peers[r]->launch(...);
}
```

**3 个 helper 串行** → 三次下腿传输**不重叠**。而每个 helper 收的是**全部 512 token**
的激活（`n_tok=512`，`n_tok * n_embd * 4` = 4 MiB），3 个 helper 各 4 MiB =
**12 MiB/层过 Gen1 x4（~1.25 GB/s）→ 理论 ~9.6 ms**，实测 `prep_sync` 15.69 ms
（含 row-plan 核与 launch 延迟）。

**这一条把 §13.4 的"P4 卡的是主卡"改写了**：主卡不是瓶颈，
**host 串行等三个 helper 的下腿传输**才是第一位的（39%）。

### 6.4 `primary_work = 8.125` 里主卡在算什么

`n_local = 0` → 主卡的 MMQ（quantize/gu/swiglu/dn）**一行都不算**，但核仍发射。
真正的活是**共享专家**（`q36_engine.cu:3219-3232`，T=512 > kGemmT=8 的分支）：
`k_to_f16`(512×2048) + `apply(sh_gate)` + `apply(sh_up)` + `k_swiglu` + `mm(sh_down)`，
其中 FS = `n_ff_shexp` = 25（共享专家很窄）。
8.1 ms/层对共享专家来说**偏大**，但这是主卡自己的、必须的活。

### 6.5 `helper_wait = 12.900` 是什么

P2−P1b = 等 `m.ev`（helper 全部完成：下腿 H2D + row-plan + MMQ + 上腿 D2H）
+ 上腿 H2D（`h_out` → `eparts`，11.2 MiB 过主卡 Gen3 x8）。
与 §13.4 的估算（9.0 + 1.4 + …）量级一致。

### 6.6 下一步的三条候选（按实测重排）

| 候选 | 实测依据 | 预估收益 | 风险 |
|---|---|---|---|
| **A. 3 个 helper 的 `prepare()` 并行** | `plan`=18.765（39%），三次下腿传输不重叠 | 最多省 ~9 ms/层（18.8 → ~10） | 中：改 `peer_plan` 的循环顺序，但 `launch()` 依赖 `prepare()` 的结果（行数）→ 要拆成"先全部 prepare，再全部 launch" |
| **B. 下腿 fp16** | 下腿 12 MiB/层过 Gen1 x4 | 12 → 6 MiB，省 ~4.8 ms/层 | 低-中：只改传输格式，helper 侧量化前转回 fp32 |
| **C. 上腿 fp16** | `helper_wait`=12.900 里上腿 11.2 MiB | 省 ~2.8 ms/层 | 低-中：改数值路径，要 logits 容差对照 |

**A 的收益最大且不动数值路径** —— 优先做 A。

**取数后要回答的三个问题**：
1. `primary_mmq`（P2−P1）到底多大？→ **helper 的等待在里面占多少**（它 = primary_mmq − 主卡自己的活）。
2. `unattributed` 还剩多少？→ 上面的分解是不是已经解释了 47.7 ms/层。
3. `prep_d2h` / `prep_sync` / `peer_plan` / `sync3022` 各是多少？→ 它们是不是被高估了
   （§13.4/§13.6 的估算都是算出来的，没测过）。
