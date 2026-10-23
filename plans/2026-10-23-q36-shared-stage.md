# q36：下腿的窗口只 stage 一次（2026-10-23）

配套：[2026-10-21-q36-peer-ima-root-cause.md](2026-10-21-q36-peer-ima-root-cause.md) §13.7–13.8、
[2026-10-23-q36-upleg-f16.md](2026-10-23-q36-upleg-f16.md)。

状态：**已实现并实测**。

## 0. 问题

三个 helper 在同一层拿的是**同一份**激活和**同一份**路由
（`peer_plan` 传给每个 `prepare_begin` 的是同一个 `hdev`、同一个 `idsdev`、同一个 `T`），
但每个 helper 都在自己的 `prepare()` 里**从主卡拷一次**：3 × 4 MiB 的阻塞 D2H 过主卡的 PCIe。

实测（`STRATA_Q36_PROMPT_TIMING=1`）：`prep_d2h` = **2.9 ms/层**（host 墙钟）。

## 1. 改法

**主卡侧**（`q36_engine.cu`）多一份**共享的 pinned 缓冲**（`peer_stage_x` / `peer_stage_ids`）：
`peer_plan` 在给任何 helper 之前做**一次** D2H，三个 helper 都从这一份读。

**helper 侧**（`q36_peer_prefill.cu`）：`prepare_begin` 多两个参数
（`h_staged` / `ids_staged`）—— 给了就用它，不再自己拷；不给（默认）就自己拷。

**P2P 路线不变**：engine 只在**有 helper 走 host 路线**时才做这次 D2H
（`PeerPrefill36::host_route()`），所以有 peer 链路的机器上多出的这一次拷贝根本不会发生。

**不变量**：字节数、行的内容、行的顺序、行契约、两个 witness 全部不变 ——
**只是同一份数据从主卡出来一次，而不是三次**。

## 2. 实测（2026-10-23 晚，MEASURED）

`bin/strata-q36.shared-stage` = `1cf78f1f`，对照 `bin/strata-q36.upleg-f16` = `165e94b7`。

| 趟 | 对照（每 helper 各拷一次） | 共享 staging |
|---|---|---|
| fp16 关，bench 1 / 2 | 311.0 / 311.7 | **317.5 / 318.7**（+2.1%） |
| fp16 开，bench 1 / 2 | 387.2 / 380.3 | **395.6 / 401.4**（+3.8%） |

**为什么只有 +2~4%，不是预期的 +6%**：`prep_d2h` 那 2.9 ms 里，大部分**与 GPU 的活重叠**
（host 在等 D2H 时，主卡在算上一层的活）。省下的是真正串行的那部分。

探针（fp16 开，共享 staging，bench 1）：

```
plan=9.403  prep_d2h=0.004  prep_sync=5.091  peer_plan=6.810
helper_wait=6.309  forward=31.995  gpu_residual=0.000
```

`prep_d2h` 从 2.9 → **0.004**：helper 侧不再拷了（那一次搬到了 engine 侧，
由报告里的 `stage=` 计时器单独报出）。

## 3. 报告的修

共享 staging 之后，helper 侧的 `prep_d2h` 计时器**测不到任何东西**（0.004），
而 engine 侧那一次 D2H **没人计时** → 报告会显得 staging 是免费的。
**修**：engine 给那一次 D2H 加了计时，报告里多一个 `stage=`。

（`bin/strata-q36.shared-stage2` = `93595e17` 带这个修。）
