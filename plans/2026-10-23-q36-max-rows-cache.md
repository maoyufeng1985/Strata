# q36: 去掉 prepare_end 的那次等待（host 侧算出行计划的两���数）

日期：2026-10-23。目标机：192.168.1.98（1× Tesla P100-12GB 主卡 + 3× P104-100 8GB helper，16 核）。

## 0. 起点

§13.9 的最终形状：**402.2 tok/s**，每层 31.995 ms。

每层分解（§13.9.1）：

| 段 | ms/层 |
|---|---|
| plan（host 等 3 个 helper 准备） | **8.140** |
| helper_wait（等 helper 的行 + 上腿 H2D） | 6.309 |
| primary_work | 8.189 |
| mixer | 7.127 |
| router_route + route_d2h + combine | 0.809 |

host 墙钟：`prep_sync` **4.957**、`stage` 0.971、`peer_plan` 7.120。

**`prep_sync` 是 `prepare_end()` 里那次 `cudaStreamSynchronize(helper 的流)`。**
它等的是这个 helper 自己这条流上排在最前面的东西：
下腿 H2D（4 MiB 的激活，走 Gen1 x4）+ 4 个 row-plan 核 + 两个 8 字节的读回。

## 1. 三个事实（读代码得到的，不是猜的）

### 事实 A：`max_rows` 在 MMQ 里是 `ncols_max`，是**上界**语义

`src/prefill/moe_mmq.cu:421-425` 把它塞进 `mmq_args` 的最后两个字段：

```cpp
const mmq_args a = {..., p.max_rows, p.max_rows};
```

`third_party/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:1384-1385`：

```
int64_t ncols_max;
int64_t ncols_opt; // value to optimize the tile size against, launch grid still uses ncols_max
```

`mmq.cuh:1413`：`const int ntx = (args.ncols_max + config.J - 1) / config.J;` —— launch 网格的**列块数**由它定。

核里多余的块自己会退出（`mmq.cuh:1024`）：

```cpp
if (jt*J >= col_diff) { return; }      // col_diff = expert_bounds[zt+1] - expert_bounds[zt]
```

`col_diff` 是**精确**的行数（来自 row plan 的 `bounds`），
`ncols_max` 只是网格的**上界**。

→ **大了安全（多出的块立刻返回），小了漏列（静默少算）。**
→ 这不是"精确值"，所以"缓存它"的前提是"每层相同"——而"每层相同"是**未经验证的推测**。

### 事实 B：host 手上**已经有**算 `max_rows` 需要的全部数据

`peer_plan`（`q36_engine.cu:3963-3968`）已经在 host 上遍历全部 entry：

```cpp
for (int64_t e = 0; e < n_ent; ++e) {
    const uint8_t o = own[(size_t) h_peer_ids[e]];
    if (o == kPeerNone) ++n_local;
    else ++rows[(size_t) o];
}
```

`h_peer_ids` 是这一层的路由 id（`:3183` 的 D2H，已经在 host 上）。
`own[]` 是 (层, 专家) → helper 的归属表（`:3933`）。

要算"helper r 那部分里最宽的专家占多少行"，只需要**每个专家的 entry 计数**：

```
ecnt[e] = 这一层里 ids == e 的 entry 数
max_rows_r = max over { e : own[e] == r } of ecnt[e]
```

### 事实 C：host 算出来的 `max_rows` 与 device 的**逐层相等**

helper 侧的 row plan（`q36_peer_prefill.cu:126-148`）：
`k_peer_count` 数的是 `owned[ids[e]]` 的 entry；`k_peer_scan` 取 `mx = max over cnt[s]`。

- helper 的 `owned_host[e] = 1` ⟺ `m.helper->slot_ptr(layer, e) != nullptr`
- engine 的 `own[e] == r`

这两个的一致性**已经被 peer_plan 的逐专家 witness 保证**（`:3945-3959`）：
对每个 (e, r) 比对 `peers[r]->owns(li, e)` 与 `own[e] == r`，不一致就结束运行。

而归属是**按专家**的（一个专家的 entry 全归同一个 helper），所以

```
device 的 cnt[pos(e)] = { ids == e 且 owned[e] } 的 entry 数 = { ids == e } 的 entry 数 = host 的 ecnt[e]
```

且 device 的 `mx` 取遍**这个 helper 自己的**专家集合 = host 的 `{e : own[e] == r}`。

→ **两边的 max_rows 完全相等**，不是"上界对下界"，是同一个数。

→ 于是 host 不需要等 device 回来。**"每层是否相同"这个前提根本不需要成立。**

## 2. 方案

### 第 1 步（先做，只测量，不改行为）

1. `peer_plan` 里加一个 per-expert 的 host 计数（`ecnt[ne]`，每层清零），
   算出每个 helper 的 `host_maxr[r]`。
2. executor 暴露 `last_max_rows()`。
3. peer_plan 在 `prepare_end()` 之后把 **device 的** `max_rows` 与 **host 的** `host_maxr[r]`
   比对——**不等就结束运行**（这是新增的第二个 witness，比今天的只查行数更强）。
4. 探针报告：每层每 helper 的 `max_rows`（device 与 host 两个值）+ 分布
   （min / max / 不同取值的个数）。

这一步回答用户的问题"每层 max_rows 是否相同"，同时**验证**host 的算法在真机上逐层相等。
行为不变（launch 仍用 device 的值）。

### 第 2 步（优化）

1. `max_rows` / `n_owned` 改用 host 算出来的值。
2. `prepare_end()` 不再 `cudaStreamSynchronize` → **那次等待整个消失**。
3. row plan 的读回保留（8 字节，D2H），但检查**推迟**：
   - 在**下一次** `prepare_begin()` 开头检查（那时上一层的读回必然已落地，见 §3）
   - forward 结束时再检查一次（最后一层）
4. 不一致 → 结束运行（和今天同样的策略：宁可不答，不要错答）。

### 第 3 步（可选，看第 2 步的实测再定）

把下腿 H2D（4 MiB）从 `prepare_begin` 挪到 `prepare_end` 之后，
这样 helper 的 row plan（只需要 16 KiB 的路由）不必排在那 4 MiB 后面。
预期收益很小（第 2 步之后 host 已经不等了），先不做。

## 3. 为什么"推迟检查"是安全的

读回 `h_plan` 的 D2H 排在 helper 流的 row-plan 核**之后**，所以：

- 下一层的 `prepare_begin` 读到 `h_plan` 时，上一层的值已经落地 ✓
  （host 走到下一层之前，先经过了下一层 route 后的 `cudaStreamSynchronize(st)`
  （`:3190`），它等到的是**上一层 combine 之后**的 `st`；
  combine 在 `peer_prefill_wait()` 之后，而 `peer_prefill_wait()` 把主卡排在 `ev` 上，
  `ev` 记录在 helper 流的最末尾 → helper 整条流都已完成 → 读回早已落地。）
- forward 结束时同样成立（主卡 `st` 同步到 combine 之后）。

**另外，host 的共享 staging 缓冲也是安全的**：下一层往 `peer_stage_x` 写之前，
上一层 helper 的下腿 H2D 已经读完它了（同上那条链）。

## 4. 第 1 步的实测（2026-10-24，192.168.1.98）

二进制 `bin/strata-q36.maxrows`（md5 `2be6b5430532b4a17f92d498ef139d5e`），
对照 `bin/strata-q36`（md5 `b7d53f977fc832d9a5a071a793c30bbf`，未动）。
旗标：`STRATA_Q36_PREFILL_PEERS=1 STRATA_Q36_PREFILL_PEERS_HOST=1 STRATA_Q36_PEER_F16=1
STRATA_Q36_HELPER_ASYNC=1 STRATA_Q36_HELPER_PARALLEL=1 STRATA_Q8_SM60=1`，`--bench-pp 512 --bench-tg 0`。

### 结论 1：`max_rows` **每层都不同** —— 原计划的前提被推翻

| helper | 40 层里的取值范围 | 不同取值的个数 |
|---|---|---|
| CUDA1 | 87 .. 508 行 | **40**（每层都不同） |
| CUDA2 | 130 .. 508 行 | 35 |
| CUDA3 | 53 .. 507 行 | 36 |

→ 「缓存一个数给所有层用」这条路**不成立**。
→ 但**方案不受影响**：`max_rows` 是 `ncols_max`，**上界**语义（核里 `if (jt*J >= col_diff) return;` 裁掉多余的块），
所以 **host 逐层算**就行，不需要"每层相同"。

### 结论 2：host 与 device **逐层完全相等** —— host 侧算法成立

120 层 × 3 helper = **360 次比较，0 次不一致**。

依据（读代码得到的，不是拟合）：归属是**按专家**的（一个专家的 entry 全归同一个 helper），
所以「entry 数」这个量谁数都一样；device 的 `k_peer_scan` 取遍的正是这个 helper 自己的专家集合，
而这个集合已被 `peer_plan` 的**逐专家归属 witness**（`q36_engine.cu:3945-3959`）证明等于 host 的 `own[e] == r`。

→ host 可以独立算出 `max_rows`，**不必等 device 回来**。

### 结论 3：那 5ms 的构成（稳态 bench 1/2）

`plan` 的 host wall 拆成三段（每层，ms）：

| 段 | bench 1 | bench 2 | 是什么 |
|---|---|---|---|
| bookkeeping | 1.143 | 1.143 | host 自己的表（归属核验 + 行计划 + 直方图） |
| prepare | 0.436 | 0.436 | `prepare_begin()`：只**排队**，不等 |
| **wait** | **5.295** | **5.295** | `prepare_end()`：等 helper 的流 + 读回 8 字节 |

稳态整层：**forward 31.735 ms**（bench 1）/ 32.124（bench 2），与 §13.9 的 31.995 同量级。
`prep_sync` 5.078 / 5.026（三个 helper 的**和**）。

**关键：`wait` 5.295 是三个 helper 的和，不是关键路径的贡献。**
三个 helper 的流在 `prepare_begin` 阶段就**并行**跑起来了（各自的 PCIe 链路物理独立），
`prepare_end` 只是依次去收割。所以关键路径上它贡献的是 **max** ≈ 5.295/3 ≈ 1.77 ms。

→ **收益上限 ≈ 1.77 ms/层**（不是 5ms），即 forward 31.735 → ~29.97，
tok/s 402 → ~426（+6%）。**远小于原估计的 +18%。**
→ 值得做，但不是大赢。

### 结论 4：探针不改行为

探针开 402.3 tok/s（bench 1）/ 397.3（bench 2）；探针关 400.2 / 405.6。
差异在噪声内（这台的重复测量波动就有 ±2%）。
`IMA=0`、`nan=0`、`rc=0`。

## 5. 第 2 步（优化）

1. `max_rows` / `n_owned` 改用 host 算出来的值。
2. `prepare_end()` 不再 `cudaStreamSynchronize` → **那次等待整个消失**。
3. row plan 的读回保留（8 字节，D2H），但检查**推迟**：
   - 在**下一次** `prepare_begin()` 开头检查（那时上一层的读回必然已落地，见 §6）
   - forward 结束时再检查一次（最后一层）
4. 不一致 → 结束运行（和今天同样的策略：宁可不答，不要错答）。

**注意**：`n_owned` 是**精确行数**（不是上界），host 必须算对。
host 算 `n_owned` 的办法：`rows[r]`（`peer_plan` pass 1 已经算的）**就是** owned 行数
——它数的是 `own[ids[e]] == r` 的 entry 数，与 device 的 `h_plan[0]`（`bounds[n_pos]`）
是同一个量（同一个归属、同一个集合）。
→ `rows[r]` 是现成的、已经在 plan 里算好的数，**不需要新的直方图**。

`max_rows` 需要直方图（每个专家的 entry 数），这是第 1 步加的那个 `peer_ecnt`。

### 第 2 步的旗标

`STRATA_Q36_PEER_PLAN_HOST=1`，函数局部静态、只认精确的 `"1"`、opt-in。
**默认路径逐字节不变**（flag 关时 `prepare_end` 仍然 sync、仍然用 device 的数）。

## 6. 为什么"推迟检查"是安全的

读回 `h_plan` 的 D2H 排在 helper 流的 row-plan 核**之后**，所以：

- 下一层的 `prepare_begin()` 读到 `h_plan` 时，上一层的值已经落地 ✓
  （host 走到下一层之前，先经过了下一层 route 后的 `cudaStreamSynchronize(st)`
  （`q36_engine.cu:3190`），它等到的是**上一层 combine 之后**的 `st`；
  combine 在 `peer_prefill_wait()` 之后，而 `peer_prefill_wait()` 把主卡排在 `ev` 上，
  `ev` 记录在 helper 流的最末尾 → helper 整条流都已完成 → 读回早已落地。）
- forward 结束时同样成立（主卡 `st` 同步到 combine 之后）。

**另外，host 的共享 staging 缓冲也是安全的**：下一层往 `peer_stage_x` 写之前，
上一层 helper 的下腿 H2D 已经读完它了（同上那条链）。

## 7. 风险与不变量

- **三个 witness 都在**：
  - 逐专家归属交叉核验（`q36_engine.cu:3945-3959`，host 纯查表，零 CUDA）——**没动**
  - 行数 device-vs-host 比对（今天就有）——**没动**
  - `max_rows` device-vs-host 比对（**第 1 步新增**）——第 2 步推迟到下一层
- **行契约不变**：`eparts` 的行数、行序、每行语义一字未动。
- **旗标 opt-in**：第 2 步用 `STRATA_Q36_PEER_PLAN_HOST=1` 门控，默认路径逐字节不变。
- **数值对照**：控制组 vs 新形状的 logits 必须逐字节相同（37 行 argmax 相同）。
- **旧二进制保留**：`b7d53f977`（对照）不动，新二进制用新文件名。

## 8. 验证清单

- [x] 第 1 步：build rc=0；md5 核对（对照 `b7d53f977` 未动）
- [x] 第 1 步：每层 `max_rows` 的 device 值 == host 值（**0 个不一致**，360 次比较）
- [x] 第 1 步：`max_rows` 的分布（**每层都不同**：40 层 40 个值）
- [x] 第 1 步：探针不改行为（开/关在噪声内），IMA=0、nan=0、rc=0
- [ ] 第 2 步：旗标关 = 今天的形状（tok/s 与 logits 都对照）
- [ ] 第 2 步：旗标开 = 更快；`prep_sync` ≈ 0
- [ ] 第 2 步：logits 与控制组逐字节相同
- [ ] IMA=0、nan/inf 行数 0、rc=0
