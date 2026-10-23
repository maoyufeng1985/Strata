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
`prepare_end` 只是依次去收割。所以关键路径上它贡献的是 **max**。

→ **后来用 per-helper 的计时直接量了**（见下），**max ≈ 4.89 ms，几乎是全部**。

### 结论 3b：三个 helper 的准备**确实是并行的**（per-helper 计时，2026-10-24）

二进制 `bin/strata-q36.waitr`（md5 `d99c2fdc0e77e1f54e8af71eec5833a0`，只加计时，行为不变，
实测 401.9 tok/s，与对照 402.2 一致）。

稳态 bench 2，每层 ms：

```
wait = 5.364  (helper CUDA1 4.892, CUDA2 0.138, CUDA3 0.074)
```

**CUDA2 和 CUDA3 几乎立刻返回**——它们在 CUDA1 被等待的那 4.9 ms 里早就做完了。
所以：

- 三个 helper 的准备**完全并行**（各自的 PCIe 链路物理独立：主卡 P100 在 root port `03.2`，
  三个 P104 在 `01.0` / `02.0` / `02.2`，各自独立 Host Bridge）。
- **关键路径上那 4.892 ms 就是「最慢的那个 helper 的下腿 H2D + row-plan 核」**，
  不是三个的和，也不是三分之一。
- 4.892 ms 走 4 MiB → 有效带宽约 **0.85 GB/s**（PCIe Gen3 x8 的理论值是约 4 GB/s，
  实测差 4.7 倍：P104 是老卡，H2D 引擎的实际吞吐远低于链路宽度）。

→ **那 5ms 值得攻的地方是「让 helper 的下腿传得更快」，不是「让 host 少等」。**

### 结论 4：探针不改行为

探针开 402.3 tok/s（bench 1）/ 397.3（bench 2）；探针关 400.2 / 405.6。
差异在噪声内（这台的重复测量波动就有 ±2%）。
`IMA=0`、`nan=0`、`rc=0`。

## 5. 第 2 步（优化）—— **做了，实测：收益为零。不合并。**

### 实测（2026-10-24，二进制 `bin/strata-q36.planhost`，md5 `ee12a33bae65568affbd5a143667bb38`）

改动：host 自己数出那两个数，`prepare_end()` 不再 `cudaStreamSynchronize`（旗标 `STRATA_Q36_PEER_PLAN_HOST=1`）。

| | 对照（`strata-q36.final`） | 旗标开 |
|---|---|---|
| `prep_sync`（三个 helper 的**和**） | 5.024 | **0.000** |
| `peer_plan` 的 host 墙 | 6.928 | **1.807** |
| `wait`（三个 helper 的和） | 5.295 | 0.286（每个 helper 0.001） |
| plan 的 GPU span | 7.544 | **9.251**（+1.707） |
| helper_wait | 7.876 | **6.291**（−1.585） |
| **forward** | **31.743** | **31.883** |
| tok/s | 402.2 | 400.3 |
| logits 对照 | — | **逐字节相同**（37/37 argmax，max\|diff\| = 0） |

### 结论：那 5ms 的等待**从来不在关键路径上**

- 等待**确实**消失了（`prep_sync` 5.024 → 0.000，`wait` 5.295 → 0.286）。
- 但 **forward 没变**（31.743 → 31.883），tok/s 没变。
- 两个变化互相抵消：`plan` 的 GPU span **变大** 1.707 ms（host 提前做完计划，GPU 队列的重叠方式变了），
  `helper_wait` **变小** 1.585 ms（helper 的算术更早开始，更早完成）。
- → 那 5ms **被 helper 的算术时间完全掩盖**：host 等多久，helper 都在同样的时间里算完。

**所以 prompt 的关键路径是：helper 的算术 + 主卡的算术 + mixer，host 的计划藏在它们的缝隙里。**
要提速 prompt，得动这三样里的**一样**，动 host 的等待没有用。

### 为什么第 2 步不合并

1. **收益为零**（实测）。
2. **它带一个竞态**，而竞态是「延迟检查」这个做法的固有陷阱：

   > 读回 `h_plan` 的 D2H 排在 helper 的流上。**排队 ≠ 完成。**
   > 下一层的 `prepare_begin` 只是把本层的下腿 H2D **排入**队列，层 N 的读回此刻**可能还没执行**——
   > 主机读到的就是 `prepare_begin` 清零后留下的 0。
   >
   > 实测：探针开（`STRATA_Q36_PROMPT_TIMING=1`）时 360 次检查全过；**探针关时第一个 pass 就崩**
   > （`the row plan counted 0 rows and a widest expert of 0`，device 读到 0，主机说 1588 / 494）。
   > 探针开时 forward 末尾多了一个 `cudaEventSynchronize`（`pp_drain`），它让读回落地了；
   > 探针关时没有。第一次检查（层 2）能过，是设备比主机快，**属于侥幸**。

   要正确地延迟核验，得给读回一个「已落地」的保证：要么每层一个独立的目标槽位（40 层 × 3 helper × 8 字节），
   在 forward 末尾统一比；要么等一个恰好排在读回之后的事件——而后者等的正是「下腿 H2D + row-plan 核」，
   **和现在 `prepare_end` 等的东西一样**。既然收益是零，这个复杂度没有理由付。

### 第 1 步的 witness 不受这个竞态影响

第 1 步的比对是**立即**的：在 `prepare_end()` 之后，而 `prepare_end()` 里 sync 过 → 读回已落地。
**所以第 1 步（提交 `c78a6662`）是安全的，保留。**

### 那 5ms 的真正去处：**helper 的下腿传输**

per-helper 的计时说：CUDA1 等 4.892 ms，CUDA2/CUDA3 几乎不���。那 4.892 ms 是
**4 MiB 的下腿 H2D + 4 个 row-plan 核**，在 PCIe 上，实测有效带宽约 0.85 GB/s。

第 2 步证明：**host 不等它，forward 也不变**——因为 helper 的算术比它长，它被掩盖了。
→ **要缩短这 4.9 ms，唯一的办法是让它本身更快**（更小的下腿、更快的链路、或者把它藏进 helper 的等待里）。
→ 而这正是 §7 的方向。

## 6. 为什么"推迟检查"是安全的 —— **不成立，见 §5**

推迟检查的想法是：读回排在 helper 流的 row-plan 核之后，所以下一层读到时上一层的值已经落地。

**这个推理错在把「排队」当成了「完成」。** 下一层的 `prepare_begin` 只是把本层的下腿 H2D 排入队列，
上一层的读回此刻**可能还没执行**（它排在 helper 流上，而 helper 流上还有那 4 MiB 的下腿 H2D 在它前面）。
实测证明了这一点：探针关时第一个 pass 就读到 0。

正确的保证只有两个，都不便宜：
- 每层一个独立的读回目标槽位（40 层 × 3 helper × 8 字节），在 forward 末尾统一比；
- 等一个恰好排在读回之后的事件——而它等的正是「下腿 H2D + row-plan 核」，和现在 `prepare_end` 等的一样。

**第 1 步的比对是立即的（在 `prepare_end` 之后，而那里 sync 过），所以不受影响。**

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
- [x] 第 2 步：旗标开 = 等待消失（`prep_sync` 5.024 → 0.000），但 **forward 不变**（31.743 → 31.883）
- [x] 第 2 步：logits 与控制组逐字节相同（37/37 argmax，max\|diff\| = 0）
- [x] 第 2 步：探针关时崩（延迟检查的竞态）→ **不合并第 2 步**
- [ ] 每层每 helper 的等待（并行还是串行）—— 保留下来的计时器
- [ ] IMA=0、nan/inf 行数 0、rc=0
