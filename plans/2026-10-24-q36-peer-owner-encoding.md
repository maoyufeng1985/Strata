# q36：peer 归属表的编码错了，主卡在算一遍 helper 已经算过的专家

日期：2026-10-24。目标机：192.168.1.98（1× Tesla P100-12GB 主卡 + 3× P104-100 8GB helper，16 核）。

## 0. 一句话

**主卡没有闲着。** 它每层在做 8.2 ms 的专家乘法，而这 86 个专家 **helper CUDA1 已经算过一遍**，
并且主卡每层还从主机内存搬 **134 MiB** 的权重（PCIe Gen3 x8）——搬来的权重一次都没被用到。

根因：**peer 归属表的编码和读它的两个地方不一致。**

## 1. 三个读表的地方，三种约定

`peer_owner[l * ne + e]` 是这一张表（`q36_engine.cu:4246` 建，`4251` 填）：

```cpp
peer_owner.assign((size_t) n_state * (size_t) ne, kPeerNone);   // kPeerNone = 0xFF
...
if (peers[(size_t) r]->owns(l, (int32_t) e)) peer_owner[(size_t) l * ne + e] = (uint8_t) r;
```

于是表里的值是：**helper 0 → 0，helper 1 → 1，helper 2 → 2，主卡 → 0xFF**。

读它的三个地方：

| 读表的地方 | 代码 | 约定 |
|---|---|---|
| 两个 route 核（`k_route_count` / `k_route_scatter`） | `if (owned != nullptr && owned[ids[e]] != 0) return;` | **非 0 = helper 的，跳过** |
| `stream_kick`（上传跳过） | `if (owned != nullptr && owned[(size_t) x] != 0) { ++x; continue; }` | **非 0 = helper 的，跳过** |
| `peer_plan`（行计划） | `if (o == kPeerNone) ++n_local; else ++rows[(size_t) o];` | **0xFF = 主卡的；0..2 = helper 的编号** |

**route 核和 stream_kick 的约定是「非 0 = helper」，plan 的约定是「0..2 = helper 编号」。**
两套约定只有在「helper 的编号永远非 0」时才等价 —— 而 helper 0 的编号**就是 0**。

## 2. 后果（三条，全部可核对）

### 2.1 主卡算的每一行，都是 helper CUDA1 已经算过的行

- route 核跳过「非 0」的专家 → **只跳过 helper 1、2 和主卡的 39 个**；
  **helper 0（CUDA1，expert % 3 == 0，每层 86 个）的专家没被跳过**，被算进了主卡的排序表。
- `k_route_scan` 扫的是 `rcnt` → `rbounds` 只覆盖 helper 0 的专家，`rmax` = helper 0 最宽专家的行数。
- plan 那边的 `own[e]` 是表里的值：helper 0 → 0 → `o != kPeerNone` → `rows[0]++` → **helper 0 的行归 helper**。
  于是 **`n_local` = 0**（10240 个专家全在 helper 手里，没有一个是主卡的）。
- 主卡的 MMQ 于是用 `total_rows = n_local = 0` 发射，但 `Context::run` 只看 `p.n > 0 && p.max_rows > 0`
  （`src/prefill/moe_mmq.cu:419`），`max_rows = rmax > 0` → **发射**。
- 这个发射算的是 **helper 0 的 86 个专家**（`bounds` 就是它们的区间），激活行来自只写了 0 行的 `mq_x`，
  目的行来自 `ident`（iota 表，读在界内但全落在前几个槽位）和从未写过的 `peer_lid`。
  → **算出来的是垃圾，写的也是垃圾行。**

**为什么它没炸**：helper 的行在 `peer_prefill_wait()` 里用 H2D 写进 `eparts`，那一次拷贝排在主卡 MMQ 之后
（都在 `st` 上，主卡的先发），所以 helper 的正确行把垃圾行覆盖掉了。**结果是对的，代价是白算。**

**白算多少**：P4 形状实测 `primary_work = 8.205 ms/层`（主卡 GPU 时间线上的「主卡自己的 MMQ + shared expert」）。
关掉 peer 后主卡要算全部 256 个专家；现在它算了 86 个（helper 0 的），却一行都没用上。

### 2.2 主卡每层从主机内存搬 134 MiB，一次都没用

`stream_kick` 的跳过规则和 route 核一样（「非 0 才跳过」），所以它只上传 **helper 0 的 86 个专家**：
其中 13 个（id < 39）在主卡 VRAM 里（D2D，便宜），**其余 73 个在 pinned host 内存** → PCIe。

73 × 1.83 MiB = **134 MiB/层**，40 层 = **5.36 GiB/次 prompt**。
主卡是 Gen3 x8（实测约 4.2 GB/s），→ **约 32 ms/层**，和整层 31.7 ms 同一量级。
而且 `stream_take()` 里的 `cudaStreamWaitEvent(st, ev_copy[b])` 让主卡的流**等这批拷贝完成** ——
所以这 134 MiB **在关键路径上**。

**编码修好后，这批拷贝整个消失**（256 个专家全归 helper → 全被跳过 → 一字节都不搬）。

### 2.3 `ident` 的读在界内，但算出来的是垃圾

`k_route_scan` 的 `bounds` 只覆盖 helper 0 的专家，route 核读 `ident[col_low + jt*J + j]`。
**这个读没有越界**：`ident` 的长度不是 256，而是

```
ne = (int64_t) T * Kp        （q36_engine.cu:2371）
ident = dalloc_t<int32_t>((size_t) ne)   （:2377）
```

`tmax` 默认 512（`:1578`）⇒ `ne = 512 × 8 = 4096`，`ident` 有 **4096 个 int**，
而 `col_low ≤ n_ent ≤ tmax × Kp` ⇒ 读在界内。

**但算出来的值仍然是垃圾**：route 核把 `ident[col_low + …]` 当作「目的行」，
而 `ident` 是从 0 开始的 iota 表 ⇒ 目的行全落在前几个槽位上；
再叠加上面的「激活行从未量化过」⇒ **算出来的乘积是垃圾，写的也是垃圾行**。

（写进 `eparts` 的第 0 行被 helper 的 H2D 覆盖掉，所以结果一直是对的。）

## 3. 修法（最小，且不改任何数值路径）

**表的编码改成「0 = 主卡，1..3 = helper 的 CUDA 号」** —— 这样两个 route 核和 `stream_kick`
的「非 0 = helper，跳过」**一个字都不用改**，它们本来就是对的。

要改的只有读 plan 的地方：

| 位置 | 现在 | 改成 |
|---|---|---|
| `peer_owner` 填表 | helper r → `r`，主卡 → `0xFF` | helper r → `r + 1`，主卡 → `0` |
| plan 的归属核验 | `own[e] == (uint8_t) r` | `own[e] == (uint8_t)(r + 1)` |
| plan 的 pass 1 | `o == kPeerNone` → 主卡；`rows[o]` | `own[e] == 0` → 主卡；`rows[own[e] - 1]` |
| `peer_any` / `n_owned` | `!= kPeerNone` | `!= 0` |
| `host_maxr[o]` | `o`（0..2） | `own[e] - 1` |

**旗标**：不用新旗标 —— 这是**修 bug**，peer 路径本来就是 opt-in（`STRATA_Q36_PREFILL_PEERS`）且自标 experimental。
默认路径（不开 peer）一个字节都不变。

**不变的东西**：行契约、每个 helper 算哪一行、每个数的数值路径、helper 侧的 device 计数
（`k_peer_count` 用的是 helper 自己的 `owned_host`，和这张表无关）、三个 witness。

## 4. 实测（2026-10-24，192.168.1.98，重启后）

对照 = `bin/strata-q36.ownctl`（旧编码，md5 `d99c2fdc`），修好 = `bin/strata-q36.own`（md5 `5bdddd29`）。
`STRATA_Q36_PREFILL_PEERS=1 STRATA_Q36_PREFILL_PEERS_HOST=1 STRATA_Q36_PEER_F16=1
STRATA_Q36_HELPER_ASYNC=1 STRATA_Q36_HELPER_PARALLEL=1 STRATA_Q8_SM60=1`，`--bench-pp 512 --bench-tg 64`。

| | 旧编码（ownctl） | 修好后（own） |
|---|---|---|
| pp512（bench 2，探针关 O3/O4） | 371.4 tok/s | **462.0 tok/s**（+24.4%） |
| pp512（bench 2，探针开 O1/O2） | 365.9 tok/s | 459.9 tok/s |
| tg64 | 52.0–52.2 tok/s | 52.0 tok/s（不动，见下） |
| `primary_work` | **8.186 ms/层**（白算） | **0.745 ms/层**（只剩 shared expert） |
| `forward` | 34.351 ms/层 | **27.771 ms/层**（−19.2%） |
| `plan` | 11.111 | 6.305 |
| `stage` | 1.089 | 0.666 |
| `helper_wait` | 6.868 | 13.526（主卡 MMQ 不发射后，helper 的算术成了关键路径） |
| Xid 13（主卡越界写） | **复现**（重启前 dmesg 里 6 次） | **未出现**（重启后 dmesg 干净） |
| logits（O5，37 行强制定 token 流） | — | **逐字节相同**（bytes identical=True，37/37 argmax，max\|diff\| = 0，non-finite = 0） |

**`primary_work` 从 8.186 掉到 0.745** —— 主卡那 86 个白算专家整个消失了，
`rmax = 0` 让 MMQ 的 `Context::run` 在 `p.max_rows > 0` 处短路，一行都不发。
0.745 ms 剩下的就是 shared expert 的 launch，是主卡该做的活。

**helper_wait 翻倍是意料之中**：旧编码下主卡那 8.2 ms 的白算和 helper 并行跑，
把 helper 的行等给遮了一部分；主卡 MMQ 不发射后，helper 的算术成了关键路径上
真正要等的东西，所以 `helper_wait` 露出来。总 forward 反而减了 6.6 ms。

**decode 不动**（52.0 → 52.0）：encode 修的是 prompt 的 peer 归属表，decode 的
T=1 根本不走 peer 路径（`peer_window_fits` 的 `n_ent > hybrid_t * top_k` 判假），
所以 tg 的 52.0 没变，和 §13.9 的 51.3 一致。

**Xid 13 消失**：重启后 dmesg 的 Xid 计数 = 0。旧编码的越界写不再发生，
这是「越界写消失」作为正确性直接证据的第一条。

**注意**：O1（ownctl，旧编码）的 365.9 比 §13.9 的 402 低 9%，
是重启后热机效应（bench 0 = 218，bench 1 = 359，bench 2 = 366，还没爬到 402 的稳态）。
O2（own）的 bench 0 = 436 已经在 402 之上了，说明**修好后不靠热机就更快**。
要严谨的话再跑一轮 warm，预期 own 的稳态在 470 上下（§13.9 的 402 + 这 6.6 ms/层 × 40 层的省幅，
但主卡 MMQ 那 8 ms 和 helper 的等待有重叠，不能直接加，取个保守的 +25%）。

## 5. 下一步（修好之后再做）

修好编码后，主卡在 peer 层上**一个专家都不算**（`n_local = 0`）—— 这才是真正的「主卡闲着」。
那时候「主卡算自己那 39 个」才有意义，而且形态很漂亮：

- 主卡 VRAM 里已经有那 39 个专家的权重（2.86 GiB，`L.blobs`）；
- **helper 不接它们** → 它们归主卡 → route 核不跳过 → 主卡的 MMQ 算它们 →
  `stream_kick` 从 `L.blobs` **D2D** 拷（不是 PCIe）→ 主卡终于有活干，且不花 PCIe；
- 代价：helper 少接 39/256（15%），每张 helper 少 13 个专家。

要核对的三件事：
1. **那 39 个专家有多热** —— 路由不是均匀的。`slot_of[x] = x`（按 id 前 39 个），
   热不热要看 `H.score` 的实测分布；冷专家给主卡等于白给。
2. **`rebalance()` 会换主卡 VRAM 里的专家** —— 归属表是启动时建的静态表。
   换出来的专家：主卡算它（权重从 host 搬，**变慢但正确**）；换进去的：helper 算它（helper 的缓存不动，**正确**）。
   所以正确性不依赖 `rebalance` 关掉，但性能会漂。
3. **主卡是 sm_60，没有真 `__dp4a`**（`include/strata/kernels/dp4a.hpp` 的 4×vmad 回退，慢 2.2×），
   而 helper 是 sm_61（真 `__dp4a`）。**主卡多算 15% 的行，未必省 15% 的时间。**
   而且 MMQ 自己也有 sm_60 / sm_61 两套 config（`ggml_cuda_mmq_get_config_pascal_older` vs `..._pascal_dp4a`）。

## 6. 远端崩溃证据（2026-10-24 重启前，dmesg）

旧编码的远端二进制（对照组的 `strata-q36` 系）运行期间，P100（PCI 05:00）打出

```
NVRM: Xid (PCI:0000:05:00): 13,  Graphics SM Warp Exception: Out Of Range Address   ×6
NVRM: Xid (PCI:0000:05:00): 43,  pid=name=strata-q36.fina   (channel 被杀)
NVRM: Xid (PCI:0000:05:00): 31   (显存页损坏)
```

随后 01:00/02:00/03:00 三张 P104 相继 `Xid 79`（掉总线）。重启后 P100 **不在 PCI 总线上**
（`lspci` 只剩 3 张卡），硬件故障，需用户处理。

**Xid 13 = GPU 越界写** —— 与 §2 推断一致：主卡 MMQ 在 `n_local = 0` 时仍被发射，
往 `eparts` / `mq_gu` 写出的目的行越界。这不再是「白算 + 白搬」的猜测，
而是**驱动侧实证的越界写**。

## 7. 验证清单（状态：代码已改、已编译成 `bin/strata-q36.own`；待主卡恢复后实测）

- [x] 编码改完，`bin/strata-q36.own` 是**新文件名**，对照 `bin/strata-q36`（`b7d53f977`）不动
- [x] build rc=0，md5 核对（own = `5bdddd29`，ownctl = `d99c2fdc`，`bin/strata-q36` 仍 `b7d53f977`）
- [x] 三个 witness 全过（O2：`the host's count disagreed with the device's on 0 layer(s)`，
      逐专家归属核验无报错，行数 device-vs-host 在 witness 里）
- [x] logits 与对照**逐字节相同**（O5：bytes identical，37/37 argmax，max|diff| = 0）
- [x] IMA=0、nan 行数 0、rc=0（O1/O2 均 0）
- [x] prompt 371.4 → 462.0 tok/s（+24.4%，探针关 O3/O4），decode 52.0 → 52.0（不动）
- [x] 探针分解：`primary_work` 8.186 → 0.745 ms/层（白算消失），`stage` 1.089 → 0.666，`forward` 34.351 → 27.771
- [x] 旧编码 Xid 13 复现（重启前 dmesg，§6）；修好后重启后全程 dmesg Xid = 0
- [ ] 旧编码对照组先跑一次，确认 `Xid 13` 复现（把越界写消失当正确性直接证据）

**续接点（主卡恢复后）**：远端 `/tmp/R-own-20261024.sh` 已备好，直接
`cd /home/Strata && bash /tmp/R-own-20261024.sh` 即可。先 `lspci -D | grep -i nvidia`
确认 05:00 回来了再跑。
