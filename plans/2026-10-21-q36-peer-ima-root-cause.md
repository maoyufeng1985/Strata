# q36 peer host 路由的非法显存访问：根因

- 状态：**缺陷 1（stripe）与缺陷 2（row-split 槽位跨层）均已在真机验证修复（2026-10-23，§13）**
- 日期：2026-10-21（根因）／2026-10-22（真机验证 + 第二个缺陷）
- 机器：192.168.1.98，P100 主卡 + 3×P104 helper，无 P2P

## 0. 结论

**根因（MEASURED）：helper 缓存里两个相邻专家的地址差，和主卡的专家步长不一致。**

MMQ 按 `w + z * (expert_bytes / type_bytes(t))` 找第 z 个专家的权重。这个除法**必须整除**，
否则第 1 个专家之后每一个都从错位的地址读，读出来的是别的字节，算出来是 NaN。

| | layer 33（正常） | layer 34（坏） |
|---|---|---|
| gate/up 类型 | Q4_K（144 字节/块） | Q4_K（144 字节/块） |
| down 类型 | Q5_K（176 字节/块） | **Q6_K（210 字节/块）** |
| 主卡 `L.stride` | 1900800 | **2041200** |
| helper 实测地址差 `delta` | 1900800 ✓ | **2041344** ✗ 多 144 字节 |
| `gu.expert_bytes / 144` | 13200 ✓ | 14176 ✓ |
| `dn.expert_bytes / 210` | 10800 ✓ | **9720.7 → 截断成 9720** ✗ |

**为什么 layer 33 侥幸没事：** 它的 `L.stride = 1900800` 恰好是 256 的整数倍，缓存那次 256 对齐没有
改变它。layer 34 的 2041200 不是 256 的倍数，对齐后变成 2041344，步长就错了。

**为什么是 NaN 而不是报错：** MMQ 不会检查地址对不对。读到的字节被当成合法权重算，算出来是 NaN 或者
巨大值。日志里 layer 34 的 `d_out[0..3] = 4.62242e+06 2.15686e+08 2.17524 -865480` 就是这个样子。

### 完整链条（全部 MEASURED）

1. layer 34 的 helper MMQ 从错位地址读权重 → 输出 NaN / 巨大值。
2. NaN 顺着后面每一层传下去（layer 35..39 全 NaN）。
3. prompt 结束 `hlast[0..3]=nan nan nan nan`。
4. `greedy()` 返回 `0x7fffffff`——那是 `k_argmax`（`q36_engine.cu:894`）在所有 logits 都是 NaN 时的哨兵值。
5. 解码第一步 `iq_embed_rows` 拿这个数当 token id：`2147483647 × row_bytes`，远超词表 248320 → 越界 → IMA。

**所以 IMA 是第 3 步以后的必然结果，不是内存被踩坏。** 给 token id 加范围检查只会把崩溃藏起来。

## 1. 证据

### 1.1 复现（MEASURED）

```
STRATA_Q36_PREFILL_PEERS=1 STRATA_Q36_PREFILL_PEERS_HOST=1 STRATA_Q36_PREFILL_DIAG=1 \
  ./bin/strata-q36 --model $M --expert-cache-device1 auto --expert-cache-device2 auto \
  --expert-cache-device3 auto --bench-pp 160 --bench-tg 8 --no-warmup --max-context 2048
```

- 报错：`cudaStreamSynchronize(st) failed: an illegal memory access was encountered`，RC=1。
- `CUDA_LAUNCH_BLOCKING=1` 追到 `iq_embed_rows`。
- `--bench-tg 0`（纯 prompt）RC=0，pp160 99.8 tok/s；`--bench-tg 1` 同样崩。
- 本机没有 `compute-sanitizer`。

### 1.2 第一个坏层（MEASURED，逐层隐藏态）

| 层 | `h[0..3]` 出来后 |
|---|---|
| 0 | 1.87988 -0.452161 1.08007 0.632513（正常） |
| 1 | 2.41783 -0.292752 0.0639428 0.175445（正常） |
| 2 | 0.838784 -0.0360916 0.0379239 0.452799（正常） |
| 3..33 | 全部正常（layer 33 的 `d_out[0..3]` 是 `0.00408607 0.00229679 -0.0493454 0.0623799`） |
| 34 | **nan nan nan nan** |
| 35..39 | nan nan nan nan |

**layer 33 和 34 唯一的结构差别是 down 的量化类型（Q5_K → Q6_K），以及由此得到的步长。**

### 1.3 步长的两个数字（MEASURED）

```
DIAG primary stride=1900800 bytes=1900544 gu_type=12 d_type=13     (layer 0)
DIAG helper dev=1 layer 0 addr check: 0 of 256 experts mismatch delta=1900800
DIAG helper dev=3 layer 33 addr check: 0 of 256 experts mismatch delta=1900800
DIAG helper dev=3 layer 34 addr check: 0 of 256 experts mismatch delta=2041344
DIAG helper dev=3 layer 34 expert_bytes=2039808 gu_type=12 d_type=14
```

- `expert_bytes` 是原始 blob 大小（2039808），`delta` 是缓存的 slot 间距（2041344），
  主卡的 `L.stride` 是 2041200。**三个数都不同。**
- 缓存报 `0 mismatch` 是对的——它检查的是自己的地址和自己的 delta 一致，那当然一致。
  **它没检查的是：这个 delta 能不能被 MMQ 的块大小整除。**

### 1.4 主卡的步长规则（MEASURED 代码）

`q36_engine.cu:1467`：

```cpp
const size_t al = lcm_sz(lcm_sz(block_bytes(y.L.gu_type), block_bytes(y.L.d_type)), 16);
y.stride = (y.L.bytes + al - 1) / al * al;
```

- 取两种 block 大小的最小公倍数再对齐，**保证 `stride` 能被两种 block 大小整除**。
- layer 34：`lcm(144, 210) = 5040`，`align_up(2039808, 5040) = 2041200`。
  `2041200/144 = 14175` ✓，`2041200/210 = 9720` ✓。

### 1.5 缓存的步长规则（MEASURED 代码）

`expert_cache.hpp:134`：

```cpp
static int64_t slot_stride(int64_t blob, int64_t guard = 0) { return (blob + guard + 255) / 256 * 256; }
```

- **只对齐到 256，不管 block 大小。** 这就是错的来源。

## 2. 修法

**让 helper 缓存的 slot 间距等于主卡的 `L.stride`，不要再做 256 对齐。**

三处改动：

1. **`NativeExpertLayout` 加 `stride` 字段**（`iq_kernels.hpp`，默认 0 = 没意见 = 旧行为）。
   q36 在填 layouts 表时把 `L->stride` 写进去。
2. **`ExpertCache::set_slot_align(int64_t)`**（`expert_cache.hpp/.cpp`，默认 256 = 旧行为）。
   `open_sized` 用这个对齐量算 slot 间距。`set_slot_align(0)` 表示「调用方给的尺寸就是最终间距」。
3. **`RemoteExperts::open`**：
   - `lay.native` 时 `cache_.set_slot_align(0)`，`sizes[i] = max(L.stride, blob_bytes)`。
   - 非 native 时 `set_slot_align(256)`，保持旧行为。
   - 显存预算 `needed` 用同一个数，否则预算不准。

**为什么不用「把尺寸向上对齐到 256」：** `1900544` 本来就是 256 的倍数，那次对齐什么都没做。
问题从来不是 blob 对齐，是**步长必须是 block 的整数倍**。256 和 block 大小没有整除关系。

**为什么不动主卡的 `L.stride`：** 它已经在做对的事（lcm 对齐）。改成额外对齐 256 会让
layer 34 的步长从 2041200 涨到 2096640（多 55440 字节/专家），40 层 × 256 专家多占约 566 MiB 显存，
而收益为零。

**旗标关闭时行为不变：** `set_slot_align` 默认 256；`stride` 字段默认 0；非 native 路径显式设 256。
q36 之外的调用方（`generate.cpp`、`verify.cpp`、`peer_experts.cpp`）一个都不用新字段。

## 3. 验收（机器恢复后执行）

同一条命令，按顺序：

1. **崩溃命令 RC=0**，stderr 没有 `illegal memory access`。
2. **`--bench-tg 0` 纯 prompt RC=0**，pp160 仍在 99.8 tok/s 附近（host 路由的收益不能丢）。
3. **layer 34 的 `delta` 变成 2041200**，且 `dn.expert_bytes / 210` 整除。
4. **`rows[0..64): nan=0 first_good=0`** 出现在每一层，包括 34。
5. **旗标关闭**：连续 3 次 `--tokens 32 --n 1` 都是 `gen: 220`。
6. **`--mtp 2` 三 helper 推荐配置**：tg128 仍是 52.5 附近，不能因为这次改动掉速。
7. **一 helper / 三 helper 各跑一次**，确认不是只有三 helper 才对。

## 4. 还要做的

- **清理所有临时诊断**：`q36_engine.cu` 里的 `DIAG`/`diag_toks`/逐层同步读，
  `q36_peer_prefill.cu` 里的 `STRATA_Q36_PREFILL_DIAG` 诊断块、`m.uniform` 实验、
  `dn.ids = m.d_ident` 实验（已恢复成 `m.d_map`）。
- **`peer_window_fits()` 的下限改动**：定案**保留**（2026-10-23，静态推理）。下限 `n_ent > hybrid_t * top_k`
  不是可调参数，是 MMQ 分支的进入条件本身：route kernel 的 skip、stream_kick 的 upload skip 和 plan 三个读者
  共用这一个谓词（`q36_engine.cu:3624-3637`）。拆掉它，解码的窄窗（T=1）会让 route 跳过 peer 持有的条目而
  plan 不跑，`stage[b]` 留下洞——撤掉才会引入缺陷，留着才是正确的。§7 已证明它不是本次崩溃的根因。
- **R1（`peer_owner` 建一次、每层 `prepare` 重读 `slot_ptr`）**：**已解决**（2026-10-23，读码确认）。
  当前 `peer_plan`（`q36_engine.cu:3682-3696`）每服务一层前，都对全部专家重读 live cache
  （`peers[r]->owns(li, e)`）并与启动时建好的 `peer_owner` 表逐格对账，两边不一致就指名该层该专家结束运行。
  `rebalance()` 只动主卡的 staging 槽（`q36_engine.cu:4017-4018` 的 `STRATA_Q36_FIXED_SLOTS` / `!m->hybrid`
  guard，且 q36 从不 attach `RemoteExpertOpt`），所以 helper 槽启动后不会换。这是 §4 列的「每层对账」那条路，
  已经在代码里，不再挂账。
- **`--max-context 512` 的 IMA** 是否同根因：**已判定为独立缺陷**（2026-10-22 晚，MEASURED）。
  `--max-context 512` 在 **peers 关闭**时同样 IMA（`cudaStreamSynchronize(st) failed: an illegal
  memory access`，`q36_engine.cu:3413`），所以与本文件的两个缺陷都无关。
  诱因看得很清楚：日志是 `context 512 cells, prompt chunk 2048`——**prompt chunk 比 context 还大**。
  `chunk` 由空闲显存决定（`q36_engine.cu:6120-6122`：`fr_gib >= 7.5 ? 2048 : ...`），**与 ctx 无关**，
  所以 `--max-context 512` 反而拿到 2048 的 chunk。**单独的缺陷，另开一条**（见 §12）。


## 5. 事故记录：P104 掉卡

2026-10-21 17:16 左右，PCI 02:00 的 P104-100 掉总线（`Xid 79: GPU has fallen off the bus`），
同时 P100 上有 `Xid 13: Out Of Range Address`（就是这次查的越界读）。

- 掉卡后 CUDA 只枚举到 3 张：cuda0=P100、cuda1/2=P104（**序号变了**，历史测量条件是 4 张）。
- 掉卡无法软件恢复：`/sys/bus/pci/devices/0000:02:00.0/reset` 超时，`nvidia-smi -i 3 -r`
  被残留进程挡住，`rmmod nvidia_uvm` 说 in use。
- 崩溃残留的 `strata-q36` 进程处于 `R` 态内核循环，`SIGKILL` 挂起但投不进去，只能重启机器。
- **重启后要重新确认 CUDA ordinal 与物理卡对应关系**，不能假设还是 cuda0=P100。

## 6. 两个必须记住的坑

**坑 1：诊断读数必须先同步。** 引擎的流是 `cudaStreamNonBlocking`（`q36_engine.cu:2180`），
helper 的流也是（`q36_peer_prefill.cu:359`）。非阻塞流与 legacy 默认流之间**没有隐式同步**。
直接 `cudaMemcpy`（走默认流）会和引擎并发，读到中间态。第一次读出的 `ewts=-inf` 就是这个假象。

**坑 2：崩溃的进程会占着显存不退。** IMA 之后进程卡在内核态，`SIGKILL` 都不管用。
下一轮跑就是 `cudaMalloc ... out of memory`。每次跑之前要看 `/proc/*/exe` 确认没有引擎。

## 7. 被推翻的假设（不要重走）

| 假设 | 怎么被推翻的 |
|---|---|
| `peer_window_fits()` 缺下限，解码步误入 peer 路径 | 加了 `n_ent > hybrid_t * top_k` 下限后，新 binary 仍以同样方式崩 |
| `eparts` 行数越界 | `peer_plan` 有 `row < 0 \|\| row >= n_ent` 检查，恒成立 |
| 主机的 token vector 被写坏 | 逐层哨兵全程没触发 |
| helper 的分组把第一个专家硬拆（`delta=0, n=1` + `delta=1900544, n=255`） | 改成等距单组后 NaN 仍在 |
| `dn.ids` 用 identity 能修 | 让部分行恢复有限值，但前 64 行仍有 54 行 NaN。identity 只是定位手段，不是修法 |
| helper 缓存的权重字节是错的 | 主卡和 helper 的专家 0 前 16 字节完全一致 |

## 8. 2026-10-22 真机验证：修复有效，但有第二个独立缺陷

机器 2026-10-22 08:29 重启过（`uptime` 11h13m @ 19:42），四张卡都回来了。
**CUDA 序号没变**：CUDA 运行时默认 `FASTEST_FIRST`，P100 仍是 CUDA0、三个 P104 是 CUDA1/2/3；
`nvidia-smi` 按 PCI 总线号排（所以它把 P100 显示成 GPU3），两者不是一回事，别按 `nvidia-smi` 的下标推理。

当日基线（推荐配置，`--bench-pp 512 --bench-tg 128`）：**pp512 158.1 / tg128 52.2**，与 10-20 的历史值一致。

### 8.1 结论

1. **步长修复有效（MEASURED）。** stripe 模式（`HELPER_PARALLEL` 不设）跑崩溃命令：**RC=0，无 IMA，
   120 个 (layer, helper) 对全部 `rows[0..64): nan=0 first_good=0`**。layer 34 的 delta 现在是 **2041200**
   （修前是 2041344），`2041200/210 = 9720` 整除 ✓。
2. **但 row-split 模式（`STRATA_Q36_HELPER_PARALLEL=1`，也就是推荐配置）仍然从 layer 0 就 NaN。**
   这**不是**步长修复失效，是**第二个、独立的缺陷**。

### 8.2 第二个缺陷的根因（MEASURED，逐字节对上）

row-split 时每个 helper 持有**每一层**的一部分专家（`expert % 3 == k`），缓存按
`(expert 递增, layer 递增)` 的顺序放槽，于是**同一层的两个相邻专家在地址上隔着全部 40 层**：

```
observed delta = 76722000
= 34×1900800 + 2041200(layer 34) + 3×1900800 + 2041200(layer 38) + 2310000(layer 39)
= 76722000          ← 逐字节相同
```

peer 路径把这个地址差当作 MMQ 的专家步长（`q36_peer_prefill.cu:782` 的 `gu.expert_bytes = g.delta`），
而 MMQ 按 `w + z * (expert_bytes / type_bytes(t))` 寻址（`src/prefill/moe_mmq.cu:423`），**这个除法会截断**：

```
76722000 / 144 = 532791.67 → 截断成 532791
→ 第一个专家之后的每一个，都提前 96 字节读 → NaN
```

stripe 模式下没有这个问题，因为一个 helper 持有**连续的若干层**，同层专家在地址上紧挨着，
delta 就是引擎自己的 stride。

**为什么现有的检查没抓到：** `STRATA_Q36_PREFILL_DIAG` 的 addr check 在 row-split 下报
`0 of 86 experts mismatch`——**它只验"间距自洽"，不验"间距等于引擎的 stride"**。
自洽的间距照样是错的间距。

**修法（两个方向，第二个是正路）：**

- **正路**：peer 路径不该从地址差推断步长。`PeerPrefill36::open()` 已经拿到了
  `NativeExpertLayout` 表（每层的 `stride` 和 `gu_type`/`d_type`），可以直接用引擎的 stride，
  并在**步长与 block 大小不能整除**时拒绝服务这一层（fallback 到本地路径）。
- **退路**：row-split 与 peer 路径互斥（`HELPER_PARALLEL=1` 时不启用 peer），代价是丢掉 +45%。

### 8.3 还有一个只在诊断块里出现的崩溃（不是产品缺陷）

`STRATA_Q36_PREFILL_PEERS=1` + `--tokens 64` 会以
`cudaMemcpy(pr, prow, 8 * 4, ...) failed: invalid argument (q36_engine.cu:2755)` 结束。
`prow` 在**该层没有被 peer 服务**时是 null，而 DIAG 块（`q36_engine.cu:2748`）只判
`peer_prefill_on && T > 8` 就无条件解引用它。**这是诊断代码的 bug，不是引擎的**——
清掉 DIAG 块（§4 的第一项）即可。

### 8.4 今日实测汇总（全部在 192.168.1.98，同一台机，同一模型）

| 配置 | 结果 | 数字 |
|---|---|---|
| 基线（推荐配置，peers 关） | ✓ | pp512 158.1 / tg128 52.2 |
| stripe + peers 关 | ✓ | pp512 158.4–158.6（纯 prompt） |
| **stripe + peers ON + HOST** | **✓ RC=0 无 IMA 无 NaN** | pp512 154.1 / 159.2 / 173.3 |
| row-split + peers 关 | ✓ | pp512 158.2–158.4 |
| row-split + peers ON + HOST | ✗ layer 0 起 NaN | pp512 220–227（**但结果是 NaN，不算数**） |
| A5 flag-off parity ×3 | ✓ | 三次都是 `gen: 220` |
| A6 `--mtp 2` 三 helper | ✓ | pp512 154.3–154.5 / tg128 52.1–53.0（没掉速） |
| A7 一 helper | ✓ | pp512 158.5–158.6 / tg128 17.1–17.3 |
| A2 纯 prompt（peers 关） | ✓ | pp160 49.6–49.7（与 §3 第 2 项的 99.8 不同：那条是 peers ON） |

### 8.5 验收清单（§3）逐条对照

| # | 要求 | 结论 |
|---|---|---|
| 1 | 崩溃命令 RC=0，无 IMA | **stripe ✓**；row-split ✗（第二个缺陷） |
| 2 | 纯 prompt RC=0，pp160 在 99.8 附近 | ✓（stripe + peers ON + HOST） |
| 3 | layer 34 的 delta = 2041200 | ✓ **MEASURED 2041200** |
| 4 | 每层 `nan=0 first_good=0` | ✓ **120/120 全绿** |
| 5 | flag 关连续 3 次 `gen: 220` | ✓ **三次一致** |
| 6 | `--mtp 2` 三 helper，tg128 在 52.5 附近 | ✓ **52.1–53.0** |
| 7 | 一 helper / 三 helper 各跑一次 | ✓（一 helper tg128 17.1 是正常的，helper 少就是慢） |

## 9. 第二个缺陷的修复：槽位按层排列（2026-10-22 晚，MEASURED）

### 9.1 根因（MEASURED，逐字节对上）

缺陷 2 的根因不是"peer 路径从地址差推断步长"这么笼统，而是**一个更靠下的事实**：

> **缓存的槽位编号是按"专家被提交的顺序"发放的，而这个顺序就是调用方的 `ranked` 偏好列表。**

`RemoteExperts::open()`（`src/core/remote_experts.cpp`）先按 `ranked` 顺序挑出 `selected`，
再依次 `admit()`（发槽位号 0,1,2,…）并 `open_sized(sizes)`（按同一顺序算字节偏移）。
row-split 的 `ranked` 是**专家外层、层内层**（`q36_engine.cu:5950`），于是同一层的两个相邻专家
在缓存里隔着整整 40 层：

```
observed delta = 76722000 = 40 层步长之和      ← §8.2 的原始观测
修正后         = 1900800  = 该层自己的 stride  ← 本次实测
```

所以 **§8.2 的"正路"（把 `lay.stride` 直接喂给 MMQ）是不成立的**：在 row-split 的几何里，
同一层的专家本来就不是 `stride`  apart，硬喂 stride 会让 MMQ 去读**根本不属于该层**的字节——
错的字节，不是截断的假象。**必须先改缓存的排布，让 stride 成为真的。**

### 9.2 修法：槽位按层排序（`src/core/remote_experts.cpp`）

在 `sizes` 构建**之前**把 `selected` 按层号做 `stable_sort`：

- 一层的专家于是拿到**连续的槽位号**，字节偏移也连续 → 真实间距 = 该层的 `stride`。
- **是 no-op**：stripe 模式的 `ranked` 本来就是层外层内，排序后完全一样。
- **只改槽位编号**：哪些专家被选中、每个占多少字节，在排序前就已定死；所有消费方都通过
  `slot_of()` / `device_slot()` 解析槽位，没有一处按编号做算术。
- **排序必须在 `sizes` 之前**：row-split 下各层的 stride 确实不同
  （实测 1900800 / 2041200 / 2310000），否则宽度和专家会对不上。

### 9.3 配套：把"步长不对"从静默变响亮（`src/qwen35/q36_peer_prefill.cu`）

原来的 `m.uniform` 只验"间距处处相等"，自洽的**错误**间距照样放行（§8.2 的最后一句）。
现在有**两道**检查：

1. `PeerPrefill36::open()`：**启动时**逐层核对 helper 缓存里该层连续专家的真实间距是否
   等于布局的 `stride`，且 `stride` 是否是两种 block 大小的整数倍。不符 → **helper 直接不开**，
   引擎拿到一条启动错误，而不是第一个答案里的 NaN。
2. `prepare()`：**每层**再验一次实测间距 == `stride`，不符 → 报错并指名该层。
   （引擎的 `peer_any` 只由 `owns()`（槽位是否存在）决定，所以这一层一旦被路由到这里就没有
   "退回本地"的选项——只能硬失败。§8.2 说的"拒绝服务这一层、fallback 本地"在当前架构下
   **没有钩子可用**。）

另外：`prepare()` 里"非等距就每专家一个 Group"的旧分支被改成**总是**每专家一个 Group。
等距时走单个 Group（快），不等距时走单专家 Group（慢但永远正确）——不再存在
"用错间距去寻址"这条路径。

### 9.4 实测（192.168.1.98，同一台机，模型 `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`）

**row-split + peers ON + HOST**（此前 layer 0 起 NaN 的那个配置）：

```
DIAG helper dev=1 layer 0 addr check: 0 of 86 experts mismatch ... delta=1900800
DIAG helper dev=1 layer 0 addrs: e0@0x...(+0) e3@0x...(+1900800) e6@0x...(+3801600) ...
DIAG helper dev=1 layer 0 stage gu nan=0/32768 h nan=0/32768
DIAG helper dev=1 layer 0 rows[0..64): nan=0 first_good=0
```

- **delta 从 76722000 变成 1900800**（= 该层的 stride）✓
- addr check **0 of 86 mismatch** ✓
- **40 层全部干净**：每层 `h[0..3]` 都是有限数，无 NaN ✓

### 9.5 未决

- §4 其余项：`peer_window_fits()` 下界、`--max-context 512` 的 IMA（**未确认**与本缺陷同源，
  不能合并断言）、R1（`peer_owner` 一次性建好但 `prepare()` 每层重读 `slot_ptr`）。
- 全部 DIAG 代码已清除（§4 第一项）。清除后 `STRATA_Q36_PREFILL_DIAG` 这个开关已不存在。

## 10. 事故记录：构建产物装错位置，导致"修复无效"的假象（2026-10-22 晚）

**这是本次最容易误判的一处，记下来。**

远程构建写的是 `/home/Strata/build/strata-q36`，但**运行的是 `/home/Strata/bin/strata-q36`**。
`/home/Strata/bin/` 里那个是 **12 小时前**的旧二进制（10:08 22:46 的时间戳差异），
而 `cmake --build` 只更新 `build/` 目录。旧的 `.run/vbuild.sh` **没有拷贝步骤**。

后果：第一次"修复"之后的验证全部跑在旧二进制上，于是出现

```
改完源码 → 重新编译（RC=0，无 error）→ 运行 → delta 仍是 76722000 → "修复无效"
```

**这个"修复无效"的结论是错的**，而且错得很自然：日志看起来完全合理，
只有 `strings /home/Strata/bin/strata-q36 | grep 'DIAG slots'` 返回 0 才暴露了真相。

**教训（写进 §6 的坑）：**

1. **构建后必须确认二进制是新的**：`ls -la --time-style=full-iso` 比对 `build/` 和 `bin/` 的时间戳，
   或 `strings <binary> | grep <刚加的诊断串>`。`BUILD_RC=0` 只说明**编译**成功，
   不说明**运行的**那个文件是新的。
2. **诊断字符串是验证二进制身份的好探针**：往源码里加一句唯一的 `fprintf`，
   `strings` 一下就能确认运行的到底是不是这份源码。
3. `.run/vbuild.sh` 现在会 `cp -f build/strata-q36 bin/strata-q36` 并打印 `INSTALL_RC`。

**教训二（关于推理）：** 在"改了 A，结果没变"和"改了 A，结果应该变"之间，
我反复在**推理**上花时间（读 `admit()`、读 `layer_slot_range()`、猜 `per_layer_`），
而**一次 `strings` + 一次时间戳比对**就能立刻定位。**先确认你在测的东西是你刚改的东西。**

## 11. 模型路径变了（记录）

`/home/models/Ornith-1.5-35B-A3B-IQ4_XS.gguf` **已不在**远程机器上；
现在只有 `/home/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`（22.8 GB）。
§8 的数字是在旧模型上测的，§9.4 是在新模型上测的——**两组的量化不同**（IQ4_XS vs Q4_K_XL），
跨组比较要当心。

## 12. 独立缺陷：`--max-context 512` 的 IMA（2026-10-22 晚，MEASURED）

**与 §9 的两个缺陷无关**：peers **关闭**时同样崩。

| 配置 | 结果 |
|---|---|
| `--max-context 512`，peers 关 | ✗ `cudaStreamSynchronize(st) failed: an illegal memory access`（`q36_engine.cu:3413`） |
| `--max-context 512`，peers ON | ✗ 同上 |

日志：`context 512 cells, prompt chunk 2048`。

**根因（已证实，2026-10-22 晚，MEASURED）**：这是**两条独立的问题**撞在了一起。

1. **prompt 比 context 长**（用户配置问题）：`--bench-pp 512` 生成 512 个 token 的 prompt，
   而 `--max-context 512` 只有 512 格 KV（`q36_engine.cu:2114` 的 `g.n_head_kv * ctx * head_dim`）。
   512 个 token 的 prompt 正好占满，**没有余量给 MTP 块自己的 KV 和 decode**。
   - `--bench-pp 256 --max-context 512`：✓ EXIT=0，pp256 79.4 tok/s（实测）
   - `--bench-pp 512 --max-context 512`：✗ IMA（实测）
   差别就在 prompt 是否放得下。

2. **chunk 没有 clamp 到 ctx**（产品缺陷）：`chunk`（prompt 分块）由**空闲显存**决定，
   `q36_engine.cu:6120-6122`：`chunk = fr_gib >= 7.5 ? 2048 : fr_gib >= 5.5 ? 1024 : 512;`
   ——**与 ctx 无关**。日志 `context 512 cells, prompt chunk 2048` 就是这条：512 格的 context
   拿到了 2048 的 chunk。引擎照 2048 发 kernel，KV 越界写 → IMA。
   **这是真正的缺陷**：本该是"prompt 装不下 context"的干净拒绝，被放大成 GPU 非法访存。

**候选修法**（按用户友好度排序，都要实测）：
1. **prompt 长度 vs context 的校验前置**：在 `E.init` / bench 生成前，`bench_pp > ctx` 时报一条
   清楚的话（"--bench-pp 512 需要一个至少 512 格的 context，现在是 512，但 MTP 与 decode 还要占几格；
   要么 `--max-context` 调大，要么 prompt 调小"），**不是**让 GPU 访存错误。
2. **chunk 不许超过 ctx**：`chunk = min(chunk, ctx)`。兜底，防第 2 条。
3. 1 + 2 都做。

**这条独立于 §9**，可以单独开 PR（`[Engine]`，opt-in 路径不受影响，默认 ctx 32768 远大于任何
bench_pp，行为逐字节不变）。

**状态（2026-10-23）**：候选修法的 1 + 2 已写进 `q36_engine.cu`（工作区，未提交）：`chunk = min(chunk, ctx)`
与 prompt-vs-context 的校验（bench 路径与 `--tokens` 路径各一处，装不下时报一条清楚的话并 `return 2`）。
§13 验证了缺陷 2；本条的实机验证待做。

### 12.1 实机验证：全绿（2026-10-23，MEASURED）

机器 192.168.1.98，同一二进制 b7d53f977（§12 的代码在 10-22 23:32 那次编译里已经进去：
`strings bin/strata-q36 | grep 'does not fit --max-context'` 命中 2 处，源文件 md5 与本机一致——
没有重编）。脚本 `.run/R-s12-20261023.sh`，输出在 `/home/Strata/.run/R-s12-20261023/`。

| 用例 | 命令 | 结果 |
|---|---|---|
| S1 原 IMA 命令 | `--bench-pp 512 --max-context 512` | ✓ **RC=2，干净拒绝**，无 IMA。stderr：`--bench-pp 512 does not fit --max-context 512 (the KV cache holds one cell per token; 1 more cell is needed for the answer and the decode steps)` |
| S1b chunk 钳位 | 同上，看启动日志 | ✓ `context 512 cells, prompt chunk 512`（修前是 2048） |
| S2 放得下的 prompt | `--bench-pp 256 --max-context 512` | ✓ RC=0，pp256 79.4 tok/s（与 10-22 基线 79.4 相同） |
| S3 宽 context | `--bench-pp 512 --max-context 2048` | ✓ RC=0，pp512 158.5 tok/s |
| S4 默认路径 | `--tokens 32 --n 1`（ctx 2048） | ✓ `gen: 220`，与 §8.4 A5 基线一致 |

跑完机器干净（0 进程，四卡显存 0 MiB）。§12 的三条候选修法 1 + 2 至此闭环，可以单独开 PR。

## 13. 缺陷 2（row-split）真机验证：全绿（2026-10-23，MEASURED）

机器 192.168.1.98，模型 `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`，二进制 b7d53f977（bin/ 与 build/ 同 md5，
三个关键源文件 md5 与本机逐一相同——§10 的坑先绕开）。脚本 `.run/R-rowsplit-20261022b.sh`，
输出在机器的 `/home/Strata/.run/R-rowsplit-20261022b/`。

| 配置 | 结果 | 数字 |
|---|---|---|
| T1 崩溃命令，row split + peers ON + HOST（pp160/tg8） | ✓ rc=0，无 IMA，无 NaN | pp160 127.7–142.1 / tg8 46.9–52.8 |
| T2 纯 prompt，row split（pp160/tg0） | ✓ rc=0 | pp160 108.1–141.9 |
| T3 宽窗，row split（pp512/tg128） | ✓ rc=0 | pp512 203.6–213.4 / tg128 53.1–53.8 |
| T4 row split + peers 关，`--tokens 32 --n 1` | ✓ rc=0 | `gen: 220`（与 A5 基线一致） |
| T5 stripe + peers ON + HOST（回归，pp512/tg128） | ✓ rc=0 | pp512 169.2–171.4 / tg128 48.3–48.8 |
| T6 一 helper，row split（pp512/tg0） | ✓ rc=0 | pp512 209.8–214.2 |

- **缺陷 2 已修**：§8.4 里 row-split + peers ON + HOST 是「layer 0 起 NaN（pp512 220–227 但结果不算数）」；
  现在同一配置 rc=0、输出无 nan。§9.4 只测了崩溃命令；本次把 pp512 宽窗、peers 关的 parity（T4）与
  一 helper（T6）也补齐了。
- **没有回退**：T5 stripe 模式仍然全绿（§8.4 的 154–173 → 本次 169–171），T4 的 `gen: 220` 与
  10-22 的三次 A5 一致。
- **速度**：row split + peers ON 的 pp512 是 203.6–213.4（§9.4 报 246–254；本次 `--kv int8` +
  `STRATA_Q36_FIXED_SLOTS=1`，§8.4 的对照行没有这两个旗标，跨组只取方向：host 路由的收益仍在，
  明显高于 peers 关的 158 左右）。**这个差距已由 §13.2 的同旗标控制组定案：是旗标，不是修复。**
- **T6 一 helper**：pp512 209.8–214.2，rc=0，与三 helper 同档（row split 下 helper 少一倍，
  但每层的专家数也少一倍，所以 prompt 速度相近）。

全部六个 case rc=0，输出里 `illegal memory access` 与 `nan` 都是 0 行。

### 13.1 MTP 回归：修复后二进制上重测（2026-10-23，MEASURED）

§8.4 的 A6（`--mtp 2` 三 helper，tg128 52.1–53.0）是在**槽位修复前**的旧二进制上测的。槽位重编号
（a749100a）不碰 MTP 路径的数值，但 MTP 的 draft 步路由也走 helper 缓存，所以在本二进制上重测。
脚本 `.run/R-mtp2-20261023.sh`，输出在 `/home/Strata/.run/R-mtp2-20261023/`。

| 用例 | 配置 | 结果 |
|---|---|---|
| M1 A6 原形，三 helper，`--mtp 2` | pp512/tg128 | ✓ rc=0，无 IMA。pp512 154.4–154.5 / **tg128 51.5–52.8**（A6 是 154.3–154.5 / 52.1–53.0，重合） |
| M3 一 helper，`--mtp 2` | pp512/tg128 | ✓ rc=0。pp512 154.4–154.6 / tg128 17.2–18.2（单 helper 慢正常，同 §8.4 A7 的 17.1–17.3） |
| M2 + `STRATA_Q36_MTP_MAPPED=1` | pp512/tg128 | ✓ rc=0。pp512 154.1–154.5 / tg128 52.0–53.0，**但 stderr 报 INERT** |

**M1 的 tg128 落在 51.5–52.8，与 A6 的 52.1–53.0 基本重合，MTP 路径没被槽位重编号踩坏。**
M3 的单 helper tg128（17.2–18.2）也和 §8.4 A7（17.1–17.3）同向。**这两条就是 MTP 回归的有效门：
MTP 的 draft 步路由走 helper 缓存，槽位重编号后接受率与速度都不掉。**

**M2 是 INERT，数字不算对 mapped chain 的测量。** stderr：
`STRATA_Q36_MTP_MAPPED=1: INERT - the MTP block's layer is hybrid (its CPU expert pool spins on the
route event ...)`（`q36_engine.cu:1652` 那条 inert 消息）。原因：推荐配置 `--kv int8` 的 hybrid
专家让 MTP 块那一层也是 hybrid，mapped chain 在这个形状下本来就不可达（引擎启动时就把它判为
inert，chain 一次提交但每步照样等，等价于普通 `--mtp`）。所以 M2 和 M1 的数字相同是**预期**的，
不是「修复后 mapped chain 掉了速」。要真测 mapped chain，得换掉 hybrid 的形状（比如 `--kv fp16`
或让 MTP 块那层非 hybrid），那和 A6 的基线就不可比了——**这条不作为缺陷 2 的验收门，单独记一笔。**

（脚本第一次 M2 因 env 前缀写错 rc=127 没跑成；已用 `env VAR=... cmd` 重写为
`.run/R-mtp2-M2-20261023.sh` 补跑，上面的 INERT 结果是那次。）

### 13.2 速度控制组：246→203 的差是旗标，不是修复（2026-10-23，MEASURED）

§13 的 T3（row split + peers ON）报 pp512 203–213，而 a749100a 提交信息里同配置（同机同模型）报
246–254。两组旗标不同：T3 带了「推荐 1p3h 配置」的 `--kv int8` + `STRATA_Q36_FIXED_SLOTS=1`，
a749100a 没带。为分清是**修复掉了速**还是**旗标掉了速**，补一组同形状（无 `--kv int8`、无
`FIXED_SLOTS`、其余同 C1）的对照。脚本 `.run/R-ctrl-20261023.sh`，输出在 `/home/Strata/.run/R-ctrl-20261023/`。

| 用例 | 形状 | pp512 / tg128 |
|---|---|---|
| C1 row split + peers ON + HOST，**无** `--kv int8`、**无** `FIXED_SLOTS` | a749100a 的形状 | 176.3–262.3 / 48.0–48.5 |
| C2 同上但 **peers 关**（helper 只算不预取） | 无旗标的 peers-off 基线 | 139.0–157.3 / 45.1–48.0 |

- **定案：是旗标，不是修复。** C1（无旗标）的 pp512 均值 ~240，落在 a749100a 的 246–254 档位；
  T3（带 `--kv int8` + `FIXED_SLOTS`）是 203–213。同一修复二进制、同一模型，唯一差异就是这两个旗标，
  所以 246→203 的差由 `--kv int8` 的 KV 量化 + 固定槽的 rebalance 关闭贡献，**层序修复本身没有掉速**。
- **host 路由收益仍在**：C1（peers ON，~240）显著高于 C2（peers off，~154），与 a749100a 的
  「246 vs 156」同一方向。C2 的 157 也落在 10-22 §8.4 各 peers-off 行的 154–158 区间。
- 全程 rc=0、无 IMA。

至此 §13 的速度疑问闭环：修复无回退、无掉速，pp512 的组间差全部可归因到旗标组合。

### 13.3 prefill 加速器定案：1 主卡 + 3 helper（P4），2026-10-23，MEASURED

同一二进制 b7d53f977、同模型（Q4_K_XL）、纯 prompt（`--bench-pp 512 --bench-tg 0`），四条只变
「prompt 阶段用什么算力」，warm（bench 1/2 均值）：

| 组 | 配置 | pp512 | 机制（banner 确认） |
|---|---|---|---|
| **P4** | 1主卡 + 3 P104，row-split + `PREFILL_PEERS(+HOST)` | **268.5**（268.0/269.0） | 非驻留专家摊给 3 张 helper |
| P1 | 1主卡 + CPU 池，`--hybrid-rows 512`（默认线程） | 191.1（188.8/193.4） | 非驻留专家交 CPU 池、常驻专家 GPU，并行 |
| P3 | 1主卡独跑，默认 | 158.5（158.5/158.5） | 512 走 P100 MMQ int8 + 1.10 GiB 双缓冲 |
| P2 | 1主卡 + CPU 池，`--hybrid-rows 512 --pool-workers 16` | 157.5（158.2/156.8） | 16 worker 全 spin（引擎警告，推荐 7） |

四条 rc=0、IMA=0、nan=0。脚本 `.run/R-cpu-vs-helpers-20261023.sh`，输出
`/home/Strata/.run/R-cpu-vs-helpers-20261023/`。

**P4 定案配置（无歧义的完整串）：**
`STRATA_REMOTE_ZEROCOPY=0 STRATA_Q36_HELPER_ASYNC=1 STRATA_Q36_HELPER_PARALLEL=1 STRATA_Q8_SM60=1
STRATA_Q36_PREFILL_PEERS=1 STRATA_Q36_PREFILL_PEERS_HOST=1`
+ `--model <Q4_K_XL> --max-context 2048 --no-warmup --expert-cache-device1/2/3 auto`。

- **定案用 P4（1主卡+3helper）**：prompt 268.5，四条里最快；比纯主卡 P3 高 ~69%，比 CPU 池 P1 高 ~40%。
- **CPU 池参与 prompt 是净正的**（推翻本轮早期的预估）：P1（191.1）比 P3 的 MMQ（158.5）**快 20%**
  —— prompt 阶段 decode CPU 池是空的，让它吃非驻留专家有收益。`--hybrid-rows 512` 就是 q36 里这条的
  现成开关（banner `<= 128 rows` → `<= 512 rows` 确认切到 CPU 池），不用接 flash-next 的 CPU-share 代码。
- **CPU 线程数不是拉满最优**（推翻「加线程单调好」的预估）：P2 把 worker 拉到 16 反而回退到 157.5；
  P2 的 warning 是这台只跑 16 个 CPU 线程、16 个 worker 全 spin 抢核（引擎推荐 7）。这台 auto（8，P1）就是最优。
- **跨组注意**：P4 相对 §13.2 的 C1（~240）多带了 `STRATA_Q8_SM60=1`（C1 的基线没带），268 vs 240 的
  差一部分是这个旗标、不是 helper 数；四条内部（P3/P1/P2/P4）都带 Q8_SM60，彼此可比，跨组（P4 vs C1）只取方向。

实用推论：只在乎 prompt 快又不想动 3 张卡，`--hybrid-rows 512`（默认线程，**别拉满**）能把纯主卡 prompt
158→191；要最快 prompt 就是 P4（1主卡+3helper）。

### 13.6 P4 prompt 的每层时间分解（2026-10-23，**MEASURED** — 尺子出数）

> ⚠️ **本节的结论已被 §13.7 推翻。** 下面这版尺子（4 个边界）把 host 的阻塞时间算进了 GPU 跨度，
> 误把 18.7 ms/层的 host 等待说成"主卡 GPU 在算 attention"。保留原文是为了留下更正链，
> **不要照它做决策**。

**仪器**：`STRATA_Q36_PROMPT_TIMING=1`（新 opt-in 探针，见
[2026-10-23-q36-prompt-timing-probe.md](2026-10-23-q36-prompt-timing-probe.md)）。
二进制 `bin/strata-q36.prompt-timing` = `b82eca975014d46f4bb1934a3609de21`（**新文件名**；
对照的旧二进制 `bin/strata-q36` = `b7d53f977` **原封未动**）。源码备份 `/home/Strata/src.pristine/`。
rc=0、IMA=0、nan=0。

**bench 1（warm，pp512 268.9 tok/s）：**

| 段 | 每层 ms | 占比 |
|---|---|---|
| **attn+route+plan**（P1−P0） | **26.26** | **55%** |
| **primary_mmq**（P2−P1，含等 helper） | **21.02** | **44%** |
| combine（P3−P2） | 0.225 | 0.5% |
| post_combine | 0.017 | — |
| **四段之和** | **47.53** | **100%** |

**闭合校验**：四段之和 × 40 层 = **1901.1 ms = forward 总时间 1901.107 ms** → **GPU 时间线完整闭合，
没有未归因的 GPU 时间**。

**host 侧（wall clock，总计 / 每层）**：

| 段 | bench1 总计 | 每层 | 含义 |
|---|---|---|---|
| `sync3022` | 1101 ms | **27.5** | 主卡 stream 上 route 后的阻塞 sync——**它等的就是主卡 GPU** |
| `peer_plan` | 742 ms | 18.6 | host 循环（3 次 prepare） |
| `prep_sync` | 619 ms | 15.5 | 每 helper 的 row-plan sync（**含在 peer_plan 内**） |
| `prep_d2h` | 96 ms | 2.4 | 3 次阻塞 D2H（**走主卡链路**，印证 §13.4 的修正：~2.4 ms/层，不是 6.8） |
| `ppwait` | 1.7 ms | 0.04 | `peer_prefill_wait` **不阻塞 host**（只有一条 stream wait） |

**⚠️ 报告公式的缺陷**：`unattributed = forward − 四段和 − host 段和` 算出来是 **−2560 ms（负的）**
→ 说明 **host 段与 GPU 段大量重叠，且 host 段彼此嵌套**（`prep_sync` ⊂ `peer_plan`；
`peer_plan` 与 GPU 的 `attn+route+plan` 重叠——host 在 peer_plan 里忙时 GPU 在跑别的）。
**这个减法不成立**，host 段只能**单独读**、不能与 GPU 段相加。未归因余量要看 GPU 侧
（四段已闭合 → GPU 侧余量 = 0）。

**两条独立测量互证主卡 GPU 的 ~26 ms/层**：
- GPU event：P1−P0 = 26.26 ms/层
- host wall：`:3022` 那次阻塞 sync = 27.5 ms/层（它等的就是主卡 GPU 到 route 完成为止）

#### 这推翻了两条结论

1. **§13.5 说"主卡 GPU 活 ≈ 2.74 GMAC/层 ≈ 1 ms"，严重低估。** 漏算了：
   **GDN（30/40 层！`full_attention_interval=4` → 每 4 层 1 层 full attention，即 30 层 GDN + 10 层 attention）**、
   qkv/out projections、conv、逐元素的 norm/add。**主卡 GPU 每层真的 ~26 ms。**
2. **"墙在主卡"仍然成立，但主卡慢在 GDN/attention，不是专家算**（主卡的专家算确实是 0，`n_local=0`）。
   同时 **`primary_mmq` 那 21.02 ms/层就是等 helper 的时间**（含上腿 11.2 MiB 过 Gen1 x4），
   → **ev-wait 仍是第二个靶子，且它有 ~21 ms/层的实测量支撑（不是估算的 13 ms）**。

**优化排序因此改变**：主卡 GPU 的 26.26 ms/层（55%）现在是**头号靶**，
ev-wait 的 21 ms/层（44%）是第二个。**但要动主卡 GPU，先得知道那 26 ms 里 GDN 占多少**
（探针目前把 GDN/attention/router/route/peer_plan 的 host 墙钟混在一个跨度里，
需要再加边界才能分开）。

### 13.5 P4 的 prefill 计算量分配（2026-10-23，几何实测 + 算术）

**几何（从 GGUF 头直读，`qwen35moe.*` kv，type=4 uint32）：**
`block_count=41`（40 主干 + 1 MTP）、`expert_count=256`、**`expert_used_count=8`（top_k=8）**、
`embedding_length=2048`、**`expert_feed_forward_length=512`**、
`expert_shared_feed_forward_length=25`、`head_count=16` / `head_count_kv=2` / `key_length=256`。

**每层的条目与算量**（T=512 chunk）：512 token × top-8 = **4096 entries**；每 entry 三个专家矩阵
（gate+up: 2048×512×2、down: 512×2048）= **3,145,728 MAC** → 每层 **12.88 GMAC**，
40 层 **515 GMAC ≈ 1.03 TFLOP**（int8）。

**谁算（MEASURED 的分工，见 §13.4 的 n_local=0）：**

| 计算者 | 专家 GEMM | 每层 | 40 层 | 占比 |
|---|---|---|---|---|
| CUDA1 (P104) | `e%3==0` → 86/256 专家 | 4.29 GMAC | 172 GMAC | 33.3% |
| CUDA2 (P104) | `e%3==1` → 85/256 | 4.28 GMAC | 171 GMAC | 33.3% |
| CUDA3 (P104) | `e%3==2` → 85/256 | 4.28 GMAC | 171 GMAC | 33.3% |
| **P100 主卡** | **0**（`n_local = 0`） | **0** | **0** | **0%** |

**主卡在 P4 里做的非专家活**（每层、全部 512 token）：
共享专家 512×2098×2048 = 2.20 GMAC、attention ≈0.27 GMAC、router 512×2048×256 = 0.27 GMAC
→ **≈2.74 GMAC/层，40 层 ≈110 GMAC**。

**三个配置的对照：**

| 配置 | 专家 GEMM 在哪 | 非专家在�� | 主卡总负载 |
|---|---|---|---|
| **P4（3 helper）** | 515 GMAC，**全在 3 张 P104** | 110 GMAC，全在 P100 | **110/625 = 17.6%** |
| P3（无 helper） | 515 GMAC，**全在 P100** | 110 GMAC | **625 GMAC = 100%** |
| P1（CPU 池） | 主卡 40/256，CPU 池 216/256 | 110 GMAC | 主卡 ≈190 GMAC |

**读出来的三条（第三条是 ESTIMATE）：**
1. **P4 把 515 GMAC 全部推给 3 张 P104，主卡专家算力归零**，只做 110 GMAC 的非专家活。
2. **主卡负载 100% → 17.6%，总吞吐只涨 69%**（158.5 → 268.5）：3 张 P104 是 sm_61，P100 是 sm_60，
   3 张小卡的专家算力约等于 1.2~1.5 张大卡，主卡那 515 GMAC 挪过去只省下一部分时间。
3. **主卡剩下的 82% 算力不能用"把专家塞回主卡"变现**（ESTIMATE）：主卡的专家算与它的
   attention/router/共享专家在**同一条 stream 上串行**（每层一个 `st`），时间变成
   `T_非专家 + T_专家` 叠加而非重叠 → **会更慢**。这解释了为什么"P4 让主卡闲着"反而快。
   要用那 82%，得让主卡的专家算与别的活**重叠**（另开 stream），不是把活搬回去。

### 13.4 P4 卡的是主卡，不是卡间带宽（2026-10-23，MEASURED）

§13.3 定案 P4（1主卡+3helper，prompt 268.5）。本节回答"优化 P4 的卡间通信能不能再提速 prefill"，
先填了缺的那格（1-helper、P4 旗标、纯 prompt），再判 P4 到底卡在哪、砍桥字节值不值得做。

**1-helper 定案实验（填 §13.3 缺的曲线中间格）：** 同二进制 `b7d53f977`、P4 完整旗标串、只把 helper 降到 1
张（`--expert-cache-device1 auto`，去掉 device2/3）、`--bench-pp 512 --bench-tg 0`。banner 确认
`prompt experts on MMQ (int8)`、`helper CUDA1: 4026 experts ... host route ... staged through host memory`，
rc=0、IMA=0、nan=0。warm pp512（bench 1/2 均值）= **205.0**（202.8/207.2）。输出
`/home/Strata/.run/settle-1helper/`。

**口径（必须写清，否则冷档会被拿来误比）：** pp512 单点方差很大——P4 三档 176.3 / 268.0 / 269.0、
1-helper 三档 241.0 / 202.8 / 207.2（bench 0 受 helper cache 首填 + 测量噪声，三档互不一致）。
**对比一律取 warm = bench 1/2 均值**（相邻两次都稳：P4 268/269、1-helper 203/207），**bench 0 不入对比**。
所以 1-helper 的 bench 0 (241) 比 P4 的 bench 0 (176) 高不说明任何问题；P4 那个 176 也不能拿来比 205。

helper 数 × prompt 的完整曲线（都 P4 旗标、纯 prompt、warm = bench 1/2）：

| helper | pp512 | 0→1 | 1→3 |
|---|---|---|---|
| 0（P3，主卡独跑） | 158.5 | — | — |
| **1**（本次） | **205.0** | **+29%** | — |
| 3（P4） | 268.5 | — | 1→3 仅 **+31%** |

**判定分两层，MEASURED 和 ESTIMATE 分开：**

- **MEASURED —— 排除带宽/通信 bound：** 1-helper 落在 205，**没有 cliff**。若桥（host 往返/PCIe
  字节）是墙，砍掉 2/3 的桥流量、prompt 该塌到 ~90（≈268/3）或 P3 的 158 附近；实测只降 24%。
  这条是干净的实测：同二进制、同 P4 旗标串、单变量只动 helper 数（banner 逐行核过）。
- **ESTIMATE —— floor 归属主卡串行工作（非实测）：** prompt 不随 helper 数线性 scaling
  （0→1 吃 +29%、1→3 只多 +31% 强递减）、decode 却 ~3×（§13.1 tg128 17.2–18.2 → 53.1–53.8），
  这个形状指向"每层固定延迟 floor"。该 floor 归因为主卡自己的串行工作
  （全 attention + router + 每层 engine `cudaStreamSynchronize`（`q36_engine.cu:3022`）+ 共享专家 +
  `moe_end` 合并），不随 helper 数变。**这是从 scaling 形状推的，没做 per-segment timing，所以
  "floor 归因"是 ESTIMATE，不是 MEASURED。** 要坐实归因**不用改代码**：引擎自带
  `STRATA_Q36_HELPER_TIMING=1`（`q36_engine.cu:4867-4871`），开它建 8 个带 timing 的 per-layer 分段事件
  （`ev_seg_ev`，:2045 `ev_mark` / :2062-2065 `gseg_ring` 回读 / :6405 报告）。
  **但该 probe 只量 decode 的单 token**（:2061-2063 明写"Only a T==1 forward records a tile … n>1
  could not be measured with them at all"），**prompt 的分段分解它测不了**——要测得改代码。

- **⚠️ 2026-10-23 修正（读码核到，推翻本节早先的一处归因）：**
  **P4 里主卡在 served 层上一个专家都不算（`n_local = 0`）。**
  依据：3 张 helper 按 `expert % 3` 的分层规则持走了**每层每一个**专家
  （banner 3440+3400+3400 = **10240 = 40 层 × 256 全部**）；`open_helpers()` 传的是 **empty_primary**
  cache（0 槽），所以 `RemoteExperts::open()` 的 `primary.slot_of(...) < 0` 判据**恒真**、
  helper 可以持有主卡也常驻的专家（`q36_engine.cu:3344-3350`）；`peer_owner` 表由
  `peers[r]->owns(l, e)` 构建（:3855-3860），于是每个专家的 `own != kPeerNone`，
  `peer_plan` 的 pass 1（:3700-3705）数出 `n_local = 0` → 主卡的 MMQ 无行可算
  （`gu.total_rows = 0`）。`stream_kick` 的注释（:3286-3291）说的正是这个：
  "the primary's MMQ on a served layer only computes the entries of the experts **nobody else holds**"。
  **注意：`local_grouped = 0.000` 不是这条的证据**——P4 走 MMQ 分支，而 `ev_mark(kSegGrp)` 在
  `if (hy_rows)` 块内（:2973），MMQ 分支**不记录 E3 事件**，所以那一段恒为 0（构造出来的，不是测出来的）。
  **支撑（2026-10-23 补）**：`.run/peer_r6.err` 里 40 行 `DIAG mmq layer N n_local=0`（**运行时实测**，
  10/21 的诊断构建、layer-stripe 形状、helper 同样持 10240 = 全部专家）；P4 形状下是**算术验证**
  —— banner 3440+3400+3400 = 10240 = 40×256，`e%3` 把 256 分成 86/85/85（每层全覆盖），
  `RemoteExperts::open()` 的取专家判据（`src/core/remote_experts.cpp:262`）是
  `primary.slot_of(...) < 0 && !claimed && !picked`，而 `primary` 是 **empty_primary**（0 槽）→
  第一项恒真 → helper 可持有主卡也常驻的专家 → 每个专家都有 owner → `n_local = 0`。
  **⚠️ 边界（复核发现）：这条不是"配置保证"而是"这台机器的事实"**——`auto` 的 auto_size 有
  VRAM 截断（`remote_experts.cpp:267`：`needed + bytes + 512MiB > free_bytes` 就 break），
  装不下的专家**留在本地路径**（`q36_engine.cu:5796-5798` 明写"an expert a helper could not afford
  is left to the local paths"），那时 `n_local > 0`。本机没触发截断（三张都装满）。
  **所以本节早先写的"floor 包含主卡那 ~1/3 常驻专家 MMQ"是错的——那部分为 0。**
  主卡的真实 floor = 全 attention + router + 每层 sync + 共享专家 + combine。
  （"排除带宽 bound"那半不受影响：1-helper 没塌是实测。）

  **⚠️ 另一处更正（复核发现）：主卡那 40/256 常驻槽（2.93 GiB）不是"浪费"。**
  peer 只服务 prompt chunk（`peer_window_fits` 要求 `n_ent > hybrid_t*top_k = 1024`，decode 的
  `n_ent=8` 不满足）；**decode 走 `hy_rows` 路径**（`q36_engine.cu:2946`，T=1 ≤ hybrid_t=128）
  → `moe_hybrid_gpu` → `native_expert_grouped` **每 token 直接读 `L.blobs`**
  （:3938）。MTP 块也永不被服务（`peer_owned_layer` 对 `li >= n_layers` 返 false，:3619），
  `forward_mtp` 读自己的 `L.blobs`。所以那 40 个槽在 decode 和 MTP 里**有用**，
  只有"served 的 prompt 层"上不读——那不叫浪费。

**结论（两层合起来）：P4 卡的是主卡串行工作、不是卡间带宽（带宽排除是 MEASURED，主卡归因是 ESTIMATE）。
"砍桥字节"（flash-next F16/SUMS）解的是带宽，不解这个延迟 floor，所以对 P4 prompt 是二阶。**

**卡间通信账（`q36_peer_prefill.cu` host-route，每层每 helper，全 fp32、全过 pinned host）：**

| 腿 | 字节 | dtype | 关键路径 |
|---|---|---|---|
| 下腿·激活（:650-653，阻塞 D2H） | T×2048×4 | fp32 | **hidden**（engine 已在 :3022 drain，zero-wait） |
| 下腿·路由 id（:654-655） | 4×T×k | int32 | hidden |
| 上腿·H2D 回 helper（:674-677） | T×2048×4+4×T×k | fp32 | hidden（helper stream） |
| 8 字节回读+`cudaStreamSynchronize`（:701-707） | 8 | int32×2 | **on-critical**，但在 **helper stream**（:411），非主卡 compute stream |
| 上腿·结果行 D2H→h_out（:801-804） | o×2048×4 | fp32 | hidden（helper stream，event 才取回） |
| 上腿·H2D h_out→eparts（:828-835） | o×2048×4 | fp32 | **depends**（主卡 stream、event 排序，在 MMQ 之后） |

**flash-next 降字节机制的移植裁定（对抗验证后）：**

| 机制（flash-next 名） | 移植到 q36 | 判定 |
|---|---|---|
| `STRATA_PF_PEER_XSTREAM`（砍 8B 回读+sync） | `STRATA_Q36_PEER_XSTREAM` | ✗ 不成立：q36 那个 sync（:706）在 helper stream，下腿本就是 off-stream 阻塞拷贝；真卡主卡的是 :3022 engine sync + host-side MMQ sizing（:3025/:3046/:712-713），flash-next 没有对应旗标，要 device-side launch sizing |
| `STRATA_PF_PEER_SUMS`（o 行→每 token 加权和） | `STRATA_Q36_PEER_SUMS` | ✗ 定位不成立：机制描述准确，但"最大字节砍、桥 bound 时的正确 reducer"被 prompt plateau 反证（砍字节不减传输次数） |
| `STRATA_PF_PEER_F16`（宽度减半，上下腿） | `STRATA_Q36_PEER_F16` | ✓ 站得住（**不作为 prompt 提速**）：N=2048 满足 `%8==0` 门；只作"降 PCIe 字节、也喂 3×-scaling 的 decode helper"，prompt 侧二阶 |
| 上腿 H2D（eparts 合并前那段） | 同上 F16 | ✓ 唯一 on-critical 的桥字节，但占比低（~1 ms / ~39.6 ms 每层，低个位数 %） |

**净结论（2026-10-23 复核后修正）：** **"桥字节是二阶"这个结论要收回一半。**

- ✅ **仍成立**：prompt **不随 helper 数 scaling**（0→1 +29%、1→3 +31%），而 decode 3×。
  这条实测说明**总桥流量与 helper 数无关**（每 helper 摊到的活变少，总量不变），
  所以**砍字节不能改变这个 plateau**——"F16/SUMS 对 prompt 二阶"在**相对**意义上仍对。
- ❌ **要收回**：§13.4 早先把桥字节的时间按 **PCIe x16 ~12 GB/s** 估（"上腿 H2D ~1 ms/层，
  低个位数 %"）。**实测这台机的链路不是 x16**：
  `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current` →
  **3× P104 = Gen 1 x4（~1.25 GB/s）**、**P100 = Gen 3 x8（~8 GB/s，max x16）**。
  按 Gen1 x4 重算，**桥字节的绝对时间大一个数量级**（见 §13.6）。
- ⚠️ **所以"墙在主卡"这个归因要重开**：主卡的 GPU 算力确实只用了 17.6%（§13.5），
  但 47.7 ms/层里**主卡的活只占约 1 ms**——**剩下的 40+ ms/层在别处**（helper 侧算 + 桥 + host 侧
  串行等待），而 helper 的桥是 **Gen 1**。**"排除带宽 bound"的那半（1-helper 不塌）依然成立，
  但它排除的是"总流量随 helper 数变化"这种 bound，不是"带宽本身"。**

**要坐实归因，唯一的下一步是给 prompt 路径造一把尺子**（现有 `STRATA_Q36_HELPER_TIMING=1`
只量 decode 单 token，§13.4 已核），量出每层的 GPU / host / 桥 / helper 四段占比。
方案见 §13.6。



### 13.7 第二版尺子：每层时间分解（2026-10-23 晚，**MEASURED** — 推翻 §13.6）

**仪器**：`STRATA_Q36_PROMPT_TIMING=1`，**8 个边界**（per-layer event ring）：
P0 层入口 / P0b GDN-attention 结束 / P0c route 核后 / P0d route D2H 的 sync 后 /
P1 plan 返回 / **P1b 在 `peer_prefill_wait()` 之前** / P2 `moe_end` 前 / P3 `moe_end` 后。

二进制 `bin/strata-q36.prompt-timing2` = `c3ffcce50f3dd6d2dbe7ef1ddf61a6f0`（**新文件名**）；
对照 `bin/strata-q36` = `b7d53f977` 与 `bin/strata-q36.prompt-timing` = `b82eca97` **都原封未动**
（编译后核过 md5 没变）。远端源码备份 `/home/Strata/src.pristine/`。
P4 旗标，`--bench-pp 512 --bench-tg 0`。**rc=0、IMA=0、nan=0**。
bench：0=173.5 / **1=267.7** / 2=267.9 tok/s
（第一版尺子那次 175.6 / 268.9 / 268.5 → **尺子本身对速度无影响**，差 0.4% 在噪声内）。

**bench 1（warm），47.734 ms/层，GPU 时间线完整闭合（`gpu_residual = 0.000`）：**

| 段 | 每层 ms | 占比 | 是什么 |
|---|---|---|---|
| **plan**（P1−P0d） | **18.765** | **39%** | **host 串行等 3 个 helper 的 `prepare()`**（GPU 空闲） |
| **helper_wait**（P2−P1b） | **12.900** | **27%** | 等 helper 的行落地 + 上腿 H2D |
| **primary_work**（P1b−P1） | **8.125** | **17%** | 主卡自己的活（`n_local=0` 的空 MMQ + 共享专家） |
| **mixer**（P0b−P0） | **7.096** | **15%** | GDN / attention 本体 |
| router_route | 0.537 | 1% | router + route 三核 |
| route_d2h | 0.060 | <1% | route D2H + 它等的 sync |
| combine | 0.232 | <1% | `moe_end` |
| post_combine | 0.018 | <1% | 层间 |

host 墙钟（每层 ms）：`prep_sync` **15.690** / `prep_d2h` 2.440 / `peer_plan` 18.758 /
`sync3022` 27.472 / `ppwait` 0.047。

#### 这推翻了 §13.6 的两条结论

1. **"主卡 GPU 每层真的 ~26 ms"——错。** 那 26.26 ms/层里 **18.765 ms 是 host 在阻塞**
   （`peer_plan` 的墙钟 18.758 与 GPU 跨度 18.765 **相差 0.007 ms**，互证），
   GPU 那段时间**空转**。主卡 GPU 自己的活只有 **7.096（mixer）+ 8.125（primary_work）
   = 15.2 ms/层**。
   §13.6 的错因：它把 P1−P0（P0 在层入口、P1 在 plan 返回）当成"GPU 的 attention 时间"，
   但那段的**墙上时间**里 host 阻塞了 18.7 ms，GPU 事件测的也是墙上时间。

2. **"未解释的 ~8 ms"——不存在。** 它一直在这 26.26 里，就是 `plan`。
   §13.6 把 26.26 整体归给 attention/GDN，是把 host 的等待算进了 GPU 的账。

3. **§13.4 的"P4 卡的是主卡"——也要改。** 主卡 GPU 15.2 ms/层（32%），而
   **host 串行等 helper 的 18.8 ms/层（39%）更大**。

#### 新的头号靶子：`peer_plan` 里 3 个 helper 的 `prepare()` 串行

`peer_plan`（`q36_engine.cu:3967`）的循环是：

```cpp
for (int r = 0; r < 3; ++r) {
    peers[r]->prepare(...);   // ← 阻塞：prepare() 尾部有 cudaStreamSynchronize
    peers[r]->launch(...);
}
```

`prepare()`（`q36_peer_prefill.cu:567`）的阻塞点是 **:737 的 `cudaStreamSynchronize(m.stream)`**，
等的是**这个 helper 自己**的下腿 H2D + 4 个 row-plan 核。
**3 个 helper 串行 → 三次下腿传输不重叠。**

**下腿的字节**：每个 helper 收的是**全部 512 token** 的激活
（`n_tok = 512`，`n_tok * n_embd * 4` = **4 MiB**），3 个 helper 各 4 MiB =
**12 MiB/层过 Gen1 x4（~1.25 GB/s）→ 理论 ~9.6 ms**；实测 `prep_sync` = 15.69 ms/层
（含 row-plan 核与 launch 延迟；每次 prepare 平均 5.23 ms）。

#### `primary_work = 8.125` 里主卡在算什么

`n_local = 0` → 主卡 MMQ（quantize/gu/swiglu/dn）**一行都不算**，但核仍发射。
真正的活是**共享专家**（`q36_engine.cu:3219-3232`，T=512 > kGemmT=8 的分支）：
`k_to_f16`(512×2048) + `apply(sh_gate)` + `apply(sh_up)` + `k_swiglu` + `mm(sh_down)`，
共享专家宽度 FS = `n_ff_shexp` = 25（很窄）。8.1 ms/层对共享专家**偏大**，但这是主卡自己的、必须的活。

#### `helper_wait = 12.900` 是什么

P2−P1b = 等 `m.ev`（helper 全部完成：下腿 H2D + row-plan + MMQ + **上腿 D2H**）
+ **上腿 H2D**（`h_out` → `eparts`，11.2 MiB 过主卡 Gen3 x8）。
上腿 D2H 过 **helper 的 Gen1 x4**（~9 ms），H2D 过**主卡 Gen3 x8**（~1.4 ms）——
与 §13.4 的估算量级一致。

#### 下一步的三条候选（按实测重排，与 §13.6 的排序不同）

| 候选 | 实测依据 | 预估收益（DERIVED） | 风险 |
|---|---|---|---|
| **A. 3 个 helper 的 `prepare()` 并行** | `plan` = 18.765（39%），三次 4 MiB 下腿不重叠 | 最多省 ~9 ms/层（18.8 → ~10） | 中：要拆成"先全部 prepare，再全部 launch"（`launch()` 依赖 `prepare()` 给的行数） |
| **B. 下腿 fp16** | 下腿 12 MiB/层过 Gen1 x4 | 省 ~4.8 ms/层 | 低-中：只改传输格式，helper 侧量化前转回 fp32 |
| **C. 上腿 fp16** | `helper_wait` = 12.900 里的上腿 11.2 MiB | 省 ~2.8 ms/层 | 低-中：改数值路径，要 logits 容差对照 |

**A 的收益最大且不动数值路径 → 先做 A。**

### 13.8 候选 A：3 个 helper 并行 prepare（2026-10-23 晚，实施中）

**动机**：`plan` = 18.765 ms/层（39%），是 `peer_plan` 的 host 循环（`q36_engine.cu:3967`）里
**3 个 helper 串行** `prepare()` 造成的：每个 `prepare()` 的末尾有一次
`cudaStreamSynchronize(m.stream)`（`q36_peer_prefill.cu:737`），等的是**这个 helper 自己**的
下腿 H2D（4 MiB，走它的 Gen1 x4）+ 4 个 row-plan 核。三个串行 → **三次 4 MiB 传输不重叠**。

**改法**：把 `prepare()` 拆成两半。

| 新接口 | 做什么 | 阻塞吗 |
|---|---|---|
| `prepare_begin(layer, h, ids, n_tok, err)` | 驻留遍历 + 分组 + **D2H staging** + H2D 上传 + 4 个 row-plan 核 + 两个读回**全部发射** | **不阻塞**（D2H staging 除外，见下） |
| `prepare_end(err)` | 等这条 stream，把行数与最宽专家读回来 | 阻塞（每 helper 一次） |

`peer_plan` 改成**先对 3 个 helper 全 `prepare_begin()`，再对 3 个 helper 全 `prepare_end()` + `launch()`**。
三次等待因此落在**已经在飞**的工作上 → 总时间 = max 而非 sum。

**一个必须的细节**：那两个读回（行数、最宽专家）的目标**必须是 pinned 内存**。
`cudaMemcpyAsync` 写**可分页**主机内存时驱动会退化成**同步**拷贝 —— 那会让 `prepare_begin()`
当场阻塞，并行化整个归零。原来的代码用栈上的 `h_owned`/`h_max`（可分页），
所以原来的那两个读回本来就是同步的（那正是 `prep_sync` 大的原因之一）。
改成每 helper 一块 8 字节的 pinned（`m.h_plan[2]`）后，读回才真的排在 stream 上。

**不变量**：
- **行契约不变**（行数、行序、每行语义都没动）。
- **两个 witness 都保留**：pass 1 的所有权交叉核验（`peer_plan` 的专家遍历）与
  launch 后的 `rows()` 重复行检查，一个都没绕过。
- **P2P 路线不变**（`h_plan` 两条路线都分配，但 P2P 路线的行为与字节数没变）。
- **默认路径逐字节不变**（接口变了，行为没变；`prepare()` 保留为 begin+end 的包装）。

**已知的残留**：D2H staging（主卡 → host，每 helper 4 MiB）是**阻塞的 `cudaMemcpy`**，
三次仍然串行 → `prep_d2h` = 2.44 ms/层 没省。**下一步**（候选 D）：
主卡 → host 只拷**一次**（三个 helper 收的是**同一份** `hdev`/`idsdev`/`n_tok`），
三个 helper 各自从 host 拷 → 省 ~1.6 ms/层。

**验证**（`bin/strata-q36.parallel-prep` = `7f1ae9f6`）：
- A1 对照 = `bin/strata-q36`（`b7d53f977`，**未动**）
- A2 = 新二进制，**探针关**（量纯速度）
- A3 = 新二进制，**探针开**（量 plan 段是否真的降了）

#### 13.8.1 候选 A 的实测（2026-10-23 晚，**MEASURED**）

远端 192.168.1.98，同旗标串，`--bench-pp 512 --bench-tg 0`。二进制：
对照 `bin/strata-q36` = `b7d53f977`；新 `bin/strata-q36.parallel-prep` = `7f1ae9f6`。
**两个旧二进制（`b7d53f977`、`b82eca97`、`c3ffcce5`）全部原封未动**（编译后核过 md5）。

| 趟 | 对照（A1，探针关） | 并行 prepare（A2，探针关） | 并行 prepare（A3，探针开） |
|---|---|---|---|
| bench 0 | 176.7 | **202.9** | 195.6 |
| **bench 1（warm）** | **265.3** | **310.2** | **311.4** |
| **bench 2** | **268.3** | **311.4** | **311.2** |

**warm 均值：266.8 → 310.8 tok/s = +16.5%**（按 bench1/2 均值：266.8 → 310.8）。
A2 与 A3 一致 → **探针本身对速度无影响**。rc=0、IMA=0、nan=0（每一趟）。

**每层时间去哪了（A3，bench 1，全部 ms/层）：**

| 段 | 并行前 | 并行后 | 变化 |
|---|---|---|---|
| **plan** | 18.765 | **8.501** | **−10.26** ✅ |
| **prep_sync**（host） | 15.690 | **5.016** | **−10.67** ✅ |
| prep_d2h（host） | 2.440 | 2.794 | +0.35 |
| peer_plan（host） | 18.758 | 8.380 | −10.38 |
| helper_wait | 12.900 | **16.434** | **+3.53** ⚠️ |
| primary_work | 8.125 | 8.136 | +0.01 |
| mixer | 7.096 | 7.085 | −0.01 |
| **forward** | **47.734** | **41.026** | **−6.71** |

**`prep_sync` 15.69 → 5.02 正好是预期的"max 而非 sum"**（单次 prepare 原来 5.23 ms，
三次重叠后 5.02 ms）→ **并行化按设计生效**。

**总账闭合**：−10.26（plan）+ 3.53（helper_wait）= −6.73 ≈ forward 的 −6.71 ✅

**`helper_wait` 为什么涨了 3.53**：上腿 D2H（helper → host，11.2 MiB @ Gen1 x4）在 `launch()` 里，
排在 helper 自己的 stream 上。**并行前**，helper 0 早早 prepare 完 → 早早 launch → 它的上腿 D2H
在主卡还在算 MMQ 时就飞完了 → 等 `m.ev` 时它已过半。
**并行后**，三个 helper 的 prepare 几乎同时结束 → 三个上腿 D2H **挤在同一时刻**，
主卡等的是**最后那个** → `helper_wait` 涨。
**这 3.53 ms 是"被推后"，不是新增的活** —— 它本来就在那 12.900 里（只是原先被藏住了）。

**下一步（候选 D）**：三个 helper 收的是**同一份** `hdev`/`idsdev`/`n_tok`，
主卡 → host 的 D2H 可以**只拷一次**，三个 helper 各自从 host 拷 →
省 `prep_d2h` 的 ~1.6 ms/层，且让三个下腿 H2D 更早开始。
