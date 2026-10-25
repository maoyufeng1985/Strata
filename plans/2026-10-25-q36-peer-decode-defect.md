# q36：「prompt 后接 decode」的那条缺陷（2026-10-25 查清）

状态：查清了。机器 192.168.1.98（1× Tesla P100-12GB 主卡 + 3× P104-100 helper），CUDA only。
二进制：`bin/strata-q36.bps`（= HEAD 减去 `open_helpers` 里 MTP_VRAM 的一个 hunk，旗标关时是 no-op）。

## 0. 一句话

**那条缺陷是三件事叠起来的，最后崩的不是内存被踩坏，是一个哨兵 token id 当成了行号：**

```
helper 缓存的 slot 间距（256 对齐）≠ 主卡的 L.stride（两种 block 大小的 lcm 对齐）
  → helper 的 MMQ 从错位的地址读专家权重 → 算出 NaN
  → NaN 顺着后面的层传下去 → 最后一层的 logits 全是 NaN
  → greedy 的 argmax 在「没有比 -inf 大的值」时返回哨兵 0x7fffffff
  → decode 的第一步把它当 token id：0x7fffffff × 行字节 → 越界读 → IMA（在 prompt 后第一次同步时报）
```

**根因是 10-21 查出来的、10-22/23 修掉的（`plans/2026-10-21-q36-peer-ima-root-cause.md`）**，
peer 归属表的编码错是第二个（10-24 修的，`53e05a83`）。
**今天（2026-10-25）那条崩溃命令不再复现**（§2）。**引擎里那句警告是旧的，没跟着修**（§4）。

## 1. 缺陷是什么（记录，MEASURED 2026-10-21/22）

`plans/2026-10-21-q36-peer-ima-root-cause.md` §0 的原表：

| | layer 33（正常） | layer 34（坏） |
|---|---:|---:|
| down 的量化 | Q5_K（176 字节/块） | **Q6_K（210 字节/块）** |
| 主卡 `L.stride` | 1900800 | **2041200** |
| helper 实测地址差 | 1900800 ✓ | **2041344** ✗（多 144 字节）|
| `dn.expert_bytes / 210` | 10800 ✓ | **9720.7 → 截断 9720** ✗ |

- 主卡的步长按 `lcm(block(gu), block(dn))` 对齐（`q36_engine.cu:1467`），**保证能被两种 block 大小整除**；
- helper 缓存的步长只按 256 对齐（`expert_cache.hpp:134`）⇒ layer 34（2041200 不是 256 的倍数）
  的第 1 个专家之后每一个都从错位的地址读。
- 修法：`NativeExpertLayout::stride`（`iq_kernels.hpp:56`）+ `ExpertCache::set_slot_align` +
  `RemoteExperts::open` 用 `max(L.stride, blob_bytes)`（`remote_experts.cpp:305`）——
  **helper 的 slot 间距改成主卡的步长**。10-23 在真机验证过。
- 第二个缺陷（row-split 模式：槽位跨层排序）由 `a749100a` 修（helper 缓存按 layer-major 排序）。

## 2. 今天还复现吗（MEASURED，2026-10-25，`bin/strata-q36.bps`）

旗标：`STRATA_Q36_PREFILL_PEERS=1 STRATA_Q36_PREFILL_PEERS_HOST=1 STRATA_Q36_PEER_F16=1`
+ `STRATA_REMOTE_ZEROCOPY=0 STRATA_Q36_HELPER_ASYNC=1 STRATA_Q36_HELPER_PARALLEL=1 STRATA_Q8_SM60=1
STRATA_Q36_HELPER_PARALLEL_MARSHAL=1 STRATA_Q36_FIXED_SLOTS=1`；
`--max-context 2048 --no-warmup --expert-cache-device1/2/3 auto`。
**dmesg 的 Xid 计数在每趟前后各记一次**（这台机器历史上累计 138 次 Xid）。

| 趟 | 命令 | rc | IMA | Xid 前后 | pp | tg |
|---|---|---:|---:|---|---:|---:|
| C1 | `--bench-pp 8 --bench-tg 128` | 0 | 0 | 138→138 | 162.1 | 60.4 |
| C2 | `--bench-pp 128 --bench-tg 128` | 0 | 0 | 138→138 | 44.9 | 59.0 |
| **C3** | **`--bench-pp 256 --bench-tg 128`（记录里的崩溃配置）** | **0** | **0** | **138→138** | **454.7** | **59.4** |
| C4 | 同 C3 再来一次 | 0 | 0 | 138→138 | 454.4 | 59.2 |
| C5 | `--bench-pp 512 --bench-tg 2`（记录里的 tg2 崩溃） | 0 | 0 | 138→138 | 493.9 | 53.7 |
| C6 | `--bench-pp 512 --bench-tg 128` | 0 | 0 | 138→138 | 495.5 | 59.3 |

（数字是 bench 2，`Xid 前后` 是 dmesg 的 Xid 计数：这台机器历史上累计 138 次，
**六趟跑完还是 138 次** ⇒ 没有新的驱动侧越界。）

**⇒ 2026-10-21 的那条 IMA 不再复现，dmesg 也没有新的 Xid。** 崩溃的原因是那两个已修的缺陷，
不是「prompt 之后接 decode」这个组合本身有什么错。

## 3. 剩下的不同：不是崩溃，是**数字**

**peer 路径里 prompt 的专家由 helper 算**（q8_1 激活 + `native_expert_grouped`），
**默认路径里由主卡的 MMQ int8 算** ⇒ 两条数值路径不同 ⇒ KV 不同 ⇒ 生成的 token 可能不同。
**这不是 race，是这条路的性质，而且仓库里早就写明了**（`q36_engine.cu:4081-4092`）：
「the VALUES are not the local path's arithmetic … enabling a helper moves batch entries between two
different (each internally consistent) implementations, **and the token stream diverges accordingly** —
measured on 192.168.1.98 with a forced 36-token prompt: the per-row RMS logit difference is 1.5e-1 to 3.3e-1
(relative ~5e-2) even where argmax still agrees」。

### 3.1 三次「peer 开 vs 关」的比较，只有两次有效

**（一）10-25 早先那次「37 行 logits 里 36 行 argmax 相同、max|diff| 0.837」**：
用 `--tokens 1 --force <37 个 id>`。**`--tokens` 后面跟的是一串 id**（`parse_ids` 把它切成一条
prompt，`q36_engine.cu:7768`），而 `--tokens 1` 就是**一个 token 的 prompt**；`--force` 是被强制的
**生成**序列，不是 prompt。peer 路径只对宽于 hybrid_t(128 行) 的 prompt chunk 生效 ⇒
**那次 peer 的算术根本没跑**，量到的只是**专家放置的差别**（peer 开 = 主卡每层留 39 个专家，
关 = 40 个）⇒ decode 的数值路径变了 ⇒ 一个 token 不同。**这条比较当 peer 路径的证据是不成立的。**

**（二）10-25 的 N1–N4 那次**（`--tokens <216 个 id> --n 32`）：**prompt 是 216 个 token，peer 路径真的跑了**
（stderr：`3 helper(s) serve prompt chunks (8680 experts held by helpers, 40 layers served …)`）：

| | prompt 216 token | 生成 32 token | 文字 |
|---|---:|---:|---|
| peer 关（N2）| 3273.8 ms（66.0 tok/s）| 567.6 ms（56.4 tok/s）| `… 220 17 25 …` |
| peer 开（N3）| **543.4 ms（397.5 tok/s）** | 535.8 ms（59.7 tok/s）| `… 220 **16** 25 …` |

**⇒ peer 路径把这个 prompt 快了 6 倍（66.0 → 397.5 tok/s），而生成的文字第 12 个 token 不同。**
默认路径自己两次（N2、N4）**逐字节相同**（确定性 ✓）；
peer 路径的确定性没测到（N1 被我自己并发跑的另一个脚本误杀，rc=137）。

**（三）10-25 的 M1–M4**（`--tokens <216 个 id> --n 32` + `--dump-full-logits`，
新二进制 `bin/strata-q36.test`，peer 开/关各两次，比 dump 的 32 行 logits）：

| 比较 | 非有限值 | argmax 相同 | max\|diff\| | 第一处不同的行 |
|---|---:|---:|---:|---:|
| M1 vs M3（peer 开 vs 它自己）| 0 | **32/32** | **0** | 无 |
| M2 vs M4（peer 关 vs 它自己）| 0 | **32/32** | **0** | 无 |
| M1 vs M2（peer 开 vs peer 关）| 0 | **9/32** | **34.4717** | **第 0 行** |

**⇒ 两条路各自都是确定的（两次跑出来的 logits 逐字节相同，max|diff| = 0）；
两条路之间的数字差得很大（32 行里只有 9 行的 argmax 相同，第一行就不同）。**
这就是「prompt 的专家由 helper 算」的结果：**KV 从第一行起就不是同一个**。
生成的文字因此从第 10 个 token 起分叉（peer 开 `38663 25 2972 198 256 471 …`，
peer 关 `6065 220 17 25 498 760 220 18 15 53066 24455 75654 198 256 471 …`）。

**（四）新旧二进制逐字节相同**：同一组配置下，旧二进制（`bin/strata-q36.bps`）和新二进制
（`bin/strata-q36.test`）的文字**逐字节相同**（peer 开与 peer 关各一对）⇒
§5 的那个 token-id 检查不改任何一个比特。

## 4. 引擎里那句警告是旧的

`q36_engine.cu:4738`（启动时每个走 host 路由的 helper 印一行）：

> `... experimental: a decode step after the prompt is not safe yet`

这句话是 **10-22 移植 peer 路径时写的**（`c36e6bb9`），在 10-23 的步长修复和 10-24 的编码修复之后
**没有跟着改**。`peer_host_route()` 的注释（`q36_engine.cu:233-248`）也还写着
「the route does not survive a decode step yet」和 10-21 的那些测量数字。
**今天实测：不再崩。** ⇒ 这两处文字要改成现在的实测结论。

## 5. 还没关的那扇门：哨兵 token id 仍然能当行号用（**已关**）

`k_argmax`（`q36_engine.cu:904`）：`float v = -INFINITY; int id = 0x7fffffff;`
⇒ **一行 logits 全是 NaN（或全是 -inf）时，它返回 0x7fffffff**。
这个数接着被当成 token id 去查嵌入表（`iq_embed_rows`，`q36_engine.cu:3773`）：
`0x7fffffff × 行字节` → 越界 → IMA。

**今天 peer 路径的 NaN 来源已经修掉，所以这条链走不通了；但链本身还在引擎里**：
任何一处产生 NaN 的 logits（以后的 bug、CPU 池的路径、别的量化组合）都会重演同一个 IMA。
（`plans/2026-10-21-q36-peer-ima-root-cause.md` §0 第 5 条写的是同一件事。）

**已关（提交 `a8605d5f`）**：每个把 token id 交给设备之前的地方都查一遍范围 ——
decode 步的 token、prompt chunk 的 token、批行的 token、MTP 行的 token、MTP 块自己的 token、
草稿步的 token、verify window 的 token。**不在词表里就打印一行并结束运行**
（`strata-q36: <哪里>: the token id N is outside the vocabulary (it holds M ids): the logits row
is not finite; the run ends here`）。**mapped chain 的设备侧交接（`k_chain_publish`）把哨兵夹成 0**，
因为那里下一步是在设备上嵌入它的，host 还来不及看。
（**那条 mapped chain 在这台机器上是死的**：`mapped_chain_live()` 在 P100+3×P104 上恒假，
`k_chain_publish` 走不到 —— 这个夹住是给别的机器上的那条路准备的。）

### 5.1 新二进制的验证（MEASURED，2026-10-25，192.168.1.98）

`bin/strata-q36.test`（md5 `724892736631a3d9300900f079be8892`，从提交后的源码编译，BUILD_RC=0）。

| 趟 | 内容 | 结果 |
|---|---|---|
| V1 | 新二进制，默认路径，parity 门（`--tokens 32 --n 1`）| `gen: 220` |
| V2 | 对照二进制（`bin/strata-q36.bps`），同一道门 | `gen: 220` ⇒ **与新二进制逐字节相同** |
| V3 | 新二进制 + peer 旗标，同一道门 | `gen: 220` ⇒ **旗标不改 32 token 的文字** |
| V4 | bench（默认路径）| pp512 **156.4** / tg128 **58.2**（改动前 156.1 / 58.5）|
| V5 | bench（peer 路径）| pp512 **495.3** / tg128 **59.3**（改动前 493.6 / 59.4）|
| V6 | 崩溃配置 pp256+tg128（peer 路径）| rc=0、IMA=0、nan=0 |
| V7 | MTP 路径（`--mtp 3`，草稿步读那些 id）| rc=0、IMA=0、nan=0 |

**⇒ 数字不动、文字逐字节相同、崩溃配置干净。** 每一趟 rc=0、IMA=0、nan=0。
**§3.1（四）**：同一组配置下新旧二进制的文字也逐字节相同 ⇒ 这个检查不改任何比特。

### 5.2 记录里那条崩溃命令本身（MEASURED，2026-10-25）

`plans/2026-10-21-q36-peer-ima-root-cause.md` §1.1 的原命令（peer + host 路由，`--bench-pp 160 --bench-tg 8`）：

| 趟 | 加了什么 | rc | IMA | nan | pp160 | tg8 |
|---|---|---:|---:|---:|---:|---:|
| D1 | 原命令 | 0 | 0 | 0 | 320.9 | 52.7 |
| D2 | + `STRATA_Q36_PROMPT_TIMING=1` | 0 | 0 | 0 | 319.6 | 52.2 |
| D3 | + `CUDA_LAUNCH_BLOCKING=1`（记录里用来点名那个核的那一种）| 0 | 0 | 0 | 161.9 | 27.2 |
| D4 | 纯 prompt（`--bench-tg 0`，记录里干净的那一种）| 0 | 0 | 0 | 403.3 | — |

**⇒ 连 `CUDA_LAUNCH_BLOCKING=1` 那个点名核的跑法也不崩了。**

## 6. 结论与要做的

| # | 结论 | 状态 |
|---|---|---|
| 1 | 缺陷 = helper 缓存的 slot 间距 ≠ 主卡的步长 → NaN → 哨兵 token id → 越界读 | 已修（10-23），今天不复现（§2）|
| 2 | 第二个缺陷 = peer 归属表的编码（`53e05a83`，主卡越界写 Xid 13）| 已修（10-24）|
| 3 | 引擎里「decode after the prompt is not safe yet」那句话 | **已改**（`8b844007`，§4）|
| 4 | 剩下的不同是**数字**（prompt 的专家由 helper 算），不是崩溃 | 仓库里早就写明（§3）；**要写进用户侧的文档**（还没写）|
| 5 | 哨兵 token id → 越界读这条路还在 | **已关**（`a8605d5f`，§5）|

## 7. 走查被推翻的假设（省得以后重走）

1. **「`wait()` 发出的 H2D 与下一层 `launch()` 复用 pinned `h_out` 之间没有流序保证」**（10-20 的旧结论）：
   **不成立。** `q36_engine.cu:3540` 每个 MMQ 层都有一次 `cudaStreamSynchronize(st)`（route D2H 之后），
   它在 `peer_plan_or_exit()`（:3557）之前 ⇒ 上一层 `wait()` 发出的 H2D 已经落地，
   下一层 `launch()` 写 `h_out` 之前队列是空的。`pending_n` 只清计数不是漏洞。
2. **「host route 的 H2D 被烤进 decode 的 CUDA graph」**：不成立。graph 只 capture decode 的
   `forward(1, ...)`（`q36_engine.cu:5049`），peer 的 H2D 在 prompt 的路径上，不在 capture 里；
   而且 capture 前的 pending work 按序在 graph 之前执行。
3. **「T ≤ hybrid_t 的 prompt 从没覆盖 peer 的算术，所以那些『干净』不算数」**（10-20 的说法）：
   事实没错（`peer_layer()` 要 T > 128），但它只说明**旧的小 prompt 测不到 peer**，
   不说明它会崩。今天 C3 用 pp256 直接覆盖了那条路径。
