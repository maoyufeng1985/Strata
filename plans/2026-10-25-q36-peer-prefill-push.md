# q36：把 peer 路径的 pp512 往上推（2026-10-25，12 小时无人值守）

目标（用户）：以 **pp512 = 493.9** 为基线继续往上推；**推不动了就转 tg64（59.1）**。

## 0. 基线（MEASURED，2026-10-25，192.168.1.98）

机器：Tesla P100-12GB 主卡 + 3× P104-100 helper。模型：Qwen3.6-35B-A3B-UD-Q4_K_XL。CUDA。
二进制：`bin/strata-q36.flag1`（= HEAD `8ed0c2f3`）。命令：`--bench-pp 512 --bench-tg 64 --no-warmup
--max-context 2048 --expert-cache-device1/2/3 auto`。旗标：`STRATA_REMOTE_ZEROCOPY=0
STRATA_Q36_HELPER_ASYNC=1 STRATA_Q36_HELPER_PARALLEL=1 STRATA_Q8_SM60=1
STRATA_Q36_HELPER_PARALLEL_MARSHAL=1 STRATA_Q36_FIXED_SLOTS=1` + `STRATA_Q36_PREFILL_PEERS=1
STRATA_Q36_PREFILL_PEERS_HOST=1 STRATA_Q36_PEER_F16=1`。

| 跑 | peer | pp512 bench 0/1/2 | tg64 bench 2 |
|---|---|---|---|
| F3（旗标 = 1）| 开 | 471.7 / 491.4 / **493.9** | **59.1** |
| F1（旗标 = 0）| 关 | 156.0 / 156.1 / 156.1 | 58.0 |
| **q8_1 window（2026-10-26，§3.4）** | 开 | 555.3 / 585.1 / **589.6** | 58.9 |
| 同一个二进制的对照（HEAD，fp32 window）| 开 | 471.5 / 490.6 / 492.9 | 59.3 |

**噪声**：同一趟里 bench 0/1/2 差 2-4%（471.7 → 493.9），**bench 2 是稳态**。
⇒ 每个候选都跑 3 趟取 bench 2 的中位数，A/B 用**同一个二进制的两趟**比。

## 1. 纪律

- 一次只动一个东西；每个候选都要有**同条件的对照**。
- 旗标默认关；关掉时默认路径**逐字节不变**（parity 门 `gen: 220`）。
- 箱子上**一次只跑一个引擎**；每趟前后看 dmesg 的 Xid 计数。
- 只用 CUDA（这台机器没有 ROCm）；改到 .cu/.cuh 就只编 CUDA，并写明 HIP 没编。

## 2. profile：peer 路径的 prompt 时间去哪了（MEASURED，2026-10-25）

`STRATA_Q36_PROMPT_TIMING=1`（引擎自带的探针）的 **bench 2**，单位 ms/层（40 层）：

| GPU 段 | ms/层 | 占 25.78 的 |
|---|---:|---:|
| mixer（GDN/attention 的投影、conv、SSM）| 6.771 | 26% |
| router_route | 0.240 | 1% |
| route_d2h | 0.023 | 0% |
| **plan（主机侧的 plan 与等待）** | **6.382** | **25%** |
| primary_work（主卡自己那份 MMQ）| 4.662 | 18% |
| **helper_wait（等 helper 的行落地）** | **7.505** | **29%** |
| combine | 0.192 | 1% |
| **forward 合计** | **25.777** | |

plan 的主机墙 = bookkeeping 0.740 + prepare 0.175 + **wait 5.454**；
wait 分摊到三个 helper：**CUDA1 5.083、CUDA2 0.010、CUDA3 0.101**（等的是最慢的那个）。

**⇒ 与 helper 有关的 = plan 6.38 + helper_wait 7.51 = 13.9 ms/层（54%），
主卡自己的活 = mixer 6.77 + primary_work 4.66 = 11.4 ms/层（46%）。**
**helper 那条链是临界路径，主卡在等。**

三个候选的出处：
1. **分配不均**：如果三个 helper 的 entry 数不是各 1/3，最慢的那个定整层的临界路径。
   （探针新加的一行「the entries per layer」量的就是这个。）
2. **主卡闲着**：主卡还有 4.78 GiB 空闲，却只留了每层 39/256 个专家（15% 的 entry）。
   把专家从 helper 挪回主卡 ⇒ helper 的活变少、临界路径变短（helper 少拿 ⇒ 每层更少）。
   **不用改代码**：`--expert-cache-device1/2/3 N`（N = 该卡装多少专家）就是那个旋钮。
3. **回程的行**：每个 entry 2048 个 fp16 值 = 4 KB；512 token 的 prompt 每层 4096 个 entry
   ⇒ 16.8 MB/层，下行 + 上行 = 671 MB + 671 MB。按本机 ~6 GB/s ≈ 0.22 s（prompt 总共 1.04 s）⇒ 21%。

## 3. 候选（按「先量后改」排）

### 3.1 已量：把活挪回主卡（`--expert-cache-device1/2/3 N`）= **否**

主卡闲着看着像是个机会。量了（每个 N 一趟，bench 2，旗标同上，探针开着）：

| 每个 helper 装 N 个专家 | pp512 | tg64 | 主卡的 entry/层 | helper 的 entry/层（三个合计）|
|---:|---:|---:|---:|---:|
| **auto（2880-2920）** | **490.5** | **59.3** | 640 | 3456 |
| 2700 | 481.7 | 49.4 | 785 | 3311 |
| 2500 | 404.8 | 41.7 | 1018 | 3078 |
| 2300 | 341.1 | 35.8 | 1281 | 2815 |
| 2100 | 293.1 | 30.9 | 1559 | 2537 |

**⇒ 越往主卡挪越差，pp 和 tg 都差。** 原因：主卡自己的 MMQ（int8，从 K-quant 现重量化）**每个
entry 比 helper 的 native q8_1 路径更贵**（主卡 7.3 µs/entry，helper 约 4.9 µs/entry），
而且主卡多拿的专家是 helper 装不下时**剩下的高 id 专家**（路由对它们更热）⇒ 主卡的活涨得比线性快。

**⇒ 反方向（helper 多拿）也是死的**：helper 的卡已经装满（auto 装到「再放一个专家就不到 512 MiB
空闲」为止；每张 P104 装 5.14-5.22 GiB 的专家，跑起来后 8 GB 的卡只剩 2.6 GiB，而那 2.6 GiB 里
还要放 executor 的行缓冲（约 0.4 GiB）和 512 MiB 的预留）。**auto 就是上限。**

**⇒ 分配不均也不是问题**：auto 下三个 helper 的 entry/层是 1069.8 / 1181.4 / 1205.0（26% / 29% / 29%），
**基本是平的**。探针里那三个 wait 数（CUDA1 5.08、CUDA2 0.01、CUDA3 0.10）只是**等的是最慢的那个**，
不是各自的工作量。

### 3.2 已做：plan 拆两半（让 GPU 在等待里干活）= **数字没动**

GPU 在 plan 的主机等待里闲着 6.5 ms/层（25%）看着像纯浪费。**把 plan 拆成两半**
（`peer_plan()` 记账 + 入队，`peer_plan_wait()` 收数 + 发 helper 的活），中间插进主卡自己的 MMQ
（`peer_plan_wait_or_exit()` 挪到 shared expert 之后），让 GPU 在 host 等的时候有活干。

| | plan | primary_work | helper_wait | forward | pp512 |
|---|---:|---:|---:|---:|---:|
| 拆之前 | 6.382 | 4.662 | 7.505 | 25.777 | 490.5 |
| 拆之后 | 1.058 | 5.555 | 11.980 | 25.815 | **494.5** |

**⇒ 每层的时间一点没变（25.8 ms），pp512 也没变（490.5 → 494.5，在噪声里）。**
原因：helper 的**算术**本来就和主卡的 MMQ 重叠着（旧顺序里 helper 的 launch 在 plan 末尾，
主卡的 MMQ 在它后面发出，两者并行）。把 MMQ 提前，只是把 GPU 的空闲挪了个地方；
**helper 的链条（copy+plan 5.03 + 算术与回程 12.25 = 17.3 ms/层）才是临界路径**，
而主卡在等它。

**⇒ 后来的结论：这个拆分撤掉了**（见 §3.6：q8 窗口把 plan 的主机墙压到 0.5 ms/层，
拆分要藏的那 6.5 ms 已经不在了）。

⇒ **下一步只能缩短 helper 那条链**（它的算术、回程的字节、或者 window 的拷贝），
或者让主卡的 mixer（6.77 ms/层）和 helper 的链条**并行**（现在 mixer 在前，helper 的 copy 依赖它的输出 ⇒
串行 ⇒ 这 6.77 ms 是纯串行段）。

### 3.3 关键发现：**helper 的 PCIe 是 x1，只有 0.81 GB/s**

在箱子上量了 pinned host→device 的带宽（`.run/pcbw.cu`）：

| 卡 | 4 MiB | 64 MiB |
|---|---:|---:|
| 主卡 Tesla P100-12GB | **6.03 GB/s** | 6.06 GB/s |
| helper NVIDIA P104-100 (1) | **0.81 GB/s** | 0.81 GB/s |
| helper NVIDIA P104-100 (2) | 0.81 GB/s | 0.81 GB/s |
| helper NVIDIA P104-100 (3) | 0.79 GB/s | 0.79 GB/s |

**⇒ helper 的链路比主卡慢 7.5 倍，而且每个 prompt 在这条慢链上搬两次大东西：**
- **上去**：window（激活）T × 2048 × 4 字节 = 512 token 时 **4 MiB** ⇒ 4.9 ms/层
  （探针的 prep_sync = 5.031 ms/层 ✓ 对上了；pp160 的 prep_sync = 1.391 ms ✓ 正好按 window 大小缩放 ✓）
- **回来**：每层 1152 个 entry × 2048 × 2 字节（fp16）= **4.7 MiB** ⇒ 5.8 ms/层

两条合计 **≈ 10.7 ms/层 = 每层 25.8 ms 的 41%**。剩下的：mixer 6.77（主卡自己的）、
helper 的算术 ≈ 6.6、plan 的记账 1.7。

⇒ **要提速度，只能少搬字节**（helper 的算术是它们自己在算，主卡帮不上）。

### 3.4 已做：window 用 q8_1 上传（**无损**）—— 第一版有三个错，改完才算数

主卡先把 window 量化成 q8_1（**helper 的 product 读的就是这个格式**），再上传：
4 MiB → **1.11 MiB**（每行 2304 字节：128 个值一块 + 比例尺，16 字节对齐）。
helper 改成**按计划顺序 gather 那些行**，不再自己量化 ⇒ **字节与它自己量化出来的完全一样**
（同一个函数、同一批行值）⇒ **peer 路径的数字一个比特都不动**。

**第一版（2026-10-26 00:36 的二进制）有三个错，验收门把它抓出来了**（C2 失败：peer 两轮在
token 7 / 32 上不同；S5：6 个 prompt 只有 2 个与上一版逐字节相同）：

1. **布局错了（主因）。** MMQ 的激活**不是按行连续的**：它的布局是**分带的**——一个带
   （128 个值 × 每一行）的块是连续的，**行在带里面**：`quantize_mmq_q8_1_cuda` 写的是
   `y[b*nrows + r]`，MMQ 的核读的也是 `y + ncols_y*(带)*sz`（third_party/llama.cpp 的
   ggml/src/ggml-cuda/quantize.cu:534 与 mmq.cuh:911）。
   ⇒ 一行的字节**不是一段连续的内存**，gather 必须**按块搬**（块 = 144 字节 = 9 个 16 字节片），
   而且源的行数（window 的 token 数）和目标的行数（helper 自己的行数）不同 ⇒ 带内的间距也不同。
2. **拷贝没有等量化。** 量化走引擎自己的 stream（`cudaStreamNonBlocking`），而 staging 的
   `cudaMemcpy` 走默认 stream ⇒ **默认 stream 不等 non-blocking stream** ⇒ 拷贝可能读到
   还没写完的 window。⇒ 拷贝改成 `cudaMemcpyAsync(..., st)` + `cudaStreamSynchronize(st)`。
3. **设备 buffer 小了 18 KiB。** `MMQ::quantize` 写的是 `q8_bytes(T, N)`（行 + 整块 buffer
   尾巴的 128 块 pad），而 buffer 只按 `tmax * q8_row_bytes` 分配 ⇒ 最后 128 块落在分配之外。
   ⇒ 按 `MMQ::q8_bytes(tmax, N)` 分配。

另外，**验收门自己也有一个错**：门里 `Eng` 从进程自己的环境变量起步，所以被 export 的
peer 旗标会漏进「默认路径」那一路 ⇒ 「默认」其实跑的是 peer 路径（C1 失败、两边速度一样）。
⇒ 门现在先把四个 peer 旗标从继承的环境里清掉，peer 那一路只从 `--peer-env` 拿。

**代价**：主卡多一次量化（512 行 × 2048 值 ≈ 0.1 ms/层），省下 3.9 MiB/层的上行。

**验收（2026-10-26，同一台机器，两个二进制的对照）**：

| | HEAD（fp32 window） | q8_1 window |
|---|---:|---:|
| pp512 bench 2 | 492.9 tok/s | **589.6**（+19.6%）|
| tg64 bench 2 | 59.3 tok/s | 58.9（噪声里）|
| 前向 ms/层 | 25.903 | **21.643** |
| 里的主机墙 ms/层 | 6.578 | **2.305** |
| prep_sync ms/层 | 5.128 | **1.397** |
| plan 的记账 ms/层 | 0.741 | 0.266 |
| mixer / primary_work / helper_wait | 6.758 / 4.662 / 7.449 | 6.763 / 4.661 / 7.454（都不动）|

⇒ 省下的 4.26 ms/层 全在上行那一条上（prep_sync -3.73 + 记账 -0.475），
helper 那条链（helper_wait 7.45）和主卡自己的活（mixer 6.76 + primary_work 4.66）一动不动。

**token 逐字节相同**：两个二进制各跑一遍验收门（6 个 256-token 的 prompt，每个 64 个贪心 token），
**peer 路径 6/6 逐字节相同，默认路径 6/6 逐字节相同** ⇒ 这个改动**不改变任何一个数字**，
只是少搬了字节。

（要动的地方：engine 侧的 `d_peer_q8` + pinned staging、executor 侧的 `d_x8` + 一个 gather 核
（`k_gather_q8`，按块搬）、`prepare_begin` 多一个 `q8_window` 标志、MMQ 头里加
`q8_row_bytes()` / `q8_block_bytes()` / `q8_row_blocks()`——`q8_bytes()` 带一个整块 buffer 的
尾巴 pad，逐行的大小得用两行相减才准，而带内的行间距是 `nrows` 个块。）

**已知的天花板**：P2P 那条路（helper 从主卡的显存直接拷 fp32）上这次量化是**白做的**
（主卡多 0.1 ms/层），因为那条路上 window 本来就不经主机。要省掉它，量化得等 plan 算出
「有没有 helper 走主机路」再决定——那是 plan 里已有的判断，代价是把它提到量化之前。
本机没有 P2P（十二对都答 0），所以这里量不到差别，先记着。

### 3.6 拆 plan 的那两半：**撤掉了**

拆开之后量到的是**中性**（25.815 vs 25.777 ms/层，pp512 490.5 → 494.5，在噪声里），
原因是 helper 的链条（拷贝 + 它的算术 + 回程）本来就是临界路径，主卡的 MMQ 本来就和它重叠。
**q8 窗口把 plan 的主机墙从 6.5 ms 压到 2.3 ms/层以后，拆分的理由（藏住 6.5 ms 的主机等待）
就没有了** ⇒ 撤掉，diff 里只留 q8 窗口本身。

### 3.5 其它候选（量过之后剩下的）

- **回程的行改 int8**：**比原估的小很多**。4.7 MiB 是三条链的**总和**，而三条链是并行的
  （三条 PCIe 是不同的 PHB 路径）⇒ 临界路径是每条约 1.5 MiB ⇒ 1.9 ms/层。
  改 int8 只省 0.95 ms/层 ⇒ **≈ +4.6%**，而且**有损**（peer 路径的文字本来就跟默认路径不同，
  这会让差距更大）⇒ **不做**。
- **plan 的记账**：q8 窗口之后只剩 0.266 ms/层（原来 0.741）⇒ 全搬走也只有 +1.2%，
  而且那个核对是防「表与缓存不一致」的护栏 ⇒ **不做**。
- **剩下的 21.6 ms/层在哪儿**：mixer 6.76（31%，主卡自己的 GDN/attention）、
  helper_wait 7.45（34%，三条链并行的临界路径）、primary_work 4.66（22%，主卡自己的 MMQ）
  ⇒ **一半以上是主卡自己的活**，peer 路径能动的部分基本动完了。
  ⇒ **转解码（tg64）**，见 §4。

### 3.7 prompt 还能动的最后一块：主卡自己那 39 个专家（**没做，先记着**）

- 主卡每层留了 39/256 个专家在 VRAM 里（2.86 GiB），它用 MMQ 的 K-quant 路径算：
  **7.3 µs/entry**；helper 走 native 路径是 **4.9 µs/entry** ⇒ 主卡那 640 个 entry/层
  要 4.66 ms，换条路大约 3.1 ms ⇒ **-1.5 ms/层 ⇒ ≈ +7.5%**。
- 引擎里已经有更快的 K-quant 核：`STRATA_PF_F16_CUDA=1` 的 fused FP16 路径
  （docs/QWEN36.md 的那一节：核本身快了 46%，232.9 → 125.4 µs/专家的 64 行 tile），
  而且**它在混合放置下没显示出来是因为 PCIe 搬专家盖住了它**——peer 路径的主卡专家在 VRAM 里，
  核的快慢是露出来的 ⇒ ⇒ 值得试。
- **为什么不在这轮做**：① 它现在对 served layer 是关的（`!peer_here`），
  而 peer 路径的行号和那条路的不同（peer 的 lid 要重排）⇒ 要改的是行契约这一层；
  ② 换了算术 ⇒ **peer 路径的数字会变**（q8 窗口的「逐字节相同」那条保证只对窗口的格式成立，
  对算术不成立）⇒ 文档里那句「peer 的文字不是默认路径的文字」要跟着改；
  ③ 一轮改不动 ⇒ 单开一次。

## 4. 下一步：解码（tg64 58.9）

### 4.1 量到的（MEASURED，2026-10-26，q8 window 的二进制 `bin/strata-q36.q8fix`）

`.run/R-decode2-20261026.sh`，三张 helper 与旗标同前，另加 `STRATA_Q36_HELPER_TIMING=1`：

| | tg64 | pp512 |
|---|---:|---:|
| D1 三张 helper | **58.8** | 589.0 |
| D2 一张 helper | 20.1 | 303.9 |
| D3 没有 helper（主卡自己） | 15.3 | 156.4 |

⇒ **helper 让解码快 3.8 倍**（没有 helper 时主卡要从主机内存搬专家：每层 192 MiB）。

**每一层 447 µs（17.02 ms/token，40 层）的去向**（D1 bench 2 的 GPU 时间线，µs/层）：

| 段 | µs/层 | 在谁身上 |
|---|---:|---|
| attn_ssm（mixer 的 dense GEMV）| **204** | 主卡 GPU（MMQ 的 Q8_0 解码核）|
| router | 20 | 主卡 GPU |
| helper_dispatch | 64 | 主卡等路由的 D2H + 把活交给 worker |
| local_grouped | 2 | 主卡 VRAM 里的专家 |
| local_shared | 32 | shared expert（MMQ）|
| cpu_pool | 109 | **GPU 空着**（主机在等 helper 的行）|
| helper_merge | 15 | 把 helper 的行并进 eparts |

⇒ 主卡自己真的在算的 273 µs/层（61%），**helper 的一个来回 ≈ 173 µs/层（39%）**。
每个 helper 的 host 侧：ms_begin 45 µs/层、ms_wait 37、ms_sync_overhead 3.3。

### 4.2 剩下的候选，和为什么先做不了

| 候选 | 预计 | 在哪个文件 | 能做吗 |
|---|---:|---|---|
| mixer 的 204 µs/层（MMQ 的 Q8_0 解码 GEMV）| 大 | 第三方 llama.cpp + 共享的 mmq | **不能**：那是 fork 出来的依赖 |
| helper 的一个来回 173 µs/层 | 大 | `src/core/remote_experts.cpp`（CUDA+HIP 共用）| **不能**：本机没有 ROCm，编不了 HIP |
| shared expert 的 32 µs/层 | ~10 µs | 共享的 mmq | 同上 |
| S2 `launch_check` 的 `cudaGetLastError`（每层 11.5 次 × 1.31 µs）| 15 µs/层（+4%）| `native_mmvq.cu`（HIP 也编）| 同上 |
| S3 `DeviceScope` 的换设备（drain 里 get + 2×set）| 9.5 µs/层（+2%）| `remote_experts.cpp` | 同上 |

⇒ **解码剩下的力气都在共享的核心文件里，而本机编不了 HIP** ⇒ AGENTS.md 的
「改了哪个后端就编哪个」这一条过不去 ⇒ 先记着，不动。

### 4.3 不改代码能拿的（旗标，MEASURED 2026-10-26，同一台机器，同一个二进制）

`.run/R-decode-flags-20261026.sh`，bench 2：

| | tg64 | pp512 |
|---|---:|---:|
| E1 现在的旗标 | 59.3 | 589.7 |
| E2 `--kv int8` | **61.5**（+3.7%）| 590.3（不变）|
| E3 `--kv int8 --mtp 2` | 59.6 | 521.8（-11%）|
| E4 `--kv int8 --mtp 3` | 61.2 | 527.7（-11%）|

⇒ **`--kv int8` 是白拿的 3.7%**（KV 缓存的 int8：每 kKVG 个值一个 FP16 比例尺 ⇒ 字节减半）。
⇒ **`--mtp` 在这个 bench 里没有**：2026-10-25 记的「D=2 是 81.5 tok/s」是另一套量法
（serve 路径的吞吐，不是这个 bench 的 tg64）⇒ 以这里为准：**MTP 的草稿没有变成墙钟**，
而它的 MTP 块占显存 ⇒ prompt 掉 11% ⇒ **不用**。

⇒ 剩下的路：要么在有 ROCm 的机器上编 HIP 之后动那两个共享文件（+6%），
要么动第三方 llama.cpp 的解码核（大，但要维护一个 fork）⇒ 两件都不是这台机器能定的事。

## 产出

- 每个候选一条记录（数字 + 条件 + 结论），落在本文件。
- 赢的改动一个旗标（默认关），默认路径不变。
- 最后的数字进 `docs/QWEN36.md`。
