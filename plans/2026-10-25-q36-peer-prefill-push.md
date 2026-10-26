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

## 3.8 回程那条链：先量，再动（2026-10-26 夜，MEASURED）

§3.5 里我把回程估成「三条链的总和 4.7 MiB ⇒ 每条 1.5 MiB ⇒ 1.9 ms/层 ⇒ int8 只省 0.95 ms（+4.6%）」。
**这个估法是错的**：4.7 MiB 是**每个 helper 每层**的量（1152 个 entry × 2048 × 2 B），不是三条链的总和。

**量法（不用改代码，一个旗标）**：同一个二进制，行的传输格式 fp16 / fp32 各跑一趟（fp32 的下行字节翻倍）。

| | pp512 | 前向 ms/层 | helper_wait | primary_work |
|---|---:|---:|---:|---:|
| D1 行走 fp16（现在的旗标）| **590.7** | **21.603** | 7.538 | 4.663 |
| D2 行走 fp32（去掉 `STRATA_Q36_PEER_F16`）| 418.2 | 30.51 | | |

⇒ 多搬的 4.72 MB/层（每个 helper）= **8.91 ms/层**；其中主卡那侧多一次 H2D（4.72 MB ÷ 6.03 GB/s = 0.78 ms）
⇒ **helper 的 D2H = 8.13 ms/层 ⇒ 4.72 MB ÷ 8.13 ms = 0.58 GB/s**（比它的 H2D 的 0.81 还慢）。

**⇒ helper 那条链 = 上行 1.40 + 算术 ≈2.7 + 下行 8.13 = 12.2 ms**（helper_wait 7.538 + primary_work 4.663 ✓ 对上）。
**⇒ 下行是整层最大的一块：8.13 ms / 21.6 ms = 38%。** 算术只占 2.7 ms ⇒ **helper 在这 2.7 ms 里是闲着的，
GPU 在算的时候拷贝引擎空转**（两条路都在同一条 helper 的 stream 上 ⇒ 串行）。

**候选（按「字节 / 时间」排）**：

| 做法 | 下行字节 | 下行 ms | 前向 ms/层 | pp512 | 备注 |
|---|---:|---:|---:|---:|---|
| 现在（fp16）| 4.72 MB | 8.13 | 21.60 | 590.7 | |
| **int8 下行（每 128 值一个比例尺）**| 2.4 MB | 4.2 | 17.3 | **738.3（+25.7%，实测）** | 有损 |
| 每 token 的加权和（少 2.25 倍的行）| 2.1 MB（fp16）/ 1.1 MB（int8）| 3.6 / 1.9 | 17.1 / 14.8 | ≈746 / ≈860 | 无损，但要动 combine |
| 下行与算术重叠（分块 + 映射表）| 4.72 MB | 与算术重叠 | 18.9 | ≈677（+14%）| 无损，逐字节相同 |

⇒ 三条路可以叠：int8 + 重叠 ⇒ 5.6 ms 的链 ⇒ 前向 14.9 ⇒ **≈856（+45%）**。
⇒ 但「重叠」要改行契约（分块 ⇒ 行按专家的顺序 ⇒ 主卡要知道每行属于哪个 entry），
⇒ 而「每 token 的加权和」不能与「重叠」叠（token 的加权和要等所有专家算完）。

### 3.9 回程走 int8（做了，`STRATA_Q36_PEER_I8=1`）

- 每行 2048 个值 → **2048 字节 + 16 个 fp32 比例尺**（每 128 个值一个）= 2112 字节
  ⇒ 每 helper 每层 **2.4 MB**（fp16 的 4.72 MB 的一半）。
- 量化在 helper 上做（**一个 warp 一行**：warp 内归约出块内最大值 ⇒ 比例尺 ⇒ 量化），
  解量化在主卡上做（同样一个 warp 一行）⇒ **`eparts` 里仍然是 float**，combine 读的东西一个字都没变。
- **行契约一个字都没变**：还是「每个 entry 一行，按 rank 排」⇒ 只有线上元素的宽度变了。
- **有损**：块内的量化误差 ≤ 块内最大幅值 / 254。引擎把这个数**打印出来**
  （`the int8 down leg packed rows up to X in magnitude and the largest error was Y of the block's own magnitude`）
  ⇒ 代价是数字，不是承诺。

**量到的（MEASURED，2026-10-26，同一个二进制 `bin/strata-q36.i8`，旗标是唯一的差别）**：

| | pp512（bench 2）| tg64 | 前向 ms/层 | helper_wait | plan |
|---|---:|---:|---:|---:|---:|
| I1 行走 fp16（对照）| 587.6 | 59.3 | 21.716 | 7.591 | 2.234 |
| **I2 行走 int8**| **738.3**（+25.7%）| 59.2 | **17.269** | **3.159** | 2.215 |

⇒ 下行那一段省了 **4.43 ms/层**（helper_wait 7.591 → 3.159），整层省 4.45 ms ⇒ **+25.7%**。
⇒ 打包自己量到的代价：**行的最大幅值 18.6674，最大误差是块内幅值的 0.00394（0.39%）**。
⇒ 无 IMA、Xid 138→138。

**验收（2026-10-26，同一台机器）**：

| | 结果 |
|---|---|
| 默认路径的 parity（`gen: 220`）| **220**（不变 ⇒ 旗标关着时默认路径逐字节不变）|
| 门（`q36_peer_gate.py`，6 个 256-token 的 prompt，peer 跑两轮 + 默认跑一轮）| **C1..C5 全过**；两轮的 token 相同；peer 632.1 tok/s vs 默认 78.2（8.09x）|
| peer 的 token：int8 vs fp16（同一批 prompt）| **6 个里 5 个逐字节相同**，p01 在第 4 个 token 上不同（25 个里的第 4 个）|
| Xid | 138 → 138；无 illegal memory access |

### 3.10 下一轮的设计：「每 token 的加权和」（无损，写在纸上了）

下行搬的是**每个 entry 一行**（token × 路由槽），但主卡要的是**每个 token 一个加权和**。
⇒ helper 可以在自己那边把同一 token 的几个 entry 按路由权重加起来，回程就只搬 token 的个数 ⇒ 字节少 2.25 倍
（1152 行 → 512 行），而且**传输量不再随路由的扇出（top_k）涨**，只随 prompt 的 token 数涨。

**设计（还没做）**：

1. **helper 侧**：把路由权重传上去（每 entry 一个 fp32，16 KB/层；引擎的 route D2H 顺手多搬一份就行），
   算完之后按 token 归并（一个 token 一个线程，遍历它在 helper 这边的 entry 的行，乘上权重累加）
   ⇒ 512 行 × 2048 ⇒ 和 int8 那条一样打包回程（1.08 MB/层 ⇒ 下行 1.9 ms）。
2. **主卡侧**：partial 和**不进 `eparts`**，进一个单独的 `[3][T][N]` 的缓冲。
   combine（`k_moe_out`，引擎自己的核）现在是「每个 token 的 K 个槽，各自有行和权重」⇒ 改成：
   helper 拿走的那些槽**权重置 0**（引擎已经有每 entry 的归属数组 ⇒ 一个核的事），
   combine 在 K 个槽之外**再加上 3 个 partial 和**。
   ⇒ `eparts` 的行数和 combine 的行契约都不用动（helper 的那些行只是不再被写、也不再被读）。
3. **代价**：算术的加法顺序变了 ⇒ **peer 的数字会动**（和 int8 那条一样，不是逐字节相同）；
   行契约要重写（「每个 entry 一行」变成「helper 给每个 token 一个加权和」）。
4. **和「重叠」不叠**：token 的加权和要等所有专家算完 ⇒ 回程无法与算术重叠。
   ⇒ 只能选一条：这条（下行 4.2 → 1.9 ms ⇒ 整层 17.3 → 14.8 ⇒ **≈860 tok/s**）
   或重叠（要映射表，+15%）。

### 3.11 做了：每 token 的加权和（`STRATA_Q36_PEER_SUM=1`）

按 §3.10 的设计做的，三处改动：

1. **helper 侧**：路由权重随 window 一起上去（每 entry 一个 fp32，16 KB/层；引擎的 route D2H 顺手多搬一份），
   helper 算完之后按 token 归并（`k_peer_toksum`：一个 token 一个 warp，遍历它在 helper 这边的行，乘权重累加）
   ⇒ 回程从 1152 行变成 512 行（int8 打包 ⇒ 1.08 MB/层）。
2. **主卡侧**：和进一个单独的 `[3][tmax][N]` 的缓冲（不进 `eparts`）；
   **helper 拿走的那些槽的权重置 0**（`k_peer_zero_wts`：引擎有每 entry 的归属数组 ⇒ 一个核的事），
   combine（`k_moe_out`）在 K 个槽之外**再加上那几份和**。
   ⇒ `eparts` 的行数和 combine 的行契约都不用动：helper 那些行**不再被写**。
   **但它们仍被读**（combine 的行映射给每个 entry 一个行）⇒ 权重是 0，而 **0 × NaN = NaN**
   ⇒ plan 之后把那一片行 `memset` 成 0（512 行 × 8 KB = 4 MB，几微秒）。
3. **代价**：加法的**顺序**变了（helper 按自己的行顺序加，主卡按 pick 顺序加）⇒ 数字差在加法的次序上，
   不差在传输的精度上 ⇒ **peer 的文字会动**（和 int8 那条一样）。

**门抓到了一个错（第一版，`STRATA_Q36_PEER_SUM=1`）**：第一版的加权和核把**两个行空间搞混了**。
helper 里有两套行号：
- **plan 的行**（entry 顺序，前缀计数 ⇒ **每次运行一样**）：结果行 `d_out` 就是它（MMQ 写 `dst row = ids[row]`）；
- **helper 自己的行**（专家顺序，组内的名次来自 `atomicAdd` ⇒ **每次运行不一样**）：token 和权重在它里面。

第一版让加权和核用 **plan 的行**去取 `d_out`、用**同一套行号**去取 `src`（token）和 `wgt`（权重）
⇒ **每一行都配错了自己的 token 和权重**，而且**加法的顺序每次运行都不一样**（helper 自己的行序是 atomic 的）。
⇒ 门的 **C2 失败**（peer 的两轮 token 不同：p01 在第 0 个 token、p02 在第 7 个、p05 在第 4 个）。
⇒ 修法：加一张反向表 `d_inv`（**plan 的行 → helper 自己的行**），
加权和核走 **plan 的行**（顺序固定 ⇒ 每次运行一样），每一行通过 `d_inv` 找到**它自己**的 token 和权重。
⇒ **教训**：helper 里有两套行号，跨边界的东西（token、权重）必须跟着 `d_inv` 走。


**量到的（MEASURED，2026-10-26，同一台机器，同一个二进制 `bin/strata-q36.sum`（**由最终源码编出**），
旗标是唯一的差别）**：

| | pp512（bench 2）| tg64 | 前向 ms/层 | helper_wait | plan | primary_work |
|---|---:|---:|---:|---:|---:|---:|
| S1 行走 int8（对照）| 740.5 | 59.5 | 17.219 | 3.102 | 2.242 | 4.668 |
| **S2 每 token 的加权和**| **842.0**（+13.7%）| 59.1 | **15.135** | **1.041** | 2.237 | 4.609 |
| S3 和 + fp16 传输（对照是 S2 的 int8）| 782.3 | — | 16.294 | 2.118 | 2.291 | 4.632 |

⇒ 下行那一段省了 **2.06 ms/层**（helper_wait 3.102 → 1.041），整层省 2.08 ms ⇒ **+13.7%**。
⇒ 打包自己量到的：和的最大幅值 **4.77422**（行的最大幅值是 18.6674 —— 和是加权的平均，小得多），
   最大误差仍是块内幅值的 **0.00394**（打包的代价与 int8 那条一样，与内容无关）。
⇒ **解码不动**（它不走这条路径）：只跑解码的 A/B：tg64 **60.7**（和关）vs **59.3**（和开），
   两臂的三个轮次都在 59.3-60.7 之间跳 ⇒ 差在**轮次之间的波动带**里，不是这条旗标的性质。
⇒ 无 IMA、Xid 138→138。

**验收（2026-10-26，同一台机器，最终源码编出的二进制）**：

| | 结果 |
|---|---|
| 默认路径的 parity（`gen: 220`）| **220**（旗标关着时默认路径逐字节不变）|
| 门（`q36_peer_gate.py`，6 个 256-token 的 prompt，peer 跑两轮 + 默认跑一轮）| **C1..C5 全过**；两轮的 token 相同；peer 676.9 tok/s vs 默认 78.1（8.66x）|
| peer 的 token：和 vs int8 的行（同一批 prompt）| **6 个里 4 个逐字节相同**，p01 在第 8 个 token 上不同、p05 在第 13 个 |
| 和 + fp16 传输（另一个打包传输）| 跑得通，782.3 tok/s（线上字节是 int8 的两倍）|
| Xid | 138 → 138；无 illegal memory access |

**pp512 的总账（同一个二进制，一串旗标）**：493.9（基准）→ 589.6（q8 window）→ 738.3（int8 回程）→
**842.0（每 token 的加权和）** ⇒ **1.70x**。

**接下来（prompt 还剩什么）**：前向 15.14 ms/层里，**主卡自己占 11.4 ms**
（mixer 6.767 + primary_work 4.609），peer 路径自己的只剩 3.3 ms（plan 2.237 + helper_wait 1.041）⇒
**peer 这条链已经不再是瓶颈**。⇒ 下一轮是 §3.7 的那条：主卡自己那 39 个专家/层
（primary_work 4.609 ms/层）换 fused FP16 的 K-quant 核（§3.7 量的核本身快 46%）
⇒ 若换路后是 2.5 ms/层，整层 13.0 ms ⇒ **≈980 tok/s**。
但 §3.7 的三条都还在：① 那条路现在对 served layer 是关的（`!peer_here`），peer 的行号要重排
（**想法**：主卡侧加一个小的 scatter 核，把 fused 路径写的「排序行」搬到 plan 的行 ——
不动共享的 fused 文件 ⇒ **只编 CUDA**，HIP 那边一个字都不用碰）；
② 换了算术 ⇒ **peer 的数字会变**（不再是逐字节相同）；
③ 要单独一轮，不能和别的混在一起。

### 3.13 这一轮：主卡自己那 39 个专家走 fused FP16（书面大纲，2026-10-26）

**为什么是它**：前向 15.14 ms/层里，主卡自己占 11.4 ms（mixer 6.767 + primary_work 4.609），
peer 路径自己只剩 3.3 ms。mixer 在第三方 llama.cpp 里（不动），primary_work 是我们的文件。
§3.12 的探针量到主卡那侧快了 **1.5 ms/层**（4.668 → 3.091），但探针的行号是错的 ⇒ 路由变了 ⇒ 答不了净收益。

**做法（只编 CUDA）**：给 fused 的 down 核加一张「排序行 → 计划行」的表（ids），
主卡侧把 `!peer_here` 的限制去掉，让 served layer 也走 fused。

| 步 | 改哪里 | 做什么 |
|---|---|---|
| 1 | `src/prefill/moe_f16_kq_cuda.cuh` | down 核多一个 `ids`（可空）：写 `out[ids[row]*n_out+col]`，空则和现在一样 |
| 2 | `src/prefill/moe_f16_kq_cuda.cu/.hpp` | 加 CUDA 专用的 `f16kqc::down_rows(...)`（**不动共享的 moe_f16_kq.hpp** ⇒ HIP 那个文件一个字都不用碰）|
| 3 | `src/qwen35/q36_engine.cu` | fused 的调用点改用 `down_rows(..., lid, ...)`，并去掉 `!peer_here`；非 peer 时 `lid = ident` ⇒ 逐字节不变 |
| 4 | 箱子上 | 编 CUDA（没有 ROCm，**HIP 没编**）、A/B 测、跑验收门 |

**要记住的坑**：
- `lid` 在 peer 上是 `peer_lid_map`（计划行），在非 peer 上是 `ident`（恒等）⇒ 同一个调用点两边都对。
- 换了算术（fp16 FMA vs int8 DP4A）⇒ **peer 的文字会动**（和 int8 那条一样，不是逐字节相同）。
- 开了 `STRATA_PF_F16_CUDA` 会多占 ~16 MB 显存（f16 的缓冲）⇒ 专家的分配会变 ⇒ A/B 的
  「主卡 entry/层」要一起读（探针会打印），别把它当成噪声。

**先撤掉的**：§3.11 之后试的「window 走自己的 stream + 先小后大」—— **两轮都没动**（840.3 vs 842.0，
前向 15.166 vs 15.135）⇒ 撤掉，记在这里。

### 3.14 做了：served layer 也走 fused FP16（`STRATA_PF_F16_CUDA=1` + peer 旗标）

改了四处（**只编 CUDA**）：`src/prefill/moe_f16_kq_cuda.cuh` 的 down 核多一个 `ids`（行表），
`moe_f16_kq_cuda.cu/.hpp` 加一个 CUDA 专用的 `f16kqc::down_rows(...)`（**共享的 moe_f16_kq.hpp 一个字没动**
⇒ gfx11 那份看不到），`src/qwen35/q36_engine.cu` 的调用点改用 `down_rows(..., lid, ...)` 并去掉 `!peer_here`。
非 peer 时 `lid = ident`（恒等）⇒ **逐字节不变**（旗标关着时默认路径也不变：parity `gen: 220`）。

**量到的（MEASURED，2026-10-26，同一个二进制 `bin/strata-q36.f16`，旗标是唯一的差别）**：

| | pp512（bench 2）| 前向 ms/层 | primary_work | helper_wait | plan |
|---|---:|---:|---:|---:|---:|
| F1 走 MMQ（对照）| 839.7 | 15.177 | 4.608 | 0.985 | 2.331 |
| **F2 走 fused FP16**| **875.9**（+4.3%）| **14.545** | **3.115** | 2.000 | 2.176 |

⇒ **主卡自己那 39 个专家/层的确快了 1.49 ms/层**（4.608 → 3.115，-32%），
和 §3.12 的探针（4.668 → 3.091）几乎一样 ⇒ 探针量的那 1.5 ms 是真的。
⇒ **但整层只快了 0.63 ms（-4.2%）**：省下的 1.49 ms 里有 **1.02 ms 被 helper_wait 吃掉了**
（0.985 → 2.000）—— **helper 那条链才是临界路径**（§3.15）。

⇒ 「每层 entry 数」没变（primary 631.6 / CUDA1 1068.7 / CUDA2 1182.2 / CUDA3 1213.4）⇒
旗标带来的那 16 MB 缓冲没有动专家的分配 ⇒ 这次 A/B 是干净的。

**代价**：换了算术（fp16 FMA vs int8 DP4A）⇒ **peer 的文字会动**（和 int8 那条一样）。

**验收（2026-10-26，同一台机器，`bin/strata-q36.f16`）**：

| | 结果 |
|---|---|
| 默认路径的 parity（`gen: 220`，旗标关着）| **220** ⇒ 默认路径逐字节不变 |
| 门（`q36_peer_gate.py`，6 个 256-token 的 prompt，peer 跑两轮 + 默认跑一轮）| **C1..C5 全过**；两轮的 token 相同；peer 714.1 tok/s vs 默认 78.1（9.14x）|
| **peer 的 logits：fused vs MMQ（同一个 256-token 的 prompt，8 个 token）** | **argmax 8/8 相同**；rms(diff)/rms(logits) = **0.074**（最大 0.287）|
| peer 的 token：fused vs 每 token 的加权和 | 6 个里 4 个逐字节相同，p01 在第 4 个、p05 在第 13 个不同 |
| Xid | 138 → 138；无 illegal memory access |

⇒ **行表是对的**：logits 只差 7.4%（RMS），argmax 全同。
⇒ **而且离默认路径更近了**：门报「与默认路径的第一个不同 token」——
MMQ 那条是 p01 第 4 个、p05 第 13 个，**fused 这条是 p01 第 20 个、p05 第 19 个**
（fp16 比 int8 准 ⇒ peer 的文字更靠近默认路径的）。

### 3.15 前向 14.545 ms/层 现在是什么（MEASURED，2026-10-26，F2）

| 段 | ms/层 | 在谁身上 |
|---|---:|---|
| mixer（GDN/attention 的投影、conv、SSM）| **6.778** | 主卡（第三方 llama.cpp + 共享的 gemm.cu）|
| router + route D2H | 0.264 | 主卡 |
| plan（主机侧：记账 0.34 + prepare 0.34 + 等 helper 的行计划 ~1.5）| 2.176 | 主机在等，**GPU 空着** |
| primary_work（主卡自己那 39 个专家，fused FP16）| 3.115 | 主卡 |
| helper_wait（等 helper 的行回来）| **2.000** | 主机在等 |
| combine | 0.212 | 主卡 |

⇒ **主卡自己的活 = 6.778 + 3.115 + 0.48 = 10.4 ms；helper 那条链 ≈ 7.5 ms**
（从 plan 开始到 helper 的行落地：主机 0.68 + window 上行 1.4 + 行计划 0.1 + 算术 ~3.4 + 下行 1.9）。
⇒ **整层 = 主卡开头的 7.04 + max(plan 2.18 + primary_work 3.115 + combine 0.21, 链 7.5) = 14.5**。
⇒ **helper 那条链是墙**（7.5 ms），主卡自己的 10.4 ms 里有 2.9 ms（primary_work 的一部分）在链的影子里。

⇒ **能动的只剩两处**：
1. **helper 那条链（7.5 ms）**：window 上行 1.4（q8_1，1.08 MB @ 0.81 GB/s）+
   算术 ~3.4（helper 的 native int8 MMQ）+ 下行 1.9（1.08 MB @ 0.58 GB/s）。
   算术那 3.4 里 **1.18 是权重流量**（每层每 helper 读 85 个专家 × 3.1 MB = 264 MB @ 224 GB/s），
   每专家的行数只有 ~13.6 ⇒ 形状很瘦 ⇒ 数学效率只有 int8 峰值的 35%。
2. **mixer（6.778 ms）**：dense 投影走 `prefill::Gemm::native`（Q4_K → FP16 的 scratch + **一次
   `cublasGemmEx`，`CUBLAS_GEMM_DEFAULT`**）⇒ **cuBLAS 的默认算法没被挑过**
   （gfx11 那边有 tune_hipblaslt 的算法表，CUDA 这边没有）⇒ 下一个要量的就是它。

**mixer 的形状（从 gguf 的头读出来的，`.run/ggufshapes.py`）**：
GDN 层：`attn_qkv` 2048×8192、`attn_gate` 2048×4096、`ssm_out` 4096×2048；
attention 层：`attn_q` 2048×8192、`attn_k`/`attn_v` 2048×512、`attn_output` 4096×2048。
⇒ 一层 28-34 GFLOP，512 token ⇒ **mixer 那 6.778 ms/层 ≈ 4.4 TFLOP/s**。
⇒ P100 的 FP16：向量 9.3 TFLOP/s、张量核（FP32 累加）9.3、（FP16 累加）18.7
⇒ **cuBLAS 现在大概是向量路径的一半**：要么它已经在用张量核（47%），
要么向量路径就是 47% ⇒ **下一个实验：把 cuBLAS 的 24 个算法 + 两种累加方式都量一遍**
（`.run/gemmprobe-20261026.cu`）。

### 3.12 探针：fused FP16 的核在主卡上到底快不快（**量到了，但结论是「探针答不了」**）

§3.7 的那条路（fused FP16 的 K-quant 核）在 served layer 上是关着的（`!peer_here`），
因为它写的是**排序行**而 combine 读的是**plan 的行**。**先量一下值不值得做**：
临时去掉 `!peer_here`（**行号没重排 ⇒ 答案是错的，只量时间**），同一个二进制两趟：

| | pp512（bench 2）| 前向 ms/层 | primary_work | helper_wait | 主卡自己的 entry/层 |
|---|---:|---:|---:|---:|---:|
| P1 MMQ 那条路（对照）| 737.6 | 17.287 | 4.668 | 3.080 | 642.6 |
| P2 fused FP16（**行号错的探针**）| 745.9 | 17.094 | **3.091** | **4.621** | 630 左右 |

⇒ **主卡那侧确实快了**：primary_work 4.668 → 3.091 ms/层（-34%），而两边的活是同一份
（主卡的 entry 642.6 → 630，差 2%）⇒ **fused 那个核在专家在显存里的时候是快的**。
⇒ **但 helper 那侧慢了同样多**（helper_wait 3.080 → 4.621）⇒ 前向只动了 0.19 ms。
⇒ **而「helper 为什么慢」没有答案**：探针的行号是错的 ⇒ 隐藏状态是错的 ⇒ **路由变了**
（entry 的分布变了，虽然总数一样）⇒ **helper 做的不是同一份活**，两个效应分不开。
⇒ **⇒ 这个探针量不了净收益**（它只证明了「主卡那侧快 1.5 ms/层」）。
⇒ **要做就得做对**：行号的重排（主卡侧一个 scatter 核：fused 写排序行，scatter 把它搬到 plan 的行）
⇒ 行号对了，路由就一样，A/B 才干净。**那个核在引擎自己的文件里** ⇒ **只编 CUDA**
（fused 那条路整个在 `#if !defined(__HIPCC__)` 里）。
⇒ 代价还是那两条：**peer 的算术会变**（文字会动），**而且要单独一轮**。

### 3.16 探针：mixer 的 dense GEMM（MEASURED，2026-10-26）

§3.15 说下一个要量的是「cuBLAS 的默认算法没被挑过」。量了三件事，结论是**没有免费的午餐**：

| 变体 | 速度（T=512，28-34 GFLOP/层的形状）| 相对引擎的路径 | 相对误差 rms(diff)/rms |
|---|---|---:|---:|
| 引擎的路（`CUBLAS_GEMM_DEFAULT`，fp32acc）| 6.5–7.3 TFLOP/s | 1.00x | — |
| A 张量核算法（`CUBLAS_GEMM_ALGO0_TENSOR_OP`）| 6.5–7.2 | **1.00x** | **0（逐位相同）** |
| B cuBLASLt fp32acc（16 个启发式里最快的）| 6.5–7.2 | 0.96–1.05x | **0（逐位相同）** |
| **C fp16acc**（`CUBLAS_COMPUTE_16F`，ALGO6）| **11.2–13.5** | **1.57–1.98x** | **0.36–0.38%** |
| D fp16acc 分 4 段 K + fp32 求和 | 6.7–9.5 | 0.94–1.45x | 0.23–0.26% |

⇒ **cuBLAS 的默认已经是最好的 fp32 累加核**（A、B 都是 1.00x 且逐位相同）⇒ 「挑算法」这条路是死的。
⇒ P100 的 fp32 累加 FP16 张量核是 9.3 TFLOP/s，实测 6.5–7.3 ⇒ **已经在那条路径的 70–78%**，
剩下的是这些形状本身的效率（N 只有 2048–8192、T=512 的瘦矩阵）。
⇒ **唯一的 2x 是 fp16 累加**（12.9 TFLOP/s ≈ 峰值的 69%），代价是累加精度：**每层 0.36% 的相对误差**。
⇒ D（分段）把误差降到 0.23%，但速度只剩 1.09–1.45x，而且每层多 T×N 的 partial 流量 ⇒ 不划算。

**下一步（本轮）：** 把 C 做成 opt-in（`STRATA_DENSE_F16_ACC=1`），只在 compute capability 6.x 上开，
量 pp512 的净收益和 logits 的差。**判据**：增益 < 3% 就回退（0.36%/层的误差要拿速度换）。

#### 3.16.1 做成了，量到了（MEASURED，2026-10-26）

**算法与形状、T 都有关，误差也一样**（`gemmacc-table`，P100，CUDA 12.8，引擎自己的输出布局：FP16 结果写在
FP32 行首、就地加宽）：

| 算法 | 速度 | rms(diff)/rms |
|---|---|---|
| ALGO 4 / 5 / 6 | 1.2–2.05x | **0.36%** |
| ALGO 1 / 2 / 3 | 0.4–1.94x | **1.2–1.35%**（K 4096 的 ssm_out 到 **15%**）|

⇒ 只收 ALGO 4/5/6。ALGO 1/2/3 有时更快（T 2048 的 wide 形状 1.94x 对 1.59x），但那是拿 3.5 倍误差换的，
不是这笔账该做的交易。表在 `gemm.cu` 的 `kDenseAcc`，按 (N, K, T band) 查。

**就地 vs 单独 FP16 buffer**（`gemmacc-layout`）：就地只在 2% 以内慢一点（1.489 vs 1.494 ms），
所以不多要 T×N×2 的内存。

**A/B（同一个二进制 `bin/strata-q36.acc`，只差旗标，bench 2）：**

| | pp512 | tg64 | mixer ms/层 | forward ms/层 |
|---|---:|---:|---:|---:|
| G1 旗标关（对照）| 830.6 | 59.3 | 6.766 | 15.288 |
| G2 旗标开 | **972.0** | 58.6 | **4.811** | **13.101** |

⇒ **pp512 +17.0%**，mixer −1.96 ms/层。判据是 3%，差了一个数量级 ⇒ 留下。

**和 fused FP16 experts 叠起来**（`R-acc-stack`，两条路管的是不同的东西：dense 投影 vs 路由到的专家）：

| | pp512 | mixer ms/层 | forward ms/层 |
|---|---:|---:|---:|
| G3 全套不开 dense flag（对照）| 864.0 | 6.770 | 14.748 |
| G4 全套开 | **1001.7** | 4.805 | 12.711 |

⇒ **+15.9%**，**破 1000**（默认路径 156.4 的 **6.40x**，起点是 493.9）。

**输出（`R-acc-logits`，256 token prompt，同一个二进制）：** argmax **8/8** 相同，
rms(diff)/rms(logits) 平均 **0.085**（之前收下的 fused FP16 路径是 0.075，同一量级）；
`gen:` 的 8 个 token 逐位相同。


### 3.17 到顶了：三条候选都量过、都否掉（MEASURED，2026-10-26）

prompt 从 493.9 推到 **1001.7**（+103%）。再往上要找的地方都量了，全部到顶：

**(a) dense GEMV 已经打满带宽.** `.run/q8gemv-20261026.cu` 在 P100 上比「纯流式读同样字节」和「引擎自己的
Q8_0 row-dot」：

| 形状 | 纯读 GB/s | row-dot GB/s | 纯读 ms | row-dot ms |
|---|---:|---:|---:|---:|
| attn_qkv 8192×2048 | 414.5 | **433.3** | 0.043 | 0.041 |
| attn_gate 4096×2048 | 387.8 | **402.5** | 0.023 | 0.022 |
| ssm_out 4096×2048 | 387.7 | **403.2** | 0.023 | 0.022 |

⇒ row-dot **比纯读还快**（它边读边算，读/算重叠），已贴着 HBM 的实际带宽（P100 峰值 732，这个访问模式到 ~410）。
`STRATA_Q8_SM60=1` 早已开着（所有测量都带它），Pascal 的 packed q8 kernel 已在跑。**没有空间。**

**(b) 解码的 dense 权重地板.** 每 token 的 dense 权重 = 37.9 MB/层 × 40 = **1.51 GB**；在 ~410 GB/s 的实测带宽上
**地板 3.7 ms/token**。实测 `attn_ssm` 0.150 × 40 = 6.0 ms，其中 3.7 是这四个 GEMV 的地板，剩下 2.3 ms 是
conv / SSM 递推 / norm。`k_gdn_seq_cols` 对 T 严格串行（每 token 依赖上一 state），分块算法（`gdn_chunk.cu`，
T/64 步）只给 Volta 开，且它改算术、要单独一轮验证。

**(c) `cpu_pool` 的 0.105 ms/层是自旋，不是工作.** 报告写着
`61440 of 61440 batch entries on the GPU (100.0%), 0 on the CPU pool`，`route_spin=0.119 ms/tok`（6018 次调用）
和 `cpu_pool=0.110 ms/层` 是同一个数。

⇒ **试过、量过、否掉**：`moe_helpers()` 无条件 `cudaStreamSynchronize(st)`，而 `ev_route` 记在它**之后**的
空 stream 上 ⇒ 那次自旋必然立刻返回。加 `stream_synced_this_layer` 跳过它后
**`route_spin=0.000`**、logits **逐位相同**（0.08483，8/8 argmax），但 **tg64 61.2 vs 61.4，噪声内**：
时间只是从 `route_spin` 搬到了 `prebegin_sync`（5.34 → 5.42 ms/tok）。host 本来就在等 GPU，少一次自旋不改变
它等多久。⇒ **已回退**，不留没收益的改动。

**(d) helper 链贴着 PCIe 上限.** 每层 6.38 MiB（3 份 q8_1 窗口上行 + 3 份 int8 行下行），
三条 helper 全走 host staging（无 P2P），实测 H2D 0.81 / D2H 0.58 GB/s ⇒ 理论 9.79 ms/层，
实测 4.56 ms/层（传输重叠）⇒ **已在物理上限附近**。

### 3.18 两条剩下的路也量过了（MEASURED，2026-10-26）

**(a) SSM 递推的分块算法：不值得移植。**

`.run/R-acc-stack` 之后在引擎里加了 `STRATA_Q36_GDNPROF=1`（只计一次，默认关），量到
**递推 + norm = 1.906 ms/层**（mixer 4.81 的 40%，T=512，P100）。

但 barrier 的成本单独量了（`.run/barrier-20261026.cu`，同样的 128 block × 512 线程）：
**0.26 µs/barrier**，1024 次 = **0.263 ms**。所以 1.906 里只有 0.26 ms 是 barrier，
剩下 1.64 ms 是算术本身（每 token 每列 16 次 FMA + 1 次 `__expf`）。

chunk 化省的是 barrier（512 次串行降到 8 次），算术还在。**收益上限 ~0.3 ms/层**
（forward 12.7 的 2%）——不够 3% 的判据。而且现成的 `gdn_chunk.cu` 是给 Flash-Next 的
（48 V heads、3 个共享 k/q），q36 是 32 V heads、2 个共享，不能直接用。

**(b) helper 的传输格式：已经在上限。**

`PEER_I8` + `PEER_SUM` 已经把下行压到每 token 一行 int8（512 × 2048 × 1 B = 1 MiB/helper）。
三条 helper 全走 host staging，实测 H2D 0.81 / D2H 0.58 GB/s。
每层 6.38 MiB 的流量在这个带宽上是 9.79 ms，实测 helper 链 4.56 ms（传输重叠）⇒ **已在物理上限附近**。

helper 的计算也贴着带宽：每 helper 每层 ~1100 entries × 1.87 MB ≈ 2 GB 权重，
P104 的 ~320 GB/s ⇒ 计算地板 ~6 ms/层，和传输链的 4.56 ms 同量级。

### 3.19 MTP 在编码任务上有收益（MEASURED，2026-10-26）

`--bench-tg` 测不出 MTP：它喂随机 token，草稿几乎不会命中，接受率接近 0。
换成**引擎自己的 tokenizer**（`tools/strata_tokenizer.py`，从 GGUF 读）tokenize 的编码 prompt，
生成 256 token，接受率完全不同。

两个 prompt（168 / 172 token），`--mtp 0` 对照，`--mtp 1..6` 加 `STRATA_Q36_MTP_VRAM=1
STRATA_Q36_MTP_MAPPED=1`（MTP 层全驻留主卡，mapped chain live）：

| --mtp | prompt 0 | prompt 1 | 接受率 | vs 60.5 |
|---|---:|---:|---|---:|
| 0 | 60.4 | 60.6 | — | 1.00x |
| 1 | 80.6 | 82.3 | 90–96% | 1.35x |
| 2 | 89.5 | 92.6 | 87–92% | 1.51x |
| **3** | **94.1** | **95.9** | 82–85% | **1.57x** |
| 4 | 91.5 | 95.0 | 76–81% | 1.54x |
| 5 | 84.8 | 91.2 | 68–74% | 1.45x |
| 6 | 88.5 | 93.0 | 69–72% | 1.50x |

逐深度接受率（`--mtp 4`，两个 prompt）：d=0 86/95%，d=1 81/88%，d=2 73/75%，d=3 61/66%。
d=4（`--mtp 5`）降到 47/48%。`--mtp 6` 的 d=4 是 56/54%、d=5 是 54/43%——
D=4 起单步接受率掉到一半，多出来的草稿不再抵得过它的成本，峰值停在 `--mtp 3`。

**`--mtp 3` 是峰值：94–96 tok/s，比 60.5 快 57%。** 代价在 prompt 上，量过了
（`R-mtp-pp-20261026`，同一个二进制 `bin/strata-q36.acc`，只差 MTP，bench 2）：

| | pp512 | tg64 | 主卡 slot/层 | 主卡 entry/层 | helper entry/层（三个合计）| forward ms/层 | primary_work | helper_wait |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `--mtp 0` | 1002.3 | 61.1 | 38 | 633.3 | 3462.7 | 12.703 | 2.977 | 2.084 |
| `--mtp 3` + `MTP_VRAM` | **965.7**（-3.7%）| 60.4 | 33 | 510.1 | 3586.0 | 12.763 | 2.646 | 2.530 |

⇒ pp512 掉的是 **3.7%**，不是之前写的 4–15%（那个数字是 2026-10-20 旧二进制上的，见
`2026-10-20-q36-1p3h-optimal.md` §2.2，当时的 pp512 基线是 158）。

掉的原因不是 MTP 的计算。prefill 里 `forward_mtp(..., draft=false)` 只给 MTP 块的 KV 预热，
不生成草稿（`q36_engine.cu:5358`），而且 512 token 的 prompt 是一个 chunk，只跑一次。
掉的是**主卡的专家预算**：MTP 层的 256 个专家占了主卡约 470 MiB，主层每层从 38 个 slot 降到 33，
主卡每层少算 123 个 entry（633 → 510）。这 123 个 entry 被 `keep_resident` 放给了 helper
（3463 → 3586），而 helper 的 native 路径每个 entry 比主卡的 MMQ 便宜——但 helper 已经是临界路径
（helper_wait 2.084 → 2.530 ms/层，+0.45），主卡省下的 0.33 ms 被它吃掉还多。
净结果：forward 12.703 → 12.763 ms/层，pp512 -3.7%。

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

### 4.1b 在新二进制上复量（MEASURED，2026-10-26，`bin/strata-q36.acc`）

`R-dec-acc`，全套旗标（peer + `STRATA_PF_F16_CUDA=1` + `STRATA_DENSE_F16_ACC=1`）：

| | tg64 | pp512 |
|---|---:|---:|
| E1 `--kv f16`（对照）| 59.2 | 1010.1 |
| E2 `--kv int8` | **61.4**（+3.7%）| 1003.4（不变）|

⇒ 老的 `--kv int8` 结论在新二进制上仍然成立，**tg64 现在是 61.4**（基线 59.1，+3.9%）。
⇒ dense 的 FP16 累加**没有动解码**（解码的投影是 GEMV，走 `mmq` 不走 `Gemm::f16`），符合预期。

**解码每一层 0.39 ms 的去向**（E3 的 `GPU primary` 时间线，bench 2）：

| 段 | ms/层 | 占比 |
|---|---:|---:|
| attn_ssm（mixer 的 dense GEMV）| **0.150** | 38% |
| cpu_pool（**主卡在等 helper 的行，GPU 空转**）| **0.105** | 27% |
| helper_dispatch | 0.067 | 17% |
| local_shared | 0.031 | 8% |
| router | 0.021 | 5% |
| helper_merge | 0.015 | 4% |
| local_grouped | 0.002 | 0% |

三个 helper 的 host 侧：ms_begin 1.65–1.76、ms_wait 1.45–1.65、ms_sync_overhead 0.11 ms/token。

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
