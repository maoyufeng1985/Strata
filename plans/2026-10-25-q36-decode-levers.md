# q36：decode 的下一步（按墙钟排序，2026-10-25）

状态：进行中。墙钟模型见 `2026-10-25-q36-mixer-graph.md` §1；mixer graph 测完（没用，回退）。
**这一轮量完的**：helper 的 begin 全是 CUDA 调用（§3）、D5 的 BPS 扫描（§5）、
S1 的 SM 数查询（§6，没收益）、D6 的小形状 block 尺寸（§7，没收益）。
**留下的**：S2/S3 要动 CUDA+HIP 共用的文件，本机没有 ROCm 编不了 HIP（§3 末）。
机器 192.168.1.98（P100-12G 主卡 + 3×P104-100 helper，CUDA only）。

## 1. 现在的墙钟（MEASURED，2026-10-25，marshal 开，tg128）

一 token ≈ 16.4 ms = 410 µs/层 × 40 层 + 1.34 ms（head/embedding/采样）。
host 侧的六段（每层，÷40 得 µs/层）：

| host 段 | µs/层 | 是什么 |
|---|---:|---|
| prebegin_sync | 134 | 等 GPU（mixer 135 + router 21）|
| mixer | 56 | **GPU 的活**（graph 实测：host 的 launch 只占 6.6 µs/层）|
| begin | 5.5 | 把活投给 worker |
| local | 25 | 本地分组的 launch（GPU 的 local_grouped 只有 2 µs/层）|
| shared | 22.7 | **GPU 的活**（shared expert 的 25 µs/层）|
| cpu | 8 | CPU 池的簿记（decode 里它一个专家都不算）|
| finish | 104 | **helper 的来回**（等 begin 完 + 收割）|
| 合计 | 355 | host_layer 377，host_own 242 |

**helper 侧（每层）**：ms_begin 41.7 µs（begin_slot 的主机活）、ms_wait 45 µs
（收割时等 helper 的流）、ms_sync_overhead 3.3 µs。

**（2026-10-25 复测，这三个数与上面不同，以新为准）**：同机、同旗标（marshal 开、
`STRATA_Q36_HELPER_TIMING=1`、tg64、192 token），每层 40 层换算：
**ms_begin 76.8 µs/层、ms_wait 19.1 µs/层、ms_sync_overhead 9.5 µs/层**
（每张 helper 各一份，AGG 行：`ms_begin=597.7 (3.113 ms/tok) ms_wait=145.8 (0.759)
ms_sync_overhead=73.0 (0.380)`）。**ms_wait 和 ms_sync_overhead 与旧数差得远**，
旧的 45 / 3.3 µs 不知道是怎么除出来的——**这里只留新数**。

**所以每层 410 µs 是这样来的**（同一次跑的 GPU 时间线行
`PHASE3 bench 2 GPU primary: ... attn_ssm=0.134 router=0.020 helper_dispatch=0.074
local_grouped=0.002 local_shared=0.027 cpu_pool=0.111 helper_merge=0.016 | tile=0.384
segsum=0.384`（每 ms/token，除以 40 得 µs/层））：
- **199 µs：主卡 GPU 真的在算**（attn_ssm 134 + router 20 + local_shared 27 + 本地 2 + merge 16）
- **177 µs：helper 的一个来回**（时间线的 helper_dispatch 74 = 主卡等路由的 D2H +
  host 投给 worker；host 的 finish 段 103.5 = 等 helper 的结果回来）
- 36 µs：host 自己的活 + head/embedding/采样

（时间线的 `cpu_pool=0.111` 那 111 µs/层不是 CPU 池在算——decode 里它一个专家都不算，
那是 host 在 finish 里等 helper 时 GPU 队列空着的那段时间。）

**两个 graph 实验（mixer 的、shared expert 的）都证明：host 各段的时间大多是 GPU 的活，
不是 host 的 launch 开销。所以能换墙钟的只有「少算」或者「少一个来回」。**

## 2. 排好序的候选

| # | 候选 | 预计 | 代价 | 状态 |
|---|---|---:|---|---|
| A | begin_slot 的主机活（42 µs/层，在关键路径上：begin 排队完 helper 的 GPU 才能开始）| 10–30 µs/层 | 中（remote_experts.cpp 是共享文件）| **量完了（§3）：全是 CUDA 调用，没有别的**|
| B | 整窗 H2D 64 KiB（PCIe，merge 在等它）| 26 µs/层？| 小不了（8 个 entry 全是 helper 算的，数据是必需的 fp32）| 死路 |
| C | helper 的结果直接 P2P 进主卡 | — | — | **死路**：12 对 cudaDeviceCanAccessPeer 全 0 |
| D | 主卡与 3 个 helper 的活量配比 | ? | 小 | 待测 |
| E | shared expert 的 25 µs/层（只有 206–224 GB/s，grid 填不满）| ~10 µs/层 | 小 | 待测 |
| F | mixer 的 135 µs/层（79–80% 的 roofline）| 小 | 大 | 难 |
| G | CPU 池那 8 µs/层的簿记（decode 里它什么都不算）| ~6 µs/层 | 很小 | 待测 |
| S1 | **q8_sm60 的 SM 数查询**（每次 launch 都 `cudaGetDevice`+`cudaDeviceGetAttribute`）| 0.8–16 µs/层 | 很小 | **量了：没有收益，撤了**（§6）|
| S2 | **launch_check 的 `cudaGetLastError`**（每层 11.5 次 × 1.31 µs）| 15 µs/层 | 中（共享文件）| 待定（微基准说这类调用本身很便宜，见 §3）|
| S3 | **DeviceScope 的 3 次运行时调用**（drain 里的 get+2×set，ms_sync_overhead 9.5 µs/层）| 9.5 µs/层 | 中（共享文件）| **三个里最值钱的**（drain 只是等别的卡上的流，理论上不用切设备）；待定 |
| D6 | **小形状一个 block 走一行**（router 32 block、shared expert 64 block）| ~10–20 µs/层 | 小（CUDA-only 文件）| **量了：没有收益，撤了**（§7）——小形状不是 grid 限制的 |

## 3. A 的拆解（量完了）

**A2 的量法**：一个 LD_PRELOAD 的 shim（`.run/api_time.c`），给每次 CUDA runtime 调用
计时，按线程和按调用类型分开打印。三个 marshal worker 是各自的线程 → worker 的总和
就是 helper 那边的开销，主线程的是主卡的。不动仓库的代码。

MEASURED（2026-10-25，192.168.1.98，marshal 开，tg64，192 token，shim 在）：

**主线程（主卡）每层 62 次 CUDA 调用：**

| 调用 | 次数/层 | 每次 | 每层 |
|---|---:|---:|---:|
| launch | 20.7 | 5.37 µs | 111 µs |
| stream_sync | 2.0 | 63.1 µs | 127 µs |
| memcpy_async | 4.2 | 6.80 µs | 28 µs |
| get_last_error | 11.5 | 1.31 µs | 15.3 µs |
| get_device | 8.3 | 1.46 µs | 12.1 µs |
| set_device | 1.9 | 2.06 µs | 4.0 µs |
| event_record / event_query | 2.2 | ~3.2 µs | 7.1 µs |

（`memcpy` 9696 次 × 780 µs 是**模型装载**，不是 decode。）

**worker（每张 helper 的 begin_slot）每层 77 µs（= ms_begin 76.8）**，全部是 CUDA 调用：
launch 3.8 × 6.0 = 23、memcpy_async 2.87 × 7.2 = 20.4、stream_sync 0.96 × 15.2 = 15.1、
set_device 3.8 × 2.1 = 8、get_device 2.87 × 1.6 = 4.5、get_last_error 3.8 × 1.34 = 5.1。

**⇒ 那「差掉的 30 µs」不在别处：begin_slot 就是 CUDA 调用。** 里面真正的活是
launch（23）和拷贝（20.4）；**15.1 µs 的 stream_sync 是 worker 在等自己那张卡的
上一批**（helper 的队列没排空）；**17.6 µs 是簿记**（get_device + get_last_error +
set_device），跟 A1 微基准说的「launch 0.21 µs」不矛盾：那是空卡上的 launch。

**shim 自己的代价**（要减掉再看）：带 shim 57.6–57.9 tok/s，不带 61.3–61.8 ⇒
墙钟差 4.2%，每个 token 被拦的调用 4109 次 ⇒ **每次约多 0.17 µs**。上面的数减掉这一份。

**微基准**（`.run/drv_cost.cu`，同一台机器，紧循环、空卡、20 万次）：
`cudaGetLastError` **0.029 µs**、`cudaGetDevice` **0.045 µs**、
`cudaDeviceGetAttribute` **0.046 µs**、**`cudaSetDevice` 6.2 µs**。
紧循环的数和引擎里的差很多（缓存冷热、驱动状态的上下文不同），**两个数都留着**：
紧循环说这些调用本身很便宜，引擎里（shim 的表，减掉 0.17）它们是 1.1–1.9 µs。

**新的三个候选（都在 S1/S2/S3）**：
- **S1**：`q8_sm60.cuh` 的 `sm_count()` **每次 launch** 都 `cudaGetDevice` +
  `cudaDeviceGetAttribute`，而 decode 每层 launch 它 8.3 次（attention 的四个投影 +
  shared expert 的三个 + router 的一个）⇒ 主线程那 8.3 次 `get_device` 就是它。
  **改了**（§6）：按线程缓存，编译通过，测量见 §6。
- **S2**：`native_mmvq.cu` 的 `launch_check()` 每次 MMVQ 调用一次
  `cudaGetLastError`（shim 表里 1.31 µs），每层 11.5 次 ⇒ 15 µs/层。
- **S3**：`remote_experts.cpp` 的 `DeviceScope`：drain 里 get + 2×set
  （shim 表里 1.46 + 2×2.10 = 5.6 µs），和 ms_sync_overhead 的 9.5 µs/层对得上
  （差的是分支和两次时钟读）。**这三个里最值钱的一个**：drain 只是等一条别的卡上的
  流，理论上不用切设备。

**S2 和 S3 都在共享文件里**（`native_mmvq.cu` 的 HIP 分支也编、
`remote_experts.cpp` 是 CUDA+HIP 共用），**本机没有 ROCm，编不了 HIP**——
要么把改动关进 `#if !defined(__HIPCC__)`，要么先在别的机器上编。


## 4. MTP 方向（用户要的）

MTP 一轮 = D 个 draft step + 1 个 verify 窗。draft step 的两段用
`STRATA_Q36_MTP_TIMING=1` 量（三个 cudaEvent：step 开始 / MTP 层结束 / draft head 结束）：
- layer：MTP 块自己那一层（**它的专家原来不在任何一张卡的 VRAM 里 → CPU 池算**，
  见下面；给了 GPU 之后 0.37–0.53 ms）
- head：draft head 的 vocab GEMV（Q8_0，248320 × 2048 = 515 MiB/步，1.26 ms）

`--draft-vocab N` 让 draft 只在前 N 个 token id 里选（N=0 = 全词表）。门槛：
committed 的文本不变、接受率（dp_ok）不掉。

**量完了（2026-10-25）：MTP 块自己那一层的专家不在任何一张卡的 VRAM 里**，
每步 8 个 entry 全落 CPU 池，层 1.458 ms（草稿步 2.725 ms 的一半）。
两条解法都量了，都 opt-in，都逐字节相同：

| 旗标 | MTP 路径 tok/s | 草稿步的层 | decode 的代价 |
|---|---:|---:|---|
| 无 | 78.7 / 80.3 | 1.484 ms | — |
| `STRATA_Q36_MTP_HELPER=1`（helper 拿那一层）| **92.0** | 0.531 ms | 无（61.2 → 61.8）|
| `STRATA_Q36_MTP_VRAM=1`（主卡自己留那一层）| 90.9 | **0.369 ms** | -1%（61.3 → 60.7，主卡每层 40 → 35 个槽）|

细节在 `2026-10-25-q36-mtp-helper.md`（§0 和 §5）。

**草稿头的 vocab 子集是死路**（同一份 plan 的 §4）：头快 8.6 倍，接受率掉 22 个点，
净收益为负。要有收益得用按频率挑的子集表。

## 5. D5：`STRATA_Q8_SM60_BPS` 的扫描（零代码，量过了）

`q8_sm60.cuh` 的 grid = min(groups, SM 数 × BPS)，BPS 默认 4（是 Flash-Next 的形状
上调出来的）。q36 的形状不同，所以扫一遍。

MEASURED（2026-10-25，192.168.1.98，marshal 开，同一个提示 64 token，
`--bench-tg 128` 的 bench 2）：

| BPS | decode tg128 | MTP tok/s | 草稿步的头 |
|---:|---:|---:|---:|
| 3 | 61.0 | 79.0 | 1.300 ms |
| **4（默认）** | 61.6 | **79.6** | 1.265 ms |
| 6 | 61.6 | 78.5 | 1.267 ms |
| 8 | **62.0** | 78.7 | 1.330 ms |
| 12 | **62.0** | 78.2 | 1.257 ms |

**文本在 3/4/6/8/12 之间逐字节相同**（脚本最后比了）——BPS 只改 grid，不改数值。

**⇒ 默认的 4 最好**：decode 在 8/12 高 0.6%（噪声带里），但 MTP 的 tok/s 在 4 最高
（79.6，别的 78.2–79.0），头的 1.257–1.330 ms 差在跑的抖动里。**BPS 不用改。**

**顺带量到的一件事**：小矩阵的 grid 不是被 BPS 卡住的，是被自己的行数卡住的
（`groups = ceil(n_out / 8)`，因为一个 256 线程的 block 走 8 行）：router（n_out 256）
只有 32 个 block，shared expert（512）64 个，attention 的 k/v（1024）128 个，
而 P100 有 56 个 SM。**⇒ 那是 D6（小矩阵的 block 尺寸），不是 BPS。**

## 6. S1：`q8_sm60` 的 SM 数查询（量了，**没有收益，撤了**）

`q8_sm60.cuh` 的 `sm_count()` 每次 launch 都 `cudaGetDevice` +
`cudaDeviceGetAttribute`，而 decode 每层 launch 这个 kernel 8.3 次（attention 的四个
投影 + shared expert 的三个 + router 的一个）⇒ shim 的表里主线程那 8.3 次
`get_device` 就是它（1.46 µs/次 = 12 µs/层 = 3%）。**改成按线程缓存后量了：**

| | decode tg128 | MTP tok/s | 文本 |
|---|---:|---:|---|
| 缓存前（对照，同一天的同一个二进制）| 61.6 | 79.6 | — |
| 缓存后 | 61.2 | 80.0 | **逐字节相同** |

**⇒ 没有收益，撤了。** 这台机器上 decode 的跑动带是 **61.0–62.0**（同一个二进制、
只换 `STRATA_Q8_SM60_BPS` 环境变量，5 个值给出 61.0 / 61.6 / 61.6 / 62.0 / 62.0），
所以 -0.4% 在噪声里。**⇒ 那些驱动调用在引擎的上下文里也不贵**（紧循环的微基准：
`cudaGetDevice` 0.045 µs、`cudaDeviceGetAttribute` 0.046 µs）。

**⇒ shim 的表要这样读**：它给每个调用加的两次时钟读+一次 PLT，在引擎的上下文里
大约 0.17 µs/次（墙钟差 4.2% ÷ 每个 token 4109 次），**所以 1.3–2.1 µs 那一档的数
是真的，不是 shim 的开销**（微基准的紧循环才是 shim 之外的世界）。

**顺带记下一个坑**：这个头文件是在 `native_mmvq.cu` 的 `strata::kernels::{匿名}`
命名空间**里面**被 include 的（native_mmvq.cu:1946），**标准头不能在那里 include**
——先写了 `#include <atomic>`，编译直接炸了（`namespace "strata::kernels::<unnamed>::std`
has no member "true_type"`，100 个错误）。要在这个文件里用标准的东西，只能用
`thread_local` 这种不需要头文件的东西。

## 7. D6：小矩阵的 block 尺寸（一个 block 走一行）

**问题**：`q8_sm60.cuh` 的 block 是 256 线程，一个 block 走 8 行（TPR 32 个 lane 一行），
所以 grid = ceil(n_out/8)：router 32 个 block、shared expert 64 个、attention 的 k/v 128 个，
P100 有 56 个 SM。32 个 block 的时候 24 个 SM 闲着；64 个的时候有 8 个 SM 背双份，
**最后一步是 kernel 时间的 75%**。

**量到的代价**（MEASURED 2026-10-25，192.168.1.98，GPU 时间线）：
shared expert 的三个乘积（3 MB）27 µs/层 = **111 GB/s**（P100 带宽的 15%），
router 的（512 KB）20 µs = 26 GB/s。**离带宽远得很 ⇒ 卡的是并行度，不是字节数。**

**改法**：小形状（`groups < SM 数 × BPS`，即行数自己是瓶颈的那些）用**一个 warp 一个
block**（一个 block 走 1 行）：warp 数不变（同样的并行度），block 数多 8 倍，
最后一步从 75% 降到 9%。

**数值不动**：一行的和在它自己那 32 个 lane 里用 shuffle 加完（`__shfl_xor_sync`，
width=TPR），block 尺寸只决定一个 block 走几行，不改一行的算法。
**⇒ 文本必须逐字节相同**（测量的门）。

**代价**：每个 block 都要把激活复制进 shared memory（小形状多 8 倍，L2 的流量，
不是 DRAM 的）；多 8 倍的 block 数（warp 数不变，所以 launch 的代价不变）。

### 7.1 量了：**没有收益，撤了**

MEASURED（2026-10-25，192.168.1.98，marshal 开，`--bench-tg 128`）：

| | decode tg128 | MTP tok/s | 文本 |
|---|---:|---:|---|
| 改之前 | 61.2（同一批的对照 61.6）| 80.0 | — |
| 改之后 | 61.2 | 78.9 | **逐字节相同** |

**⇒ 假设错了：小形状不是 grid 不平衡限制的。**

**为什么（看 kernel 就明白）**：`RPB = 8` 是说**一个 block 里的 8 个 warp 各走一行**
（不是顺序走 8 行）——`row = g * RPB + rloc`，`rloc = threadIdx.x / TPR`。
所以两条路的**内存级并行度一样**：大 block 是 64 个 block × 8 个 warp = 512 个 warp
各读一行，小 block 是 512 个 block × 1 个 warp = 同样 512 个 warp 各读一行。
**每行 2 KB 的读是 512 路并发，两条路一样 ⇒ 一样的带宽。** 我把「block 走 8 行」
当成了「顺序走 8 行」，这是错的。

**⇒ 那 shared expert 的 111 GB/s（27 µs/层，3 MB）是被什么卡的？** 不是带宽：
3 个 GEMV 各 1 MB，一次 launch 的固定开销（P100 的 launch 2.64 µs + 激活进
shared memory + 尾巴）就占了大头——**1 MB 的读根本填不满带宽**。
**⇒ 要提速得把三个 GEMV 融进一个 launch**（省两次 launch ≈ 5 µs/层），
或者把小矩阵的活搬到一个更宽的 kernel 里。**不是 grid 的事。**

**⇒ 这也说明 D2/D6 这类「小矩阵」的方向要重新想**：q8_sm60 的小形状每次 launch
只读 1 MB，天生就是 launch-bound 的。

