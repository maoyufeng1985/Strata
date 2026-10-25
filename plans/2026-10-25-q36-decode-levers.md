# q36：decode 的下一步（按墙钟排序，2026-10-25）

状态：进行中。墙钟模型见 `2026-10-25-q36-mixer-graph.md` §1；mixer graph 测完（没用，回退）。
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

**所以每层 410 µs 是这样来的：**
- 199 µs：主卡 GPU 真的在算（mixer 135 + router 21 + shared 25 + merge 16 + 本地 2）
- 211 µs：**helper 的一个来回**（begin 的主机活 42 + 收割 45 + PCIe 的往返）
- 27 µs/token：head / embedding / 采样

**两个 graph 实验（mixer 的、shared expert 的）都证明：host 各段的时间大多是 GPU 的活，
不是 host 的 launch 开销。所以能换墙钟的只有「少算」或者「少一个来回」。**

## 2. 排好序的候选

| # | 候选 | 预计 | 代价 | 状态 |
|---|---|---:|---|---|
| A | begin_slot 的主机活（42 µs/层，在关键路径上：begin 排队完 helper 的 GPU 才能开始）| 10–30 µs/层 | 中（remote_experts.cpp 是共享文件）| 待测（先跑微基准）|
| B | 整窗 H2D 64 KiB（PCIe，merge 在等它）| 26 µs/层？| 小不了（8 个 entry 全是 helper 算的，数据是必需的 fp32）| 死路 |
| C | helper 的结果直接 P2P 进主卡 | — | — | **死路**：12 对 cudaDeviceCanAccessPeer 全 0 |
| D | 主卡与 3 个 helper 的活量配比 | ? | 小 | 待测 |
| E | shared expert 的 25 µs/层（只有 206–224 GB/s，grid 填不满）| ~10 µs/层 | 小 | 待测 |
| F | mixer 的 135 µs/层（79–80% 的 roofline）| 小 | 大 | 难 |
| G | CPU 池那 8 µs/层的簿记（decode 里它什么都不算）| ~6 µs/层 | 很小 | 待测 |

## 3. A 的拆解（要先量）

begin_slot 的主机活 42 µs，理论上只有 4 次 CUDA 调用（2 次 H2D + 2 次 launch ≈ 12 µs）
+ 几 KB 的 memcpy + 一次 DeviceScope。差的 30 µs 在哪？**未知**——先量，别猜。

量法 A1：`.run/begin_cost.cu`（本机微基准）：每种 CUDA 调用在主卡/辅助卡上的主机代价，
含"2 launches + 2 copies + D2H + sync"整套，外加 PCIe 在 GPU 时间线上的实际带宽。
量法 A2：在 `remote_experts.cpp` 的 begin_slot 里分段计时（共享文件，改动是后端无关的 C++，
但本机不能编 HIP）。

## 4. MTP 方向（用户要的）

MTP 一轮 = D 个 draft step + 1 个 verify 窗。draft step 的两段用
`STRATA_Q36_MTP_TIMING=1` 量（三个 cudaEvent：step 开始 / MTP 层结束 / draft head 结束）：
- layer：MTP 块自己那一层（专家在主卡 sm_60 上算，主卡没有硬件 __dp4a）
- head：draft head 的 vocab GEMV（Q8_0，248320 × 2048 = 515 MiB/步）

`--draft-vocab N` 让 draft 只在前 N 个 token id 里选（N=0 = 全词表）。门槛：
committed 的文本不变、接受率（dp_ok）不掉。
