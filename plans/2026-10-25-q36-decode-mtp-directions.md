# q36：decode 与 MTP 的下一轮方向（2026-10-25）

状态：方向清单。机器 192.168.1.98（P100-12G 主卡 + 3×P104-100 helper，CUDA only）。
这一版比同名的初稿多了 2026-10-25 当天在机器上量到的东西：decode 的八段探针
修好之后的那张时间表，和四个新的计数器。带 MEASURED 的是量过的；带「静态分析」
的是读代码推的。

## 0. 这一版改了什么（与初稿的差别）

初稿（同一个文件名的上一版）把「把 6.258 ms/层的主卡 drain 拆开」排在 D1，
因为没有工具能拆。现在有了：

- **decode 的八段探针本来是坏的**：q36_engine.cu 的 `gp_push(a, b, ms)` 把时间
  记进 `gseg_ms[b - a]`，而调用处传的是 `(j, j + 1)`，所以 `b - a` 恒等于 1——
  七段的时间全部落进了第 1 段（router），另外六段打印 0.000。MEASURED（修之前）：
  报告是 `router=0.428 tile=0.428 segsum=0.428 resid=0.000`，别的段全是 0.000。
  一行改成 `gseg_ms[a]` 就修好了（本次提交）。
- **新增 host_layer / host_own 计数器**：每层 layer() 调用在 host 上花多久，
  减去里面的 drain，得到 host 自己的活。这条数是本轮所有排序的依据。
- **新增 host 的五段拆分**：mixer / begin / local / cpu / finish，host 的活花在
  哪一段里。

## 1. 基线（MEASURED，2026-10-25，binary bin/strata-q36.f16cuda3，md5 0131784d…）

配置：`STRATA_REMOTE_ZEROCOPY=0 STRATA_Q36_HELPER_ASYNC=1 STRATA_Q36_HELPER_PARALLEL=1
STRATA_Q36_FIXED_SLOTS=1 STRATA_Q8_SM60=1 --kv int8 --max-context 2048 --no-warmup`
（2026-10-20 那轮扫描的最优配置）。

| 项 | 数 | 出处 |
|---|---:|---|
| decode（--bench-tg 64，三趟） | 54.0 / 54.9 / 54.6 tok/s | .run/R-probe1-20261025 |
| decode（--bench-tg 128） | 54.8 / 54.9 / 54.7 tok/s | .run/R-decode-mtp-20261025 |
| MTP（--tokens 路径，64 token） | D=0 53.4，D=1 64.5，D=2 73.7，**D=3 74.7**，D=4 73.3 tok/s | .run/R-decode-mtp-20261025 |
| MTP，开 STRATA_Q36_HELPER_PARALLEL_MARSHAL=1 | D=0 59.2，D=1 71.5，**D=2 81.5**，D=3 79.7 tok/s | .run/R-marshal-mtp-20261025 |
| MTP 的每深度接收率（D=3） | 94.7% / 88.9% / 76.5% | 同上 |

即 decode 一 token 18.25 ms（54.8 tok/s），MTP 在 D=3 是 13.4 ms/token（+39.9%）。
与 2026-10-20 那轮（52.5 / 66.2-67.2）同形：甜点位仍是 D=3，盈亏线在 D=0..1 之间。

## 2. decode 的时间去哪了（MEASURED，2026-10-25，STRATA_Q36_HELPER_TIMING=1）

八段探针修好之后，最后一层的 GPU 时间线（每 token，每层一个样本，单位 ms/tok，
2560 次 = 40 层 × 64 token）：

| 段 | ms/tok | 每层 µs | 说明 |
|---|---:|---:|---|
| attn_ssm | 0.135 | 135 | 主卡自己这一层的 dense GEMV（GDN 或注意力）+ 归一化 |
| router | 0.021 | 21 | 路由 GEMV + top-k |
| helper_dispatch | 0.173 | 173 | **drain（host 等 GPU）+ 三张 helper 的 begin** |
| local_grouped | 0.002 | 2 | 主卡 VRAM 槽里的专家（每层约 1.25 个） |
| local_shared | 0.025 | 25 | shared expert |
| cpu_pool | 0.051 | 51 | CPU 池这一段（decode 里 CPU 池其实没活，见下） |
| helper_merge | 0.016 | 16 | helper 的行 merge 进 eparts |
| **tile（E0..E7 合计）** | **0.427** | **427** | 整层的 GPU 时间线 |

host 侧（每 token）：

| 项 | ms/tok | 每层 µs | 说明 |
|---|---:|---:|---|
| prebegin_sync（drain） | 5.55 | 139 | host 等主卡流排空 |
| route_spin | 0.158 | 4 | CPU 池等 ev_route |
| **host_layer（layer() 的墙钟）** | **16.9–17.2** | **421–430** | host 在每层里花的时间 |
| **host_own（host_layer - drain）** | **11.3–11.6** | **282–285** | **host 自己的活** |

**结论（这条是本轮最重要的数）：decode 是 host 瓶颈，不是 GPU 瓶颈。**
一个 token 18.25 ms 里，host 自己的活占 11.4 ms（62%），drain（等 GPU）占 5.5 ms（30%），
剩下的 1.4 ms 是 head、embedding、采样这些。GPU 的时间线 427 µs/层里真正在算的只有
约 199 µs（135 + 21 + 2 + 25 + 16），其余是 GPU 在等 host。

各张卡的侧翼：helper 的 ms_begin 1.37–1.52 ms/tok（每张卡各算一份，3 条独立
PCIe 链路并行），ms_wait 0.22 / 0.26 / 0.43–0.55 ms/tok（CUDA3 最慢，是收割顺序的
观测效应，见 D4）。

## 3. 什么已经排除（带把它们杀掉的数据）

- **dense 侧没有大空间可挖**：q36 的 dense 权重本来就是 Q8_0（模型里就是 Q8_0，
  attn_qkv 17 MiB/层、attn_gate 8.5、ssm_out 8.5、shexp 三个各 1.06，40 层
  1.15 GiB，加 head 515 MiB，合计约 1.66 GB/token）。引擎自己的 q8_sm60 GEMV 在
  P100 上的实测带宽（tools/q8_sm60_check.cu，2026-10-25）：head 433 GB/s、
  attn_qkv 351、attn_gate 380、ssm_out 373，而 attn_k/v 180、shexp 206/224。
  P100-12GB 的规格带宽是 547 GB/s，所以大矩阵已经在可读带宽的 80% 上下，
  小矩阵（n_out 512/640/2560）因为 grid 填不满只有一半。dense 侧可挖的总量
  约 0.3–0.5 ms/token，不值得单独立项。
- **drain 里没有「驱动等待」可省**：drain 的 139 µs/层 ≈ 该层的 dense GEMV 时间
  （28.7 MiB ÷ ~400 GB/s ≈ 75 µs）加上注意力/SSM 的小核和启动开销。初稿 D2
  （把 drain 推迟到下一层入口）、D3（窄等待）都在打这个 6.258 ms，量下来之后
  它们打的是真的 GPU 活，不是 host 空等。窄等待的历史实测是 5.4 → 5.7 ms/token
  （更慢，已删，q36_engine.cu:3751-3758 注释）。
- **CUDA graph 在 hybrid 摆放下根本用不上**：`use_graph = false` 当 main_hybrid
  （q36_engine.cu:5302「the host takes part in every layer」），而 moe_helpers 的
  drain 和 RemoteExperts::finish 的同步在 graph capture 里非法。所以「把 decode
  变成一个 graph」这条路是封死的，除非先把 host 参与的那部分拿掉。
- **这机器上 nsys/CUPTI 不可用**：nsys 能起、能出报告，但只抓到第一个 kernel
  （pack_kernel），后面的 kernel 一个都没有（2026-10-25 实测，trivial 程序
  也一样）。所以这台机器上没有 profiler 可用，engine 自己的探针是唯一仪器。
- **启动开销（本机实测，.run/launch_overhead.cu）**：eager 一次 launch 2.45–2.49 µs
  （host 的 CPU 时间也是 2.49 µs），放进 captured graph 后 0.99 µs。decode 一层
  主卡大约 15 次 launch、helper 侧大约 12 次，所以 launch 开销一层约 70 µs、
  一个 token 约 2.8 ms（15%）。**这是 host 那 282 µs/层里最大的一块可合并项。**
- **每层的 CUDA runtime 调用数（LD_PRELOAD 计数，.run/api_count.c，2026-10-25）**：
  3068 次/token = **76.7 次/层**（launch 31.6、memcpy_async 12.5、stream_sync 3.8、
  set_device 11.2、get_device 15.6、event_record/query 各 1）。按 2.45 µs 一次
  launch 算，光 launch 就 77 µs/层，device 切换 27 次又是几十 µs。**host 的 282 µs
  里有大约一半是 runtime 调用的开销。**
- keep-resident（把热专家搬回主卡）：decode 51.8 → 45.3 tok/s = -12.5%，已关。
- HELPER_RING=2：38.9 vs 41.9 = -7.16%，未合。
- 主卡 FP16 expert GEMM：expert 权重是 K-quant，PCIe 313 µs/expert vs FP16 90 µs。
- P2P 全对不可用（12 对 cudaDeviceCanAccessPeer 全 0），跨卡只能走 pinned host。
- up-leg 在 decode 上只有 24 KiB/token，减半省 4.7 µs，不值得。
- --bench-tg 从不跑 spec_step，MTP 速度只能在 --tokens 路径上读。

## 4. 方向（按本轮量到的东西重排）

### D1（第一名，已落地并量过）：把 host 每层 282 µs 的活砍下来

MEASURED 的依据：host_own 11.4 ms/token = 62%。GPU 的时间线 427 µs/层里只有
199 µs 在算，228 µs 在等 host。

**已落地的一条（STRATA_Q36_HELPER_PARALLEL_MARSHAL=1）**：三张 helper 的 begin()
与 finish() 各自搬到自己的 worker 线程上跑（主线程做 helper 0），三次派发重叠。
- MEASURED（192.168.1.98，2026-10-25，tg128 三趟）：
  control（bin/strata-q36.f16cuda3）54.5 / 54.6 / 54.5 tok/s；
  新二进制旗标关 55.0 / 55.2 / 55.1（默认路径没动）；
  **新二进制旗标开 60.2 / 60.5 / 60.3 tok/s = 比 control 快 10.4%**。
- 门：旗标开关生成的文本逐字节相同（`gen:` 行一致）——改的只是派发的墙钟。
- 每次 launch 在本机 2.45 µs（.run/launch_overhead.cu），一次 begin 约 12 次
  CUDA runtime 调用，所以一次 begin 约 38 µs 的 host 时间，本来三张卡可以同时做。
- MTP 侧同一面旗标（--tokens 路径，同一个二进制开/关，D=0..3）：
  **53.5 → 59.2 / 64.4 → 71.5 / 76.3 → 81.5 / 74.2 → 79.7 tok/s**，
  每档的接收数不变（31/33、41/44、47/54）。甜点位从 D=3 挪到 **D=2**（81.5），
  因为 verify 窗口便宜了以后，深一档的草稿不再划算。
- 剩下的：begin/finish 之外的 host 活（mixer 56 µs/层、local 22、cpu 9）与
  每层 76.7 次 CUDA runtime 调用里的其余部分。下一个方向：把主卡每层的 kernel
  序列 capture 成 graph（一次 launch 2.45 µs，graph 里 0.99 µs），但 capture
  不能跨 drain 与 helper 的同步，所以边界只能落在 drain 两侧。

现在还差一件事：282 µs 具体花在哪五段里（mixer / begin / local / cpu / finish）。
本轮已经加了五个计数器（d_h_mixer_ms / d_h_begin_ms / d_h_local_ms / d_h_cpu_ms /
d_h_finish_ms），下一次上机就能读出来。下面按已知的量级排可能性：

1. **launch 开销（约 70 µs/层，占 282 的 25%）**：把主卡每层的 kernel 序列
   capture 成一个 graph（graph 里不能有 drain 和 helper 的同步，所以 graph 的
   边界只能落在 drain 两侧：pre = mixer+router+D2H，post = local+shared）。
   收益的上限：host 的 70 → 约 10 µs/层，GPU 侧 15 × 1.5 = 22 → 15 µs/层，
   合计约 65 µs/层 = 2.6 ms/token = 14%。风险：capture 的边界必须精确，
   capture 里不能出现 host 阻塞（drain、helper 的 finish 同步）；默认路径不动
   （新旗标 opt-in）。
2. **helper 的 begin/finish（每层 3 次 begin + 3 次 finish）**：begin_slot 的
   主机侧有 5 次 memcpy、一次 DeviceScope（cudaSetDevice）、2–3 次 API 调用和
   几次 launch；finish 有 D2H + host 侧 scatter + 等 helper 的流。能不能少一次
   API 调用、少一次 cudaSetDevice，量一下 begin 的 5 个 memcpy 里有没有大的。
   收益未知，等五段拆分出来。
3. **CPU 池那段（tile 的 cpu_pool 段 51 µs/层）**：decode 里 CPU 池其实一个
   专家都不算（报告「0 on the CPU pool in 0 ms」，因为 40 槽 VRAM + 3 张
   helper 覆盖了全部 256 个专家），但 moe_hybrid_cpu 仍然每层做一遍 spin、
   建 jobs map、清向量。n_miss == 0 时能不能整段跳过（它已经有 early return，
   但 spin 和簿记在 return 之前）。
4. **route_group 的 host 侧分组**：decode 的 hy_rows 路径下它把 8 个 entry 按
   专家分组，全在 host 上做。

怎么量：五段计数器（已加）+ tg128 墙钟两跑对比。停：五段里没有任何一段
占 host_own 的一半（>140 µs/层），就换方向。

风险：低（全是调度，不改算术）。旗标关时默认路径必须逐字节不变。

### D2：MTP 链的每步结构（草稿头的 515 MiB 与每步的 host 往返）

- **草稿头**：head_draft 每条草稿步读一遍打包的 Q8_0 head（248320 × 2048 =
  515 MiB），按本机 q8_sm60 的 433 GB/s 算是 1.2 ms/步，D=3 一轮 3 步 = 3.6 ms。
  一轮（MTP 的整个投机轮）在 74.7 tok/s 下约 25 ms，所以草稿头占一轮的 14%。
  两条路：(a) --draft-vocab 取子集（配置项，0/65536/8192/4096；65536 量过是
  中性的，更小的没量；风险是「前 N 个 token id」的语义会让低频 id 被拒收，
  门是 dp_ok 总和不能掉）；(b) 草稿头用一份低精度的副本（Q4_K = 254 MiB），
  草稿只需要 argmax，精度差一点只影响草稿被不被接受，不影响 commit 出来的
  文本——但会改接收率，量。
- **每步的 host 往返**：默认链每条草稿步一次 H2D + 一次 D2D + 一次 D2H +
  一次 cudaStreamSynchronize（D=3 时 2 次）。STRATA_Q36_MTP_MAPPED=1 能一次
  提交全部链式步，但本机 INERT：MTP 块自己的层是 hybrid（helper_layer 拒绝
  li >= n_layers，布局表只覆盖 40 主层，expert_layout_set_native 检查
  fmt.size() == n_layers，41 != 40 直接拒）。把 MTP 层加进布局表，256 个专家
  摊到 3 张 P104 是每张约 85 个 × 1.88 MiB = 160 MiB（每张现在占 5.14 GiB，
  8 GiB 装得下）→ lhy(n_layers) 转假 → mapped 链活 → 每步的 host 往返消失。
  顺带 MTP 层自己的专家从 P100 的 sm_60 4×vmad 回退搬到 P104 的真 __dp4a。
  收益：两个静态估计叠加（每步 1–3 ms 的 host 往返 + 专家那条 2.31× 的 dot
  差距），都没量过。**先量：MTP 块自己的层现在每步花多少。**
- **MTP 侧还没有 kernel 级计时**：草稿的时间花在哪只有墙钟和 --mtp-calibrate
  的整步 ms。加一个 STRATA_Q36_MTP_TIMING（只插 1 步，8 个共享 event 槽）。

怎么量：--tokens 路径，--mtp 3，MTP-DEPTH 的 dp_tested/dp_ok 必须不变
（这是调度改动，不许改哪条草稿被接受）。

### D3：helper 的收割顺序（0.2–0.5 ms/token，小）

helper_finish 与 moe_helpers_wait 按固定 r=0,1,2 顺序 finish，最后收的那张卡
背所有卡的剩余算术：ms_wait 0.22 / 0.26 / 0.43–0.55 ms/token。改法：按
ms_begin 降序收，或每卡 cudaStreamQuery 轮询、谁好先收谁；另一条：把那条
eparts H2D 拆成三条并发。**前置：每层每 helper 的行数直方图**（imbalance 复核
自己定的第一步，一直没打；位置：moe_hybrid_cpu 自旋后 h_ids 已在 host）。

停：ms_wait 摊不匀，或 tg128 落在 0.4% 噪声带里（2026-10-20：两连跑
49.1 / 49.3）→ 撤。

### D4：decode 下腿改 q8_1（1.5%）

helper 的下腿每层每 helper 8 KiB（1 token × 2048 × 4 字节），40 层 × 3 张 =
960 KiB/token，在 0.85 GB/s 的 Gen1 x4 上是 1.1 ms，其中 helper 等它到齐才能算，
所以约 9.4 µs/层在关键路径上（0.38 ms/token）。主卡侧先压成 q8_1（2.03 KiB）
→ 2.4 µs，省 0.28 ms/token（1.5%）。风险：数值路径变了，主卡侧量化必须与
helper 的 quantize_q8_1_rows 逐字节同才有合并资格。

### D5：Q8_SM60 的 blocks-per-SM 旋钮（零代码，先量）

STRATA_Q8_SM60_BPS 默认 4，是在 Flash-Next 的形状上调出来的。q36 的形状不同
（n_in 2048/4096，n_out 8192/4096/512/640/2560/248320），小矩阵只填到 64 个
block（180–224 GB/s）。扫 2/3/4/6/8/12 六个值，看 tg128 有没有动。

### D6：dense 侧的小矩阵（0.3–0.5 ms/token，低）

q8_sm60 的 grid = min(groups, SM 数 × BPS)，n_out=512 的 shared expert 只填
64 个 block。用 BPS 或别的手段把小矩阵填满。量级不大，排最后。

## 5. 下一个动作

1. 上机读五段拆分（mixer / begin / local / cpu / finish），定 D1 的哪一条值得做。
2. 同时跑 STRATA_Q8_SM60_BPS 的扫描（D5，零代码）。
3. MTP 侧加 STRATA_Q36_MTP_TIMING，量草稿步里 MTP 块自己的层和草稿头各占多少
   （D2 的前置）。
4. 草稿头的 vocab 子集扫一遍（门在 dp_ok）。
5. MTP 层进 helper 布局表（VRAM 装得下才做：每张 P104 约 160 MiB 的增量）。
