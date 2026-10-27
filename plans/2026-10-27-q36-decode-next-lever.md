# q36 decode 的三个大块：复核与下一个杠杆（2026-10-27）

## 为什么复核

12 小时报告 §9 的第二条（`plans/2026-10-26-q36-12h-report.md:223-227`）写：

> 真正的三个大块是 attention 的读（32K 上约 25 ms/token）、helper 的 PCIe 往返
> （…`finish` 4.24 ms/token）、专家 GEMM 的带宽地板 3.7 ms/token。

三个数都要核：**25 ms 是哪来的**、**4.24 是什么**、**3.7 是不是「专家」的**。
（PCIe 那一处已经修过一次，提交 `617b2753`：0.58 GB/s 是引擎那条路的有效速率，不是链路的带宽。
但那一修把 **prefill** 的数放进了 **decode** 的一条里 —— 这次一并改掉。）

复核方法：三个独立读者各查一个数（attention 的读 / helper 的往返 / dense 地板），
只读代码与盒子上的日志，每条结论带 file:line 或日志行；每个数标 MEASURED / DERIVED。

## 结论先说

1. **这三个数不能相加**：它们在不同的资源上，而且有包含关系
   （3.7 ms 那块在 mixer 的 5.65 ms GPU 段里面，4.24 ms 与 GPU 的活是重叠的）。
   ⇒ 墙钟是**重叠部分的最大值**，不是和。
2. **「attention 的读 32K 上约 25 ms/token」没有测量支撑**：`git log -S` 只有 §9 那一处写过它。
   探针实测是 **默认核 14.274 ms/token、GQA 共享核 4.787 ms/token**（32K decode cell，ctx 131072 那一趟）
   ⇒ 25 高了约 75%。
3. **「专家 GEMM 的带宽地板 3.7 ms/token」标错了**：3.7 ms 是 **dense（非 routed）权重**的地板
   （1.51 GB ÷ 410 GB/s），源码自己写的是「dense 权重地板」。
   routed 专家那一侧（每 token 580 MiB）在当前 peer 配置里 **88% 在 helper 上** ⇒ 走 0.8 GB/s 的 PCIe，
   已经算在「helper 往返」那条里 ⇒ 单列一条是重复计数 + 归错地方。
4. **「helper 的 PCIe 往返」是同步税，不是带宽**：decode 每 token 下行只有约 100 KiB
   ⇒ 纯传输 0.04 ms，而 `finish` 是 4.24 ms ⇒ **差 100 倍**，差在 3 条 helper 流上的
   阻塞 `cudaStreamSynchronize` + 主机侧 scatter + 整窗 H2D 合并。
   ⇒ 少字节没用，要少同步。
5. ⇒ **下一个杠杆是 attention 的核本身**（它占墙钟最多），而且它现在**不是带宽受限**：
   GQA 共享形状把流量砍到 1/8，只换来 **3.03 倍**的时间（寄存器涨到 40、block 数少了 8 倍）⇒
   缺的是**并发度**，不是流量。

## 三条各自是什么（复核后）

### 1. attention 的读

| | 值 | 是什么 | 出处 |
|---|---:|---|---|
| 墙钟 @32K（ctx 40960, peer, `--kv int8`）| 28.8 ms/token（GQA 关）/ **19.3 ms/token（GQA 开）** | tok/s 34.4 → 51.8 | MEASURED 盒子日志 `aaoff.out` / `aaauto.out`（12h 报告 §4） |
| attention 核 @32K decode cell，默认核 | **14.274 ms/token** | `k_attn_part` 的核时间 | MEASURED 探针（`STRATA_Q36_MIXER_SPLIT=1`，ctx 131072，`--bench-pp 32768 --bench-tg 32`）：`plans/2026-10-26-q36-kv-stream.md:213` |
| attention 核 @32K decode cell，GQA 共享核 | **4.787 ms/token** | `k_attn_part_gqa` 的核时间 | 同上：`:299`（merge 0.549 → 0.548 ms/token） |
| 流量（读一次）| 470 MB/token | 528 B/head/cell（int8：256 码 + 4 个 fp16 scale）× 2 head × 40960 cell × 11 层 | DERIVED（`q36_engine.cu:423` kKVG=64；12h 报告 :49 的 528） |
| 8 倍冗余 | 默认核每份 KV 被读 8 次 | grid `dim3(nc, nt*H)`，16 个 query row 里 8 个映射到同一个 `kvh`，地址只依赖 `kvh` | 代码：`q36_engine.cu:4145` + `:550-551` |
| 默认核的有效读带宽 | ≈ 211 GB/s | 3.02 GB ÷ 14.274 ms | DERIVED；P100 的本地地板 ≈ 410 GB/s（q8 GEMV 实测，12h 计划 :35）|
| 共享核 | 流量 /8，时间只 /3.03 | block 数 `nc*16` → `nc*2`，寄存器 40/线程 | MEASURED 探针 + `kv-stream.md:305-307` |

⇒ **「25 ms」是流量×假定带宽的推算，不是测量**；正确的说法是
「默认核 14.3 ms/token，GQA 共享核 4.8 ms/token（探针实测，32K decode cell）」。

### 2. helper 的往返

| | 值 | 是什么 | 出处 |
|---|---:|---|---|
| `finish` | 4.24 ms/token（ctx 2048 那趟探针 4.088） | 主机在 `moe_helpers_wait` 里等 3 个 helper 排空 + 主机把 compact 行 scatter 回 `helper_out` + 整窗 H2D 合并 | MEASURED `.run/brk_short.err`；代码 `q36_engine.cu:4548-4551`、`:5146`、`:5176-5177` |
| 其中真正的传输 | **≈ 0.04 ms/token** | decode 每层每 helper 只回 2-3 行（≈ 32 KiB）⇒ 32 KiB ÷ 0.81 GB/s | DERIVED：`remote_experts.cpp:565` 的 `compact_bytes` + 链路的 0.81 GB/s |
| 差的 4.2 ms | 阻塞同步 + 主机 scatter | `RemoteExperts::finish()` → `drain()` → `cudaStreamSynchronize(stream_)`（`remote_experts.cpp:633`）+ 主机 `std::memcpy`（`:654-655`） | 代码 |
| 链路的带宽 | H2D/D2H 都 0.79-0.81 GB/s（gen1 x4，上限 ~1.0） | 直测 | MEASURED `.run/pcbw2.cu` 2026-10-27 |
| **0.58 GB/s 是什么** | **prefill** peer 路径的下行（512-token 窗、每 helper 1152 行、fp16 4.72 MB/层） | 两次引擎跑的差分 | DERIVED `plans/2026-10-25-q36-peer-prefill-push.md:247-248` ⇒ **对 decode 没有解释力** |

⇒ decode 的 helper 路径是**延迟/同步受限**，不是带宽受限 ⇒ 少字节没用，要少同步。
（prefill 侧才值得减字节，而且 `STRATA_Q36_PEER_I8=1` 已经做了，21.7 → 17.3 ms/层，+25.7%。）

### 3. dense GEMV 的地板（原来写成「专家」）

| | 值 | 是什么 | 出处 |
|---|---:|---|---|
| dense 权重 | 1.51 GB/token | 40 层的 attn_qkv 17 MiB + attn_gate 8.5 + ssm_out 8.5 + 共享专家 3×1.06（Q8_0） | DERIVED `plans/2026-10-25-q36-decode-mtp-directions.md:76-78` |
| 本地地板 | ≈ 410 GB/s | P100 的 q8 GEMV 实测（row-dot 402-433） | MEASURED `plans/2026-10-25-q36-peer-prefill-push.md:584-591` |
| **3.7 ms/token** | 1.51 GB ÷ 410 GB/s | **dense 块的地板**，墙钟真的付（它在 mixer 的 5.65 ms GPU 段里） | DERIVED |
| routed 专家的字节 | 580-588 MiB/token | 8 专家 × 40（+MTP）层 × 1.8125 MiB | DERIVED |
| routed 专家在哪儿 | **88.3% 在 helper 上**（9040/10240），主卡 VRAM 只 3.1%，CPU 池 0 | 当前 peer 配置 | MEASURED 启动行（`hybrid experts: 30 of 256 per layer in VRAM (2.20 GiB)` + 三个 helper 的人数） |
| ⇒ helper 那部分的地板 | 2.08 MB/层/helper ÷ 0.81 GB/s ≈ **2.6 ms/层** | 走 PCIe，不是 HBM | DERIVED `peer-prefill-push.md:615-617` |

⇒ 「专家 GEMM 的带宽地板 3.7 ms/token」**标错了**：3.7 是 dense 的地板；
专家那一侧在当前配置里就是「helper 往返」那条 ⇒ 单列一条既重复计数又归错地方。

## 现在的墙钟（MEASURED，peer 全开，`--kv int8`）

| 配置 | tok/s | ms/token |
|---|---:|---:|
| ctx 2048 | 60.3-60.6 | 16.5-16.6 |
| ctx 40960，默认核 | 34.4 | 28.8 |
| **ctx 40960，GQA 共享核** | **51.8** | **19.3** |

## 下一个杠杆（按证据排）

1. **把 GQA 共享形状开起来**（`STRATA_Q36_ATTN_GQA=a` 或 `=8`）：
   32K 上 **34.4 → 51.8 tok/s（+50%）**，字节不变（探针 + 三道 parity 门）。
   ⚠ 它现在是**未写进用户文档的旗标** ⇒ 要补 `docs/QWEN36.md`（用户看不到 +50% 的开关）。
   ⚠ prefill 的 GQA 路径此前**没有门**；2026-10-27 补了一道：8192 cell 的同一 prompt，
   旧核与共享核 **96 个 token 逐字节相同**（`.run/R-kcurve256-gqa-20261027.sh`）。
2. **attention 核的并发度**（不是流量）：流量已经砍到 1/8，只换来 3.03 倍
   ⇒ block 数少了 8 倍、寄存器 40/线程 ⇒ 缺的是并发。
   可试的方向（都要先量）：把一个 chunk 的 256 个 cell 拆到 2-4 个 block；
   减少 `vsc` 表带来的共享内存/寄存器压力；`GB` 取 4 而不是 8。
3. **helper 的 decode 路径减同步**：`finish` 4.24 ms 里只有 0.04 ms 是传输
   ⇒ 3 条流上的阻塞 `cudaStreamSynchronize`（`remote_experts.cpp:633`）和主机 scatter（`:654-655`）。
   ⇒ 少字节没用。
4. **prefill 的 helper 下行减字节**（与 decode 分开算）：那里才真的带宽受限，
   而且 `STRATA_Q36_PEER_I8=1` 已经做了一半（+25.7%）。

## 不知道的

- **GQA 共享核在 32K 上的核时间**只有探针那一趟（ctx 131072、`--bench-pp 32768`、
  「每层同步一次 ⇒ 那趟的 tok/s 不可比」）⇒ 4.787 ms 是**核时间**，不是墙钟。
- **attention 核在 ctx 40960（而不是 131072）的核时间**没量过。
- **helper 路径的字节在当前配置下的精确数**（每 token 约 100 KiB 是从代码推的，
  没在日志里逐条数过）。
- **Quest 路径那 12.9 ms/token**（12h 报告 §7）还是在 32K 上量的，262144 未量。
