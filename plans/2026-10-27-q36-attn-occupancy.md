# q36 的 attention 核：占用率**不是**杠杆（2026-10-27）

## 为什么要做这个

§9 复核（`plans/2026-10-27-q36-decode-next-lever.md`）的结论：decode 最大的那块是 attention 的核，
而它**不是带宽受限**：

| | 值 | 出处 |
|---|---:|---|
| 默认核 `k_attn_part` @32K decode cell | 14.274 ms/token | MEASURED 探针（`STRATA_Q36_MIXER_SPLIT=1`）：`plans/2026-10-26-q36-kv-stream.md:213` |
| GQA 共享核 `k_attn_part_gqa` @同一处 | **4.787 ms/token** | 同上：`:299` |
| 流量（读一次）| **381 MB/token** | DERIVED：528 B/head/cell（K 和 V 各 264 B = 256 个 int8 码 + 4 个 fp16 scale）× 2 head × **32768 cell** × 11 层 |
| P100 的本地读带宽地板 | ~410 GB/s | MEASURED（q8 GEMV）：12h 计划 :35 |
| 381 MB ÷ 410 GB/s | **0.93 ms** | DERIVED ⇒ 共享核离带宽地板差 **5.2 倍** |

⇒ **缺的不是流量，是并发度。** 共享核把流量砍到 1/8，只换来 **3.03 倍**的时间。

## 这一轮做的改动：queries 与 scores 共用一份共享内存

`qs[GB][HD]` 和 `sc[GB][kChunkCells]` **形状相同、生命周期不重叠**（queries 在 K 点积之后就没用了，
scores 在那之后才写）⇒ 让它们共用一份字节：

```
__shared__ float smem[GB * HD + (VI8 ? (HD / kKVG) * kChunkCells : 1)];
float* const vsc = smem + GB * HD;   // V 的 scale，4 KB
float* const sc  = smem;             // scores 住在 queries 的字节里
```

⇒ 共享内存 **20 KB → 12 KB**，多一道屏障（scores 覆盖 queries 之前，所有线程必须读完 queries）。

## 结果：**没有加速**（MEASURED 2026-10-27，`.run/R-smem-20261027.sh`）

两个二进制从同一棵树编出来（`base` = HEAD，`smem` = HEAD + 上面那处共用）：

| | 共享内存 | 寄存器 | **tg64（tok/s）** |
|---|---:|---:|---:|
| `base`（HEAD）| **20480 B** | 58 | **51.4 / 51.7** |
| `smem`（共用之后）| **12288 B** | 60 | **50.7 / 50.7** |

（`--bench-pp 32768 --bench-tg 64`，ctx 40960，peer 全开，`--kv int8`，旗标 `STRATA_Q36_ATTN_GQA=8`；
共享内存和寄存器是 ptxas 自己报的（`-Xptxas=-v`），不是推的。）

⇒ **占用率从 3 block/SM 提到 4 block/SM（37.5% → 50%），速度反而掉了约 2%。**
⇒ **所以这个核不受占用率限制**（至少不是被共享内存卡住的那种）——**这条把「占用率」这条路关掉了**：
变体 B（把 `m`/`l` 挪进 shared、降寄存器）也就没有意义了。
⇒ 顺带一条解释：共用之后多了一道屏障，它把 K 点积那段**串行化**了
（写 scores 之前所有线程都得读完 queries）⇒ 拿到的并发度被这道屏障吃掉。
⇒ **bit 精确性是保住的**（旗标开着的路径 token 与旧二进制逐字节相同，见下面的门），
**但没有收益 ⇒ 这个改动不进仓库**（HEAD 的树就是现在的树）。

## 门（都过了，但没有收益）

| 门 | 结果 |
|---|---|
| `base` 的默认路径 | `gen: 220 16 15 15 15 20332 1740 369`（= 出货基线）✓ |
| `base` 旗标 GQA=8 | 同上 ✓（GQA 路径本来就是逐字节相同）|
| `smem` 默认路径 | 与 `base` 逐字节相同 ✓ |
| `smem` 旗标 GQA=8 | 与 `base` 逐字节相同 ✓（**这个改动是位精确的**）|

⇒ 位精确但不快 ⇒ 不合。

## 那这个核到底卡在哪（两个负结果 + 一个对照）

读代码 + ptxas + 上面这个负结果能说的：

- **不是带宽**：381 MB/token 对 410 GB/s 的地板是 5.2 倍的余量（实测只用 **86.4 GB/s**）。
- **不是占用率**：3 → 4 block/SM 没有变化。
- **不是算力**：5.9 G ops/token ÷ 4.412 ms = 1.34 T ops/s，P100 的 FP32 峰约 10 T ops/s ⇒ 13%。
- **⇒ 剩下的是指令流本身**：这个核每个线程大约 7500 条指令，其中
  **V 段占 8/9**（256 个 cell ×（1 次全局字节读 + 1 次 scale 读 + **8 次 score 读** + 8 mul + 8 fma））
  ⇒ 每线程 **2048 条 shared 读**（`sc[g][k]`，g=0..7）是大头；
  K 段只有 32 次装载 + 约 800 条指令。
- **把 scale 挪到 FMA 的另一边**（`fmaf(q, c8[x]*sg, a[g])`、`fmaf(sc[g][k], wr[k]*vk, acc[g])`）
  能把乘法从 2048 次降到 256 次，**但那会破位精确**（两个写法不是同一个舍入）
  ⇒ 那是一条「不精确」的路，和 Quest 一样要单独标。

### 对照：int8 路径 vs fp32 路径（MEASURED 2026-10-27，`.run/R-kvprobe-20261027.sh`）

同配置（ctx 40960、`--bench-pp 32768 --bench-tg 64`、peer 全开、`STRATA_Q36_ATTN_GQA=8`、
`STRATA_Q36_MIXER_SPLIT=1`），只换 `--kv`：

| | **attention ms/tok** | 流量/token | 实际带宽 | GDN ms/tok | merge | **tg64** |
|---|---:|---:|---:|---:|---:|---:|
| `--kv int8` | **4.412** | 381 MB | 86.4 GB/s | 4.090 | 0.422 | **49.9 / 50.0** |
| `--kv fp32` | **30.402** | 1.52 GB | 50.2 GB/s | 4.129 | 0.422 | **21.6** |

⇒ fp32 慢 **6.9 倍**，而流量只多 4 倍 ⇒ **它的实际带宽反而更低** ⇒
**fp32 路径不是被 DRAM 卡的，是被它自己的指令流卡的**：
它的内层是 `for g { for i { a += qs[g][i] * kp[i] } }` ⇒ **同一个 cell 的 K 被读 8 遍**
（每个 head 一遍）⇒ 每线程 2048 次全局读，而 int8 路径只有 32 次。
⇒ **所以这个对照没有把「算术 vs 带宽」分开**（两条路径差的不只是字节数）
⇒ 但它钉住了一件事：**int8 路径的 4.412 ms 里没有 DRAM 的账**（86.4 GB/s，离 410 GB/s 很远）。

### 下一个实验（零改码，已排上）：**Quest + GQA 的速度**

Quest 的旧账「~13 ms/token 的 attention 侧开销」是在**旧的 8 倍冗余读**的核上量的。
现在 GQA 共享形状下，稠密读只要 4.412 ms/token；而 Quest 在 k=8 时只留 8/128 个 chunk（6.25% 的 cell）
⇒ 如果 Quest 的开销主要在旧核的掩码上，**Quest + GQA 会是净赢**，
而 32K 上 k=8 的输出**逐字节等于稠密**（96 token 实测）⇒ 又快又准。
脚本 `.run/R-questgqa-20261027.sh`（GQA=8 × QUEST=0/8/16/32，加 GQA=0 × QUEST=8）。

## 还没做的变体（按新证据重排）

| | 改动 | 位精确 | 预期 | 判断 |
|---|---|---|---|---|
| ~~D~~ | queries 与 scores 共用共享内存 | 保持 | 占用率 37.5% → 50% | **实测无收益（−2%），不采纳** |
| ~~B~~ | 把 `m[8]`/`l[8]` 挪进 shared、降寄存器 | 保持 | 占用率再上一档 | **连带作废**：占用率不是杠杆 |
| E | K 的 int8 读改成 `uint4`（16 B 一次而不是 8 B）⇒ 装载指令少一半 | 保持 | 若卡在装载 ⇒ 有收益 | 待测 |
| F | 把 scale 挪到 FMA 的另一边（乘法 2048 → 256 次）| **破** | 若卡在算术 ⇒ 大收益 | 破位精确，要单独标成不精确路径 |
| C | 两棵 8 级树改成 warp 内 shfl + 跨 warp 树 ⇒ 少约 10 道屏障 | **破** | 约 2-4% | 门值要重发 |
| — | 把一个 chunk 的 256 个 cell 拆到多个 block | 要重写 merge | 尾波（4.57 block/SM）| 大 |

## 另一条路：helper 路径（复核过，结论是**先别做**）

`finish` 4.24 ms/token 的性质：**同步税，不是带宽**（decode 每 token 只下 ~10 KiB/helper，
纯传输 0.007 ms；helper 的整窗拷贝在 zero-copy 下是 0）。

| | 改动 | 位精确 | 预期 | 工时 |
|---|---|---|---|---|
| V1 | drain 的 `cudaStreamSynchronize` 不再带 `DeviceScope`（跨卡 set_device 实测 2 µs/次）| 保持 | 0.3-0.5 ms/token | 1 h |
| V2 | 设备侧 scatter（读 helper 的 mapped 内存）替掉主机侧 `memcpy` + 64 KiB H2D | 保持 | 0.1-0.2 ms/token | 3-4 h |
| V3 | `cudaLaunchHostFunc` 让三个 drain 互不串行 | 保持 | 4.24 → 1.5-2.5 ms | 大 |

⇒ **但它在关键路径上吗？** ctx 2048：墙钟 = max(mixer 5.65, helper 4.24, dense 3.7) ⇒
helper 减半不改变墙钟。ctx 40960：helper 的 4.24 **完全被盖住** ⇒ **这一轮不做 helper 路径。**
（出处：`plans/2026-10-27-q36-decode-next-lever.md` §三条各自是什么。）

## 不知道的

- **这个核到底卡在哪**：上面那个 int8-vs-fp32 的对照是第一个直接证据（已排上）。
- **PTXAS 的两个数**（58 寄存器 / 20480 B）现在是实测的（ptxas 自己的报告），不再是推的。
- **GQA 共享形状在 ctx 40960（而不是探针那一趟的 131072）的核时间**没量过。
