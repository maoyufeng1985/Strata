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

### Quest + GQA 的速度（MEASURED 2026-10-27）

同配置（ctx 40960、`--bench-pp 32768 --bench-tg 64`、peer 全开、`--kv int8`）。
脚本 `.run/R-questgqa-20261027.sh` / `.run/R-questtime-20261027.sh`。

**墙钟（`.run/R-questgqa-20261027.sh`）：**

| | tg64 | vs dense |
|---|---:|---:|
| GQA=8 QUEST=0 | **51.6 / 51.3** | — |
| GQA=8 QUEST=8 | **53.4 / 53.3** | **+3.5%** |
| GQA=8 QUEST=16 | 50.6 / 50.7 | −2% |
| GQA=8 QUEST=32 | 46.4 / 46.4 | −10% |
| GQA=0 QUEST=8 | **53.9 / 54.1** | ≈ QUEST=8 旧核 |

⇒ **只有 k=8 是净赢**。k 再大，读回来的 chunk 多过省下的钱。

**核时间（`STRATA_Q36_MIXER_SPLIT=1` + `STRATA_Q36_QUEST_TIMING=1`，GQA=8）：**

| | attn ms/tok | merge | bounds | select† | tg64 |
|---|---:|---:|---:|---:|---:|
| QUEST=0 | **4.411** | 0.421 | — | — | 49.7 / 49.8 |
| QUEST=8 | **3.803** | 0.270 | 0.105 | 0.408 | 51.0 / 51.3 |
| QUEST=32 | **6.696** | 0.356 | 0.094 | 0.408 | 44.3 / 44.4 |

† `QUEST:` 行的「select+pick」是 event 1→2，止于 `k_quest_sel`；`k_quest_pick` 是 event 2→3，**不在这个数里**（`q36_engine.cu:4024-4099` / `:8756-8768`）。

⇒ k=8：attn −0.608、merge −0.151，另加 select 0.408 + bounds 0.105 ⇒ **净约 −0.25 ms/tok**，和墙钟 +3.5% 对得上。
k=32：attn 反而涨到 6.696（读 32/128 chunk + 掩码开销）⇒ 墙钟输。

### 质量：64K 上的 k 曲线（MEASURED，GQA=8）

`.run/R-kcurve-fast-20261027.sh`，`--tokens-file /tmp/ctx64k.ids`，96 token：

| k | vs dense |
|---|---|
| 8 | 第 **5** token 分叉 |
| 16 | 第 **5** token 分叉 |
| 32 | 第 **7** token 分叉 |
| 64 | 第 **5** token 分叉 |

32K 上 k=8 曾逐字节等于稠密；**64K / 128K 上同一个 k 已经早分叉**。

| ctx | k | fork |
|---|---:|---:|
| 32K | 8 | 无（96 token 逐字节同） |
| 64K | 8/16/32/64 | 第 5/5/7/5 token |
| 128K | 8 | 第 **3** token（1227 s） |
| 128K | 16 | 第 **2** token（1235 s） |
| 128K | 32 | 第 **2** token（1231 s） |

**代码嫌疑（仍成立，但不是唯一病因）：** GQA 共享核用**组内第一个 query head** 的掩码决定整组是否读这个 chunk（旧 `k_attn_part_gqa`），而 `k_attn_merge` 按**每个 head 自己的掩码**合并（`:928-934`）。
若头 0 丢、头 1 留 ⇒ merge 读到从未写过的 `pm`/`pacc`。
`k_quest_sel` 是**每 query head 一行**，旧注释「同组共用同一掩码」错了。

**对照结论（MEASURED 2026-10-27，`.run/R-quest64ctl-20261027.sh`，旧二进制）：**

| 对比 | 结果 |
|---|---|
| g0 dense vs gqa8 dense | **IDENTICAL 96** |
| g0 dense vs g0 k8 | **第 5 token 分叉** |
| g0 k8 vs gqa8 dense | 第 5 token 分叉 |
| g0 k8 vs gqa8 k8 | 第 11 token 分叉（两边都早分叉，但路径不同） |

⇒ **GQA=0 路径本身没错**（dense 对齐）。
⇒ **GQA=0 × k=8 仍第 5 token 分叉** ⇒ 64K 早分叉**主要是 Quest 选块本身不够**，不是单靠 GQA 掩码错位能解释的。
⇒ any-head 修仍是**必要**（代码 bug），但是**非充分**。

**最小修（已推盒编装）：** `q36_engine.cu:677-688` early-return 改成「组内**任一** head 要留才算留」。
`bin/strata-q36.s3` md5=`f05bf102909f1ac955ee4667309fa56b`；短门 parity `gen: 220`。

### 修后重测（MEASURED 2026-10-27，`.run/R-quest-remeas-20261027.sh`）

| 对比 | 结果 |
|---|---|
| fix dense vs OLD dense | **IDENTICAL 96** |
| fix k8 vs g0 k8 | **IDENTICAL 96** |
| fix k8 vs fix dense | 第 **5** token 分叉 |
| fix k8 vs OLD dense | 第 5 token 分叉 |
| fix k8 vs OLD k8 | 第 11 token 分叉 |

臃时：k=0 313 s，k=8 317 s。

⇒ **掩码修生效**（fix k8 == g0 k8）；dense 路径位不变。
⇒ **长 ctx 质量仍限于 Quest 选块**（仍第 5 token 分叉）。

### 验证清单

1. ~~等 k 曲线~~ ✓
2. ~~读对照~~ ✓（走「Quest 本身」支）
3. ~~推盒编译 any-head~~ ✓
4. ~~重跑 remeas~~ ✓（掩码修生效；Quest 本身仍早分叉）
5. ~~SparseEngine 的默认预算映射到 q36~~ ✓（**负面**，见下）
6. **下一步**：变体 F/E/C（Quest 这条线已关，见下）。

### SparseEngine 的默认预算映射（MEASURED 2026-10-27，**负面**）

SparseEngine（`D:\work\src\SparseEngine`）的默认：`sink_keep_tokens=64`、`decode_keep_tokens=4096`、
`recent_keep_tokens=512` ⇒ 共 4672 token；`quest_skip_layers=2`；`quest_chunk_size=16`。

映射到 q36 的 **256-cell chunk** 单位（`kChunkCells = 256`，`q36_engine.cu:256`）：

| SE | SE 原值 | q36 的 chunk 数 |
|---|---:|---:|
| sink | 64 token | 1 chunk |
| recent | 512 token | 2 chunk |
| 合计 keep | 4672 token | 19 chunk（4864 token）|
| skip layers | 2 | 2 |

**注意：SE 的原生页是 16 token，不是 256。** q36 的 256 是 attention 的 split-K chunk；
把 q36 改成 16 的页要重写 `k_attn_part` / `k_attn_part_gqa` 和全部共享内存尺寸，不是改一个常数。

实测（`.run/R-quest-se-20261027.sh`，`bin/strata-q36.mtpse` md5 `2cf4d14d…`，64K/128K，96 token）：

| 臂 | 墙钟 | vs dense |
|---|---:|---|
| SE 默认（19/1/2/2）ctx 65536 | 314 s | **第 7 token 分叉** |
| SE 默认（19/1/2/2）ctx 131072 | 1228 s | **第 5 token 分叉** |
| 显式 19/1/2/2（不设 SE_DEFAULTS）ctx 65536 | 314 s | 与 SE_DEFAULTS **逐字节相同 96** |

⇒ 显式臂与 SE_DEFAULTS 臂**逐字节相同**，说明**映射和接线是对的**；
⇒ 但 SE 的默认预算**依然不等于 dense**（64K 第 7 token、128K 第 5 token 分叉）。
⇒ MTP 深度在这个预算下**没有收益**（mtp=1 与 mtp=3 的 tg64 都在 ~51 tok/s，噪声内）。

**结论：Quest 这条线关闭。** 在实测过的 k 范围里**没有不 fork 的 k**（64K：k=8/16/64 第 5 token、
k=32 第 7 token；128K：k=8 第 3 token、k=16 第 2 token、k=32 第 1 token），
加 k 或加 sink/recent 都不对齐 dense，128K 比 64K 更早分叉。
`src/qwen35/q36_engine.cu` 里这轮的研究脚手架（SE 默认值、`STRATA_Q36_QUEST_SINK`/`_RECENT`/`_SKIP_LAYERS`）
**已全部撤掉**（回到 HEAD），只留下两个**真 bug 修复**：
① `k_attn_part_gqa` 的 any-head 掩码（组内任一 head 要留才算留；旧码用组内第一个 head 的掩码，
   而 `k_attn_merge` 按每个 head 自己的掩码合并 ⇒ 读到未写过的 `pm`/`pacc`），
② `q_ev` 的 `has_mtp` 尺寸（`--mtp` + QUEST_TIMING 会越过 event 向量末尾）。
两处都只在 **Quest 开启**路径上生效，默认路径逐字节不变。

## 同族的另一个实现：kvmem-qw3（2026-10-27 读到）

`https://github.com/kvmem/kvmem-qw3`（Apache-2.0，C++，27★，最近 push 2026-10-13）。
**架构同族**：它的 QW3 跑 `qwen35` 混合模型；Strata 的 q36 模型元数据也是
`general.architecture = qwen35moe`、`full_attention_interval = 4`（40 层里只有 **10 层**
是标准 attention，其余是 DeltaNet 递归层，KV 不随上下文增长）。
论文：`https://arxiv.org/abs/2609.04852`。

**规模不同，结论不能直接搬**：它的效率实验跑在 **RTX PRO 6000 Blackwell 96 GB + 125 GiB RAM**
（`docs/kvmem_performance_evaluation_20260726.md`），`--ctx 655360`、KVMem budget 200K、
CPU tier 64 GiB、block 32 token —— 它把**冷的 KV 块摊到主机内存和 NVMe**，
所以它的问题是「怎么把历史从 GPU 搬出去又搬回来」。
q36 的测试机是 **P100-12 GB + 31 GiB RAM**，没有 NVMe tier 也没有 paged KV，
q36 要解决的是「12 GB 卡上 150K 的上下文怎么读得快」。
**两者只共享「查询条件化的块选择」这一层想法，不共享工程前提。**

它做的正是 Quest 这一族的事：**查询条件化的 KV 块检索**（按当前 query 选历史 KV 块），
但**机制和 Quest 完全不同**，有两点值得记下来：

| | q36 的 Quest | kvmem-qw3 的 KVMem |
|---|---|---|
| 选什么 | decode 时**原地跳块**（被丢的 chunk 不读，位置不动） | 选中的块**重映射**成一个连续窗口，**重烘 RoPE**（de-rotate 再 re-rotate），位置变成「窗口内的位置」 |
| 打分 | 每维 int8 min/max 的**上界**（Quest bound），在 RoPE 后的空间比 | **Mean-K** 索引：每块一个平均 K 向量，在**内容帧**（de-RoPE 到位置 0）比 `q_raw · k̄_raw`（位置不变） |
| 保留 | top-k chunk（+ sink/recent） | top-k 块（+ sink/recent），块大小 32–512 token |
| 结果 | 64K/128K **早分叉** | AgentLongBench 52 条诊断样本：**40/52**，比最好的 RAG（29/52）多 11 条 |

**为什么这个差别重要：** Quest 在**位置不动**的前提下丢块，剩下的块之间隔着一段
「空洞」——被丢掉的位置在相对距离上仍然存在，query 与保留 key 的相对位置对不上模型
见过的分布。KVMem 的做法是把选中的块**压紧**并重烘位置，让模型看到的相对位置仍在
训练分布内（它自己文档里写：*"the active window length stays within the trained regime,
and RoPE is re-baked so relative positions remain in-distribution"*）。
**这可能就是 Quest 在 64K/128K 早分叉的真正原因**：不是「选的块不够多」，
而是「跳块不压紧 ⇒ 位置分布跑偏」。q36 实测「加 k 或加 sink/recent 都不对齐 dense」
与这个解释一致——加 k 只是多读几个块，位置空洞还在。

**但 KVMem 自己也撞在同一堵墙上**（它的 `docs/kvmem_known_issues.md`）：
- **KVMI-007（P2, RESEARCH）**：*"mean-k 会稀释和碎片化长 session 中的相关证据"*；
- **KVMI-012（P0, RESEARCH）**：normal-attention 块重选后，**DeltaNet 的递归状态仍来自原时间线 checkpoint**，
  两者不同步；
- **KVMI-011（P0）**：fp16/fp8 的 K 反复**原地 re-RoPE 会累积漂移**（它用 immutable raw-K 做有界重置）；
- 它自己把 Quest/InfLLM 归为 *retrieval* 一路，把自己的 cumulative-attention 归为 *retention*（H2O 式），
  并承认 retention 掉的历史块**再也回不来**，只有 query-conditioned retrieval 能把它**复活**。

⇒ **结论一致**：块级 KV 检索在长上下文上换速度必然换质量，q36 的 Quest 和 kvmem 的 KVMem
是同一族的两个实现，都还没解决「选得准 + 位置不跑偏」这两件事同时成立。
q36 侧**如果**要再走这条路，值得抄的是 **RoPE 重烘 + 位置压紧**，不是它已经有的上界打分。
但这条线需要一个 **paged KV**（KVMem 的窗口装配靠「页表重排 + 指针别名」，零拷贝），
而 q36 现在是 **dense cell 布局**，没有 page table——**这不是一个局部改动**。

### q36 侧要接 KVMem 式做法的三个硬约束（读代码确认，2026-10-27）

1. **没有 paged KV**：q36 的 KV 是 `int8_t* kc` + `__half* ks` 的 **dense cell 布局**
   （`q36_engine.cu:500-509`：`kp = ((int8_t*) kc) + (kh * ctx + pos) * HD`），
   位置就是数组下标。KVMem 的窗口装配靠「页表重排 + 指针别名」零拷贝搬块，
   q36 要做同样的事，得先把 KV 改成 page table —— **大改**。
2. **K 是 RoPE 烘焙后量化的，没有 raw-K**：`k_attn_prep` 里 RoPE 先作用在 `v` 上，
   然后 `v` 直接量化成 int8 + per-group scale（`:499-509`）。
   KVMem 的内容帧打分（de-RoPE 到位置 0 再比 `q_raw · k̄_raw`）和重烘都要求**未烘焙的 K**。
   q36 要走这条路得**额外存一份 raw-K**（显存代价：150000 cell × 10 层 × 2 KV head × 256 dim）。
   KVMem 自己也被这件事咬过：**KVMI-011（P0）**——fp16/fp8 的 K 反复原地 re-RoPE 会累积漂移，
   它靠 immutable raw-K 做有界重置。
3. **GDN 没有块级状态快照**：q36 的 30 层 GDN 是 fused 递归层
   （`k_gdn_seq` / `k_gdn_seq_cols`，`:1420` / `:1497`），只带一个整体状态，
   没有「块边界状态」的概念。KVMem 的 DeltaNet 召回打分器
   （`E_j = S_j − a_j·S_{j-1}`，每块存净编辑和累计 log-decay）
   在 q36 上需要**在 GDN 核里加块边界快照**（它自己说这条要新加 kernel 侧通道：
   *"forces the qw3 delta kernel when capturing (the ported warp kernel has no snapshot side-channel)"*）。
   而且 q36 的 `k_gdn_seq` 是**一个 launch 跑整个 prompt chunk**，块边界要在 chunk 内再切。

⇒ **q36 侧接 KVMem 式做法 = 三个大改（paged KV / raw-K / GDN 快照），不是一个局部补丁。**
这与「Quest 这条线关闭」的判断一致：**q36 现在不做块级 KV 检索**。
KVMem 的价值在于它证明了这条路的**上限**（AgentLongBench 40/52，比最好 RAG 多 11 条）
和它的**代价**（mean-k 稀释证据 KVMI-007、re-RoPE 漂移 KVMI-011、DeltaNet 状态不同步 KVMI-012）。

## 还没做的变体（按新证据重排）

| | 改动 | 位精确 | 预期 | 判断 |
|---|---|---|---|---|
| ~~D~~ | queries 与 scores 共用共享内存 | 保持 | 占用率 37.5% → 50% | **实测无收益（−2%），不采纳** |
| ~~B~~ | 把 `m[8]`/`l[8]` 挪进 shared、降寄存器 | 保持 | 占用率再上一档 | **连带作废**：占用率不是杠杆 |
| E | K 的 int8 读改成 `uint4`（16 B 一次而不是 8 B）⇒ 装载指令少一半 | 保持 | 若卡在装载 ⇒ 有收益 | **位点已钉，未改码**：`k_attn_part` `:582-587`、`k_attn_part_gqa` `:705-712` 现用 `uint2`/`i0+=8`。改 `uint4`/`i0+=16`/`x<16`；`kKVG=64` 内 scale 同、FMA 序同 ⇒ 位精确。但计划正文钉住 **V 段占 8/9**，K 装载本就小 — 杠杆弱，等盒空再测 |
| F | 把 scale 挪到 FMA 的另一边（乘法 2048 → 256 次）| **破** | 若卡在算术 ⇒ 大收益 | **位点已钉，未改码**：K `fmaf(q*sg, c8, a)` → `fmaf(q, c8*sg, a)`（`:587` / `:712`）；V `fmaf(sc*wr, vk, acc)` → `fmaf(sc, wr*vk, acc)`（`:767`）。破舍入，须单独标不精确路径；V 段是大头，杠杆比 E 大。等盒空再测 |
| C | 两棵 8 级树改成 warp 内 shfl + 跨 warp 树 ⇒ 少约 10 道屏障 | **破** | 约 2-4% | **位点已钉，未改码**：`block_max`/`block_sum` `:264-284`（one-head 走这里 `:595/:598`）；GQA 就地树 max `:737-742` + sum `:779-783`（每级一道 `__syncthreads`，两棵共 ~16 道）。同文件已有 shfl 样板（`:375/:505/:822/:886`）。改 warp shfl + 8-slot 跨 warp 树可砍大半屏障，但 sum 树换序破位精确，门值要重发。杠杆次于 F；等盒空 |
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

- **这个核到底卡在哪**：int8-vs-fp32 钉住「不是 DRAM」；变体 E/F 位点已钉，未改码未测。
- ~~**64K/128K 早分叉是不是 GQA 掩码错位**~~：对照钉主因 Quest；any-head 已推盒，remeas：`fix k8 == g0 k8`，仍第 5 token ≠ dense。
- ~~**128K 上 k 曲线**~~：齐。k=0/8/16/32 fork 无/3/2/2（1227/1235/1231 s）。
- ~~**GQA 共享形状在 ctx 40960 的核时间**~~：已量（QUEST=0 → attn 4.411 ms/tok）。
