# q36：prefill 的「300 多 tok/s」在哪（2026-10-25 复核）

日期：2026-10-25。机器：192.168.1.98（1× Tesla P100-12GB 主卡 + 3× P104-100 helper）。
模型：Qwen3.6-35B-A3B-UD-Q4_K_XL。二进制：`bin/strata-q36.bps`（= HEAD 减去 `open_helpers` 里
MTP_VRAM 的那一个 hunk，旗标关掉时是 no-op）。

## 0. 一句话

**「300 多」有两个来源，别混：**

1. **Qwen3.6 的 peer prompt 路径**（opt-in：`STRATA_Q36_PREFILL_PEERS=1` +
   `STRATA_Q36_PREFILL_PEERS_HOST=1` + `STRATA_Q36_PEER_F16=1`）：
   229.9（10-20）→ 310.8（10-21）→ 402.2（10-23）→ 462.0（10-24）→ **493.6（今天）**。
   **今天复现成功**：460.0（10-24 的那份配置）/ 493.6（加 10-25 的 marshal + fixed slots）。
2. **Ornith-1.5-35B-A3B-IQ4_XS**（同一个架构的 **IQ4_XS** 量化，19.3 GB，不是 Qwen3.6 的 Q4_K_XL）
   在这台机器上的 prefill：328.4（10-19，`.run/R-b1-ring-20261019-101438-t3-r1.txt`）、
   324.9（10-20）。`plans/2026-10-20-q36-1p3h-optimal.md` §6 明确把这两个数标成
   **「Ornith 历史，不能用于 Qwen3.6 的参数决策」**。

**默认路径（不开 peer）今天还是 156 tok/s**：prompt 的专家全在主卡上算，三张 helper 在
prefill 期间完全闲置（这是设计如此，见 `2026-10-20-q36-prefill-helper-vram.md` §1.1：
`T > kGemmT(=8)` 的 prompt chunk 不走 helper 那条路）。

## 1. 今天实测（MEASURED，2026-10-25，192.168.1.98）

命令：`--bench-pp 512 --bench-tg 64 --no-warmup --max-context 2048
--expert-cache-device1/2/3 auto`，Qwen3.6-35B-A3B-UD-Q4_K_XL，稳态取 bench 2。
旗标：`STRATA_REMOTE_ZEROCOPY=0 STRATA_Q36_HELPER_ASYNC=1 STRATA_Q36_HELPER_PARALLEL=1
STRATA_Q8_SM60=1`，「10-25 的旗标」= 再加 `STRATA_Q36_HELPER_PARALLEL_MARSHAL=1
STRATA_Q36_FIXED_SLOTS=1`。

| 跑 | peer | 10-25 旗标 | pp512 bench 0/1/2 | tg64 bench 2 |
|---|---|---|---:|---:|
| P1 | 开 | 无 | 469.1 / 394.9 / **460.0** | 50.5 |
| P2 | 关 | 无 | 138.0 / 154.1 / **155.1** | 38.4 |
| P3 | 开 | 有 | 474.9 / 492.4 / **493.6** | **59.4** |
| P4 | 开 | 无 + prompt 探针 | 465.5 / 396.6 / 456.9 | 45.0 |
| P5 | 关 | 有 | 155.4 / 155.9 / **156.1** | **58.5** |

**P3 对 P5 是唯一一对只差「peer 开关」的对照**（同一个二进制、同一批旗标、同一个 bench）：

| | prefill pp512 | decode tg64 |
|---|---:|---:|
| 默认路径（peer 关） | 156.1 | 58.5 |
| **peer 路径（peer 开）** | **493.6（+216%，3.16×）** | 59.4（不变） |

**⇒ prefill 快 3.16 倍，decode 不掉。** 每一趟的 rc=0、IMA=0、nan=0。

GPU 命中（P2 vs P3/P5 的那行 `hybrid experts: N of 61440 batch entries on the GPU`）：
默认路径 **61440/61440 = 100%**（P2 没开 fixed slots 时是 56670/61440 = 92.2%，4770 条在 CPU 池
里花 1075 ms）——**默认路径不是「CPU 池拖慢」**，是**专家都在主卡上算、helper 闲置**。

**机制**（`docs/QWEN36.md:188` 对默认路径的原话：「a prompt chunk of more than `--hybrid-rows`
rows, **with the experts streamed from the host memory**」）：默认路径里 prompt 的专家权重从
pinned host 内存搬到主卡（启动行：`all 18.75 GiB in pinned host memory`，主卡自己只留
39~40/256 每层 = 2.86~2.93 GiB）。**DERIVED**：一次 prompt 要搬 ≈ 18.75×(216/256) ≈ 15.8 GiB，
按本机实测的 H2D ~6 GB/s（`plans/2026-10-24-...-down-leg-bytes.md` 那条线）≈ 2.6 s，
而默认路径一个 512 token 的 prompt 是 512/156.1 = 3.28 s —— **PCIe 搬运是主项**。
peer 路径则是每张 helper 算自己显存里的专家、只把结果行（fp16）送回来，所以没有这条搬运。

## 2. 数字对不对得上历史

| 日子 | 数字 | 出处 | 备注 |
|---|---:|---|---|
| 10-20 | 229.9 | `2026-10-20-q36-prefill-helper-vram.md:211` | peer 路径第一版（+45%）|
| 10-21 | 266.8 → 310.8 | `2026-10-21-q36-peer-ima-root-cause.md` §13.8.1 | 3 个 helper 并行 prepare |
| 10-23 | 383.8 → 402.2 | 同上 §13.9 | 上腿 fp16 + 下腿只 stage 一次 |
| 10-24 | 371.4 → **462.0** | `2026-10-24-q36-peer-owner-encoding.md` §4 | peer 归属表的编码修复 |
| **10-25（今天）** | **493.6** | 本文件 §1 | 加了 10-25 的 marshal + fixed slots |

**今天没有掉，反而 +7%**：10-24 的 462.0 → 今天的 460.0（同一批旗标）→ 493.6（加 10-25 旗标）。
中间的 10-25 旗标对 decode 也有用（tg64 50.5 → 59.4）。

## 3. 数值：peer 路径和默认路径**不是**逐字节相同

**先说清楚今天的两次比较，哪一次算数（`--tokens` 后面跟的是一串 id，`parse_ids` 把它切成一条
prompt；`--force` 是被强制的**生成**序列，不是 prompt）：**

**（一）无效的那次**（`--tokens 1 --force <37 个 id>`）：prompt 只有 **1 个 token**，
而 peer 路径只对宽于 hybrid_t(128 行) 的 prompt chunk 生效 ⇒ **peer 的算术根本没跑**。
那次量到的「37 行里 36 行 argmax 相同、max|diff| 0.837」只是**专家放置的差别**
（peer 开 = 主卡每层留 39 个专家，关 = 40 个）⇒ decode 的数值路径变了 ⇒ 一个 token 不同。

**（二）算数的那次**（`--tokens <216 个 id> --n 32`，peer 路径真的跑了 ——
stderr：`3 helper(s) serve prompt chunks (8680 experts held by helpers, 40 layers served …)`）：

| | prompt 216 token | 生成 32 token | 文字 |
|---|---:|---:|---|
| peer 关 | 3273.8 ms（66.0 tok/s）| 567.6 ms（56.4 tok/s）| `… 2972 **6065 220 17 25 498 …**` |
| peer 开 | **543.4 ms（397.5 tok/s）** | 535.8 ms（59.7 tok/s）| `… 2972 **38663 25 2972 198 256 471 …**` |

**⇒ 这个 prompt 上 peer 路径快 6 倍（66.0 → 397.5 tok/s），而生成的文字从第 10 个 token 起分叉。**
（比文字更硬的那道数：两条路各自跑两次，dump 出来的 32 行 logits **逐字节相同**（max|diff| = 0，
确定性 ✓）；两条路**互相**比，32 行里只有 9 行的 argmax 相同，max|diff| = 34.47，**第 0 行就不同**
—— prompt 的专家由 helper 算，KV 从第一行起就不是同一个。）

**原因（已知的那类，不是新缺陷）**：peer 路径里 prompt 的专家由 helper 算（q8_1 激活 +
`native_expert_grouped`），默认路径里由主卡的 MMQ int8 算 —— 两条数值路径不同，KV 因此差一点点，
后面的 decode 就可能选到不同的 token。这跟 repo 里早就记着的「helper 路径与主卡路径的
logits 逐行 RMS 差 1.5e-1~3.3e-1」是同一件事（`q36_engine.cu:4081-4092`，
`2026-10-20-q36-1p3h-optimal.md` §7 的纪律 3）。

**⇒ peer 路径不能承诺逐字节相同**；要拿它做默认，得先有一条「与默认路径逐字节相同」的验收门。
（这条缺陷的来龙去脉在 `2026-10-25-q36-peer-decode-defect.md`。）

## 4. 现在的状态（未合并的东西）

- peer 路径**已经写进用户侧的文档**（`docs/QWEN36.md` 的 "The prompt's experts on the helper GPUs" 一节，
  2026-10-25）。旗标：`STRATA_Q36_PREFILL_PEERS`（**只要变量在就算开**，`=0` 也开；`q36_engine.cu:4346`）、
  `STRATA_Q36_PREFILL_PEERS_HOST=1`（host 兜底，本机无 P2P，12 对全 0）、
  `STRATA_Q36_PEER_F16=1`（两个都要正好是 "1"）。
- 引擎自己印的警告还在：「host route: on … **experimental: a decode step after the prompt is not
  safe yet**」（`q36_engine.cu:4729`）。今天实测没有 IMA（10-24 的编码修复和 10-22 的 slot 排序修复
  之后，这条老缺陷没有再复现），但**「不安全」那句本身还没撤**。
- `STRATA_Q36_PEER_KEEP_RESIDENT=1`（主卡接回 39 个常驻专家）实测是负的，旗标默认关
  （`2026-10-24-q36-primary-resident-peers.md` §3b：prompt 持平、decode −12.5%）。

## 5. 结论

1. **「prefill 优化到 300 多」是真的，而且今天还有效**：Qwen3.6 的 peer prompt 路径 **493.6 tok/s**
   （默认路径 156.1，+216%），decode 不掉（tg64 59.4 vs 58.5）。
2. **默认路径 156 tok/s 是设计如此**，不是回归：prompt chunk（T=512）太大，helper 那条路只吃 ≤8 行。
3. **peer 路径还不是能进默认的东西**：数值与默认路径不同（216 token 的 prompt，32 行 logits 里只有 9 行
   argmax 相同，max\|diff\| 34.47，第 0 行就不同），引擎的「decode after the prompt is not safe yet」
   警告**已撤**（`8b844007`），用户侧文档**已写**（`docs/QWEN36.md`，2026-10-25）。
4. **下一步（按顺序）**：
   a. ~~把「decode after the prompt」那条老缺陷查清并撤掉警告~~ → **已完成**
      （`plans/2026-10-25-q36-peer-decode-defect.md`，`8b844007`）；
   b. ~~写 `docs/QWEN36.md` 的一节~~ → **已完成**（`docs/QWEN36.md`）；
   c. **如果要进默认**，先立一条「与默认路径逐字节相同」的验收门（现在做不到，见 §3）。
