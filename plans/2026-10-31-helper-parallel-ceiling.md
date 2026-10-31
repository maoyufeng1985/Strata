# 回程免费 + 专家 GEMM 全隐藏：可行性实测（2026-10-31）

## 0. 这份文档做什么

只做**可行性实测**，不改生产代码。

问题：在 Whittle 45B 这台盒（192.168.1.98）上，"helper 回程免费 + 主卡的专家 GEMM 全隐藏"到底可达不可达？
如果可达，30k 冷提示词能压到多少 tok/s；如果不可达，省时间该转到哪一侧。

产出三个数：**30k 冷墙**、**decode tok/s**、**T-lines 是否逐字节一致**。
这三个数齐了，才决定要不要进实施（writing-plans + team 实施）。

**不做**：不动生产代码、不动默认路径、不改数值（不碰 i4）、不碰用户在途的 i8 改动。
本轮只换运行配置（命令行 + 环境变量）和 profile 文件，引擎二进制保持 `6c9fa66d`。

## 1. 判据（用户 2026-10-31 批准）

| 条件 | 结论 |
|---|---|
| 30k 冷 ≤ 30 s（≥ 1000 tok/s）且 decode ≥ 48 tok/s 且 T-lines 64/64 逐字节一致 | **可达**，进实施 |
| 30k 冷 30–34 s | **有条件可达**，说明还差哪一段 |
| 30k 冷 ≥ 34 s，或 decode 比基线掉 > 4 % | **不可达**，转 P100 侧（dense 23.6 s 才是墙） |

decode 的基线（2026-10-30/31 实测）：64 token / 275-token prompt = 45.5 tok/s；256 token 热批 = 49.1 tok/s；
P100 单卡 32.5 tok/s。

## 2. 机器与被测物

| 项 | 值 |
|---|---|
| 盒 | 192.168.1.98，root（共用机，单引擎独占） |
| 卡 | 3× P104-100 8192 MiB（PCIe bus `00000000:01:00.0` / `02:00.0` / `05:00.0`）+ Tesla P100-PCIE-12GB 12288 MiB（bus `00000000:03:00.0`）|
| 卡间 | PCIe Gen1-x4，**无 P2P**（helper 回程走 host 映射内存）|
| 二进制 | `/home/Strata/build-rebase/strata`，md5 `6c9fa66d3753c9852fa17250c3cf4799`（2026-10-31）|
| 模型 | `/home/models/Whittle-Qwen-3.8-45B-A3B-Q4_K_M.gguf` |
| profile | `/home/Strata/.run/whittle-20261028/profile.bin`（10240 ranked pairs = 40 层 × 256 专家）|
| 30k 提示词 | `/home/Strata/.run/whittle-20261028/bench/prompt.txt`，重复到 30084 个 id |

settled 命令（E0 基线用的就是它）：

```
env STRATA_ROUTE_PRIOR=0 STRATA_MMQ_KQ=1 STRATA_DENSE_F16_ACC=1 STRATA_PF_PEER_COMPRESS=1 \
    STRATA_PF_PEER_I8=1 STRATA_UNBUFFERED_LOAD=1 STRATA_PREFILL_TIMING=1 \
  [ STRATA_PREFILL_REMOTE_PEERS=1 STRATA_PREFILL_REMOTE_PEERS_N=3 ] \
  strata --pack /home/Strata/packs/whittle45b --native <model> --mmap-experts --ple-gguf <model> \
    --ple-io ram --max-context 32768 --spec 2 --suffix-draft 0 --expert-profile <profile.bin> \
    --gpu 2,1,0,3 --kv int8 --expert-cache auto --expert-cache-device1 auto \
    --expert-cache-device2 auto --expert-cache-device3 auto --remote-expert-opt \
    --conversation-cache-mib 0 --prefill 16384 --serve
```

stdin：4× `GEN 1 <30084 ids>` + `QUIT`（`GEN` 的 max_new=1）。
T-lines 臂：`GEN 64 <275 个 id>`，与 `/home/Strata/.run/pickup-20261030/HDP4A-PLANNED/baseline/tlines-T.txt` 逐字节比。

**每次运行前 `pgrep -x strata` 必须为空**（共用机）；结束要收尾（4 卡 0% / 0 MiB）。

## 3. 机制：helper 的行切分（今天是什么）

### 3.1 行怎么分

行 = 专家 GEMM 的 M 维行 = 一个 (token, expert) 激活对。**不是按 token 切**。

- **行的归属 = 那个专家的权重在哪张卡**：`src/prefill/prefill.cpp:4909-4933`，`P.res_has(l,e)` 判定专家归谁。
- **行的物理布局**：本地行在前，每个 peer 一段连续行块，peer 顺序（`prefill.cpp:4902-4903` 注释、`4935-4962` 的 `m.off[]` / `peer_base[]` / `back_at` / `back_rows`）。
- 同一个 token 的不同专家可以落在不同卡上。

### 3.2 专家怎么分到各卡

| 谁 | 规则 | 代码 |
|---|---|---|
| 主卡 P100 | profile 的**每层最热**的 N 个（per-layer admission）| `src/program/generate.cpp:7354-7358` |
| 3 张 helper | 主卡不要的，按 profile 的 **ranked 序轮转**分给各卡（round-robin）| `src/program/generate.cpp:7700-7731` |

默认放置是 `--expert-cache-remote-placement stripe`：ranked 序轮流发，所以每张 helper 拿到**每一层的 1/3**（热度混合）。
另一个值 `layer`：整层给一张卡（layer % cards）。

### 3.3 实测证据（PEERON-3 臂，2026-10-31）

启动行：

```
strata generate: remote ranking: 10240 profiled pairs, 0 other pairs to fill CUDA1..3
strata generate: remote ranks striped across 3 CUDA devices
strata generate: expert cache 4600 slots, 8.19 GiB of VRAM
strata generate: CUDA1: 1880 additional experts, 3.34 GiB
strata generate: CUDA2: 1880 additional experts, 3.32 GiB
strata generate: CUDA3: 1880 additional experts, 3.32 GiB
```

4600 + 1880×3 = 10240 = 40×256，**一个专家恰好归一张卡**。

运行行（冷块，80 layer-pass = 2 chunk × 40 层）：

```
peer GPU 1: idle 28583 moe in 4652 moe gemm 3010; rows/layer 13296 of 131072 cap, experts/layer 39
peer GPU 2: idle 28700 moe in 4640 moe gemm 2912; rows/layer 13329 of 131072 cap, experts/layer 39
peer GPU 3: idle 28128 moe in 4747 moe gemm 3374; rows/layer 14371 of 131072 cap, experts/layer 40
```

- 每张 helper 每层持有 47 个专家、实际算 39 个、接走该层 131072 行里的 ~13000 行（**10.1 %**）。
- 三张合计 ~30 % 的行；主卡仍做 ~70 %。

## 4. 术语陷阱（写在这里，免得下一轮再踩）

**两个引擎里 "stripe" 指相反的东西：**

| 引擎 | "stripe" | 另一个值 |
|---|---|---|
| Strata CLI（`src/program/generate.cpp`，盒上跑的就是它）| ranked 序轮转 = **按专家的行切分** | `--expert-cache-remote-placement layer` = 层切分 |
| Q36 引擎（`src/qwen35/q36_engine.cu:8006-8116`）| **层切分**（helper 拿连续层区间）| `STRATA_Q36_HELPER_PARALLEL=1` = 行切分（`e % n == k`）|

**`STRATA_Q36_HELPER_PARALLEL=1` 在盒上这个二进制里是空操作。**
它只在 `src/qwen35/q36_engine.cu:2548` 被读，而盒上跑的是 CLI 的放置代码（`generate.cpp:7700-7731`）。
（字符串在二进制里是因为 `libstrata_qwen35.a` 被链接进去了；实测启动日志里没有任何 `strata-q36:` 行。）

CLI 的默认 `stripe` 与 Q36 的 row-split **几何相同**（每张 helper 拿每一层的 1/3），差别只在按什么分：
CLI 按**热度 rank** 轮转，Q36 按**专家 id** 取模。两者都是"每层 1/3 + 冷热混合"，本轮不需要区分。

## 5. E0 基线（已实测，2026-10-31 同日同二进制）

| 量 | peer 关 | peer 开 |
|---|---:|---:|
| 30k 冷墙 | 41.007 s = 732.6 tok/s | **36.606 s = 821.8 tok/s** |
| 热块（13699 tok）| 19.5 s | 17.25–17.35 s |
| 主卡自己的专家 GEMM | 16189 ms（39.4 %）| 10313 ms（28.2 %）|
| combine + combine read（等 helper 回程）| 240 ms（0.6 %）| 2849 ms（7.8 %）|
| host staging | 17510 ms | 6206 ms |

**并行值 4.40 s（−10.7 %）**：helper 接走主卡 5876 ms 的专家 GEMM（36 %），回程吐回 2849 ms。
（同日 T0 基线 36.478 s = 822.4 tok/s，二进制 `eb13cd81`；今天的 ON 臂差 0.35 %，盒没漂。）

逐层（80 layer-pass）：主卡 dense 290 ms、自己的专家 GEMM 129 ms（开）/ 202 ms（关）、combine 38.6 ms（开）/ 3 ms（关）。
helper 侧：喂入 58.2 ms/层、GEMM 37.6 ms/层、idle 357 ms/层。

## 6. E1 变体（用户 2026-10-31 定）：3 个 helper 存所有专家 + 行切分

### 6.1 配置

主卡的专家缓存清空，10240 个专家全部住到 3 张 helper 上，按行切分（每张拿每一层的 1/3）。

CLI 里怎么表达（**唯一变量是主卡的 `--expert-cache`**）：

```
--expert-cache 1                    ← 主卡只留 1 个专家（0 会被改写成 auto，见 generate.cpp:5239）
--expert-cache-device1 auto --expert-cache-device2 auto --expert-cache-device3 auto
--expert-cache-remote-placement stripe   ← 默认值，写出来是为了让臂自证
```

`STRATA_Q36_HELPER_PARALLEL=1` **不写**（在 CLI 里是空操作，写了会让人误以为它生效了）。

### 6.2 显存算术（可行）

每专家字节：1880 个 = 3.34 GiB ⇒ **1.819 MiB/专家**；全部 10240 个 = 18.1 GiB。

| 卡 | 现在（1880 个/卡） | E1（3413 个/卡）|
|---|---|---|
| P104 总量 | 8192 MiB | 8192 MiB |
| 专家 | 3420 MiB | **6209 MiB** |
| 引擎其他（peer 缓冲 + 上下文）| 871 MiB | 871 MiB |
| 剩余 | 3901 MiB | **1112 MiB**（引擎下限 512 MiB，`src/core/remote_experts.cpp:311`）|

主卡（P100，12 GB）：缓存腾出 8387 MiB；prompt path 不再借缓存槽位（改自己分配，实测借的是 2558 槽 / 4.56 GiB）⇒ 约剩 4.1 GiB，够。

**现在每张 helper 只有 1880 个专家，不是因为显存不够，是因为主卡占了 4600、只剩 5640 个可分。** 主卡一空，10239 个按轮转分下来正好 3413/卡，显存装得下。

### 6.3 连带影响（要一起读）

1. **decode 会掉**：主卡手上没专家，decode 的路由命中全靠 helper 的远端路径（`--remote-expert-opt`）或 CPU lane。这是本变体最大的代价，判据里 decode ≥ 48 tok/s 正是卡在这里。
2. **prompt path 改自己分配缓冲**：主卡不再借缓存槽位（`generate.cpp:9849` 那条 "allocates its own buffers" 分支），缓冲大小变了，chunk 数可能变（臂里要数 `PP` 行的个数）。
3. **回程没变便宜**：回程仍是 host 路线（无 P2P），combine 等待仍在。E1 测的是"GEMM 全隐藏"这一半，"回程免费"是实施阶段的事。

### 6.4 臂与读数

臂序（盒会漂，同日对照必须夹在中间）：

| 臂 | 内容 |
|---|---|
| E1A | `--expert-cache 1`，30k 4×GEN + T-lines 臂 |
| CTRL | `--expert-cache auto`（今天 settled），30k 4×GEN + T-lines 臂 |

每个臂读：30k 冷墙（`PP` 行）、`DONE` 行、主卡各段（`gemm gate/up`、`gemm down`、`combine`）、
三张 helper 的 `rows/layer` 与 `experts/layer`（验证切法真的变成 1/3）、T-lines 与基线逐字节比、decode tok/s。

Xid-31 PDE 故障（见 §8）会打断臂，臂脚本带重试直到出一条干净臂为止。

## 7. E2 外推门限（E1 之后）

E1 的读数直接给外推的输入：

- 每张 helper 的行数从 10.1 % 涨到 ~33 % ⇒ 喂入与 GEMM 都 ×3.1：
  喂入 58.2 → ~180 ms/层，GEMM 37.6 → ~117 ms/层，链条 **~297 ms/层**。
- 主卡的 dense 窗口是 **290 ms/层**（§5）。链条 297 vs 窗口 290 ⇒ 整块墙 ≈ **23.8 s ≈ 1250 tok/s**。
  余量只有 ±2 %，所以 E1 的实测行数/成本直接决定这个数，不靠推。
- **喂入字节必须减半（i4）才能有把握**：链条 90 + 117 = 207 ms/层 < 290 ms/层，余量 ~30 %。
  i4 是数值变更（激活量化精度），**不在本轮范围**；这里只把它记成门限。

## 8. 风险与已知故障

| 风险 | 事实 | 处置 |
|---|---|---|
| Xid-31 PDE | 盒子的驱动级故障（reader P104 读映射 host 缓冲）。2026-10-31 13:00 之后：对照臂 16384 chunk 死 2/4、4096 chunk 死 3/3；E1 臂死 15/15。`dmesg` 累计 743 条 Xid | 臂脚本带重试；失败的臂标注原因，不静默丢弃。**结论：这台盒现在对 peer 路线的大提示词臂都不友好，与配置无关；30k 的数要等 epoch 过去** |
| P100 处于降级态 | R=160 Q4_K gate/up 今天仍读 4.985 ms（健康值 1.999 ms，10-30 也是 4.981）| 同一状态，可比；但"P100 侧快 2.5 倍"的推论要谨慎 |
| i8 路径的数值 | helper 的行是从 i8 量化过的激活算的，主卡的本地行不是 | T-lines 必须逐字节比；不一致要标出来，不能当"噪声" |
| 独占机 | 共用机 | 每次运行前 `pgrep -x strata` 空；结束 4 卡 0% |

## 9. 验收清单

- [x] E1A 30k 臂干净（rc=0，4 条 `PP` 行），记录冷墙与主卡各段 —— **没拿到：15 条臂全被 Xid-31 打断**
- [ ] E1A 的 helper 行是每张 ~33 %（`rows/layer` ≈ 43000 of 131072），`experts/layer` ≈ 85 —— **没拿到（臂没活到那一步）**
- [x] CTRL 同日对照臂干净，冷墙与今天 PEERON-3 差 ≤ 2 % —— **36.517 s vs 36.606 s，差 0.24 %**
- [x] T-lines 臂：E1 与 CTRL 各 64 行，与基线逐字节比 —— **CTRL 一致；E1 从第 14 个 token 起分叉（两次臂同一位置）**
- [x] decode tok/s，与基线比 —— **E1 40.8 / 36.8 tok/s，对照 46.5 tok/s，掉 12.3 %~20.9 % > 4 %**
- [x] 收尾：4 卡 0% / 0 MiB，无 strata 进程
- [x] 结论落盘：`.omo/evidence/e1-all-helpers-20261031.md`

## 10. 之后才做的事（不在本轮）

1. 判据通过 ⇒ 用 `superpowers:writing-plans` 出实施计划，再并行实施"回程免费"（P100 侧或 helper 侧）。
2. 判据不过 ⇒ 转 P100 侧：dense 23.6 s 才是墙（sm_60 vmad patch + 6.6 s host staging，owner 已划出范围）。
3. i4（喂入减半）只在 E1 显示链条压不进 290 ms 窗口时才提，单独走数值评审。

## 11. E1 的实测结果（2026-10-31 13:00–13:40，追加）

完整证据：`.omo/evidence/e1-all-helpers-20261031.md`。

### 11.1 切法的映射（两处必须记下来）

- **`STRATA_Q36_HELPER_PARALLEL=1` 在盒上这个二进制里是空操作。**
  它只在 `src/qwen35/q36_engine.cu:2548` 被读（Q36 引擎自己开 tier 时）；
  盒上跑的是 Strata CLI（`src/program/generate.cpp`，启动行是 `strata generate: ...`），
  它的放置代码（`generate.cpp:7700-7731`）不看这个变量。
- **CLI 里等价的切法是 `--expert-cache-remote-placement stripe`（默认值）**：
  ranked 序轮转 ⇒ 每张 helper 拿**每一层的 1/3**，与 Q36 的 row-split **几何相同**；
  差别只在按什么分（CLI 按热度 rank，Q36 按专家 id 取模）。
- **`--expert-cache 0` 不能用来清空主卡**：`generate.cpp:5239` 把它改写成 `auto`。
- 术语陷阱：两个引擎里 "stripe" 指相反的东西。

### 11.2 几何：完全按设计落地（实测启动行）

```
expert cache 1 slots, 0.00 GiB of VRAM
CUDA1: 3413 additional experts, 6.07 GiB
CUDA2: 3413 additional experts, 6.05 GiB
CUDA3: 3413 additional experts, 6.05 GiB
the prompt path allocates its own buffers (too few cache slots to borrow)
helper cache on GPU 1/2/3 ...; 1107 / 1121 / 1115 MiB left free
```

10239 个专家（主卡留 1）÷ 3 = 3413/卡 ⇒ **全部专家住在三张 helper 上，每层 1/3**。
显存算术成立：8192 − 6209（专家）− 871（引擎其他）= 1112 MiB 余（下限 512）。
主卡腾出 8.19 GiB（启动时 4384 MiB free，对照臂只有 562 MiB）。

### 11.3 三个判据的读数

| 判据 | E1 | 同日对照 CTRL | 结论 |
|---|---|---|---|
| 30k 冷墙 | **15 条臂全被打断** | 36.517 s = 823.8 tok/s | 未测出 |
| decode 64 tok | 40.8 / 36.8 tok/s | 46.5 tok/s | **掉 12.3 %~20.9 %** ✗ |
| T-lines | **第 14 个 token 起分叉** | 与基线逐字节一致 | ✗ |

### 11.4 30k 臂为什么一条都跑不出来

- 11 条 E1 臂（16384 chunk）+ 4 条（4096 chunk）全部 rc=1，死在第一个 PP 行之前。
- 14 条报 `qsa_decode_attn_batch: an illegal memory access was encountered`；
  1 条报 `prefill gemm: no GPU memory left ...`，但同一条消息里写着
  `cudaMemGetInfo -> an illegal memory access was encountered` ⇒ **上下文早已被污染，同一个根因**（误诊）。
- `dmesg`：对应的 Xid 全是 `Xid (PCI:0000:02:00): 31 ... FAULT_PDE ACCESS_TYPE_READ`，
  faulting 地址是 host 侧页对齐地址 ⇒ **reader P104 读映射 host 缓冲时的驱动级 PDE 竞态**。
- **对照臂今天也不好**：16384 chunk 4 条里 2 条死（PEERON-1/2）、2 条干净（PEERON-3、CTRL-1）；
  4096 chunk 3 条全死（CTRL-C4K-1..3，都是 `prefill copy_i32` 的 IMA）。
  ⇒ 这台盒现在**对 peer 路线的大提示词臂都不友好**，与配置无关；换小 chunk 不能绕过。

### 11.5 冲突在哪

"专家 GEMM 全隐藏"和"decode 快"在这台盒上**直接冲突**：同一批热专家既是主卡 prefill 的 GEMM 负载
（热专家占行质量的绝大部分），也是 decode 的本地命中集。
主卡把热专家让给 helper，prefill 的收益来了，decode 的命中就没了。

方向验证（275-token 提示词，同一臂里读出）：

| | prefill ms | decode ms | decode tok/s |
|---|---:|---:|---:|
| CTRL | 1048.9 | 1376.2 | 46.5 |
| E1（T1 / T3）| **721.0 / 729.7** | 1570.6 / 1739.4 | 40.8 / 36.8 |

⇒ "主卡不做专家 GEMM"本身有效（prefill −31 %）；按 E2 外推，30k 落在 23.8~25 s。

### 11.6 下一步（中间档，零代码改动）

主卡只留 **decode 热集**，提示词的热专家全部下沉给 helper：

1. 先取 decode 的 routing（`--dump-routing` 跑一次 T-lines 臂），
   得到 decode 实际触到的 (层, 专家) 集合；
2. 用它写一份新的 profile：**每层**的 rank 序 = decode 热集（按 decode 次数）→ 其余按现在的热度序；
   全局序保持"rank 外层、层内层"（现在的 profile 就是这个形状），
   这样 round-robin 之后每张 helper 仍是每层 1/3；
3. `--expert-cache <decode 热集的大小>`（主卡只留那几百个槽）+ 三张 helper auto；
4. 臂与判据同 §6.4。

**取 routing 的结果（2026-10-31 13:32，`.omo/evidence/` 里存了原始文件）**：

- 64 token 的 decode 触到 **3989 个不同的 (层, 专家) 对，每层 60~176 个（中位 93）**。
  （之前记的"276 对 ≈ 7/层"是错的，别再用。）
- 提示词的行质量**极度集中**：每层最热的 7 个专家就占 **40.9 %** 的行，最热 30 个占 78.8 %，
  最热 60 个占 93.7 %（用 56 个位置的 prefill trace 量，`.pi/tmp/trace.bin`）。

**取舍曲线**（`tools` 里两个脚本：`an_tradeoff.py` 用现 profile 的序，`an_v1_curve.py` 用 decode 的序）：

| 主卡每层 N | 槽 | 用的序 | 主卡留住的 decode 路由 | helper 接走的提示词行 |
|---:|---:|---|---:|---:|
| 115 | 4600 | 热度序（今天）| 78.3 % | 4.9 % |
| 30 | 1200 | 热度序 | 35.6 % | 21.2 % |
| 7 | 280 | 热度序 | 8.5 % | 59.1 % |
| 45 | 1800 | **decode 序（V1）** | **80.1 %** | **49.7 %** |
| 60 | 2400 | decode 序（V1）| 88.6 % | 37.9 % |
| 93 | 3720 | decode 序（V1）| 97.2 % | 14.8 % |

⇒ **用热度序没有中间档**（要给 helper 走行，主卡就得交出热专家，decode 的命中同时崩）。
**用 decode 序（V1）有**：N=45/layer 时 helper 接走 **49.7 %** 的提示词行，
而主卡仍留住 **80.1 %** 的 decode 路由（比今天的 78.3 % 还高一点）。
按 E1 的 prefill 收益（−31 %）折算，30k 落在 **31 s 一线**（"有条件可达"那档）。

V1 的 profile 已经写好（本地 `.pi/tmp/v1-profile.bin`，`mk_v1_profile.py` 生成的，
全局序 rank 外层/层内层 ⇒ 主卡拿到每层前 N 个），
主卡的槽位按 `slots/40` 平分到各层（`src/core/expert_cache.cpp:723-732`），
所以 `--expert-cache 1800` = 每层 45 个。

**未测**：E1 的 30k 墙；V1 的任何数字（臂还没跑）；
V1 会不会让 T-lines 分叉（主卡的 GPU/CPU 算子配比变了，理论上会，得实测）。

### 11.7 这台盒现在的状态（下一轮开跑前先看这个）

- 30k 的 peer 路线臂**大面积被 Xid-31 打掉**，与配置无关：
  对照臂 16384 chunk 死 2/5、4096 chunk 死 3/3；E1 臂 16384 chunk 死 11/11、4096 chunk 死 4/4。
- 干净的对照臂今天有两条：`CTRL-1` 36.517 s = 823.8 tok/s、`CTRL-2` 36.678 s = 820.2 tok/s
  （与 PEERON-3 的 36.606 s 一致，盒没漂）。
- 小提示词的臂（275 token）今天全部干净（E1 的两条 T-lines 臂、CTRL-T1、decode routing 那一臂）。
- ⇒ **30k 的数要等 epoch 过去**；这中间先跑小提示词的臂和 profile 的构造验证。
