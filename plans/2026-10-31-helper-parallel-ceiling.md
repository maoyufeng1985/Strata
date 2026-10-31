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
| Xid-31 PDE | 盒子的驱动级故障，今天 4 个 peer 臂打挂 2 个（`dmesg` 累计 737 条 Xid）| 臂脚本带重试；失败的臂标注原因，不静默丢弃 |
| P100 处于降级态 | R=160 Q4_K gate/up 今天仍读 4.985 ms（健康值 1.999 ms，10-30 也是 4.981）| 同一状态，可比；但"P100 侧快 2.5 倍"的推论要谨慎 |
| i8 路径的数值 | helper 的行是从 i8 量化过的激活算的，主卡的本地行不是 | T-lines 必须逐字节比；不一致要标出来，不能当"噪声" |
| 独占机 | 共用机 | 每次运行前 `pgrep -x strata` 空；结束 4 卡 0% |

## 9. 验收清单

- [ ] E1A 30k 臂干净（rc=0，4 条 `PP` 行），记录冷墙与主卡各段
- [ ] E1A 的 helper 行是每张 ~33 %（`rows/layer` ≈ 43000 of 131072），`experts/layer` ≈ 85
- [ ] CTRL 同日对照臂干净，冷墙与今天 PEERON-3 差 ≤ 2 %
- [ ] T-lines 臂：E1 与 CTRL 各 64 行，与基线逐字节比（一致 / 不一致，两种都要写下来）
- [ ] decode tok/s（64 token 与 256 token 热批），与基线比，掉 > 4 % 就记"不可达"
- [ ] 收尾：4 卡 0% / 0 MiB，无 strata 进程
- [ ] 结论落盘：`.omo/evidence/` 一份证据 + 本文档补一节实测数

## 10. 之后才做的事（不在本轮）

1. 判据通过 ⇒ 用 `superpowers:writing-plans` 出实施计划，再并行实施"回程免费"（P100 侧或 helper 侧）。
2. 判据不过 ⇒ 转 P100 侧：dense 23.6 s 才是墙（sm_60 vmad patch + 6.6 s host staging，owner 已划出范围）。
3. i4（喂入减半）只在 E1 显示链条压不进 290 ms 窗口时才提，单独走数值评审。
