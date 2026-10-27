# 把 QSA 接进 q36：工作量复核（2026-10-27）

## 为什么要问这个

q36 在 ctx 262144 上跑不起来（实测 2026-10-26，`plans/2026-10-26-q36-kv-stream.md`）：
KV（int8，11 层）2.84 GiB + hybrid staging 1.10 GiB > P100 的 12 GB。
上游 Flash-Next 有一条「KV 放主机内存、显存只留滑窗」的路（`src/kernels/cuda/kv_stream.cu`），
但它是给**稀疏 QSA** 设计的 ⇒ 「把 QSA 接进 q36」和「KV 流式」是同一件事的两半。

## 这份复核要回答的

1. **工作量**：接进 q36 要动哪些文件、多少行、什么顺序、要过哪些门 ⇒ 几个人天。
2. **哪些能直接搬、哪些必须重写**：QSA 的核已经在 `strata_kernels` 里编好了（CMake:413-421）
   ⇒ 搬的是「接线 + 改数据布局 + 改 KV 写法」，不是搬核。
3. **硬阻塞**：VMM（这台 P100 不支持：`cuMemMap` → 801）、Q4_0 的 Hadamard 旋转（PR #21，不在 q36）、
   Qwen3.8-Flash-Next 的几何 vs q36 的 16/2/head_dim 256。
4. **结论**：这条路是不是把 ctx 262144 弄上 P100 的**最省**办法；不是的话，最省的是什么。

## 先量一件事：更长的上下文今天值不值（MEASURED 2026-10-27，`R-longctx`）

peer 路径全开，`--bench-pp 32768 --bench-tg 64`，`--kv int8`，192.168.1.98：

| ctx | 主卡每层的专家（VRAM 里的）| pp32768 | **tg64** |
|---:|---:|---:|---:|
| 32768 | 38 / 256 | 606 | 34.4 |
| 65536 | 24 / 256 | 605 | 34.4 |
| 131072 | **9 / 256** | 599 | **34.3** |

⇒ **上下文变长，decode 速度不掉**（34.4 → 34.3 tok/s）：decode 只读当前位置以前的 cell，
prompt 长度不变 ⇒ 读量不变；主卡的专家被挤掉（38 → 9 个）由 CPU pool 和 helper 顶上，
**每 token 的专家活本来就是那么多** ⇒ 不掉速。
⇒ **上下文长度的代价是显存，不是速度** ⇒ 「把 262144 弄上 P100」这件事的收益是
「能装下 256k 的 prompt 和答案」，**不是**「decode 更快」。
⇒ 而 **q36 的 int8 KV 在 262144 是 2.84 GiB**：换成 Q4_0（144 B/head/cell 对比 int8 的 528）
⇒ **0.77 GiB**，省下的 2.07 GiB 加上 staging 的 815 MiB 的洞 ⇒ 262144 直接装得下，
**不需要 QSA、不需要 KV 流式**。

## 方法

六个独立读者各读一个子系统（流式层 / 选择器 / decode attention / q36 侧 / 接线与门 / 阻塞项），
一份综合，再两个对抗读者（一个查估计是否乐观/悲观，一个查结论是否选错了路）。
只读，不改代码；每条结论带 file:line，推断要标明是推断。

## 结果

### 0. 一句话

**「把 QSA 接进 q36」这句话要改写**：能接的只有「**池 + 流式**」那半；
打分那半对 q36 是**死的**（q36 没有 indexer），decode attention 那个核是 **G=12 编死的**、接不了。
⇒ 实际要写的是「**q36 自己的 Quest 当选择器 + q36 自己的 G=8 分块核 + 上游的 kv_q8/kv_stream 池**」。
工作量 **15-24 人天**；三个门**量了两个**（pin 得住、读带宽 6.0 GB/s、散点读一样快），
**只剩「Quest 在 262144 上留多少 chunk 才不掉质量」没量**。
⇒ 而「最省的路」不是这条：**staging 缩小 + Q4_0** 先把 262144 启动起来，这条当第二个项目。

下面每条都标了 **【实测】** / **【读代码，未实测】** / **【推断】**。

### 1. 哪些能搬，哪些不能（全部【读代码】）

| QSA / 上游的件 | q36 能用吗 | 为什么（file:line） |
|---|---|---|
| `qsa_block_scores` / `qsa_index`（打分那半）| **不能** | q36 没有 indexer：`family_layout.hpp:66` 给 Qwen35Moe 设 `idx_q_heads = idx_key_dim = 0` ⇒ `qsa_select.cu:1712-1714` 直接 `exit(1)`；`qsa_select.cu:21-23` 的 `static_assert` 还把 `IDX_DIM=128/IDX_HEADS=4/R=4` 钉死 |
| `qsa_block_topk`（取 top-k）| **不能** | `qsa_select.cu:2557` 的第一件事就是 `if (s.idx_block != R || …) exit(1)`，`R` 是 `constexpr 4`（`:21`）⇒ 传 `idx_block=1` 一定 exit |
| `qsa_decode_attn.cu`（稀疏 decode attention）| **不能** | `qsa_decode_attn.cu:24-25`：`constexpr int G = 12` + `static_assert(HD==… && G==n_head/n_head_kv)`；`:1189` 运行时 `n_head != G*n_head_kv` 就 exit。q36 是 16/2 = **G=8** |
| `kv_q8.cu` / `kv_stream.cu`（流式层）| **能，但要改读法和写法** | 见 §2、§3；它们不依赖 indexer |
| `kv_q4.cu`（Q4_0 + FWHT256）| **能，q36 从没用过** | 已经在 `strata_kernels`（CMake:414）⇒ 链进 q36 不用改 CMake；但 `q36_engine.cu` 里 grep `kv_q4|fwht256` **0 命中**，`--kv` 现在直接拒绝 `q4_0`（`:8084-8097`）⇒ 是新活，不是 cherry-pick |
| q36 自己的 Quest（`k_quest_bounds/sel/pick`）| **已经在树里** | `q36_engine.cu:798-942`；它才是选择器 |

⇒ **「接 QSA」的净内容 = 搬上游的池和流式层，加上 q36 侧的三处改写。**

### 1.1 上游已经有 `--kv-resident`，但它绑在 QSA 的层上（【读代码】）

上游**已经出货**了 KV 流式：`--kv-resident N`（`src/program/generate.cpp:836`）——
「keep N cells of each **QSA layer** in VRAM (min 20480) and the whole K/V in pinned RAM;
the freed VRAM goes to expert slots」，默认 0（全部留在显存里）。
`docs/DETAILS.md:216` 还记着它的代价：「about 13.7 KB of RAM per context token (1.7 GB at 128K)」，
并且「Never under WSL」（就是 §3 第 1 号门那条 pin 的限制）。
⇒ 两件事：
- **好消息**：流式层、pin 的兜底、报错路径**上游都写好并出货了**（`kv_host_alloc`），不是从零开始。
- **坏消息**：它是**按 QSA 的层**写的（每层一个池、每层一个 resident 窗口）。
  q36 是**稠密 attention、11 层、没有 QSA 层** ⇒ **不是开个旗标就能用**，得把
  「每层一个池」接到 q36 的 11 个 attention 层上，这正是 §2 的活。

### 2. 必须重写的地方（【读代码，未实测】）

1. **`k_attn_part_gqa` 的读法（最大的一块）**：现在 K/V 直接按 `[kvh][ctx][HD]` 索引
   （`q36_engine.cu:580`、`:692`），**没有页表**。分页读不是换指针：那个核把 V 的权重
   先乘进 shared memory（`:746-760`），shared 表是按**块内 cell 偏移**索引的，
   所以要先把一个 chunk 的 256 个 cell 的码**经页表搬到 shared memory** 再算
   ⇒ 访问模式和占用都变了（综合估 150-250 行，**实际更贵**，见 §4 的修正）。
2. **`k_quest_bounds` 也要跟着改**（对抗读者补上的，综合漏了）：它读的是**码**和 **scale**
   （`:798-838`：`kc[(kh*ctx+p0+k)*HD+d]`、`ks[(kh*ctx+p0)*nsg]`）⇒ 分页后两个地址都得翻译。
   ⇒ **我自己的补充**：最省的做法不是「写完再扫一遍」，而是**写的时候顺手累加**
   （每个 cell 写进去时对这个 dim 做一次 min/max）——写一次、读 256 次，摊薄几乎不要钱；
   写后重扫要过 PCIe。
3. **Quest 的 keep-mask → ids/steps**：`d_sel` 是 per-(row, chunk) 的 uint8（`:3329`），
   而 `kv_stream_resolve` 要的是 per-query 的**cell id 升序表**（`kv_stream.cu:89-165`）
   ⇒ chunk id 要在**设备上**展开成 256 个 cell id ⇒ 新写一个核，**约 150-200 行**（综合估 40-80，低了）。
4. **码的写法和布局不一样，直接复用会让门挂掉**：
   q36 的 int8 存法（`:500-515`）scale 是 `__half`、**按浮点 scale 量化**、布局 `[kvh][pos][group]`；
   `kv_q8` 的（`kv_q8.cu:33-73`）scale 是 uint16 装 fp16、**按存下来的 fp16 scale 量化**、
   布局 `[page][kvh][cell_in_page][group]`。
   ⇒ 两边**不是同一件事**：直接调用 `kv_append_q8_step`，**码会差最后一个 bit**
   ⇒ 「开池 + k=nc 等于稠密」那道验收门会**静默地挂**（它测的是「翻译对不对」，
   码不一样就说明翻译本身错了）。
   ⇒ **纠正对抗读者一处**：它说「q36 是按 chunk 量化、kv_q8 是按 cell 量化」——不对，
   两边都是**按 cell、按 64 维一组**（`KV_Q8_GROUP=64`）；真的差别是 **scale 的舍入**和**数组的布局**。
   ⇒ 修法比「写一个 chunked 变体」小：照 q36 的口径重写那个 append（十来行）+ 一个布局转换。
5. **分配**：`alloc_session`（`:3101-3152`）现在每 session 每层平铺分配 K/V；分页要多三样：
   宿主机上 pin 住的整份 KV（11 层 int8 ≈ 3.04 GB，**逐层 pin**，照上游 `layer.cpp:905-919` 的做法）、
   显存里的槽池、页表。
   ⇒ 还要分清两个数：**页表长度 = n_blocks**（262144 个 cell 的页数），
   **槽池大小 = n_slots**（显存里能放多少页）——综合把它们混成了一个（对抗读者纠正的）。
   ⇒ `kv_stream.cu:225-228` 有个硬门：`n_slots < 1024` 直接 exit。
   ⇒ 上游的 resident 窗口下限是 **20480 cell/层**（`generate.cpp:836`）⇒ 折算成 q36 的 chunk
   是 80 个 chunk/层 ⇒ 槽池至少要装得下这个。

### 3. 三个门：两个已经量了（MEASURED 2026-10-27，`.run/R-pinprobe-20261027.sh`，探针 `.run/pinprobe.cu`）

| # | 要量的 | 结果 |
|---|---|---|
| 1 | **能不能 pin 住 3.04 GB 的宿主机副本** | **量了：能。** P100 用 `cudaHostAlloc(Mapped\|Portable)` 一次 pin **1 / 2 / 4 GiB 都成功**（盒子上当时 30 GiB 空闲内存）⇒ 3.04 GB **不会被驱动拒绝**。剩下只有一件事没量：「引擎跑着的时候（专家 arena 已经 pin 了 ~18.75 GiB）还能不能再 pin 3.04 GB」—— 那是内存账，不是驱动的账（31 GiB 里还剩 ~12 GiB） |
| 2 | **读宿主机副本有多快** | **量了：散点几乎没有惩罚。** kernel 侧 UVA 连续读 512 MiB = **6.0 GB/s**（`cudaMemcpy` H2D 6.1 GB/s）；**散着读 270336 B 的 chunk**（11 层，每层 4/8/16/32/64 个）= **5.6~6.0 GB/s** ⇒ 和连续读一个速度。**每 token 的代价**：8 chunk/层 ⇒ 22.69 MB/token ⇒ **4.23 ms/token**；16 chunk/层 ⇒ 45.38 MB ⇒ 7.99 ms/token |
| 3 | **Quest 在 262144 上留多少 chunk 才不掉质量** | **没量，还是唯一的未知。** Quest 的 `k` 是旋钮不是测量（`quest_k()`，`:2898-2907`）；已测的只有 16k/32k（k=8 在 32k 上逐字节等于稠密，k=4 在 16k 上分叉）。262144 上 nc=1024，k=8 ⇒ 只留 **0.78%** ⇒ 这个数同时定 n_slots、显存、和每 token 的 PCIe 量 |

⇒ 计划里那个 **6.03 GB/s** 被复核了：**探针量到的连续读是 6.0 GB/s** ⇒ 计划用的数是对的，
而且它是**这台机器上主卡的 H2D**（helper 的链路实测 H2D/D2H 都是 0.79~0.81 GB/s，
链路上限 ~1.0 GB/s：PCIe gen1 x4，`lspci` 实测；`.run/pcbw2.cu` 2026-10-27 直测）。
⇒ **对抗读者担心的「突发读比连续读慢」不成立**（270 KB 的粒度上散着读和连续读一样快）——
它担心的**真正**的门是「保留集要小于 ~6% 才划算」，那个门依然在（门 3）。

⇒ **对抗读者各修正一次估计**：综合说 10-18 人天，两个对抗读者分别指出
「`qsa_block_topk` 桥接是死的」「chunk 展开的核被低估」「`k_quest_bounds` 也要改」
「`k_attn_part_gqa` 不是换指针而是重排访问」「pin 内存是 show-stopper 而不是风险」
⇒ **修正为 15-24 人天**（含测量；现在门 1、2 量完了 ⇒ 剩下的基本是写码）。
⇒ **纠正对抗读者一处**：它把 6.03 GB/s 说成「另一张卡上 Flash-Next 的数」——
不对，**6.03 就是这台机器上主卡的 H2D**，探针也复核了（6.0 GB/s）。

### 4. CUDA 图：流式这条路必须走 eager（【读代码】）

mixer 现在每层一张图（提交 `c5e71fad`）。图的规矩是「**地址固定的设备内存，capture 之前填好**」。
Quest 的 `d_sel` 每 token 由 `k_quest_pick` 重算 ⇒ 这个前提**破了**：
要么把 chunk→cell 的展开也塞进图里（但它每 token 变的，还是破），
要么**流式那条路不进图**。
⇒ **决定：流式走 eager，不进 mixer 图。**
⇒ 不花钱：实测图的墙钟收益是 **0**（60.6 vs 60.3 tok/s，见 `plans/2026-10-26-q36-12h-report.md` §6）。

### 5. 顺序（9 步，前 3 步是量）

1. **量** Quest 的 k 曲线 @262144（pin 上限和读带宽已经在 2026-10-27 量了，§3）。
2. `qsa.hpp`：加一个 q36 的 `QsaShapes`（16/2/256 + 选定的 page_size）。
3. `alloc_session`（`:3101-3152`）：`STRATA_Q36_KVPOOL=1` 之下分配宿主机池 + 槽池 + 页表（默认关 ⇒ 逐字节不变）。
4. `k_attn_prep`（`:499-533`）：int8 的写改成「宿主机那份永远写、槽里只写驻留的那份」（`kv_append_q8_step` 的约定），
   **按 q36 的量化口径**（见 §2.4）。
5. `k_attn_part_gqa`（`:685-763`）：分页读；保住 G=8 分块 + load8 的对齐。
6. Quest 块（`:4086-4101`）：`d_sel` → ids/steps（新核），在 attention 之前调 `kv_stream_resolve`。
7. `k_quest_bounds`（`:798-838`）：跟着改（或者改成写的时候累加）。
8. **缩 staging**（计划 path A，`:126-139`，约 300 MiB 的洞），省出来的显存喂给专家的槽。
9. 验收：parity 门（旗标关时逐字节不变，`gen: 220 16 15 15 15 20332 1740 369`）、
   「开池 + k=nc 等于稠密」、ctx 262144 真能启动、decode 的 ms/token 实测。

⇒ **第 8 步和「流式端口」是同一件活**（对抗读者指出综合算了两遍）：
先做缩 staging，流式端口就只剩「读 PCIe 上的那几块」这一件事 ⇒ **省掉约 3 天**。

### 6. 验收门

- 旗标关 ⇒ 逐字节不变（parity `gen: 220 16 15 15 15 20332 1740 369`）。
- 开池 + Quest `k=nc`（全留）⇒ 输出等于稠密路径。**前提是码一样**（§2.4）⇒ 所以第 4 步必须先对。
- 开池 + 真 k ⇒ 单独一道质量门（token 一致 / KL 对稠密参考）。
- ctx 262144 **真能启动**（今天会打印「not enough VRAM left for the hybrid experts」并退出，`:6719-6724`）。
- 262144 下的 decode ms/token **实测**，不许外推。

### 7. 那最省的路是什么（结论）

| 路 | 代价 | 结论 |
|---|---|---|
| **staging 缩小 + Q4_0 KV** | Q4_0 是**有损**的：Flash-Next 那边 ppl +8~12%（`docs/DETAILS.md:215`），**q36 没量过**。而且它**不是「不用改核」**：q36 从没调用过 `kv_q4.cu`，要加 fwht256 旋转（写在 `k_attn_prep`、读在 `k_attn_part`/`k_kv_f16`）+ 一个 dequant 分支 + `--kv` 的分发，**约 200-300 行**（综合估 80-200，低了）。但它**不花每 token 的 PCIe** | **先做这个**：把 262144 启动起来（计划自己的算：KV q4_0 ⇒ free 2122 MiB ⇒ 24 槽/层，`:99-105`） |
| 只缩 staging | 主卡只剩 **7 槽/层**（现在 38）⇒ 能启动，decode 变慢 | **单独不够** |
| 把 KV 搬到三个 helper 上 | 要**新的 helper 侧 attention 核**（`RemoteExperts` 是给专家 GEMM 用的），还要把 ~2000 个专家退回 CPU 池 | 最贵 |
| **Quest + 池 + 流式（本复核的对象）** | 15-24 人天；三个门量了两个（§3），**Quest 的 k 曲线还没量**；每 token 要付 PCIe（**已量：4.23 ms/token**，见下） | **第二个项目，lossless** |

⇒ **最终建议**：
**先做 staging 缩小（和流式端口第 8 步是同一件活）+ Q4_0** ⇒ 262144 能启动，速度大约不变。
**流式端口留作第二个项目**：它的收益是**显存**（容量 + 专家的槽），不是 tok/s ——
这跟上面量到的「上下文长度的代价是显存，不是速度」是同一件事。
⇒ 而且流式自己**不免费**（**MEASURED 2026-10-27**）：Quest 在 262144 上留 8 个 chunk（2048 cell/层）时，
每 token 要搬 22.69 MB ⇒ 散点读 5.62 GB/s ⇒ **4.23 ms/token**（16 chunk/层 ⇒ 7.99 ms/token）。
保留的块每 token 都在变 ⇒ 几乎每 token 都要重搬。
⇒ **它换的是显存，不是速度**：换 2.84 GiB 的显存，花 ~4 ms/token 的 PCIe。

### 8. 这份复核自己改了哪几处（留给下一个人）

1. **对抗读者改综合**：10-18 → **15-24 人天**；`qsa_block_topk` 的桥接是死的；
   chunk 展开的核 40-80 → **150-200 行**；`k_quest_bounds` 也要改；
   `k_attn_part_gqa` 不是换指针而是**重排访问**；pin 上限是 show-stopper 不是风险。
2. **我改对抗读者**：q36 与 kv_q8 的差别**不是** per-chunk vs per-cell（两边都是 per-cell、按 64 维一组），
   而是 **scale 的舍入**和**数组的布局**；6.03 GB/s **是这台机器主卡的 H2D**，不是别家卡的数。
3. **我改综合**：`k_quest_bounds` 那一步的最省解法是**写的时候累加 bounds**，不是写后重扫。
4. **我改两边的 pin 说法**：不是「一次 pin 3.04 GB」，上游是**逐层 pin**（一层 277 MB，
   `layer.cpp:905-919`），而且兜底（`cudaHostRegister`）**在树里** ⇒ 这是「累计多少能 pin 住」的未知，
   不是「这个设计能不能跑」的未知。
5. **综合漏掉的**：上游**已经出货**了 `--kv-resident`（`generate.cpp:836`），
   但它绑在 **QSA 的层**上 ⇒ q36 不是开个旗标就能用。
6. **没变的**：q36 没有 indexer ⇒ QSA 的打分半边是死的；`qsa_decode_attn.cu` 是 G=12 编死的；
   这两条两个对抗读者都复核过，**没有反驳**。
7. **探针量的**（`.run/pinprobe.cu`，2026-10-27）：P100 能一次 pin 4 GiB；
   散点读 270336 B 的 chunk 是 5.6~6.0 GB/s（≈ 连续读）⇒ 「突发读更慢」这个担心不成立；
   Quest 留 8 chunk/层 时流式每 token 4.23 ms。探针和日志在 `.run/pinprobe.cu` 与盒子的
   `.run/quest-20261026/pinprobe.log`。
