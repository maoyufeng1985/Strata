# q36：prefill 的下腿字节 —— 唯一完全暴露在关键路径上的 PCIe 时间（2026-10-24）

目标机：192.168.1.98（1× Tesla P100-12GB 主卡 + 3× P104-100 8GB helper）。
前置：`plans/2026-10-24-q36-prefill-primary-fp16-review.md`（关键路径分解）、
`plans/2026-10-24-q36-primary-fp16-verdict.md`（主卡 FP16 三道门）。

## 0. 这一轮改什么

**helper 的下腿（主卡 → helper 的激活）从 fp32 改成更窄的格式**，每个 helper 每层少搬字节。

**先说清为什么是这条而不是别的**（§1 的时间线）：

| PCIe 段 | 时间/层 | 暴露在关键路径上吗 |
|---|---|---|
| 下腿 H2D（激活，4 MiB/卡） | 5.18 ms | **是，全额**（host 阻塞在 `prepare_end`，GPU 空转） |
| 上腿 D2H（结果行，5.33 MiB fp16/卡） | 6.90 ms | **否，大部分被主卡 MMQ 的 4.68 ms 掩盖** |

上腿的 D2H 在 helper 算完的那一刻就在 helper 自己的流上发起了，主卡的 MMQ 同时在跑，
所以上腿只把约 `6.90 − 4.68 ≈ 2.2 ms` 露在关键路径上。
**下腿是唯一一处「GPU 空转等 PCIe」的地方** —— 所以它的每一毫秒都直接加在 forward 上。

## 1. 现状（MEASURED，2026-10-24）

`STRATA_Q36_PROMPT_TIMING=1`，`bin/strata-q36.krd`，bench 0-2 聚合（40 层 × 3 pass）：

```
mixer=7.04 | router_route=0.26 | route_d2h=0.05 | plan=6.79 | primary_work=4.68
| helper_wait=7.65 | combine=0.20 | forward=26.67
HOST WALL: prep_sync=5.11  peer_plan=6.77  sync3022=18.66
plan 的 host 墙: wait=5.583 (helper CUDA1 4.947, CUDA2 0.007, CUDA3 0.164)
```

**下腿的字节**（`q36_peer_prefill.cu:841` 的 `cudaMemcpyAsync(m.d_x, m.h_x_use, ...)`）：
每 helper 每层 `n_tok × n_embd × 4` = 512 × 2048 × 4 = **4.00 MiB** 激活 + 16 KiB 路由。

**这 4 MiB 是 fp32，而 helper 收到后立刻把它量化成 q8_1**
（`q36_peer_prefill.cu:945` 的 `MMQ::quantize`）—— **fp32 的后 16 位精度在传输路上就扔了。**

## 2. 收益表（helper 链路实测 0.81 GB/s）

| 激活格式 | 每 helper 每层 | 传输时间 | 省 | forward 26.67 → | prompt 提速 |
|---|---|---|---|---|---|
| **fp32（现状）** | 4.00 MiB | **5.18 ms** | — | 26.67 | — |
| fp16 | 2.00 MiB | 2.59 ms | 2.59 ms | 24.08 | **+10.7%** |
| q8_1 | 1.00 MiB | 1.29 ms | 3.88 ms | 22.79 | **+17.0%** |

**三卡的传输是并行的**（`prepare_begin` 对三张卡依次入队，各自的链路），
所以省的是**一次传输的时间**，不是三倍。

## 3. 两条路的取舍

### 路线 A：fp16（改动小，数值会变）

- **怎么走**：主卡把 `peer_stage_x`（host pinned 的 fp32 窗口）改成 fp16 存，
  helper 收到后转回 fp32 再量化。
- **改哪**：`q36_engine.cu:2350` 的 `peer_stage_x` 分配 + `:4130` 的 D2H；
  `q36_peer_prefill.cu` 的 `m.h_x_use` 类型 + `:841` 的上传 + `:945` 前的转换。
- **数值**：fp16 的相对误差 < 1e-3，而 helper 侧本来就要量化成 q8_1（8 bit，
  相对误差 ~1e-2）—— **fp16 这一步的误差比 q8_1 量化小一个量级，数值上几乎免费**。
  但**仍然是不同的数值路径**，logits 不会逐字节相同，必须实测并记录 max|diff|。

### 路线 B：q8_1（改动大，**可以 bit-exact**）

- **怎么走**：主卡侧就把窗口量化成 `block_q8_1_mmq`，helper 直接用，不再量化。
- **能不能 bit-exact**：MMQ 的 q8_1 量化是**按行**的（每行独立 scale，
  `quantize_mmq_q8_1` 的 grid 是 `(ne1, block_num_y)`，ne1 = 行数），
  **行的量化与别的行无关** → 主卡量化与 helper 量化同一行，**结果逐字节相同**。
  但有两个前提要核：
  1. **DS layout**：`mmq_get_q8_1_ds_layout(type_src0)` 按**权重的类型**选布局
     （Q4_K → `DS4`？Q5_K → `D2S6`？）。主卡量化时必须传**同一个 type**，
     否则 layout 不同，字节就不同。
  2. **行数不同**：helper 量化的是它自己那 n 行（`m.d_src` 选的），主卡量化的是
     整个窗口 T 行。MMQ 的 q8_1 布局按行分组转置，**行数不影响单行的字节**
     （每行独立算 scale 和部分和）—— 但 `ne1` 参与 grid 划分，不影响数值。
  3. **量化 kernel 的 `ids` 路径**：`quantize_mmq_q8_1` 支持 `ids`（取行）。
     主卡量化整个窗口（`ids=nullptr`）与 helper 量化它的 n 行（`ids=d_src`），
     **只要行的 fp32 源相同，量化结果相同**。
- **改哪**：主卡侧加一个 q8_1 staging + 一次量化调用；helper 侧改成「收 q8_1、
  按 `d_src` 选行」，跳过自己的量化。
- **代价**：`d_hq`（down 投影的 q8_1 激活）还是在 helper 上量化，那部分不变。

### 选哪条

**先做路线 A（fp16），量到了再说路线 B。** 理由：
- A 的改动面小（4 个点），能快速验证「下腿字节减半 = forward 减 2.59 ms」这个因果链；
- A 的数值影响可预测（fp16 误差 < q8_1 量化误差）；
- B 要核 DS layout 和行数无关性两件事，做错了会**静默改变数值**（不是崩溃，是 logits 变），
  该在 A 验证过因果链之后再做。

**如果 A 证到了因（forward 减 2.59 ms），B 就值得做**（再多省 1.29 ms，且 bit-exact）。
**如果 A 没证到因**（forward 没减），说明 5.18 ms 里不止传输时间（可能是 helper 的行计划核、
或 `prepare_end` 的同步本身），那 B 也不必做 —— 该回去找那 5.18 ms 的真实构成。

## 4. 改动清单（路线 A）

- [ ] `q36_engine.cu:1591` `peer_stage_x` 的类型：`float*` → `__half*`
      （注释要写：这是下腿的传输格式，不是数值格式）
- [ ] `q36_engine.cu:2350` 的分配：`cudaHostAlloc(..., win * N * sizeof(__half))`
- [ ] `q36_engine.cu:4130` 的 D2H：主卡显存是 fp32，host 是 fp16 →
      **不能直接 cudaMemcpy**，要「主卡先转 fp16 到一块 device buffer，再 D2H」
      （多一次 device 转换，但主卡那条腿是 6.07 GB/s，转换 4 MiB 只要 ~0.05 ms）
- [ ] `q36_peer_prefill.cu`：新增 `m.h_x_use` 的 fp16 变体 + helper 侧
      「fp16 → fp32」的 device 转换（在 `MMQ::quantize` 之前）
- [ ] **旗标**：`STRATA_Q36_PEER_DOWN_F16=1`，只认精确的 `"1"`，
      函数局部静态，默认关（默认路径逐字节不变）
- [ ] `peer_stage_x` 的大小：`win * N * sizeof(__half)`，win = tmax * Kp

## 5. 验证清单（AGENTS.md：改了必须实测）

- [ ] 编译成**新文件名** `bin/strata-q36.dlf`（对照 `bin/strata-q36.krd` = `fa39ea5f`，
      `bin/strata-q36` = `b7d53f977` 全程不动），核 md5
- [ ] **默认（旗标关）**：逐字节等于 `bin/strata-q36.krd`（旗标关 = 现状）
- [ ] **旗标开**：pp512 / tg64，对照 krd；预期 prompt +10% 左右、decode 持平
- [ ] **logits 逐行对照** krd：记录 argmax 一致数与 max|diff|
      （fp16 下腿**预期不同**，理由见 §3；差异应远小于 q8_1 量化误差）
- [ ] IMA=0、nan=0、rc=0、Xid=0
- [ ] `STRATA_Q36_PROMPT_TIMING=1` 的 span：`plan` 的 host wait 应从 5.58 降到 ~3.0
- [ ] 三卡的 `wait` 分布（CUDA1 4.95 → ?，CUDA2/CUDA3 仍 ~0）

## 6. 不变量

- **旗标 opt-in**：默认路径逐字节不变（`peer_stage_x` 的类型在旗标关时必须仍是 float，
  分配和 D2H 都要走老路 —— **不能只把类型改了再靠 if 分支**，那会让默认路径的
  分配大小和拷贝类型都变，虽然数值相同但不是「逐字节不变」的代码路径）
- **数值路径**：helper 侧仍然量化成 q8_1，MMQ 的输入类型不变
- **行契约 / eparts 行序 / 三个 witness**：不动
- **旧二进制**：`bin/strata-q36`（`b7d53f977`）不动；`bin/strata-q36.krd`（`fa39ea5f`）是本轮的对照基线

## 7. 与硬件的关系

helper 的 0.81 GB/s 是 **Gen1 x4 插槽的链路上限**（pinned+async 与 pageable+sync 同速，
§`2026-10-24-q36-prefill-primary-fp16-review.md` §4.3）—— **软件压不动，只有换槽/换板**。
软件能做的只有**少传字节**，这就是这一轮的全部。
