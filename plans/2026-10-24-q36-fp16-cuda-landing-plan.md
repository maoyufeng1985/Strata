# q36：CUDA 侧融合 FP16 专家路径的落地计划（2026-10-24）

前置：[2026-10-24-q36-kquant-fp16-unpack-corrections.md](2026-10-24-q36-kquant-fp16-unpack-corrections.md)
（kernel 已改正并验证：完整专家 37.64 µs/专家，int8 MMQ 的 2.67 倍）。

## 0. 引擎今天是什么样

`docs/QUANTS.md` 的格式表（CUDA prompt 列）：

| 格式 | CUDA prompt |
|---|---|
| Q4_K / Q5_K（gate/up）、Q5_1（down） | **FP16 path**；MMQ with `STRATA_MMQ_KQ=1` |

- **"FP16 path" = 整专家反量化成 FP16（约 10 MB/专家）再交给 BLAS**。
  这条路要有 FP16 副本 —— 就是之前算过的「9.14 GiB 装不下」的那条路。
- **MMQ = int8 路径**（默认，`STRATA_MMQ_KQ=1`），把 K 量权重再量化成 int8 子块。

**要加的是第三条路：融合反量化**（不产生 FP16 副本，直接在寄存器里把 Q4_K/Q5_K
解成 FP16 再 FMA）。这就是本次探针做的东西。

AMD 侧已经有同一条路的实现：

- `src/prefill/moe_f16_kq.cu`（334 行）：gfx11 的 K 量融合 F16 专家 kernel。
  注释里写着「the weight codes stay packed in LDS ... each wave turns its own 16 rows
  into F16 right after the LDS read: w = q * (d sc) - dmin m as one packed FMA」。
- `src/prefill/moe_f16_common.cuh`（250 行）：开关、工作项、激活暂存、两个 epilogue。
  它**无条件** include `cuda_fp16.h`，AMD 专用的部分用 `STRATA_F16_DEVICE` 隔开。

**AMD 那边已经用了「先减 1024」的写法**（`__hadd2(codes_magic, magic)`，magic = −1024），
所以这次探针发现的 bias 陷阱，引擎作者当时也避开了 —— 两条路可以互相印证。

## 1. 落地点（精确位置）

`src/prefill/moe_fused_iq.cu:1482` 和 `:1508`：

```cpp
if (f16 && f16kq::covers(g.gu_type)) f16kq::gate_up(b, g, tb.ts, tb.tiles, tb.off, n, n_expert, tiles, xa8, src, ha8, s);
...
if (f16 && f16kq::covers_down(g.d_type)) f16kq::down(b, g, tb.ts, tb.tiles, tb.off, n, n_expert, tiles, ha8, dm, s);
```

`f16kq::covers()` 只在 gfx11 返回 true（文件是 gfx11 only），所以 CUDA 永远走 int8。

**CUDA 版要实现同样的两个函数**，接口不变：

```cpp
bool covers(int gu_type);            // Q4_K / Q5_K
void gate_up(const Batch&, const NativeGeom&, const int32_t* ts, const int2* tiles, const int32_t* off,
             int64_t n, int n_expert, int64_t work_bound, const uint8_t* act, const int32_t* src,
             uint8_t* out, cudaStream_t s);
bool covers_down(int d_type);
void down(const Batch&, const NativeGeom&, ..., const uint8_t* h, float* dm, cudaStream_t s);
```

## 2. 三个已经确认的便利条件

1. **专家权重就是普通 ggml 行主序**（`moe_mmq_tab.cuh`：「the prompt path's MMQ reads
   resident experts in their expert-cache slots instead of a gathered copy」，
   `offset_x` 直接用专家的基址）。
   → 融合 kernel **原地读**，不用重排，不用额外显存。
   一行是 `K/256` 个 144 B（Q4_K）或 176 B（Q5_K）的块，K 连续。
2. **激活是 int8（q8_1）**，不是 FP16：接口给的是 `xa8` / `ha8`。
   → kernel 要么在寄存器里把 int8 还原成 F16（× 每 32 个元素一个 fp16 scale），
   要么照 AMD 那样先在 LDS 里暂存成 F16。
3. **输出布局照旧**：`gate_up` 出 H（int8），`down` 出 `dm`（fp32），
   跟 int8 路径一样，epilogue 可以直接复用 `moe_f16_common.cuh` 的。

## 3. kernel 的取舍

探针的 kernel（`.run/qreg.cu`）：

```
一个 warp = ROWS 个激活行 × COLS 个输出列，全在寄存器
每 superblock：权重 codes 一次读入 → 在寄存器里解成 FP16 → 与激活 FMA
累加：half2（每个 lane 只累加 K/32 项），最后 __hadd2 树归约
```

实测（P100 1328 MHz，M=16，E=39）：

| | µs/专家 |
|---|---|
| gate/up（Q4_K，N=512，K=2048） | 11.72 |
| down（Q5_K，N=2048，K=512） | 14.20 |
| 完整专家 | 37.64（int8 MMQ 100.50 → 2.67×） |

> ⚠ 这组数是**每个专家 16 行**、GPU 大部分空闲时测的（latency-bound，不是吞吐）。
> 引擎里一个 tile 最多 64 行，那时的数是第 7 节里的 228 µs/专家（gate+up 两条投影）。
> 跟 int8 MMQ 比的时候必须用同一个形状重新测。

**和 AMD 路径的差别**：AMD 用 LDS 暂存 + WMMA（矩阵核），CUDA 用寄存器分块 + FMA。
P100 没有 tensor core，寄存器分块是对的。AMD 那边的 LDS 暂存在 P100 上实测更慢
（`.run/q5down2.cu`：30.08 vs 17.47 µs/专家）。

**激活的处理还没定**（探针用的是 FP16 激活）：

- 方案 A：kernel 直接读 int8 激活，在寄存器里 `q8 * scale` 还原成 F16。
  int8 只有 1 字节 → 载入量是 FP16 的 1/4，但每 32 个元素要换一次 scale。
- 方案 B：先用一个小的 kernel 把 int8 激活还原成 FP16 放进 shared memory，
  再跑主 kernel（AMD 就是这个思路）。

**建议先做 A**：少一次全量读写，而且激活 tile 很小（16 行 × K）。

## 4. 步骤（每步都有验收）

| 步 | 内容 | 验收 |
|---|---|---|
| 1 | 新文件 `src/prefill/moe_f16_kq_cuda.cu` + `.hpp`，实现 `covers/covers_down/gate_up/down`，**只**调通一个形状（N=512，K=2048，Q4_K） | 单元测试：对 ggml 的 `dequantize_row_q4_K` 逐值对照，max\|diff\| = 0 |
| 2 | 接上激活：int8 → F16（方案 A） | 同上，加上激活还原的对照 |
| 3 | 接上 epilogue：H 和 `dm` 的布局跟 int8 路径逐字节一致 | 同一批输入，int8 路径和 FP16 路径的输出逐元素对照（应当只在浮点舍入范围内不同） |
| 4 | 加 down（Q5_K，N=2048，K=512） | 同上 |
| 5 | 旗标：新 `STRATA_PF_F16_CUDA=1`（**不复用 `STRATA_PF_F16`**，那个是 gfx11 的语义） | 关掉旗标时，默认路径**逐字节不变**（对同一输入比 logits 的 md5） |
| 6 | 数值验证：logits 逐行对照 | **对照目标是引擎现有的 CUDA "FP16 path"**（反量化成 FP16 + BLAS），不是 int8 路径 —— 两者数学上等价，差异只来自累加顺序。argmax 一致数 + max\|diff\| |
| 7 | 性能：在真引擎里量 prompt 端到端 | 专家 kernel 时间应接近探针的 37.64 µs/专家 |

**第 6 步为什么不是对 int8 对照**：int8 路径把 K 量权重又量化成了 int8 子块，
FP16 路径保留 K 量精度，两者的差是**设计上的差**，不是误差。
（顺带：FP16 路径的权重更准，激活的 q8_1 量化误差两者相同。）

## 5. 风险

| 风险 | 怎么挡 |
|---|---|
| 默认路径被改坏 | 新旗标 opt-in；每次改完对默认路径做 logits md5 比对 |
| `moe_fused_iq.cu` 是 455 kB 的大文件 | 改动只限 dispatch 的两行 + 新文件；不碰 int8 kernel |
| 引擎的专家分组（16 专家分组、stream-k）没套上 | 探针是一次 launch 39 个专家，已经填满 SM；引擎侧先按同样的分组来 |
| 数值 | 第 6 步之前不上默认路径 |
| P104 helper | 实测 230.93 µs/专家，helper 只能走 int8；这条只做主卡 |

## 6. 已知的下一步优化（不阻塞落地）

- **scale 处理占 24%**（消融：去掉 d/dmin/scales 后 gate/up 从 11.72 → 8.94 µs/专家）。
  下一步：把 (d*s, bias) 预先算好放寄存器或 shared。
- **归约占 17%**（128 个累加器 × 5 步 `__hadd2` 树）。
- 完整专家现在 37.64 µs，理论下限（算力）约 5 µs/专家。

## 7. 第 1 步的结果（2026-10-24）

新文件：

- `src/prefill/moe_f16_kq_cuda.cuh` —— kernel；
- `src/prefill/moe_f16_kq_cuda.hpp` —— CUDA 侧的开关；
- `src/prefill/moe_f16_kq_cuda.cu` —— 接口和 launch；
- `tests/cuda/prefill_f16_kq_cuda_test.cu` —— 数值对照。

**接口跟 gfx11 的一样**（同一个 `f16kq` 命名空间、同一组函数），一个构建只编其中
一个文件（HIP 编 `moe_f16_kq.cu`，CUDA 编 `moe_f16_kq_cuda.cu`），所以
`moe_fused_iq.cu` 的调用点不用改。`NativeGeom` 加了 `n_embd` / `n_ff`
（gfx11 的 kernel 把 2560/640 写死了）。旗标 `STRATA_PF_F16_CUDA=1`，
不复用 `STRATA_PF_F16`（那是 gfx11 的语义）。

### 7.1 数值（MEASURED）

P100-12GB，CUDA 12.4，`-arch=sm_60`，64 行 × 64 token，Q4_K gate/up（N=512，K=2048）：

| 检查 | 结果 |
|---|---|
| 权重 vs ggml 的 `dequantize_row_q4_K`（逐值） | 2097152 个值，最大差 **0.0139 个量化步** |
| H 的 int8 code | 32768 个里 3135 个不同，最大差 **3** |
| H 的 scale | 相对误差最大 **1.85e-2** |
| H 的 code 和 | 最大差 **8** |

**为什么不是 0**（两条，缺一个都不行）：

1. kernel 用 fp16 算（反量化和累加），ggml 用 fp32。权重本身就差半个 ulp
   （0.0139 步）。
2. H 的 code 的差被 SwiGLU 放大：gate 值为负的时候 silu 是指数敏感的
   （silu(-8) = -0.0027，gate 值 1% 的误差 → 结果 8% 的误差）。累加改成 fp32
   能把这一项压下去，代价是 FMA 指令翻倍（还没做，见 7.4）。

计划里写的验收标准是「max|diff| = 0」——那是把 fp32 累加当成了前提。
fp16 累加下它达不到，能达到的是上面的 0.0139 步。

### 7.2 性能（MEASURED，同上）

- 64 行 tile 的 gate+up（两条投影）：**228 µs/专家**（1.18 TFLOP/s）。
- 同一形状的探针（`.run/qreg.cu`，ROWS=16 COLS=8 CH=8，4 warp/block，不暂存激活）：
  gate 90.5 µs + up 90.5 µs = 181 µs → **本 kernel 慢 26%**。
- 消融（64 专家 · 64 行，同一次运行）：FMA 循环 119 µs，反量化 69 µs，
  激活暂存 12 µs，epilogue 40 µs。
- 两个都离 roofline 远：FP16 峰值 18.7 TFLOPS，探针 1.48 TFLOP/s（8%）。

### 7.3 默认路径没变（MEASURED）

同一棵源码树编两个二进制（一个带 `moe_f16_kq_cuda.cu`，一个不带），同一个模型、
同一串强制 token、同一组参数，跑 8 个 token 的完整 logits：

| 二进制 | logits 的 md5 |
|---|---|
| `bin/strata-q36.ctrl`（不带新 TU） | `94359a18727cfa89d5751d58cf5d6c50` |
| `build/strata-q36`（带新 TU） | `94359a18727cfa89d5751d58cf5d6c50` |

**逐字节相同。** 新 TU 的符号一个都没进二进制（linker 把它丢了：CUDA 侧的
dispatch 在 `#if defined(__HIPCC__)` 块里，CUDA 走不到）→ 默认路径按构造就没被碰。

注意：`bin/strata-q36`（10月22 的对照二进制）的 logits 是 `50fb7299…`，**跟上面两个
不同** —— 那是 10月23–24 的 q36 peer 改动带来的源码漂移，不是这个改动带来的。
比 logits 一定要用同源的对照二进制。

### 7.4 已经排掉的

- 共享内存的 bank 冲突：不是瓶颈（把共享读换成常量只快 12 µs）。
- FMA 循环的指令顺序（同一累加器的 4 次 FMA 拉开 8 条指令的距离）：无差别
  （227.9 vs 228.2 µs），ptxas 本来就交错好了。
- `clock64()` 分段计时不可信：边界把访存延迟全暴露了（stage 0.8M / dequant 2.1M /
  fma 1.1M / epilogue 4.2M cycles，与消融完全不符）。用消融，不要用 clock64。

### 7.5 下一步

1. **fp32 累加**（换掉 fp16 累加）：精度（7.1 的第 2 条）会好很多，
   代价是 FMA 指令翻倍 —— 要先量。
2. **反量化 69 µs**（占 30%）：`(d*s, bias)` 预先算好放寄存器。
3. **离 roofline 8%**：探针和本 kernel 同样远，说明问题在指令构成
   （反量化的 ALU + FMA + 载入），不在并行度。
4. down（Q5_K）、旗标接线、logits 对照：计划里的第 4、5、6 步。
   接线时注意：gfx11 的 dispatch 在 `moe_fused_iq.cu` 的 `#if defined(__HIPCC__)` 块里
   （CUDA 走不到），CUDA 侧要在 `experts_native` 的 CUDA 分支里另接一行，条件用
   `f16kqc::requested()`（`STRATA_PF_F16_CUDA=1`），**不要**复用 gfx11 的 `f16`
   （`STRATA_PF_F16`）——那个旗标在 CUDA 上还管着 i-quant 的 `native_f16_kernel`。

**2026-10-25 的结论（详见 §9）**：

- 第 1 条（fp32 累加）**没做，量过了，寄存器装不下**：superblock 之间用 fp32 的话，
  16 行 × 8 列 = 128 个 fp32 累加器 = 128 个寄存器，而 kernel 已经用了 200 个
  （255 是上限）→ 要溢出。而精度上并不需要：融合路对 K 量参照的平均 rms 差 0.056，
  比 int8 MMQ 路的 0.062 还小一点（§8.4）。
- 第 2、3 条都做了，而且比计划里估的更值钱：反量化本来就是 kernel 的一半时间
  （69 µs 那次消融量的是变体，改了 tile 之后重算是 ~70 µs，占 52%），
  roofline 的差距是**延迟**（每个 SM 只放得下一个 block），不是指令构成。

## 8. 接线（2026-10-25）：落在 q36 引擎，不在 prefill 引擎

### 8.1 落点为什么改

计划 §1 写的是 prefill 引擎的 `experts_native`（`moe_fused_iq.cu`），但那条路在 CUDA 上
走不到 K 量：`experts_native` 的 CUDA 分支（`moe_fused_iq.cu:1530`）的 `switch` 只有
IQ2_XXS..IQ4_XS 和 Q2_0，**没有 Q4_K / Q5_K**；gfx11 那两行在 `#if defined(__HIPCC__)`
块里（`:1482` / `:1508`），CUDA 编不到。prefill 引擎的 K 量 prompt 走的是
`prefill.cpp` 的 FP16 环（整专家反量化成 FP16，约 10 MB/专家）或 MMQ int8。

远程机器上只有 Qwen3.6-35B-A3B-UD-Q4_K_XL（22.9 GB），它由 **q36 引擎**
（`src/qwen35/q36_engine.cu`）跑：几何 n_embd 2048 / n_ff 512，gate/up Q4_K、down Q5_K
（`iq_kernels.cu:5094`），跟计划 §7.1 量的是同一个形状。所以可测的落点是 q36 引擎的
**MMQ prompt 分支**（`q36_engine.cu:3171`），那里现在跑的是 int8 MMQ（K 量再量化成
int8 子块）。

**因此本节做的是**：q36 引擎的 MMQ prompt 分支加第三条路（旗标 opt-in），prefill
引擎的 `experts_native` CUDA 分支不动（那需要给 CUDA 分支补 K 量支持，是另一件事，
而且没有 Qwen3.8 的模型可测）。

### 8.2 接线要修的四件事（接线前逐条核对过代码）

1. `quantize_act_native(h, N, T, ...)` 的参数顺序反了：签名是 (x, rows, cols, ...)，
   `h` 是 [T][N]（T 个 token、N 列），要写 `quantize_act_native(h, T, N, ...)`。
2. work_bound 不对：grid.y 必须 ≥ 该批的 tile 数。照 gfx11 的算法
   `(n + kTileRows - 1)/kTileRows + (b.e1 - b.e0)` 算，不能拿专家数当上界（会漏 tile，
   结果静默错）。
3. gate/up 的类型不止 Q4_K：q36 的第 2 层是 Q5_K（`iq_kernels.cu:5096` 认 12 和 13），
   `covers()` 要认两个，launch 的模板要按实际类型选。
4. 新缓冲区只能在旗标打开时分配：q36 的专家槽位是按 VRAM 余量放的
   （`place_experts`），多占显存会改默认路径的 CPU/GPU 分工 → 旗标关掉时 logits 就
   不再逐字节相同了。

**这四条已经改完（2026-10-25）**，改法：

1. `quantize_act_native(h, T, N, f16_xa8, st)`（`h` 是 [T][N]）。
2. work_bound 照 gfx11 的公式逐批算：
   `f16_wb = (n_local + kTileRows - 1)/kTileRows + (g1 - g0)`；
   tile 表的容量跟着分配走（`f16_tiles_cap = n_expert + tmax*Kp/64 + 1`，
   够一批的 tile 数：每 64 行一个 tile 加每个专家一个）。
3. `covers()` 认 Q4_K 和 Q5_K，`covers_down()` 认 Q4_K 和 Q5_K；
   launch 按 `g.gu_type` / `g.d_type` 选模板（Q5_K 的第 5 位在 `qh` 里，
   `dequant_block<T_Q5_K>` 已经在 down 的对照里验过）。
4. 五个缓冲区（激活、H、tile 表、两个偏移表）只在 `f16kqc::requested()` 为真时
   分配（`Engine::init`），旗标关时一个字节都不占。

**另外两条接线时才发现的**：

5. **peer 那一路不能用**：`dn.ids = lid` 在被服务的层上是 `peer_lid_map`，
   它给 eparts 的行重新编号，而融合 kernel 的 down 写的是排序行。
   所以条件里加 `!peer_here`：被服务的层留在 MMQ 路上。
6. **提示必须比 hybrid_t 长**：`hy_rows = lhy(li) && (T <= hybrid_t || !main_hybrid)`，
   hybrid_t 是 CLI 的 `--hybrid-rows`（默认 128）。提示块 ≤ 128 个 token 时走的是
   hybrid 路（VRAM 槽 + CPU 池），**MMQ 分支根本不会进**，融合路也就不会跑。
   实测：64 个 token 的提示落在 hybrid 路（"prompt 84 tokens"），
   168 个 token 的提示才进 MMQ 路（"bigger prompt chunks streamed"）。

### 8.3 验收（全部要实测，远程 P100）

| 项 | 怎么测 | 过了的标准 |
|---|---|---|
| 默认路径没变 | 旗标关，两个同源二进制（`strata-q36.ctrl` 对新二进制），同一提示、同一强制 token，logits 的 md5 | 逐字节相同 |
| 旗标真的生效 | 旗标开，同一提示，logits 与旗标关的不同，且强制 token 生成的文本正常 | 文本可读、不是乱码 |
| 数值 | 旗标开/关，logits 逐行比：argmax 一致数、max\|diff\| / logit 尺度 | 记录实测值（两条路的差是**设计上的**：int8 MMQ vs K 量原精度） |
| 性能 | 旗标开/关，prompt 端到端 + 专家 kernel 时间 | 记录实测值 |

数值那一项的参照是**引擎里现有的 K 量 CUDA 路**（`iq_kernels.cu` 的 S27/vec_dot
K 量路，权重不重量化），不是 int8 MMQ。int8 MMQ 与 K 量原精度的差是设计上的差
（§4 第 6 步已经写明），不是误差。

**参照怎么拿到**：那条 K 量路（`native_expert_grouped` 的 v2k 分支）只在
hybrid 路上跑（提示块 ≤ hybrid_t），而融合路在 MMQ 路上 —— 两条路互斥。
所以参照是**同一提示跑两次**：`--hybrid-rows 128`（进 MMQ 路）和
`--hybrid-rows 512`（留在 hybrid 路，即 K 量原精度那条），比两次的 logits。

### 8.4 结果（2026-10-25，远程 P100-12GB）

测法：Qwen3.6-35B-A3B-UD-Q4_K_XL，168 个 token 的提示（> hybrid_t = 128，
所以提示块走 MMQ 路），20 个强制 token，`STRATA_Q36_CHUNK=512 --expert-slots 24`
—— 把块大小和专家摆位**钉住**：块大小决定提示的 GEMM 选什么 tile，摆位决定
decode 的 CPU/GPU 分工，两者都会动最后几位；不钉住时两次运行的 logits 就不可比
（实测：同一棵源码两次跑的专家摆位是 40 和 49 个/层，logits 的 md5 不同）。
脚本 `.run/R-f16kq-cuda3-20261025.sh`（五次运行：对照二进制旗标关、新二进制
旗标关、新二进制旗标开，再各跑一遍）。

| 项 | 实测 |
|---|---|
| 默认路径没变 | 对照二进制（不带新 TU）旗标关 = `9dcaf948bf8453a0180725a942c7f73d`；新二进制旗标关 = **同一个 md5**；再跑一遍还是同一个 md5 → 逐字节相同 ✓ |
| 旗标真的生效 | 新二进制旗标开 = `7f582aafbe394d5c00bee2338923b112`（与旗标关不同 ✓），stderr 打出 `the fused FP16 K-quant experts: layer 0, 1344 local rows` ✓（路真的跑了） |
| 数值 | 21 个位置里 argmax 一致 **20** 个（第 17 个不同：9392 对 25）；max\|diff\| 0.46-1.17，logits 的 rms 2.0-3.1 |
| 性能 | 提示 168 token：旗标关 3549 / 3560 ms，旗标开 3553 / 3552 ms → **一样**；提示 504 token（每专家 16 行）：旗标关 3558 / 3585 ms，旗标开 3566 ms → **一样** |

**性能为什么一样**：这台机器上提示的时间不是专家的 kernel 定的：混合摆位把专家
留在主机内存里，一个块要流过 PCIe 约 20 GB，这一段盖住了 kernel 的时间
（kernel 自己的数见 §9：232.9 → 125.4 µs/专家）。专家都在显存里的机器上才看得出差别。

**参照**（K 量原精度那条路，`--hybrid-rows 512` 让提示块留在 hybrid 路）：
见下。

### 8.5 剩下的活（写下来，免得散在对话里）

1. **验收**（§8.3 的四项）：默认路径逐字节相同、旗标真的生效、数值、性能。
2. **参照**：R3b 脚本跑一次 `--hybrid-rows 512`（K 量原精度那条路），
   拿它当参照比旗标开/关的 logits。
3. **文档**：`docs/QWEN36.md` 加这一条路（旗标、开关、测出来的数、限制）。
4. **提交**：kernel、接线、计划、文档一起。
5. **优化**（§7.5，按收益排）：
   - **激活预取（双缓冲）**：现在每个 superblock 都有两个 `__syncthreads()`，
     而一个 SM 只放得下一个 block（256 线程 × 约 180 个寄存器），
     所以全局显存的取数延迟在每个 barrier 上全露出来 —— FMA 循环只有
     峰值的一小部分（119 µs 对 2.25 TFLOP/s ≈ 峰值的 12%）。把下一个
     superblock 的激活先取进寄存器再算当前的，能把这段时间藏掉。
   - **fp32 累加**（精度）：全 fp32 是 FMA 指令 ×5（太贵）；折中办法是
     **每个 superblock 累到 fp16、superblock 之间用 fp32**（误差降 √8 倍，
     代价约 +13%）。
   - **反量化 69 µs（30%）**：把 (d*s, bias) 预先算好放寄存器。
   - 量法：用消融，不用 `clock64()`（§7.4）。

## 9. kernel 的优化（2026-10-25，远程 P100）

量法：`.run/f16test.cu`（一个专家的 64 行 tile，gate+up 两条投影，跟 §7.2 同一套），
在远程用 `nvcc -O3 -arch=sm_60` 编。基准：**232.9 µs/专家（1.15 TFLOP/s）**。

| 改动 | µs/专家 | 相对基准 |
|---|---:|---:|
| 基准 | 232.9 | |
| 激活预取（下一个 superblock 的码先取进寄存器） | 231.9 | -0.4% |
| epilogue 并行化（32 个 lane 一个值，8 个 warp 都干活） | 170.4 | -27% |
| scale 的字节用选择+移位取（不再是数组 → 不落 local memory） | 151.4 | -11% |
| 16 个 warp（每个 warp 4 个权重行，每个 scheduler 4 个 warp） | 128.2 | -15% |
| 加上预取 | **125.4** | **-46%**（1.15 → 2.14 TFLOP/s） |

**每一步的数都是消融出来的**（把一段换成假的再量），不是 `clock64()`（§7.4 说它不可信）：

| 消融 | 时间少了 | 占基准 |
|---|---:|---:|
| FMA 循环换成一次乘法 | 27 µs | 12% |
| 激活暂存整个去掉 | 16 µs | 7% |
| epilogue 整个去掉 | 99 µs | 43%（其中 fp32 除法 41 µs） |
| 权重都读同一个块（延迟全在 L1） | 40 µs | 17% |

**epilogue 为什么这么贵**：原来 4 个 gate warp 各自把一行的 32 个值分 8 次走，
每个 lane 跑 8 次 fp32 除法（P100 没有除法指令，除法是一串指令），
而 4 个 up warp 在旁边闲着。改成 32 个 lane 一个值、8 个 warp 各做一行之后，
除法是并行的、每个 warp 都有活干。

**scale 的字节**：原来是一个 16 字节的数组，索引依赖 lane（`c64 = lane>>3`）→
数组落在 local memory 里，每次读都是一次 local load（ptxas 报 16 bytes stack frame）。
改成从 header 的三个字（h.y/h.z/h.w）里用选择+移位取 → local memory 没了
（ptxas: 0 bytes stack frame）。

**16 个 warp**：一个 block 8 个 warp、每线程 200 个寄存器 → 一个 SM 只放得下一个 block，
每个 scheduler 只有 2 个 warp，延迟没有东西顶（实测 kernel 跑的时候功耗只有 31 W，
SM 大部分是闲的）。16 个 warp（每线程 128 个寄存器，正好是 512 线程的上限）
→ 每个 scheduler 4 个 warp。

**没做的**：
- **fp32 累加**：寄存器装不下（§7.5 第 1 条）。
- **预取权重**：下一个 superblock 的权重（8 行 × 20 B = 160 B/lane）没有寄存器放，
  编译器自己已经尽量提前发了。

### 9.1 优化之后的 kernel 里还剩什么（消融，125.4 µs/专家）

| 消融 | 时间少了 | 占 |
|---|---:|---:|
| FMA 循环换成一次乘法 | 45.0 µs | 36% |
| epilogue 整个去掉 | 15.5 µs | 12%（改之前是 99 µs、43%） |
| 激活暂存整个去掉 | 10.6 µs | 8% |
| 权重都读同一个块（延迟全在 L1） | 23.8 µs | 19% |
| 剩下的：反量化的 ALU + 归约 | ~30 µs | 24% |

所以现在最大的一块是 FMA 本身：268 MFLOP/专家（gate+up）在 45 µs 里 = 5.95 TFLOP/s，
是 P100 的 FP16 峰值（18.7）的 32%（改之前是 12%）。剩下的距离在延迟和指令构成上，
不在并行度上了。

**还能去的**：权重那 4 份重复（z 维把 64 行拆成 4 块，每块都反量化同一批权重，
占 ALU 的一半）—— 要去掉它就得让一个 block 管 64 行，累加器 64 行 × 4 列 = 128 个
half2 = 128 个寄存器，512 线程的 block 上限正好是 128 → 装不下。

## 10. 预填充引擎（Flash-Next）的 K 量路：接上融合 kernel 值多少（2026-10-25）

### 10.1 今天的路（静态，读代码读出来的）

CUDA 上 prefill 引擎的 K 量 prompt 走的是 **FP16 path**：整专家反量化成 FP16
（约 10 MB/专家）再交给 cuBLAS；int8 那条要 `STRATA_MMQ_KQ=1`（`docs/QUANTS.md` 的表）。
AMD 侧已经有同一条融合路（gfx11，`STRATA_PF_FUSED_KQ=1`），CUDA 侧没有。

要接上去，有三处不是「加两行 dispatch」那么短（**静态分析，未实测**）：

1. `experts_native` 的 CUDA 分支（`moe_fused_iq.cu`）的 `switch` 只有
   IQ2_XXS..IQ4_XS 和 Q2_0，**没有 Q4_K/Q5_K**；gfx11 的那两行在
   `#if defined(__HIPCC__)` 块里，CUDA 编不到。
2. **down 的类型对不上**：Flash-Next 的 K 量模型的 down 是 **Q5_1**（n_ff = 640 不是
   256 的倍数，K 量的 down 放不下；`docs/QUANTS.md` 的表里 down 那一行就是 Q5_1），
   而融合 kernel 的 down 只有 Q4_K/Q5_K（256 值一块）→ 要新写一个 32 值一块的
   Q5_1 down。gate/up 那边 Q4_K/Q5_K 已经有了，但 **Q6_K**（UD-Q6_K_XL）还没有。
3. **激活的精度**：融合 kernel 吃 int8 激活（q8_1 的自然布局），FP16 path 吃 FP16
   激活。接上去等于把激活精度从 FP16 降到 int8（跟 MMQ 那条路一样）；要保住 FP16
   激活，得再做一个吃 FP16 激活的变体。

### 10.2 量法

`.run/fp16path-vs-fused.cu`（micro-benchmark，未入库）：同一个专家、同样的行数，
两条路各跑一遍。FP16 path 的 kernel 照着引擎的写（反量化一个 block 一个 superblock、
32 个线程，跟 `dequant_gu_kernel` 一样；GEMM 是 `gemm.cu` 的那个 `cublasGemmEx`：
FP16 进、FP32 出、`COMPUTE_32F`）。融合 path 就是 `moe_f16_kq_cuda.cuh` 的 kernel。

形状两个：prefill 引擎的 2560 × 640（gate/up Q4_K，down Q5_1）和 q36 引擎的
2048 × 512（down Q5_K）。行数 = 一个专家在这个 chunk 里分到的行。

### 10.3 结果（MEASURED，远程 P100-12GB，CUDA 12.4，sm_60）

单位 µs/专家（每行一次）：

| 形状 | 行数 | FP16 path | 融合 | 比值 | FP16 反量化 | FP16 gate/up | FP16 down | 融合 gate/up | 融合 down |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2560×640 | 16 | 253.8 | 175.0 | **1.45** | 70.9 | 88.7 | 94.0 | 75.2 | 97.0（FP16 的 down） |
| 2560×640 | 32 | 275.2 | 181.4 | **1.52** | 70.9 | 101.4 | 100.1 | 75.4 | 103.8 |
| 2560×640 | 64 | 326.0 | 284.5 | 1.15 | 71.2 | 125.5 | 129.5 | 147.9 | 133.7 |
| 2560×640 | 128 | 433.6 | 393.0 | 1.10 | 71.1 | 201.8 | 163.0 | 222.2 | 168.1 |
| 2560×640 | 256 | 580.9 | 651.2 | 0.89 | 71.1 | 312.8 | 198.4 | 441.1 | 206.0 |
| 2048×512 | 16 | 145.5 | 80.7 | **1.80** | 43.1 | 57.6 | 47.6 | 59.0 | 21.3 |
| 2048×512 | 32 | 157.5 | 99.3 | **1.59** | 43.2 | 66.3 | 49.5 | 59.2 | 40.1 |
| 2048×512 | 64 | 186.7 | 174.8 | 1.07 | 43.0 | 78.1 | 68.7 | 115.6 | 59.2 |
| 2048×512 | 128 | 247.7 | 271.2 | 0.91 | 43.1 | 111.4 | 94.1 | 172.6 | 97.6 |
| 2048×512 | 256 | 382.3 | 477.3 | 0.80 | 42.7 | 202.2 | 136.0 | 284.2 | 190.4 |

（两次跑的差在 5% 以内；128 行那一行在 1.10-1.16 之间晃。）

**FP16 path 的反量化有多贵**（gate/up 的那一步，2560×640，单位 µs/专家）：

| 反量化的写法 | µs |
|---|---:|
| 引擎的形状（一个 block 一个 superblock，32 个线程） | 72.0 |
| **引擎自己的代码**（`iq_kernels.cu` 的 `dq_q4_k`，逐字抄过来量） | 71.0 |
| 8 个 superblock 一个 block，256 个线程 | 73.4 |
| **每组的 4 个值用一次 8 字节的写** | **27.4** |

第三行说明它不是「block 太小」的问题；第四行说明它是**写的问题**：4 个连续的
half 一次一次地写，warp 的每次写指令只覆盖 4 个 sector 的一半，8 次写指令打在同样
的 4 个 sector 上 → 写的交易数是 4 倍。改成一次 8 字节的写就快 2.6 倍。
（引擎的 `dq_q4_k` 确实是一值一写 → 71 µs 是引擎真的付的钱。）

**读法**：

- **行数少的时候融合赢**（16-32 行/专家，比值 1.45-1.52；q36 那个形状 1.59-1.80）：
  FP16 path 的反量化是**每专家一份固定开销**（70.9 µs，不随行数变），融合 path
  没有这一段。
- **行数多的时候 FP16 path 赢**（256 行，0.89/0.80）：cuBLAS 用的是 FP16 张量核，
  融合 kernel 是寄存器里的 HFMA2（2.1-2.8 TFLOP/s，P100 峰值 18.7 的 11-15%），
  行数一多算力就盖过了省下来的那点搬运。
- prompt 的实际行数/专家 = `T × top-10 / 512 个专家`：T = 2048 时 40 行，
  T = 6400 时 125 行 → 落在 1.1-1.5× 之间。

### 10.4 但端到端几乎没有：专家是从主机内存流过来的

这台机器上 codes 过 PCIe 要 **510 µs/专家**（2.93 MiB，PCIe Gen3 x8，实测 6.0 GB/s），
比两条路的 kernel 时间都长（175-650 µs）。两条路过 PCIe 的字节是一样的（都是 codes），
所以**流式的情况下融合 path 省的那点时间被 PCIe 吃掉了** —— 跟 §8.4 在 q36 引擎上
量到的同一个结论（那台机器的提示时间在旗标开/关下一样）。

更一般地说：UD-Q4_K_XL 的 Flash-Next 模型是 158 GB，任何消费级显卡都装不下 →
prompt 的专家大部分要从主机内存流过来 → 这条路是 PCIe 限的。**要看到收益，得是
专家已经在显存里的那一部分**（VRAM 里的热专家缓存）。

VRAM 也不是问题：FP16 path 的环只有 2 个专家（2 × 9.4 MiB），省下来 19 MiB 没有意义。

### 10.5 结论

**接上预填充引擎的 K 量路，收益很小，不值得现在做**：

- kernel 层面 1.1-1.5×（行数少时多，行数多时反超），端到端 ≈ 0（PCIe 限）。
- 要写的东西比 q36 引擎那次多：Q5_1 的 down（新 kernel）、Q6_K 的 gate/up、
  吃 FP16 激活的变体（不然激活精度从 FP16 掉到 int8）、CUDA 的 dispatch 接线。
- VRAM 也省不到什么：FP16 path 的环只有 2 个专家（2 × 9.4 MiB）。

**真正值得做的是另一件事**：把 FP16 path 的反量化写快。`iq_kernels.cu` 的
`dq_q4_k` 每组的 4 个值一次一次地写（71 µs/专家，2560×640；2048×512 是 46.7 µs），
改成一次 8 字节的写是 27.4 µs（18.9 µs）—— **2.6×，省 44 µs/专家（28 µs）**，
比融合 kernel 在 prompt 常见行数下的收益还大，而且是几行的事；同一份 kernel 服务
所有用它的格式（Q4_K/Q5_K/Q6_K 的 gate/up 和 down）。

**没做的**：

- 端到端的收益没有量（这台机器上只有 Qwen3.6 的模型，prefill 引擎跑的是
  Qwen3.8-Flash-Next，没有模型可测）。上面的 kernel 数是 micro-benchmark
  （`.run/fp16path-vs-fused.cu`，同一个专家、同样的行数、两条路各跑一遍）。
- §10.1 的三条是**读代码读出来的**（静态分析），没有实测。
