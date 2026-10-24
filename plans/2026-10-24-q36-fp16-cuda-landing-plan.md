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
