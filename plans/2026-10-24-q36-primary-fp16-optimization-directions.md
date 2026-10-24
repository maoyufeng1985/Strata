# q36：主卡走 FP16 的优化方向（2026-10-24）

用户的方案：**主卡权重存 Q4，显卡内反量化成 FP16 直接计算**。
本文回答「详细说明后续优化方向」。

**先更正一处我自己的错误**：上一份文档（`...-skinny-expert-shape-review.md` §5）说
「FP16 权重副本 9.14 GiB 装不下」，那是拿**另一个方案**（预先反量化、常驻 FP16 副本）
去卡用户的方案。**用户的方案不需要 FP16 副本** —— VRAM 里只有 Q4（2.86 GiB），
反量化在寄存器里发生，多出来的是每次 launch 几 MiB 的临时缓冲。**那条「装不下」不成立。**

但更正之后，**真正的问题浮出来了**，而且比 VRAM 严重。

## 0. 三个数摆在一起（同一个形状、同一个打包度）

P100 (sm_60)，M=16，gate 投影（N=512, K=2048），E=39 专家一次 launch：

| 方案 | 权重在 VRAM | 每专家 | 读什么 |
|---|---|---|---|
| **int8 DP4A**（引擎现状） | Q4 | **17.43 µs** | 0.56 MiB 权重（int8 尺寸） |
| FP16，权重预先反量化好 | FP16 | 22.33 µs | 2.00 MiB 权重 |
| **FP16，融合反量化**（用户的方案） | Q4 | **65.29 µs** | 0.56 MiB 权重 |

**用户的方案比 int8 慢 3.7 倍**，尽管两者从 VRAM 读**同样多的字节**。
慢的原因不是字节，是**算术**。

## 1. 瓶颈在哪：M 扫描把两件事分开了

融合 Q4→FP16，E=39（`q4fused3.cu`）：

| M（行数/专家） | 每专家 | 带宽 | TFLOP/s |
|---|---|---|---|
| 1 | 5.18 µs | 113.9 GB/s | 0.405 |
| 4 | 65.20 µs | 9.0 GB/s | 0.515 |
| 16 | 65.20 µs | 9.0 GB/s | 0.515 |
| 60 | 232.33 µs | 2.5 GB/s | 0.542 |

**TFLOP/s 几乎不变（0.405 → 0.542），时间却涨了 45 倍。**
kernel 是纯流量受限，算力利用率从头到尾只有峰值的 3%。

### 流量账：激活被读了 M 遍

一个 warp 负责一个输出列，要读这一列的权重（K 个值）和**全部 M 行的激活**（M×K 个值）。
权重读 1 遍，激活读 **M 遍**：

| M | 权重（M=16 时的 gate 段） | 激活（N=512 个输出列各读一遍） | 比例 |
|---|---|---|---|
| 1 | 2.00 MiB | 2.00 MiB | 1× |
| 4 | 2.00 MiB | 8.00 MiB | 4× |
| 16 | 2.00 MiB | 32.00 MiB | **16×** |
| 60 | 2.00 MiB | 120.00 MiB | **60×** |

**M=16 时激活流量是权重的 16 倍，M=60 时是 60 倍。**
这就是「M 越大越慢而 TFLOP/s 不变」的原因。

**int8 DP4A 为什么没这个问题？** 它也不该有 —— 但引擎的 MMQ kernel
**把 M 行激活放进 shared memory**（`s2_expert_grouped.cu` 的做法：一个 CTA 处理多行，
`row_dot_s2_q8` 里 lane 分块读激活），激活只读一遍。
**这是 int8 路径快 3.7 倍的直接原因，不是 DP4A 指令更快。**

## 2. 1Cat-vLLM 的解法（逐行读过的源码）

`benchmarks/csrc/sm70_awq_m5_batched_gemv.cu` 的结构：

| 常量 | 值 | 含义 |
|---|---|---|
| `kM` | 5 | 一次算 5 行激活 |
| `kCtaN` | 4 | 一个 CTA 处理 4 个输出列 |
| `kInterleave` | 4 | 输出列的交错布局 |
| `kThreads` | 128 | 4 个 warp |
| `kCtaK` | 1024 | K 方向的分块 |
| `kTileK` | 64 | 每个 thread 一次处理的 K 长度 |

**四个关键手法，逐个对应我们的瓶颈**：

1. **融合反量化**（`:126-285`）：量化权重**只读一次进寄存器**，
   `convert_u4x8_to_half`（`:90-123`）用 `lop3` 魔数一次转 4 个 half2，
   scale 和 zero-point 用**一次 `__hfma2`** 折进去（`:202-204`）：
   ```cpp
   weights[elem][0] = __hfma2(__halves2half2(q0[elem], q1[elem]), scale01, bias01);
   ```
   **一次 hfma2 同时完成缩放和偏置** —— 我们的版本要两次。
2. **激活在寄存器里复用**：`acc_h[kM][kCtaN/2]`（`:139`）——
   **kM=5 行激活 + 4 个输出列的累加器常驻寄存器**，
   所以每个权重字节只读一次、每行激活只读一次。
3. **warp shuffle 归约 + shared memory**（`:259-284`）：
   `__shfl_xor_sync` 规约到 8 个 lane，再写 shared，最后由 128 个 thread 合并。
4. **split-k 归约用 lock + spin**（`sm70_awq_splitk_reducer_micro.cu`）：
   瘦长形状下 K 方向切分，split 之间用 `ld.global.acquire` / `st.global.release`
   的锁序列化成**部分和相加**（不是求和重排），
   保证**数值确定**（可复现），不引入 atomics 的顺序不确定性。

**1Cat 的 batched GEMV 加速比 2–3×**（对 `torch.mm` 基线，router 层 K=4096、M=1），
来源是 `benchmark_sm70_deepseek_v4_fp16_gemv.py:255-263`（`speedup` 字段）。

## 3. 我们的优化方向（按收益/难度排）

### 方向 1：把 M 行激活放进寄存器，权重只读一遍 【收益最大】

**问题**：激活被读 M 遍（16–60×）。

**做法**（照搬 1Cat 的 `kM`/`kCtaN` 结构）：
- 一个 CTA 处理 **4 个输出列 × 16 行激活**（我们的 M=16 比 1Cat 的 5 大）；
- 16 行激活 × 4 个输出列的累加器常驻寄存器
  （16×4 = 64 个 half2 = 64 个 32-bit 寄存器，太多 → 降到 4 列 × 8 行 = 32 个）；
- **权重每个字节只读一次**，激活每行只读一次。

**预期**：激活流量从 16× 降到 1×，M=16 的每专家时间从 65.29 µs
降到接近权重带宽上限（0.56 MiB @ 732 GB/s = 0.8 µs，加 FMA 算力时间）。
**保守预期 10–20 µs/专家**，即与 int8 的 17.43 µs 同一量级。

**风险**：寄存器压力。`__launch_bounds__` 要给足，16 行 × 4 列的累加器可能溢出到 local memory。
**必须用 `cuobjdump -res-usage` 核对寄存器数**（1Cat 的 kernel 用 `__launch_bounds__(128)`）。

### 方向 2：一次 `__hfma2` 折 scale 和 bias 【容易，收益明确】

我们的 `q4fused3.cu` 现在是：
```cpp
const __half2 coef = __floats2half2_rn(d*sc, -dmin*mn);
const __half2 nib  = __floats2half2_rn(byte & 0xF, byte >> 4);
wq[j] = __hmul2(coef, nib);          // 一次 hmul2
```
这已经接近 1Cat。1Cat 用 `__hfma2(w, scale, bias)` 是因为它的 AWQ 权重还需要加 zero-point。
**Q4_K 的 min 已经在 coef 里了，所以我们的写法等价。**

**但还有一处可省**：nibble 转 half2 现在走 `__floats2half2_rn`（两次 float 转换）。
1Cat 的 `lop3` 魔数（`:99-118`）一次 `lop3.b32` 就把 4 个 4-bit 整数转成 4 个 half2。
**Q4_K 的 4-bit → half2 可以用同样的魔数**（`0x64006400` 那个技巧）。
**预期：反量化算术减半。**

### 方向 3：一次 launch 处理 16 个专家 + stream-k 【调度，不是算法】

引擎的 MMQ 已经有了（`moe_mmq.cu:531`，目标 ≥90% tile 效率）。
FP16 路径要拿到同一个调度结构，**照搬 `mmq_resident_sort.hpp` 的分组 + stream-k 逻辑**，
只换算子。

**预期**：M=60 时单专家 553.20 µs → 88.41 µs（6.3×，实测 `probe4`）。

**但注意**：这条只解决「填不满 SM」，不解决「激活被读 M 遍」。
**方向 1 和方向 3 是乘法关系，不是加法。**

### 方向 4：权重用 `__ldg` / L2 复用 【便宜，可能有惊喜】

同一个 CTA 的 4 个输出列共享激活行 → 激活可以放 shared memory。
但 16 行 × 2048 × 2 B = 64 KiB，超了 shared memory。
**分块：K 方向切 512 一段** → 16 × 512 × 2 = 16 KiB，可进 shared。
**这是 llama.cpp `vec_dot` 的做法**（分块 + shared）。

### 方向 5：Q5_K 的 down 投影单独处理 【必要】

down 投影是 Q5_K（type 13），代码不同（5-bit 解包），
且形状不同（N=2048 输出列、K=512）。
**Q5_K 的 5-bit 解包没有 `lop3` 魔数可用**，得用移位+掩码，比 Q4_K 慢。
**实测占比未知，必须分开量。**

## 4. 数值：这条路会改变结果吗

会。三处变化：
1. **权重从 Q4_K 变 FP16**（Q4_K 反量化本来就是 FP16 值的近似，所以这一步**几乎无损**）；
2. **累加从 int32 变 FP16/FP32**：
   - 引擎的 int8 路径：Q4_K → int8 → DP4A int32 累加，**精确**；
   - 我们的路径：Q4_K → FP16 → FP16 累加，**有误差**（K=2048 项的 FP16 累加误差大）。
   - **必须用 FP32 累加**（1Cat 的 `AccumMode=2`），或者分块累加（每 256 项提升一次精度）。
3. **激活不再量化成 q8_1**：这是**精度提升**（q8_1 量化会掉精度）。

**「a wrong answer is worse than a failed run」** → 必须做 logits 逐行对照。

## 5. 实施顺序

| 步 | 内容 | 预期 | 风险 |
|---|---|---|---|
| 1 | 探针：方向 1（4 列 × 8 行的寄存器版融合 GEMV） | M=16 每专家 65.29 → 10–20 µs | 寄存器溢出，要用 `cuobjdump -res-usage` 核对 |
| 2 | 探针：方向 2（4-bit → half2 的 `lop3` 魔数） | 反量化算术减半 | 数值要对（先做 bit-exact 单测） |
| 3 | 探针：Q5_K 的 down 投影（分开量） | 拿到全专家的完整时间 | 5-bit 没有魔数 |
| 4 | 引擎改动：旗标 `STRATA_Q36_PRIMARY_F16=1`，把 MMQ 的分组+stream-k 套过来 | 组合收益 | 默认路径必须逐字节不变 |
| 5 | 数值验证：logits 逐行对照，argmax 一致数 + max\|diff\| | — | 累加精度 |
| 6 | VRAM：Q4 常驻 2.86 GiB（不变），临时缓冲几 MiB | 无压力 | 低 |

**独立可做（和 FP16 无关）**：下腿 fp32 → fp16/q8_1
（`plans/2026-10-24-q36-down-leg-bytes.md`，+10.7% / +17.0%）。

## 6. 一个可能的更优解（未测）

如果方向 1 做完，融合 Q4→FP16 每专家到了 10–20 µs，
那么**「主卡 FP16 vs helper int8」的分工就变成了纯粹按 PCIe 带宽算**：

- 传一个专家过 PCIe（Q4，1.81 MiB）= **313 µs**（6.07 GB/s）
- 在主卡上算它 = 10–20 µs

**PCIe 比算术贵 15–30 倍。** 所以规则很干脆：
**凡是 PCIe 传得动的专家，放 helper 上用 int8 算（helper 有真 `__dp4a`）；
凡是 PCIe 传不动的（常驻主卡的），主卡用 FP16 算。**

这和「主卡 FP16 / helper int8」的直觉一致，但**依据是 PCIe，不是算力**。

**skip: FP16 权重常驻主卡（不需要，VRAM 不是瓶颈）；
skip: 在 P100 上找 cuBLAS 的「更好的 algo」（24 个扫完，DEFAULT 最优）；
skip: 照搬 1Cat 的 ALGO9_TENSOR_OP（P100 上要么慢 2 倍要么 unsupported）；
skip: 「一次只算一个专家」（实测 6.3× 的损失，方向 3 已覆盖）。**

## 附：本文所有探针

| 探针 | 远端路径 | 状态 |
|---|---|---|
| `q4fused3.cu` | `/tmp/q4fused3` | **可用**（方向 1 的基线，65.29 µs/专家） |
| `q4fused2.cu` | `/tmp/q4fused2` | 可用但有结构体拷贝（106.77 µs） |
| `q4fused.cu` | `/tmp/q4fused` | **grid 错，数字不可用** |
| `probe3.cu` | `/tmp/probe3` | cuBLAS strided-batched，FP16 权重 |
| `probe4.cu` | `/tmp/probe4` | cuBLAS strided-batched，完整三段 |
| `probe5.cu` | `/tmp/probe5` | int8 dp4a，同形状同打包度（17.43 µs） |
| `probe6.cu` | `/tmp/probe6` | 激活重读的流量账 |
| `fp16lt.cu` | `/tmp/fp16lt` | algo × 累加方式交叉扫 |

**所有数字只用于时间，不用于数值。**
