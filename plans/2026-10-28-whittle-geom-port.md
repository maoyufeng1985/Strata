# Whittle-45B：把整个引擎的几何参数化（2026-10-28）

## 0. 现状

模型能加载（`session is up`），停在 `prefill: geometry differs from the artifact's`。
上游 v0.2.7 的 `kCompiledGeometry` 与 v0.2.2 完全一样（Flash-Next 专用），**没有捷径**，必须逐文件移植。

## 1. 尺寸映射（Flash-Next → Whittle）

| 字面量 | 含义 | Whittle | 来源 |
|---:|---|---:|---|
| 2560 | N = n_embd | 2048 | `kCompiledGeometry.n_embd` |
| 640 | FF = n_ff | 512 | `kCompiledGeometry.n_ff` |
| 1280 | GW = 2*FF | 1024 | `2*FF` |
| 10240 | D = N*HC **且** C = ssm_conv_channels | 8192 | 两者在两个模型里都相等 ⇒ 数值无歧义 |
| 6144 | ZV = ssm_value_dim **且** n_head*head_dim | 4096 | 同上 |
| 24 | NH = n_head | 16 | `kCompiledGeometry.n_head` |
| 48 | HV = ssm_v_heads | 32 | `kCompiledGeometry.ssm_v_heads` |
| 512 | n_expert | 256 | 运行期值（`kCompiledGeometry.n_expert` 仍是 512，作容量上限）|

`head_dim=256`、`idx_key_dim=128`、`idx_q_heads=4`、`hc=4`、`hc_lr=320`、`ssm_state_size=128`、
`ssm_k_heads=16`、`idx_block=4`、`n_rot=64` **都不变**。

## 2. 树的选择

远端树 = v0.2.2 + Q36 工作 + 已打的 Whittle 补丁；本地树 = v0.2.6 + 105 个 Q36 提交 + 已打的 Whittle 补丁。
差 577 个上游提交（src 40 文件 / include 47 文件）。**决定：推本地树**（v0.2.6 更新、含全部 Q36 优化），
远端旧树保留为回退。pack 读取代码未变 ⇒ `/home/Strata/packs/whittle45b` 可直接复用。

远端布局：
- `/home/Strata/src-flash` = 新的本地树
- `/home/Strata/build-flash` = 新构建目录（`-DSTRATA_ISA_FLOOR=avx -DSTRATA_ENABLE_CUDA=ON
  -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES="60;61" -DSTRATA_GGML_DIR=<已取到的 pinned llama.cpp>`）
- `/home/Strata/src` + `/home/Strata/build-avx` = 回退，不动

## 3. 实际改了哪些文件（19 个，+259/-162）

| 文件 | 改动 |
|---|---|
| `include/strata/core/geometry.hpp` | n_embd 2048、n_layers 40、ssm_v_heads 32、conv 8192、value 4096、n_head 16、n_ff/n_ff_shexp 512；idx_top_k 改成"预算不是形状"的注释 |
| `include/strata/artifact/gguf_reader.hpp` | 把 `attention.indexer.top_k` 从 pin 表里去掉（它是预算，宽度 = `min(n_kv, top_k+block-1)`）|
| `include/strata/kernels/cpu/expert.hpp` | H 2048、FF 512、NE 256（CPU 专家路径）|
| `include/strata/kernels/ngram.hpp` + `src/kernels/ngram.cpp` | PLE：行宽 256、每 n-gram 4 头（总 8 头）、39,040,000 行、新格式 Q4_K（144 B/行）、Whittle 的 hash 常数 |
| `include/strata/kernels/qsa.hpp` | `qsa_real_shapes()` 从 `kCompiledGeometry` 取（原来钉死 n_head=24）|
| `include/strata/core/remote_experts.hpp` | open() 的默认 n_embd/n_ff/k |
| `include/strata/plan/plan.hpp` | 层数、QSA 层数、GDN value heads |
| `src/prefill/prefill.cpp` | 头部的 N/HC/D/LR/C/ZV/HV/NH/QHD/FF/GW/QW/K/NE 全部从 `kCompiledGeometry` 取；68 处 640→FF、30 处 1280→GW、2560→N、12288→QW；`T*24`→`T*NH`；几何检查从 `ss.k != K` 放宽成范围检查（K/NE 是运行期值）；加 `static_assert(ZV == NH*QHD)` |
| `src/prefill/kernels.cu` | 头部的形状；`split_q_kernel` / `gate_attn_kernel` 的 24 heads / 512 → NH / 2*QHD |
| `src/kernels/cuda/fused_gr.cu` | N=2048、H_TILE=1024、Q8_KC=D/16、down 的 tile 用 N 与 N/2（原来钉死 2560/1280）|
| `src/kernels/cuda/iq_kernels.cu` | D_FMTS 加 Q4_K(12)（Whittle 有 20 层 down 是 Q4_K）；S26_XMAX/S26_HMAX 从几何取 |
| `src/kernels/cuda/ple.cu` | static_assert 从字面量改成不变式 `PLE_N_HEADS*PLE_HEAD_DIM == NG_N_EMBD` |
| `src/kernels/cuda/native_ple_postops.cu`、`s2_expert_grouped.cu`、`qsa_decode_attn.cu`、`qsa_prompt_attn.cu` | N/H/FF/G 改成几何值（G 12→8）|
| `src/kernels/cpu/native_expert.cpp`、`src/core/remote_expert_opt.cu` | 自检的行宽 |
| `src/program/generate.cpp` | 层数 48→`n_layers`；PLE 探针的输出宽度 2560→`n_embd`；**新增 max_context 守卫**（indexer 的选择预算必须装得进单块 top-k 的容量）|

## 4. 死路径不动

`moe_fused.cu`（Q2_0 canonical blob 布局，只有非 native pack 走）里的 2560/1280/640 **不改**：
它们不在 native pack 的路上。判据：调用点是否被 `lay.native` 门控。

## 5. 验收

- 编译过（远端 CUDA 构建，sm_60/61）。
- 加载测试出 token。
- `--cpu-oracle-q8-0` 与 CPU 路径对一遍。
