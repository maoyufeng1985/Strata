# Whittle-45B 的 PLE 记忆表移植（2026-10-28）

## 0. 一句话

引擎已能加载 Whittle-45B 的**专家、权重、token embedding**（934 MiB 权重 + 818 MiB 原生矩阵 + 273 MiB embedding，
实测通过），停在最后一项：**PLE 的 n-gram 记忆表**。它的几何与编译期不同，且表格格式是新的 Q4_K。

## 1. 实测：Whittle 的 PLE 几何（从下载好的 GGUF 读，`.run/ple_probe.py`）

| 项 | 编译期（Flash-Next） | **Whittle 实测** | |
|---|---:|---:|---|
| `embedding_length_per_layer_input`（行宽） | 160 | **256** | 改 |
| `ple.heads_per_ngram` | 8 | **4** | 改 |
| `ple.ngram_size` | 3 | 3 | 同 |
| `ple.conv_kernel` | 4 | 4 | 同 |
| 表张量 | `[160, 320001536]` IQ4_NL | **`[256, 39040000]` Q4_K(type 12)** | 改 |
| `head_offsets` | 16 个，~2e7 each | **8 个，4,880,000 each** | 改 |
| `head_vocab_sizes` | 16 个不等 | **8 个，全 4,880,000** | 改 |
| `layer_multipliers` | 23703573157769 / 20109073645365 / 8052911324071 | **32473892291597 / 22010589867355 / 6403636407469** | 改 |
| `ple.eos_token_id` | 248044 | 248044 | 同 |

**不变量**（两边都成立，这是结构没变的证据）：
`PLE_N_HEADS × PLE_HEAD_DIM = n_embd`
- 编译期：16 × 160 = 2560 ✓
- Whittle：8 × 256 = 2048 ✓

`n_embd / (NGRAM_SIZE-1)` = 每头的行数：Whittle 2048/2 = 1024？但表是 39,040,000 行。
⇒ 表的行数与 n_embd 无关，是词表大小决定的：`8 × 4,880,000 = 39,040,000` ✓
（编译期 `16 × ~2e7 = 320,001,536` ✓）

⇒ **每头一个词表**，头数 = `heads_per_ngram × (ngram_size-1)` = 4×2 = 8（编译期 8×2 = 16 ✓）。
词表大小 4,880,000（编译期 ~20,000,000）。

## 2. 与 Q36 的关系：Whittle 的 PLE 头数是 Q36 的两倍

`heads_per_ngram` 8 → 4 但 `ngram_size` 仍是 3 ⇒ 头数 16 → 8。
**头数减半，宽度 160 → 256 ⇒ 每头覆盖的 n_embd 宽度不变**（见上面的不变量）。

## 3. 改动清单（四处，都要编译器兜底）

| 文件 | 改动 | 风险 |
|---|---|---|
| `include/strata/kernels/ngram.hpp` | `PLE_HEAD_DIM` 160→256；`HEADS_PER_NGRAM` 8→4；`PLE_TABLE_ROWS` 320001536→39040000；`PLE_ROW_BYTES_*` 随之（256/32=8 块） | 低：都是 constexpr，编译期暴露 |
| `src/kernels/ngram.cpp` | `ple_artifact_consts()` 的 multipliers/vocab/offset 换成 Whittle 的；`PleFormat` 加 `Q4_K`，行宽 = Q4_K 的一行 | 中：新格式要写反量化 |
| `src/kernels/ngram.cpp` | `PleTable::open` 的 `want_ne0`：Q4_K 的表 ne0 是**行宽 256**（Q4_K 的 ne0 是元素数，不是字节数） | 中 |
| `src/kernels/cuda/ple.cu` / `native_ple_postops.cu` | 若有按 160 硬编码的行宽 | 查 |

## 4. Q4_K 的行长

Q4_K 的块是 256 元素（`QK_K=256`），一行 256 元素 = **1 块** = `sizeof(block_q4_K)` = 144 B + 2 个 fp16 scale
（`d`, `dmin`）…… 实际 ggml 的 `block_q4_K` 是 `{half d; half dmin; uint8_t scales[12]; uint8_t qs[128];}` = 144 B。
⇒ **Q4_K 行 = 144 B**，与 256 元素一一对应（比 IQ4_NL 的 90 B 宽）。

表大小：39,040,000 行 × 144 B = **5.62 GB** —— 与 HF 卡片说的「n-gram 表在 Q4_K_M 里约 5.6 GiB」**完全吻合**。
⇒ 这是交叉验证：行宽、行数、格式三者互相印证。

## 5. 不做的事

- 不把 Q4_K 的 PLE 表转成 IQ4_NL（会改数值，且要多 10 GB 磁盘 + 一次全表量化）。
- 不动 `PLE_N_HEADS` 的使用点（`verify.cpp` / `mtp.cpp` 的 `rows[PLE_N_HEADS]` 数组随 constexpr 自动跟随）。
- 不给 Q4_K 加 scale 元数据（Q4_K 自带 d/dmin）。
