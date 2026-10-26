# DCA / MInference 1.0 — verdict for q36

Plain facts from five research reports, each adversarially verified. For an engineer deciding
what to build. No design is proposed; only what exists, what it costs, and what it requires.

## 1. WHAT THEY ARE

**DCA** is not a single named paper the user specified. The closest verified matches are two
distinct things: (a) **Dual Chunk Attention** — arXiv 2402.17463, ICML'24 ("ChunkLlama"),
a training-free length-extrapolation that changes only the RoPE position indexing (local +
chunked global), requires no new model components, and integrates with FlashAttention at
near-zero kernel overhead (VERIFIED, PDF read); and (b) **the DeepSeek Sparse Attention (DSA)
"lightning indexer"** — arXiv 2512.02556 (DeepSeek-V3.2), a trained 64-head/128-dim indexer
that picks the top-k=2048 KV tokens per query, reducing main-attention from O(L²) to O(Lk)
(VERIFIED, paper §2.1/§2.3). "MInference" (often written MInference 1.0) is the **Microsoft
training-free block-sparse method**, arXiv 2407.02490 (NeurIPS'24): offline per-head sparse
pattern assignment (A-shape / Vertical-Slash / Block-Sparse) + online index approximation,
a one-time ~15 min offline search on an A100, no model modification (VERIFIED, abstract +
Appendix C.2). It is FOR prefill latency: 1.8x/4.1x/6.8x/10x at 100K/300K/500K/1M on a single
A100 (VERIFIED, paper Latency section). Decode is left dense for all baselines in that paper.

## 2. THE DECISIVE QUESTION FOR q36

q36 geometry: 40 layers, GQA 16 q-heads / 2 KV-heads / head_dim 256, 256 experts top-8,
full-attention only on layer%4==3 (10 trunk layers) + 1 MTP layer = **11 attention layers**;
the other 30 layers are GDN (linear/recurrent, constant ~4.8 ms/tok regardless of context).

| | DCA-style (Dual Chunk Attention, 2402.17463) | DCA-style (DSA lightning indexer, 2512.02556) | MInference 1.0 (2407.02490) |
|---|---|---|---|
| Needs a component the model lacks? | No (RoPE indexing only) | **Yes: trained indexer weights (64×128, ~2.1B tokens warm-up + 943.7B tokens sparse stage)** | No (offline search produces a static per-head pattern, no weights) |
| Acts at | prefill (position indexing) | both (indexer scores at decode, picks top-k) | **prefill only** in 1.0; decode stays dense |
| Reported accuracy cost | comparable quality (Llama2 70B to 100k+) | "no substantial degradation" vs V3.1-Terminus, parity prose only, **no k-sweep numbers** | RULER effective window held (LLaMA-3-262K→32K, GLM-4-1M→64K), VERIFIED |

q36 specifics:

- **DSA / trained-indexer path**: q36 has **no indexer** in its config.json (verified: the
  MiniCPM-V-4.7-35B-A3B card, which has identical 16/2/256 geometry, has no sparse_attn /
  attn_topk / indexer_* keys — q36 is the same design). Adding DSA requires **continued
  training** (the paper trains it: 1000 steps dense warm-up at lr 1e-3, then 15000 steps
  sparse at lr 7.3e-6, KL against head-summed L1-normalized main-attention distribution).
  The indexer input is detached; the indexer signal is only from the KL loss. This is a
  training workload, not an inference-time retrofit. The sparse GEMM kernel assumes MLA-MQA
  (one shared latent KV per token, used by every query head) — q36 is GQA-16/2 with
  per-head K/V, so the **selection logic transfers but the fast kernel does not**.
  P100 (sm_60, no FP8 tensor cores, no mma) cannot run any merged kernel: DeepGEMM #200
  is SM90 FP8, FlashMLA #98 is Hopper, llama.cpp's merged CUDA path (#25545) is F16/F32
  vector/wmma with no P100-validated numbers; open #28888 is explicitly a Volta cuBLAS
  fallback. **No public source named "DCA" in the long-context sparse-attention sense was
  found** (arXiv search returns video-commentary co-attention only).

- **MInference 1.0 path**: training-free, but **offline-static** — the pattern is
  per-layer/per-head, searched once on a 30K-token KV-retrieval sample (~15 min on an A100),
  then reused for all lengths. It targets **prefill**, not decode. The merged kernel
  (vllm-project/flash-attention PR #33, merged 2025-01-15, by minminsun) implements
  vertical + slash patterns only (A-shape and Block-Sparse not in that kernel). MInference
  2.0 (arXiv 2501.07724) adds a generation-stage claim (up to 5.7x) but is a different
  paper and its kernel still needs sm_80+. No Qwen-family MoE or GQA-16/2 model appears
  anywhere in the MInference 1.0 paper, repo configs, or supported list (grep of arXiv v2
  + 400-path repo tree confirms: models are LLaMA-3-8B-262k/1M, GLM-4-9B-1M, Yi-9B-200K,
  Phi-3-Mini-128K, Qwen2-7B-128K, Qwen2.5 dense 7/32/72B + 1M). **MInference's LeanK decode
  variant requires a learned per-layer channel mask (.pth) and reorder_linear_weights** —
  not training-free at decode (VERIFIED, LeanK/leank/patch/llama.py).

## 3. THE q36 COST MODEL (from the q36-specific report)

Numbers all traceable to `plans/2026-10-26-q36-kv-stream.md` (MEASURED 2026-10-26):

**Today, dense decode at 262144 cells:**

| | bytes | source |
|---|---:|---|
| KV int8, 11 layers, 262144 | **2.84 GiB** | plan L111 |
| Actual read traffic with 8× GQA redundancy | **22.5 GB/token** | plan L223 (2.84 GiB × 8) |
| @ P100 410 GB/s (measured HBM ~410) | **~55 ms/token** | plan L224 |
| Cost-model fit (0.6655 + 0.10496·nc, nc=1024) | **108.1 ms/token** (extrapolated, not measured) | plan L273 |

**Decode baseline (MEASURED, this box):** 54.8 tok/s = **18.25 ms/token** at ctx 2048
(`plans/2026-10-25-q36-decode-mtp-directions.md` L31). The "16 ms/token" figure in
`q36_engine.cu:2577` and `plans/2026-10-26` L35/L275 is UNVERIFIED against the measured
2026-10-25/26 numbers (18.25 ms is the measured value; 16 ms may be a different config).

**If a top-k selection read instead (k = 2048, the trained value from DSA):**

The q36 engine's own `qsa_select.cu` uses `idx_top_k = 2048`, `idx_block = 4`
(`qsa.hpp:109-117`), giving a selection width of `min(n_kv, 2048+4-1) = 2051` cells
(`qsa.hpp:133-136`). At 2051 selected cells per layer × 2 KV heads × head_dim 256 ×
int8 (1 B) = **1.04 MB/layer selected** vs 264 MB/layer dense (262144 × 2 × 256 = 134 MB
per KV head, × 2 = 268 MB, ~264 MiB int8). 11 layers: **~11.5 MB/token sparse** vs
**~2860 MB/token dense** — a **249× reduction** in bytes read.

**Selection pass cost (q36's own kernel, sm_60 P100):**

The P100 (sm_60) **cannot** run `qsa_block_scores_tc` (needs sm_80) nor
`qsa_block_topk_cluster` (needs sm_90). The adversarial check REFUTED the claim that
the P100 uses the "Turing STRATA_TOPK_SPLIT path": `topk_split_device()` gates on
`sm75_default` (Turing sm_75) or `STRATA_TOPK_SPLIT=1` env — P100 is sm_60, not sm_75,
so it falls through to the register/wide top-k kernel (`qsa_select.cu:2246-2258`).
The warp-level block scorer does `128-dim × 4-head = 512 MACs per pooled block`
(VERIFIED, `qsa_select.cu` IDX_DIM=128, IDX_HEADS=4). At 262144 cells, block-pooled
scoring is ~35 G MACs/step (block granularity 4 → 65536 blocks × 512 MACs × 11 layers),
on P100 FP32 4.2 TFLOPS ≈ **~9 ms/step** for the selection pass alone — this is a
derived estimate, not a measured number. The in-tree `qsa_decode_attn.cu` measured on
RTX 2080 Ti (sm_75) at 2051 selected cells: **34.5 µs/query** (line 339 comment).

**Net at 262144 with top-k=2048:**

| | dense (today) | sparse top-k=2048 |
|---|---:|---:|
| Attention bytes/token | 22.5 GB (×8 redundancy) / 2.84 GiB (after GQA fix) | ~11.5 MB (11 layers, 2051 cells) |
| Attention ms/token | 108.1 (extrapolated, dense, ×8) / ~14 (dense, GQA-fixed) | not measured, est. ~9 (selection) + attention over 2051 cells |
| Decode ms/token | 18.25 (measured, ctx 2048) | dominated by selection + expert GEMV |

The k=2048 assumption comes from DSA's trained value (`index_topk=2048` in
config_671B_v3.2.json, VERIFIED) and q36's own `qsa_select.cu` `idx_top_k=2048`.
There is **no published k-sweep quality number** for any of these (DSA paper gives
only parity prose; MInference has no k-sweep either).

## 4. THE VERDICT

Ranked by applicability to q36 on a P100 + 3×P104 box, no P2P:

### 1. "Just fix the 8× redundant read" — ALREADY DONE, MEASURED

The q36 engine's `k_attn_part` grid has y-dim = query-head count (16), so each KV head's
cells are read 8×. Flash-Next's `attn_chunk_kernel` uses y-dim = n_head_kv, so each KV
cell is read once per block, with all G=8 query heads sharing the K/V read inside the
block. The fix was implemented in `k_attn_part_gqa` (flag `STRATA_Q36_ATTN_GQA=8`):
- **Bit-identical** output (P0 flag-off vs known-good: gen byte-identical; P1 flag-on
  vs P0: gen byte-identical) — VERIFIED, plan L377-387.
- **+19%** at ctx 32768 (17.7→21.2 tok/s), **+50%** at peer config (34.2→51.3 tok/s),
  **−0.5%** at ctx 2048 (noise) — MEASURED, plan L395-447.
- The slope of the cost model drops from 0.105 to ~0.013 ms/tok per 256 cells
  (plan L283). At 262144: 108.1 → ~14 ms/token (extrapolated).
- **What it took**: ~50 lines in q36's own kernel, no new dependency, no training,
  no P2P, no new GPU. Already in the tree.

This is the path the evidence supports as the **first** fix. It does not make attention
sparse — it makes the dense read 8× cheaper. At 262144, attention goes from dominating
decode (108 ms) to being comparable to it (14 ms).

### 2. MInference-style training-free block-sparse — DOES NOT APPLY to q36 decode

- MInference 1.0 is **prefill-only** in its 1.0 form; decode stays dense for all
  baselines (VERIFIED, paper Experiments setup).
- The offline pattern search was validated on dense LLaMA-3/Qwen2/GLM-4/Yi/Phi-3 —
  **no Qwen3.x, no MoE, no GQA-16/2 model** (VERIFIED, 400-path repo grep).
- The merged kernel (FA PR #33) is vertical+slash only, targets prefill, needs sm_80+
  (A100/H100). P100 sm_60 cannot run it.
- MInference 2.0's generation-stage claim (5.7x) is on A100/Hopper, not sm_60.
- LeanK's decode variant requires a **learned channel mask** (.pth) — not training-free.
- **What it would take for q36**: run a 15-min offline search on a 30K-token KV-retrieval
  sample on an A100 (which this box does not have), produce a per-head pattern, then
  write a sm_60 sparse decode kernel (none exists in-tree for GQA-16/2). No quality
  evidence for GQA-16/2 MoE.

### 3. DSA-style trained indexer — DOES NOT APPLY without training

- q36 has no indexer. Adding one requires the V3.2 training recipe (2.1B + 943.7B tokens,
  KL distillation) — a training run, not an inference-time retrofit.
- The MLA-MQA assumption (shared latent KV) does not hold for GQA-16/2.
- No merged kernel runs on sm_60. The P100 has no FP8, no mma, no Hopper instructions.
- No published k-sweep quality data to tune against.
- **What it would take**: a full fine-tune of a 64×128 indexer on q36's data, then a
  sm_60 sparse gather kernel (the llama.cpp merged path #21458 GGML_OP_GATHER is a
  fallback, not a fast kernel), plus host-memory KV streaming (the `kv_stream.cu` layer
  exists in Flash-Next but is designed for the sparse QSA path, not dense).

## 5. WHAT WAS NOT VERIFIED

- **The "16 ms/token" decode baseline** in `q36_engine.cu:2577` and plan L35: the
  measured value on this box (2026-10-25) is 18.25 ms/token (54.8 tok/s). 16 ms may
  be a different config (with/without MTP, different flags). Read the exact benchmark
  config that produced 16 ms to resolve.
- **The selection-pass cost on P100 sm_60** (~9 ms/step derived): the in-tree
  `qsa_select.cu` warp-scoring kernel was not measured on this box at 262144 cells.
  The 34.5 µs/query number is from RTX 2080 Ti sm_75. Run `qsa_select` on the P100
  at full context to get a real number.
- **DSA k-sweep quality**: the paper gives no table. If k=2048 is the trained value,
  what does quality look like at k=1024 or k=4096 on q36? No source.
- **MInference 2.0 generation-stage numbers** (5.7x on MInferBench): arXiv 2501.07724
  was not read in full; the 5.7x figure is from the abstract's second report only.
- **The exact "11 vs 13 QSA layers" discrepancy** in `plans/2026-10-26-q36-kv-stream.md`:
  L5 says 13, L30/L111 use 11. 40 layers % 4 == 3 gives layers 3,7,...,39 = 10 trunk
  layers; + 1 MTP = 11. The "13" appears to be an error inside the plan (double-counted).
  All PCIe cost numbers depend on which count you use.
- **Whether the GQA fix's 108→14 ms extrapolation holds at 262144**: the cost-model
  fit is over 4 measured points (max 65536). 262144 is 4× the largest measured point.
  Run the actual binary at 262144 (if VRAM allows after the GQA fix) to confirm.
- **P100 sm_60 vs P104 sm_61 `__dp4a`**: the P104 (sm_61) has true `__dp4a` (2.31×
  faster than the P100's 4×vmad fallback, MEASURED). The q36 attention fix is
  architecture-independent, but expert GEMM speed on the primary card is 2.31× slower
  than the helpers. This does not affect the attention verdict but affects total decode.
- **"DCA" as a specific paper**: no arXiv id was found for a paper literally named "DCA"
  in the long-context sparse-attention sense. The two things the user may mean are
  Dual Chunk Attention (2402.17463) and DSA (2512.02556). If the user intended a
  different paper, supply the arXiv id.

## 6. SOURCES

| URL | What it supports |
|---|---|
| https://arxiv.org/abs/2402.17463 | Dual Chunk Attention: training-free RoPE indexing, FlashAttention integration, zero kernel overhead |
| https://arxiv.org/abs/2512.02556 | DSA lightning indexer: O(L²)→O(Lk), trained recipe (2.1B+943.7B tokens), MLA-MQA assumption, parity prose |
| https://arxiv.org/abs/2407.02490 | MInference 1.0: training-free offline pattern + online index, 1.8x-10x prefill at 100K-1M on A100, decode stays dense |
| https://arxiv.org/abs/2501.07724 | MInference 2.0: generation-stage claim (5.7x), not read in full |
| https://github.com/vllm-project/flash-attention/pull/33 | MInference kernel merged 2025-01-15: vertical+slash only, by minminsun |
| https://github.com/vllm-project/vllm/pull/11844 | vLLM dual-chunk backend merged 2025-05-13 |
| https://github.com/vllm-project/vllm/pull/54873 | vLLM QSA sparse GQA for Qwen3.8-Flash-Next: topk buffer [N, 2048+3], decode 10.0-10.6 µs at ctx 1024-51200 |
| https://github.com/flashinfer-ai/flashinfer/pull/6804 | FlashInfer NVFP4 QSA sparse paged attention: SM80-90, split-K, 14.5 µs/query at 2051 tokens from 8K on sm_80 |
| https://github.com/MiniMax-AI/MSA | MiniMax Sparse Attention: trained Index Branch, SM100/SM103 kernels, 7.6x decode at 1M on H800 |
| https://github.com/microsoft/MInference | MInference repo: LeanK channel mask (.pth), tilelang flash-decoding, configs for Qwen2.5/GLM-4/LLaMA-3 |
| https://github.com/ggml-org/llama.cpp/pull/21458 | llama.cpp GGML_OP_GATHER merged: DSA sparse decode via gather (not MMA) |
| https://github.com/ggml-org/llama.cpp/pull/25545 | llama.cpp CUDA lightning indexer F16/F32 merged (no P100 numbers) |
| https://github.com/sgl-project/sglang/pull/5327 | SGLang MInference kernel merged 2025-04-12 |
| https://github.com/sgl-project/sglang/pull/35657 | SGLang MSA: OPEN (not merged), TRT-LLM-GEN block-sparse decode, SM103 |
| https://github.com/deepseek-ai/DeepGEMM/pull/200 | DSA indexer logits kernel: SM90 FP8, merged |
| https://github.com/deepseek-ai/FlashMLA/pull/98 | DSA sparse attention kernel: Hopper, merged 2025-09-29 |
| https://hf-mirror.com/Qwen/Qwen2.5-14B-Instruct-1M/raw/main/config.json | DCA+MInference 1M config: dual_chunk_attention_config, rope_theta 10M |
| https://qwenlm.github.io/blog/qwen2.5-1m/ | Qwen 2.5 1M blog: DCA+MInference integration, 3.2x-6.7x prefill, custom vLLM branch |
| D:\work\Strata\plans\2026-10-26-q36-kv-stream.md | q36 cost model: 2.84 GiB int8 KV at 262144, 108.1 ms/token extrapolated, GQA fix +19%/+50% measured |
| D:\work\Strata\plans\2026-10-25-q36-decode-mtp-directions.md | q36 measured decode 18.25 ms/token (54.8 tok/s), MTP D=3 peak 74.7 tok/s |
| D:\work\Strata\src\kernels\cuda\qsa_select.cu | q36's own selection kernel: idx_top_k=2048, idx_block=4, warp scoring 512 MACs/block |
| D:\work\Strata\src\kernels\cuda\qsa_decode_attn.cu | Flash-Next QSA decode: G=12 hard-coded (won't run on q36 G=8), 34.5 µs/query sm_75 |
| D:\work\Strata\src\kernels\cuda\kv_stream.cu | Flash-Next KV streaming: host KV + selected-block pull, designed for sparse QSA |
