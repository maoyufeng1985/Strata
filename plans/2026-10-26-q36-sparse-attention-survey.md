# DCA / MInference / Quest 对 q36 的适用性（2026-10-26）

## 已查证（VERIFIED，2026-10-26，一手来源）

### 1. MInference 1.0 —— **训练无关，但作用在 prefill**（用户复核**成立**）

**arXiv 2407.02490 摘要原文**（`https://arxiv.org/abs/2407.02490`）：

> "Our proposed technique can be directly applied to existing LLMs **without any modifications to the
> pre-training setup or additional fine-tuning**."
> "We determine the optimal pattern for each attention head **offline** and dynamically build sparse indices
> based on the assigned pattern during inference."
> "...significantly reduce the latency in the **pre-filling stage** of long-context LLMs."

⇒ **训练无关 ✓**（不改预训练、不微调）。
⇒ **作用在 prefill ✓**。

**全文（`https://arxiv.org/html/2407.02490v2`）§Experiments 的决定性一句**：
> "For all baselines, we perform sparse computation **only during the pre-filling stage**, while
> **retaining dense computation during the decoding stage**."

⇒ **decode 保持稠密，这是论文自己的实验设定**，不是我们的推断。

**论文把 decode 的优化归到另一类**（§Related Work）：
> "optimizations for the decoding stage are divided into: 1) Reusing attention KV to reduce KV cache storage;
> 2) Static KV cache dropping; 3) Dynamic KV cache dropping; 4) **Dynamic KV cache offloading**; 5) Methods for
> retrieval-augmented decoding"

⇒ **第 4 类（动态 KV offloading）就是 Quest 那一族**——MInference 1.0 自己不做那个。

**加速比（§Latency，原文）**：
> "At 100K, 300K, 500K, and 1M tokens, our method achieves speedups of **1.8×, 4.1×, 6.8×, and 10×**,
> respectively. It reduces the **pre-filling latency** from 30 mins to 3 mins on a single A100 for a prompt of
> 1M token."

**选择 pass 的成本（Appendix F，原文）**：
> "when the context windows exceed 200k, the actual sparsity of all three patterns surpasses 90%.
> **Even considering a 20% index-building overhead**, this ensures that the kernel achieves a speedup of over 8×."
> "when the context windows exceed 500k, the sparsity relative to FlashAttention exceeds 95%,
> with a theoretical speedup of over 15×"

⇒ **"20% index-building overhead"是本文给的唯一一条选择 pass 的成本数**——
对 q36 的稠密 decode 路径，选 top-k 也要付这个代价，得算进账里。

**但"训练无关"≠"零准备"**：pattern 是**离线**定的。
官方 repo `experiments/README.md` 有专章 **"Offline Kernel-Aware Sparse Pattern Search"**，
跑的是 InfiniteBench 的 `kv_retrieval` 任务，产出 `*_best_pattern.json`。
⇒ **每个模型都要先离线跑一遍 pattern 搜索**，只是它不更新权重。

**官方 e2e benchmark 全部是 prefill 的数**（`experiments/README.md`）：
> "we measured the **pre-filling stage** latency **without using KV cache**"

**支持模型里没有一个 MoE**：Qwen2 / Qwen2.5 / LLaMA-3 / GLM-4 / Yi / Phi-3
（`https://raw.githubusercontent.com/microsoft/MInference/main/README.md`）。

### 2. 官方给 MInference 定的位置：**① KV Cache Generation**

README 把它按 KV cache 生命周期分类：
- ① **KV Cache Generation**：MInference、xAttention、FlexPrefill、A-shape、Tri-shape…
- ② KV Cache Compression：StreamingLLM、SnapKV、PyramidKV、KIVI
- ③ KV Cache Retrieval：CacheBlend
- ④ **KV Cache Loading**：**Quest**、RetrievalAttention

⇒ **MInference 1.0 = prefill 时怎么生成 KV**；
⇒ **decode 侧的那个是 ④ 里的 Quest**。

### 3. Quest —— **训练无关、query-aware、decode 侧**（arXiv 2406.10774）

> "Quest, a **query-aware** KV cache selection algorithm. Quest keeps track of the **minimal and maximal Key
> values** in KV cache **pages** and estimates the criticality of a given page using Query vectors.
> By only loading the **Top-K critical KV cache pages** for attention, Quest significantly speeds up
> self-attention without sacrificing accuracy."
> "up to **2.23x self-attention speedup**, which reduces **inference latency by 7.03x**"

**对 q36 的意义**：Quest 需要的三个东西，q36 都能凑齐：
1. **K 的 per-page min/max** —— q36 的 K 是 int8 量化（`kKVG = 64` 个码一个 fp16 scale），
   min/max 可以**量化后**维护（每 64 码一组已经有 scale，再存 min/max 就是 2 个 int8）
2. **per-query 的 page 重要性打分** —— q36 的 decode 是 1 个 query row，**每层只有 1 个 query**，
   打分极便宜（不是 prefill 那种 T 个 query）
3. **只读 Top-K page** —— 这就是 KV 流式要的东西

⇒ **Quest 才是 q36 _decode 侧_ 的那条路**（MInference 是 prefill 侧，q36 的 prefill 不是瓶颈）。

**Quest 报告的数**：self-attention 2.23×，**inference latency 7.03×**（页面加载的收益远大于注意力本身）。

### 4. DeepSeek DCA —— 待 workflow 核实

（DeepSeek-V3.2-Exp 的 lightning indexer + block-sparse main attention。
**关键问题**：indexer 是**训练过的**吗？需要 MLA 吗？对一个没有 indexer 的 GQA 模型能用吗？）

## 还没核实（等 workflow）

- DCA 的 indexer 是否必须训练、是否必须 MLA
- "DCA + MInference 1M 长上下文配置" 到底是哪个具体产物（DCAer? llama.cpp PR? HF 模型?）
- RTPurbo 是什么、要求什么架构
- llama.cpp / vLLM / SGLang 各自的稀疏 attention 支持现状
- **block-sparse decode 的选择 pass 本身要多少**（对比省下的 KV 读）
- head_dim 256 的模型在 decode 侧做 block-sparse 的精度代价（有没有公开数）

### 4. "DCA + MInference 1M 长上下文配置" —— **找到了，是 Dual Chunk Attention**（VERIFIED 2026-10-26）

`Qwen/Qwen2.5-14B-Instruct-1M` 的 `config.json`（`hf-mirror.com` 可读）：

```
dual_chunk_attention_config: {chunk_size: 262144, local_size: 8192,
                               original_max_position_embeddings: 262144}
rope_theta: 10000000.0
```

⇒ **DCA = Dual Chunk Attention**（arXiv 2402.17463, ICML'24, "ChunkLlama"），
**不是** DeepSeek 的 sparse attention。机制：**只改 RoPE 的位置索引**（局部窗口 + 分块全局），
**不改模型结构、不需要训练、不改权重**——是"长度外推"的方法，不是"省流量"的方法。
（vLLM 侧的实现是 PR #11844，已 merged 2025-05-13。）

⇒ **所以用户说的那个"1M 长上下文配置"是一个 `config.json` 的字段 + 一个 vLLM 后端**，
不是一个新的模型。**它不解决 q36 的 decode 流量问题**（它解决的是"能不能在 1M 上跑"）。

### 5. Qwen3.8-Flash-Next 在 vLLM 上已经是稀疏 QSA（VERIFIED 2026-10-26）

PR **#54873**（gau-nernst，**2026-09-04 已 merged**，15 commits，vllm-bot 合入 main）：
**"[Qwen3.8-Flash-Next] Improve QSA sparse GQA for prefill and short-ctx decode"**

PR 正文原话：
> "QSA sparse attention read topk indices from **[num_tokens, 2048+3]** buffer, where each token can attend
> up to **2048+3** past KV tokens."

**它的 decode attention 微基准**（tp1，N=1，DQL=1，单位 µs）：

| ctx | before → after |
|---:|---|
| 1024 | 11.9 → **10.0** |
| 2048 | 12.7 → **10.4** |
| 8192 | 12.9 → **10.6** |
| 51200 | 12.8 → **10.6** |

⇒ **ctx 从 1024 到 51200，decode 的 attention 一直是 ~10.6 µs**（**不随上下文涨**）。
**对比 q36 今天的稠密路径**：32768 cell 就要 **14.5 ms/token**（11 层合计，实测）。
⇒ **差 1400 倍。**

**但这个 PR 本身只改"有效 topk 数量"（valid_counts），不引入稀疏**——
它假设**稀疏选择已经有了**（那个 2048+3 的索引 buffer 是 Qwen3-Next 自带的机制）。

**E2E（TP4 GB300，BF16，无 MTP）**：并发 64 时 tok/s +1.9%、TTFT p50 -5.4%、TPOT p50 -2.0%。
⇒ **稀疏 QSA 已经在生产里跑这个模型了。**

**关键未决问题**：那个 `2048+3` 的 top-k 索引是**模型自带的训练过的 indexer** 算的，
还是**推理时算的**？⇒ **已核实：是模型自带的训练过的权重。**

### 6. Flash-Next 的 QSA 依赖模型自带的 indexer 权重（VERIFIED 2026-10-26，仓库内）

- `src/core/layout.cpp:78-79`：加载层时检查 **`indexer.q_proj.weight`** 和 **`indexer.k_proj.weight`**，
  标记为 `qsa_only = true` ⇒ **这些权重只存在于 Flash-Next（Qwen3.8-Flash-Next）的模型文件里**。
- QSA 的 attention kernel 用的是这**两个投影的输出**来算稀疏分数，**不是**用现有 q/k 投影临时算的。
- q36（Qwen3.6-35B-A3B）的权重清单里**没有** `indexer.*`；
  `src/qwen35/q36_main.cpp` 里没有任何 indexer 引用，q36 的几何里也没有 indexer 字段。

⇒ **q36 不能照搬 Flash-Next 的 QSA 稀疏路径**——它缺的是**训练过的 indexer 权重**，
不是缺代码（代码在树里，但没有权重就是没有打分依据）。

⇒ 这也解释了为什么 Flash-Next 的稀疏是"免费的"：**它的稀疏能力是训练时买来的**。

## 对 q36 的四条路（当前判断）

| 路 | 训练无关？ | 作用在 | q36 的瓶颈是 decode ⇒ 适用？ | 代价 | 实测 |
|---|---|---|---|---|---|
| **DCA = Dual Chunk Attention** | **是**（只改 RoPE 位置索引）| 长度外推 | **不解决流量**（解决"能不能跑到 1M"）| 改 RoPE + vLLM 后端 | 未测 |
| DSA / lightning indexer（DeepSeek）| **否**（要训练 indexer：2.1B + 943.7B token）| decode | **否**（q36 没有 indexer，而且它是 MLA-MQA，q36 是 GQA-16/2）| 改架构 + 微调 + 蒸馏 | 未测 |
| MInference 1.0 | 是（但要离线 pattern 搜索）| **prefill** | **否**（q36 的 prefill 不是瓶颈）| 离线搜索 + sm_80+ kernel | 未测 |
| **Quest** | **是** | **decode** | **是** | per-page min/max + 选择 pass | 未测 |
| **GQA 共享（已做）** | **是**（不改权重，逐位不变）| decode | **是** | ~50 行，已落地 | **+50% @32K，交叉点 4K** |

⇒ **GQA 共享是当前唯一"改完就赚"的路**：decode 读 32768 cell 时 **+50%**，
交叉点在 decode 读 2000–4000 cell（**ctx < 4K 会亏 2.6–3.6%**），数值**逐位不变**。
⇒ **Quest 是下一条路**（如果 262144 的稠密读仍是瓶颈：GQA 之后 attention 仍是 ~36 ms/token）。
⇒ **DSA 那条路对 q36 关死**（没 indexer 权重、架构不匹配、P100 无 FP8/Hopper kernel）。

### 为什么 Flash-Next 能稀疏而 q36 不能

**Flash-Next（Qwen3.8-Flash-Next）自带的 indexer 权重是训练时买来的能力。**
q36（Qwen3.6-35B-A3B）的权重里没有 `indexer.*` ⇒ 没有打分依据 ⇒ 只能用稠密。
**这不是代码差距，是模型差距。** 想补上只能训练（DSA 的配方：2.1B token 热身 + 943.7B token 稀疏阶段）。
