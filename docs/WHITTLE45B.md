# Whittle-Qwen-3.8-45B-A3B: running it on Strata's Flash-Next engine

This page covers [logic65/Whittle-Qwen-3.8-45B-GGUF](https://huggingface.co/logic65/Whittle-Qwen-3.8-45B-GGUF)
(45B mixture of experts, 3B active, 256 experts, top-8) on the `qwen4exp` engine path — the one
Qwen3.8-Flash-Next uses. It runs on NVIDIA cards. `Q4_K_M` on a Tesla P100 (12 GB, sm_60) is the
configuration every number below was measured on.

> **This is not an officially supported model.** The engine's kernels are compiled for one shape
> (`include/strata/core/geometry.hpp`), and this model's shape differs from Flash-Next's. The port
> that makes the two agree is in the working tree; a model file of another shape is still refused at
> load with the metadata key that differs.

## What you need

| | |
|---|---|
| Model file | `Whittle-Qwen-3.8-45B-A3B-Q4_K_M.gguf` (about 26 GB) |
| Expert pack | a native pack made from that file (`tools/iq_pack.py`) — 18.16 GiB, read through the file cache |
| PLE | the same GGUF carries the PLE table (39,040,000 rows, 5.62 GiB); `--ple-gguf` reads it from there |
| RAM | 31 GiB is enough for one engine and nothing else |
| VRAM | 12 GB on the main card; three 8 GB cards help decode |
| CPU | any x86-64 with AVX. The older-CPU build (`-DSTRATA_ISA_FLOOR=avx`) is for pre-AVX2 chips |

## The command

```sh
M=/path/Whittle-Qwen-3.8-45B-A3B-Q4_K_M.gguf
PACK=/path/packs/whittle45b
PROF=/path/profile.bin          # built from a routing trace, see below

strata --pack "$PACK" --native "$M" --tokens-file PROMPT.txt \
    --ple-gguf "$M" --ple-io ram \
    --max-new 256 --max-context 4096 \
    --spec 2 --suffix-draft 0 \
    --expert-profile "$PROF" --expert-cache auto \
    --gpu 0 --kv int8 --prefill 2048
```

Add three more cards to hold experts, and `--remote-expert-opt` (see below):

```sh
    --gpu 0,1,2,3 \
    --expert-cache-device1 auto --expert-cache-device2 auto --expert-cache-device3 auto \
    --remote-expert-opt
```

`--gpu N` is an **nvidia-smi index**, not a CUDA ordinal: on this box the P100 is index 2 and the
three P104s are 1, 0 and 3, so `--gpu 2` is the P100 and `--gpu 2,1,0,3` is the P100 with the three
helpers behind it.

### `--remote-expert-opt` needs the serve mode

The flag is refused without `--serve`, because it changes how a helper returns its work and the
author only validated it there. The serve mode is the same engine reading `GEN <max_new> <ids>`
lines on stdin and writing `T <id>` lines on stdout; the app and the Terminal both drive the engine
this way, so this is the production path:

```sh
printf 'GEN 256 %s\n' "$IDS" | strata --pack "$PACK" --native "$M" \
    --ple-gguf "$M" --ple-io ram \
    --max-context 4096 --spec 2 --suffix-draft 0 \
    --expert-profile "$PROF" --expert-cache auto \
    --gpu 2,1,0,3 --kv int8 --prefill 512 \
    --expert-cache-device1 auto --expert-cache-device2 auto --expert-cache-device3 auto \
    --remote-expert-opt --serve
```

## The PLE table belongs in RAM

`--ple-io ram` maps the table and locks it in RAM at start (`PLE table locked in RAM (--ple-io ram)
in 0.5 s`). It costs 5.62 GiB of the machine's RAM — the expert arena is another 18.16 GiB, so on a
31 GiB box there is room for exactly one engine.

The table is the model's n-gram memory ("engram"): 16 rows per token, gathered from a 39-million-row
Q4_K table. In RAM it costs about **0.04 ms a token** (the prompt's 276 tokens took 11.4 ms of PLE
in total), so it is not on the token's critical path. `direct` (the default) reads the same rows
from the SSD on every token instead.

## Measured (Tesla P100, 2026-10-28)

276-token prompt, 128 tokens generated, `--kv int8`:

| Configuration | decode |
|---|---|
| P100 + three P104s (expert cache) | 42.1 tok/s |
| the same, `--remote-expert-opt` | **47.2 tok/s** |
| the same, `--remote-expert-opt` and `--expert-cache-remote-placement layer` | **47.8 tok/s** |

`--remote-expert-opt` is the whole difference: each helper reduces its expert outputs to one
weighted partial sum per token **on its own card** before the result comes back, so the bytes that
cross PCIe drop about eight times (16.4 MiB instead of 131.5 MiB of full rows per helper over 128
tokens) and the time the main card waits for the helpers falls from 223 ms to 129 ms.

With 256 tokens generated the same configuration gives **43.9 tok/s** without the flag; the number
with it is in [the summary](../plans/2026-10-28-whittle-summary.md).

Without the helpers (the P100 alone, `--gpu 2`): 32.5 tok/s with a 700-token prompt and 64 tokens
generated, 38.0 tok/s with the three helpers. A long generation is faster than a short one: the
graph capture, the profile fill and the PCIe warm-up are fixed costs.

Prefill scales with the prompt (275 tokens 215 tok/s, 699 tokens 340, 1799 tokens 467) and does not
benefit from the other three cards at all: 340.5 vs 340.1 tok/s on 699 tokens, 470.9 vs 472.3 on 1799.
The PCIe link is 6.1 GB/s on this box, which is not enough to pay for shipping the prompt's expert
rows. The helper cards can still help the prompt, but only by computing rows instead of receiving
them: see `STRATA_PREFILL_REMOTE_PEERS` below.

## The prompt can use the helper cards (`STRATA_PREFILL_REMOTE_PEERS=1`)

Off by default. With it on, the three helper cards (`--expert-cache-device1..3`) each compute a
share of the prompt's expert rows on their own card and send the rows back over the host route,
instead of the P100 computing every row itself. The decode does not change.

| Arm (276-token prompt, P100 + three P104s, Q4_K_M) | prompt |
|---|---|
| the flag off | 242 tok/s |
| the flag on, one helper | 258 tok/s |
| the flag on, two helpers | 305 tok/s |
| the flag on, three helpers | **330 tok/s** (+36 %) |

Measured on the box on 2026-10-28 with the MMQ prompt path (`STRATA_MMQ_KQ=1`). The decode stayed
at 44 tok/s in every arm.

Two things to know:

- **The flag needs `STRATA_MMQ_KQ=1`** on a K-quant pack. Without it the MMQ plan is empty and the
  engine answers "needs the MMQ prompt path (set STRATA_MMQ_KQ=1 for a K-quant pack)" and the
  prompt quietly stays on the P100.
- **`--peer-prefill-rows` does not apply here.** The helper caches take the rows their own caches
  hold; the engine says so when the flag is set.

The numbers, the output check and the review are in
[the plan](../plans/2026-10-28-whittle-remote-prefill-peers.md).

## Two things about the numbers

**The profile decides most of the speed.** `--expert-cache auto` fills its slots in the profile's
order. Built by filling in order across the layers, the profile leaves about 1.4 experts a layer off
the card; built from a real routing trace, 1.1 (single card) or 0.05 (with the three helpers). On the
same prompt that is 18.4 vs 29.8 tok/s.

```sh
strata ... --dump-routing trace.bin        # one ordinary run
python tools/make_profile.py trace.bin --no-base --reorder \
    --n-layer 40 --n-expert 256 --out profile.bin
```

**There is no usable speculation for this model.** Whittle ships no MTP block in its GGUF (only
`blk.0`..`blk.39`) and the prompt-lookup drafter accepts nothing on it (0 of 32 drafts over 32 rounds).
A verify window of T rows then runs T-1 rows whose answer the round throws away, at 5.2–6.5 ms a row:
T1 22.58, T2 29.13, T3 34.30, T4 41.67 ms a round. The engine now takes a one-row window when no
drafter is present, which is 32.4 → 40.3 tok/s. `--spec 2` is still required by a native pack; it now
only sizes the captured graphs and the buffers. `--suffix-draft 0` is the fastest setting.

## What did not help

| Setting | Result |
|---|---|
| `--spec 3` / `4` / `5` | Slower (14.9 / 22.7 tok/s): wider windows, no draft accepted |
| `--expert-cache 512` (too few slots) | 10.2 tok/s |
| `--peer-device 1` | 33.8 tok/s; it also cannot be combined with `--expert-cache-device1..3` |
| `STRATA_PF_F16_CUDA=1` (fused FP16 K-quant experts) | Reaches the kernels, no measurable change |
| `STRATA_DENSE_F16_ACC=1` (FP16-accumulate GEMM on Pascal) | No measurable change |
| `STRATA_HC_PERSIST=1` | A HIP kernel; the CUDA build has an empty stub, so it does nothing |
| `STRATA_HC_UPMIX=1`, `STRATA_HCD_EXACT=1` | Prompt path only |
| `STRATA_GDN_CHUNKED=1` / `2` | Within noise (+1.3% / +1.0%) |
| Four cards for prefill | No change |

## Where the time goes

A one-row round on the P100, 22.58 ms, from `STRATA_VERIFY_PROFILE=1`:

| Stage | ms |
|---|---|
| the resident experts' GEMMs (24 layers x 8 experts) | 3.13 + 1.02 |
| the output head (Q5_K, 417 MB) | 2.23 |
| the GDN's qkv projection | 1.99 |
| the hyper-connection down/up (eight of them) | ~3.5 |
| the shared expert and the activation quantization | 1.59 |
| waiting for the helper cards' ring | 0.63 |

The head is a single token against 417 MB of weights, so it is bandwidth and nothing else — but only
187 GB/s of the P100's 730 GB/s, because the Q5_K rows are dequantized block by block. The
hyper-connection reads are eight small products a layer.

## Limits

- The compiled shape is this model's. Another `qwen4exp` file of a different shape is refused at load.
- `--max-context` is capped at 32768: the model's `attention.indexer.top_k` is 262144, equal to its
  context, so the attention is dense and the selection kernel's single block only reaches 32768 cells.
  The engine checks this at load.
- One engine at a time on a 31 GiB machine: the expert arena (18.16 GiB) and the PLE table (5.62 GiB)
  do not fit twice. A leftover engine also drives the load average past 50, which makes every prefill
  number afterwards meaningless.
- No MTP, so no speculative decoding. The suffix drafter is on by default for other models; here it
  costs nothing to turn off and a little to leave on.
- `--remote-expert-opt` was measured on three helper cards but only documented for two, and it
  changes the order of the floating-point summation, so the generated text is not byte-identical to
  a run without it. It was coherent on every run here.
