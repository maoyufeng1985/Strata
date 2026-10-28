# Qwen3.6-35B-A3B: pictures, several requests at once and speed

This page is for people who run [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (35B mixture of
experts, 3B active, 256 experts, top-8) with Strata and want to know how pictures, parallel requests and the prompt
chunk behave. For sizes and memory, see [MODELS.md](MODELS.md).

Strata runs [Unsloth's GGUFs with the MTP layer](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)
(UD-Q4_K_XL, UD-Q6_K_XL, Q8_0) on its own engine path (`src/qwen35`; `strata` switches to it from the file's
architecture). It runs on NVIDIA and AMD cards. The Intel Arc and Mac engines do not run it.

## The MTP block as its own file (`--mtp-gguf`)

The MTP head is a normal part of the model GGUF: `blk.40` plus four `nextn.*` tensors.  `--mtp-gguf FILE` reads
that block from a separate file instead, which is how a head of different precision can be used without
repacking the 23 GB model:

```sh
python tools/q36_mtp_extract.py --from-checkpoint --out q36-mtp-f16.gguf   # from Qwen's BF16 checkpoint
strata-q36 --model model.gguf --mtp 2 --mtp-gguf q36-mtp-f16.gguf ...
```

`--from-checkpoint` range-reads only the 19 `mtp.*` tensors out of the official `Qwen/Qwen3.6-35B-A3B`
safetensors (about 1.6 GB of BF16) and writes them as FP16, so the weights are the original ones and were never
quantized.  `--from-gguf FILE` instead widens a block that is already in a GGUF; that recovers nothing, because
the source was already Q4_K / Q5_K / Q8_0.

An FP16 block is 1.57 GiB and is computed by the cuBLAS F16 GEMM, whose accumulation stays in FP32.  `--mtp 0`
(the default) loads no drafts at all.

The block's routed experts are FP16 as well, and the engine has no grouped FP16 kernel: it runs them on cuBLAS's
FP16 GEMM, one call per expert, with the row ranges brought back to the host (so this block is never inside a
CUDA graph).  Its experts stay in the main card's VRAM - the CPU expert pool has no FP16 format - and they are
built from the GGUF one expert at a time, so no host copy of them is made either.

Two rules make the extracted file read like the model's own block, and both were measured (2026-10-28, the file
against the model's `blk.40`, element by element):

- the expert tensors keep the checkpoint's axis order (`[E, FF, H]` for the gate and the up, `[E, H, FF]` for the
  down), because the GGUF writes the dimensions reversed;
- the seven RMS norm weights are one larger than the checkpoint's, which stores them shifted by -1.

Check a file with the comparison the extract is written for:

```sh
python tools/q36_mtp_extract.py --verify model.gguf --out q36-mtp-f16.gguf
```

It reads the sidecar and the model's own block in each file's memory order and compares the values.  The error
must be the quantization error of the model's copy (about 7% for the Q4_K gate and up, 4% for the Q5_K down, 0.6%
for the Q8_0 weights, and exactly zero for the F32 and BF16 ones).  A bigger error means the layout is wrong, and
it shows up as a non-finite logit, not as a load error.

### What the FP16 head costs, and what it buys

The block's routed experts are 1.5 GiB and they stay in the card (the CPU expert pool has no FP16 format), so the
main layers get fewer of their slots.  MEASURED 2026-10-28 on the P100 12 GB + 3x P104 8 GB box, one prompt of 38
tokens, 128 tokens generated, greedy, the same prompt in every row:

| the head | drafts accepted | tok/s, the helpers on | tok/s, the helpers off | the main layers' slots |
| --- | --- | --- | --- | --- |
| the model's own block (Q4_K / Q5_K) | 59 of 68 (87%) | 24.9 | 25.8 | 42 of 256 per layer |
| the FP16 sidecar | 59 of 68 (87%) | 24.5 | 20.9 | 21 of 256 per layer |

The acceptance column is the helpers-on run; with the helpers off the two heads accepted 58 of 69 (84%) and 59 of
67 (88%).

The two heads draft the same tokens: the FP16 weights change the logits, but not the token the head picks, so the
acceptance is the same.  The same holds deeper - with `--mtp 3` both heads accepted 86 of 121 (71%), and `--mtp 4`
on the FP16 head accepted 93 of 137 (68%).

Where the FP16 head costs is the card: 1.5 GiB of the primary's VRAM, which is the main layers' slots (42 per layer
against 21).  With the helpers serving the decode that is worth about 2% (24.5 against 24.9 tok/s); without them
most of the experts fall to the CPU pool and the decode drops to 20.9 against 25.8 tok/s.

One placement rule follows from it: an FP16 block's experts never stream, so they must not raise the prompt
staging either.  When they did, the staging grew from 1.10 GiB to 3.00 GiB, the main layers' slots fell from 42 per
layer to 0, and the decode fell to the CPU pool with them: 14.6 tok/s against the quantized head's 25.7.

## Quick start

Pick it in the app's model list, in `strata launch` or in setup:

```sh
strata launch --family qwen36 --model ud-q4_k_xl
strata launch --family qwen36 --model ud-q4_k_xl --vision on --parallel 2
```

| Option | Effect |
| --- | --- |
| `--vision on\|off\|gpu\|cpu` | Terminal: pictures and which reader. Source setup uses `--vision yes\|no\|gpu\|cpu` and `--vision-tokens N`. |
| `--parallel N` | How many requests are answered at once, 1 to 8. Source setup has the same option. |
| `--prefill 512` | Engine flag in the config's `args`, not a `strata launch` option: the prompt chunk size ([Prompt chunk](#prompt-chunk)). |

## Pictures

The vision encoder is Unsloth's `mmproj-F16.gguf` (0.9 GB, pinned revision, SHA-256 checked by setup): a 27-layer ViT
with a 2x2 merger, the same projector family as the Flash-Next models' (`qwen3vl_merger`; only its output width
differs: 2048 here, 2560 there). Strata runs it with the same helper (`strata-vision`, llama.cpp's `mtmd`), so the
encoder choices are the Flash-Next ones ([Images](DETAILS.md#images-vision)).

Pictures are on by default, by the same rule as for every model on NVIDIA and AMD cards:

- The CPU reader runs under 24 GB of graphics memory. It takes a few seconds per picture and no VRAM.
- The card's reader runs from 24 GB (GPU encoder, about 1.2 GB of VRAM at up to 1,024 image tokens). On AMD it is
  `strata-vision-hip`, with the CPU reader as its fallback.

Your explicit choice is kept, with a note. A picture becomes up to 1,024 tokens of the context (the CPU default is 300).
The engine puts the encoder's rows where the prompt has `<|image_pad|>` tokens and gives every row its 2-D position,
interleaved M-RoPE as llama.cpp does for this model (rotary sections 11 / 11 / 10).

Pictures work with several requests at once, with the MTP drafts and with the conversation cache. The same conversation
sent again continues from the held state when its pictures are the same (matched by the server's key for each
picture). A different picture starts that conversation again.

> **Measured on:** RTX 3060 12 GB (CUDA) and an R9700 box (32 threads) and a 3060 box (10 threads), UD-Q4_K_XL, Strata
> 0.1.45.

| Encoder | Time per picture (640x400 = 260 tokens / 800x450 = 350 tokens) | VRAM |
| --- | --- | --- |
| GPU, RTX 3060 | 0.10 s / 0.15 s | 1,179 MiB at up to 1,024 tokens (1,337 MiB at 2,048), CUDA context included |
| CPU, 32 threads (R9700 box) | 0.9 s / 1.1 s | none |
| CPU, 10 threads (3060 box) | 1.6 s / 2.1 s | none |

## Several requests at once

`"parallel": N` in the model's config (`strata-qwen36-<size>.json`) or `--parallel N` makes the engine keep N
conversations open (`--batch N`, up to 8) and decode them together. Every window carries one token of each
conversation, so the dense weights, the head and every routed expert that conversations share are read once for all of
them. The server speaks the same engine protocol as for Flash-Next ([BATCHING.md](BATCHING.md)).

A new request waits only for the read of its own prompt, which runs between the decode windows of the others. A
conversation continues in the slot that holds its start, or the engine copies the longer held start from another slot.
A request that runs alone (the solo path, with its MTP drafts) moves into a batch slot when another arrives, and back
when it is alone again, and gives the same text either way.
`STRATA_Q36_READ_WINDOWS=4` lets the decoding slots produce more tokens during a long prompt read, at the cost of a
slower read. The default is one decode window per chunk.

**MTP drafts.** A request that has the engine to itself keeps its MTP drafts. With several requests the batch window has
no drafts unless `--batch-mtp` is in the config's `args` (opt-in). Each slot's window then also carries the proposal its
MTP block made, one per slot, up to the rows to spare in the 8-row window.

**What a slot costs.** Each slot keeps its own cache in VRAM: 1.5 GB at a context of 32K (fp32 keys and values of the 10
attention layers and the MTP block, 45 KB per token) and 0.07 GB of recurrent state. Setup writes the context for one
conversation, so N slots take N times that from the room the experts would have. On a card that does not hold every
expert (12 and 16 GB, the hybrid mode) the expert slots lose that room, and a batch reads more distinct experts from RAM.

Setup offers `--parallel` where the sessions do not eat the experts' room, says the price in the same breath and asks
"How many requests at once?" Enter keeps one at a time, and any number you give is kept.

| Card type | Slots setup offers |
| --- | --- |
| Every expert still fits | Up to 3 to 4 (3 on a 32 GB card at 32K) |
| Hybrid mode | The count whose sessions take at most 35% of the room the experts would have. 12 GB: 4 slots at 8K, none at 32K. 16 GB: 2 slots at 32K. |

The engine says what the sessions take before it allocates them and refuses, with the numbers, a count that cannot fit.

In the hybrid mode the text of a request can differ a little between running alone and running beside others. The
experts that sit on the GPU or on the CPU change with the routing of all the requests, and the two compute with
different rounding. With the slot tables fixed (`STRATA_Q36_FIXED_SLOTS=1`, a test setting) every slot equals its solo
run there too.

## Prompt chunk

The MoE prompt kernels read every expert once per chunk, so a bigger chunk feeds them more rows per expert. With every
expert in VRAM the engine takes the largest of 4096 / 2048 / 1024 that fits beside the experts, the 0.77 GiB reserve
and the sessions' caches.

| Situation | Chunk |
| --- | --- |
| Every expert in VRAM, 8K to 64K context | 4096 |
| Every expert in VRAM, 100K context | 2048 |
| Every expert in VRAM, 131K context | 512 |
| `--batch 2..4` | 1024 at most, so the other slots are not held for a whole chunk |
| `--batch 8` | 512 |
| Hybrid mode | 2048; 1024 or 512 on a card with 7.5 or 5.5 GiB free |

In the hybrid mode a bigger chunk takes expert slots (an RTX 3060 at chunk 4096 read prompts 5% faster and decoded 13%
slower). `--prefill 512` (or `STRATA_Q36_CHUNK=512`) gives the 512 size, and then the text is the base engine's, bit
for bit. The chunk size changes the GEMM tiles the BLAS picks and with them the last bits of the prompt state. This is
the size of the llama.cpp CPU-against-HIP difference on the same file, not a quality loss. A prompt shorter than the
chunk (every chat turn after the first) is one chunk either way.

## Speed and exactness

Every change on the decode path gives the old kernels' bits. `STRATA_Q36_VERIFY=1 strata ... --no-graph` runs the old
kernel beside each new one on the real activations and compares them bit for bit. `STRATA_Q36_OLD_KERNELS=1` switches
the new kernels off. Gate/up for Q4_K experts uses the V2K kernels (`STRATA_EXPERT_V2K`, the default on gfx12).

> **Measured on:** AMD Radeon AI PRO R9700 32 GB (HIP gfx1201), UD-Q4_K_XL, Strata 0.1.45 against the engine before it,
> 5 interleaved pairs, medians; llama.cpp 972b5fc HIP on the same file (`-fa 1`).

| | Before | 0.1.45 | llama.cpp |
| --- | ---: | ---: | ---: |
| Decode tg128 | 83.2 tok/s | 117.8 | 81.9 to 82.1 |
| Prompt pp4096 | 3,453 | 6,220 (chunk 4096) | 3,794 (`-ub 2048`) / 3,042 (`-ub 512`) |

More numbers are in [History](#history--older-measurements).

## Limits

- Eight slots at most (the window holds 8 rows). `--batch-mtp` shares those rows between slots and proposals.
- A sampling request in a batch decodes one token per window without proposals.
- `logit_bias` is not available on this engine path. The server refuses a request that has it (as it does with
  `--batch` for Flash-Next).
- A slot that stopped being alone goes back to the solo path only through the server's own hand-over. The MTP block's
  cache has holes for the tokens a slot decoded in batch windows, so the drafts of a later solo stretch are a little
  worse. The text is not affected.
- Intel Arc (SYCL) and Mac do not run this family.
- The helper-GPU prompt path (`STRATA_Q36_PREFILL_PEERS=1`) is opt-in and experimental: it is faster, but the
  text it gives is not the default path's text, and it is not on by default.

## Testing

The gates are in `tools/qwen35/` and talk to the engine through its serve protocol (no server needed). Run them under
the box's GPU lock.

| Script | Checks |
| --- | --- |
| `q36_solo_gate.py` + `q36_cmp_gate.py` | The 10 parity prompts x 128 greedy tokens through `GEN`, compared with another build's. |
| `q36_batch_gate.py --slots N --waves --long 2500` | Every slot's tokens equal its solo tokens, slots reused, a long prompt read beside decoding slots. |
| `q36_proto_edge.py` | `max_new` 1, `BSTOP`, a busy slot, `STOP` during a prompt read, held-state reuse, sampled requests, the context's end. |
| `q36_interleave_probe.py` | What the decoding slots see while a long prompt is read. |
| `q36_vision_cmp.py` (with `q36_mtmd_dump.cpp` built against llama.cpp) | The picture comparison against llama.cpp. |
| `q36_vision_proto.py` | Pictures through `GENI` / `BGENI` (solo, in a batch, the conversation cache). |
| `q36_vision_serve.py`, `q36_serve_par.py`, `q36_serve_handoff.py` | The same through the server. |
| `q36_par_bench.py` | The aggregate-throughput A/B against `llama-server -np N`. |
| `q36_peer_gate.py` | The peer prompt path's acceptance gate: the helpers served the prompt chunks (the prompts
  are wider than 128 tokens), the path gives the same tokens twice, the default path is unchanged, nothing
  crashed, and the peer path's prompt is not slower than the default path's. The gap to the default path is
  reported, not judged. |

## History / older measurements

### Pictures checked against llama.cpp

> **Measured on:** R9700 32 GB (HIP), UD-Q4_K_XL, Strata 0.1.45. llama.cpp's `mtmd` and its own graph, the encoder on the
> CPU on both sides, the same prompt ids, llama.cpp's answer fed to both, logits compared step by step over 24 steps.
> The floor is llama.cpp's CPU backend against its own HIP run.

| Picture and question | Prompt tokens | First token | Mean KL, llama.cpp HIP vs Strata | Floor |
| --- | ---: | --- | ---: | ---: |
| 640x400, text in the image | 288 | same | 0.029 | 0.056 |
| 640x400, list the shapes | 294 | same | 0.035 | 0.039 |
| 800x450, text in the image | 378 | same | 0.021 | 0.031 |
| 800x450, describe the shapes | 381 | same | 0.0005 | 0.0019 |
| 1024x1024, the title (1,024 image tokens, 3 chunks) | 1,042 | same | 0.0022 | 0.0026 |
| two pictures in one request | 655 | differs (a near tie; the floor's KL at this step is 0.16) | 0.014 | 0.022 |

The answers are the same text as llama.cpp's. With the picture options on, a request without pictures gives the same
tokens as without them (10 prompts x 128 greedy tokens).

### Aggregate decode speed, several requests at once

> **Measured on:** same prompts, greedy, 128 new tokens each, wall time from first request to last answer, 5
> interleaved pairs, medians, Strata 0.1.45 with `--batch 8 --mtp 0` against `llama-server -np N`.

| Card | Requests | Strata tok/s | llama.cpp | Ratio |
| --- | ---: | ---: | ---: | ---: |
| R9700 32 GB, UD-Q4_K_XL | 1 | 83.9 | 73.8 | 1.14 |
| | 2 | 173.5 | 107.4 | 1.61 |
| | 4 | 252.0 | 165.0 | 1.53 |
| | 8 | 325.4 | 208.1 | 1.56 |
| RTX 3060 12 GB, hybrid (llama.cpp `-ncmoe 28 -t 8`, the best split of a sweep) | 1 | 64.0 | 41.1 | 1.56 |
| | 2 | 85.1 | 52.5 | 1.62 |
| | 4 | 99.3 | 63.0 | 1.58 |

All pairs were faster for Strata. Every slot's tokens equal the tokens of the same request alone.

### Several requests with the base engine, and with `--batch-mtp`

> **Measured on:** R9700 32 GB, 5 interleaved pairs, Strata 0.1.45 (before the speed work below).

| Requests | `--mtp 0` | `--mtp 2 --batch-mtp` | Ratio |
| ---: | ---: | ---: | ---: |
| 1 | 83.7 tok/s | 134.0 | 1.60 |
| 2 | 173.1 | 207.0 | 1.20 |
| 4 | 251.0 | 275.0 | 1.10 |
| 8 | 324.0 | 320.5 | 0.99 (no row to spare: no proposals) |

> **Measured on:** hybrid cards, UD-Q4_K_XL, medians of 5 interleaved pairs, `--mtp 2`; "together" is the total tokens
> per second of N simultaneous requests against one request at a time with the draft layer.

| Card, context | Slots | One request alone, against 1 slot | N together, against one at a time |
| --- | ---: | ---: | ---: |
| RTX 3060 12 GB, 8K | 2 / 4 / 8 | -3.5% / -4.3% / -12% | 0.99x / 1.15x / 1.10x |
| RTX A4000 16 GB, 32K | 2 / 4 | -10.6% / -29.9% | 1.03x / 0.76x |
| Tesla P100 16 GB, 32K | 2 / 4 | -7% / -23% | 0.98x / 0.81x |

Without the draft layer (`--mtp 0`) the same slots give 1.26-1.64x together (3060 at 8K: 4 slots 1.64x for -5.5% alone), but a single request
with drafts is already about 1.6x as fast as one without, so on a small card the main gain is that a second user starts at once; the
speed gain is small. On a card that holds every expert (R9700 32 GB, 8K prompts) 4 slots run 1.45x one-at-a-time-with-drafts (1.59x with
`--batch-mtp`). The engine says what the sessions take before it allocates them and refuses (with the numbers) a count that cannot fit.

In the hybrid mode the text of a request can differ a little between running alone and running beside others: the experts that sit on the
GPU or on the CPU change with the routing of all the requests, and the two compute with different rounding. With the slot tables fixed (`STRATA_Q36_FIXED_SLOTS=1`, a test setting) every slot equals its
solo run there too, with and without `--batch-mtp`.

### Prompt read beside decoding slots

> **Measured on:** R9700 32 GB, Strata 0.1.45, three decoding slots.

A 3,008-token prompt was read in 0.95 s (3,200 tok/s, 6 chunks). Each decoding slot paused at most 160 ms (one chunk)
and produced 20 tokens meanwhile. With `STRATA_Q36_READ_WINDOWS=4` they produced 65 tokens during a 1.14 s read.

### Speed work in 0.1.45

> **Measured on:** R9700 32 GB (HIP gfx1201), UD-Q4_K_XL, 5 interleaved pairs, 3 repetitions each, medians; base = the
> engine before 0.1.45.

| | Base | 0.1.45 | Ratio |
| --- | ---: | ---: | ---: |
| `--mtp 2`, 10 prompts greedy 128 | 167.7 tok/s | 183.7 | 1.10 |
| 1 / 2 / 4 / 8 requests (`--batch 8 --mtp 0`) | 85.2 / 182.4 / 271.9 / 361.1 | 122.4 / 205.1 / 303.3 / 386.9 | 1.44 / 1.12 / 1.12 / 1.07 |
| Prompt pp4096 at chunk 512 / 1024 / 2048 / 4096 | | 3,453 / 4,786 / 5,766 / 6,223 | |

UD-Q6_K_XL on the same card (every main layer's experts in VRAM, chunk stays 512): decode 80.4 to 106.9 tok/s, pp4096
2,160 to 2,200 (llama.cpp 3,486 at `-ub 2048`, 2,738 at `-ub 512`), `--mtp 2` 154.7 to 167.3. `--chunk 2048` there moves
the experts to the hybrid placement (205 of 256 per layer in VRAM): prompt 3,500 tok/s, decode 94. That is an option,
not the default.

> **Measured on:** RTX 3060 12 GB (CUDA sm_86, 20 threads), UD-Q4_K_XL in the hybrid mode (37 of 256 experts per layer
> in VRAM), Strata 0.1.45.

Decode tg128 58.5 to 59.7 tok/s (+2%), pp2048 1,958 to 2,022 (+3.3%), 10/10 prompts identical, plain and with `--mtp 2`.
llama.cpp 972b5fc CUDA (`-ncmoe 26 -t 8 -ub 2048`): pp2048 1,328, tg128 48 to 52. The GPU part of a token is the dense
Q8_0 at the card's bandwidth and the rest waits for the CPU pool, so only the small kernels' share is gained.


## The speed work in detail


Decode and prompt kernels of this engine path were rewritten for speed; **every change on the decode path gives the old kernels' bits**
(`STRATA_Q36_VERIFY=1 strata ... --no-graph` runs the old kernel beside each new one on the real activations and compares them bit for bit:
0 mismatches in 18,360 / 8,187 / 6,399 comparisons on the R9700 and the RTX 3060; `STRATA_Q36_OLD_KERNELS=1` switches them all off). The
10 parity prompts (greedy 128) give the base engine's tokens, plain and with `--mtp 2`. Where the prompt's chunk size changes, the numbers
change a little: see "Prompt chunk".

**What was slow** (rocprofv3, R9700, UD-Q4_K_XL, plain decode, 11.25 ms of kernel time per token): the big Q8_0 products (attention, GDN
projections, head: 1.9 GB per token) run at 530 GB/s, the memory bound. The rest did not: the shared expert's and the attention k/v products
(140 per token) at 18 us, the Q4_K gate/up kernel at 250 GB/s, `k_moe_out` (one block) 17 us, `k_group` (one thread walking 256 experts) 9 us.
Now: those products one warp per row (a launch of 2048 warps took 21 us on gfx1201 where 1024 warps took 5 us, so the grid is capped), shared
gate+up+SwiGLU+q8_1 in one launch, the MoE end with 1024 threads and the next layer's q8_1 image, add+norm+q8_1 in one launch, router top-k and
grouping in one launch, the router and the shared gate in one product, Q4_K gate/up on the V2K kernels (`STRATA_EXPERT_V2K`, gfx12 default).
The prompt's GDN conv runs in tiles of 16 tokens side by side and the delta rule loads one token ahead with two barriers per token (both exact).

R9700 32 GB (p620, HIP gfx1201), UD-Q4_K_XL, base = the engine before this work, 5 interleaved pairs, 3 repetitions each, medians; llama.cpp
972b5fc HIP on the same file (`-fa 1`, best of `-ub` 512 / 2048):

| | base | 0.1.45 | ratio | pairs faster | llama.cpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode tg128 | 83.2 tok/s | 117.8 | 1.42 | 5/5 | 81.9-82.1 |
| prompt pp4096 | 3,453 | 6,220 (chunk 4096) | 1.80 | 5/5 | 3,794 (-ub 2048) / 3,042 (-ub 512) |
| `--mtp 2`, 10 prompts greedy 128 | 167.7 | 183.7 | 1.10 | 3/3 | 129 (draft-mtp, n-max 2, earlier run) |
| 1 / 2 / 4 / 8 requests at once (`--batch 8 --mtp 0`) | 85.2 / 182.4 / 271.9 / 361.1 | 122.4 / 205.1 / 303.3 / 386.9 | 1.44 / 1.12 / 1.12 / 1.07 | 5/5 | |

UD-Q6_K_XL on the same card (every main layer's experts in VRAM, so the prompt chunk stays 512): decode 80.4 -> 106.9 tok/s (1.33), pp4096
2,160 -> 2,200 (llama.cpp 3,486 at -ub 2048, 2,738 at -ub 512; decode 81.1), `--mtp 2` 154.7 -> 167.3. `--chunk 2048` there moves the experts
to the hybrid placement (205 of 256 per layer in VRAM): prompt 3,500 tok/s, decode 94 - an option, not the default.

RTX 3060 12 GB (P2003, CUDA sm_86, 20 threads), UD-Q4_K_XL in the hybrid mode (37 of 256 experts per layer in VRAM): the GPU part of a
token is the dense Q8_0 at the card's bandwidth (6 ms of the 17) and the rest waits for the CPU pool (9 ms of experts), so only the small
kernels' share is gained: decode tg128 58.5 -> 59.7 tok/s (+2%), pp2048 1,958 -> 2,022 (+3.3%), 10/10 identical, plain and `--mtp 2`
(84.3 -> 84.7). llama.cpp 972b5fc CUDA (`-ncmoe 26 -t 8 -ub 2048`): pp2048 1,328, tg128 48-52.

**Prompt chunk (the default changed).** The MoE prompt kernels read every expert once per chunk, so a bigger chunk feeds them more rows per
expert: R9700 pp4096 at chunk 512 / 1024 / 2048 / 4096 = 3,453 / 4,786 / 5,766 / 6,223 tok/s. With every expert in VRAM the engine now takes the
largest of 4096 / 2048 / 1024 that fits beside the experts, the 0.77 GiB reserve and the sessions' caches without giving the packed decode
copies back (8K-64K context: 4096, 100K: 2048, 131K: 512; `--batch 2..4`: 1024 at most so the other slots are not held for a whole chunk,
`--batch 8`: 512). The hybrid mode keeps 2048 (a bigger chunk takes expert slots: RTX 3060 chunk 4096 reads prompts 5% faster and decodes 13%
slower) and goes down to 1024 / 512 on a card with 7.5 / 5.5 GiB free. **`--prefill 512` (or `STRATA_Q36_CHUNK=512`) gives the old size, and
then the text is the base engine's, bit for bit.** The chunk size changes the GEMM tiles the BLAS picks and with them the last bits of the
prompt state: KL against the 512 chunk on six real 1.5-3.2K-token documents (48 teacher-forced rows each): chunk 1024 4.0e-3, 2048 4.2e-3,
4096 3.6e-3 (per prompt 2.3e-3 .. 5.9e-3), top-1 agreement 44-48 of 48 rows - the size of the llama.cpp CPU-vs-HIP difference on the same
file (4.2e-3), not a quality loss. Greedy 64 tokens after those six documents: the default chunk gives the base engine's text for 2 of 6,
`--prefill 512` for 6 of 6. A prompt shorter than the chunk (every chat turn after the first) is one chunk either way: unchanged.
The maintainers keep the new default's reference outputs (six documents, greedy 64) and check later changes against them.

**Tried and dropped.** The MTP drafts' head over a subset of the vocabulary (`--draft-vocab N`, kept as an opt-in; the answer never
changes): the first 16K-131K rows gain 5-8% on English and code but lose 10-15% on Chinese / Russian / Arabic / Hindi (70% of a Chinese
prompt's token ids are above 65,536), and a candidate list built from the tokens seen so far loses on both. The shared expert on a second
stream (captured into the decode graph) decodes at 36 tok/s instead of 117 with this ROCm. Chunked (WY) GDN and the PCIe-assisted hybrid decode
were not attempted.

**The fused FP16 K-quant prompt path (opt-in, `STRATA_PF_F16_CUDA=1`).** A K-quant model (Q4_K/Q5_K gate/up, Q5_K down) has three ways
to run the prompt's experts in this engine. The FP16 path dequantizes the whole expert to FP16 - about 10 MB per expert, so it needs
room for the experts. The MMQ int8 path (`STRATA_MMQ_KQ=1`) requantizes the K-quant weights to int8 sub-blocks. The third one, opt-in,
is the fused path: the K-quant codes are dequantized into the registers and multiplied with HFMA2, so no FP16 copy of the weights exists
at all and the expert's room is only its codes.

It covers this engine's MMQ prompt branch: a prompt chunk of more than `--hybrid-rows` rows, with the experts streamed from the host
memory. The decode windows and a served layer (the peers) keep the paths they have. With the flag off nothing changes - the same
prompt gives byte-identical logits.

Measured (P100-12GB as the primary, Qwen3.6-35B-A3B-UD-Q4_K_XL, a 168-token prompt, 21 forced tokens, `STRATA_Q36_CHUNK=512
--expert-slots 24`: the chunk and the experts' placement are pinned, because either one moves the last bits of the logits):

| | measured |
|---|---|
| the flag off (the control binary without the new code, and the new binary, three runs) | the same logits md5 (`9dcaf948...`) |
| the flag on | different logits (`7f582aaf...`); the argmax the same in 20 of 21 positions |
| against the K-quant path (the same prompt with `--hybrid-rows 512`, the native grouped kernels) | the fused path's mean rms difference 0.056 of the logits' rms, the int8 MMQ path's 0.062; the argmax 20 of 21 (the int8 path 21 of 21) |
| the prompt's time (168 tokens, 504 tokens) | 3.55 s, the same with the flag on; 3.56 s (504 tokens), the same with the flag on |

The prompt's time does not move because on this card it is not the experts' kernel that decides it: the
hybrid placement keeps the experts in the host memory and streams them over PCIe (about 20 GB per chunk), and
that covers the kernel's time. The kernel itself got 46% faster while it was being landed (232.9 -> 125.4 us
per expert, one expert's 64-row tile of the gate and up projections, 1.15 -> 2.14 TFLOP/s; the plan has the
steps), and that shows on a card that holds the experts in its VRAM.

The numbers come from [the landing plan](../plans/2026-10-24-q36-fp16-cuda-landing-plan.md) (§8 and §9).


## The prompt's experts on the helper GPUs (opt-in, `STRATA_Q36_PREFILL_PEERS=1`)

the helpers can take the prompt's routed experts: each helper holds its share of the experts in its own VRAM, computes the
rows it owns and sends them back; the primary's MMQ then computes only the entries nobody else holds, and the experts the
helpers own are not read from the host memory at all. The engine prints what it did:

```
strata-q36: STRATA_Q36_PREFILL_PEERS: 3 helper(s) serve prompt chunks (8680 experts held by helpers, 40 layers served,
windows up to 2048 tokens; host route: on); the primary's MMQ computes the rest
```

That is the last big piece of the prompt's time on such a box. In the default path the primary reads every expert it does
not hold from the host memory over PCIe (18.75 GiB of expert weights sit there on the box below) and the helpers sit idle
while it does. The peer path only takes a prompt chunk that is wider than 128 tokens and no wider than the helpers' row
budget, so a short prompt and every decode window stay on the default path.

The flags, all opt-in and all off by default:

| flag | what it does |
|---|---|
| `STRATA_Q36_PREFILL_PEERS=1` | the prompt's experts go to the helper GPUs. The value must be exactly `1`: `=0` and `=x` leave the path off |
| `STRATA_Q36_PREFILL_PEERS_HOST=1` | the rows come back through pinned host memory. Needed on a box without a GPU-to-GPU link (on the P100 + 3x P104 box all twelve ordered pairs answer 0, so that is the route there) |
| `STRATA_Q36_PEER_F16=1` | the rows come back packed as fp16 (half the bytes) |
| `STRATA_Q36_PEER_I8=1` | the rows come back packed as int8, one scale per 128 values (about half of fp16 again). It wins over `STRATA_Q36_PEER_F16=1` when both are set, and it costs precision: the rows are the experts' outputs, and the engine prints what the packing measured |
| `STRATA_Q36_PEER_SUM=1` | the rows come back as one weighted sum per token instead of one row per routed entry, so the bytes follow the prompt's tokens and not the routing's fan-out. It needs a packed transport (`STRATA_Q36_PEER_I8=1` or `STRATA_Q36_PEER_F16=1`) and the host route, and the sum is the same addition in the helper's order - so the peer's text moves a little |

The path needs the helpers open (`--expert-cache-device1/2/3 auto`) and the MMQ prompt path; without one of them the flag is
ignored. The numbers below use the row-split helper mode (`STRATA_Q36_HELPER_PARALLEL=1`), the one where every helper holds
a share of every layer's experts - the mode the decode needs for its speed anyway.

Measured (2026-10-26, Tesla P100-12GB as the primary + 3x P104-100 as helpers, Qwen3.6-35B-A3B-UD-Q4_K_XL, CUDA,
`STRATA_REMOTE_ZEROCOPY=0 STRATA_Q36_HELPER_ASYNC=1 STRATA_Q36_HELPER_PARALLEL=1 STRATA_Q8_SM60=1
STRATA_Q36_HELPER_PARALLEL_MARSHAL=1 STRATA_Q36_FIXED_SLOTS=1` plus the three above, `--bench-pp 512 --bench-tg 64
--no-warmup --max-context 2048`, bench 2, the same binary in both arms - the peer switch is the only difference between the
two columns):

| | default path | the peer path |
|---|---:|---:|
| prompt pp512 | 156.4 tok/s | **589.6** (3.77x) |
| prompt pp512, with `STRATA_Q36_PEER_I8=1` as well | | **738.3** (4.71x) |
| prompt pp512, with `STRATA_Q36_PEER_SUM=1` as well | | **842.0** (5.38x) |
| prompt pp512, with `STRATA_PF_F16_CUDA=1` as well | | **875.9** (5.60x) |
| prompt pp512, with `STRATA_DENSE_F16_ACC=1` as well | | **1001.7** (6.40x) |
| decode tg64 | 58.1 tok/s | 58.9 |
| a 216-token prompt (the prompt part of `--tokens <216 ids> --n 32`) | 65.9 tok/s | **450.0** (6.8x) |

**The KV cache in int8 (`--kv int8`) is worth another 4% on the decode** (measured 2026-10-26 on the same box with the
peer path: tg64 59.3 -> 61.5 tok/s, the prompt unchanged). It is a flag of its own, so it is off by default.

**The window goes up in q8_1.** The helpers' links are the slow part of this path: on the box above a helper's pinned
host-to-device bandwidth is 0.81 GB/s against the primary's 6.03, and the prompt's activations (the "window") were the
biggest thing sent up - 4 MiB per layer in fp32, which was 5.1 ms of the layer's 25.9. The primary now quantizes the
window once, into the format the helpers' products read anyway, and the helpers pick their rows out of it instead of
quantizing the same window again: a quarter of the bytes (1.1 MiB). The numbers do not move - the bytes are the ones
the helpers' own quantization would have written - and the path's tokens are byte-identical to the build before
(measured 2026-10-26: the same six 256-token prompts of the gate, token for token).

**The rows come back in int8 (`STRATA_Q36_PEER_I8=1`).** The way back is the slow part of this path: on the box above a
helper's device-to-host bandwidth is 0.58 GB/s, and the rows (one per routed expert, 2048 values each) were 8.13 ms
of the prompt's 21.6 ms per layer - 38% of the whole prompt. (The same binary with the rows as fp32 instead of fp16
runs at 418 tok/s against 590: the extra 4.7 MB per helper per layer is 8.9 ms.) The rows now travel as one byte a
value with one scale per 128 values, about half of fp16 again. Measured 2026-10-26 on the box above, the same
binary, the flag the only difference between the two arms: **pp512 587.6 -> 738.3 tok/s (+25.7%)**, the forward
21.7 -> 17.3 ms per layer, the decode unchanged (tg64 59.3 -> 59.2). It costs precision: the rows are the experts'
outputs, and the engine measures what the packing costs - the largest magnitude it saw (18.67) and the largest error
as a fraction of the block's own magnitude (0.0039) - and prints both. The number of rows and their order do not
change, so `eparts` still holds the same floats when the combine reads them. The peer's text moves a little with it:
on the gate's six 256-token prompts, five are byte-identical to the fp16 peer's and one differs at token 4.

**The rows come back as one weighted sum per token (`STRATA_Q36_PEER_SUM=1`).** What the primary needs is one weighted sum
per token, not one row per routed expert - and the helper has the rows and, with the routing weights the primary now
sends up, the weights. So the helper adds up its own rows per token and sends one row per token: 512 rows for a
512-token window where the rows were 1152. The bytes then follow the prompt's tokens and not the routing's fan-out.
Measured 2026-10-26 on the box above, the same binary, the flag the only difference between the two arms: **pp512
740.5 -> 842.0 tok/s (+13.7%)**, the forward 17.2 -> 15.1 ms per layer, and the wait for the helpers' rows 3.10 ->
1.04 ms - that is the down leg, the part this flag shortens. With the fp16 transport instead of int8 the same sums
give 782.3 tok/s (twice the bytes on the wire). The decode does not use this path, and it does not move: a decode
layer's sum count is zero, so the combine reads the router's weights and no sums (measured the same day, a
decode-only run: tg64 60.7 with the sums off and 59.3 with them on, with both arms' rounds bouncing between 59.3 and
60.7 - the difference is inside the run-to-run band).

The sum is the same addition the primary would have done, in the helper's row order instead of the primary's, so the
numbers move by the order of the additions and not by the transport's precision. The peer's text moves a little with
it: on the gate's six 256-token prompts, four are byte-identical to the int8 peer's and two differ (at token 8 and
token 13).

**The primary's own experts go through the fused FP16 K-quant path too (`STRATA_PF_F16_CUDA=1`).** On a served layer
the primary still computes the experts nobody else holds - 39 of 256 per layer - and it used to compute them with the
same int8 MMQ the helpers do not use. The fused path (the K-quant codes dequantized in the registers, no FP16 copy of
the weights) is faster for them: the primary's share of the layer's work drops from 4.61 to 3.12 ms. The fused down
kernel now takes a row map, so a served layer can use it: the plan numbers its rows in the entry order and the
combine reads that order, and without a helper the map is the identity - the same bytes as before. Measured
2026-10-26 on the box above, the same binary, the flag the only difference between the two arms: **pp512 839.7 ->
875.9 tok/s (+4.3%)**, the forward 15.18 -> 14.55 ms per layer. The gain is smaller than the 1.49 ms the primary's own
work loses, because the helpers' chain is what the layer waits for (the wait for their rows grows from 0.99 to 2.00 ms
to cover it). The arithmetic changes (fp16 instead of int8), so the peer's text moves - and it moves towards the
default path's: the logits of the same 256-token prompt keep the same argmax at all eight positions (the RMS
difference is 7.4% of the logits' own RMS), and the gate's prompts now agree with the default path's text for 20 and
19 tokens, where the int8 rows agreed for 4 and 13.

**The dense projections accumulate in FP16 (`STRATA_DENSE_F16_ACC=1`).** The prompt's biggest block is not the experts
but the mixer's dense projections - 6.8 of a layer's 14.5 ms. On a Pascal card those run at 6.5-7.3 TFLOP/s with the
FP32 accumulator, and the same product with an FP16 accumulator runs at 12-13, twice as fast. The price is the
accumulation's precision: 0.36% RMS against the FP32-accumulate result on the model's own shapes, measured projection
by projection. Measured 2026-10-26 on the box above, the same binary, the flag the only difference between the two arms:
**pp512 830.6 -> 972.0 tok/s (+17.0%)**, the forward 15.29 -> 13.10 ms per layer, the mixer's 6.77 -> 4.81 ms. It adds
to the fused FP16 experts' gain: with both flags on, the same box reads the prompt at **1001.7 tok/s** against
864.0 (measured the same day, same binary, the dense flag the only difference), i.e. 6.40x the default path's 156.4.
The
decode does not use this path (its projections are GEMVs) and it does not move: tg64 59.3 -> 58.6, inside the
run-to-run band. The text moves a little: the same 256-token prompt keeps the same argmax at all eight positions (the
RMS difference is 8.5% of the logits' own RMS, against the 7.4% the fused FP16 experts already cost) and the eight
generated tokens are identical. It is opt-in and Pascal-only: newer cards run the FP32-accumulate product at the same
rate, and the cards below lack the FP16 tensor cores. Which cuBLAS algorithm wins moves with the shape and the chunk
size, so the engine carries a measured table; only the algorithms that keep the 0.36% error are in it - the ones that
split the K sum inside the FP16 accumulator (ALGO 1, 2 and 3) reach 1.2-1.35%, and 15% on the K 4096 shape, for at
most a third more speed.

**The draft layer's depth is a peak at `--mtp 3`.** On the box above, with two coding prompts (168 and 172 tokens) and the
MTP layer resident on the primary (`STRATA_Q36_MTP_VRAM=1`, the mapped chain live), `--mtp 3` gives 94.1 and 95.9 tok/s
against 60.4 and 60.6 without it (1.57x), with an acceptance of 82-85%. `--mtp 6` gives 88.5 and 93.0 (1.50x, acceptance
69-72%): from the fourth draft on, one step's acceptance falls to about half, so the extra drafts no longer pay for
themselves and the peak stays at 3.

The decode benchmark cannot see this gain: `--bench-tg` counts the tokens the engine commits, and the drafts are not in
that number (tg64 61.1 without the draft layer, 60.4 with `--mtp 3`). The draft layer's cost shows in the prompt instead:
pp512 1002.3 -> 965.7 (-3.7%) with the layer resident on the primary, and -11% in the flag bench. The depth is therefore
worth measuring on the serve path, not with `--bench-tg`.

**The text is not the default path's text, and that is the price.** The prompt's experts are computed by the helpers (q8_1
activations, `native_expert_grouped`) instead of by the primary's MMQ int8, so the prompt's state is a different one: of the
32 rows of logits dumped from the same 216-token prompt, 9 keep the same argmax as the default path's (the largest difference
34.5 on logits of that size), and the generated text is a different text. Each path on its own is deterministic - two runs of
the same path give byte-identical logits - so this is a change of implementation, not a race. With the flags off nothing
changes: the same prompt gives byte-identical logits.

**It was a crash for a while: the defect was recorded on 2026-10-21 and the fixes landed on 2026-10-23.** The helpers read
the experts from a misaligned address (their
cache's slot spacing did not match the primary's step), the NaN walked the rest of the layers, and the greedy pick's fallback
(0x7fffffff, which is not a token id) became an out-of-range row in the embedding lookup - the read past the end of the
table, not corrupted memory. It is fixed: the crash commands of the record are clean now (measured 2026-10-25: `--bench-pp 256
--bench-tg 128`, `--bench-pp 512 --bench-tg 2` and `--bench-pp 160 --bench-tg 8`, with and without
`CUDA_LAUNCH_BLOCKING=1` - no illegal memory access, and no new Xid in dmesg). The engine's own warning on that path says
what is true now.

**The gate that watches this path** is `tools/qwen35/q36_peer_gate.py` (see "Testing"). It runs the peer path twice and the
default path once, on prompts wider than 128 tokens - a narrower prompt never reaches the peer arithmetic, so a gate that
ran on one would pass without ever testing the path - and it checks that the helpers really served the prompt chunks, that
the path gives the same tokens twice, that the default path is unchanged, that nothing crashed, and that the prompt is not
slower than the default path's. It also reports the gap to the default path (the first token where the two paths differ),
which is the number a decision to make this path the default would need.

The numbers and the history are in [the peer plan](../plans/2026-10-25-q36-peer-prefill-today.md), [the push that
took the prompt from 493.9 to 1001.7](../plans/2026-10-25-q36-peer-prefill-push.md) and [the defect
record](../plans/2026-10-25-q36-peer-decode-defect.md).

## Reading the KV once instead of eight times (opt-in, `STRATA_Q36_ATTN_GQA=a`)

The decode's attention has one job per group of query heads: 8 of the 16 query heads share the same keys and values, and
the old kernel read them 8 times, once per head. `STRATA_Q36_ATTN_GQA=a` (or `=2`, `=4`, `=8`) makes one block serve the
whole group, so the keys and values are read once. The tokens do not change - the same expressions over the same numbers,
in the same order - only the traffic changes.

**Measured on:** 192.168.1.98 (Tesla P100-12GB primary, 3x NVIDIA P104-100 helpers), Qwen3.6-35B-A3B-UD-Q4_K_XL, the peer
path on, `--kv int8`, `--bench-pp 32768 --bench-tg 64`:

| context | the flag | decode |
|---|---|---:|
| 40960 | off (the shipped default) | 34.4 tok/s |
| 40960 | `=a` | **51.8 tok/s** (+50%) |
| 2048 | off or `=a` | 60.3 tok/s (the context is too short for the shared shape to pay for itself) |

`=a` decides by itself: the shared shape is used when the decode will read at least 2048 cells. Below that the old kernel
is faster - measured 2.6% slower with the shared shape at 576 cells, 2.1% faster at about 4160 - so 2048 is the
conservative side of the crossing (being wrong there costs about 1%).

The prompt uses the same shape, and gives the same answer: on an 8192-cell prompt the old kernel and the shared shape
give the same 96 tokens (measured 2026-10-27). The traffic drops by 8 and the time by 3, because the shared shape also
runs 8 times fewer blocks and needs 40 registers a thread.

## Reading only the chunks that can matter (opt-in, `STRATA_Q36_QUEST=K`)

`STRATA_Q36_QUEST=K` (K = how many chunks of 256 cells a query row keeps) makes the decode read only the K chunks a cheap
upper bound says can matter, instead of every chunk before the query's position. It is an approximation: the bound is an
upper bound, not the truth.

**Measured on:** the same box and model, peer path on, `--kv int8`, `--bench-pp 32768 --bench-tg 64`
(2026-10-27, `.run/R-questgqa-20261027.sh` / `.run/R-questtime-20261027.sh`):

| | decode | attention ms/tok |
|---|---:|---:|
| GQA=8, every chunk | 51.6 / 51.3 tok/s | 4.411 |
| GQA=8, K = 8 (6% of 128) | **53.4 / 53.3** (+3.5%) | **3.803** |
| GQA=8, K = 16 | 50.6 / 50.7 | — |
| GQA=8, K = 32 | 46.4 / 46.4 | 6.696 |
| GQA=0, K = 8 (old kernel) | 53.9 / 54.1 | — |

Only K = 8 is a net win. The select pass costs ~0.4 ms/tok (bounds another ~0.1); at K = 8 the attention
saving covers it, at K ≥ 16 it does not.

The kept chunks are not exact, so it needs a quality check of its own: at 32K, K = 8 gave the dense answer token for
token over 96 tokens (and K = 4 did not), at 16K K = 4 already differs. At 64K under GQA=8 the same K = 8 already
forks at token 5 (K = 16 / 32 / 64 also early); at 128K K = 8 forks at token 3 (1227 s), K = 16 at token 2
(1235 s), and K = 32 at token 2 (1231 s). A GQA=0 control at 64K (`.run/R-quest64ctl-20261027.sh`) pinned the cause:
g0 dense == gqa8 dense (96 tokens), but g0 k=8 still forks at token 5 — so the early fork is Quest selection itself,
not only the GQA mask mismatch. The shared shape's Quest path had a real bug (kept or dropped a chunk from the
group's first query head's mask while merge reads each head's own mask); the any-head gate in `q36_engine.cu` is
now on the box (`strata-q36.s3` md5 `f05bf102…`). Remeasure (`.run/R-quest-remeas-20261027.sh`): fix dense == OLD
dense (96), fix k=8 == g0 k=8 (96) — mask fix works — but fix k=8 still forks dense at token 5. Details:
`plans/2026-10-27-q36-attn-occupancy.md`.

**What this buys you, in one line:** at 32K the shared shape reads the same answer 50% faster, and the chunk
selection adds 3.5% on top of it — but the chunk selection changes the answer at 64K and above, so it stays off
unless you want the approximation. No K in the measured range (8 to 64) keeps the answer equal to the dense one
past 32K.

## At a 150K context (measured 2026-10-27)

The numbers above use a 32K prompt in a 40K context. Here is the same run with the context at 150,000 cells, the
prompt at 32,768 tokens and 500 decode steps, MTP depth 3, and the fastest flag set measured so far
(`STRATA_Q36_HELPER_ASYNC` + `STRATA_Q36_HELPER_PARALLEL` + `STRATA_Q8_SM60` + `STRATA_Q36_HELPER_PARALLEL_MARSHAL` +
`STRATA_Q36_FIXED_SLOTS` + `STRATA_Q36_PREFILL_PEERS` + `STRATA_Q36_PREFILL_PEERS_HOST` + `STRATA_Q36_PEER_SUM` +
`STRATA_Q36_PEER_I8` + `STRATA_PF_F16_CUDA` + `STRATA_DENSE_F16_ACC`, the three helper caches on, `--kv int8`).
Box: 192.168.1.98 (Tesla P100-12GB primary, 3x P104-100 helpers), Qwen3.6-35B-A3B-UD-Q4_K_XL.

| | tok/s |
|---|---:|
| prompt, 32,768 tokens (random ids, `--bench-pp 32768`) | **483.0 / 481.9 / 482.1** |
| decode, 500 steps (`--bench-tg 500`), GQA unset | **33.8 / 33.9 / 33.9** |
| decode, same shape, `STRATA_Q36_ATTN_GQA=auto` | **50.2 / 50.2 / 50.3** |

The context needs `--chunk 512`: at 150,000 cells the default 2048 asks for more VRAM than the card has left. The 34 figure is the shipped default (GQA off); set `STRATA_Q36_ATTN_GQA=auto` for ~50 tok/s — see below.

A real task at the same size: the prompt is 32,768 **real** tokens (the repository's own source plus an
instruction), and the model writes the code that was asked for.

```
prompt 32768 tokens in 67750.9 ms (483.9 tok/s), 500 generated in 15100.3 ms (33.1 tok/s),
MTP 3 drafts: 340 of 452 accepted (75%)
```

The MTP depth, measured per draft position: d=0 134 of 154 accepted (87.0%), d=1 113 of 152 (74.3%),
d=2 93 of 146 (63.7%).

The 34 → 51 tok/s gap is **only** `STRATA_Q36_ATTN_GQA`. Unset keeps the old one-head-per-block kernel
(`attn_gqa()` returns 0 at `q36_engine.cu:2886`); `=auto` picks the shared shape once the decode reads
≥ 2048 cells. Same binary, peer path on, `--kv int8`:

| GQA | shape | decode |
|---|---|---:|
| unset (default) | ctx 40k, chunk 2048, tg 500 | 33.8 / 34.0 / 33.9 tok/s |
| unset (default) | ctx 150k, chunk 512, tg 500 | 33.8 / 33.9 / 33.9 tok/s |
| `=auto` | ctx 40k, chunk 2048, tg 64 | 50.9 / 51.8 / 51.6 tok/s |
| `=auto` | ctx 150k, chunk 512, tg 500 | 50.2 / 50.2 / 50.3 tok/s |
| unset (`aaoff`, 2026-10-26) | ctx 40k, chunk 2048, tg 64 | 34.0 tok/s |
| `=auto` (`aaauto`, 2026-10-26) | ctx 40k, chunk 2048, tg 64 | 51.0 tok/s |

With GQA on, the 150K decode matches the 40K one (~50 tok/s). Set `STRATA_Q36_ATTN_GQA=auto` for the
long-context decode path.

Ruled out (same binary, GQA unset → all ~34 tok/s):

| context | decode steps | MTP | prompt | decode |
|---|---:|---:|---:|---:|
| 150000 | 500 | 3 drafts | 483.9 | 33.8 / 33.9 / 33.9 tok/s |
| 40960 | 500 | 3 drafts | 489.2 | 34.1 / 34.1 / 34.1 tok/s |
| 150000 | 64 | 3 drafts | 482.6 | 33.6 / 34.1 / 34.1 tok/s |
| 40960 | 500 | none | 561.3 | 34.0 / 34.1 / 34.1 tok/s |
| 150000 | 500 | none | 551.7 | 33.8 / 33.8 / 33.8 tok/s |
| 40960 | 500 | 3 drafts, chunk 2048 | 561 | 33.8 / 34.0 / 33.9 tok/s |

So context, decode-step count, MTP, and prompt chunk do **not** move the decode. MTP and chunk do cost the
**prompt**: ~489 tok/s with drafts / chunk 512 against ~561 tok/s without drafts / chunk 2048, because the
prompt path copies draft seeds and the chunk sets the expert batch size.
