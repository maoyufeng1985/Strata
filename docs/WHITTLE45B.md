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

## The two configurations this box settled

A 12-hour sweep on the same box (the log is in
[the plan](../plans/2026-10-28-whittle-12h-unattended.md)) ended with one command for the prompt and
one for the decode. Both are opt-in switches; the default path is unchanged.

**The prompt: three helper cards, the rows and the activations in per-row INT8, the chunk at 16384.**

```sh
printf 'GEN 1 %s\n' "$LONG_IDS" | \
env STRATA_ROUTE_PRIOR=0 STRATA_MMQ_KQ=1 STRATA_PREFILL_REMOTE_PEERS=1 \
    STRATA_PREFILL_REMOTE_PEERS_N=3 STRATA_DENSE_F16_ACC=1 STRATA_PF_PEER_COMPRESS=1 \
    STRATA_PF_PEER_I8=1 STRATA_UNBUFFERED_LOAD=1 \
  strata --pack "$PACK" --native "$M" \
    --mmap-experts \
    --ple-gguf "$M" --ple-io ram \
    --max-context 32768 --spec 2 --suffix-draft 0 --expert-profile "$PROF" \
    --gpu 2,1,0,3 --kv int8 --expert-cache auto \
    --expert-cache-device1 auto --expert-cache-device2 auto --expert-cache-device3 auto \
    --remote-expert-opt --conversation-cache-mib 0 \
    --prefill 16384 --serve
```

`STRATA_PF_PEER_I8=1` is what took the 30k cold prefill from 48.8 s to 36.4 s (824-827 tok/s).

`STRATA_PF_PEER_I8_FUSED=1` (with `PEER_I8`) is a further opt-in: it drops one FP32 buffer round-trip on
the helper's activation quantize (the i8 codes go straight to the MMQ's int8 activations instead of
through an FP32 copy - the same numbers, one fewer pass). Measured on this box (2026-10-30, the 30k
arms against the plain i8 arm, two runs each): the cold 30k is 36.41 s without it, 36.48-36.58 s with
it, and the warm batches 17.0 s against 16.9-17.0 s - no measurable difference, so it stays off.

**Persistence mode is on for every visible card by default now.**  The engine turns it on at start-up
(NVML, best effort, `STRATA_PM=off` skips it): all four cards on this box ran with persistence mode
off, parked at the 139 MHz idle clock between the prompt's GEMM waves (measured 2026-10-30).  The
30k wall is unchanged (within the run-to-run noise): the time it buys is on bursty workloads - the
short prompts and the decode steps, where each GEMM wave used to start from the idle floor.  It does
not survive a reboot; the engine re-enables it on every start.

**The helper's DP4A GEMM is already at its best tile for the shape it runs at.**  The MMQ's J (the
tile's output columns) was swept on a P104, Q4_K, one product at a time: at 160 rows per expert,
J=40 is 10 % faster than the default (4.48 ms against 4.98 ms, gate/up); but this prefill moves ~48
rows per expert, and at that shape the default's own choice (J=48, 1.331 ms) is the fastest of the
sixteen, and the 30k wall with J=40 pinned is 1.8 % *slower* warm (17.26 s against 16.96 s,
2026-10-30).  `STRATA_MMQ_KQ_J=<8..128>` pins one J for the K-quant experts (bitwise the default's
products - the tiles change, not the sums), for tuning a shape the switch misses; the default keeps
the upstream choice.  Measured 2026-10-30 on the P104 helpers (192.168.1.98), the `mmq_expert_bench`
Q4_K cases added the same day time it at the engine's shapes.

`--mmap-experts` (with `STRATA_UNBUFFERED_LOAD=1`, which the engine ignores on this flag and
reads the GGUF in place) is what removed the 18.16 GiB pinned expert arena: the 19.5 GB of expert
weights live in the OS page cache (reclaimable), the P100 holds its 4602 profile-ranked experts in
8.19 GiB of VRAM and each P104 its 1880 in 3.3 GiB, and the PLE table is the only pinned RAM
(5.62 GiB, `--ple-io ram`).

A 30,000-token prompt on the P100 with the three P104s (measured 2026-10-29, one binary, three
batches in one process so the first is cold and the next two warm):

| | wall | rate |
|---|---:|---:|
| the first batch, cold (the expert weights not yet in the page cache) | 48.8 s | **614 tok/s** |
| the first batch, cold, with `STRATA_PF_PEER_I8=1` | 36.4 s | **824-827 tok/s** |
| the first batch, cold, `STRATA_PF_F16_PEER=1` + `STRATA_PF_F16_HELPER=dp4a` + `STRATA_PF_PEER_I8=1` | 38.5 s | **778 tok/s** |
| the second and third batches, warm | 22.4-22.6 s | **1335 tok/s** |

**`STRATA_PF_PEER_I8=1` is the best cold configuration measured**: 36.29, 36.39, 36.42, 36.43 s over
four consecutive 30k runs on one binary (`5346c1e53091`) = **824-827 tok/s**, against the same
binary's 48.62 s = 617 tok/s with the switch off (**+33.5 %**). It carries the activations to the
helper cards *and* the helpers' result rows back in per-row INT8: one 8-bit code and one FP32 scale
per row, the grid of amax/127 (a maximum error of amax/254, ~0.394 % of the row's peak; ~0.228 %
rms). The MMQ route's activation transfer was FP32, so its 67 MB per layer becomes 17 MB (**4x
less**), and the return route's FP16 rows halve again to N+4 bytes a row.

Two defects were found and fixed while making this trustworthy, both in the route itself:

- **The FP16-source quantizer addressed the wrong row.** A `uint64_t` holds four halves, and the
  kernel read eight out of one word with a `cols / 2` row stride. The FP16 source is what a remote
  helper cache selects by default, so this was the live path.
- **The return rows were read at the wrong offset.** The helpers write their rows at group-local
  offsets (their bounds are already relative to `rows_local`), but the primary's combine added
  `back_at` (= `rows_local`) on top, so it dequantized from an unwritten part of the buffer. That
  cost time and produced the intermittent `illegal memory access` (1 run in 4) this route showed
  until it was fixed; four consecutive clean runs followed.

The numerical check: a 275-token prompt + 64 decode tokens with the switch on and off printed
312-byte `output` lines that compare **byte-identical**. (An earlier version of this check grepped a
line form this build does not print and silently compared two empty streams - the verdict above
comes from comparing the real output lines.)
The prefill's timeline says where the time went (the 37.6 s run of an earlier binary; the offsets are
the same on the 36.4 s one):

| segment | baseline (48.8 s) | `STRATA_PF_PEER_I8=1` (37.6 s) |
|---|---:|---:|
| `combine` (waiting for the helpers' rows over the links) | 17196 ms (35.2 %) | 3005 ms (8.0 %) |
| `gemm gate/up` (the P100's own expert GEMMs) | 5412 ms | 7348 ms |
| `gemm down` | 2819 ms | 3783 ms |
| helper link route (host staging) | 6231 ms | 9226 ms |

The helpers' link time nearly vanishes and the P100's own GEMMs are what the cards' freed time
buys. **The 1000 tok/s target (30 s) is not reached cold, and this box has not been shown to make it
impossible.** Two bounds are worth keeping apart:

- **A free helper return** (the `combine` 3005 ms removed, the experts left where they are) leaves
  ~34.5 s ≈ 870 tok/s: not enough, so the link is no longer what stands between this run and 30 s.
- **Moving the expert GEMMs off the P100** is the only remaining large block (gate/up 7352 + down
  3783 = 11135 ms). The three P104s are idle 62 % of the wall (measured: idle 23500 / 23375 / 23731 ms
  of 37573; `moe in` 3391 / 3491 / 3429, `moe gemm` 2309 / 2303 / 2001 ms), so there is capacity, but
  per row they are *slower* than the P100: over the prompt the P100 carries 68 % of the routed rows
  for 11135 ms (~2.2 µs a row) and the three P104s carry 32 % for 6613 ms (~2.7 µs a row; their
  memory bandwidth is the lower one). That is why the one measured attempt in that direction lost
  time instead of gaining it. Whether a large enough share moved, with the return route sized for it,
  lands under 30 s is **not established by any measurement here**.

An earlier version of this page claimed the P100's "dense path alone is ~31 s" and therefore a hard
~970 tok/s ceiling. That was wrong: it counted the expert GEMMs (mobile work) as dense (immobile)
work. The non-expert segments of the 37573 ms timeline are hc read 5084 + qsa attn 4570 + gdn 4231 +
qsa select 1950 + qsa proj 1696 + gdn recurrence 1430 + gdn out proj 1051 + hc write 1006 + router 920
+ combine 3040 + small items, about 26 s. The warm prefill (1335 tok/s) does clear 1000.

A unit test for the three quantize/dequantize kernels lives in
`tests/cuda/prefill_peer_i8q_test.cu`: it covers `cols` below, at and above one 256-thread pass
(128 / 2048 / 6144 - the multi-pass case is where a wrong write index hid before), rows whose
amplitudes differ by five orders of magnitude (a dropped warp's amax shows there as a saturated
last element), the FP16 and FP32 sources side by side, all-zero / single-value / at-amax rows, and
asserts the documented bounds (max error amax/254, rms below 0.30 %).

**Two directions that do not help, measured on top of the i8 route (2026-10-29):**

| change | 30k cold | why |
|---|---:|---|
| fewer resident experts on the P100 (`--expert-cache 3600` → 3834 resident) | 38.4 s = 782 | the P100 hands expert GEMMs to the helpers over the Gen1 x4 links, which is slower than computing them locally |
| more resident experts (`--expert-cache 5000`, `5200`) | 37.6 s / 37.7 s | **this arm did not change anything**: its INFO still reports `expert_slots_primary=4602`, the default, so it is a repeat of the baseline and says nothing about raising the count (`5104` quoted earlier is the other arm's `experts_streamed`) |
| the leftover VRAM holds the prompt's KV (`--kv int8 --kv-resident 32768`) | 37.5 s = 801 | on the guarded binary it runs and is a wash; on the earlier binary the same arm reported an illegal access, so this row is one measurement on `a9ca603eff2e`, not a reproduction |
| the same with rotated Q4_0 V (`--kv k8v4`) | 37.9 s | the mixed precision costs more than the bytes it saves |
| the hyper-connections split (`STRATA_HC_SPLIT=1`) | 37.6 s = 799 (earlier binary) | a wash on this configuration |
| the hyper-connection up-mix fused (`STRATA_HC_UPMIX=1`) | — | this build's kernel reports an illegal access on the P100 |
| the hyper-connection GEMM on int8 codes (`STRATA_HC_REQ8=1`) | 38.7 s = 776 | the expansion scratch costs more than the read it saves (`hc read` unchanged at 5073 ms) |
| BF16 SSM/router/indexer projections (`--native-bf16 --native-bf16-extra`) | 37.6 s = 798 (earlier binary) | no segment moves |
| the pinned CUDA QSA/GDN/rope kernels (`--native-qsa --native-qsa-indexer --native-gdn --native-rope`) | — | this build's kernels report an illegal access on the P100 |

So `--expert-cache auto` (4602 on the P100) is the balance point, and the KV stays streamed. Every
one of these is inside a ±1 % band of 37.6 s (or worse): **the cold prompt is now P100-compute-bound,
and no card-arrangement or dense-path switch in this build moves it.**

The P100's own share of the 37.6 s timeline, unchanged by any of the switches above:

| segment | ms | share |
|---|---:|---:|
| expert GEMM, gate/up | 7352 | 19.6 % |
| hyper-connection read | 5084 | 13.5 % |
| QSA attention | 4570 | 12.2 % |
| GDN | 4231 | 11.3 % |
| expert GEMM, down | 3783 | 10.1 % |
| the helpers' rows back (`combine`) | 3040 | 8.1 % |
| QSA select / project / GDN tail / HC write / router | 7053 | 18.8 % |

The expert GEMMs alone are 11.1 s and the helpers' return is 3.0 s; a completely free helper link
would still leave ~34.5 s, and the P100's dense work (~31 s) is the floor.

The ~26 s gap between the cold and warm numbers is the first-touch of the expert weights (the cold
batch reads 34 GB of expert bytes through the page cache and the P104s' links; the warm ones read
8.6 GB, the profile-ranked resident set). What is still bounded is the P100's own dense work (its
BF16 GEMMs run through FP32 on Pascal, already at the card's ~1-12 TFLOPS measured); the helper
cards' Gen1 x4 links (~0.8 GB/s) no longer are.

What each switch is worth, measured on this box:

| Switch | Effect |
|---|---|
| `STRATA_PF_PEER_I8=1` | the helper route's bytes in per-row INT8: **48.6 → 36.4 s on a 30k cold prompt (+33.5 %, 617 → 824-827 tok/s)**, the combine wait 17.2 → 3.0 s |
| `STRATA_PF_PEER_COMPRESS=1` | the helper's rows come back in FP16 (the pre-i8 default), so the PCIe return route carries half the bytes: 483.9 → 585.5 tok/s (**−17.3 %** of the prefill's time) |
| `STRATA_DENSE_F16_ACC=1` | FP16-accumulate GEMM on Pascal: **−5.0 %** of the prompt's time |
| `STRATA_PREFILL_REMOTE_PEERS=1` + `_N=3` | the helper cards compute their share of the prompt's expert rows instead of the P100 computing all of them |
| `--prefill 16384` (over 512) | **−16.2 %** of the prompt's time (the chunk is the biggest single lever) |
| `STRATA_IO_PREFETCH_DEPTH=4|8`, `--prefill 8192` | **no effect** (48.7-49.0 s, inside the run-to-run noise): the staging granularity is not the cold path's limit |

**The decode: two helper cards, and three switches on top.**

```sh
printf 'GEN 256 %s\n' "$IDS" | \
env STRATA_ROUTE_PRIOR=0 STRATA_MMQ_KQ=1 \
    STRATA_QFUSE=1 STRATA_NV_DEFAULTS_PRIMARY=1 STRATA_ADAPT_LAG=2 STRATA_ADAPT_NOWAIT=1 \
  strata --pack "$PACK" --native "$M" \
    --ple-gguf "$M" --ple-io ram \
    --max-context 4096 --spec 2 --suffix-draft 0 --expert-profile "$PROF" \
    --gpu 2,1,0,3 --kv int8 --prefill 512 --expert-cache auto \
    --expert-cache-device1 auto --expert-cache-device2 auto \
    --remote-expert-opt --conversation-cache-mib 0 --serve
```

**49.1 tok/s** for 256 tokens, against 47.7 tok/s for the plain two-helper configuration (**−3.10 %**).
The three switches are `STRATA_QFUSE=1` (the fused int8 tails), `STRATA_NV_DEFAULTS_PRIMARY=1` (the
sm_60 per-arch table; the bigger contributor, −1.22 % on its own, against `QFUSE`'s −0.60 %) and
`STRATA_ADAPT_LAG=2 STRATA_ADAPT_NOWAIT=1`. `--remote-expert-opt` is worth +1.45 % by itself, and
**two** helper cards beat three by 0.16 % on the decode.

Both numbers were measured in one session with one binary, against one base arm, and they are the
hot-batch numbers (the expert arena and the graphs are already warm) — they are the numbers to
compare a configuration with, not to compare with the cold single-batch numbers above.

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

The same flag on a 30,000-token prompt, with the settled switches (the FP16 rows back, the
FP16-accumulate GEMM, the chunk at 16384), gives **median 633.4 tok/s** — see
[the two configurations](#the-two-configurations-this-box-settled).

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
| `STRATA_PF_F16_PEER=1` (the fused FP16 K-quant prompt path, the primary and the helper cards) | **4.4× slower** on the prompt path (614 → 138 tok/s on a 30k prompt, 2026-10-29, all 40 layers on the path): the helper cards' fused FP16 GEMM is ~17× slower than their MMQ GEMM, because Pascal has no FP16 tensor cores. Off by default; it is for tensor-core cards. |
| the same, with `STRATA_PF_F16_HELPER=dp4a` (the primary on the FP16 kernels, the helpers keep their integer DP4A/MMQ kernels) | **Back to the MMQ number: 48.3 s = 621 tok/s on the 30k cold prompt** (the helper's per-layer GEMM falls from ~1.8 s to ~0.1 s). This is the per-card split: on this Pascal fleet the primary's FP16 GEMMs cost its MMQ GEMMs, so the split is speed-neutral against the all-MMQ configuration, and the answer drifts the same way (the first 43 of 64 seeded decode tokens agree with the MMQ arm, then the sampler diverges - the FP16 accumulation, not the helper's path, is what differs from the int8 arithmetic). Opt-in, off by default. |
| `STRATA_PF_PEER_DMA=1` (the helper's FP16 rows back on the copy engine, then converted on the P100's VRAM) | **Slower than the default zero-copy read** (48.8 s → 52.8 s on a 30k prompt): the P100's kernel read of the mapped rows already runs near the link's speed, and the engine's H2D adds a second pass over the same bytes. Measured 2026-10-29, off by default, do not use on this box. |
| `STRATA_DENSE_F16_ACC=1` (FP16-accumulate GEMM on Pascal) | No measurable change on the decode; on the prompt path it is **−5.0 %** (see above) |
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

- **1000 tok/s on a 30k cold prompt is not reachable on this box.** With every switch on
  (`STRATA_PF_PEER_I8=1`, the best measured) the cold prefill is 36.4 s, and the helper links are
  down to 3.0 s (`combine`). **1000 tok/s is not reached, and this box is not shown to make it
  impossible.** Making the helper return free leaves ~34.5 s (~870 tok/s), so the link is no longer
  the obstacle. The only large block left is the expert GEMM itself (gate/up 7352 + down 3783 =
  11135 ms), and the three P104s are idle 62 % of the wall — but per row they are slower than the
  P100 (~2.7 µs against ~2.2 µs), which is why the one measured attempt to move experts to them lost
  0.8 s. Whether a large enough share, with the return route sized for it, lands under 30 s is not
  established by any measurement here. The warm prefill does reach 1335 tok/s.
- `STRATA_PF_PEER_I8=1` is **not numerically verified**: the codes are an amax/127 grid (maximum
  error amax/254, ~0.23 % rms), the "identical token streams" check compared empty streams, and the
  FP16-source quantization kernel has an addressing defect. Treat this switch as an experiment:
  timing measured, correctness open.
- The FP16-primary arm with i8 (`STRATA_PF_F16_PEER=1` + `STRATA_PF_F16_HELPER=dp4a`) still shows
  an intermittent CUDA illegal access at 30k (2 of 4 runs, clean under `CUDA_LAUNCH_BLOCKING=1`);
  use the MMQ + i8 configuration above, which was clean in every run.
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

## The helper levers, measured (P104 Gen1-x4 box, 192.168.1.98, 2026-10-30)

The helper-dp4a plan (the P104 helpers' integer-DP4A MMQ prompt path) closed 2026-10-30, each lever as
one arm: one flag, one 30k arm set (cold + warm x3), one 275-token T-lines arm. The re-runnable arm
script is [`.run/hdp4a-plan-arms.sh`](../.run/hdp4a-plan-arms.sh); every run's source is in the plan
ledger (the wave reports `t0-baseline.md`, `t2-box.md`, `t3-box.md`, `t4-box.md`).

**The frozen baseline** is binary `eb13cd81` on the settled command above: the 30k warm median is
**17.008 s** (the mean of the two reusable runs, 16.973 s and 17.043 s, inside the 16.9-17.1 s range),
the P104 bench cases at R in 32/48/64 (Q4_K gate/up 1.101 / 1.333 / 1.678 ms), and a P100-STATE note:
that day the P100 ran in its degraded state (its R=160 Q4_K gate/up case reads 4.981 ms against the
healthy 1.999 ms; the link and throttle reasons were recorded, not repaired). The P100-side wall
numbers below come from runs on that state.

### SHARED D2H (`STRATA_PF_PEER_I8_SHARED=1`): crash-gated, MEASURED-NEUTRAL, the flag stays off

One D2H: the P100's i8-quantized activations go to one pinned host buffer owned by peer 0, and the
three P104s read the same bytes from their own contexts (the PCIe host reads are independent, no
helper-to-helper dependency). What it moves is the P100-side field of the i8 feed line: the per-peer
cumulative i8 D2H over the 4-GEN arm is 5814 MB a peer, x 3 links = **17.44 GB** of D2H at the
baseline, versus **5.81 GB** with SHARED=1 (the single copy moves the same byte stream once instead
of three times; one third by construction, measured against the 3-GEN shape that gives 7.44 GB /
2.47 GB). The per-peer feed-rate field is unchanged: ~0.53 GB/s on the cold chunk, ~1.70 and ~2.24 GB/s
on the warm chunks (the warm runs reuse the 16384 context and feed less new i8 per layer). The
baseline arm's own peer lines (P104, warm chunk): idle ~13.0-13.2 s, moe in ~2.1 s, moe gemm
~1.4-1.6 s per peer.

The 275-token T-lines arm is **byte-identical** settled vs SHARED=1 (64/64 T-lines against the
frozen baseline; at 64 tokens the i8 readers never run, so the shared mapping is never exercised),
and the default-OFF path is byte-identical to the base: the diff is +132/-9 lines, the 9 removals
being base statements re-homed into the guard's `else` branches, same statements, same order.

The 30k arm never completed: every SHARED=1 run at 30k scale faults on the reader P104, and dmesg
logs hundreds of `NVRM: Xid (PCI:00000000:01:00): 31 ... FAULT_PDE ACCESS_TYPE_READ` entries, all
`name=strata`, all on the reader P104. The fault is driver-level, not index math: the pinned buffer
is allocated under the owner helper's own P104 context, and the two reader P104s' i8 dequant
kernels dereference that host range under their own contexts for the first time at 30k; the
driver's PDE mapping of the cross-context host range is what faults, and the IMA is then sticky
in the reader's context, first reported by the next checked call (`prefill copy_i32: an illegal
memory access`). The tried fixes were timing masks, not fixes: the owner-stream and P104-side
syncs shortened the window, and the 1-byte / 1-word / full-range map warm-ups all still faulted
(the full-range warm stopped at the stop-loss with a new `prefill mmq: iota: invalid argument` at
peer open). No 5th variant was built. The baseline's SHARED=0 arm measures **17.056 s** warm median
(GENs 2-4 of the 4-GEN arm, against T0's 17.008 s); the SHARED=1 arm has no median. The lever closes
**MEASURED-NEUTRAL (crash-gated)**: not byte-safety-gated (T-lines byte-identical, the indexing in
bounds, the default-OFF path byte-identical), but no warm 30k median exists. The flag stays off by
default, and the settled command line is unchanged.

Measured on the P104 Gen1-x4 box, 192.168.1.98, 2026-10-30.

### The K-quant J probe (`STRATA_MMQ_KQ_J_PROBE=1`): off by default, +0.32 % on the 30k

The probe picks the K-quant GEMM's J (the tile's output columns) per (type, row bucket, cc) at
first use: up to 3 candidates (the upstream switch's pick plus the legal J of the same or one more
tile), each warmed and timed once, the winner cached, one printed line per miss. The cost is
ms-class on the first prompt only; the probe is off by default and is not in the settled command
line.

On the P104 (bus `00000000:01:00.0`, `mmq_expert_bench`, Q4_K):

- **R=160: the probe picks J=40, 4.482 ms**, against the default's J=64 at 4.978-4.981 ms. The J=40
  pin reproduces the docs' J-sweep reference 4.482 exactly (0.0 %).
- **R=48, the prefill's ~48-rows/expert shape: the probe picks J=48, 1.331 ms**, the upstream
  default's own pick (0.0-0.3 % against the default bench; J=64 is 1.673 ms, slower).

The host probe test passes 40/40 (the bucket boundaries, the candidate-set construction, the
`STRATA_MMQ_KQ_J` override precedence, the second-call no-probe; ctest 1/1).

The 30k arm, settled env, probe=0 vs probe=1: warm medians **16.928 s vs 16.983 s = +0.32 %**,
under the 0.5 % gate. What it moved in the peer lines (P104, warm chunk): moe gemm +90 ms
(1348 -> 1442 ms), idle -62 ms, moe in +13 ms. At 30k rows the P104's DSO probes its first
K-quant product's bucket (the printed `rows~>512` key) and picks J=64 there, which is the ~90 ms;
the J-axis wins are at the R=48/R=160 bench shapes, not the 30k rows. The probe's arms 2-3 of the
epoch hit the box's Xid-31 PDE-fault epoch (the same driver race the SHARED section records), so
the delta is decided on the clean arms only. With the probe off by default, the settled binary is
unchanged and its 275-token T-lines stay byte-identical to the settled arm; the flag rides only the
probe=1 arm lines.

Measured on the P104 Gen1-x4 box, 192.168.1.98, 2026-10-30.

### The sm61 I=128 overlay (`STRATA_MMQ_SM61_CONFIG=ON`, build-time): ctest PASS, bench neutral, T-lines byte-identical

A CMake option (default OFF, available only when `CMAKE_CUDA_ARCHITECTURES` contains 61 and
`STRATA_EXPERIMENTAL_SM60=ON`), in the TQ-overlay shape: a configure-time copy of the pinned
ggml-cuda header set + `quantize.cu` into the build dir, keyed on the pin `3cf03257f` and a
SHA-256 whitelist, so when the option is OFF the copy is byte-identical to the pinned file. The
change adopts upstream ggml-org/llama.cpp **PR #31352** (issue #30235, open 2026-10-29) verbatim:
I=64 -> I=128 on the two J=64 rows, 11 types including our Q5_1/Q8_0/Q2_0, the tile-only axis,
bit-safe; the issue measured **+12 % pp512 on sm_61**, the P104 class. The copy attaches to both
`strata_mmq` and `strata_kq` (the pinned K-quant instances compile into `strata_kq`; attaching to
`strata_mmq` alone leaves the P104 K-quant path on the pinned table).

Evidence, P104 bus `00000000:01:00.0`, one build with the overlay on (`build-rebase-sm61`):

- The overlay ctest is **PASS 2/2**: every copied header equals the pinned file except the intended
  CASE, and the off-build test proves the OFF path neutral.
- **R=48 Q4_K gate/up: off 1.480 ms vs on 1.466 ms = -0.95 %**, MEASURED-NEUTRAL within ~1 % on the
  same-card pair (no >=10 % gain; against the 1.331 ms J-sweep reference of a different epoch, ON is
  +10.4 % worse).
- A sm_75-only configure prints `STRATA_MMQ_SM61_CONFIG: unavailable - '75' compiles no sm_61 (cc 610)
  code: the overlay stays neutral`, the copy byte-identical to the pin, not attached: sm_75+ release
  builds are byte-identical by construction.
- The 275-token T-lines byte-identical check: **PASS.** The T-lines arm on the sm61 build dir is
  byte-identical to both settled references (`cmp` rc=0 vs the T0 baseline `tlines-T.txt` and the
  T2 SHARED=0 `tlines-T.txt`; all three files md5 758a2343..., 64 lines, arm rc=0, experts_streamed=77).
  This was the last open line of t4-box.md (it read DONE-PARTIAL when the docs section was first
  drafted because the T0 reference was not yet frozen; it now reads STATUS: DONE).

Measured on the P104 Gen1-x4 box, 192.168.1.98, 2026-10-30.

### Two candidate levers that closed without code

**The input double buffer (the old PIPE).** Not built, by the 2026-10-30 feed-cadence research:
producer lock: layer l+1's i8 quantize reads the P100's own layer-l+1 mixed output, which is
enqueued only after layer l's combine, which waits on every helper's `ev_done`; while the P104s
read layer l, layer l+1's bytes do not exist, so an input double buffer has nothing to prefetch.
Its recoverable term is the D2H only, 10.1 ms/layer x 40 layers = **~404 ms** per 40-layer arm,
below the 0.5 s ship threshold. The 0.53 GB/s "feed rate" is the serial span diluted (40 x
116.2 ms matches the measured 4673 ms `moe in` span within 0.5 %), not a link speed; the wire-only
i8 read is 77.0 ms/layer, and the 39 ms/layer dilution is the producer lock, which no buffer
recovers. (The research report's sections 1-2, measured on this box.)

**CUDA graphs on sm_61 (upstream PR #27721).** Closed on a static read, no code, no flag: the
PR's +40 % MoE / +7 % dense figures are ggml-backend graph-run figures, and the prefill path
performs no ggml graph execution: every expert GEMM runs through `mmq::Context::run` (the direct
stream-launch path, our own kernels), and the only `ggml_graph_compute` call in the tree is a CPU
test (`src/kernels/cpu/native_hadamard_test.cpp:128`). The upstream gain does not transfer to this
path, so the candidate closes on the read.

### Deferred and excluded, one line each, with the evidence

- **i4 activations: deferred (owner, for precision).** Halving the i8 wire bytes (the read
  77.0 -> ~38.5 ms/layer, ~1.5 s off the P104 chain per 40-layer arm), now the top helper-side
  byte lever. The i4/7 grid bounds are 7.14 % max / 4.12 % rms of the sub-block peak, ~9.07 / 5.24
  LSB of the q8_1 step, the q8_1 requantize does NOT absorb it; the weakest gate is the
  teacher-forced top-1 >= 272/275 + 62/64 decode; the link bytes are 0.625 N.
- **The P100 sm_60 vmad dp4a patch: deferred (owner-ruled out of scope).** The TQ-style overlay
  mechanism is documented and reusable, and the `dp4a.hpp` vmad form is 2.2x standalone bit-exact,
  but the P100's sm_60 has no hardware DP4A. And until P100-side work lands, the 30k wall is
  P100-bound: the P100's per-layer gate/up + down is 270.5 ms against the P104's whole chain of
  161 ms/layer, so helper levers buy P104 idle time and shorten the combine tail, not the wall.
- **The P104 Gen2-x4 unlock: EXCLUDED (owner, 2026-10-30), and dead by capability.** nvidia-smi
  reports `pcie.link.gen.max = 1` on all three P104s: the negotiated capability ceiling is Gen1,
  so there is no Gen2 target to unlock, and the register-level LNKCAP question is moot at this
  topology.
- **The P104/P100 clock locks: EXCLUDED (owner).** Persistence mode is already default-on: the
  engine turns it on at start-up (NVML, best effort, `STRATA_PM=off` skips it), and there is
  nothing to lock on top of it.

Every measurement in this section was taken on the P104 Gen1-x4 box, 192.168.1.98, 2026-10-30.
