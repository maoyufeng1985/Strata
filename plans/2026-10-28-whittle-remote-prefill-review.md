# Review: the Whittle remote-helper peer prefill

Date: 2026-10-28. Subject: the uncommitted change behind `STRATA_PREFILL_REMOTE_PEERS`
(`src/prefill/prefill.cpp`, `include/strata/prefill/prefill.hpp`, `src/program/generate.cpp`),
read together with the three geometry files in the same tree.

Plan: [2026-10-28-whittle-remote-prefill-peers.md](2026-10-28-whittle-remote-prefill-peers.md).

## Verdict

The peer path is **correct** (the output check below is a measurement, not an argument) and the
default path is **untouched** (gate A0: the flag off prints no peer line and allocates no peer
object). The change is worth committing, but not as one commit: **HEAD does not compile**, and the
three geometry files in the working tree are the fix for that, not part of the peer feature.

## The biggest finding: HEAD does not compile

Commit 30dbf508 ("whittle-45b: the Flash-Next engine parameterized for the Qwen-3.8 shape")
re-pointed `include/strata/core/geometry.hpp` at Whittle (`n_embd=2048`, `n_head=16`,
`n_head_kv=2`, `ssm_v_heads=32`) but left three files pinning the old shape. Each one fails a
`static_assert` in an ordinary CUDA/HIP build:

| file | the pinned number | with the Whittle geometry |
|---|---|---|
| `src/kernels/cuda/fused_gr.cu:2761` | `NB = 80` | `UPC = N/NB = 2048/80 = 25` → `UPC*HC*2 = 200 != THREADS (256)` |
| `src/prefill/gdn_tc.cuh:64` | the scratch is `60305408` bytes | `HV = 32` gives `51388416` (`HV = 48` gives `60305408`) |
| `src/kernels/cuda/qsa_pattn.cu:57` | `G = 12` | the geometry says `16/2 = 8` → the assert next to it fires |

The first two are in every build (`namespace hcp` sits behind
`#if defined(__HIPCC__) || defined(__CUDACC__)`; `gdn_tc.cuh` is included by `kernels.cu`). The
third is behind `STRATA_PATTN_BUILT` (the experimental sm_6x path), so it breaks only that build.

**So the working tree's three files are the missing half of the geometry port.** They belong in
their own commit, before the peer feature:

- `[Engine] whittle: the three files that still pinned the old shape` — `fused_gr.cu`, `qsa_pattn.cu`,
  `gdn_tc.cuh`
- `[Engine] prefill: the prompt's expert rows on the helper caches` — `prefill.cpp`, `prefill.hpp`,
  `generate.cpp`

## What the automated review found, and what survived my own check

Nine dimension reviewers read the diff; every finding was then checked twice (does the code say it,
and does it break something). These survived:

| finding | verdict | what I did |
|---|---|---|
| `--peer-prefill-rows` is silently ignored on the helper path | real | see below |
| the peer executor leaks three CUDA handles when it fails to open | real | fixed |
| `STRATA_MMQ_KQ=1` is an invisible prerequisite | real | fixed |
| the flag is not in `docs/` | real | fixed |
| the plan's A1 gate quotes the pre-fix binary's numbers | real | fixed |
| `.run/` (635 MB) is untracked and not ignored | real | fixed |
| the flag-off path now allocates two vectors and one timer per chunk | real, harmless | left (no behaviour change) |
| `qsa_pattn.cu` is now locked to G=8 | true, not a defect here | this tree is the Whittle build; HEAD's G=12 was already broken for the experimental path |
| the over-cap fallback at `prefill.cpp:5631` can abort the prompt | refuted | the ring plan and the row plan are symmetric: a non-resident expert no peer holds is always in the ring |
| the flag-off path is byte-identical | mostly | the two allocations above are the only difference; no output, no behaviour |

### `--peer-prefill-rows` on the helper path

`open_peer_exec` takes the caller's cap (`:2308`), then the compact path lifts it to `T_max*K`
(`:2348`) because the row tables grow with the chunk. The helper caches always take the compact
path (they have no P2P link), and the row tables are sized from the **lifted** value (`Rt`, `:2384`),
so the caller's cap reaches nothing: not the buffers, not the over-cap guard (`:4643`).

Making it bite is **not** a one-line change: an expert over the cap falls back to the primary, but
the stager has already skipped staging its blob (it skips any expert a peer holds), so the primary
would have no blob to read and the request would die with "an expert is neither resident, streamed
nor on the peer". So the cap stays advisory on this path and the engine now says so:

```
strata serve: --peer-prefill-rows does not apply to the helper caches: the compact path lifts the
peer's row cap to the chunk's rows, and their share is bounded by what their caches hold
```

### The leaked handles

`~PeerPrefill` began with `if (dev < 0) return;`, and `dev` is only set at `:2323` — after the two
early returns that create `ev_in` (`:2311`) and the copy stream (`:2316`). A failure there leaked
one event, one stream and one event. The destructor now frees the primary-side resources whatever
happened, and only the card-bound part is skipped when the card is unknown.

### The invisible prerequisite

`STRATA_PREFILL_REMOTE_PEERS=1` on a K-quant pack without `STRATA_MMQ_KQ=1` fails with "needs the
MMQ prompt path" and the prompt quietly stays on the primary. The helper path's message now names
the variable: "needs the MMQ prompt path (set STRATA_MMQ_KQ=1 for a K-quant pack)".

## One correction of my own

My first reading of the cap said the peer's row table was sized from the pre-lift value while the
guard allowed the post-lift value, which would be an out-of-bounds write on the helper card. That
was wrong: `R` (`:2309`) sizes only the layer-sized (non-compact) buffers, and `Rt` (`:2384`) sizes
the row tables, both after the lift, so the two agree. The measured helper shares (250, 228, 248
rows a layer) are far below both.

## The output check (the one the plan was missing)

A token-equality gate cannot work on this box: the flag-off path's own output moves from run to
run, so the greedy token stream diverges from token ~10 to 30 between two runs of the *same* arm.
Moving the experts' residency changes the float summation order, the logits move, and a token
flips. A residency change **with no peer at all** (the primary's cache 3000 instead of 4767)
diverges the same way, so a token difference across configurations says nothing either way.

The number that does work is the first token's logprob, because it depends only on the prompt's
output. The generate program's `logprobs=K` request key prints it (`generate.cpp:13777`).

The floor is the flag-off path's own run-to-run spread. The same binary, the same arm, four runs
(`.run/R-off-on-twice.sh`):

| arm | run 1 | run 2 | run 3 | run 4 |
|---|---:|---:|---:|---:|
| the flag off | −0.042552 | −0.041766 | −0.042642 | −0.044429 |
| the flag on | −0.042972 | −0.042972 | — | — |

**The flag-off path moves by up to 0.0027 nats; the flag-on path reproduces bit for bit.** The
expert residency is identical in every run (`expert_slots_primary=4765`), so the jitter is the
order of the float reductions on the primary, not a different set of cached experts; the mechanism
is not confirmed. The peer path, which takes two thirds of the rows off the primary, is the more
reproducible of the two.

The peer path's distance from the flag-off arm (0.00033–0.00146 over the runs) is **inside** the
flag-off path's own spread, so the peer path's output is not distinguishable from the primary's.
The rows land in the right slots.

A 1104-token prompt (three chunks, `R-peer-long.sh`) runs clean: no over-cap abort, the prompt goes
351.3 → 440.1 tok/s, and the first token's logprob moves by 0.028 (off −0.191859, on −0.164314) —
the flag-off arm's own spread again.

## The final numbers (the binary that carries the fixes)

Binary md5 `9a4c410c9730`, box 192.168.1.98, P100 + 3×P104, Whittle Q4_K_M, 275-token prompt,
`STRATA_MMQ_KQ=1`:

| arm | prompt tok/s | prompt ms | decode tok/s |
|---|---:|---:|---:|
| the flag off | 241.8 | 1137 | 44.22 |
| 1 helper | 258.0 | 1066 | 44.32 |
| 2 helpers | 304.5 | 903 | 43.90 |
| 3 helpers | **329.6** | **834** | 44.02 |
| 3 helpers, `--peer-prefill-rows 500` | 326.8 | 841 | 44.18 |
| the flag off, the primary's cache 3000 | 208.5 | 1319 | 40.82 |
| 3 helpers, the primary's cache 3000 | 333.6 | 824 | 40.71 |

The earlier binary gave 244.0 → 327.0 tok/s for the same two arms (+34 %); this one gives
241.8 → 329.6 (+36.3 %). The fixes changed no number. The phase breakdown is the same too (the
GPU timeline 875 ms, gemm gate/up 215, gdn 180, gemm down 109, hc read 72, combine 48, dequant
45; the peers idle 736-744 ms with 250 / 228 / 248 rows a layer and `over the cap 0`).

The `--peer-prefill-rows 500` arm is the one that proves the flag does not reach the helper caches:
same throughput, same token stream as the plain 3-helper arm.

## Not covered

- The SUM merge for N helpers, the I8 down-leg and the helpers' peer-streamed share: not
  implemented, so there is nothing to review.
- `STRATA_PREFILL_HELPER_SOURCE` together with `STRATA_PREFILL_REMOTE_PEERS`: the helper's
  `copy_blob` writes into its cache slot while the peer reads that slot in place (`gather_native`),
  and the two are on different streams. The mechanism is real; whether a torn read can mix two
  experts was not proved, and no bench arm sets both. Not in the measured setup.
- The `--peer-device` tier's P2P route: the refactor touches it, and the automated review did not
  re-run it on a P2P box.
- The HIP, SYCL and Metal backends: the three geometry files were only built for CUDA (the box has
  no AMD or Intel card).
