# Whittle / Flash-Next: remote-helper peer prefill → ~1000 tok/s

Date: 2026-10-28. Goal: same box (P100 + 3×P104, no P2P), Whittle Q4_K_M,
bring prompt pp512 from ~214 toward the Q36 peer ladder's **1001.7**.

## Why docs have no switch

`STRATA_Q36_PREFILL_PEERS*` lives only in `src/qwen35/` (`PeerPrefill36`).
Flash-Next Whittle never calls it. Closest upstream pieces:

| Piece | File | Gap |
|---|---|---|
| `--peer-device` + `Prefill::set_peer` | `prefill.cpp`, `generate.cpp:9830` | **1** peer card; **mutex** with `--expert-cache-device1..3` (`generate.cpp:3912`) |
| `PeerPrefill` host route + `STRATA_PF_PEER_SUMS` / `F16` | `prefill.cpp:845-862,2360` | Already has byte-reduction knobs; still one `PeerExperts*` |
| `RemoteExperts::holds` / `slot_ptr` | `remote_experts.hpp:61-69` | Written for q36 peer path; Flash-Next prefill unused |
| `STRATA_PREFILL_HELPER_SOURCE` | `generate.cpp:9280` | Copies blobs only; helpers still **idle** on compute |

Direction was Flash-Next → q36 (design, not copy). Reverse port: **not done**.

`PeerPrefill36` builds only into `strata_qwen35` (`CMakeLists.txt:861`), not the Flash-Next
prefill library. Do not link it into Whittle; re-bind Flash-Next `PeerPrefill` to remotes.

## Ladder (Q36 MEASURED, same box, `docs/QWEN36.md:449-455`)

| Step | Switch stack | pp512 |
|---|---|---:|
| 0 | default (helpers idle on prompt) | 156.4 |
| 1 | peer base + host | **589.6** |
| 2 | + I8 down-leg | 738.3 |
| 3 | + per-token SUM | 842.0 |
| 4 | + `PF_F16_CUDA` | 875.9 |
| 5 | + `DENSE_F16_ACC` | **1001.7** |

Whittle baseline today: **~214** tok/s on short prompt (`noprior/best256`, MEASURED).
`--remote-expert-opt` stays for decode; this work is prompt-only.

## v1 design (Adapter, chosen 2026-10-28)

Flag: `STRATA_PREFILL_REMOTE_PEERS=1` (exact `"1"`, default off → byte-identical).
Host route always on this box (no P2P): mirror q36 `PREFILL_PEERS_HOST` behavior;
optional `STRATA_PREFILL_REMOTE_PEERS_HOST=0` refuses without P2P.

1. **Do not** call `RemoteExperts::begin` (window `n_tok ≤ MAXT≈8`, `remote_experts.cpp:440`).
2. Bind **one `PeerPrefill`-shaped executor per** open `RemoteExperts` (like q36's
   `peers[3]`, `q36_engine.cu:5426`), reading weights via `holds` / `slot_ptr`.
3. Row contract: local rows first, then helper 0..N-1 contiguous runs (same rule as
   `q36_peer_prefill.cuh` row contract).
4. Reuse existing `STRATA_PF_PEER_SUMS` / `STRATA_PF_PEER_F16` before inventing I8.
5. Mutex: refuse if `m.pp` (`--peer-device`) or `help_pp` already set; keep
   `--expert-cache-device*` + `--remote-expert-opt`.

Geometry already matches (`geometry.hpp`: n_embd=2048, n_ff=512, n_expert_used=8).

### Files

| File | Change |
|---|---|
| `include/strata/prefill/prefill.hpp` | `set_remote_peers(...)` |
| `src/prefill/prefill.cpp` | N× remote `PeerPrefill` + row plan |
| `src/program/generate.cpp` | opt-in wire after remotes open (default off) |
| this plan | acceptance |

`generate.cpp` touch is opt-in only (same pattern as `STRATA_PREFILL_HELPER_SOURCE`).
Default path: no new objects, no new stderr.

### Out of v1

- I8 packed down-leg (Q36 `PEER_I8`) — add when SUMS+F16 plateau.
- Changing `RemoteExperts::begin` MAXT.
- Copying `q36_peer_prefill.cu` verbatim (q36 MMQ/eparts contract ≠ Flash-Next Dm).

## Acceptance

| Gate | Pass |
|---|---|
| A0 | flag off: stderr has 0 `PREFILL_REMOTE_PEERS` lines; gen tokens match prior best256 |
| A1 | flag on + host: helpers show prompt launches; no IMA; pp512 ≫ 214 |
| A2 | target band after SUMS/F16: pp512 ≥ ~500 (step-1 band); decode tg not worse than −3% vs 48.2 |
| A3 | full ladder toward ≥900 when I8 (+ optional PF_F16) lands |

## Ceiling honesty

The Q36 ladder's step 1 was 3.8× (156.4 → 589.6) because *Q36's* prompt baseline was
expert-bound. Whittle's is not: the peer share measured here is **+38 %** (238.7 → 327.9 tok/s),
and 1000 tok/s is not reachable on this box by peer steps — the prompt's floor with the expert
share split is ~800-840 ms (330-345 tok/s), and the rest of the prompt is the engine's own work
(the GDN alone is 21 % of it). Whittle already saw `PF_F16`/`DENSE_F16` **neutral without peer**
(`WHITTLE45B.md:132-133`); re-measure **with** remote peers (the primary's remaining experts
only) before spending anything on them.

## What was built (2026-10-28, v1)

**Shape.** One `PeerPrefill` executor per helper cache (the q36 `peers[3]` shape), each on its own
card, reading the weights through `holds` / `slot_ptr` (`PeerSource`, `prefill.hpp:34`).
`Prefill::open_peer_exec` is the one executor body (the old `set_peer` body, unchanged except for
the `PeerSource` adapter); `--peer-device` keeps its slot (`m.pp`), the helpers get `m.rpp[0..2]`.

**Row contract.** The layer's row plan gives every peer its own contiguous run of `Dm`:
the local rows first, then helper 0, 1, 2 in order (`peer_of` / `peer_kind` / `peer_base` /
`peer_rows` in `prefill.cpp:4606+`). The peer's rows land in `Dm` at its own base, so the plain
`moe_combine` covers every slot — **the rows mode**, no merge kernel, no new kernel at all.
The per-token SUM mode stays the one peer tier's (`m.pp`): N helpers' sums would need a merge
(the combine takes one), and the rows mode needs none.

**Switches.** `STRATA_PREFILL_REMOTE_PEERS=1` (opt-in; default off: no objects, no stderr),
`STRATA_PREFILL_REMOTE_PEERS_N` (how many helper caches take part, default: all of them).
Mutex: `--peer-device`, `--layer-split` (a split assigns each layer's experts to one stage).

**The MMQ gate (found on the box).** The peer path is the MMQ prompt path, and the MMQ plan is
empty for a K-quant pack unless the K-quant MMQ kernels are opened: **`STRATA_MMQ_KQ=1`** (the
`strata_kq` DSO next to the engine; `mmq_kq_loader.cpp:67`). Without it the plan has no layer and
the peer path refuses with "needs the MMQ prompt path". So every arm runs with it.

**Not done (v1):** the SUM merge for N helpers (a CUDA-only kernel + a second PCIe read of the
sums), the I8 down-leg, the peer-streamed share for the helpers (the helpers compute what they
hold; `ps_frac` is forced to 0 for them).

## Measured (box 192.168.1.98, P100 + 3×P104, Whittle Q4_K_M, 275-token prompt)

Every arm runs with `STRATA_MMQ_KQ=1` (the MMQ prompt path); the prompt's own time is the `PP` line.
All the rows below are the same binary (md5 `9a4c410c9730`, the one with the review's fixes), run
on 2026-10-28 after 19:04.

| Arm | primary cache | prompt tok/s | prompt ms | decode tok/s |
|---|---:|---:|---:|---:|
| A0 (FP16 path, no MMQ) | 4767 | 214.9 | 1279 | 44.35 |
| **A0 (MMQ path)** | 4767 | 241.8 | 1137 | 44.22 |
| A1, 1 helper | 4767 | 258.0 | 1066 | 44.32 |
| A1, 2 helpers | 4767 | 304.5 | 903 | 43.90 |
| **A1, 3 helpers** | 4767 | **329.6** | **834** | 44.02 |
| A1, 3 helpers, `--peer-prefill-rows 500` | 4767 | 326.8 | 841 | 44.18 |
| A0 (MMQ path) | 3000 | 208.5 | 1319 | 40.82 |
| A1, 3 helpers | 3000 | 333.6 | 824 | 40.71 |

(The FP16 row is from the earlier binary; it is the baseline the MMQ path replaced.  The decode
differs from the 14:12 `best256` (48.2 tok/s): that run used the older `build-flash` tree.  The
earlier binary gave 244.0 → 327.0 tok/s for the same two arms (+34 %), so the two binaries agree
within the run-to-run noise.)

### Output correctness (measured 2026-10-28, the check this plan was missing)

A0's token gate does not hold on this box, so it needs a number that does. The generate program's
`logprobs=K` request key prints one `LP` line per token with that token's logprob and the top-K
(`generate.cpp:13777`). The **first** token's logprob depends only on the prompt's output, so it is
the prompt path's own number: the same 275-token prompt, four arms, `.run/R-peer-lp.sh`:

| arm | what changes | first token's logprob |
|---|---|---:|
| off | the flag off (the primary computes every expert row) | −0.044429 |
| on | the flag on, 3 helper caches | −0.042972 |
| offc3 | the flag off, the primary's expert cache 3000 slots (a residency change with NO peer) | −0.048266 |
| on1 | the flag on, 1 helper cache | −0.038820 |

**The noise floor is the flag-off path's own.** The same binary, the same arm, twice
(`.run/R-off-on-twice.sh`):

| arm | run 1 | run 2 | run 3 | run 4 |
|---|---:|---:|---:|---:|
| the flag off | −0.042552 | −0.041766 | −0.042642 | −0.044429 |
| the flag on | −0.042972 | −0.042972 | — | — |

So the flag-off path's own first token moves by up to **0.0027 nats** from run to run, while the
flag-on path reproduces its value **bit for bit**. The peer path's distance from the flag-off arm
(0.00033–0.00146 over four runs) is inside that band.

(The expert residency is the same in every run — `expert_slots_primary=4765` — so the jitter is not
a different set of cached experts; it is the order of the float reductions on the primary.  The
mechanism is not confirmed.  The peer path, which takes two thirds of the rows off the primary, is
the more reproducible of the two.)

**So the rows land in the right slots.** The token divergence (a0 vs a1 first differ at token 30) is
this setup's own float-order sensitivity, not a peer bug: the flag-off arm differs from itself by
token 24 with the same 90 % rate.  A token difference between two arms is therefore no evidence of
a bug either way — the first token's logprob is the number that is.

### Gates

- **A0 ✓** — the flag off: 0 `PREFILL_REMOTE_PEERS` lines, the default path untouched.
- **A1 ✓** — the flag on: all three helper caches take the prompt's expert rows, no IMA, the
  prompt **+36.3 %** (241.8 → 329.6 tok/s), the decode flat (44.22 → 44.02 tok/s).
- **A2 ✗ on the prompt** — 329.6 < 500. **Measured, not guessed**: the primary's own MoE is
  324 ms of the 875 ms prompt (the phase breakdown below), and moving 17 % more of the expert
  rows to the helpers (a 3000-slot primary cache) bought 4 % on the prompt (208.5 → 333.6 tok/s)
  and cost the decode 8 % (44.2 → 40.7 tok/s). The rest of the prompt is the engine's own work,
  not the experts'.
- **A3 ✗** — not reachable by peer steps here: the prompt's floor with the expert share split is
  ~830-880 ms (330-345 tok/s) and the Q36 ladder's ratios do not transfer (Q36's step 1 was 3.8×
  because *its* baseline was expert-bound; Whittle's is not).

### What the ladder says

Each helper takes a third of the expert share. The gain is not linear: one helper alone is +6.7 %
(one helper's rows do not hide the primary's wait for them), two are +25.9 %, three +36.3 %.

### Where the prompt's time goes (A1, 3 helpers, `STRATA_PREFILL_TIMING=1`)

GPU timeline 875 ms, wall 876 ms, host staging 1 ms. The phases:

| phase | ms | | phase | ms |
|---|---:|---|---|---:|
| gemm gate/up | 215 | | hc read | 72 |
| gdn | 180 | | combine | 48 |
| gemm down | 109 | | dequant | 45 |
| qsa proj | 49 | | gdn out proj | 43 |
| router+shared | 32 | | gather / host grouping | 29 / 14 |

The peers, per card: **~110 ms of work each** (moe in 58-60 + moe gemm 44-54) and **~736-744 ms
idle** — they are not the bottleneck, they wait for the primary. Their share is 250 / 228 / 248
rows a layer (the cap is 4096, `over the cap 0`). The primary's own MoE (gate/up + down) is
324 ms; the rest (~515 ms) is the prompt's own work — the GDN alone is 21 % of it.

**The next lever is not another peer step.** The SUM and I8 down-leg (the ladder's steps 2-3)
shrink the *return* traffic, which is ~64 ms of the 875 ms prompt (the rows come back over the
6.1 GB/s link) → at most a few percent here. The primary's cache size trades the prompt for the
decode (measured above). What is left is the prompt's own work: the GDN (180 ms), the HC read
(72 ms), the combine (48 ms), the dequant (45 ms).

## Review (2026-10-28)

[2026-10-28-whittle-remote-prefill-review.md](2026-10-28-whittle-remote-prefill-review.md): nine
dimension reviewers over the diff, every finding then checked twice (does the code say it, and
does it break something). What it changed:

- **HEAD did not compile.** The geometry port (30dbf508) left `fused_gr.cu`, `qsa_pattn.cu` and
  `gdn_tc.cuh` with the old shape, and each fails a static_assert in an ordinary CUDA/HIP build
  (`UPC*HC*2 = 200 != THREADS`, the scratch's `60305408 != 51388416`, `G = 12 != 16/2`). The three
  files in this tree are that fix, not part of the peer path: they belong in their own commit.
- The peer executor leaked three CUDA handles when it failed to open (the destructor returned
  before the card was known). Fixed.
- `STRATA_MMQ_KQ=1` was an invisible prerequisite: the flag failed with "needs the MMQ prompt path"
  and the prompt quietly stayed on the primary. The message now names the variable.
- `--peer-prefill-rows` does not apply to the helper caches (the compact path lifts the row cap,
  and the row tables are sized from the lifted value). The engine now says so instead of taking
  the flag silently. Making it bite is not free: an expert over the cap falls back to the
  primary, but the stager has already skipped staging its blob, so the request would die.
- `has_peer()` was dead code; `set_peer_from` is now private.

The output correctness check above is the plan's own: the token gate cannot work on this box, and
the first token's logprob can.



