# Task 4: the three tqLLM borrowables, measured before being built — and an audit of the refuted routes

Owner asleep, unattended run, 2026-11-09. Two parts:

1. **Task 4** — the three items left over from the tqLLM survey: per-layer slot allocation, `rebalance()`'s
   stream-sync → event ordering, per-slot independent decay eviction. Each one is measured first, because each
   one is cheap to build and the question is whether there is anything to win.
2. **The refuted-route audit** — the routes this tree has already killed, re-examined one at a time for whether
   the kill still holds against today's code and today's numbers.

Everything below is measured on this box unless it says **static analysis**.

---

## Task 4.1 — per-layer slot allocation: NOT APPLICABLE to this model

tqLLM's `expert_layer_slots()` (`expert_bank_plan.hpp:91`) gives each layer a different slot count when the
layers' expert tensors differ in size, so a layer with small experts does not starve one with large ones. The
q36 port of that arithmetic is already in the tree (`include/strata/qwen35/expert_plan.hpp`,
`plan_expert_slots` + `layer_slots`, 123 unit tests) — but it deliberately returns **the same count for every
main layer**, and the measurement says that is correct here:

- `L.stride` comes from `L.bytes`, which comes from `K::native_expert_layout(type, g.n_embd, g.n_ff)`
  (`q36_engine.cu:2334`).
- `g.n_embd` and `g.n_ff` are **model-wide scalars**, not per-layer fields. Every main layer of Q36 therefore has
  an identical expert size, so a per-layer allocation is a constant vector: the uniform answer already is the
  per-layer optimum.

**Verdict: nothing to change. Winner-take-all is not a defect in this model.** A heterogeneous model (a
different expert size per layer, which is what tqLLM's GLM/MiniMax paths target) would need it, and the header
already has the shape for one — `layer_slots(plan, is_mtp, mtp_f16)` is the one place it differs.

## Task 4.2 — `rebalance()`'s trailing sync: measured, and closed

### What it is

`rebalance()` (`q36_engine.cu:7984`) ends with

```cpp
CK(cudaStreamSynchronize(st));   // slot_of's host copy is pageable: the upload is done before it changes
```

comment explains the reason: `slot_of` is a pageable host vector, so the H2D of `H.slot_of_d` must complete
before the host mutates it again — and the only handle on that is a full stream drain. On a decode step the
stream holds the whole window's work, so this waits for far more than the upload.

### Does it run during decode?  Yes.

MEASURED `.backup/probe_swap.py` (binary `370fabfe`), two sessions, one 39-token prompt, decode length the
only variable. Exit line: `... N slot replacements`.

| decode tokens per slot | swaps | of which decode |
|---:|---:|---:|
| 1     | 243 | — (prompt baseline) |
| 96    | 268 | **+25** |
| 192   | 292 | **+49** |

Perfectly linear, ~0.26 swaps a token. And `rebalance()` is called every `kRebalanceEvery = 32` **windows**
(`q36_engine.cu:8320`), so a 192-token decode makes ~10-12 calls.

### How much does the sync cost?  Almost all of it.

MEASURED with a new off-by-default probe, `STRATA_Q36_REBAL_PROF=1`, which brackets `rebalance()` and splits
its host time at the sync. **The first version of the probe pooled the prompt's calls with the decode's** and
printed one average, which a review caught: rebalance is called with **2048** allowed swaps at a prompt
boundary and with **8** every 32 decode windows, so one prompt call (hundreds of expert blobs) dominates a
dozen decode calls and the "per call" figure would have been the prompt's. The counters are now two sets,
`[0]` prompt and `[1]` decode, and the report prints a line for each.

The first (pooled) run, binary `8e716fb7`, is still worth recording because it is what it is — everything
together:

| arm | calls | swaps | work ms | sync ms | per call (work / sync) |
|---|---:|---:|---:|---:|---|
| dec_32  | 3  | 252 | 4.899 | **78.390** | 1.633 / **26.130** |
| dec_256 | 10 | 308 | 3.121 | **96.367** | 0.312 / **9.637** |

In both arms `work` is a third of a millisecond a call while `sync` is ten to twenty-six: **96 % of
`rebalance()`'s host time is waiting for the stream to drain.**

### The split counters, and why the item is closed

MEASURED, second run, binary `b743d034` (two sets of counters, `prompt` and `decode`):

| path | arm | calls | swaps | work ms | sync ms | per call (work / sync / swaps) |
|---|---|---:|---:|---:|---:|---|
| decode | dec_32  | 1 |  8 | 0.114 |  2.479 | 0.114 / **2.479** / 8 |
| decode | dec_256 | 8 | 64 | 0.992 | 19.780 | 0.124 / **2.473** / 8 |
| prompt | dec_32  | 2 | 244 | 5.618 | 74.148 | 2.809 / 37.074 / 122 |
| prompt | dec_256 | 2 | 244 | 3.157 | 75.721 | 1.579 / 37.861 / 122 |

The decode figure is stable across the two arms to six microseconds a call, which is what makes it
interpretable. Now the arithmetic:

- One expert blob is `18.75 GiB / 40 layers / 256 experts = **1.875 MiB**`.
- A decode call swaps **8** of them: **15 MiB**.
- The box's P100 link was measured at **6.11 GB/s** (`box-pcie-link-measured`). 15 MiB / 6.11 GB/s =
  **2.45 ms**.
- Measured sync: **2.473 ms**.

**A 1 % match. The trailing sync is waiting for rebalance's own 15 MiB of expert uploads, and nothing else.**
That re-frames the whole item: there is no unrelated window work being waited on. And every call site has
already drained the stream immediately before calling `rebalance()` (`:8226` prompt, `:8308` run_rows,
`:4164` after_rows), so the stream at that moment holds *only* rebalance's copies.

**So an event-ordering change cannot help.** The next window's forward reads the slots this call just filled;
the copies must land before it starts, whoever waits for them. Replacing the full-stream sync with an event
would move the wait, not remove it — the same verdict `q36-gpu-wait-dig-closed-20261106` reached for every
other synchronisation reshuffle in this engine, arrived at here from the bytes rather than from a proposal.

### What the whole thing costs, which is the number that actually decides it

- **Decode**: 8 calls over a 256-token, two-session run = **19.78 ms** of sync. That run produces 512 tokens;
  at the ~100 tok/s this stack runs at that is ~5.1 s of wall, so rebalance is **0.39 % of decode**.
- **Prompt**: 244 swaps, ~75 ms of sync across two sessions' prompts, against prompt reads of 674 and 1093 ms
  — so roughly **4 % of the prompt's read**, and it is the placement catching up after a prompt that touched
  many experts at once.

**Verdict: closed, not built.** The decode cost is 0.39 %, it is real data movement, and the only lever is
fewer or smaller swaps — not a different way of waiting for them.

### A side measurement that closes a different question

The CPU pool was measured on the same arms, because "the decode waits on host dashes" is the other candidate
for idle time:

| arm | CPU-pool entries | CPU pool ms | total entries |
|---|---:|---:|---:|
| dec_1   | 19,360 | 1678 | 24,960 |
| dec_96  | 20,323 | 1681 | 85,760 |
| dec_192 | 21,261 | 1848 | 147,200 |
| dec_32  | 19,707 | 1603 | 44,800 |
| dec_256 | 21,878 | 1976 | 188,160 |

The CPU-pool **entry count barely moves with decode length** (19.4 k → 21.9 k while total entries go 25 k →
188 k). So the pool is a prompt-phase cost; during a decode it is **1.7 % of entries** and a few tens of ms —
under 0.2 % of a 256-token decode. That is consistent with `q36-nohost-experts-prompt-gain` and
`q36-serve-prompt-cpu-pool-20261105`: the pool's cost is the prompt's, and nothing about decode.

### A second finding, unplanned: `rebalance()`'s period is inconsistent between the two paths

- decode: `if (++win_count % kRebalanceEvery == 0) rebalance(8);` (`:8320`) — every 32 **windows**.
- prompt: `after_rows()` uses `pos % kRebalanceEvery == 0` (`:4175`) — every 32 **tokens**.

Static analysis only; not measured. The prompt path's rule is the stricter one and the decode path's is the
looser one, which is the safe direction (fewer mid-decode swaps). It is not obviously wrong, so it is recorded,
not changed. Anyone touching this should know the two exist.

## Task 4.3 — per-slot independent decay eviction: the current design is already the conservative one

tqLLM decays a slot's value continuously (`slot_value = freq / (1 + 0.05*age) + prefetch_bonus`) and evicts on
events. q36 halves every count at each rebalance (`for (double& v : H.score) v *= 0.5;`, `:8058`).

The measurement that decides whether to port the tqLLM form is the swap rate, and it is **0.26 swaps a token**
(above). The slot tables are close to stable; there is no churn for a better policy to fix. tqLLM's event-driven
eviction is a response to its own problem (many more experts per byte of VRAM, and a disk tier under them);
neither condition holds here — 6 of 256 experts a layer, and the tier below is a host arena with 18.75 GiB
already fully resident.

**Verdict: not built. There is no measured churn to reduce.** If a future placement makes slots scarce (a
smaller primary card, or a model with more experts), the `refetch_soon`-style counter tqLLM keeps
(`mixed_expert_cache.cpp:90-101`) is the metric to add *first*, because it is what would tell you the churn is
real. Adding the policy before the metric is how the 67→4.9 tok/s incident tqLLM documents happened.

---

## The refuted-route audit

Each entry: what was refuted, on what evidence, and whether that evidence still holds. **Nothing here is a new
measurement** unless it says so — this is a re-read of the code and the recorded numbers, per the owner's
request to go through the refuted routes one at a time.

### 1. GDN micro-optimisation — still dead, and now superceded

`q36-gdn-recurrence` records five refuted levers: column groups 4→16 (neutral), barrier-free (−29 %),
half-warp shuffles (−5.6 %), `__launch_bounds__` occupancy (no better), two-deep load-ahead (neutral). **All
five attacked the sequential scan's stall structure and none of them moved it**, which is the evidence that the
scan's cost is its dependency chain and not its instructions.

The port that landed tonight is that conclusion acted on: the win came from removing the chain (chunks of 32
with 3 barriers a chunk instead of a barrier a token), i.e. changing the decomposition rather than the loop.
**3.312 against 3.825 ms, 13.4 %, disjoint.** So this route is closed *as a micro-optimisation route* and the
remaining headroom is in the same direction (UT, longer chunks), not in the five refuted ones.

### 2. `HELPER_RING` / cross-layer harvesting — still dead

`q36-helper-ring-is-dead` and the re-measurement in `q36-helper-idle-is-the-real-idle`. The mechanism claim is
static and still true as far as this audit checked: `begin_win()` has no caller in the tree, the engine only
calls `begin()` (which is hardwired to slot 0), and `rings_ > 1` therefore only changes which branch `drain()`
takes. The measured kill is −7.16 % (`plans/2026-10-25-q36-decode-mtp-directions.md:106`).

**This audit re-verifies the structural part**: the helper's idle is 55-59 % a layer and it is idle *waiting for
the primary to finish the mixer*, so there is nothing to hide there — the two are sequential by data dependency,
not by scheduling. Cross-layer harvesting would move the wait, not remove it. Route stays closed.

### 3. Two-window interleave — closed by a fixed cost, not by the mechanism

`q36-two-window-interleave-verdicts` and `q36-two-window-fixed-cost`: splitting a window costs **~120 us a
layer** (not the 267 assumed), and the helper already has 55-59 % idle, so the shared card is not the ceiling
and the primary is. `q36-ring-per-slot-stream` shows the per-slot stream implementation landed and **did not
help** (−7.32 % → −6.46 %, and 2 sessions unchanged at −41.5 %), so the root cause was scheduling order, not a
shared stream.

**Still holds.** The implementation is in the tree and is correct to keep; the feature is off and should stay
off. Anything that revisits it must first beat the ~120 us a layer split cost, and no measurement has.

### 4. "Fill the helper idle" — four routes, all negative, all re-checked

`q36-serve-cost-model-and-dead-ends` lists DFlash2 in multi-session (−15.7 %), `--batch-mtp` at N=8 (−11.4 %),
two groups of four interleaved at half a window (−27 %), and widening the window to 10 rows (no change).
The cost model behind them, `T(R) = 267 us a layer + 89.4 us a row a layer`, still fits to a few percent.

**Static check that the blocking conditions are unchanged**: (a) `df2_slot_cap = max(1, min(df2_d, vrows/n-1))`
still collapses to 1 for n ≥ 5, so the fifth slot's window is dropped; (b) `run_rows()` still begins with
`cudaStreamSynchronize(st)` because the pinned staging buffer is single; (c) `mtp_rows()` still synchronises
twice a window. None of the three has been touched since. **Route stays closed.**

### 5. N = 8 plain rows — the one that survived, and it is the current operating point

`q36-nsess-adversarial-verdicts`: 188.6 tok/s aggregate at N = 8 plain (against 135.5 for N = 2 with drafts),
with the four claimed optimisations around it KILLED and four of their numbers falsified. The extrapolation to
N = 10/12 (197 / 205) was never measured and needs `kVerifyMax` raised plus a rebuild.

**Still the only surviving large lever in this family, and it is already taken.** N = 10 is the only untested
point and it is a measured-first question, not a code question.

### 6. "Hide the wait" in general — closed with 15 proposals refuted

`q36-gpu-wait-dig-closed-20261106`: R1-R4, 15 proposals, 0 survived. The reasons are specific: `prebegin` 206 us
is a dead zone (narrow events were tried and were slower), `finish` 311 us is the helper round-trip latency
itself (`ms_wait` 176-216 us, not DMA bytes), and the prefill plan's remaining 1.2 ms is link bytes.

**Still holds, and tonight's work does not change it**: the GDN port removed a *compute* dependency chain, which
is the one category this closed survey explicitly left open ("再往下只有结构性破链"). Every proposal in that
survey was a synchronisation reshuffle, and every one was refuted.

### 7. GQA attention — negative at the sizes q36 actually runs

`q36-attn-gqa-negative`: gqa4 +0.3 % (noise) and gqa8 −1.4 % at 2048 cells, with every md5 identical, so it is
a real measurement of a null. The automatic rule wants ≥ 2048·256 cells before turning it on.

**Still holds.** Not revisited tonight; no code has changed in that path.

### 8. `--bench-tg` as a decode measurement — verified still invalid

`q36-decode-real-path`: `--bench-tg N` runs `E.process(&one, 1, ...)` in a loop, i.e. **one full forward at
T = 1 per token, bypassing `spec_step` entirely.** Every "decode tok/s" produced that way is the plain
non-speculative path with a drafter that never runs.

**Re-verified against tonight's source, `q36_engine.cu:11803-11807`**: the loop is still
`E.process(&one, 1, ...)` with `E.greedy()` between tokens, and there is no `spec_step` in it. So the claim
holds and, more importantly, **it validates tonight's measurements**: the swap/REBAL probes used `--serve` with
`BGEN`, which is the real speculative path, not `--bench-tg`. The GDN speed A/B used `--max-context` plus a
prompt, which reaches the prompt branch (`chunk(T)`), not the decode one.

### 9. GDN decode occupancy — closed by a missing feature, not by a measurement

`q36-gdn-decode-occupancy`: `k_gdn_seq` runs at 100 regs / 1 block / SM / 25 % occupancy while
`k_gdn_seq_cols` gets 48 regs / 2 blocks / 50 % — but `k_gdn_seq_cols` **lacks `rows_mode`**, which is the thing
that makes the decode's MTP windows work. So the higher-occupancy kernel cannot simply be swapped in.

**Still holds, and it is a code-structure blocker rather than a performance verdict.** It is the same shape as
Task 4.1: the better answer exists but the surrounding contract does not admit it.

---

## What was changed tonight

One file, `src/qwen35/q36_engine.cu`, and only an off-by-default probe:

- `STRATA_Q36_REBAL_PROF=1`: brackets `rebalance()`, splits its host time into work and the trailing sync, and
  prints a `REBAL prompt:` and a `REBAL decode:` line at exit. **Off by default; no clock is read with the flag
  unset; the default path's behaviour and output are unchanged.**
- `rebalance()` grew a `bool prompt = false` second parameter so the two populations are counted apart, and the
  three prompt-side call sites pass `true`. **This is a signature change, not a behaviour change**: the only
  reader of the new parameter is the probe, which is off.
- The four probe counter sets are reset wherever `hy_swaps` is (`:9062`, `:9094`), so a probe run with
  `--mtp-calibrate` cannot have the discarded pass leak into the reported totals.

### The review, and what it caught

`VERDICT: FAIL`, one blocker, and it was right: the probe pooled the prompt's calls (up to **2048** swaps each)
with the decode's (at most **8**) into one average, so the `26.13 ms/call` sync it first reported was mostly the
prompt's, and the label said "on the decode path". That is the "arithmetically correct, describes the wrong
population" failure this tree keeps hitting. Fixed by the two counter sets above; **no second review was run**,
per the one-review-per-task rule. The reviewer's nits were also taken: the counters reset with their
neighbours, and the comment now says that a pageable source's staging is charged to `work`, not to `sync`.

Reviewer's UNCERTAINTIES, recorded honestly: it could not compile (no `nvcc` on the working machine, no
configured build tree), so its confirmation of the probe is by reading. The box build (`b743d034`, rc=0) is
what establishes that the code compiles.

Built on the box three times: `8e716fb7` (the pooled version), `b743d034` (the split version), and `7c94ded3`
(the final one). The flag string is present in all three and every run quoted above is one of the probe scripts.

### And it is verified not to touch the default path

The only artefact that predates the edit is the recorded run `.run/ab-gdnwy-20261108.5/{base,base_b}.out`,
produced by binary `370fabfe` on the same 39-id prompt: **`1edb0e03`, 190 tokens, two arms agreeing**. Final
binary `7c94ded3`:

```
probe_off (new binary, flag unset):      1edb0e03/190   BYTE-IDENTICAL to the reference
probe_on  (new binary, REBAL_PROF=1):    1edb0e03/190   same tokens with the probe running
```

So the `bool prompt = false` parameter, the four counter arrays and the extra `fprintf` changed nothing on the
default path, and turning the probe on does not perturb what it measures.

## Two script bugs that produced wrong output first, both mine

Recorded because both would have silently produced a confident wrong answer:

1. **`ab_gdn_wy_speed.py` measured the wrong chunk.** It assumed GDNPROF profiles the prompt it sends; GDNPROF
   actually profiles the **warmup** chunk of `min(tmax, ctx)` rows, so with `--max-context 2048` it reports
   T = 2048 whatever the prompt is. The script then compared that reading against a bar measured at 1024 rows
   and printed "FASTER BUT ABOVE THE BAR" for what is a clear pass. Fixed by parsing `T=` out of the engine's
   own line and refusing a verdict when the arms disagree on it.
2. **`probe_rebal2.py` ran and exited 0 with no output.** Rewriting `main()`'s tail dropped the
   `if __name__ == "__main__"` guard, so the file defined `main` and did nothing. It took two runs to notice
   because the exit code was 0 both times — the same "a silent success is indistinguishable from a bug"
   failure the verdict gates exist for. The lesson is not "check the guard"; it is **check that a script
   printed its first line before believing its exit code**.

## What this does not say

- Nothing about the probe's effect on throughput: every arm was run *with* the probe on, and 3-4 clocks a call
  is not a measurable cost, but the probe-off arm was not run either.
- Nothing about N = 10.
- The refuted-route audit is a code-and-record re-read. Where it says "still holds" it means *the stated
  blocking condition is still present in the source*, not that the effect was re-measured tonight.
- The 0.39 % decode figure divides a measured 19.78 ms by a wall time derived from an assumed ~100 tok/s. The
  19.78 ms and the 512 tokens are measured; the wall is arithmetic.

## The one number worth carrying out of all of this

`rebalance()` costs **2.47 ms of trailing sync per decode call, and that 2.47 ms is 15 MiB at the box's
measured 6.11 GB/s**. The sync is not a synchronisation defect; it is the link. Any future proposal to make
decode faster by touching `rebalance()` must first explain how it will move 15 MiB faster, or how it will move
fewer bytes.
