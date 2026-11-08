# The GDN recurrence as two phases: what makes tqLLM's chunked scan 4.15x, and the port

## The question

Q36's prefill spends 3.80 ms per 1024-token half in the GDN recurrence
(`k_gdn_seq_cols`), 97 % of it the sequential chain. A chunked (WY) form was
already implemented here on 2026-11-04 and **lost by 12 %**
(`k_gdn_chunked`, `STRATA_Q36_GDN_CHUNKED=1`, 6.862 against 3.934 ms GDNPROF).
tqLLM ships the same recurrence, chunked, and reports **4.15x on GB10**. The
task is to find what is different and port it.

## What I read

- `.ghsurvey/tqllm/src/models/qwen/qwen4exp/gdn_seq_fast.cuh` (472 lines, read in full)
- `.ghsurvey/tqllm/src/models/qwen/qwen4exp/qwen4exp.cu:569-679` (the two kernels),
  `:1330-1399` (`fn_gdn`, the dispatch)
- `src/qwen35/q36_engine.cu:1535-1735` (our `k_gdn_seq_cols` and `k_gdn_chunked`)

This is static analysis of both codebases. **No measurement yet** - the box has
not been touched for this.

## The finding: the barrier count, and it is not the one our note expected

Our note (`q36-gdn-chunked-wy-derivation`) concluded the fix was "the UT
transform plus L=64". Reading tqLLM says that is **not** what it does, and the
real difference is structural:

| | our `k_gdn_chunked` | tqLLM (`fn_gdn_wy_k` + `fn_gdn_seq_fast`) |
|---|---|---|
| (I+A)^-1 | forward substitution **inside** the scan, one `__syncthreads()` per token (`:1714-1718`) | computed **whole** in a separate parallel kernel, no barrier in the scan |
| barriers per chunk in the scan | L (=16 in the current constant) | 3 (load, E ready, before S update) |
| S layout | `(row v, dim d)`, so the outer-product update walks dim with stride S | `(d, v)` transposed, so a `float4` of v columns is contiguous |
| chunk length | 16 | 32 |
| width of the scan | `(hv, 4 col groups)` | `(vh, dim/VS)` = `(48, 2)` with per-block 64-column slabs |

So the whole of the 4.15x is available without any UT transform, and without
changing the arithmetic at all: the same [(I+M)^-1] is formed, just in the
parallel phase where 32 chunks of one head run at once instead of one after
another. Our own implementation did the substitution in the sequential phase and
paid 16 barriers a chunk for it.

The two phases, as tqLLM runs them:

- **phase 1** `fn_gdn_wy_k`, grid `(vh, chunks)` - every (head, chunk) at once.
  Forms `A[t][j] = beta_t * P(t,j) * (k_t . k_j)` and `M[t][j] = P(t,j) * (q_t.k_j)`,
  then `T = (I+A)^-1` by forward substitution with **one lane per column** (the
  32 columns of the inverse are independent, so this is 32 lanes in parallel, no
  barrier), then `U = T diag(beta) V`, `W = T diag(beta) diag(D) K`, and `lg = log D`.
- **phase 2** `fn_gdn_seq_fast`, grid `(vh, dim/64)` - walks the chunks in order
  with only the affine recurrence `E = U - W S^T`, `O = g (Q S^T) + M E`,
  `S = g_C S + sum_j (g_C/g_j) e_j k_j^T`. Three barriers a chunk, no solve.

The `D_t` trap from our derivation is the same one tqLLM works around: it keeps
`lg = log D` and forms every ratio as `expf(lg[t] - lg[j])`, which is why its
per-chunk factors never need the `g_0` correction that cost us a version.

## Port shape for q36

Shapes: `dim = 128`, `vh = 48`, `kh = 16` (3 v-heads a k-head), prefill T = 1024
in halves. Intermediate buffers at T = 1024: `U`+`W` 50.3 MB, `M` 6.3 MB, `lg` 0.2 MB
= **56.8 MB**, allocated once and grown only.

1. `k_gdn_wy` (new): grid `(hv, chunks)`, block 256. Mirrors `fn_gdn_wy_k` at
   C = 32. Writes `U`, `W`, `M`, `lg` to a workspace.
2. `k_gdn_seq_wy` (new): grid `(hv, dim / VS)`, block 256, `S` transposed. Mirrors
   `fn_gdn_seq_fast`.
3. Behind `STRATA_Q36_GDN_WY=1`, **default off**. The summation order differs from
   `k_gdn_seq_cols`, so the default path must stay byte-identical.

## Before any of it: the arithmetic check (done)

`tools/q36_gdn_wy_check.py` is new and compares the two-phase form against the
sequential recurrence in fp64, written independently in the kernels' own
notation. **7 cases, all OK**, the real shape (DK = DV = 128, T = 1024, C = 32)
at **3.25e-12** relative. It covers a ragged tail and, deliberately,
`DK != DV` - tqLLM takes one `dim` for both because Flash-Next has 128 of each,
and a single symbol is exactly what lets a transposed state or a swapped index
pass every shape check.

Rejected on the way, both worth keeping in mind for the kernel:

- **A gate with mean zero is not a valid test.** The first version drew
  `gate ~ N(0, 0.3)`, so `g = exp(gate)` exceeded 1 half the time, the recurrence
  amplified, and 1024 tokens reached 1e157 - past fp64's range, where `exp()`
  overflows and every case reads as a total mismatch. Real GDN gates are a decay.
  With `gate < 0` the same code is exact to 1e-12.
- `gate` in Q36 **is** log g (the kernel does `g = exp(gate)`); tqLLM's `gates`
  already holds `exp()`ed values, which is why its kernel calls `logf` on them.
  Copying the kernel without that flips the scan's input.

## Build notes (both cost a round trip)

- `k_gdn_wy` needs **58 KiB** of shared memory at C = 32 (three `C x 129` panels, two
  `C x 33`, two `C`), past the 48 KiB static ceiling: `ptxas error: uses too
  much shared data (0xe380 bytes, 0xc000 max)`. Both kernels therefore take
  `extern __shared__` and set `cudaFuncAttributeMaxDynamicSharedMemorySize`.
- After that change the panels are `float*`, so `sK[t][d]` no longer compiles
  (15 errors). Flat `sK[t * P + d]`. Worth knowing before writing more of these.

## The first real verdict: DIFFERENT WORDS, and why that is not yet a bug report

Three arms on the fixed prompt, 2026-11-09 01:27-01:34, binary
`strata-q36` md5 `370fabfe27ad3d99dc5349379d271446`:

```
base    rc=0  md5=1edb0e03  190 tokens [95, 95]  OK
wy      rc=0  md5=10cc532f  190 tokens [95, 95]  OK   (wy_ran=True)
base_b  rc=0  md5=1edb0e03  190 tokens [95, 95]  OK
```

`base` reproduces exactly, so the prompt is stable and the middle arm may be
read. It does not agree with it. The engine also printed, unaided,
`GDN_WY: the two-phase recurrence ran (T=2048, 64 chunks, 32 heads)` - so the
branch is definitely what produced those tokens, not a silent fallback.

**Why this is not yet "the recurrence is wrong".** The two paths sum the same
terms in a different order, so they cannot be bit-identical; the question is
whether the difference is a reordering's ~1e-6 or a defect's ~1e-1. The token
stream answers a different question, and it answers it in the way a near-tie
does: the first **101 of 190** tokens are identical, then slot 0 flips at its
52nd token and slot 1 at its 53rd. An index or layout error does not survive 101
tokens - B2/B3 in the table above are both defects that would have diverged on
the first token. So the shape of the failure says "float noise", but that is an
argument, and this project does not ship arguments.

**The decisive measurement.** `.backup/ab_gdn_wy_dump.py` uses the engine's
existing `STRATA_Q36_SPLIT_DUMP_MIX` hook to write the GDN branch's own output
(`gdny`, after `k_gdn_norm`, layer 0) for the same prompt under both flags, and
compares those tensors directly. Bars: relative difference `<= 1e-5` means the
port is right and the flip is a near-tie; `> 1e-2` means it is wrong; between
the two, no verdict. `base` vs `base_b` is dumped too, so a box that cannot
reproduce its own GDN output is caught before the middle arm is read.

### The answer: the recurrence agrees to 2.8e-06

```
l000_gdny_t02048_a00000.bin   n=8,388,608  peak=1.298  max|d|=3.692e-06  rel=2.843e-06  differing=7,206,721
l000_gdny_t00009_a00000.bin   n=36,864     peak=1.298  max|d|=3.722e-06  rel=2.867e-06  differing=32,410
```

Both the 2048-token prompt chunk and the 9-token decode step land at **2.8e-06
relative**, which is the size a reordering of these sums produces in fp32 - and
three orders of magnitude below anything an index or layout error would give
(those produce O(1), and would have shown on the very first token, not the 101st).
`max|d| = 3.7e-06` against `peak = 1.298`.

So: **the two-phase recurrence is arithmetically correct**, and the token flip
is the sampler resolving a near-tie that the last few ulp decided. That is
exactly why `STRATA_Q36_GDN_WY` is opt-in and must stay opt-in - it cannot be
byte-identical, and the default path's text has to be.

The rest of the dumps make the attribution exact. Every tensor the port does not
touch is **bit-identical**, and the ones downstream of it are not:

| dump | what it is | rel diff |
|---|---|---|
| `qkv` | the GDN branch's projection, the port's input | **0** (16,777,216 floats) |
| `al`, `be` | the recurrence's gate and beta, its other inputs | **0** |
| `gdny` | **the recurrence's output - the port's own result** | **2.84e-06** |
| `y` | the mixer's exit, one `ssm_out` later | 4.56e-04 |
| `ep`, `sy` | the MoE combine inputs | 0.88 / 2.2e-03 |

The inputs are identical and the output differs by 2.8e-06: that is the port, and
nothing else. The growth from 2.8e-06 to 4.6e-04 across one matmul is the
amplification that eventually flips a token 52 steps later.

`ep` and `sy` are **not** evidence either way and should not be quoted as such.
The `base` vs `base_b` comparison settles it: with the port not involved at all,
two runs of the same binary give `ep_t02048` **rel 8.9e-01** (33.4 million of its
33.6 million floats differ) and every other dump **rel 0.000e+00**. `ep` is a
MoE expert buffer whose contents depend on which experts were resident and which
lane each row took; it varies between identical runs, so its difference under the
port says nothing.

That same run is what makes the whole attribution possible: **the box reproduces
`gdny` bit for bit when nothing changes**. So the 2.8e-06 in the middle arm is
the port and only the port.

### What the dump script got wrong first, and why it matters

The first version compared *every* dumped file and took the worst. `base` vs
`base_b` then came back at 8.9e-01 (from `ep`), the script declared the box
irreproducible, and - worse - it printed **both** `WRONG ARITHMETIC` (from that
same `ep` number) and `NO VERDICT` in the same run. Two contradictory verdicts,
because the "which tensor is the port's" question had never been answered before
the thresholds were written.

The fix is the one this project keeps relearning: name the tensor the change
actually produces (`gdny`), establish its reproducibility floor separately
(0.0), and judge only that. Everything else is printed for the record and
explicitly excluded from the verdict - with `ep`'s own floor quoted next to it so
a reader cannot mistake it for evidence.

The dump also corrected one number in this document: `HV` is **32**, not 48
(`l000_al_t02048` is `T*HV*4 = 262144` bytes, so `HV = 65536/2048 = 32`). Every
buffer size above that used 48 is that much too large.

## Success criteria

- fp64: the T-matrix form equals the sequential recurrence to ~1e-14. - **DONE**
  (3.25e-12, `tools/q36_gdn_wy_check.py`, 7 cases).
- box, byte-identical text on a fixed prompt with the flag on and off. - **NOT
  MET, and it cannot be.** Measured: 2.84e-06 relative on the recurrence's own
  output, which flips a token at step 52 of 95. The arithmetic is right; the
  text is not and will not be. This criterion was wrong when it was written - it
  is the criterion for a change that does not reorder a sum.
- box, GDNPROF: the recurrence below **3.80 ms** per 1024 tokens (the number to
  beat). Below 2.0 ms would be the 1900 tok/s the earlier note estimated as the
  ceiling. - **MET: 3.312 ms, 13.4 % faster, disjoint ranges.** See the
  measurement section below. It is below the bar but far from 2.0 ms.

So the honest state is: **correct, measured, and worth carrying behind the flag** -
13.4 % off the GDN recurrence, which is 97 % of a kernel that is 26 % of the
whole prefill forward, so ~3.4 % off prefill in total. Not the 4.15x tqLLM
reports on GB10, and the reason is worth stating: tqLLM's baseline is its own
sequential scan on a card with ~20x this one's shared-memory bandwidth per SM,
so its 4.15x is a ratio between two kernels on hardware where the barrier is not
the dominant cost. Ours starts at 3.7 us a token, where the barrier *is* the cost,
and the port removes the barriers it can (L per chunk -> 3) but not the chunk
count: phase 2 still walks 32 chunks a half, each with 3 barriers.

## The speed measurement: 3.312 against 3.825 ms at T = 1024

`.backup/ab_gdn_wy_speed.py`, box 2026-11-09 01:5x, binary md5 `370fabfe`.
Interleaved arms (base, wy, base, wy), same prompt, the profiler on in both.

```
base_a  3.825 ms (T=1024)      wy_a  3.307 ms (T=1024)
base_b  3.824 ms (T=1024)      wy_b  3.318 ms (T=1024)

base: 3.824 .. 3.825  (the box's own spread: 0.001 ms)
wy  : 3.307 .. 3.318  (spread 0.011 ms)
DISJOINT: faster by 13.4 % (3.312 against 3.825 ms)
BELOW THE BAR: 3.312 against 3.800 ms
```

The base arm reproduces the recorded 3.80 ms bar to a thousandth, which is what
makes the middle arm readable. The `wy` arm printed
`GDN_WY: the two-phase recurrence ran (T=1024, 32 chunks, 32 heads)` in **both**
of its runs, so both readings are the port's and not a silent fallback's.

A second run at `--max-context 2048` gives the same ratio on twice the chunk:
7.612 against 6.521 ms, **14.3 %**, spreads 0.006 and 0.004. So the win is a
constant factor, not a fixed saving that would wash out at longer prompts - the
two-phase form is linear in T like the sequential one, just with a smaller
constant.

**The first version of this script measured the wrong thing and nearly hid it.**
It assumed GDNPROF profiles the prompt it sends, and normalised nothing. GDNPROF
actually fires once, inside `warmup()`, on the throw-away chunk of
`min(tmax, ctx)` rows - so with `--max-context 2048` it reports T = 2048
whatever the prompt is - and the script then compared that 6.521 ms against a
3.80 ms bar measured at 1024 rows, printing "FASTER BUT ABOVE THE BAR" for what
is in fact a clear pass. The fix: the T is parsed out of the engine's own line
(`(T=(\d+)`) and never assumed, the bar is scaled by `T / 1024` only after saying
so out loud, and a run whose arms disagree on T refuses to produce a verdict.
Both runs above were then measured with the corrected script.

## Status 2026-11-09: the first A/B run wedged the box's NVIDIA driver

`.backup/ab_gdn_wy.py` was launched on 2026-11-08 23:35 and produced only
`START base GDN_WY=False`. The `base` arm's engine then sat at 100 % CPU in
**kernel** time (`utime=2 stime=143301` ticks) with an RSS of 161 MB and never
mapped the model, so it never reached the point where the flag is read. **No
verdict was produced; nothing about the port's equivalence is known from a box
run.**

What the box shows now (verified, not inferred):

- `strata-q36` (pid 2672429) is `R`, spinning in the kernel, and **`SIGKILL` is
  queued but not delivered** (`SigPnd=0x100`, still `R` minutes later) — it is
  stuck inside the driver, not merely busy.
- Three `nvidia-smi` invocations are `D` in `uvm_va_space_destroy`, blocked on a
  mutex; `nvidia_uvm` reports 8 users, so `rmmod nvidia_uvm` fails.
- `dmesg` has **no `NVRM`/`Xid` line for this episode** — the last Xid is from
  uptime 486936, long before. So this is not the familiar Xid-31 MMU fault.
- The rest of the box is healthy: load 4.3, 20 GB free, both GPUs still report
  their model through `/proc/driver/nvidia/gpus/*/information`.

**Likely cause, and it is mine.** The previous A/B that ran cleanly on this box
(`.backup/ab_expert_plan.py`) never resets the cards. My new script calls
`nvidia-smi --gpu-reset` on **all four** indices before every arm — the three
helper P104s *and* the P100 at index 2, which is the display/modeset card
(`nvidia-modeset` is resident). The reset returns an error on that card, and the
driver was then found wedged. This is unconfirmed (there is no dmesg line to
date it), but the ordering and the blast pattern both fit, and the reset is not
needed: the old script's `helpers_free()` wait did the same job by waiting for
memory to be returned.

**Do not repeat this.** A recovery needs a reboot of the box, which is the
owner's decision, not something to take unilaterally. Until then:

- the A/B equivalence run is **not done**;
- the GDNPROF timing run is **not done**;
- nothing in this document has box evidence behind it.

The next script change, before anything else runs: drop `--gpu-reset` entirely
and go back to waiting on free memory per helper card, the way
`ab_expert_plan.py` does.

## The review that found the port could not have run

The box was rebooted (as the owner approved) at 01:05 on 2026-11-09 and the
driver came back clean: four cards, a fresh CUDA context, `cudaMalloc` fine. In
the same pass a review of the new code found **five blockers, four of which would
have produced wrong numbers rather than a crash**. All five are fixed; the fixes
are what the code now says.

| | what was wrong | what it would have done |
|---|---|---|
| B1 | `k_gdn_seq_wy` mapped 256 threads over a 32-token x 64-column slab that needs 512 | tokens 16..31 never had `E` corrected or an output written; value columns 32..63 never updated |
| B2 | both kernels used `C = 32` as the hidden state's row stride | every q/k/v read came from a different token's row - a wrong recurrence with no fault |
| B3 | the state was read and written as `[value][head][key]` | correct prefill from a zero state, then a corrupted decode: the transposition is invisible on zero |
| B4 | `float4` loads from a panel whose stride was `D + 1 = 129` | sticky misaligned-address fault on 3 of every 4 rows |
| B5 | 58 KiB and 97 KiB of dynamic shared memory | **could never run at all**: sm_60/sm_61 cap every block at 48 KiB, opt-in included |

B5 is the one worth remembering. The build succeeded and the binary carried the
flag, and the plan recorded the 48 KiB *static* ceiling as a problem already
solved by `extern __shared__` - but the dynamic opt-in ceiling on Pascal is the
same 48 KiB. The 64 KiB opt-in starts at Volta. Both kernels now declare their
panels `__shared__` and fit in **40.75 KiB** and **42.875 KiB**, with the video
memory traded for a few coalesced global reads of Q and K.

Two habits this round bought: the reviewer was given the file and the intent and
found five things I had read past twice, and every one of them is a defect whose
symptom is *a plausible number*, not a crash.

## What the box A/B has cost so far (four runs, no verdict yet)

Each of these was a real gate doing its job, not a script bug:

1. `--gpu-reset` wedged the driver (above).
2. `--expert-vram 2` + `NO_HOST_EXPERTS=1`: refused at load, because a primary
   slot can only be filled from the host arena. `--expert-vram 0.01`.
3. `NO_HOST_EXPERTS=1` alone: refused, because it wants one helper to hold
   **every** expert while this box's deployment splits them three ways by
   `expert % 3`. Dropped - it is not part of what this A/B varies.
4. The 20-id probe prompt made slot 0 emit EOS after 2 tokens
   (`BDONE 0 2 stop`), so the two slots produced 2 and 96 tokens. Replaced with a
   longer prompt, and the completeness gate changed from `n == NREQ * NTOK` to a
   per-slot floor - an EOS stop is a legitimate output, and demanding the full
   count would have refused a run that was fine.

## What this is not

- Not the UT transform. If the port lands near the estimate and more is needed,
  UT is a separate change to phase 1 only, and phase 2 does not move.
- No claim about decode: `T` there is a handful of rows, and tqLLM's own dispatch
  sends `n < 64` to a different kernel entirely (`qwen4exp.cu:1349, :1356`).
  The branch here is guarded by `chunk(T)`, so decode never takes it.

## Where the remaining time goes, and why the next step is UT

Phase 2 walks 32 chunks a half with 3 barriers each, on 4 blocks a head
(`128 / kWyVS`) - so 128 blocks over 32 heads, and the P100 has 56 SMs. The
chunk-to-chunk dependency is one `float`-wide state slab, which is why the
barriers cannot be removed by scheduling alone. What can still be removed is
**how many chunks there are**: the UT (Cholesky) transform lets the step be
taken with half the multiplication depth rather than a triangular solve, which
is the change that would let `kWyChunk` grow from 32 to 64 without the Gram work
growing as C^2 per token. That is phase 1 only. It is also what the old note
predicted, and it is now the *second* step rather than the first - the first one
was structural and is done.
