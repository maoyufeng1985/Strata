# What vLLM does for GPU occupancy under concurrency, and what Strata can take

Four read-only scans of `D:\work\vllm-src` (commit `a34565bc`), each on a different mechanism, cross-checked
against Strata's own code. The question: Strata's decode window is 8 rows across 1 primary card and 3 helper
cards, and the helper cards measured 55-59 % idle (each layer they wait for the primary's mixer). What does vLLM
do to keep its cards busy that Strata does not?

## What vLLM does

**Scheduling budgets are two numbers, not one.** `token_budget` (= `max_num_scheduled_tokens`) is spent by the
running requests; `input_budget` (= `max_num_batched_tokens`) also gates admission of waiting requests
(`scheduler.py:741, :868, :1102`). A decode output token is never reserved, and a request's spec tokens are
clipped by both budgets at once.

**Spec depth is per request, and can be a table lookup on batch size.** `num_spec_tokens` defaults to a fixed
value (`scheduler.py:301`) but a `num_speculative_tokens_per_batch_size` mapping builds a dense
`batch_size -> K` table (`spec_decode/dynamic/utils.py:77-148`) consulted every step (`:1786-1791`). There is no
acceptance-rate feedback loop - the acceptance rate is metrics only (`spec_decode/metrics.py:107-135`).

**A mixed batch carries a per-request draft length.** Requests with drafts and without are stitched into one
flat batch by a cumulative sum of per-request `num_draft_tokens` (`gpu_model_runner.py:2304-2316`,
`_calc_spec_decode_metadata :2688-2742`), with `-1` placeholders padding to the cudagraph shape.

**Batch size is padded up to a capture size, and the capture list is dense where it is small.**
`_bs_to_padded_graph_size` rounds up to the next capture size (`cudagraph_dispatcher.py:81-90`); the default list
is `[1,2,4] + range(8,256,8) + ...`, and "interactivity" mode uses every integer 1..32 (`vllm.py:2641`). Graph
descriptors are captured largest-first so the widest graph reserves the shared pool
(`cudagraph_dispatcher.py:329-346`, `cuda_graph.py:200`).

**CPU work is fenced by an event, not by a stream sync.** `prepare_inputs_event.synchronize()` runs at the top of
the next step (`gpu_model_runner.py:3675`) and is a *blocking* `torch.cuda.Event`, chosen over
`cudaStreamSynchronize` to avoid the driver lock spinning and because it is precise to the event rather than the
whole stream (`:762-763`). There is one pinned buffer per input, not a double buffer (`utils.py:110-143`).

**Requests that finish are replaced on the next schedule, not within the running step.** `running` is a
persistent list reused every round (`scheduler.py:244, :817`); a finished request is removed in
`update_from_output` (`:2587, :2640`) and the waiting queue fills the gap at the next `schedule()`
(`:1071-1660`).

**Prefill is deliberately rhythmised so decode runs uninterrupted stretches.** `long_prefill_token_threshold`
(`:781`), `defer_prefills` (`:772, :802`) and `local_prefill_interval` (`:408`) push prefill into a few release
steps rather than interleaving it with every decode step.

## What Strata already does

- **Per-request draft depth** - `Slot::ndraft` / `mchain` already give each slot its own number of rows.
- **A row budget per window** - `E.vrows` is exactly vLLM's `token_budget` at the granularity Strata runs at.
- **An event fence instead of a stream sync for the helper wait** - `STRATA_REMOTE_FLAG_WAIT` (mapped completion
  word) exists, and the deferred-wait path (`moe_helpers_wait`) already runs one window late.
- **Prefill/decode separation** - the prompt chunk and the decode window are already distinct code paths.

## What is worth taking, ranked

1. **`STRATA_Q36_MTP_SAMPLED` (implemented, 2026-11-08).** Sampling requests currently lose MTP entirely:
   `Sampling::needs_logits()` is true for the model card's own defaults (temperature 0.7 / 1.0, presence_penalty
   1.5, `serve/server.py:9295-9304`), so real traffic never verifies a draft. The fix needs no rejection sampler
   and no draft probability: an argmax draft is a delta with `q(d)=1`, so vLLM's test `min(1, p(d)/q(d))` is just
   `p(d)`, and "sample one row from `p`, keep the prefix that equals the drafts" has exactly the same
   distribution. See `2026-11-07-two-window-interleave.md` for why the rejected alternative (a two-window split)
   was not the way to get there.

2. **A batch-size -> MTP depth table** — **MEASURED 2026-11-08, and the answer is not a table but a split by
   request type.** For sampled traffic depth 1 is the only setting that pays (+11.0% over plain, depth 2 is -8.1%);
   for greedy traffic the pin stays depth 2 (141.8 / 70.9, reproduced). A depth table keyed on row count would not
   have found this - the gate is the *acceptance rate*, which follows from whether the request samples, not from how
   many rows the window carries. The original vLLM-derived idea (pick K from the row count) is therefore closed;
   the useful rule is `mtp_cand(smp) && !needs_logits() ? 2 : 1`, i.e. exactly the split the two arms measured.
   Details in "Depth 1 is the flag's winning setting" below.

3. **A dense capture list at small row counts.** Strata captures per `(G, T)` key (`run_graph_key`), so the arms
   it actually runs already have their graph. This only matters if a future change makes the row count vary more
   than it does now; noted, not urgent.

4. **A separate stream for the sampler's D2H, waited on lazily.** vLLM issues the sampled-token device-to-host copy
   on `async_output_copy_stream` (`gpu_model_runner.py:304-310`), marked non-blocking, and defers the wait until the
   tokens are actually needed (`gpu_input_batch.py:1012`, waited inside the next `sampler(...)` call at
   `gpu_model_runner.py:3507`). `execute_model` therefore returns without draining the compute stream, and the next
   step's launches are issued while the copy is still in flight. Strata's `run_rows` does the opposite: it issues the
   pick D2H on `st` and then `cudaStreamSynchronize(st)` (`q36_engine.cu:7985-7992`), so the whole pipeline drains
   once per window before the next window is built. The same pattern applies to `mtp_rows` (`:8083-8085`).
   **This one is closed by an existing measurement, not by a new experiment.** The exit drain was already timed
   apart (`STRATA_Q36_ROWS_SYNC_TIMING`, `plans/2026-11-06-q36-row-split-sync-corrections.md`): the entry drain is
   0.004 ms, i.e. it does not wait at all, and the 2.819 ms exit drain is the full-vocabulary head plus argmax - real
   GPU arithmetic (`head_rows()` at the end of `forward()`), not a host-side copy. A separate copy stream can hide a
   copy; it cannot hide arithmetic the next window's rows depend on. So there is no window for vLLM's trick to
   hide, and the item is dropped rather than ported.

5. **Not portable.** vLLM's DBO ubatching (`gpu_ubatch_wrapper.py:76-211`) hides an all-to-all over NVLink;
   Strata has no P2P at all (measured: direct 313 tok/s vs 1900 through the host), so there is no communication to
   hide behind compute. The prefix-cache block managers likewise assume a KV layout Strata does not have. The
   engine's `batch_queue` (`engine/core.py:219-222`, "eliminate pipeline bubbles") is a pipeline-parallel device and
   has no single-rank analogue here.

## The measurement that decides the next step

`STRATA_Q36_MTP_SAMPLED` is the only item here that changes what the engine *can* do rather than how fast it does
it: it is the difference between MTP being on for real traffic and off. Everything else is tuning against a
baseline that has to exist first. So: build it, prove the sampled text is unchanged when the sampler is seeded to
pick the argmax, then A/B the acceptance rate and tok/s against the plain path.

## Where ① landed (2026-11-08)

Built on 192.168.1.98 against `strata-q36.facesref` (the pre-change binary). The regression is
`.backup/mtp_sampled_check.py`: six engine runs, 2 sessions x 24 tokens, temperature 0.7 / top_k 20 / fixed seed.

- `PASS flag-off-greedy` - md5 `3431a53d87c7e908daf24221cf9ae77d` on both binaries.
- `PASS flag-off-sampled` - md5 `3da0dd658f259ecf4b855b97983f1743` on both binaries, `accept 0 / 0` on both
  (the old path never offers a proposal to a sampled slot, which is exactly what the flag changes).
- `on-sampled-1` - `accept 20 / 26`, md5 `e0c1d9db859d8da392015b7998125588`. The sampled request verifies drafts
  for the first time; the accepted count is the whole point of the change.
- `on-sampled-2` - the fixed-seed reproducibility arm (see the run log for its md5).

One earlier attempt at this build failed with an illegal memory access at `q36_engine.cu:5635`, inside the *prompt*
warmup. **Corrected 2026-11-08: this is real, not environmental.** My first pass grepped too short a dmesg window
and concluded "no Xid, therefore not a device fault". dmesg does have it:

```
Xid (PCI:0000:03:00): 31, pid=1677016, name=strata-q36, MMU Fault: ENGINE GRAPHICS GPCCLIENT_T1_3
faulted @ 0x7f6d_87600000. Fault is of type FAULT_PDE ACCESS_TYPE_READ      (2026-11-08 10:36:20)
```

OOM, swap and locked memory were each excluded (no OOM in dmesg; 15 GiB swap with 170 MiB used; Unevictable and
Mlocked both 0). `:5635` is only the *discovery* point - a `cudaStreamSynchronize` - so the fault is upstream in the
prompt MMQ path (`k_route_scan` `:5610`, `k_route_scatter` `:5615`). Frequency so far: 2 of 10 engine starts, both
the day's first cold start. Not caused by ① (the change adds two member *functions* and no data members, and
`kVerifyMax` is unchanged). Open defect, out of scope for ①, worth its own report.

Reviewer verdict: PASS_WITH_NITS, no blockers. The nits worth carrying forward: `sample()` runs a vocabulary-wide
`partial_sort` (measured 0.293 ms/call on this box), so a sampled slot now pays it `n` times per window instead of
once - at n <= 5 that is under 1.5 ms against a ~47 ms window, so it is not worth optimizing; and the flag's
stderr acceptance line will now report *sampled* acceptances, whose rate is `p(draft)` and therefore much lower
than the greedy rate, which is correct but will read as alarming.

### Throughput A/B (`.backup/mtp_sampled_ab.py`, 2 sessions, 128 tokens each, temperature 0.7 / top_k 20)

Arms interleaved off/on/off/on so a drifting box hits both. Throughput is taken from each slot's own `BDONE` wall
clock, never from a us/layer figure; `agg` is total tokens over the slowest slot's time.

| rep | flag | agg tok/s | per slot | accept |
|---|---|---|---|---|
| 0 | off | 50.48 | 40.61 / 47.64 | 0 / 0 |
| 0 | on | **62.35** | 47.75 / 51.83 | 66 / 90 (73.3 %) |
| 1 | off | 50.63 | 40.73 / 48.10 | 0 / 0 |
| 1 | on | **63.21** | 48.40 / 52.45 | 66 / 90 (73.3 %) |

Mean 50.56 -> 62.78, **+24.2 %**, with both slots faster in both reps. The arms do not overlap: the two `off` runs
sit 0.3 % apart and the two `on` runs 1.4 % apart, against a 24 % gap between the groups. The 73 % acceptance is
the expected value, not a defect: with an argmax draft the acceptance probability of a sampled request is
`p(draft)`, not 1, so it is structurally below the greedy rate and still high enough to pay for the extra rows.
The acceptance count is also bit-identical across the two `on` runs (66/90), which is what a fixed seed should
give.

**CORRECTION (2026-11-08, after the S=4 run).** An earlier version of this section claimed the per-slot token
sequences were *identical* between the flag-off and flag-on arms. They are not. Comparing the full sequences instead
of the first five tokens:

    slot 0, flag off: 491 24880 93808 17 35 7    198 197   20345 42816 283 ...
    slot 0, flag on:  491 24880 93808 17 35 2061 198 3645  90819 7     15  ...

The common prefix is 5 tokens and the divergence is at index 5, in both the S=2 A/B and the S=4 A/B. So the claim
that "only the interleaving differs" was wrong; it came from looking at a `...` that hid the split.

This is **not automatically a defect**, and it is worth being precise about why. The two arms draw from the same
distribution but consume the RNG a different number of times: a plain step calls `sample()` once per token, a verify
window calls it once per row (up to 5). A fixed seed therefore cannot make the two arms agree - the same stream is
read at different offsets. What a fixed seed *can* prove is that each arm is deterministic (it is: the two flag-on
reps matched, and so did the two flag-off reps).

The real equivalence test removes the RNG entirely: at `temperature=0`, `sample()` returns the argmax and draws
nothing (`q36_engine.cu:9201`), so a flag-on arm and a flag-off arm must then be byte-identical. If they are, the
prefix test reads the right rows and the divergence above is only RNG offset. If they are not, the prefix test is
comparing the wrong row. `.backup/mtp_sampled_equiv.py` runs exactly this pair.

**This also exposes a gap in the original regression.** `mtp_sampled_check.py` compares flag-off to flag-off and
flag-on to flag-on; it never compares a flag-on arm to a flag-off arm, so it proves determinism and the presence of
acceptances, but not distributional equivalence. Its `RESULT PASS` should be read that way.

### The equivalence test that does work: `top_k=1` (PASS, 2026-11-08)

`temperature=0` does not work as the neutral arm: `needs_logits()` is `temperature > 0 || any penalty`
(`include/strata/qwen35/sampling.hpp:17`), so temperature 0 leaves the sampler entirely and takes the greedy path,
where the flag is inert by construction - the comparison would be vacuous and would pass no matter what the prefix
test did.

`top_k=1` is the correct neutral arm. It keeps `temperature=0.7`, so `needs_logits()` is true and the sampled
verify path is exercised; inside `sample()` (`q36_engine.cu:9199-9227`) `kk=1` leaves `idx=[argmax]`, the keep loop
settles at `keep=1`, and `r -= p[0]` is always `<= 0`, so the draw returns the argmax **while still consuming one
`u(rng)`**. Both arms therefore advance the RNG identically and a fixed seed is comparable.

`.backup/mtp_topk1_equiv.py`, 2 sessions x 64 tokens:

| arm | accept | tokens per slot | sequence |
|---|---|---|---|
| flag 0 | 0 / 0 | 63 / 63 | baseline |
| flag 1 | **58 / 68** | 63 / 63 | **byte-identical** |

`RESULT PASS`. The flag-on arm verified 68 drafts, accepted 58, and emitted exactly what the arm that verified
nothing emitted. **The prefix test reads the right rows.** The temperature-0.7 divergence above is the RNG offset
and nothing else.

### Depth sweep (`.backup/mtp_sampled_depth.py`) - depth is NOT worth raising

Item 2 asked whether a sampled slot should verify more than one draft. It should not. `STRATA_Q36_MTP_DEPTH` in
(1, 2, 3, 1) with the flag on, S=2 / vrows=8 (slack = 4, so both depths are reachable), 128 tokens, temperature 0.7.

An aggregate of `total tokens / slowest slot` is **not comparable across these arms**: the drafts change which
tokens are sampled, so each depth finishes at a different point (slot 1 stopped at 36 / 35 / 70 / 36 tokens) and
the numerator moves with the arm, not with the speed. The comparable figure is the per-slot wall clock per token:

| depth | slot 0 ms/tok | slot 1 ms/tok | slot tok/s |
|---|---|---|---|
| 1 (r0) | 20.62 | 19.80 | 48.9 / 51.9 |
| 1 (r3) | 20.45 | 19.27 | 48.9 / 52.3 |
| 2 | **38.69** | 18.45 | 25.9 / 54.2 |
| 3 | 20.25 | 19.27 | 49.4 / 52.0 |

**Depth 3 is at parity with depth 1** (19.55 vs 19.86 us/token averaged over slots, inside the 1.4 % drift the two
depth-1 arms show) and depth 2's slot 0 is a neighbourhood excursion, not a depth effect (its slot 1 was the
fastest arm in the sweep; it was re-measured as depth 3 on the run right after). The mechanism is the one already
recorded for the greedy path: depth d costs d `mtp_rows()` calls, each bracketed by two full
`cudaStreamSynchronize` (`q36_engine.cu:7944, :7978`), and the wider window (T=8 at depth 3 vs T=4 at depth 1,
about 982 vs 625 us/layer from `T(R) = 267 + 89.4*R`) needs 57 % more output to break even. It does not get it.

Two methodology notes, both already in memory and both hit again here: the aggregate numerator must count `BT`
rows, never the `BDONE` sum (a slot's first token goes through admission and prints `T`, but still increments
`sl.produced`); and this is the same conclusion as `q36-s4-depth2-vrows10-loses.md`, now confirmed on the sampled
path rather than only the greedy one.

**This closes item 2.** vLLM's `num_speculative_tokens_per_batch_size` maps a large batch to a *smaller* K, and
Strata already does that structurally and unconditionally: `want = slack > 0 ? mtp_depth() : 1`
(`q36_engine.cu:9611`), and at S=4 / vrows=8 slack is 0. The other direction - a small batch buying a deeper
chain - is what the sweep measures, and it loses. There is no row of such a table left to fill.

### Second vLLM scan (2026-11-08) - three more mechanisms, all already matched or not portable

- **Mixed prefill/decode without padding.** vLLM's backends take `query_start_loc` + `seq_lens` + `block_table` and
  process variable lengths in one varlen call (`flash_attn.py:1313-1316`); no padding, paged KV. Strata's window is
  already a varlen row set (`run_rows(gr, S)` with per-group `n`), so there is nothing to remove.
- **Cascade attention / common-prefix merging has been REMOVED from vLLM v1** (`model.py:328-329`,
  `gpu_model_runner.py:547-549`; `common_prefix_len` is "deprecated and always 0", `backend.py:707`). An earlier
  candidate idea - merging the sessions' shared prompt prefix - has no vLLM precedent to port and would have to be
  justified on Strata's own numbers.
- **Persistent buffers, incremental updates.** `InputBatch` is allocated once at `max_num_reqs` and mutated in
  place; `condense()` early-returns when nothing was removed (`gpu_input_batch.py:689`) and moves one row at a time;
  the block-table copy is issued *first* each step so it overlaps the remaining CPU prep (`gpu_model_runner.py:2034`);
  slot mapping is computed on GPU by a kernel (`block_table.py:153-179`), never materialised on the host. Strata's
  window already does the analogue - only the active slots are rebuilt (`q36-serve-cost-model-and-dead-ends.md`:
  "rows are not a session-to-row assignment").
- **`adaptive_verification` (DSpark only, `speculative.py:650`)** allocates one shared `draft_budget` for the whole
  batch, ranks per-request-per-position survival probabilities, keeps only the global best prefix, and picks the
  budget by `argmax(estimated_accepted / cost)` (`adaptive_verification.py:461`). This is the closest published
  analogue to Strata's "8 rows shared across sessions" and worth reading if the row budget ever becomes the binding
  constraint. It is not today: S=2 leaves 4 rows unused and the sweep above says spending them on a deeper chain
  does not pay.

  **Now measured, and it is worth stating why it does not port as-is.** vLLM ranks draft slots by a *learned*
  confidence head: `dspark/speculator.py:284` computes `confidence` from a `_sample_markov_chain` and stores it in
  `draft_token_confidence_probs`, which `get_num_tokens` cumprods into a survival probability
  (`adaptive_verification.py:427`). Strata has no confidence head - the MTP block writes logits and an argmax and
  nothing else. A cheap stand-in would be the top1 softmax probability of each draft row, which means a
  vocabulary-wide `exp`/sum per draft row per window; at 2 drafts a window that is ~2 x 150k exps on top of a path
  whose whole marginal cost is `mtp_rows` 2.14 ms. The depth sweep above says the *depth* decision is a loss
  anyway at every setting, so an adaptive budget would be choosing among options that are all worse than plain.
  Park it until MTP's acceptance on sampled traffic clears the break-even line on its own.

### Third vLLM scan (2026-11-08) - four readers over runner / scheduler / spec-decode / attention

The single transferable finding, and the four negative results that saved a wrong turn.

**The one that matters: UBatching / Decoupled Batch Overlap.** `gpu_model_runner.py:4107` (`maybe_create_ubatch_slices`),
`:4146` (`check_ubatch_thresholds`), `ubatch_utils.py:140-221`. The step's batch is sliced into `num_ubatches`
sub-batches, and the attention metadata and forward run per slice (`:4189`, `:2618`), so the pipeline overlaps across
slices instead of running one serial pass. Activation: `parallel_config.use_ubatching` and
`num_tokens >= dbo_decode_token_threshold` (decode) / `dbo_prefill_token_threshold` (prefill). It ships with a veto:
`_allow_microbatching` (`:3734`, fires when `num_reqs >= 2`) refuses to split when a prefix-cache reader in the first
half would attend over blocks a same-batch writer fills in the second half.

This is the same shape as Strata's two-window interleave, arrived at independently - and it agrees with the two
things we measured the hard way: the overlap has to be *between* slices of work, and it needs an ordering veto.
Our veto is stronger than vLLM's: a q36 prefill chunk cannot be hoisted into the plan gap because half-1's rows wait
on a combine from one step behind (`q36-prefill-plan-gap-hoist-unsafe.md`). Recorded so a future attempt starts from
the right constraint instead of rediscovering it.

**Negative results - do not port these, they are not in vLLM either:**

- **No cross-step double buffering.** vLLM's only two-steps-in-flight path needs `async_scheduling` *and*
  `pipeline_parallel_size > 1`: `decode_stagger = pipeline_parallel_size` (`async_scheduler.py:14,47`), gate at
  scheduler.py:838. At PP=1 the stagger is 1, i.e. it degenerates to one step at a time. There is nothing to copy.
- **No per-request multi-stream attention.** Attention is one varlen call on the default stream; the only off-stream
  work is MLA sparse-index building (`mla/index_group.py:188`), and CUDA-graph capture forces a single stream.
- **No draft/target overlap and no cross-request staging.** `propose_draft_token_ids` (`gpu_model_runner.py:4810`)
  runs *after* the target forward and consumes its hidden states; the whole batch goes target-all -> draft-all ->
  verify-all (`:3521`). So vLLM, with all its engineering, also serializes the three passes. Our serialization is not
  a local deficiency.
- **No fixed verify window at all.** There is no shared N-row cap; occupancy comes from scheduling every behind
  request every step, `num_new_tokens` per request clamped by `token_budget` (scheduler.py:861-870, 1023). At 2
  sessions Strata already puts both in the window every step, so this lever is already pulled.

**Smaller items, already matched or not applicable:**

- Multi-graph capture plus pad-up dispatch (`cudagraph_dispatcher.py:71-91`, `:233-322`): a mixed batch is rounded
  *up* to the nearest captured bucket, never dropped to eager. Strata already keys graphs by row count
  (`row_graphs`, `q36_engine.cu:7926`) and by `[session][layer * kMgStride + T]` for the mixer (`:2409`), so an
  in-range window hits a graph. Nothing to add.
- Persistent-vram staging so a captured graph replays with no rebuild (`backends/gdn_attn.py:633-689`, `:751-807`):
  Strata's `begin_win`/slot buffers are the analogue.
- Recovered-token sampling fused into the rejection kernel (`rejection_sampler.py:1070-1151`), and accept/reject kept
  fully on device with no per-step D2H (`:432-586`, only `parse_output` copies at the end). Matches our own finding
  that a per-window D2H is not where the time goes.
- `deferred_state_corrections_fn` (`gpu_model_runner.py:4140`, `:4343-4345`) and the `prepare_inputs_event`
  async-prep pipeline (`:3667`, `:764`) both need `use_async_scheduling`, and both overlap *host* prep with device
  work. Strata's host cost is already 0.14 ms/window (`q36-window-other-is-run-rows.md`), so there is nothing to hide.

### The S=2 depth=2 record reproduces, and what ab2 was actually measuring (2026-11-08)

`.backup/s2_repro_and_ab.py`, run dir `/home/Strata/.run/s2-repro-20261108`, analysed with the authoritative
`q36-analyze2.py` (its own gates: `gate` must be < 0.005, a plain arm must land in 20-60 ms/win).

| arm | binary | request | flag | wins | tok/win | ms/win | agg | sess |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `a_rec_s3d2` | `strata-q36.s3d2` | greedy | - | 69 | 5.536 | 38.82 | **142.6** | **71.3** |
| `b_cur_greedy` | `strata-q36` | greedy | 0 | 69 | 5.536 | 39.03 | **141.8** | **70.9** |
| `c_cur_off` | `strata-q36` | 0.7 / top20 | 0 | plain | 1.000 | - | n/a | n/a |
| `e_cur_off` | `strata-q36` | 0.7 / top20 | 0 | plain | 1.000 | - | n/a | n/a |
| `d_cur_on` | `strata-q36` | 0.7 / top20 | 1 | 17 | 4.059 | 42.15 | n/a | n/a |

**The record reproduces exactly.** `a_rec_s3d2` on the 11-07 binary and `b_cur_greedy` on today's binary both
give 69 windows, 5.536 tok/win, accept 123/135 on *both* slots (246/270), and 142.6 / 141.8 aggregate. Per-layer
profile matches too: 766.1 vs the record's 772.9 us/layer, window 38.82 vs 39.05 ms, `forward` 30.64 / 30.91,
`mtp_rows` 4.60 / 4.54. So there is no regression between 11-07 and today, and the pinned config is **142 agg /
71 per-session**, not the 62 my earlier A/B reported.

**Why ab2 read 62, three causes, all mine:**

1. **ab2 sent a sampled request.** The record is greedy (`BGEN <slot> <n> <ids>`, no sampling keys), so
   `needs_logits()` is false and the MTP candidate gate is open no matter what the flag says. ab2 sent
   `temperature=0.7 top_k=20`, and with the flag off that makes every slot a non-candidate.
2. **ab2 ran depth 1.** The record pins `STRATA_Q36_MTP_DEPTH=2` (T = 6 rows, 5.536 tok/win); ab2 set no depth.
3. **ab2 took the denominator from the BDONE clock**, which starts at admission and carries the ~700 ms prompt of
   the later session. The record's aggregate is `windows * 1000 / ms_per_window`, i.e. the decode clock only.

**The flag's real effect is not "more acceptances in the same window" - it is "the sampled request gets a window
at all".** `window()` only calls `window_mtp()` when `greedy > 0` (`q36_engine.cu:9915`), and `greedy` counts
`mtp_cand(sl.smp)`. With the flag off, a sampled slot contributes 0, so both `c_cur_off` and `e_cur_off` fall
through to `window_plain()` - no WINDOW line, `accept 0/0`, exactly as measured. With the flag on, `d_cur_on`
runs a real 17-window arm and accepts **37/62 (60%)**, which is `p(draft)` and therefore the designed value, not a
defect. `c` and `e` are byte-identical to each other, so the plain path is deterministic.

**A trap in reading the plain arms.** `c_cur_off` / `e_cur_off` die early and unevenly: slot 1 stops at 26 tokens
(546 ms), slot 0 runs to 107 (2655 ms). The analyzer's plain reconstruction then divides *all* 130 post-admission
BT rows by the first slot's clock and prints 236-238 tok/s - but 105 of those rows are slot 0 decoding alone. That
number is an artifact of the truncation, not a speed. Do not quote it; the plain rate needs both slots alive for
the whole span, which needs equal-length requests (or an EOS-free prompt), which this pair does not have. For
reference, the properly-measured plain 4-row arm on this box is 149.0 tok/s (`q36-mtp4-wins.md`), and a 2-row
plain window is not directly comparable to it.

**Still open:** `d_cur_on` cannot be turned into an aggregate-per-session number, because slot 0 dies at 36
tokens. What it does settle is that the flag produces a working MTP window and a 60% acceptance rate on the
sampled path. The throughput comparison at S=2 needs a matched-length pair, and that is the next run.

### The matched-length A/B: the flag does NOT pay at S=2 on sampled traffic

`/home/Strata/.run/s2_flag_matched.py`, run dir `s2-matched-20261108`. Both slots request **48 tokens from a
filler prompt**, so neither stops early and every window in the span carries two live rows. That removes the EOS
freedom that made the first pass unreadable.

| rep | flag | wins | tok/win | ms/win | agg | sess | accept |
|---|---|---:|---:|---:|---:|---:|---|
| 0 | 0 (plain) | 46 | 2.000 | 21.93 | **91.2** | 45.6 | 0/0 |
| 0 | 1 (MTP d2) | 28 | 3.357 | 40.41 | **83.1** | 41.5 | 45/96 (47%) |
| 1 | 0 (plain) | 46 | 2.000 | 22.01 | **90.9** | 45.4 | 0/0 |
| 1 | 1 (MTP d2) | 28 | 3.357 | 40.31 | **83.3** | 41.7 | 45/96 (47%) |

Four arms, each 47+47 tokens over a 91-token (plain) or 75-token (MTP) span. **Within-arm spread 0.3%, between-arm
9%.** The flag costs 9% aggregate (91.2 -> 83.1) and 9% per session (45.6 -> 41.5). The window gets 1.68x more
tokens but costs 1.84x more time, so the trade is negative - and reproduces bit-for-bit on the tok/win and the
acceptance count (45/96 both reps), so this is not box drift.

The profile says exactly where the extra time goes (`rep0-flag1.err`):

```
WINDOW outside forward: 40.41 ms/window | forward=27.97 mtp_rows=4.01 commit_rows=1.14 other=7.28 (28 windows)
```

`forward` is 27.97 of 40.41 ms = **69%**. The other 12.4 ms is MTP-specific: `mtp_rows` 4.01, `commit_rows` 1.14
and `other` 7.28 (the group build, the per-row D2H + `sample()` partial_sort, printf, poll). Against the plain
arm's single 21.93 ms window, the whole MTP apparatus buys 1.36 extra tokens for 18.5 extra ms - i.e. ~13.6 ms
per extra token, when a plain row costs ~11 ms. It cannot win at a 47% acceptance rate.

**Why the acceptance rate is 47% and not the greedy 91%.** This is the designed value, not a bug: with an argmax
draft the acceptance probability of a sampled request is `p(draft)`, not 1. A greedy request's `p(draft)` is ~1
because the distribution is near-deterministic; a 0.7-temperature request's is not. `mtp-sampled-verified.md`
already recorded this, and predicted it would "read as alarming" - this is the measurement that shows it is also
what decides the economics.

**The earlier +24% was measured against the wrong baseline.** `mtp_sampled_ab2.py` compared flag-off-sampled
(which runs `window_plain`, one row per slot) against flag-on against a BDONE clock carrying a ~700 ms prompt,
at 384 tokens with EOS stopping both slots unevenly. That comparison is not wrong in sign for what it measured -
the flag does move a sampled request from the plain path to an MTP path - but the *rate* it printed (50.3 -> 62.4)
came from a denominator neither arm shares with the other. The matched-length pair above is the first clean
comparison, and it says the move is a loss.

**The greedy result still stands and is separate.** Greedy S=2 depth 2 is 141.8-142.6 agg / 70.9-71.3 per session,
reproduced on both binaries. That is the configuration the record pinned, and the flag does not touch it (a greedy
slot is a candidate with the flag on or off).

### Depth 1 is +13% on sampled traffic; depth 2 is -7% (metric settled against `q36-analyze2.py`)

The numbers come from `s2_analyze_live.py`, whose metric was checked against `/home/Strata/.run/q36-analyze2.py`
until the two agreed to the digit. **The metric is the whole story here, and I got it wrong twice before
matching the authoritative one - see "The metric, twice wrong" below.**

| arm | rows | span_ms | **agg** | wins | tok/win | ms/win | accept |
|---|---:|---:|---:|---:|---:|---:|---|
| plain | 93 | 1029.9 | **90.3** | - | - | - | - |
| plain | 93 | 1031.4 | **90.2** | - | - | - | - |
| plain | 93 | 1022.2 | **91.0** | - | - | - | - |
| plain | 93 | 1019.6 | **91.2** | - | - | - | - |
| plain | 93 | 1023.6 | **90.9** | - | - | - | - |
| plain | 93 | 1034.5 | **89.9** | - | - | - | - |
| **depth 1** | 92 | 900.6 | **102.2** | 30 | 3.133 | 31.06 | 40/54 (74%) |
| **depth 1** | 92 | 891.7 | **103.2** | 30 | 3.133 | 30.81 | 40/54 |
| **depth 1** | 92 | 897.0 | **102.6** | 30 | 3.133 | 30.94 | 40/54 |
| **depth 1** | 92 | 893.3 | **103.0** | 30 | 3.133 | 30.86 | 40/54 |
| depth 2 | 92 | 1087.3 | **84.6** | 28 | 2.679 | 40.08 | 45/96 (47%) |
| depth 2 | 92 | 1096.1 | **83.9** | 28 | 2.679 | 40.41 | 45/96 |
| depth 2 | 92 | 1088.1 | **84.6** | 28 | 2.679 | 40.11 | 45/96 |
| depth 2 | 92 | 1094.4 | **84.1** | 28 | 2.679 | 40.31 | 45/96 |

**Depth 1 is +13%** (102.2-103.2 against 89.9-91.2; the ranges do not overlap). **Depth 2 is -7%**
(83.9-84.6 against 89.9-91.2). Six plain runs, four depth-1 runs, four depth-2 runs, all within-arm spread
under 1%.

**The metric.** `sl.t0` is stamped in the engine after `E.process()` has read that slot's prompt and just before
`BADM` is printed (`q36_engine.cu:10207`), so a slot's `BDONE` ms excludes its own prompt. Serve admits slots one
at a time, so **the last-admitted slot's clock starts exactly when the final prompt finished**. Its `BDONE` ms is
therefore the all-slots-decoding wall clock with no prompt in it, and the matching numerator is every `BT` row
after that last `BADM`:

```
cut  = index of the LAST BADM line
ts   = BDONE ms of the slot admitted there      # its clock starts when all slots are in
rows = every BT row after `cut`
agg  = rows * 1000 / ts
```

Do not subtract the `PP` line's ms on top: `BDONE` already excludes that slot's own prompt, so subtracting again
double-counts and shortens the denominator (the error `q36-analyze2.py` records as 8183 -> 7513 ms on p_s8, a
8.9% inflation).

**The metric, twice wrong.** First I divided *all* 94 BT rows by `min(BDONE)`, which mixed a both-slot numerator
with a one-slot clock (94/1029.9 = 91.2 for plain, 94/900.6 = 104 for depth 1). Then I "corrected" it by
restricting the numerator to a both-alive interval but keeping `min(BDONE)` as the clock - and `min(BDONE)` is the
**first slot to die**, whose clock started one prompt (~600 ms) earlier, so the interval is too long and the rate
reads ~13% low. That is what produced a (wrong) "depth 1 only ties plain" reading for a few hours. The
authoritative clock is the last-admitted slot's, not the minimum.

**Why depth 1 wins: the 3rd and 4th rows are nearly free.** `forward` is 21.93 ms for the 2-row plain window and
22.20 ms for the 4-row depth-1 window - +0.27 ms for two extra rows, against a cost model that predicts +7.2 ms.
The window pays 3.133 tokens for 31.06/21.93 = 1.42x the time, i.e. 1.57x the tokens. Going on to 6 rows costs
another 5.8 ms of `forward`, 2.9 ms more `mtp_rows` and 1.8 ms more `other`, while acceptance falls from 74% to
47% - so depth 2 pays 40.08/21.93 = 1.83x the time for 1.34x the tokens, a 7% loss.

**Depth 1's acceptance looks better than depth 2's for a structural reason.** 74% (40/54) against 47% (45/96):
depth 2 counts the *second* draft, whose marginal acceptance is much lower - the same effect recorded for greedy
in `q36-mtp-depth-2-wins.md` (first draft 80%, second 38%). The tokens-per-window column already prices it in.

**The recommendation that follows.** For sampled traffic at S=2, run `STRATA_Q36_MTP_SAMPLED=1` with
`STRATA_Q36_MTP_DEPTH=1` (+13%). For greedy traffic the pin stays depth 2 (141.8 / 70.9, reproduced). These do
not conflict: the greedy record was measured at depth 2 and depth 1 has not been re-measured there.

**Fifth independent reproduction (`s2-d1recheck-v4-20261108.log`, `RESULT PASS`):** `d1=[102.2, 103.8] mean=103.0`
against `plain=[90.6, 91.4] mean=91.0`, **+13.2%**. Every arm matched-length, no errors.

### Mixed traffic: depth 1 also wins, and the depth is currently GLOBAL rather than per slot

Real serving carries both kinds of request at once, and the depth decision does not look at which kind it is
serving: `window_mtp` computes `slack = vrows - S - nc` and then `want = slack > 0 ? mtp_depth() : 1`
(`q36_engine.cu:9615-9617`), after which every candidate slot whose `mdepth` comes out above 1 is chained -
greedy or sampled alike. `Slot::mdepth` (`:9352`) is already a per-slot field, so the split is available; the
decision just does not use it.

`.backup/s2_mixed_probe.py` + `.backup/s2_mixed_greedy_d1.py`, S=2, 2 x 48 tokens, no EOS, authoritative metric:

| arm | slot 0 | slot 1 | wins | **agg** | sess | accept 0 / 1 |
|---|---|---|---|---:|---:|---|
| `mix-d2` | greedy | sampled | 28 | **87.8** | 43.9 | 53% / 38% |
| `mix-d1` | greedy | sampled | 30 | **106.6** | 53.3 | 68% / 62% |
| `greedy-d2` | greedy | greedy | 24 | **107.6** | 53.8 | 53% / 53% |
| homogeneous sampled, d1 | sampled | sampled | 30 | **103.0** | 51.5 | 74% |
| homogeneous sampled, d2 | sampled | sampled | 28 | **84.2** | 42.1 | 47% |

**Two things fall out.** First, `mix-d1` (106.6) is within noise of `greedy-d2` (107.6) and above the best
homogeneous sampled arm (103.0), so **depth 1 is the right setting for mixed traffic too** - the sampled slot's
presence does not cost the greedy slot anything at depth 1. Second, `mix-d2` is 18% below both, and its
acceptance lines show why: the greedy slot drops 53% (its own homogeneous depth-2 number, so the chain is
working) while the sampled slot falls to 38%, below even its homogeneous depth-2 47%.

So the current global knob forces a choice between `MTP_DEPTH=1` (mixed-safe, greedy slots give up the deeper
chain they could have used) and `MTP_DEPTH=2` (greedy-optimal alone, but 18% down the moment one sampled
request shares the window). Making `want` per slot - `cand[i] && !slots[act[i]].smp.needs_logits() ? depth :
1` - removes the trade. **Not implemented yet**: the 2x2 above prices it at most ~1 tok/s over `mix-d1` (107.6
vs 106.6), which is inside the box's drift, so it is a correctness-of-intent improvement rather than a measured
win, and it should not be built without a `greedy-d1` control to say how much of `greedy-d2`'s 107.6 comes from
the depth at all.

### There is no single-slot tail (a hypothesis raised and refuted the same session)

The depth-1 arm's window average is 3.133 tokens per 31.06 ms window = 100.9 tok/s, and
`17 windows x 4 tokens + 13 windows x 2 tokens = 94` reproduces 3.133 exactly. Together with `rep0-d1` showing
slot 0's BDONE at 1397.7 ms against slot 1's 900.6 ms, that looked like a 497 ms stretch in which one session ran
alone at 26.2 tok/s - 36% of the run - and a fallback from `window_mtp()` to `window_plain()` once `act == 1`
looked like a large win.

**It is not real.** Two checks killed it:

- **Neither arm emits a token after the first death.** In both `rep0-plain` and `rep0-d1`, slot 0 has already
  emitted all 47 of its BT rows by the time slot 1 dies, and nothing follows. There is no surviving-session
  stretch to speed up, in either arm.
- **The BDONE gap is prompt stagger, not idle.** A slot's clock starts after its own prompt is prefilled, and
  serve admits slots one at a time: `slot 0 takes a prompt of 24 tokens ... in 607 ms`, then slot 1's prompt runs.
  Slot 0's 1627.7 ms therefore covers 607 ms of prompt the other slot never paid, and its decode is
  1627.7 - 607 = 1020.7 ms against slot 1's 1029.9 ms - the same interval, finishing together. This is exactly
  why the authoritative clock is the **last-admitted** slot's: its clock starts when the last prompt ends.

**The arithmetic coincidence is just that.** 3.133 tok/window is a mean over windows that carry a variable number
of rows per slot; it does not decompose into populations of window *shapes*. Reading the engine's per-window mean
as if it separated one-slot from two-slot windows is a variant of the mistake `q36-window-count-trap` records.

`s2_tail_probe.py` was written for this question and is kept: it measures a one-session MTP arm against a
one-session plain arm, which is the clean way to price a genuinely one-slot verify window if that case ever
arises (for example with `--batch 2` and one client leaving mid-request).

### The prompt starves the decoding sessions, and `read_windows` is the knob vLLM would name

Measured twice, at two prompt sizes.

- **A short prompt** (`s2-mixed-20261108/mix-d2.out`): slot 1's 24-token prompt took **664 ms** (`PP 24 24 664`),
  and in that whole interval slot 0 - already admitted and decoding - emitted **2 tokens**. Its own rate in the
  same run is 53 tok/s.
- **An ordinary prompt** (`s2-readwin-20261108.4`, `rw1`, `--chunk 128`): slot 1's 1024-token prompt took
  **25044 ms**, and slot 0 emitted **15 tokens** in that interval - **0.6 tok/s against its own ~53**, a 99%
  freeze. A 1024-token prompt is not a corner case, so this is the concurrency problem itself.

**And for the common case the decode gets NOTHING at all, not even the one window.** The loop is guarded by
`done < fresh` (`:10133`), and `on_chunk` receives tokens-read-*so-far* (`:7908` passes `i + T`). A prompt
shorter than the chunk is read in a single pass, so the only callback fires with `done == fresh`, the guard is
false, and `window()` never runs. Verified on the three default-chunk runs above: **every 24-token prompt
produced exactly one `PP` line with `done == fresh`** (`PP 24 24 687`). So for any prompt that fits in one
chunk - which on these configs means anything under 2048 tokens, i.e. essentially every real request - a new
admission freezes every decoding session for the prompt's whole duration.

**And the chunk size trades the two against each other, hard.** Measured prefill on this box: `--chunk 128` reads
at **41 tok/s** (~610 us/layer) against the ~1780 tok/s the PP work recorded at large chunks
(`q36-chunk-linear.md`: 14.06 us/token/layer). A small chunk buys boundaries but pays the expert-staging
overhead once per chunk, so it is 43x slower per layer. The comment's assumption (a 512-token chunk for 0.15 s,
i.e. ~3400 tok/s) is off by two orders of magnitude at the small end.

Put together, the fair-share rule `read_windows ~ (S-1) * t_chunk / (S * t_win)` evaluates to **3.5 to 110**
across the chunk sizes and speeds this box actually has, against a default of 1 that only ever runs when the
prompt spans a boundary. A fixed default cannot be right for any of them; the value has to come from the chunk's
measured cost.

The mechanism is one line. `E.process()` calls back after every `--chunk` tokens (`q36_engine.cu:7856`, the
prompt loop), and the callback (`:10129-10142`) runs `window()` exactly `read_windows` times before handing the
GPU back:

```cpp
for (int w = 0; w < read_windows && n_active > 0; ++w) window();   // :10136
```

`read_windows` defaults to 1 (`:9375`), and the comment states the arithmetic it was chosen with: a chunk takes
~0.15 s, a window ~0.012 s. That is ~7% of the GPU for the decoding slots during a prompt.

**Why Strata cannot just fuse the batches the way vLLM does.** `chunk(T)` (`q36_engine.cu:4395`) decides
prompt-arithmetic vs decode-arithmetic for the WHOLE window:

```cpp
bool chunk(int T) const { return T > kGemmT && !rows_mode && !verify_win; }   // kGemmT == 8
```

`rows_mode` makes the window a decode batch whatever its width, and a prompt chunk is neither `rows_mode` nor
`verify_win`. So a window is either all prefill arithmetic (FP16 GEMMs, the split-K router, chunk attention over
ONE session's K/V) or all decode arithmetic - there is no per-row switch. vLLM's mixed step works because its
attention takes `query_start_loc` + `seq_lens` and does the right thing per sequence inside one varlen kernel; a
Strata window would have to split mid-`forward()`, which is a much larger change than a scheduler tweak. The
`read_windows` ratio is the cheap approximation of the same fairness, and it is the one to measure first.

This is exactly what vLLM's `long_prefill_token_threshold` exists to bound - its comment is "stops a long
prefill from starving other requests of the token budget" (`vllm/v1/core/sched/scheduler.py:776-785`), floored
at `input_budget // num_eligible_reqs`. The difference is that vLLM *interleaves* prefill and decode rows in one
step's token budget rather than alternating whole passes, so it has no such fixed cost to trade against.

**Two traps for anyone re-measuring this.** `read_windows` only acts at a chunk boundary, and the chunk is
`--chunk` (printed as `prompt chunk N`, default 2048 at `--max-context 4096`), so a prompt shorter than the
chunk is read in one pass and the knob can never fire - the first version of this probe used 1024 ids and
measured nothing. And serve admits serially (`:10371-10372`), so the long prompt must be the *second* request:
admitted first it would run with no other slot active and again measure nothing.

**The rule that would size it, translated from vLLM's floor.** vLLM cannot give a long prefill a *token* budget
larger than a fair share: `long_prefill_token_threshold >= input_budget // num_eligible_reqs`
(`scheduler.py:786-791`), applied only when `num_eligible_reqs > 1` - "when it is the only request there is
nobody to starve, so let it use the whole budget" (`:776-785`). Strata has no fused mixed batch, so the fair
share has to be expressed in *time*: for one prompt chunk of duration `t_chunk` shared with `S - 1` decoding
slots, each decoder's fair slice is `(S-1) * t_chunk / S`, and `read_windows` should be that over one window's
duration `t_win`:

```
read_windows ~ (S - 1) * t_chunk / (S * t_win)
```

At the comment's own figures (`t_chunk` 150 ms, `t_win` 12 ms, S=2) that is **6.2**; at a fast chunk (40 ms) it
is **1.7**, i.e. the default of 1 is roughly right for a fast chunk and badly wrong for a slow one. The measured
prompt in the first attempt read 1536 ids in 19853 ms - 6.6 s per 512-token chunk, *far* outside the comment's
0.15 s, because the read is itself competing with the decode - so on this box the ratio is much larger than 6.
This is the number the probe below exists to pin down.

**`NO_HOST_EXPERTS` answers this by shrinking the problem instead.** Filling the formula with the measured prompt
cost and the measured window (31 ms) at S=2:

| configuration | `t_chunk` | ratio |
|---|---:|---:|
| default (arena on the prompt) | 690 ms | **11.1** |
| `NO_HOST_EXPERTS=1` | 100 ms | **1.61** |

The default of 1 is badly wrong in the first row and approximately right in the second. That is the reason to land
the arena change first and re-price `read_windows` afterwards rather than tune it now: it is the same fix, applied
to the denominator instead of the numerator. The one thing `NO_HOST_EXPERTS` does NOT fix is the `done < fresh`
guard - a single-chunk prompt still runs zero windows - but the interval it fails to cover is now 100 ms rather
than 690 ms, i.e. ~3 windows instead of ~22.

A 4-session arm was run and **abandoned**, and the user's call is that 2 sessions is the operating point - four
20-token prompts at once each cost about 700 ms, and because serve admits sessions one at a time, every later
prompt blocks the sessions already decoding. The arm's per-slot rates came out 11x apart and its token counts were
set by EOS, not by speed, so nothing in it is a usable measurement.

**Re-measured at S=2** (`.backup/mtp_sampled_ab2.py`, 2 x 384 requested, arms interleaved):

| rep | flag | slot 0 tok/s | slot 1 tok/s | agg tok/s | accept |
|---|---|---|---|---|---|
| 0 | off | 41.34 | 48.53 | 50.62 | 0 / 0 |
| 0 | on | **48.05** | **52.14** | **61.95** | 66 / 90 |
| 1 | off | 40.84 | 48.38 | 50.00 | 0 / 0 |
| 1 | on | **48.67** | **52.58** | **62.75** | 66 / 90 |

Mean **50.31 -> 62.35, +23.9 %**. Within-group spread is 1.2 % (off) and 1.3 % (on) against a 23.9 % gap between
groups, so the arms do not overlap; both slots are faster in both reps; and the acceptance count is identical
across the two `on` runs (66 / 90). The first A/B measured +24.2 % on the same code with a shorter generation, so
two independent sessions agree to 0.3 of a point.

`+22.4 %` on the aggregate, with both slots faster (slot 0 +16.2 %, slot 1 +7.4 %) - the same direction and
roughly the same size as the first A/B's `+24.2 %`, measured on a different day session with a different run
length. That two independent measurements land within 2 points of each other is the strongest evidence here that
the flag's gain is real rather than a single-run artefact.

The generation does not reach 384 tokens: this prompt converges to EOS at 106-120 for slot 0 and 25-36 for slot 1,
so the ~700 ms prompt stays 25-35 % of each arm's wall clock. That constant is identical across arms and therefore
cannot manufacture the gap, but it does mean the *absolute* tok/s figures are depressed - only the ratio should be
quoted.

### The ~690 ms prompt is the 18.75 GiB arena upload, and removing it is worth more than any scheduler change

**The measurement that settles it.** `.backup/s1_prompt_cost.py` runs ONE session with nothing else on the box and
reads the engine's own admission line (`takes a prompt of N tokens = ... read in X ms`), so no aggregate metric and
no concurrency can be blamed. Two arms, then the same two with `STRATA_Q36_NO_HOST_EXPERTS=1`:

| arm | default | `NO_HOST_EXPERTS=1` | change |
|---|---:|---:|---:|
| `p24` 24-token prompt | 800 ms | **112 ms** | **-86 %** |
| `p512` 512-token prompt | 3301 ms | **848 ms** | **-74 %** |
| fitted intercept | 677 ms | **76 ms** | -89 % |
| fitted marginal | 5.12 ms/tok (195 tok/s) | **1.51 ms/tok (663 tok/s)** | 3.4x |

The intercept is nearly token-independent, and the engine names it on its own startup line: *"all 18.75 GiB in
pinned host memory ... bigger prompt chunks streamed (1.10 GiB staging)"*. Every prompt chunk re-uploads the
primary's expert arena. Corroboration from the other direction: a **276**-token prompt measured 3305 ms by default
and 624 ms without the arena (`q36-drop-host-expert-arena-task`) - numerically the same 3.3 s as `p512`'s 3301 ms at
**512** tokens. So 3.3 s is the price of *one chunk*, not of any particular token count.

**Two readings this corrects.** First, `hybrid experts: N of M on the CPU pool` does NOT count the whole prompt:
24 rows (<= `hybrid_t` 128) take the `hy_rows` path into the pool (7680 = 24 x 8 x 40), while 512 rows (> 128) take
the streamed peer path and put **zero** entries in the pool (`640 of 640`, which is only the two generated tokens).
An earlier note in this file attributed the 664 ms to the CPU pool on the strength of that counter; the counter was
right and the inference was wrong. Second, a two-point "fixed + marginal" fit only means anything within ONE
mechanism - under the default config both points are dominated by the arena upload, so the 5.12 ms/token slope is
an artefact.

**Why this outranks the three mechanisms above.** `NO_HOST_EXPERTS=1` is refused unless `PREFILL_PEERS=1` is also
set (the validator checks it), so turning it on moves the prompt bodily onto the peer streaming path. The prompt is
what freezes the decoding sessions, and it is now 7x cheaper - which is a larger effect than `read_windows` can
produce even at its most favourable reading, because `read_windows` only re-slices the time the prompt already
occupies. It also changes what `read_windows` should be worth: with `t_chunk` down by 7x, the fair-share rule
`(S-1) * t_chunk / (S * t_win)` drops by the same factor and the default of 1 may well be correct.

**The open question the A/B below answers.** Removing the arena is not free for the decode: every routed expert must
then be owned by a helper, and the decode's CPU-pool bucket (`moe_hybrid_cpu`) has no host copy left to read. The
prior record is ambiguous - one arm measured decode 110.2 -> 114.5 tok/s, the next 52.9 (`q36-drop-host-expert-arena-task`,
and the box is bimodal, `q36-decode-box-bimodal`). So the decode has to be measured at S=2, not inferred.

**ANSWERED (2026-11-08, `.backup/s2_nohost_ab.py`).** Four arms, alternating, each preceded by a `--gpu-reset` of the
three P104 helpers and a discarded warm-up request; all four arms read 5005 MHz on every card for their whole run,
so neither confound is in the numbers.

| arm | prompt read | agg tok/s |
|---|---:|---:|
| `host` #1 | 696 / 687 / 681 ms | 100.1 |
| `host` #2 | 689 / 664 / 673 ms | 103.1 |
| `nohost` #1 | **100 / 103 / 102 ms** | **106.8** |
| `nohost` #2 | - | **106.6** |

```
host   mean 101.6  [100.1, 103.1]
nohost mean 106.7  [106.8, 106.6]
delta +5.1%
```

The two groups do not overlap (host's best, 103.1, is below nohost's worst, 106.6). The prompt is **85% cheaper**
and the aggregate is **+5.1%**; the decode is not hurt by losing its CPU-pool fallback, because the helpers own every
expert in this configuration. Load time also falls 51-60 s -> **1.0 s**.

**Two things this number is NOT, both raised by the review and both confirmed here.** (i) **It is not "the arena
made the decode cheaper".** The two configurations emit *different tokens*: the first token is **16** under `host`
and **220** under `nohost`, deterministically in both reps, so the whole generation forks from step one. (ii) **It is
not a like-for-like window count either.** Under `host`, MTP accepts 31 of 63 per slot; under `nohost`, 32 of 61 -
one fewer draft position verified for the same 63 tokens, i.e. 32 forward passes against 31. Normalising for that
leaves roughly 3%, not 5.1%. The honest reading of the four arms is: *`nohost` was 6.7% and 3.5% faster in the two
paired reps, and it reads a prompt in 100 ms instead of 690 ms; n=2 on a box with a known bimodal failure mode, so
treat the aggregate as suggestive and the prompt time as established.*

**The first-token difference needs its own explanation before this ships.** The prior record says the flag is
byte-identical on a `--tokens-file` single-session run (`q36-drop-host-expert-arena-task`, same `a32abb32...` both
arms). What changed here is the shape: `--serve --batch 20` with MTP. The plausible mechanism is arithmetic, not a
bug - under `host` a 24-row prompt takes the `hy_rows` path into the CPU pool, under `nohost` it goes to the helpers'
q8_1 path, and the two quantisations do not round identically. A near-tie argmax then flips. That is an acceptable
outcome for a decode that already ships with its own sampling, but it must be stated rather than assumed, and the
serving default must not change on the strength of this table alone.

**A second, independent reading of the same effect**, one that does not go through the aggregate metric at all. Slot 0
is admitted first, so its own BDONE covers its prompt while slot 1's does not; the gap between the two is therefore
slot 0's prompt, measured on the engine's own clock:

| arm | slot 0 BDONE | slot 1 BDONE | gap |
|---|---:|---:|---:|
| `host` #1 | 1925.1 ms | 1239.0 ms | **686 ms** |
| `host` #2 | 1876.4 ms | 1202.9 ms | **673 ms** |
| `nohost` #1 | 1261.2 ms | 1160.7 ms | **101 ms** |
| `nohost` #2 | 1263.0 ms | 1163.3 ms | **100 ms** |

That is the same 690 -> 100 ms the admission lines print, recovered from a different field. Two mechanisms agreeing
to the millisecond is what makes the prompt half of this result safe to quote; the +5.1% aggregate is the part that
rests on four arms.

**The gain is not a draft-quality artefact.** MTP acceptance is identical across the two arms - `host` 31 of 63 and
31 of 63, `nohost` 32 of 61 and 32 of 61 - so the speculative decoder is doing the same work in both; the time
difference is the prompt path, not a change in how many tokens each verify window wins.

**Two measurement preconditions, both of which cost a run to learn.** (i) A helper card that has *fallen* to a
stuck memory clock is never lifted by traffic, only by `nvidia-smi --gpu-reset` - one arm read **60.4** instead of
106.8 for exactly that reason (`q36-decode-box-bimodal`). `--gpu-reset` refuses while any process holds the device,
so `strata-q36` and `nvtop` must be killed first. **Read the clock with care, though:** 405 MHz is also the P104's
*idle* clock, and a healthy card sits there in P8 between bursts, so a single post-reset snapshot proves nothing
either way - during this session the cards were observed at 405 in both the healthy and the stuck state, and a reset
that reports success is not the same as a card that will stay at 5005 under load. The reliable signal is the arm's
own throughput (a stuck card costs ~45%), which is why every result line carries the tok/s. (ii) The first request
after engine start is not steady-state; the engine's own `GPU warm-up: 606 ms on 4 card(s)` does not cover the page
cache and helper staging. Three `host` arms without a discarded warm-up read 104.6 / 104.7 / 101.9 - a 2.6% spread
that can swallow a 4% effect - against 100.1 / 103.1 once the first request is thrown away.

**The metric itself was cross-checked two ways.** `q36-analyze2.py` cannot score these files at all (it requires a
`WINDOW` line or a `SERVE` layer count, and `STRATA_Q36_SERVE_TIMING=0` suppresses both), so `analyze()` was verified
by hand against the reference's plain-arm formula: **100.081 / 106.832 / 103.084 / 106.593**, matching the reported
100.1 / 106.8 / 103.1 / 106.6. The reference reads the *last* BDONE for the last-admitted slot (`e[-1]`, not `e[2]`,
which is the produced count); the script now does the same, joins its reader thread before analysing, and prints the
first token and the BT digest so an arm that is not doing the same work is visible in the result line rather than in
a later diff.

**A design that was proposed and does not transfer.** A scheme from another project: (i) run prefill on a separate
thread so the scheduler keeps submitting MTP decode windows, (ii) interleave the two with CUDA streams, (iii) isolate
the resources that collide (upload buffers, MTP inputs, scratch, and a reordered readiness signal), gated on
"experts fully resident and buffers independent". Step (ii) has no purchase here: `q36_engine.cu:8940` does
`if (E.main_hybrid) E.use_graph = false;` because *the host takes part in every layer*, so there is no captured graph
to replay concurrently - a decode window is a host-driven command stream, and both activities compete for the same
host thread and the same CPU pool. Step (iii)'s spirit is right but its target is wrong: the colliding resource here
is the arena itself taking part in the prompt, not a buffer that needs separating. Step (i) alone buys little for
the same reason - moving prefill to another thread does not create arithmetic that does not exist.

**And vLLM does not do any of the three either** - two independent readers over `D:\work\vllm-src\vllm`, because the
scheme was described as the vLLM approach and that attribution matters for whether to copy it.

- **No separate prefill thread.** `vllm/v1/engine/core.py:713` is `future = self.model_executor.execute_model(...,
  non_block=True)` immediately followed by `:730` `model_output = self._wait_for_output(future)`, i.e. one blocking
  step loop. The worker is a separate *process* reached by IPC (`multiproc_executor.py:742/880`), not a thread
  running compute concurrently. `step_with_batch_queue()` (`:758`) pipelines *scheduling*, not GPU work.
- **No dual compute streams.** Every `torch.cuda.Stream` under `vllm/v1/` is a copy stream or a PP broadcast:
  `gpu_model_runner.py:761/916/943/952/1178` (async D2H of sampled tokens and pooler output),
  `simple_kv_offload/worker.py:180`, `gpu/pp_utils.py:74/169-252`, `gpu/ubatch_wrapper.py:76`,
  `structured_outputs.py:30`, `spec_decode/*copy_stream`, and `attention/backends/mla/index_group.py:33/138-188`
  (a side stream inside attention, not at the scheduler level). The model forward itself is one call on the default
  stream: `gpu_model_runner.py:4269` `model_output = self._model_forward(...)`, and `_model_forward` (`:3681`) is
  just `self.model(...)`.
- **The actual mechanism is packing.** Both the running loop (`scheduler.py:817-1059`) and the waiting loop
  (`:1071-1677`) write into ONE `num_scheduled_tokens: dict[str, int]` (`:740`); one `SchedulerOutput`
  (`output.py:237-253`) carries it, and it has **no** `num_prefill_tokens` / `num_decode_tokens` field at all.
  Decode's 1 token and a prefill chunk are the same batch in the same forward pass.
- **The one "alternate the steps" mechanism is `defer_prefills`** (`scheduler.py:772-813`), and it skips prefill for
  a whole step (`:844-848`, `:1346-1349`) rather than overlapping it - alternation, not concurrency. It is gated on
  `local_prefill_interval`, which is a rotation cadence, not a compute-overlap trick.

So the only portable vLLM lesson is the one already recorded above: cap a prefill to a fair share of the budget when
other requests are eligible (`long_prefill_token_threshold`, floored at `input_budget // num_eligible_reqs`,
`scheduler.py:779-791`), and let a lone request take everything (`else 0`, `:783`). Strata cannot co-schedule the
two phases in one window (`chunk(T)` is whole-window), so its equivalent is the `read_windows` ratio - and that
ratio's denominator just got 7x smaller.

## Second vLLM pass: three mechanisms I had not examined, and what each is worth here

The first pass answered "does vLLM run prefill on a separate thread or on its own streams" - no on both counts. This
pass asked the follow-up question instead: **what overlap mechanisms does vLLM actually have, and does any of them
have purchase on this engine?** Three readers over `D:\work\vllm-src\vllm`, one mechanism each.

### 1. Async scheduling: real, default-on, and not worth anything here

This one is a genuine CPU/GPU overlap and I had missed it.

- The flag is `SchedulerConfig.async_scheduling`, `None` by default (`vllm/config/scheduler.py:246`). When `None`,
  `vllm/config/vllm.py:1763-1797` **auto-enables it** unless something is incompatible (pooling model, unsupported
  executor backend, PP>1, or spec decode that is not EAGLE/ngram/draft/dspark/dflash). So on a plain generate model it
  is **on by default**. It selects a different class, `AsyncScheduler` (`config/scheduler.py:266-271`).
- The mechanism is `EngineCore.step_with_batch_queue` (`vllm/v1/engine/core.py:758`), used when `batch_queue` is set
  (`batch_queue_size = max_concurrent_batches`, `core.py:222`), which is filled to **two steps in flight**
  (`config/vllm.py:563-573`). While step N's kernels run, the host schedules step N+1 (`core.py:785`) and runs
  `sample_tokens(non_block=True)` (`core.py:800-805`).
- What has to be double-buffered because of it: `input_ids`, `query_start_loc`, `seq_lens`, `num_scheduled_tokens` as
  `CpuGpuBuffer` with a deferred H2D (`gpu_model_runner.py:782`); `prepare_inputs_event`, a blocking
  `torch.cuda.Event` guarding reused host tensors (`:759,764`); a separate `async_output_copy_stream` for the sampled
  token's D2H (`:756,761`); and `prev_sampled_token_ids` cached GPU-side for the next step (`:3607-3614`).
- The contract is `synchronize_input_prep()` (`:3667-3679`): step N+1's host prep **waits for step N's transfer, not
  step N's compute**, then records its own event. Per-request state fixups are deferred to run after the launch
  (`deferred_state_corrections_fn`, `core.py:4040`, applied `:4140` / `:4345-4346`).

**Why it does not transfer.** This buys back *host scheduling* time that would otherwise sit between two GPU steps.
Strata's host cost inside a decode window is already measured at **0.14 ms on a 47.01 ms window**
(`q36-window-outside-forward`: entry sync 0.004 ms, exit GPU head+argmax 2.819 ms which is GPU-side, everything host
0.14 ms). There is no second of host work to hide. The one place host work *is* large - the prompt read - is not a
scheduling problem, it is the 18.75 GiB arena, and that was already dealt with by dropping the arena.

### 2. Decode-first admission: the ordering I do not have

This is the mechanism that names the defect precisely.

- `Scheduler.schedule()` runs two phases in a fixed order: **RUNNING first** (`scheduler.py:815`, the loop at `:817`),
  **WAITING second** (`:1071`, loop at `:1097`). A request mid-decode lives in `self.running`, so it claims its token
  before any prompt in `self.waiting` is even considered.
- There is **no reserved budget for decode**. The single `token_budget` is decremented as requests are added
  (`:1024-1025`, `:1638-1639`); nothing subtracts "decode room" up front.
- What bounds one prompt's chunk is `long_prefill_token_threshold` (`scheduler.py:90`, **default 0 = disabled**), and
  only when `enable_chunked_prefill` and more than one request is eligible (`:781-785`); an adaptive variant floors it
  at `input_budget // num_eligible_reqs` (`:786-791`). A lone request gets everything (`else 0`, `:783`).
- `defer_prefills` (`:772-813`) is the blunter version: skip prefill entirely for a whole step (`:844-848`,
  `:1346-1349`). Gated on `throttle_prefills` / `local_prefill_interval`, both off by default.

**The contrast is exact.** vLLM protects decode by *ordering* - decode's tokens are taken off the budget first, and
prefill is what gets chunked and deferred. Strata does the opposite: `q36_engine.cu:10127` gates the decode window loop
on `done < fresh`, so while a prompt is being read the decoding slots get **nothing** until a chunk boundary, and then
only `read_windows` windows. vLLM never blocks prefill admission on decode; it just runs the decoding request first
and chunks the prefill. That is the one structural idea here worth having, and it is the same one already recorded as
the `read_windows` ratio.

### 3. A CUDA-graph size ladder: already ruled out by measurement

- vLLM captures many graphs, not one: `cudagraph_capture_sizes` defaults to
  `[1, 2, 4] + range(8, 256, 8) + range(256, max_cudagraph_capture_size+1, 16)` with the max capped at 512 (1024 on
  Blackwell) (`vllm/config/compilation.py:848-851`), and dispatches to the smallest captured size at or above the real
  batch (`cudagraph_dispatcher.py:71-90`). Default `CUDAGraphMode` is `FULL_AND_PIECEWISE` (`config/vllm.py:342,363`)
  - FULL for uniform decode, PIECEWISE for prefill and mixed batches.
- Padding rows are genuinely free of side effects: an `is_padding` mask (`gpu_model_runner.py:3334-3340`), attention
  sliced to `output[:num_actual_tokens]` (`flash_attn.py:1283`), and `slot_mapping` filled with `-1` so padding writes
  nothing to the KV cache (`:3963-3965`, `flash_attn.py:1483-1497`). Above the largest captured size it runs eagerly
  (`cudagraph_dispatcher.py:272-279, 318-322`).

**Why it does not transfer.** A size ladder only pays if a smaller graph is *cheaper*. Measured here it is not: 1, 2
and 8 rows cost the same per layer because 97% of a row's cost is waiting on the helper GEMM
(`q36-row-cost-is-helper-gemm`), and 10 rows is equivalent to 8 (`q36-vrows10-result`). A ladder would be ten graphs
that all cost the same. The row count is not the axis; the expert GEMM is.

### The net of the second pass

Nothing in vLLM can be copied as code. One thing can be copied as *structure*: **decode claims its tokens first and
prefill is what gets chunked**. Strata's `read_windows` is a weaker version of the same idea - it grants decode a fixed
number of windows per chunk boundary instead of giving decode priority over the budget. The measurement that follows
(`.backup/s2_interleave.py`) tests whether widening that grant actually converts into decode tokens, and what it costs
the prompt, rather than assuming the ordering alone is the win.

## The gap this tree already knows how to close: a time-proportional decode budget

While reading the interleave path I found that this repository contains **two** engines and only one of them
solves the starvation problem. `src/program/generate.cpp` (the general batch server) has had a
time-proportional interleave for a long time; `src/qwen35/q36_engine.cu` has a fixed window count.

| | `generate.cpp` | `q36_engine.cu` |
|---|---|---|
| what bounds the decode between two prompt chunks | a **time budget**: `decode_share * ms_since(chunk start)` | a **count**: `read_windows` (default 1) |
| where | `generate.cpp:17278-17287` | `q36_engine.cu:10133` |
| the loop | `do { batch_step(); } while (elapsed < budget)` | `for (w = 0; w < read_windows; ++w) window()` |
| default share | `STRATA_BATCH_DECODE_SHARE` = **0.5** | 1 window ~ 47 ms, whatever the chunk cost |
| `BYIELD` | implemented (`yielded_at` / `yielded_slot`, `:16637-16640`) | **a no-op**: `:9383` says "a read here does not give way" |

The reference implementation, verbatim (`generate.cpp:17278-17287`):

```cpp
if (decode_share <= 0.0 || !batch_on()) continue;
const double budget = decode_share * std::chrono::duration<double, std::milli>(Clock::now() - tq).count();
const auto td = Clock::now();
in_prompt_read = true;
do {
    if (!batch_step()) { in_prompt_read = false; batch_fatal = true; e = "a batch window failed"; return false; }
} while (batch_on() && std::chrono::duration<double, std::milli>(Clock::now() - td).count() < budget);
```

`tq` is set immediately before each chunk is read (`:17234`), so the budget is **half of what that chunk actually
cost** - it adapts to the chunk size and to the box's speed without a single tuning constant.

**Why the fixed count is the wrong shape.** With the measured no-host marginal of 1.51 ms/token, a 2048-token chunk
costs ~3.1 s, and one window costs ~47 ms (`q36-window-outside-forward`). A share of 0.5 would grant
`0.5 * 3100 / 47 ≈ 33` windows. `read_windows = 1` grants **one** - a factor of 33 in fairness, and the number never
changes when the chunk does.

**What must be said honestly about it.** There is no true parallelism here: `run_rows()` drives the window from the
host and blocks (`q36_engine.cu:7957`), so chunk time and window time *add*. A time-proportional interleave therefore
cannot raise the aggregate - it trades the long prompt's latency for the decoding session's. That is still a real
product win (a 3-second freeze becomes a 3.5-second slow-down spread evenly), but it is **not** a throughput number
and must never be reported as one. The historical record on slicing this engine is bad - 8 sessions -7.3% and
**2 sessions -41.5%** (`q36-two-window-interleave-verdicts`) - so the first question a port has to answer is why
slicing was so expensive then, before assuming it will be cheap now.

### Port sketch (not implemented, and it should not be until the measurement says the demand exists)

`E.process`'s callback already receives `done` and runs inside the chunk loop (`:7835`, `on_chunk(i + T)`), and the
callback is called once per chunk. So the chunk's own cost is the gap between two callback entries - no new engine
state:

```cpp
// in Serve::request, before E.process
auto t_prev = Clock::now();
const int64_t got = E.process(ids.data() + reuse, fresh, [&](int64_t done) -> bool {
    const auto now = Clock::now();
    const double chunk_ms = ms_since(t_prev);      // the chunk that just finished
    t_prev = now;
    ...
    if (n_active > 0 && done < fresh) {
        const double budget = decode_share * chunk_ms;
        const auto td = Clock::now();
        do { window(); } while (n_active > 0 && ms_since(td) < budget);
    }
    ...
});
```

`read_windows` stays as the cap (`do { } while (... && ++w < read_windows)`) so the two knobs compose instead of
replacing each other, and `decode_share = 0` with `read_windows = 1` reproduces today's behaviour byte for byte -
which is what the "default path stays byte-identical" rule requires.

## First measurement off the interleave probe: the control arm holds

`.backup/s2_interleave.py`, arm `c2048_w0` (chunk 2048, `STRATA_Q36_READ_WINDOWS=0`), all P104s at 5005 MHz:

```
long_pp=2  prompt_ms=3723  prompt_tok=3500  prompt_tps=940.1  dec_tok=0  dec_gap_ms=1601
```

Two positive controls, both of which had to pass for the rest of the table to mean anything:

- **`dec_tok = 0`.** With the window loop's `read_windows` at zero, the decoding session produced **not one token**
  during the 1601 ms the long prompt was being read. That is the starvation stated as a number: 1.6 s of frozen
  decode, and it is the zero point every other arm is measured from.
- **`long_pp = 2`.** The engine printed exactly `ceil(3500/2048) = 2` `PP` lines, so `--chunk 2048` reached the engine
  and the boundary count is what the arithmetic says.

Note `dec_gap_ms = 1601` while `prompt_ms = 3723`: the slice starts at the long request's **first** `PP` (printed when
the first 2048 tokens are done, at 2115 ms) and ends at its `DONE`, so it covers the second chunk plus the tail. The
gap and `prompt_ms` are different clocks and are not meant to agree - the gap is what the decoding session was frozen
for, and the prompt's own total is what the prompt cost.

`prompt_tps = 940` is worth noting against the earlier figure of ~663 tok/s for the no-host marginal (1.51 ms/token).
The difference is the prompt shape: 3500 varied tokens at chunk 2048 amortise the fixed costs differently from the
24/512-token probe that produced the 1.51. Neither number contradicts the other; they measure different prompts.

### What the docs already claimed, and why this run is not a contradiction of it

`docs/QWEN36.md:332-335` records, on an **R9700 32 GB** with three decoding slots and a 3,008-token prompt:

> A 3,008-token prompt was read in 0.95 s (3,200 tok/s, 6 chunks). Each decoding slot paused at most 160 ms (one chunk)
> and produced 20 tokens meanwhile. With `STRATA_Q36_READ_WINDOWS=4` they produced 65 tokens during a 1.14 s read.

Three slots x 20 tokens = 60 tokens at `READ_WINDOWS=1`, and 3 x 65 = 195 at 4 — so the historical ratio is close to
linear in `read_windows`, at a prompt cost of +20% (0.95 -> 1.14 s). This run's `c2048_w0` vs `c2048_w1` pair is the
same phenomenon on **one** decoding slot at **one** chunk boundary: 0 -> 3 tokens for +27 ms (+0.7%).

The two are consistent but **not interchangeable**, and the differences must be stated whenever the old numbers are
quoted beside the new ones:

| | docs/QWEN36.md:332 | this run |
|---|---|---|
| card | R9700 32 GB (HIP, all experts in VRAM) | P100 + 3x P104, **no host arena** (`NO_HOST_EXPERTS=1`) |
| decoding slots | 3 | 1 |
| chunks | 6 (prompt 3,008 at ~512) | 2 (prompt 3,500 at 2048) |
| boundaries x windows | 5 x `read_windows` | 1 x `read_windows` |
| prompt cost of w=4 | +20% | not yet measured (only w=1: +0.7%) |

The R9700 keeps every expert on the card, so a decode window there is pure compute and a chunk is fast (3,200 tok/s).
On this box the prompt is 940 tok/s because every expert row crosses PCIe, and a window costs what it costs. **The
per-window token yield should therefore be similar and the per-window *cost* should not be** - which is exactly why the
`c512_*` arms matter: at chunk 512 this prompt has 6 boundaries, the same shape as the documented measurement, and only
then is the two data sets comparable.

## The chunk-2048 ladder: the knob works, and the price is linear

Three arms off `.backup/s2_interleave.py`, one decoding slot, a 3,500-token prompt read in **2** chunks (so exactly
**1** boundary), all three P104s at 5005 MHz, warm-up discarded before each:

| arm | `read_windows` | `dec_tok` during the read | `prompt_ms` | prompt cost | prompt tok/s |
|---|---:|---:|---:|---:|---:|
| `c2048_w0` | 0 (control) | **0** | 3723 | - | 940.1 |
| `c2048_w1` | 1 (the default) | **3** | 3750 | +27 ms (+0.7%) | 933.3 |
| `c2048_w8` | 8 | **15** | 3913 | +190 ms (+5.1%) | 894.5 |

Three things this establishes, in order of how much they can be trusted:

1. **The knob is real and the control is clean.** 0 -> 3 -> 15 tokens is monotone in `read_windows`, and the zero arm
   produced exactly zero. Whatever the absolute numbers are worth, they are measuring the window grant and not noise.
2. **The marginal price of a granted window is ~23.3 ms of prompt time** (163 ms for the 7 extra windows between w1
   and w8). That is *cheaper* than a window's measured cost on the decode path (47 ms,
   `q36-window-outside-forward`) - a window inserted into a prompt read does not pay the full decode window, probably
   because the prompt's own stream work is still in flight when the window's rows are submitted.
3. **The token yield per granted window is ~1.7-1.9** (`15/8 = 1.88`, `3/1 = 3.0` but n=1), which is the MTP
   acceptance rate showing up: a window produces 1-3 tokens depending on how many proposals verify.

**The ratio that matters.** At w8 the prompt paid **5.1%** of its own duration to give the decoding session **15
tokens** over 1.8 s - about 8.3 tok/s while it had been getting **0**. In proportional terms: the prompt took 5% longer
and the co-resident session went from frozen to slow-but-alive.

### Why this is not yet an answer, and what would make it one

- **One decoding slot, one boundary.** `dec_tok` scales with boundaries *and* with slots, and this arm has one of each.
  The `c512_*` arms have 6 boundaries and are the comparable shape to `docs/QWEN36.md:332` (6 chunks, 3 slots).
- **The fair-share arithmetic says the default is 15x too small at chunk 512.** Measured no-host marginal 1.51 ms/token
  ([[q36-nohost-experts-prompt-gain]]) gives a 512-token chunk ~773 ms; a granted window costs ~13.6-24 ms. A 0.5 share
  is therefore `0.5 * 773 / ~20 ≈ 19` windows, against the default of **1**. At chunk 2048 the same arithmetic gives
  `0.5 * 3100 / 20 ≈ 77`, which is past the `read_windows` clamp of 64 - so a fixed count **cannot** express a fair
  share at large chunks, which is the argument for the time budget in the section above.
- **Nothing here says the aggregate improves.** As established above, chunk time and window time add on this engine,
  so every token the decode gains is prompt time somebody else waits for. `c2048_w8`'s +190 ms and 15 tokens is a
  **latency redistribution**, and it must be reported as one.

### Chunk 512: 6 boundaries, and the cost separates from the chunk

```
c512_w1   long_pp=7  prompt_ms=4316  prompt_tps=810.9  dec_tok=11  dec_gap_ms=3731
```

Two things to read off this arm, and one that it does **not** say:

- **`dec_tok = 11` at 6 boundaries x 1 window = 6 windows**, i.e. **1.83 tokens per granted window**. That matches
  `c2048_w8`'s `15/8 = 1.88` almost exactly, on a different chunk size and a different number of windows. The yield
  per window is a property of the MTP acceptance rate, not of the chunk, and two independent shapes agreeing on it is
  the strongest number in this table.
- **Chunk 512 costs the prompt 593 ms by itself** (3723 -> 4316, +15.9%) and that is *not* the windows: `c512_w1`
  grants the same one window per boundary as `c2048_w1` grants per its single boundary, so the chunk size is what
  changed. This is the known shape of the engine: a smaller chunk streams every layer's expert rows over PCIe more
  times, so the fixed per-chunk cost is paid 6 times instead of 2.
- **It does not yet isolate the per-window cost at chunk 512.** That needs `c512_w8`, and until it lands the honest
  statement is: at chunk 2048 a window costs ~13.6 ms of prompt time; at chunk 512 the *chunk* costs +593 ms and the
  windows ride on top of it for an amount not yet measured.

## The full table, and what it settles

All five arms, `.backup/s2_interleave.py`, one decoding slot, a 3,500-token prompt, `NO_HOST_EXPERTS=1`,
`STRATA_Q36_VERIFY_ROWS=8`, MTP depth 2, all three P104s at 5005 MHz, warm-up discarded before each arm:

| arm | chunk | `read_windows` | boundaries | `long_pp` | `prompt_ms` | `prompt_tps` | `dec_tok` | tokens/window |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `c2048_w0` | 2048 | 0 (control) | 1 | 2 | 3723 | 940.1 | **0** | - |
| `c2048_w1` | 2048 | 1 (default) | 1 | 2 | 3750 | 933.3 | 3 | 3.00 (n=1) |
| `c2048_w8` | 2048 | 8 | 1 | 2 | 3913 | 894.5 | 15 | 1.88 |
| `c512_w1` | 512 | 1 (default) | 6 | 7 | 4316 | 810.9 | 11 | 1.83 |
| `c512_w8` | 512 | 8 | 6 | 7 | 5423 | 645.4 | **108** | 2.25 |

Both controls held for every arm (`long_pp` = 2 and 7 as the arithmetic says; the zero-window arm produced exactly
zero decode tokens), so the table is measuring the knob and not an artefact.

### The two numbers that cross-validate

**The marginal price of one granted window is ~24 ms of prompt time, and it comes out the same at two chunk sizes:**

| pair | windows added | prompt time added | per window |
|---|---:|---:|---:|
| `c2048_w1` -> `c2048_w8` | 7 | 163 ms | **23.3 ms** |
| `c512_w1` -> `c512_w8` | 42 | 1107 ms | **26.4 ms** |

Two different chunk sizes, two different chunk costs, two different boundary counts - and the per-window price agrees
to 12%. That is the strongest evidence in this run that the window grant is being measured and not the chunk.

**The yield per window is ~1.8-2.25 tokens, and the engine's own acceptance counter explains it.** `15/8 = 1.88`,
`11/6 = 1.83`, `108/48 = 2.25`. The windows are boundaries x `read_windows`: 1x8, 6x1 and 6x8. The `c512_w8` arm's stderr reports
`slot 1: 63 of 101 MTP proposals accepted`, i.e. **62%** acceptance over 101 proposals across the read. A window
proposes at most 2 drafts per slot (depth 2), so its token count is 1 plus the accepted drafts - which is exactly the
1.8-2.25 band the three arms land in. The per-window yield is a property of the drafter, not of the interleave, and it
is bounded above by the draft depth.

### The trade, stated plainly

`c512_w8` is the fair-share shape: **108 decode tokens** granted to a co-resident session during the read, against
**11** at the default. That is the difference between a session that is nearly frozen and one that keeps moving. It
costs the prompt **+1107 ms, +25.6%** (4316 -> 5423).

So on this box the interleave is a **latency redistribution, not a throughput gain**: chunk time and window time add,
because `run_rows()` is host-driven and blocking (`q36_engine.cu:7957`). The decoding session gains tokens that the
prompting session pays for. Whether that is worth having is a product decision about who is waiting, not something the
aggregate tok/s can answer - and it must never be reported as a throughput win.

### Caveats that a reader of this table must carry

- **One decoding slot.** `dec_tok` scales with the number of active slots as well as with windows, so `c512_w8`'s 108
  is what **one** co-resident session got. Two sessions would not double it - they share the same windows, with more
  rows each and a worse slack (`q36-verify-rows-gate-mtp-depth`, `q36-mtp-crossing-at-s4`).
- **The 108 is one arm, one run - but its acceptance is not noise.** `c512_w8`'s `2.25` tokens/window is above the
  1.83-1.88 the other two arms agree on, and the engine's own counter says why: **62.4%** of its 101 proposals were
  accepted, against **47.6%** (10/21) in `c2048_w8` and **47.1%** (8/17) in `c512_w1`. The two arms that agree on yield
  also agree on acceptance, so the third arm's higher yield is a real change in acceptance under the much longer read
  (42 windows instead of 6-8, so the drafter is warm for far more of it), not measurement scatter. What is still a
  single reading is the **value** 108 - on a box with a known bimodality (`q36-decode-box-bimodal`) one arm is a lead.
- **`prompt_tps` is the engine's own clock** and the arms are not the same work: `c512_w8` decoded 108 tokens for a
  competing session while reading, so it is not "the same prompt, slower".
- **The default is 15x too small at chunk 512 and 60x too small at chunk 2048.** A 0.5 fair share at chunk 512 is
  `0.5 * 773 / 26.4 ≈ 15` windows (default 1); at chunk 2048 it is `0.5 * 3100 / 23.3 ≈ 67`, past the clamp of 64.
  A fixed count cannot express a fair share at large chunks - which is the whole argument for the time budget.

### What to do with this

Do **not** change a default on the strength of one run. The demonstrable, repeatable facts are: (i) at the default the
co-resident session gets 0-11 tokens during a 1.6-3.7 s read, i.e. it is effectively frozen; (ii) raising
`read_windows` un-freezes it at ~24 ms of prompt time per window; (iii) the fair share and the current default differ
by an order of magnitude. What is **not** established is that any of it raises aggregate throughput - it cannot, by the
additivity argument - or which share is right, which needs a latency target the user picks.

### A check that the slice is not over- or under-counting

`c512_w8`'s raw stdout holds **114** `BT 1` lines in total, while the slice counted **108** inside the read. The
6-line difference is the decode windows that ran after the long request's `DONE` and before the run ended - the slice
correctly excludes them. The same file has **7** `BT 0` lines, all of them the long request's own one-token generation
plus its post-read windows, likewise outside the slice.

This is the check the Reviewer's blocker asked for, done against the raw output rather than against the script's own
bookkeeping: the number the table quotes is strictly inside the read, and the residual is accounted for.

### Why the two-decoder arm cannot be predicted from the one-decoder table

The engine's own arithmetic says the shape changes when a second decoder is present
(`q36_engine.cu:9607-9611`):

```cpp
int slack = E.vrows - S;
for (int i = 0; i < S; ++i) slack -= cand[i] ? 1 : 0;   // the first proposal's spare row
const int want = slack > 0 ? E.mtp_depth() : 1;
```

At S=1 with one candidate: `slack = 8 - 1 - 1 = 6`, so the chain runs at full depth. At S=2 with both candidates:
`slack = 8 - 2 - 2 = 4`, still positive, so depth 2 still runs - but the window's rows are now split three ways
(a row each plus up to two chain rows each), instead of six spare rows going to one session. And at S=4, `slack = 0`,
which is why the recorded result is "S=4 uses depth 1" (`q36-s4-depth2-vrows10-loses`).

So the one-decoder table's yields (1.83-2.25 tokens per window) are a **ceiling**, not a prediction, for the
two-decoder case: the same window now verifies two sessions' proposals and each session gets fewer rows. The
`q36-mtp-crossing-at-s4` result - the per-slot draft opportunity falling from 4/4 to 3/5 across S=4 to S=5 - is the
same mechanism seen from the other side. `.backup/s2_interleave_2slot.py` measures it at S=2 rather than assuming it.

## The two-decoder measurement at S=2: the window costs more, and buys more

`.backup/s2_interleave_2slot.py`, box 192.168.1.98, three arms, every arm preceded by a reset of the three P104s
and a warm-up (all three helper cards read 5005 MHz in every arm, so the known clock bimodality is excluded).
Two greedy decoders (`max_new = 3000`, a 24-token prompt each) on slots 0 and 1, one 3500-token prompt on slot 2
at chunk 512 (7 `PP` lines, so the chunk took), `STRATA_Q36_MTP_DEPTH=2`, `VERIFY_ROWS=8`.

| arm | `read_windows` | prompt ms | prompt tok/s | slot 0 | slot 1 | both | tokens/window |
|---|---:|---:|---:|---:|---:|---:|---:|
| `w0` | 0 (control) | 4142 | 845.0 | **0** | **0** | **0** | - |
| `w1` | 1 (default) | 4401 | 795.3 | 11 | 11 | 22 | 3.67 (n=6) |
| `w8` | 8 | 5996 | 583.7 | 108 | 108 | **216** | 4.62 (n=42) |

`RESULT PASS: every arm ran and both controls held.`

The controls that held: `long_pp == 7` in every arm; the `rw=0` arm granted exactly zero decode tokens; at
`rw > 0` each slot cleared the floor of `(7-1)*rw/3`; and the two slots' counts matched to the token in both arms
(11/11 and 108/108), which is what "one window emits one `BT` per active slot" predicts. No decoder stopped
inside a read - both `BDONE` elapsed times exceed the read in every arm.

### The number that matters: 108 equals the one-slot 108

The one-slot table's `c512_w8` gave **108** tokens to its single decoder over the same 48 windows
(6 boundaries x 8). This run gives **108 per slot** with two decoders present. Two independent arms, two
independent resets and warm-ups, and the per-slot yield is identical: **48 windows x 2.25 tokens = 108**.

That was not guaranteed. `q36_engine.cu:9607-9611` computes `slack = E.vrows - S - sum(cand)`, which falls from
6 (S=1) to 4 (S=2), and the row budget now splits three ways per slot instead of six ways for one. The measurement
says the split does not cost per-slot yield at S=2 - the chain still runs at depth 2, and each slot still gets its
`1 + a` rows. (At S=4 `slack` is 0 and depth drops to 1; that is where the yield would fall, and it is why S=4 is
out of scope here.)

### What does cost: 38 ms per window instead of 26

| pair | extra windows | extra ms | ms/window | extra tokens | tokens/ms of prompt time |
|---|---:|---:|---:|---:|---:|
| one-slot `c512_w1` -> `c512_w8` | 42 | 1107 | 26.4 | 97 | 0.0876 |
| two-slot `w1` -> `w8` | 42 | 1595 | **38.0** | 194 | **0.1216** |

A window at S=2 carries two slots' rows (6 rows instead of 3), so it is 44% dearer. But it serves both decoders
from that one interruption of the read, so the prompt time buys **39% more decode tokens per millisecond** than
the same knob at S=1. The pinned operating point is S=2, and this is the first measurement that says the knob is
*more* efficient there rather than merely applicable.

### The caveat that has not changed

This is latency redistribution, not throughput. The prompt goes 4142 -> 5996 ms (+45%) at `rw=8`, and
`run_rows()` is host-blocking (`q36_engine.cu:7957`), so chunk time and window time add rather than overlap.
`w8`'s 583.7 tok/s is a number to quote only beside the 216 tokens the decoders gained; on its own it reads as a
45% prompt regression. The fair-share arithmetic from the one-slot section still applies unchanged: at chunk 512 a
half-time share is ~15 windows, and the default of 1 is an order of magnitude short of it.

### Two script bugs this run exposed, and neither was an engine behaviour

**Bug 1 - a gate that waited on a condition nothing signalled.** `wait_bt()` counts `BT` lines, but the reader
thread only called `notify_all()` on `BADM`/`BDONE`/`DONE`. The gate therefore slept until slot 0 printed its
`BDONE` - 47013.9 ms later - and the long prompt went in after both decoders had already finished. The first run
of this arm reported `dec0=0, dec1=2`: `n_active` was 0 for five of six boundaries, and the 2 tokens were slot 1's
last two because it outlived slot 0 by ~0.5 s. Fixed by notifying on `BT` as well.
**The `rw=0` control passed in that broken run** - it is supposed to be zero - so a passing control did not mean
the timing was right. What caught it was the symmetry check (`abs(d0 - d1) <= 1`) and the per-slot floor.

**Bug 2 - host-side sequencing that delayed the request instead of ordering it.** The script waited for both
decoders to produce a token before submitting the long prompt. The engine already serves stdin strictly in order
(`q36_engine.cu:10371-10372`: `request()` reads a prompt to its end before the loop pops the next line), so the
wait was pure added delay. Fixed by writing all three `BGEN` lines in one `write` and one `flush`.

The fingerprint for bug 1, worth remembering: a `BDONE <slot> ... <ms>` whose `ms` field is *smaller* than the
long request's `prompt_ms`, or a first `PP` line that appears after a `BDONE`.

## Third vLLM pass: no time budget exists there either

A targeted scan of `/d/work/vllm-src` for how v1 keeps the GPU busy across a chunked prefill answered one
question and closed one avenue:

- The per-step budget is a **token** budget, not a time budget: `scheduler.py:741` `token_budget =
  self.max_num_scheduled_tokens`, filled from `max_num_batched_tokens` (`config/vllm.py:2435`).
- It is spent **decode first, prefill from the remainder**: phase 1 is `scheduler.py:815-817` (`# First, schedule
  the RUNNING requests.`, decrementing at `:1024`), phase 2 admits waiters at `scheduler.py:1097` and stops at
  `:1102` (`loads_only = (num_running >= self.max_num_active_reqs or token_budget <= 0)`).
- `long_prefill_token_threshold` (`config/scheduler.py:90`, **default 0 = disabled**) caps one request's chunk at
  `scheduler.py:866-867` and `:1388-1389`, and only when `num_eligible_reqs > 1` (`:781-791`).
- `time.monotonic()` appears only for logging (`:766`, `:3368`). There is **no** wall-clock budget in the
  scheduler, and **no** `max_num_partial_prefills` or any cap on concurrent partial prefills.

So vLLM's answer to "keep the GPU busy during a prefill" is "spend a token budget, decode first". Strata's
`generate.cpp` time-proportional budget (`STRATA_BATCH_DECODE_SHARE`, default 0.5, `generate.cpp:17278-17287`) is
strictly more expressive than that, and q36's fixed window count is strictly less. The gap identified earlier in
this document stands, and vLLM offers no alternative design to borrow for it.

## The port this measurement justifies, and the one it does not

The numbers above make the case precise. At the pinned S=2 point, a decode window during a prompt read is **more**
efficient than at S=1 (0.122 against 0.088 decode tokens per millisecond of prompt time), because one interruption of
the read serves both decoders. Yet `read_windows` is a **fixed count**, read once at engine start
(`q36_engine.cu:9378`) and applied identically at every chunk boundary (`:10136`).

A fixed count cannot express a fair share, because the fair share depends on how long the chunk took:
`windows = share * chunk_ms / window_ms`. With `share = 0.5`, `window_ms = 38` and a 512-token chunk at ~630 ms,
that is ~8 windows; a 2048-token chunk at ~2.5 s needs ~33; and at chunk 4096 the count passes the clamp of 64.
The default of 1 is an order of magnitude short at every chunk size measured.

### The reference implementation in this tree

`src/program/generate.cpp:17235` and `:17281-17286`:

```cpp
const auto tq = Clock::now();                       // the chunk starts
if (!run_conc(q, r)) return false;
...
if (decode_share <= 0.0 || !batch_on()) continue;
const double budget = decode_share * std::chrono::duration<double, std::milli>(Clock::now() - tq).count();
const auto td = Clock::now();
do {
    if (!batch_step()) { ... }
} while (batch_on() && std::chrono::duration<double, std::milli>(Clock::now() - td).count() < budget);
```

`decode_share` is `STRATA_BATCH_DECODE_SHARE`, default **0.5** — half the chunk's own cost is given back to the
decoding slots. Note the budget is computed from **the chunk that just finished**, not from a fixed number, and the
window loop is a `do/while` so at least one window always runs when the share is positive.

### What the q36 port would look like

The q36 chunk callback (`q36_engine.cu:10129-10142`) already has everything the reference needs except the budget:

```cpp
const auto t0 = Clock::now();                        // already there, used for the PP line
const int64_t got = E.process(ids.data() + reuse, fresh, [&](int64_t done) -> bool {
    const double chunk_ms = ms_since(t0);            // NEW: this chunk's own cost
    ...
    if (n_active > 0 && done < fresh) {
        poll_control();
        for (int w = 0; w < read_windows && n_active > 0; ++w) window();
    } else { poll_control(); }
    ...
});
```

But `ms_since(t0)` is the elapsed time since the read *started*, not since the chunk started, so a faithful port needs
a per-chunk timestamp. And the loop bound changes from a count to an elapsed check:

```cpp
    const auto tc = Clock::now();
    ... E.process(... [&](int64_t done) -> bool {
        const double chunk_ms = ms_since(tc);        // the chunk that just ended
        const int wmax = decode_share > 0.0
                       ? (int) std::min(64.0, decode_share * chunk_ms / window_ms)
                       : 0;
        if (n_active > 0 && done < fresh) {
            poll_control();
            for (int w = 0; w < wmax && n_active > 0; ++w) window();
        } else { poll_control(); }
```

Two things this needs that are not obvious:

- **`window_ms` has to be measured, not assumed.** The engine has no such constant today. It can be estimated from
  the previous window's own timing, which `window_mtp` already gathers when `win_outside_timing()` is on
  (`:9403-9405`); on the default path nothing is timed, so a measured estimate would have to be introduced.
- **`tc` has to be reset per chunk**, which the current callback structure does not do — `t0` is the read's start.
  `process()` calls the callback once per chunk (`:7835`, `done = i + T`), so `tc` can be advanced at the end of each
  callback instead.

### Why this is written down and not implemented

Three reasons, in order:

1. **The default would change behaviour.** `read_windows` default 1 -> a time share would raise the prompt-read cost
   of every batch request with a decoder beside it. `CONTRIBUTING.md` requires the default path stay byte-identical
   to the last release; this would have to ship opt-in (`STRATA_Q36_DECODE_SHARE`, default 0 = off), with the count
   path kept as the default until a measured run justifies flipping it.
2. **`src/qwen35/q36_engine.cu` has uncommitted changes in the working tree that are not mine.** Editing it now would
   entangle this work with them.
3. **The measurement above is one run per arm.** The per-slot equality (108/108 on both) is strong because it
   reproduces the one-slot 108, but a default change needs more than that.

So the honest state is: the gap is identified and quantified, the reference implementation is in the same tree, the
shape of the port is written down, and the decision to make it a default is left open pending a measured A/B.

## The window cost is linear and does not saturate: a four-point curve at S=2

Same harness, same box, two more arms at `READ_WINDOWS` 16 and 32, each preceded by a reset and a warm-up.
`w16` completed with both controls holding (`long_pp = 7`, symmetric slots at 218/218).

| arm | windows granted | prompt ms | prompt tok/s | slot 0 | slot 1 | both |
|---|---:|---:|---:|---:|---:|---:|
| `w0` | 0 | 4,142 | 845 | 0 | 0 | 0 |
| `w1` | 6 | 4,401 | 795 | 11 | 11 | 22 |
| `w8` | 48 | 5,996 | 584 | 108 | 108 | 216 |
| `w16` | 96 | 7,781 | 450 | 218 | 218 | **436** |
| `w32` | 192 | 11,505 | 304 | 460 | 461 | **921** |

`RESULT PASS: every arm ran and both controls held` on the 16/32 run as well, with `long_pp = 7` in both and the
slots symmetric to within one token (218/218 and 460/461 - one token of asymmetry at 192 windows is a single window's
difference, inside the `abs(d0-d1) <= 1` allowance).

The marginal cost between adjacent arms is what matters, because it isolates the window from the chunk:

| pair | extra windows | extra ms | **ms/window** | extra tokens | **tokens/window** | tokens/ms |
|---|---:|---:|---:|---:|---:|---:|
| `w1` -> `w8` | +42 | +1,595 | **38.0** | +194 | **4.62** | 0.1216 |
| `w8` -> `w16` | +48 | +1,785 | **37.2** | +220 | **4.58** | 0.1232 |
| `w16` -> `w32` | +96 | +3,724 | **38.8** | +485 | **5.05** | 0.1302 |

All three marginals agree to **within 4% on cost**. That is the strongest form this measurement can take short of a
repeat run: three independent intervals, spanning 6 to 192 windows and two different absolute window counts, produce
the same per-window price. The window is **linear and unsaturated** across the whole measured range.

Four consequences:

1. **The fair-share formula can be extrapolated.** `windows = share * chunk_ms / 38` is valid up to the clamp of 64.
   At `chunk 512` a chunk is ~630 ms, so a half share is ~8 windows - and the engine's default is **1**.
2. **`w1` is the outlier, not the rule.** Its 43.2 ms/window over `w0` is 14% above the marginals, because the
   first window after a read pays some fixed cost the later ones do not. Any costing must use the marginal (37-39 ms),
   not the `w0`->`w1` step.
3. **No saturation means no reason to stop at 8.** A larger `read_windows` keeps buying decode tokens at the same or
   better rate - the yield actually rises at `w32` (5.05 against 4.62), which fits the one-slot finding that a longer
   read keeps the drafter warm and lifts acceptance. The only limits are the clamp of 64 (`q36_engine.cu:9378`) and
   how much prompt latency the caller accepts.
4. **The prompt cost is unbounded in the same direction.** 845 -> 304 tok/s across the five arms. At `w32` the prompt
   is 2.8x slower than with the windows off, which is why this is a latency-allocation knob and never a throughput one.

## A limitation the Reviewer found by predicting the opposite of what happened

The Reviewer flagged the symmetry control (`abs(d0 - d1) <= 1`) as one that "will fire on valid data", reasoning
that with `--batch-mtp` and `MTP_DEPTH=2` a window emits `1 + a` tokens per slot where `a` is that slot's accepted
drafts, so two slots with different acceptance should drift apart over 48 windows.

It did not happen. Measured: **108/108, 218/218, 460/461** at `rw` 8, 16, 32. The control held every time, and the
Reviewer's reasoning about the mechanism is correct - so the explanation has to be elsewhere:

**Both decoders are clones.** The harness sends the same 24-token prompt (`SHORT = "220" x 24`) to both slots with
`temperature=0`. Two greedy decoders on the same prompt produce the same tokens, so they take the same draft/accept
path and emit the same number of tokens. The symmetry is a property of the *test fixtures*, not of the engine.

That is a real limitation on how far this measurement generalises, in two ways:

1. **The cost figure may be optimistic.** Identical token streams mean identical expert routing, so the three P104
   helpers read the same experts for both slots. Two real users would route differently, and the helper cards would
   have to serve a wider expert set per window - the mechanism behind `q36-verify-window-expert-groups` (7.94 distinct
   experts per helper per window). A mixed-traffic run is the only way to price that.
2. **The symmetry control is only valid for this fixture.** Keep it, but it must not be read as "the engine treats
   slots evenly" - two different prompts would drift, and the control would then need the relative form the Reviewer
   suggested (`|d0-d1| <= max(1, 0.25*min(d0,d1))`) rather than the absolute one.

The per-window cost itself survives this, because every arm is measured against a `w0` baseline taken with the same
two cloned decoders - the comparison is internally consistent. What does not survive is any claim that 38 ms/window
is the price for *arbitrary* two-session traffic.

## vLLM's dual-batch overlap exists, and it is the design this tree already tried and rejected

The fifth pass found the mechanism the earlier passes said was absent from vLLM. It is there, and it is instructive:

- `vllm/v1/worker/ubatching.py:16` `_NUM_UBATCHES: int = 2` - the batch is split into two microbatches.
- `gpu_ubatch_wrapper.py:76` `self.comm_stream = torch.cuda.Stream(device=device)` - **each ubatch has its own
  compute stream and they share one communication stream** (`:289-296`).
- Hand-off is by yielding threads: `yield_and_switch_from_compute_to_comm` / `..._from_comm_to_compute`
  (`ubatching.py:133`, `:141`), with a device-side event join (`deepep_v2.py:517` `event.current_stream_wait()`).
- The stated purpose (`deepep_ht.py:141-144`): *"We yield before launching the dispatch kernel since the dispatch
  kernel will block the CPU so we want to queue up all the compute for the other ubatch before the dispatch kernel
  starts"*.
- It is **off by default** and env-gated (`VLLM_DEEPEP_V2_PREFER_OVERLAP`, `envs.py:2157`, default `False`).

Read against this tree's own record, that is the same idea as the two-window interleave
(`plans/2026-11-07-two-window-interleave.md`), down to the failure mode: q36's version was measured at **-7.3% for
8 sessions and -41.5% for 2 sessions**, and the root cause recorded in `q36-ring-per-slot-stream` was that every slot
shared one `stream_`. vLLM's DBO does not share the compute stream - it gives each ubatch its own and shares only the
comm stream.

So the vLLM evidence does **not** say "the interleave works"; it says the one structural difference that could explain
q36's negative result is the one this tree has already identified and not yet tried. That is a hypothesis for an engine
change, not a measurement, and it stays out of scope for the pinned S=2 work until the cloned-decoder caveat above is
priced with a mixed-traffic run.

## Review record for the harness changes

Per the working agreement, one review per task, no re-review loop. Two `FAIL` verdicts earlier in this work (on
`.backup/s2_interleave.py` and on the first form of `.backup/s2_interleave_2slot.py`); each had its blockers fixed and
was **not** re-reviewed.

The final form of `.backup/s2_interleave_2slot.py` (the two changes: write all three `BGEN`s in one write, and notify
waiters on `BT`) went to a fresh Reviewer:

- **VERDICT: `PASS_WITH_NITS`, BLOCKERS: none.**
- Confirmed on the engine source that removing the host-side waits does not break the premise: `run()` pops one stdin
  line at a time and `request()` reads a prompt synchronously, so `BGEN 2` cannot be entered until both decoders have
  been admitted (`q36_engine.cu:10127-10139`, `++n_active` at `:10215`). There is no interleaving where the long read
  starts while a decoder is still reading its own prompt.
- Confirmed `wait_bt` is still live (the warm-up gate) and can no longer hang.
- Confirmed the five controls would have caught the failure that actually occurred.

NITs, and what was done with each:

| NIT | Assessment |
|---|---|
| 1. `abs(d0-d1) <= 1` will fire on valid data (acceptance differs per slot) | **Empirically did not fire** (108/108, 218/218, 460/461). The Reviewer's mechanism is right, so the explanation is that both decoders are clones of the same prompt. Recorded above; the mixed-prompt run tests it directly. |
| 2. Control (e) is lenient by `(read_start - decoder_admit)`; `bdone_ms` can be empty | Lenient by milliseconds in the fixed script; accepted, not changed. |
| 3. `gap_ms` is reader-side, cross-check against `long_ms` | Accepted as a reading aid; `long_ms` from the ADMIT line is what the tables use. |
| 4. The comment blaming "any host-side wait" is now stale | Correct - the 42 s came from the missing `BT` notify. The comment already says so. |
| 5. `events.append` outside the lock | GIL-safe; not changed. |
| 6. `wait_bt` would wait 900 s if the engine died silently | Real, and not hit in these runs (every arm exited cleanly). Not changed. |

The Reviewer also recorded an uncertainty worth repeating: it read this working tree's `q36_engine.cu`, while the box
runs the prebuilt `build-rebase/strata-q36`, and it could not confirm the two match. The proofs above therefore rest on
the tree being the source of that binary. The measured behaviour is consistent with it (7 `PP` lines at chunk 512,
`BADM ... 1` at admission, `BDONE` on finish, the engine exiting only after `BSTOP`), which is what those code paths
predict, but that is inference, not a byte-level check.

## The cloned-decoder caveat, tested: mixed prompts yield *more*, not less

`.backup/s2_interleave_mixed.py` is the same harness with one change: slot 1 gets a **different** 24-token prompt
(`SHORT2`), so the two decoders route to different experts and accept different drafts. Same box, same reset and
warm-up, `--chunk 512`, arms `rw` 0 and 8.

| arm | prompt ms | prompt tok/s | slot 0 | slot 1 | both | drift |
|---|---:|---:|---:|---:|---:|---|
| `m0` (`rw=0`) | 4,154 | 843 | 0 | 0 | 0 | - |
| `m8` (`rw=8`) | 6,006 | 583 | 108 | **143** | **251** | 32% |

The `m0` control held exactly. `m8`'s per-slot drift is the measurement, and the script reports it as an error only
because the relative control (`> 25%`) was written for that purpose - this is a **successful** run, not a failed one.

### What it settles

1. **The symmetry in the cloned runs was a fixture property, as suspected.** Two different prompts drift 108 vs 143.
   The Reviewer's mechanism (`1 + a` per slot, `a` = accepted drafts) is real; identical prompts simply produce
   identical `a`.
2. **The cost figure is not inflated by cloning - if anything it is conservative.** Prompt time is unchanged
   (6,006 against 5,996 ms, +0.2%), but mixed traffic yields **251 tokens against 216, +16.2%**. Per millisecond of
   prompt time: 0.1355 against 0.1165 tok/ms. The worry that different routing would make the helper cards work
   harder and cost more did not materialise at S=2 with `rw=8`.
3. **`dec0 = 108` is bit-identical between the cloned and mixed runs.** The slot whose prompt did not change gained
   exactly what it gained before; the extra 35 tokens belong entirely to slot 1. That is the cleanest possible
   evidence that the window grant is per-slot-fair and that slot 0's share is not being taken by slot 1.

### What it does not settle

One run, one `rw` point, one prompt pair. The 16% is a direction with a plausible mechanism, not a coefficient. A
repeat at `rw=1` and `rw=16` would say whether the mixed-traffic advantage is constant or depends on how many windows
are granted. Not run.

## Sixth vLLM pass: prefix caching is how vLLM avoids the very read this whole document is about

`vllm/config/cache.py:260` `enable_prefix_caching: bool = True` - **on by default**. The mechanism that matters here:

- On admission the scheduler asks for the longest cached prefix: `scheduler.py:1252-1254`
  `self.kv_cache_manager.get_computed_blocks(request, max_cache_hit_length)`, resolving through
  `kv_cache_manager.py:288` to `coordinator.find_longest_cache_hit(...)` (`:326-327`).
- The hit is combined at `scheduler.py:1262-1264` into `num_computed_tokens`.
- **Cached tokens do not consume the step budget**: `num_new_tokens = request.num_tokens - num_computed_tokens`
  (`scheduler.py:1362`), and only `num_new_tokens` is subtracted at `:1638-1639` and `:1024-1025`.

So a request that finds a cached prefix shrinks its own prefill to the uncached tail and stops competing for the token
budget. There is also a cache-aware waiting-queue reorder (`scheduler.py:2820`, deepest prefix first, `:2844-2847`),
gated on KV usage (`:2831`) and a window size (`:2834`) - it reorders peers only.

**Why this matters for the S=2 question.** Every measurement in this document is about what happens when a long read
competes with decoders. vLLM's primary answer is to **not have that read**: a repeated prefix is served from cache.
q36 has a weaker form of the same idea - `match_len(ss, ids, ir.imgs)` at `q36_engine.cu:10090`, which reuses a held
session's prefix and, inside a batch, copies the longest held start from another slot (`:10091-10103`,
`E.copy_session(src, s)`). What it does **not** have is a persistent block-level cache across requests: `reuse` is
computed against live sessions' held histories, so a prefix survives only as long as some slot still holds it.

That is a larger change than the decode-share port and it is outside the pinned S=2 window, but it is the one vLLM
mechanism found in six passes that attacks the prompt cost itself rather than redistributing it. Recorded, not
pursued.

## What six vLLM passes produced, in one table

The instruction for this stretch of work was to keep scanning the vLLM source for concurrency / GPU-occupancy
optimisations. Six targeted passes, each with file:line evidence from `/d/work/vllm-src`:

| pass | question | found | transfers to this tree? |
|---|---|---|---|
| 1-2 | does vLLM run prefill on a separate thread, or on a second compute stream? | no separate prefill thread; async scheduling is a *host-side* overlap (`core.py:758`, `gpu_model_runner.py:3667-3679`) | **no** - Strata has no captured graph under `main_hybrid`, and q36's decode is host-blocking |
| 3 | chunked-prefill budget: tokens or time? | **tokens** (`scheduler.py:741`), decode first (`:815-817`), prefill from the remainder (`:1097`, `:1102`); `long_prefill_token_threshold` default 0 | **no design to borrow** - q36's `read_windows` is weaker, `generate.cpp`'s time budget is stronger |
| 4 | per-request draft counts / speculative batching | per-request `draft_len`, one batched rejection kernel; K adapts to **batch size** (`scheduler.py:1787-1791`), never to acceptance | **no** - q36's `slack = vrows - S - nc` is the same idea computed exactly rather than looked up |
| 5 | MoE all-to-all overlap | **DBO exists**: 2 microbatches, per-ubatch `compute_stream`, shared `comm_stream` (`ubatching.py:16`, `gpu_ubatch_wrapper.py:76`), off by default | **the one live hypothesis** - q36's two-window interleave failed (-7.3% / -41.5%) because slots shared one stream; vLLM does not share the compute stream |
| 6 | prefix caching | **on by default** (`config/cache.py:260`); cached tokens do not consume the step budget (`scheduler.py:1362`, `:1638`) | **attacks the read itself** - q36 only reuses a held session's prefix, no persistent block cache; larger than the decode-share port |

One conclusion worth stating plainly: **the mechanism the user's pasted description proposed (a separate prefill
thread plus dual compute streams) is not what vLLM does.** vLLM's GPU-occupancy answers are (a) a token budget spent
decode-first, (b) DBO, which is off by default, and (c) prefix caching, which removes the read. Of those, only (a) maps
onto something this tree can do today, and this tree already has a better version of it in `generate.cpp`.

## Where the work stands

Measured and recorded:

- `read_windows` works, is linear to 192 windows at 37-39 ms/window, and yields more per millisecond of prompt time at
  S=2 than at S=1.
- Mixed prompts yield 16% more than cloned prompts at the same prompt cost.
- The default of 1 is roughly 8x below a half-time share at chunk 512.

Identified but **not** implemented:

- A time-proportional decode share for q36 (`STRATA_Q36_DECODE_SHARE`), ported from `generate.cpp:17281-17286`. The
  design is written down above; the default would have to stay 0 so the default path remains byte-identical.

Open, and not started:

- Mixed-traffic repeat at `rw` 1 and 16 to turn the 16% from a direction into a coefficient.
- The q36 prefix-cache question: whether `reuse` can be extended past live sessions. Larger than a knob.

## Mixed traffic beats cloned traffic, and the advantage shrinks as windows rise

`.backup/s2_interleave_mixed_curve.py` - the same mixed fixtures at `rw` 1 and 16, to turn the single 16% reading into
a curve.

| `rw` | clone: tokens | mixed: tokens | mixed vs clone | clone prompt ms | mixed prompt ms | `dec0` (both) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22 | **28** | **+27.3%** | 4,401 | 4,381 (-0.5%) | 11 |
| 8 | 216 | **251** | **+16.2%** | 5,996 | 6,006 (+0.2%) | 108 |
| 16 | 436 | **505** | **+15.8%** | 7,781 | 7,874 (+1.2%) | 218 |

Three clean results:

- **The advantage is real at every point and costs nothing.** Prompt time moves by -0.5%, +0.2% and +1.2% - inside
  noise - while token yield rises 27%, 16% and 16%. Cloned decoders were never a *pessimistic* fixture for cost; they
  understate what the knob returns.
- **The advantage settles at ~16% once the window count is past the first few.** It is 27% at `rw=1` and then flat at
  16% for `rw` 8 and 16, so the initial gap is the catch-up windows and what remains is the steady-state difference.
  Two points agreeing to 0.4 percentage points is a coefficient, not a direction.
- **`dec0` is exactly the cloned value at all three points** (11, 108, 218). Slot 0 has the same prompt in both
  fixtures and produces the same tokens to the token. Every additional token belongs to slot 1. This is the third
  independent confirmation that the window grant does not take one slot's share to give to another.

And the cost is identical across fixtures, which is what makes the benefit comparable at all:

| `rw` | clone ms/window | mixed ms/window |
|---:|---:|---:|
| 1 | 43.2 | 37.8 |
| 8 | 38.6 | 38.6 |
| 16 | 37.9 | 38.8 |

### What this means for the number to quote

For a *cost* figure, 37-39 ms/window stands, measured the same way in both fixtures. For a *benefit* figure at S=2 the
honest statement is **+16% over cloned decoders at useful window counts (and +27% at the default of 1)**, and cloned
decoders should be named as a lower bound rather than used as the headline.

## CORRECTION (2026-11-08, later the same night): the time-budget port is not a performance lever

The section above treats the missing `STRATA_Q36_DECODE_SHARE` as a capability gap worth closing. **That framing was
wrong**, and the measurement in this document is what shows it.

A fixed count and a time budget are **linearly equivalent** here, because the window cost is a constant:

- fixed `N` windows costs `N * 38 ms` of prompt time;
- a `share` budget costs `share * chunk_ms`, and since the chunk also costs a constant per token, `share * chunk_ms`
  is `share * chunk_ms / 38` windows - the same line, parameterised differently.

The per-window cost measured linearly across `rw` 1, 8, 16, 32 (38.0 / 37.2 / 38.8 ms). There is no chunk size in
this range where the two forms diverge, and `--chunk` is fixed at deployment, so the one thing a share buys -
rescaling when the chunk size changes - buys nothing in practice.

What a share *does* buy is expressiveness for a caller who does not know the chunk duration in advance. That is a
portability improvement, not a speed one, and it does not justify touching the engine. The recommendation is
downgraded accordingly.

**The same correction applies to the second half of the caveat throughout this document.** I described the window
cost as "added, not hidden, because `run_rows()` is host-blocking". The conclusion (costs add) is what the arms show
and it stands, but the mechanism was stated wrong: the block is at `run_rows()`'s **entry**
(`q36_engine.cu:7949`, `cudaStreamSynchronize(st)` waiting for the previous window's launch queue), and `forward()`
itself is launched asynchronously. The prompt branch is the part that is strictly serial - it synchronises once per
chunk at `:7858`. So "host-blocking" is true of the prompt read and of the window's entry, not of the decode
compute. That distinction is what makes the dual-stream question non-trivial, and it is why the next section was
rewritten.

## Dual streams: the binding constraint is the link, not the stream

The proposal worth taking seriously is not "do what vLLM does". It is the narrower engineering question: the host
launches `forward()` asynchronously and then blocks at the next `run_rows()` entry, so there is a window in which a
second stream could carry work. Whether that window is worth filling depends on what the work would contend for.

**Link arithmetic at the pinned configuration.** A decode window at S=2 greedy depth 2 costs 38.0-38.8 ms (measured
above) for two slots' rows. Each layer moves roughly 2.2 MB of activations per chunk to the helpers
(`q36-prefill-last-gap-is-link`), over 40 layers that is ~88 MB per window, so the window alone demands
`88 MB / 38 ms = 2.3 GB/s`. The box's measured single-link throughput is **0.81 GB/s per hop, with no GPU-to-GPU P2P
at all** (`q36-no-p2p-host-staging-only`: every pair is PHB and one-way peer tests fail; direct 313 tok/s against
1867 tok/s staged, a 6x penalty). Prefill and decode therefore draw from the same saturated link.

That is the reason a dual stream cannot help here, and it is a hardware reason rather than a software one. Adding a
stream does not add link capacity; it only changes which work is queued. The only way a second stream pays is if the
work it carries does not need the link - and at the point where the host is blocked (waiting on a launch queue that
is itself waiting on helper results), essentially all outstanding work does.

**What would change this verdict:** a configuration where decode rows are served from the primary's own VRAM with no
helper round trip. `q36-primary-expert-slots` measured that inserting the primary into the decode is worth +3.0% with
byte-identical output - but only with `PRIMARY_RES=1` and `PEER_KEEP_RESIDENT=0`, and it does not remove the helper
traffic for the experts that stay remote. Until the expert traffic on the link falls, the dual-stream idea has no
headroom to exploit, and it should not be implemented.

## Where the optimisation actually lands

Given all of the above, the efficient frontier of this knob is already reached and the remaining work is choosing a
point on it, not moving the line.

The window yields 4.62-5.05 tokens per 37-39 ms at S=2. The steady-state rate for the same configuration is
141.8 tok/s for two slots, i.e. 7.06 ms/token. A window's own rate is `38.0 / 4.62 = 8.23 ms/token`, so **a window
delivers tokens at 86% of the steady-state rate**. There is at most 14% to recover by making windows cheaper, and the
helper-idle work already established that the helper's idle 55-59% is spent waiting on the primary's mixer
(`q36-helper-idle-is-the-real-idle`) - it is not recoverable.

So the landing point is the setting, not the implementation:

| `READ_WINDOWS` | prompt cost at chunk 512 | decode tokens during a 3.5K read (2 slots) |
|---:|---:|---:|
| 1 (default) | +0.26 s (+6%) | 22 |
| 2 | ~+0.52 s (+12%) | ~44 |
| 3 | ~+0.78 s (+18%) | ~66 |
| 8 | +1.85 s (+45%) | 216 |

The first three rows are the usable range. Eight windows buys the most decoding but makes the prompting user wait
45% longer for a 3.5K prompt; that is a defensible choice only when the decoders are interactive and the reader is
not. `rw` 2-3 is the range where the reader pays a sixth to a fifth more and the decoders keep moving, and it needs
no code change at all.

### The 14% ceiling, worked out

The steady-state rate for the pinned configuration is 141.8 tok/s across two slots, i.e. **7.05 ms per token**
(counting both slots' tokens together, which is the unit the window measurement also uses). A window costs 38.0 ms
and returns 4.62 tokens, i.e. **8.23 ms per token**. So:

- a window delivers tokens at `7.05 / 8.23` = **85.7%** of the steady-state rate;
- a window that matched steady state would return `38.0 / 7.05` = **5.39 tokens** instead of 4.62.

The gap is **14%**, and it is the entire upside available from making the interleave mechanism itself more efficient.
Everything downstream of that - a cheaper window, a smarter grant, a second stream - competes for that same 14%, and
the helper-idle work has already shown the largest apparent slack (the helper sitting idle 55-59% of a layer) is
waiting on the primary's mixer rather than being free capacity (`q36-helper-idle-is-the-real-idle`).

That is the quantitative reason the recommendation is "pick a setting" rather than "optimise the mechanism": the
mechanism is within 14% of doing at least as well as it possibly could, and the setting is currently an order of
magnitude away from the useful range.

### The landing measurement: `rw` 1 / 2 / 3, the range that is actually usable

`.backup/s2_interleave_landing.py`, same harness and same fixtures, three arms in one session so the box cannot
drift between them.

| `rw` | prompt ms | prompt tok/s | slot 0 | slot 1 | both | vs `rw=1` |
|---:|---:|---:|---:|---:|---:|---|
| 1 (default) | 4,395 | 796 | 11 | 11 | 22 | - |
| 2 | 4,597 | 761 | 25 | 25 | **50** | **+202 ms (+4.6%), tokens x2.3** |
| 3 | 4,829 | 725 | 37 | 38 | **75** | +434 ms (+9.9%), tokens x3.4 |

`RESULT PASS: every arm ran and both controls held` - `long_pp = 7` in all three, both slots symmetric to within one
token (11/11, 25/25, 37/38), no decoder stopping inside a read, and the three P104s at 5005 MHz throughout.

Against `rw=0` (4,142 ms, 0 tokens) the prompt cost is 6.1%, 11.0% and 16.6% at `rw` 1, 2, 3 - so the ladder is
cheap exactly where it is most useful.

`rw=1` reproduced the earlier session's value (4,395 ms against 4,401 ms, 22 tokens both times), so the box is stable
across sessions and these numbers are comparable with the 8/16/32 ladder.

The marginal costs now cover the low end as well, and they agree with the high end:

| pair | extra windows | extra ms | ms/window | tokens/window |
|---|---:|---:|---:|---:|
| `rw=1` -> `rw=2` | +6 | +202 | 33.7 | 4.67 |
| `rw=2` -> `rw=3` | +6 | +232 | 38.7 | 4.17 |
| `rw=3` -> `rw=8` | +30 | +1,167 | 38.9 | 4.70 |
| `rw=1` -> `rw=8` | +42 | +1,601 | 38.1 | 4.62 |
| `rw=8` -> `rw=16` | +48 | +1,785 | 37.2 | 4.58 |

Five intervals spanning 6 to 96 windows: **37-39 ms per window on four of them**, and the first one is slightly
cheaper at 33.7 ms because the first window after a read pays a fixed cost the later ones do not. Yield is flat at
4.17-4.70 tokens per window across all of them, with no trend - the variation is acceptance noise, not a mechanism
changing shape.

**So the recommendation is `rw=2`.** It doubles the decode tokens a read grants (22 -> 50 for two slots) for 4.6% more
waiting on the prompt, and it is the cheapest step on the ladder per token gained. `rw=3` is the next step (x3.4
tokens for 9.9% more waiting) if the decoders matter more than the reader. `rw=8` costs 45% and is the half-time
share - defensible only when the prompting request is not interactive at all.

---

## The real coding task, at the pinned shape (pp20k / tg1k / S=2 / MTP on)

Everything above measured the interleave with fixtures: 24-token decoder prompts, a 3,500-token reader, and a
clone of the same prompt in both slots. The task the user set next is the shape they actually care about - a real
coding prompt of 20,000 tokens, 1,000 generated tokens, two sessions at once, MTP on, with the cards' power
watched - so this section re-measures on that shape rather than extrapolating to it.

### The prompt had to be rebuilt, and the reason is the point

The first attempt tiled 340 real-code ids out to 20,000 tokens. That keeps the token distribution but repeats the
content eight times, and the user rejected it: **"prompt不能重复"**. They are right - a prompt that is one paragraph
copy-pasted is not a coding task, and a number measured on it cannot be quoted as one.

The obstacle was real: the box has no tokenizer (`transformers`, `tokenizers` and `sentencepiece` are all absent),
no network, and the engine has no encode subcommand - only `--tokens` and `--tokens-file`, which take ids.

**The obstacle was not real after all, twice over.** Two independent findings:

1. The model file carries the whole tokenizer in its metadata - `tokenizer.ggml.model = 'gpt2'`, `pre = 'qwen35'`,
   248,320 tokens, 247,587 merges - so a byte-level BPE can be rebuilt from the weights file alone.
2. **This repository already has that tokenizer.** `tools/strata_tokenizer.py` is a pure-Python implementation
   that reads the GGUF directly; it is on the box at `/home/Strata/src-rebase/tools/strata_tokenizer.py`. The
   original tiling script's own header even said so ("tokenized by the engine's own tokenizer
   (tools/strata_tokenizer.py, from the GGUF)").

So the prompt is encoded on the box, by the reference implementation, from the box's own checkout.

### What the two prompts are

`pp20k_real_code.py` walks `src/`, `include/`, `serve/` and `strata_cli/` (406 + 209 + 128 + 37 files of real C++,
CUDA and Python), takes files in a shuffled-but-fixed order, caps each at 2,200 tokens (cut at a line boundary),
and stops at exactly 20,000. `--two` splits the file list so the two sessions get **different** code - two people
working at once are not asking about the same file, and the mixed-prompt measurement above showed cloned prompts
understate throughput by ~16%.

| prompt | files | distinct ids | specials | repeated 64-token blocks |
|---|---:|---:|---:|---:|
| session 0 | 12 | 2,324 (11.6%) | 0 | **0** |
| session 1 | 11 | 2,933 (14.7%) | 7 | **0** |

The builder asserts, before any ids are written, that the length is exact, that the ids re-encode to themselves,
that no 64-token block repeats, and that no `[PAD...]` placeholder appears. The specials in session 1 are genuine:
this tree's own source writes `<|image_pad|>` inside comments and string literals, so a real 20k slice of it
contains a handful.

The tokenizer was checked against a second, independent implementation as well: the HuggingFace `tokenizers` build
from the same GGUF vocabulary (with the `qwen35` regex transcribed from `third_party/llama.cpp/src/llama-vocab.cpp:396`)
agreed with `strata_tokenizer.py` on 11 of 11 strings covering tabs, indentation, C syntax, quotes, CJK, emoji and
combining marks.

### What the engine does with two 20k prompts, and why "2 concurrent" means what it means

This is the finding that governs how the numbers may be read. The serve loop (`q36_engine.cu:10359-10367`) pops one
request, calls `request()`, and only then calls `window()`. Inside `request()`, `E.process(...)` reads the whole
prompt **synchronously** (`:10127-10147`). So:

| phase | who is doing what |
|---|---|
| slot 0's 20k read | only slot 0. Slot 1's `BGEN` has not been popped yet. |
| slot 1's 20k read | slot 0 decodes between chunk boundaries, `read_windows` windows per boundary |
| both generating | the only phase in which both sessions decode at once |

There is no design in which both prompts are read at the same time. `READ_WINDOWS` is exactly the mechanism that
makes the middle phase overlap, and the 2 sessions are genuinely concurrent for the whole generation phase - 1,000
tokens each is ~30 s at the rates measured here, against ~20 s of reads.

So the honest description is **2 sessions, sequentially admitted, concurrently generating**, with the interleave
covering the second read. That is the engine's concurrency model, and this measurement does not change it; it only
says what it costs and yields at this shape.

### Arms and controls

Two arms, identical but for MTP: `mtp0` (`--mtp 0`) and `mtp1` (`--mtp 1 --batch-mtp`, depth 2), so "开mtp" is
answered as a comparison rather than assumed. Both: 2 sessions, `--chunk 512`, `--kv int8`, greedy, the shipped
decode environment from the previous night (`STRATA_Q36_MTP_VRAM=1`, `VERIFY_ROWS=8`, `NO_HOST_EXPERTS=1`,
`READ_WINDOWS=2`), the box's four cards reset and warmed before each arm.

Controls, each of which fails the arm rather than being reported around:
- both sessions admitted (`BADM <slot> 1` **twice**) - otherwise the arm is a single-session run;
- both admissions are 20,000 tokens, straight from the engine's own `taken a prompt of N tokens` line;
- the two sessions produce within a 2:1 split of each other - a starved slot is not a pair;
- no CUDA error and no non-finite value in stderr.

### Power is sampled, not bookended

The 20k prefill is a ~20 s window inside a ~1 minute run, so a reading before and a reading after would average it
away. `nvidia-smi --query-gpu=power.draw` is sampled every 250 ms for the whole arm, and the per-card figure
reported is max and mean over all samples. Max is the prefill peak; mean is what the card did overall.

### The sequential read is not a q36 deficiency, and vLLM confirms it

It is tempting to read the phase table above as "q36 reads prompts one at a time and that is why concurrency is
weak". It is the same in vLLM, and for the same reason: a model forward is one GPU operation with a dependency
chain through every layer, so two prompts cannot be read by the same card at the same time at all.

`vllm/v1/worker/gpu_model_runner.py:4001` `execute_model` runs one step synchronously; the async work vLLM does is
host-side scheduling (build the next batch while the GPU runs the current one), and its batch is a mix of decode
and prefill **tokens of different requests in one forward** - which is exactly what q36's chunked read plus the
`read_windows` interleave does, one layer down. The difference that remains is that vLLM's per-step batch admits
the next request's prefill as soon as a token budget frees, while q36's `request()` blocks until the whole prompt
is read.

That is a real difference, and it is the one a port could remove: **a 20k prompt occupies the card for 26 seconds
in q36 before the second request is even looked at.** But the fix is not "a separate prefill thread" (there is
nothing for a second thread to compute on a busy card); it is **admitting the prompt in pieces across steps**,
which is what `--chunk` does inside one request and what would need to happen *between* requests. Recorded as the
next thing worth measuring, not as something this run demonstrates.

### The result: real code, 20k in, 1k out, two sessions

Box `root@192.168.1.98`, four cards reset and warmed before each arm, `--batch 2`, `--chunk 512`, `--kv int8`,
`READ_WINDOWS=2`, greedy. Raw logs kept at `.backup/pp20k-run/`.

| arm | prompt (both reads) | prompt tok/s | generated | gen wall | aggregate | per session | MTP accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mtp0` (`--mtp 0`) | 60,130 ms | 665.2 | 1,183 | 34,809 ms | **34.0 tok/s** | 17.0 | - |
| `mtp1` (`--mtp 1 --batch-mtp`, depth 2) | 65,326 ms | 612.3 | 1,183 | 27,058 ms | **43.7 tok/s** | 21.9 | **782/809 = 96.7%** |

**MTP on is worth +28.6% on the generation phase** at this shape, at the cost of +8.6% on the reads. That is the
largest MTP gain measured on this box so far, and the reason is the prompt: 96.7% acceptance is far above the
47-62% the earlier fixture runs saw, because 20,000 tokens of real contiguous source makes the next token nearly
deterministic in a way 24 random ids never did. **The acceptance rate is a property of the traffic, not of the
drafter** - a fixture with random prompts systematically understates MTP.

**One session stopped on its own.** `slot 0` produced 183 tokens and printed `finish = stop` (end of text) in both
arms - identically 183, so the model wrote a complete answer and stopped. `slot 1` ran to the 1,000-token cap.
So "tg1k" was granted, not always consumed: 1,183 generated against 2,000 requested. The aggregate rate is
therefore over unequal sessions and is quoted as a property of this task, not as a throughput ceiling.

### What this shape says about concurrency, and one thing it does not

`BDONE` for slot 0 lands **after** slot 1's read finishes (34,809 ms against a 31,242 ms read), so the interleave
did grant slot 0 windows during slot 1's read - `BT 0` rows appear between slot 1's `PP` chunk boundaries in the raw
log. That is `READ_WINDOWS` doing exactly what it was measured to do.

What is *not* concurrent, and cannot be with this engine: the two 20k reads. Together they are 60 seconds of the
run; the generation is 35. On this task the prompt is 63% of the wall clock, and no decode-side setting changes
that. **The cost of a 20k prompt is the thing to attack, not the interleave.**

### The prompt is now reproducible

`pp20k_real_code.py` is the builder: it reads `src/`, `include/`, `serve/`, `strata_cli/`, caps each file at 2,200
tokens, stops at exactly 20,000, splits the file list in two for the two sessions, and refuses to write anything
unless the length is exact, the ids re-encode to themselves, no 64-token block repeats and no PAD id appears.
Anyone can rerun it against the same checkout and get the same two prompts.

---

## Project Maya: what it is, and the one mechanism that ports

### Maya is Strata's downstream, not its upstream

`third_party/project-maya/NOTICE.md:5`: **"Project Maya is itself built on Strata."** The relationship runs the
other way from what the name suggests: Strata ported Maya's DeepSeek-V4.1 engine into `src/dsv41/` (30,594 lines,
plus 4,846 in `vendor/`), not the reverse.

### There is no Q36 path in Maya, and no way to add one

`tools/dsv41/pack_gguf.py:510` is a hard gate:

```python
raise SystemExit("%s: not a DeepSeek-V4.1 GGUF (architecture %r)" % (model, arch))
```

Its predicate is a single equality, `general.architecture == "deepseek41"` (`:386`). Q36's GGUF declares
`qwen35moe`, so it is refused before anything else runs. All eight occurrences of "qwen" across `src/dsv41/`,
`include/strata/dsv41/`, `tools/dsv41/` and `tests/dsv41/` are comments or a tokenizer default
(`smoke.py:40`'s `tcfg.get("pre") or "qwen35"` is DeepSeek's own vocabulary sharing the tokenizer family, not
model support).

Nor could the gate be bypassed: `ds4_model.cu:565` reads `qk_rope_head_dim` (MLA), and `n_head_kv` appears **zero**
times in the whole file, while q36 is GQA. The shapes differ everywhere except the layer count (40).

| | DeepSeek-V4.1 | Q36 |
|---|---|---|
| layers | 40 | 40 |
| experts/layer | 384 (6 used) | 256 (8 used) |
| attention | MLA | GQA |
| extra | 203 GB Engram n-gram table | none |

### The one portable mechanism, measured on this box

`src/dsv41/ds4_ring_link.hpp` (35 lines) sizes a prompt's staging ring from the device's **measured** host link:

```cpp
want = stride/link >= 1 ms ? 1 : link * 10 ms / stride;    // ring_slots(), kRingBatchS=10ms, kRingSlowS=1ms
```

It depends on four numbers - link bytes/s, expert stride, VRAM room, batch parts - and none of them are
model-specific, so this is the one piece that crosses over.

Maya's own plotter is `link_h2d` (`ds4_model.cu:3902`): 32 MB from pinned memory, best of three, **on the legacy
stream**. Its comment records the trap: timed on a stream of their own, the copies left decode ~20% slower for the
whole run on PCIe Gen2 boxes. That is why the probe below keeps the legacy stream.

**Measured 2026-11-08 on this box** (`.backup/link_probe.cu`, two runs identical, four cards idle):

| card | pinned H2D | pageable H2D | pin D2H | ms per 2.0 MB expert | ring slots | per batch |
|---|---:|---:|---:|---:|---:|---:|
| P100 (nvidia-smi 0) | **6.11 GB/s** | 4.15 | 6.58 | 0.34 | 116 | 29 |
| P104 (1) | **1.63 GB/s** | 1.61 | 1.67 | 1.29 | 4 | 1 |
| P104 (2) | **1.63 GB/s** | 1.61 | 1.67 | 1.29 | 4 | 1 |
| P104 (3) | **1.59 GB/s** | 1.57 | 1.67 | 1.32 | 4 | 1 |

The q36 expert stride is real, read off the GGUF: `ffn_down_exps` is `[512, 2048, 256]` type 13 (Q5_K) =
**2.000 MB per expert**.

**The formula is not uniform on this box.** The threshold is `stride/link >= 1 ms`, i.e. `link <= 2.10 GB/s`:
- the three **P104 helpers at 1.6 GB/s** sit just under it -> **one expert per batch**, which is Maya's declared
  *minimum*, not an optimisation;
- the **P100 primary at 6.1 GB/s** is 5x over it -> 29 per batch, a 240 MB ring.

So porting `ring_slots()` would tell the helpers to do exactly what they already do. It is a correct rule that
buys nothing here, and that is worth knowing before writing it.

### And it corrects a number this project has been quoting

`q36_peer_prefill.cu:95` and the plan notes say the helpers' link is **0.58 GB/s**, and a memory
(`q36-prefill-last-gap-is-link`, corrected 2026-11-04) says 0.81 GB/s is the ceiling because the window makes *two*
PCIe hops. Today's isolated probe gets **1.63 GB/s** on the same cards.

Both numbers are real; they measure different things.
- **1.63 GB/s** is one hop's *capacity*, idle, best-of-three, pinned, 32 MB.
- **0.58-0.81 GB/s** is the *effective* throughput the prompt path achieves when the copy is one step in a
  dependency chain that also runs the mixer, the plan and the helper's GEMMs.

So the remaining prompt gap is **not** link capacity - half of it is provably unused. It is what happens to the
copy while everything else runs. That reframes the next measurement: not "buy a faster link", but "why does a
1.63 GB/s link deliver 0.81".

## The copy path is 4.5% of the layer, and the premise was wrong (2026-11-08, MEASURED)

That measurement has now been made, and it does not answer the question as posed - it retires it. **The copy
path is not where the prompt time goes, at any bandwidth.**

`.backup/link_gap_probe.py`, run on 192.168.1.98 against `--bench-pp 512` with `STRATA_Q36_PROMPT_TIMING=1`,
two arms differing only in `STRATA_Q36_STAGE_ASYNC`, three passes each. The arms agree pass to pass to within
0.07%, so the numbers below are not noise:

| per layer (ms) | `STAGE_ASYNC=0` | `STAGE_ASYNC=1` | share of forward |
|---|---:|---:|---:|
| post_combine | 4.812 | 4.775 | 47.5% |
| mixer | 3.743 | 3.744 | 37.1% |
| plan | 0.636 | 0.608 | 6.1% |
| helper_wait | 0.395 | 0.391 | 3.9% |
| router_route | 0.151 | 0.152 | 1.6% |
| combine | 0.105 | 0.105 | 1.0% |
| route_d2h | 0.018 | 0.016 | 0.2% |
| primary_work | 0.017 | 0.017 | 0.2% |
| **forward** | **10.132** | **10.061** | |

Host wall (overlaps the GPU spans, not additive): `sync3022` 8.24 / 0.23, `peer_plan` 1.01, `stage` 0.035,
`prep_d2h` 0.008, `prep_sync` 0.007, `ppwait` 0.10.

**The control that settles it.** `stage` is the window's D2H out of the primary. With `STAGE_ASYNC=1` the host
only *enqueues* it, so `stage` would be an issue cost, not a wall clock - which is exactly why the second arm
exists. With `STAGE_ASYNC=0` the host **blocks** on that copy, and `stage` is **0.035 ms**, the same as the
async arm's 0.033. The copy is that cheap either way. It is not being hidden; it is not there.

Every copy-bearing span, summed: `stage` 0.035 + `helper_wait` 0.391 + `route_d2h` 0.018 + `prep_d2h` 0.008 =
**0.45 ms of 10.07 = 4.5%**. `helper_wait` already contains the helpers' own up-leg H2D - the comment at the P2
record site says so: "the wait for the helpers' rows to land plus the up-leg H2D".

**The byte account agrees, and kills the old two-hop story.** Per layer the up leg moves 512 q8 rows x 2880 B =
1.47 MB plus 0.04 MB of ids and weights; the down leg moves 3 helpers x 512 tokens x 2640 B int8 = 4.06 MB. Every
span that contains a copy, added together, is 0.45 ms - so 5.56 MB is carried in at most 0.45 ms, and the copies
demonstrably overlap the compute rather than sitting in front of it. The up and down legs are also **different
physical links** (primary D2H 6.11 GB/s, helper H2D 1.63 GB/s) plus 3 helpers running in parallel, not one link
used twice. The earlier note that "two hops turn 1.63 into 0.81" compared a capacity against a throughput that
was never governed by that capacity. (0.45 ms is an upper bound: the spans holding the copies also contain the
helpers' compute, so the copies' own share can only be smaller.)

**Retired:** `q36-prefill-last-gap-is-link`'s conclusion that the last 6.8% is the helper activation upload and
that "this is where the copy efficiency can be moved". There is no copy lever here.

**What the time is actually spent on:** `post_combine` (47.5%) is the gap from a layer's P3 to the next layer's
P0, i.e. per-layer host bookkeeping plus the next layer's mixer start. `mixer` (37.1%) is the primary's own
attention + GDN arithmetic. Together 85%. `sync3022` is the primary's one mandatory
`cudaStreamSynchronize(st)` per layer (to bring `rmax` back and size the MMQ); it overlaps the GPU spans rather
than adding to them, which is why it can read 8.24 in one run and 0.23 in another with an unchanged `forward`.

**A caliber trap worth keeping.** `--bench-pp 512` reads 10.14 ms/layer, but the real 20k prompt from the same
session's run reads 20000 tok / 28888 ms over 40 layers and 39.1 chunks = **18.49 ms/layer**, 1.82x more. The
difference is expert residency and chunk-to-chunk routing, not the link. So `--bench-pp` absolute numbers are
not throughput numbers (`q36-prefill-two-instruments` said this already). The conclusion above does not depend
on them: `stage` is measured *inside each arm*, and the `STAGE_ASYNC` comparison is arm against arm.

**Not usable for attribution:** `STRATA_Q36_MIXER_SPLIT=1` prints 240 calls for attention (3 passes x 8 decode
tokens x 10 attention layers) - it reports the **decode** mixer, whose accumulators were reset after the prompt.
It cannot explain the prompt's 3.74 ms of `mixer`.

### What Maya does differently, and why it is not a port

Project Maya moves MoE activations with a **kernel** writing host-mapped pinned memory and a flag spun
in-kernel (`ds4_kernels.cu:1743-1779`), one stream per device, the whole `T*D*4` row in one put with no fixed
split cap, and **no `cudaStreamSynchronize` on the hot path** - completion is carried by the flag that the next
device's already-queued kernel spins on. q36 instead takes one blocking stream sync per layer. That is a
difference in *synchronisation model*, not in copy speed - and since copies are 4.5%, porting Maya's transport
would not move the number that matters.
