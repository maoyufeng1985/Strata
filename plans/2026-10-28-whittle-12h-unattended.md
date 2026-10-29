# Whittle: 12 hours unattended — the full optimization sweep

Start: 2026-10-28 20:30. Mode: unattended (no questions; anomalies recorded, run continues).
Branch: `rebase-foundation`. Box: 192.168.1.98, P100 (sm_60) + 3x P104 (sm_61), Whittle Q4_K_M.
Baseline at start (MEASURED, binary `ce487ed6bcd9`): prompt 275 tokens = 843 ms (flag off),
818 ms with the peer share (3 helpers, +36 % over 241.8 tok/s), decode 44.0 tok/s = 22.6 ms/token.

## The user's list, as work items

| # | Request | State at start | Work |
|---|---|---|---|
| 1 | 逐行阅读 flash-next 能用的都用上 | partial (the Q36 knob sweep done) | a systematic line-by-line audit of the prompt and decode hot paths |
| 2 | 同步 Q36 已实现的 | HELPER_PARALLEL's split = this tree's `stripe` (already default); ASYNC + MARSHAL missing | port ASYNC/MARSHAL to `expert_source.cpp` |
| 3 | MTP 要实现 | the engine has MtpDrafter; **Whittle has no MTP weights** | decide the weights question, then implement |
| 4 | prefill 主卡和 3 helper 并行 | done for the expert rows (+36 %); the primary's own work is now the bottleneck | parallelize the non-expert prompt phases |
| 5 | decode 也要并行 | helpers are dispatched serially on the host thread | ASYNC + MARSHAL (item 2) |
| 6 | 优化跨卡传输和通信 | host route 6.1 GB/s, remote-opt already collapses the return | measure and shrink what is left |
| 7 | prefill 的 kv 计算做流式传输 | KV stream exists (`m.kv_copy`, prefetch); peer KV not streamed to helpers | investigate KV streaming for the prompt |
| 8 | 精简内存的专家占用 | expert cache 18.6 GiB, primary 4765 slots | shrink residency without losing the hit rate |
| 9 | 发挥想象力 | — | the audit will surface what the list does not name |

## Rules for this run

- One engine at a time on the box; every bench arm takes the box alone and prints the binary md5.
- Default path stays byte-identical; every new path is opt-in behind an env var.
- No claim without a measurement; every number says what it was measured on.
- A failed idea is recorded with its number and dropped, not retried.
- Commits: one problem per commit, area prefix, validation split (passed / failed / not run).

## Where the run stands (read this first)

**Decode, best configuration found:** `--expert-cache-device1/2` (two helpers) + `STRATA_NV_DEFAULTS_PRIMARY=1`
+ `STRATA_QFUSE=1`. Measured against the table-off 3-helper baseline, decode goes **5352.4 → 5248.7 ms per
256 tokens (−1.94 %)** in the round that tested the last two together (`R-tableab.sh`, binary `315469d836bf`).
The 2-helper layout adds another −0.8 % on top of that, measured separately. Nothing here is a default yet.

**Prompt:** `STRATA_DENSE_F16_ACC=1` remains the one prompt win (−5.0 %, 842 → 800 ms) — **but it is not
bit-exact**. `src/prefill/gemm.cu:881-889` gates it to `60 <= cc < 70` (Pascal only), `:900-937` restricts it
to the cuBLAS algorithms whose error stays at 0.36 %, and the comment states the price plainly: output is
FP16 because cuBLAS takes no FP32 output with FP16 compute, **0.36-0.38 % rms** against fp32-accumulate.
So it buys 5 % prompt speed for a small, real precision loss. That is a trade to make deliberately, not a
free win, and it must be said that way.

**Measured and rejected, with numbers, do not retry:**
| idea | result |
|---|---|
| DFlash2 drafter (`incoai/Qwen3.6-35B-A3B-DFlash2`) | loads and runs; 25.90 vs 32.46 tok/s — slower |
| MTP with Qwen3.6's head, `rt3`, `--spec 4` | acceptance 8.6-9.4 %, 21.97 vs 32.46 tok/s — slower |
| `--mtp-hnorm stream` | acceptance neutral (8.6/9.0 vs 9.0/9.4); the 82→86 % upstream number does not transfer |
| `STRATA_REMOTE_HELPER_PARALLEL=1` | −3.8 % after the persistent-pool rewrite; the rest is wake/join cost |
| `STRATA_REMOTE_FINISH_PAR=1` | +0.17 %, i.e. neutral — per-card streams already overlap |
| `STRATA_VERIFY_DEVICE_PLAN=1` | +2.13 %, twice, agreeing to 0.01 % — a certain loss |
| `STRATA_ADAPT_NOWAIT` / `STRATA_ADAPT_LAG=2` alone | −0.1 % / +0.2 %, sign flips — they only matter inside the table |
| `FUSE_DECODE`, `B6_MMVQ_ROWS`, `GR_UP75` (drop-one from the table) | all within 0.11 % of the table, signs disagree — dead weight on this box |

### 2026-10-29 — the line-by-line audit, re-triaged by provenance

The read-only source audit (8 module agents + synthesizer, 44 candidate points) is re-triaged here because
several of the agents marked "measured" what was actually **upstream's number on a different card**.
Graded by provenance:

**A — upstream explicitly measured on a P100, positive, but this box's config is gated out. Test these first.**
| switch | upstream evidence | this box |
|---|---|---|
| `STRATA_ADAPT_OVERLAP=1` (+ `ADAPT_LAG=1 ADAPT_NOWAIT=0`) | `DETAILS.md:1010` — P100 **+4 to +8 %** | needs the whole arena in RAM + one GPU; the 3-helper MMQ config may not qualify — **must test** |
| `STRATA_ADAPT_LAG=2`, `SIDE_MIXER=2`, `GR_UP75=1`, `B6_MMVQ_ROWS=1`, `FUSE_WAIT=2`, `FUSE_DECODE=1` | the sm_60 table, `nv_defaults.cpp:36-48` (P100 +17.8 %) | gated off on the mixed fleet; the **whole table already measured −5.02 %** (round `_r2.log`), so these are known — only the drop-one split is open |

**B — 0.2.7's Pascal opt-in family, unmeasured here. `release-notes-0.2.7.md:53` says their speed numbers are the
authors', not ours.** Test, do not assume:
`STRATA_DOWN_WS=1`, `STRATA_MTP_Q8_SM60=1`, `STRATA_MMQ_WS=1`, `STRATA_PATTN_SM60=1`, `STRATA_EXPERT_NIBBLE=1`.
All compile in (`STRATA_EXPERIMENTAL_SM60=ON`), all need an A/B.

**C — mechanism-plausible, derivation-only (the agent's "measured X ms" is unsupported).** Worth a probe only
after A and B: the PLE compact copy-stream (`prefill.cpp:157`), the GDN recurrence kernel, `ps_frac`
residency allocation (`prefill.cpp:3539`), prompt chunk-size retuning (`prefill.cpp:3131`), offloading the
dense GDN/QSA projections to a helper (`prefill.cpp:3933,3977`), and the Q5_K native-head path
(`generate.cpp:6394,7939`).

**The Q5_K head (the "biggest per-token cost") was probed this session and the lever is smaller than the
docs' 10 % framing suggests.** Three facts, each read from code or a run:
1. It is **already on by default**: `--native "$M"` sets `native_head_gguf` (`generate.cpp:3851`), and the
   startup log reads `experimental native Q5_K head, 417177600 bytes` in every combo arm — no switch to flip.
   `STRATA_Q8_SM60` does not touch it: Whittle's `output.weight` is **Q5_K** (ggml type 13), and that switch
   only packs the Q8_0 path (`native_head.cpp:78-79`).
2. **There is no off switch**, so a clean on/off A/B cannot be built from the tree.
3. **The dedicated GPU-floor measurement does not run on Whittle**: `--gpu-only-full`
   (`generate.cpp:3058`) replays per-layer `pre`+`post` graphs, but a native/IQ pack captures **no post
   graphs** (`generate.cpp:8083`), so it exits `session_replay_full: not captured with post graphs` (rc 1).
   Whittle's head therefore cannot be isolated on its own with the measure flags.

The only reproducible head number is the **share of the per-request decode** in the serve log: 22.5-23.1
ms/token total (combo `DONE` lines: 2884-2962 ms over 128 tokens) against `WHITTLE45B.md:175`'s single-row
breakdown of **2.23 ms head of 22.58 ms = ~9.9 %**. Two independent sources agree on ~10 %. What is left
that actually moves that 10 %: quantising the 417 MB head matrix to a lower native format (Q4_0/Q2) — but
that changes the logits, so it is a quality item, not a pure speed win, and is out of scope for the
"make it faster without changing the answer" pass. **C-grade verdict: the Q5_K native head is already
optimal-as-shipped on this model; the remaining lever is a quant change, parked.**

### 2026-10-29 — C-grade PLE compact and Q8_SM60, both measured zero on this box

`R-combo2.sh`, post-rebase binary `5f63a48e`, 2-helper + `--remote-expert-opt` base, one process per
config, 3 GEN batches (batch 1 warms, batches 2-3 timed). 275-token prompt, 128 tokens out:

| arm | timed_pp (ms) | timed_dec (ms, 128 tok) | verdict |
|---|---:|---:|---|
| base | 1088.5 | 2651.95 | ref |
| `STRATA_PLE_COMPACT=1` | 1090.5 | 2652.3 | +0.01 % — **0** |
| `+ STRATA_PLE_COPY_STREAM=1` | 1090.5 | 2659.1 | noise — **0** |
| `STRATA_Q8_SM60=1` | 1092.5 | 2656.4 | +0.17 % — **0** |
| `Q8_SM60 + PLE_COMPACT` | 1093.5 | 2651.7 | noise — **0** |

Two points closed as measured-zero:
1. **`STRATA_PLE_COMPACT` / `PLE_COPY_STREAM` (7x fewer PLE bytes over PCIe):** the box runs
   `--ple-io ram` (the 39 M-row table locked in RAM, `WHITTLE45B.md:79` — PLE is 0.04 ms/token, off the
   critical path), so the PLE rows never ride PCIe and the "7x fewer bytes" lever has nothing to act on.
   The prediction (0) held: +0.01 % prompt, 0.01 % decode.
2. **`STRATA_Q8_SM60` (GP100 packed Q8_0 expert GEMV):** Whittle's experts are **Q4_K_M on the MMQ
   path**, not Q8_0 GEMV, so the packed Q8_0 kernel is not dispatched. +0.17 % is within the session's
   noise floor. The Q36-era "+5.3 % Q8_SM60" number does not transfer — different expert quant, different
   engine path.

With these two and the Q5_K head, the three remaining C-grade points that had real switches or a clean
probe are all exhausted on this box. The only C-grade item left is the **GDN recurrence kernel**
(`prefill.cpp:3952`), which upstream already measured within-noise (`WHITTLE45B.md:165`, `GDN_CHUNKED`
+1.3 %/+1.0 %), so it is not worth a dedicated run either.

**D — demoted or rejected.** `STRATA_PF_PAD` (gfx11-only, `prefill.cpp:529`); `STRATA_HANDOFF_PIPE`
(`prefill.cpp:420`, only fires with a layer split + hand-off, single-card prefill gets nothing, and its
design assumes separate PCIe links); the 512 MiB expert-cache reserve and slot-alignment items
(`remote_experts.cpp:267`, `peer_experts.cpp:202`) — VRAM-only, no speed lever; the `STRATA_GDN_CHUNKED`
item — already measured within-noise on this box.

**Box note:** the new post-rebase binary is `5f63a48e0ac1` (v0.1.53), built 08:31; re-measure any number
against it, not against the earlier `315469d836bf`.

### 2026-10-29 — combined one-load session: the best config is `QFUSE + sm_60-table + ADAPT_LAG/NOWAIT` on 2 helpers

`R-combo.sh`, post-rebase binary `5f63a48e`, 275-token prompt, 256 tokens out. Each config = one
`strata` process; the process fires 3 GEN batches and only batches 2-3 are timed, so the 18.16 GiB
expert arena and the 1217-expert prefetch are loaded **once per config**, not per batch. Six configs:

| arm | timed_pp (ms) | timed_dec (ms, 256 tok) | dec vs `h2` |
|---|---:|---:|---:|
| `h3` (3 helpers) | 1094.5 | 2664.1 | +0.16 % |
| **`h2` (2 helpers, `--remote-expert-opt`)** | **1089** | **2668.35** | base |
| `h2_nopref` (2 helpers, no `--remote-expert-opt`) | 1011 | 2707.25 | +1.45 % |
| `h2_qfuse` (`+ STRATA_QFUSE`) | 1095.5 | 2652.5 | **−0.60 %** |
| `h2_table` (`+ STRATA_NV_DEFAULTS_PRIMARY=1`) | 1090 | 2635.8 | **−1.22 %** |
| **`h2_all` (`QFUSE + table + ADAPT_LAG=2 NOWAIT=1`)** | 1096.5 | **2585.5** | **−3.10 %** |

Three findings, each now confirmed on the post-rebase binary in one coherent session:

1. **Two helpers beat three.** `h3` is 0.16 % *slower* on decode and 0.5 % slower on the timed prompt.
   The "2 is the knee" conclusion from 2026-10-28 holds on `5f63a48e`.
2. **`--remote-expert-opt` pays: +1.45 % decode.** Dropping it (`h2_nopref`) is the only arm slower than
   the base in decode. It also makes the warm prompt faster (1011 vs 1089-1096 ms) but that is the
   first batch, which carries the prefetch — not a signal.
3. **The best decode config is `h2_all`: `STRATA_QFUSE=1` + `STRATA_NV_DEFAULTS_PRIMARY=1` +
   `ADAPT_LAG=2 ADAPT_NOWAIT=1` on two helpers with `--remote-expert-opt` → 2585.5 ms, −3.10 % over the
   plain `h2` (47.7 → 49.1 tok/s).** The sm_60 table alone (`h2_table`, −1.22 %) beats `QFUSE` alone
   (−0.60 %), so the table is the bigger contributor and `QFUSE` stacks on top of it. This is the most
   solid decode number of the run: one session, hot batches, same binary, all arms against one base.

**Caveat, recorded per the run's own rule:** these timed numbers are the mean of *hot* batches 2-3 inside
one process, so the expert arena and graphs are warm; they are not directly comparable to the earlier
cold single-batch numbers (53xx ms decode on `315469d8`). The −3.10 % is a same-session, same-binary,
same-batch-state result and is the one to quote for "best Whittle config"; absolute tok/s across
different harnesses must not be mixed.

**Two "upstream says wins" switches were refuted on this box this session:**
- `STRATA_ADAPT_OVERLAP` — the startup log prints `adaptive overlap: off: not with --remote-expert-opt`;
  it does not engage under the 3-helper + remote-opt config, and its two rounds went +1.1 % / −0.8 %
  (noise). The P100 "+4..8 %" upstream figure does not transfer here.
- `STRATA_PATTN_SM60=1` — **crashes**: `dequant launch: an illegal memory access` in the prompt's
  QSA attention, zero `PP`/`DONE` lines, `rc=1`. Reproducible across two attempts. Recorded as a hard
  defect on this build, not a win.

## Log

**Two harness defects found that invalidate earlier claims:**
1. **Serve-mode token streams are not reproducible on this box** under `--expert-cache auto` (base_1 vs base_2
   diverge at token 18 with `ram_reads` 60862 vs 61602). `tok_md5` proves only the first ~17 tokens match and
   is **not** a validity check for any serve-mode arm in this run. This also explains part of the ±1.3 % spread.
2. **The box's error bar is not uniform.** Same-arm repeats varied from 0.01 % (`no_side`: 5374.2/5374.6) to
   1.3 % (`table`: 5351.0/5283.4). Only same-direction repeats across rounds plus a mechanism count as signal.

**One claim was wrong and is corrected:** an earlier entry here said the sm_60 default table was
**−1.1 %** on this box. It is not; the live A/B reads **−0.74 %** (three same-direction rounds), and the
`_r2` 5 % reading came from a configuration whose own baseline was 8 % slower. Both numbers are in the log;
the −0.74 % is the one for this box.

**Open, not done:** `QFUSE` is measured ≈−1 % five times but `gr_parity` is **red** on this box with and
without it (the q8_1 region is never written; `0x5a` poison bytes survive). It stays opt-in until that test
is green. Prefill's non-expert phases are still unaudited beyond the phase table; the two candidate
offloads (o_proj, shared expert) have no measurement.

### 2026-10-29 07:40-08:00 — 200k context: it works, and the prompt rate barely decays

Asked directly for "200k context, pp32k, tg200". The KV arithmetic that motivated the test: Whittle's QSA
layers are `l % 4 == 3`, so 10 of 40 layers carry KV; `kv_q8_bytes_per_cell` (`include/strata/kernels/kv_q8.hpp:25`)
is `8*128*2 + 8*2*2*2 = 2112` B/cell/layer, so **2112 × 10 × 200000 = 3.93 GiB** of KV plus the RoPE tables.
With the default 8.48 GiB expert arena that does not fit a 12 GB card, so the test ran two cache sizes.
Binary `315469d836bf`, `--max-context 200000 --kv int8`, `--prefill 512`, 32768-token prompt (the 275-token
bench line repeated), 200 tokens out, with peer prefill + `DENSE_F16_ACC` + the sm_60 table on.

| arm | prefill 32k | decode 200 tok | notes |
|---|---:|---:|---|
| `--expert-cache 0` | 78080 ms → **419.7 tok/s** | 4543.5 ms → **22.7 ms/tok, 44.0 tok/s** | experts all streamed, 6092 streamed |
| `--expert-cache 2000` | 84197 ms → **389.2 tok/s** | 4594.5 ms → **23.0 ms/tok, 43.5 tok/s** | 9278 streamed — the *larger* cache streams **more** |

**Both run, neither errors, and 200k context is usable.** The counter-intuitive part is that the bigger
expert cache is 7 % slower on the prompt and streams 52 % more experts; on a 12 GB card at 200k context the
KV is what should have the VRAM, so `--expert-cache 0` is the better arm here.

**Prompt throughput decays very little with length** — the engine prints a `PP` line per chunk, so the whole
curve is visible in one run (arm `ec0`):

| tokens processed | 512 | 4096 | 8192 | 16384 | 24576 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| tok/s | 477.4 | 434.4 | 430.2 | 425.8 | 422.2 | 419.7 |

**−12 % from 512 to 32k.** Whatever the KV and QSA attention cost at 200k, it has not reached these lengths.
That makes the earlier 275-token figure (348 tok/s) a *short-prompt* number, not the box's prompt ceiling:
the same configuration does **419.7 tok/s at 32k**.

**Correction to something I said in this session:** I described 419.7 tok/s as "far below" the 348 tok/s of the
275-token bench. That was backwards — 419.7 > 348, and the direction is the expected one (fixed per-prompt
costs amortise over more tokens). The number is recorded correctly above; the reading was wrong when first
spoken.

## Log

### 2026-10-29 13:22-13:27 — aclass-trim reconfirm; box idle; hold

Re-ran `R-aclass-trim.sh` after FINAL1000 idle gap (prior 13:14 run already refuted mtpq8).
Same verdict:

| arm | prompt ms | decode ms |
|---|---:|---:|
| `base_1` | 1,138 | 5,351.1 |
| `mtpq8_1` | 1,160 | 5,344.0 |
| `base_2` | 1,151 | 5,368.3 |
| `mtpq8_2` | 1,135 | 5,359.0 |

Mean decode base 5,360 / mtpq8 5,352 (−0.1 %, noise). `mtp_max=0`, drafts 0/0.

Box idle after exit. **Hold — no further engine from this agent.**

### 2026-10-29 13:18-13:21 — RAM0 settled

**RAM0 settled** (`--mmap-experts --ple-io direct`, auto expert-cache, compress on, N=2):

| PP | ms | tok/s |
|---|---:|---:|
| cold (29999/30000) | 48,842 | **614.2** |
| warm reuse | 22,567 | 603.3 |

`arena_mib=0` (no 18 GiB pinned expert arena). Prefill **−3.2 %** vs baseline ~634. File-tier
path: page-cached GGUF (`UNBUFFERED_LOAD` ignored without `--resident-budget-gib`). Cold GEN1
decode waited **24.0 s** in 12802 file fetches (OS read 18.4 GB); warm GEN2 wait **1.3 s**.
VRAM warn 418 MiB free (#2065).

**Verdict:** mmap saves host RAM arena, costs ~3 % cold 30k prefill + first-request I/O hit.
Keep for RAM-starved serves; not a path to 1000 tok/s.

### 2026-10-29 13:14-13:18 — ACLASS-TRIM confirmed; `MTP_Q8_SM60` no-op

**aclass-trim** (binary `9893029d`, GEN 256 / 275-tok prompt, `--prefill 512`
`--max-context 4096 --spec 2 --suffix-draft 0`, 2 repeats):

| arm | prompt ms | decode ms | tok/s | drafts |
|---|---:|---:|---:|---:|
| `base_1` | 1,142 | 5,353.9 | **47.8** | 0/0 |
| `mtpq8_1` (`STRATA_MTP_Q8_SM60=1`) | 1,147 | 5,359.1 | **47.8** | 0/0 |
| `base_2` | 1,138 | 5,355.4 | **47.8** | 0/0 |
| `mtpq8_2` | 1,123 | 5,374.6 | 47.6 | 0/0 |

**Refuted:** `STRATA_MTP_Q8_SM60=1` on this pack/config. All arms `drafts accepted 0 of 0`
+ `mtp_max=0` — flag never engages (`--suffix-draft 0`). Skip further mtpq8 on Q4_K_M.

**Reconfirm 13:28–13:33** (same binary `9893029d`, same script): base 5373.9 / 5379.1,
mtpq8 5370.3 / 5344.0 ms; all `mtp_max=0`. Noise-equal; MTP still no-op. Box idle after.

**Prefill settled ~634–636 tok/s** (FINAL30K / PLE / VRAM / HDGIB0 / COLD base).

### 2026-10-29 13:03-13:06 — FINAL1000 settled; `--expert-cache 0` ignored

**FINAL1000 settled** (intended: `--expert-cache 0 --ple-io direct` + `HEADROOM_GIB=0` +
`UNBUFFERED_LOAD=1`, N=2). Actual load: **expert-cache auto → 4581 primary slots** — the
`0` CLI did **not** disable main-card experts (same packing as HDGIB0). Prefill matches
baseline:

| PP | ms | tok/s |
|---|---:|---:|
| cold (29999/30000) | 47,450 | **632.2** |
| warm reuse | 22,032 | 618.0 |

VRAM warn 422 MiB free (#2065). `ple-io direct` + unbuffered load: no prefill gain vs ram/
auto. To actually empty primary cache, need a different flag path (verify CLI parse /
`expert-cache 0` vs `none`).

**COLD1000** (partial; no `== COLD1000 DONE ==`):

| arm | envs | cold ms | cold tok/s |
|---|---|---:|---:|
| `cold_base` | none | 47,267 | **634.7** |
| `cold_warm` | `PREFILL_WARM=1` | 46,870 | **640.0** |
| `cold_warm_io` | WARM + IO_PF_SCHED/AHEAD | **invalid** | PP only at 14469 tok |

`cold_warm` ≈ +0.8 % (noise). `cold_warm_io` prompt truncated — discard.

Box idle after FINAL1000. **Next (one engine):** `aclass-trim` base+mtpq8 only.

### 2026-10-29 12:58-13:04 — COLD1000 partial; FINAL1000 owned box

See above for settled numbers. `aclass-trim` was uploaded earlier, not launched then.

### 2026-10-29 12:42-12:51 — HDGIB0 settled; box idle

**HDGIB0 settled** (`STRATA_RESIDENT_HEADROOM_GIB=0`, `--ple-io ram`, expert-cache auto,
cap16384, compress on, N=3):

| PP | ms | tok/s |
|---|---:|---:|
| cold (29999/30000) | 47,172 | **636.0** |
| warm reuse #1 | 21,945 | 620.4 |
| warm reuse #2 | 21,955 | 620.1 |

Prefill matches FINAL30K / PLE / VRAM (~634–636). `HEADROOM_GIB=0` packs primary cache to
4581 slots / 8.16 GiB → `vram_free_mib=568` at load, **422 MiB free at request start**
(WARNING #2065). Prefill untouched; decode risk if driver pages. Keep default headroom
(or `--vram-reserve-mib 832`) for serve configs.

Box idle after exit (`strata=0`, GPU mem 0). Skip STEADY `cmp_on` — compress-on cold settled.
**Next (one engine, if still idle):** optional `mtpq8` A/B only.

### 2026-10-29 12:37-12:42 — VRAM30K settled; HDGIB0 owned box

**VRAM30K `ple_direct` settled** (auto expert-cache, `--ple-io direct`, compress on):

| arm | cold ms | cold tok/s | warm tok/s |
|---|---:|---:|---:|
| `ple_direct` | 47,227 | **635.2** | 620.3 |

Matches FINAL30K / PLE030K `ple_off` (634.7). `ple-io direct` at auto cache = free host
RAM (~5.6 GiB) with **no prefill loss** vs ram. Earlier MEMSWEEP e1500_direct looked slower
only because expert-cache=1500 collapsed `prefill_cap` to 4096.

**PLE030K `ple_off`:** cold 634.7 / warm 622.0 (same config, ram PLE).

### 2026-10-29 12:15-12:39 — MEMSWEEP30K done; STEADY cmp_on missing; box owned by VRAM30K

`R-vram30k.sh` owned the box then; now done (see above). `RB030K` SKIP'd on collision.

**STEADY30K2 incomplete** (`pickup-20261028/STEADY30K2/`): only `cmp_off` finished;
script died before `cmp_on`. Cold full PP without COMPRESS (3h + F16_ACC, ch32768):

| arm | cold ms | cold tok/s | warm reuse tok/s |
|---|---:|---:|---:|
| `cmp_off` | 55,997 | 535.7 | ~525 |

**MEMSWEEP30K settled** (binary `9893029d`, always `DENSE_F16_ACC` + `PF_PEER_COMPRESS` +
3 helpers; vary `--expert-cache` + `--ple-io`). Cold / warm final `PP` (29999/30000):

| arm | expert-cache | ple-io | prefill_cap | cold ms | cold tok/s | warm tok/s | vram_free |
|---|---:|---|---:|---:|---:|---:|---:|
| `e1500_ram` | 1500 | ram | 4096 | 55,605 | 539.5 | 526.9 | 6008 |
| `e1500_direct` | 1500 | direct | 4096 | 56,331 | 532.5 | 521.0 | 6008 |
| `e2500_direct` | 2500 | direct | 8192 | 51,532 | **582.1** | 569.9 | 4058 |

No crash / OOM on any arm. `ple-io direct` vs `ram` at e1500: −1.3 % (noise / SSD read),
saves ~5.6 GiB host RAM. Shrinking expert-cache to 1500 collapses `prefill_cap` to 4096 and
costs ~15 % vs FINAL30K's auto-cache + cap16384 path (**633 tok/s**). e2500 recovers some
(cap8192 → 582) but still −8 % vs auto.

`e2500_direct` was missing after the original MEMSWEEP driver died mid-run; relaunched once
with a request-file LF script (`/tmp/e2500_direct.sh`) after a CRLF upload failed and a
pipeline redirect leaked GEN tokens into the launch log.

**PLE030K `ple_off`** (auto cache, no PLE flag change): cold **634.7** / warm 622.0 — matches
FINAL30K. **VRAM30K `ple_direct`** in flight: cold PP so far 635.2.

**Next when VRAM30K exits (one engine):** finish STEADY `cmp_on` A/B (or skip — FINAL30K +
PLE already settle compress-on cold ~634); optional `mtpq8` only.

### 2026-10-29 12:08-12:18 — box back; FINAL30K settled; B-class no-fire on Q4_K_M

SSH recovered after reboot (`up ~7–9 min`).

**FINAL30K settled** (`pickup-20261028/FINAL30K/`, binary `9893029d`, 3 helpers +
`DENSE_F16_ACC` + peer compress). Cold full-prompt PP (second `PP` after `RESUME 0`,
29999/30000):

| run | ms | tok/s |
|---|---:|---:|
| r1 | 47,364 | 633.4 |
| r2 | 47,507 | 631.5 |
| r3 | 47,294 | **634.3** |

**Settled cold 30k prefill: median 633.4 tok/s** (mean 633.1). Warm reuse after
`RESUME 16384` ~618–621 tok/s.

**B-class switches vs Whittle Q4_K_M — source audit, no fire (do not spend box time):**
- `STRATA_EXPERT_NIBBLE` — only IQ2_S / IQ3_XXS / IQ3_S (`expert_nibble.hpp`)
- `STRATA_DOWN_WS` — only IQ4_NL / Q2_0 down at `n_ff==640` (`iq_kernels.cu:down_ws_on`)
- `STRATA_MMQ_WS` — gate/up IQ2_* / IQ3_* and down Q2_0 / IQ4_NL (`moe_ws.cu:supported`)
- Whittle pack is **Q4_K_M** → all three silent no-ops
- Still measurable: `STRATA_MTP_Q8_SM60=1`. Skip: `PATTN_SM60` (crash); `ADAPT_OVERLAP` (refuted)

`.run/R-aclass-trim.sh` trimmed to `base` + `mtpq8` only.

### 2026-10-29 11:32-11:50 — box SSH dead (ping OK, tcp22 False)

Anomaly recorded (resolved by reboot ~12:08). From the Windows host:
- ping 192.168.1.98 replies (1 ms TTL=64)
- TCP 22 opened then stopped accepting (`tcp22=False` / plink timeout / 15–25 s no banner)
- Classic OOM / sshd hung pattern (2026-10-18/20); many stale local `plink` killed

### 2026-10-28 20:30-21:00 — what the box settled before the run

**Won: `STRATA_DENSE_F16_ACC=1`, the prompt's only real win.** 3 interleaved rounds,
`base 842/843/842 ms` vs `+DENSE_F16_ACC 800/807/785 ms` → **median 842 → 800 ms (-5.0 %)**;
the combined arm (all four Q36 knobs) was 801/802/800, i.e. DENSE_F16_ACC is the whole gain and
nothing else adds to it. Decode unchanged (5791/5792/5814 vs 5833/5784/5785 ms).

**The sm_60 per-arch default table: gated OFF on this box, an earlier "-1.1 %" was mis-recorded, and the
real number is still OPEN.** Three readings exist and they disagree, so all three are recorded:

| source | reading | comparable? |
|---|---:|---|
| earlier note in this log | −1.1 % | no — ran under a competing build |
| `.run/_r2.log`, 3 interleaved rounds, `gate*_def`→`gate*_prim` | **−5.02 %** (median) | **no** — different binary and it carried `STRATA_PREFILL_REMOTE_PEERS=1`; its own `def` baseline read 5799-5897 ms against this batch's 5368 ms, i.e. 8 % slower overall |
| `flagsweep.log` base (table off) vs `flagsweep3.log` `table_1` (table on) | −0.51 % | no — different batches, and flagsweep3's own arms have not finished |

The `_r2` detail, for the record (3 rounds, and a `hand*_sm60` arm that sets the same seven switches by
hand rather than through the gate — it lands on the same numbers, so that batch *was* measuring the
seven switches and not the gate):

| round | `def` | `prim` | `hand` |
|---|---:|---:|---:|
| 1 | 5799.3 | 5512.1 | 5527.9 |
| 2 | 5897.8 | 5502.9 | 5525.2 |
| 3 | 5803.3 | 5518.5 | 5502.4 |

**Why that 5 % could not simply be quoted, and what the live A/B said.** Its baseline is 8 % slower than
the same configuration today (`--remote-expert-opt` was not on those arms, and the binary differs), so the
5 % was a *relative* gain inside a configuration this box no longer runs. `R-tableab.sh` settled it —
`off` / `on` / `on+QFUSE`, 3 interleaved rounds, same binary `315469d836bf`, same flags:

| round | `off` | `on` | `on+QFUSE` |
|---|---:|---:|---:|
| 1 | 5352.4 | 5263.9 | 5270.5 |
| 2 | 5386.1 | 5313.0 | 5248.7 |
| 3 | 5348.7 | 5312.9 | 5240.8 |
| **median** | **5352.4** | **5313.0** | **5248.7** |

- **The table is worth −0.74 % here, not −5 %.** Three rounds, three same-direction wins, so it is real,
  but small. The `_r2` 5 % does not transfer: it was measured in a configuration with a much slower
  baseline (5799 ms against 5352 ms), and whatever it recovered has already been recovered here by other
  means. `STRATA_NV_DEFAULTS_PRIMARY=1` stays opt-in; on this box it is worth having but not worth a
  default change on its own.
- **`QFUSE` on top of the table: −1.21 % median** (5270.5/5248.7/5240.8 against 5263.9/5313.0/5312.9 —
  two of three rounds win, the third loses by 0.13 %).
- **`QFUSE` across all five measurements: −1.03 %, −1.33 %, −1.21 %.** Every batch that tested it, on two
  different baselines, put it at about −1 %. That is the most reproducible decode result in this run.

**What the gate does, and why it exists.** `generate.cpp:3389-3435`: on a mixed fleet (sm_60 primary +
sm_61 helpers) the table is **not applied**, because the switches are all read from one process-wide
environment — a table picked for the P100 would also drive the P104s. The comment records the earlier
reading (`measured: 0 firings in every log on 192.168.1.98, where it is worth +17.8 % on the P100 alone`,
IQ3_XXS, bare P100) and the opt-in `STRATA_NV_DEFAULTS_PRIMARY=1`. `docs/DETAILS.md:1198-1207` documents the
table and states the identity gate (10 prompts plus `exact_pp` 4K/20K, fixed flags) is **identical** with the
sets on — i.e. the whole table is claimed bit-exact.

**One consequence is certain regardless of how the A/B lands:** every arm of `R-flagsweep.sh` ran with the
table **off** (confirmed: its `base_1.err` has zero `exact speed switches` lines). So five of the seven
switches — `FUSE_WAIT`, `FUSE_DECODE`, `B6_MMVQ_ROWS`, `GR_UP75`, and `SIDE_MIXER` at its table value `2`
— were never exercised in that sweep, and its `SIDE_MIXER=1` arm was a **0→1** test against a baseline
where the table's `=2` had never been applied. `R-flagsweep3.sh` runs the drop-one sweep **on top of the
table** to answer that.

### 2026-10-28 23:30-23:50 — drop-one on top of the table: one switch pays, five are dead weight, `QFUSE` is the real find

`R-flagsweep3.sh`, 7 arms × 2 rounds, all with `STRATA_NV_DEFAULTS_PRIMARY=1` (the table on), same binary
`315469d836bf`, 256-token generation, 275-token prompt. The method is drop-one — each arm turns exactly one
of the table's switches to `0` — which is the method the table's own comment cites.

| arm | round 1 | round 2 | mean | vs table |
|---|---:|---:|---:|---:|
| **table (all 7 on)** | 5351.0 | 5283.4 | 5317.2 | — |
| `SIDE_MIXER=0` | 5374.2 | 5374.6 | **5374.4** | **+1.08 %** |
| `FUSE_WAIT=0` | 5308.0 | 5293.6 | 5300.8 | −0.31 % |
| `FUSE_DECODE=0` | 5326.2 | 5317.8 | 5322.0 | +0.09 % |
| `B6_MMVQ_ROWS=0` | 5367.0 | 5276.5 | 5321.8 | +0.09 % |
| `GR_UP75=0` | 5283.5 | 5338.8 | 5311.2 | −0.11 % |
| **`+STRATA_QFUSE=1`** | **5250.3** | **5242.5** | **5246.4** | **−1.33 %** |

1. **`SIDE_MIXER=2` is the one table switch that measurably pays: +1.08 %, and its two rounds agree to
   0.01 % (5374.2 / 5374.6).** That is the tightest repeat in any arm of this run, and it is a *drop-one*
   loss, so the direction is unambiguous. It also explains the earlier sweep's `SIDE_MIXER=1` arm: on a
   table-off baseline, `1` beat `0` by ~1.2 %; on a table-on baseline, `0` loses to `2` by 1.08 %. Both
   readings say the same thing — **the side branch is worth ~1 % here either way**.
2. **`FUSE_WAIT=0` is 0.31 % faster in both rounds** — small, and the opposite sign from its `-1.6` table
   note. On this box its "fold both waits" mode appears to cost slightly more than it saves. Not worth a
   default change on 0.31 % with this box's spread; recorded, left alone.
3. **`FUSE_DECODE`, `B6_MMVQ_ROWS` and `GR_UP75` are dead weight here: their means sit within 0.11 % of the
   table and their two rounds disagree in sign.** These were measured on IQ3_XXS on a bare P100 (and
   `FUSE_WAIT`/`GR_UP75` originally on an RTX 2080 Ti); on Q4_K_M with three helper cards they cost nothing
   and buy nothing. They are harmless, so nothing is changed — but this box's copy of the table is
   effectively **three** switches, not seven.
4. **`STRATA_QFUSE=1` on top of the table is the run's second real decode win: −1.33 %, both rounds.**
   Combined with the earlier table-off sweep (−1.03 %, both rounds), `QFUSE` has now won in **four of four
   rounds across two independent baselines**. That is the strongest evidence of any flag in this run.

**`QFUSE` is not a free switch, and the release notes say so — but the defect they name is already fixed
in this tree.** `docs/release-notes-0.2.7.md:52` records: *"the fused int8 tails of `STRATA_QFUSE` used a
division that differed from the matrix kernels in about one code per million (#3948, found by a parity test,
that switch is off by default, the default path is unchanged)"*. Reading the fix: `fused_gr.cu:6` defines
`STRATA_Q81_FASTMATH_DIV`, and `include/strata/kernels/q8_1_finite.hpp:29-37`'s `q81_div` then emits
`div.full.f32` — the same fast division a `--use_fast_math` matrix kernel uses — instead of IEEE `a / b`.
The header's own comment names the two producers this covers: *"fused_gr.cu's hyper-connection read,
verify_kernels.cu's GDN output norm: STRATA_QFUSE"*. So on this tree the divergence should be gone.

**That is a claim from reading code, and the test says otherwise — `gr_parity` FAILS on this box, with
and without the switch.** `gr_parity.cpp:416-459` compares the read's own q8_1 image against
`native_quantize_q8_1`'s bytes for T 1..8, direct and as graph replays; `:464` onwards does the same for a
two-group verify window. Run on the P100 (`CUDA_VISIBLE_DEVICES=2`, build-rebase binary):

```
  QFUSE: T=1 rep 0: q8_1 bytes differ
  fused GR read + q8_1 (STRATA_QFUSE), T 1..8, direct and graph replays: FAIL
  QFUSE two groups (group-relative, direct): T=2 rep 0: group 0 differs
    token 0 block 64 byte 0 (scale/sum): got 0x5a, separate quantise 0x00
  ...
gr_read/gr_write: 3 failures
```

Two facts pin what this is and is not:
1. **`0x5a` is the test's own poison byte** (`:434` `cudaMemsetAsync(d_q, 0x5a, ...)`), and every compared
   byte still reads `0x5a` — so that region of the q8_1 image was **never written**, rather than written
   wrongly. The `separate quantise` side reads `0x00` because its input is all zeros.
2. **The same three failures appear with `STRATA_QFUSE` unset.** The control run is byte-identical in its
   failing lines, so this is not the switch turning something on — `fused_gr.cu:1414,1432,618,1623,1913,2256`
   gate the q8_1 tail on `q8_mixed != nullptr`, which `gr_parity` sets unconditionally, not on the env var.

**Conclusion recorded, not glossed:** `QFUSE` is *measured faster* (five batches, ≈−1 %) but **not
demonstrated bit-exact on this card** — the harness that would demonstrate it is already red in this tree,
before and after the switch is involved. `src/kernels/{gr_parity.cpp,cuda/fused_gr.cu}` and
`native_mmvq.cu` are untouched by this run (`git status` clean on all of them), so this is pre-existing
branch state, not something these changes introduced. It stays opt-in. Making it default would need this
test green first, and that is a separate piece of work.

**Cumulative decode best identified so far** (not yet run as one arm): table (−5.0 % from `_r2`, pending the
live A/B) + `QFUSE` (−1.33 % on top) + the 2-helper layout (−0.8 %). `R-tableab.sh` (off / on / on+QFUSE,
3 rounds) is the run that confirms the first two together.

**Noise floor, for reading the above.** Same-arm `table` repeats were 5351.0 vs 5283.4 — **1.3 % apart** —
while `no_side` repeated to 0.01 %. So this box's error bar is not uniform: an arm can repeat tightly and
still sit 1 % off another arm. Only same-direction repeats across rounds, plus a mechanism, are treated as
signal here; that is why items 2-3 above are recorded and left alone rather than acted on.

**Closed: `--peer-prefill-rows` does not reach the helper caches** (measured: same throughput and
same token stream as the plain 3-helper arm).

**Corrected twice during the run:** `--expert-cache-remote-placement stripe` (this tree's DEFAULT)
is already the Q36 row split, so `HELPER_PARALLEL`'s parallelism exists; what is missing is
ASYNC + MARSHAL (`expert_source.cpp:5013-5020` and `:5214-5220` dispatch the helpers serially on
one host thread). And MTP is **not** absent from Flash-Next — `MtpDrafter`, `draft_kv` and the
accept loop all exist and `src/core/mtp.cpp` is compiled in; what Whittle lacks is the **weights**
(the GGUF ships `blk.0..blk.39`, no MTP block), so MTP here is a weights question first.

### 2026-10-28 20:30-21:00 — decode helper sweep: the helpers help, and 2 is the knee

`R-helper-sweep.sh`, 3 interleaved rounds, 256-token generation, 275-token prompt, binary `ce487ed6bcd9`:

| helpers | decode ms (256 tok), 3 rounds | ms/token | vs 0 |
|---|---|---:|---:|
| 0 | 6312 / 6356 / 6318 | 24.73 | — |
| 1 | 5932 / 5901 / 5928 | 23.15 | −6.4 % |
| 2 | 5333 / 5314 / 5304 | 20.82 | **−15.8 %** |
| 3 | 5354 / 5445 / 5361 | 21.06 | −14.8 % |

So the decode is **not** helper-serial in the sense of "helpers only cost": adding helpers helps a lot
up to two and then flattens (the third adds nothing measurable, +1.2 % and inside the spread). Two
readings, both consistent with the numbers: past two helpers the 6.1 GB/s host route is the wall, or
the primary's own per-token work is. Either way 2 helpers is the decode configuration to carry forward,
and a 4th card would not pay.

### 2026-10-28 21:50-22:10 — 2 helpers vs 3: a 0.8 % win, and +50 % expert density per card

3 interleaved pairs, 256-token generation, binary `b624021caf32`, identical flags but the third
`--expert-cache-device3`:

| round | 3 helpers decode ms | 2 helpers decode ms | Δ |
|---|---:|---:|---:|
| 1 | 5369.1 | 5306.0 | −1.17 % |
| 2 | 5389.7 | 5337.5 | −0.97 % |
| 3 | 5348.6 | 5326.5 | −0.41 % |
| **median** | **5369.1** | **5326.5** | **−0.79 %** |

So the third card is **not free but nearly**: 0.8 % of decode. What it does buy is density — the same VRAM
serves **2738 experts/card at two helpers vs 1825 at three (+50 %)**, because the ranked list is split two
ways instead of three. For a box at 31 GiB of RAM where the 18.16 GiB expert arena is the binding
constraint, the two-helper layout also leaves a card's 8 GiB and its host-side staging slots unused.

**Recommendation for this box: two helpers.** Not because it is measurably faster (0.8 % is at the edge of
the spread) but because it delivers the same decode on one card less, with half again the experts resident
per card.

This also corrects the earlier reading of the 0/1/2/3-helper sweep (24.73 / 23.15 / 20.82 / 21.06 ms per
token): the gain is 0 → 2 (−15.8 %), and 2 → 3 is a **wash**, not a regression. The third helper's earlier
+1.2 % was noise; the honest statement is "the knee is at two, and a third adds nothing".

### 2026-10-28 21:20-21:50 — HELPER_PARALLEL: implemented, and measured NEGATIVE on this box

`STRATA_REMOTE_HELPER_PARALLEL=1` runs the helpers' `begin()` calls on worker threads instead of dispatching
them one after another on the host thread (`expert_source.cpp`, the branch at the `remote_count > 1` site).
Binary `b624021caf32`, 3 interleaved pairs, 256-token generation, 275-token prompt:

| round | off decode ms | on decode ms | off prompt ms | on prompt ms |
|---|---:|---:|---:|---:|
| 1 | 5365.6 | 7044.5 | 1147 | 1349 |
| 2 | 5395.4 | 6999.6 | 1253 | 1363 |
| 3 | 5349.5 | 6986.5 | 1299 | 1130 |
| **median** | **5365.6** | **6999.6** | 1253 | 1349 |

**+30.5 % decode time with the flag on, in all three pairs.** The parallel branch did run — the timing line
counted **10240 `parallel begin` dispatches** (40 layers × 256 tokens) — so this is the mechanism working
and being slower, not a flag that failed to take effect.

**The first explanation was wrong, and the hardware says so.** I wrote that the three helpers share one
6.1 GB/s host route. They do not: this box's helpers are **Gen1 x4** cards on their own links. Measured
directly (`.run/pcbw3.cu`, 3 helpers × 8 MB × 32 iterations):

```
serial     1030.7 ms  ->  0.78 GB/s aggregate
parallel    349.2 ms  ->  2.31 GB/s aggregate
speedup      2.95x     ->  INDEPENDENT links (parallel scales)
```

So the links were never the problem. Two other suspects were eliminated by measurement: a per-copy
`cudaSetDevice` (0.86×, i.e. free) and a cross-thread lock (each `RemoteExperts` has its own `read_mu_`).

**The real cause was the implementation: one `std::thread` per dispatch.** A dispatch is one call per
layer, so 40 per token; each `begin()` is a few microseconds of host work (a 64 KB memcpy plus launches),
while creating and joining three threads costs ~70 µs each. Measured in isolation over the same 10240
dispatches:

```
10240 dispatches x 3 memcpys of 8192 B
  serial     19.8 ms
  threaded  747.0 ms   (37.67x)
```

**Rewritten as a persistent pool** (created once, woken on a generation counter, parked again). Same
3 interleaved pairs, binary `3e702c0c1b01`+:

| round | off decode ms | on decode ms | Δ |
|---|---:|---:|---:|
| 1 | 5350.5 | 5592.9 | +4.53 % |
| 2 | 5394.8 | 5568.7 | +3.22 % |
| 3 | 5366.9 | 5537.5 | +3.18 % |
| **median** | **5366.9** | **5568.7** | **+3.76 %** |

So the pool fixed two thirds of the regression (+30.5 % → +3.8 %) and the switch is still
**measured-negative**: what is left is the wake-up/join cost against a wait it can only save once per
layer, and the helper's own `drain()` is a hard `cudaStreamSynchronize` per card per layer
(`remote_experts.cpp:624`) that parallel dispatch does not remove.

**Kept in the tree, off by default.** The default path is byte-identical either way: the flag-off
64-token stream is `ad1892f9e6afe279` on both binaries.

**What would actually help, and is the one real gap found by reading Q36:** the helper's completion is
waited for *inside the same layer* (`expert_source.cpp:5332`, after the CPU pool at `:5328`), so the
three streams are synchronized before the layer's own GPU work continues. Q36's `HELPER_ASYNC` defers
that `finish` past the layer's local work. That is a scheduler change, not a threading one, and it is the
next thing worth trying rather than more ways to call `begin()`.

### 2026-10-28 22:20-22:40 — the "three serial drains" hypothesis, tested and refuted

The reading above suggested the three helpers' drains add up: `finish()` calls `drain()` per card
(`remote_experts.cpp:648`), and for `rings_ == 1` that drain is a blocking `cudaStreamSynchronize`
(`:624`). Added `STRATA_REMOTE_FINISH_PAR=1`: poll every card's stream to completion first
(`drain_all`), then scatter — so the layer waits for the slowest card rather than the sum. Three
interleaved pairs, 256-token generation:

| round | off decode ms | on decode ms | Δ |
|---|---:|---:|---:|
| 1 | 5364.1 | 5383.8 | +0.37 % |
| 2 | 5349.3 | 5373.1 | +0.44 % |
| 3 | 5407.9 | 5362.0 | −0.85 % |
| **median** | **5364.1** | **5373.1** | **+0.17 %** |

**Neutral to within noise, and that is the informative part: the drains were never adding up.** Each
card's GPU work starts at `begin()`, so when the first `finish()` blocks, the other two cards are already
computing; by the time the loop reaches them their work is done. Flash-Next's per-card independent streams
already give the overlap that Q36's `HELPER_ASYNC` has to create, because Q36's `RemoteExperts` is a
single slot whose `finish` gates the next layer. **There is no Q36-style async win to port here** — the
mechanism is already present structurally, just not named the same way.

The flag stays (correct and harmless), off by default; the flag-off stream is unchanged.

### 2026-10-28 21:00-21:15 — DFlash2: the drafter that fits Whittle EXISTS, and the upstream says it loses

**Searched HuggingFace for a DFlash2 pack this engine can run.** `src/core/dflash2.cpp:167` used to
gate on `NH_ != 32 || NKV_ != 8 || HD_ != 128 || rank_ != 256 || N_ != 2560 || L_ > 8`, i.e. exactly
the Flash-Next drafter. Whittle is hidden **2048**, so no published Flash-Next pack could load.

Found **`incoai/Qwen3.6-35B-A3B-DFlash2`** — `base_model: Qwen/Qwen3.6-35B-A3B`, the model Whittle is
merged from. Its `config.json` reads hidden **2048**, 6 layers, heads 32 / kv 8 / head_dim 128, inter
6144, window 2048, rank 256, **8 taps** `[1,6,11,16,22,27,32,37]`. Every field except `hidden` is the
same as the Flash-Next pack, and `hidden` is Whittle's own width — the gate was the only thing in the
way, and it was a hardcode, not a kernel limit.

The target side matches too: `Qwen/Qwen3.6-35B-A3B` is 40 layers / 16 heads / 2 kv / head_dim 256 /
256 experts / top-8 / moe_inter 512 / vocab 248320, which is Whittle's geometry line for line, and it
has **no hyper-connections** — so the drafter was trained on plain hidden states, the same tensors the
engine's tap capture produces (the `hc_mean_streams` path degenerates to a copy when `hc == 1`).

**Parameterized the loader** (`src/core/dflash2.cpp`, `tools/dflash2_pack.py`, the header comment):
the geometry now comes from the pack's own `config.json`; what stays fixed is what the kernels really
fix — draft attention 32 q-heads / 8 kv / head 128, selector rank 256, `kRows` 8, at most 8 layers
(the `kring_[8]` array), and `window + 16 <= kRing`. Two widths that were hardcoded to
`TAPS × N = 12800` and `4 × N/16 = 640` now follow the pack (`taps_.size() * N_`, `4 * (N_/group_)`) —
with the old constants an 8-tap 2048 pack would have overrun its activation buffer. The `fc.weight`
check (`[N, taps × N]`) is the load-time proof that a pack's tap list and width agree.

**Built and packed it on the box**: `tools/dflash2_pack.py` wrote 96 tensors / 678.7 MB, index
`hidden 2048 / layers 6 / taps 1,6,11,16,22,27,32,37 / fc [2048, 16384]`, and the engine loaded it —
no shape refusal, the whole path reached the expert-cache stage. `tools/test_dflash2_pack.py` passes
3 + 1 skipped.

**But the upstream has already measured this and it loses.** Strata's own issues:

- **#3965** (RTX 4060 Ti 16 GB, sm_89, 2026-10-24): DFlash2 was **3.3 % slower** than MTP at 1k output
  and **10.1 % slower** at 16k. Acceptance 66.5 % vs 69.1 % (1k) and 81.1 % vs 87.8 % (16k); rounds
  5,552 vs 5,091; CPU expert pool 35.0 vs 32.6 ms/round; **expert-cache slots 3027 vs 3594 (−567,
  −15.8 %)**. Its own conclusion: the gap **grows with output length**, so long outputs do not rescue it.
- **#3457** (M4 Max): "The stock `incoai/Qwen3.8-Flash-Next-DFlash2` drafter was **slower** than MTP on
  this target (41.1-42.5 vs 42.1-43.7 tok/s). **The gain comes from a drafter fine-tuned on the
  target's own features, not from DFlash2 alone.**"
- **#3322**: even the *official* Flash-Next DFlash2 on a 256K code run gave +14.84 % decode over MTP,
  but **full request time was 0.72 % slower** because the prompt dominates.

The −567 expert slots is the part that decides it here: this box runs Whittle Q4_K_M on a 12 GB P100
with 4208–4465 slots and the rest of the experts on the CPU pool, so a drafter that costs ~16 % of the
slots trades a 15 % hit-rate drop against a draft that is *already* slower than a fallback this tree
cannot even run (Whittle has no MTP weights). Recorded as measured-negative for Whittle on this box; the
re-parameterization stays because it is what makes *any* future Whittle-trained drafter loadable.

**And it does load and run on Whittle now**, which is what the re-parameterization bought. With the
strengthened gate (below), binary `3e702c0c1b01`:

```
strata dflash2: 248320-token draft head (397.9 MiB), drafter 755 MiB of VRAM
strata generate: DFlash2 drafter on (qwen36-dflash2.pack, graphs captured in 10 ms), the MTP layer stays as the fallback
decode   16 tokens in 617.7 ms  ->  25.90 tok/s
prefill  275 tokens in 1280.1 ms -> 214.83 tok/s
```

25.90 tok/s against the no-drafter 32.46: slower, exactly as the upstream A/Bs predicted. So the whole
path is now exercised and measured rather than argued about, and the answer for this box is no.

**The gate was also made stronger than the shape list.** The Q8 GEMM rejects an input width that is not a
multiple of 512 and an output that is not a multiple of 16 (`kernels/cuda/dflash2.cu:719,723`), and the
first version checked only the config's `N_`/`I_`. It now walks every `q8_0` tensor and names the offender,
so a mis-packed or hand-edited pack fails at load with its own name instead of throwing inside a kernel.
Verified: the Qwen3.6 pack still loads, and the default path's 64-token stream is still
`ad1892f9e6afe279`.

**What the search also turned up, and it matters more than DFlash2:** `Qwen/Qwen3.6-35B-A3B`'s own
checkpoint carries **19 `mtp.*` tensors** in shards 25-26 (`mtp.fc.weight [2048,4096]`,
`mtp.layers.0.mlp.gate.weight [256,2048]`, `mlp.experts.down_proj [256,2048,512]`,
`self_attn.q_proj [8192,2048]`, `pre_fc_norm_hidden/embedding [2048]`, …) — sizes that match Whittle
exactly because the experts are [256, 2048, 512] and hidden is 2048. `docs/MODELS.md:266` already
points at `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` as the MTP weights for this family, and `tools/mtp_fetch.py`
already knows how to range-fetch `mtp.*` tensors out of a safetensors checkout. That is the item-3 path:
**Whittle's MTP weights are one fetch away**, and it is a genuine Qwen3.6-trained head rather than a
drafter borrowed from another model.

### 2026-10-28 21:00-21:30 — MTP, item 3: the block exists, and the HC gap is closable EXACTLY

**Neither checkpoint ships MTP weights.** `logic65/Whittle-Qwen-3.8-45B-A3B` has 979 tensors, prefixes
`{model: 978, lm_head: 1}`, **0 `mtp.*`** — its config's `mtp_num_hidden_layers: 1` is a declaration, not
weights. `Qwen/Qwen3.6-35B-A3B` (the model Whittle is merged from) **does** ship them: **19 `mtp.*`
tensors**, 1689.3 MB, in shards 25-26.

**The two MTP blocks are not the same block**, and this is the whole problem:

| | Flash-Next MTP (`tools/mtp_fetch.py` SHA256 table) | Qwen3.6 MTP (`mtp.*`) |
|---|---|---|
| tensors | **31** | **19** |
| `fc` | `fc_embedding` + `fc_hidden` (separate) | `mtp.fc` `[2048, 4096]` (fused) |
| hyper-connections | 11 tensors (1 mixer ×3, attn ×4, mlp ×4) | **none** |
| QSA indexer | 3 tensors | **none** |
| width | 2560 | 2048 |
| experts | 512, top-10 | 256, top-8 |

The engine's MTP is written for the first: `src/core/mtp.cpp:348` requires `fc_embedding` + `fc_hidden`,
and `:967`/`:1092` dereference `attn_hyper_connection.hc_norm` / `mlp_hyper_connection.hc_norm` through
`fused_gr_read_multi`, which `std::exit(1)`s on a null weight (`fused_gr.cu:2512`). Whittle runs with
`hc_count = 4`, so `fused_gr_supported()` (`fused_gr.cu:2507`, `hc == HC == 4`) is satisfied and the HC
path is taken — there is **no non-HC branch anywhere in the MTP**. Qwen3.6's block has no HC because
Qwen3.6 has no HC.

**But the gap is exactly closable without training and without touching the engine.** With zero matrices
the hyper-connection read *is* identity, and the arithmetic closes exactly (`fused_gr.cu:150-210`):

```
w_down = 0                    ->  lo = silu(w_down . xn / HC) = 0                       (:186)
w_up   = 0                    ->  sigmoid(w_up . lo) = 0.5                             (:205)
                              ->  mixed = (1/HC) sum_c R_c . w_norm_c . rs_c . 0.5      (:204-209)
w_norm = gamma (same row for all streams)  ->  factors out:
                                  mixed = 0.5 . gamma . mean_c(R_c . rs_c)             the pooled RMSNorm
w_inject = 0                  ->  inject = 0, and R += bo . 2.sigmoid(0/4) = + bo      (:200)
                              ->  the 0.5 above cancels EXACTLY
```

and the surviving 0.5 then rides through `native_quantize_q8_1`, which divides by `amax` and multiplies
back (`d = amax/127`, `q = round(x/d)`, result `q*d`), so it cancels there too. The fill is therefore
exact, not an approximation. `block_inject_weight` null and zero are the same result (`fused_gr.cu:163`
skips the inject rows when it is null), so those two stay zero-filled BF16.

**What that means concretely**, written as `tools/mtp_rt_qwen36.py`:
- `mtp.fc.weight [2048,4096]` splits on columns into `fc_embedding` + `fc_hidden` `[2048,2048]` each —
  the engine applies both to a 2048-wide input and adds them (`mtp.cpp:949`, `:958`, `:960`), which is
  exactly what one fused projection does.
- `hc_norm` = the layer's own `input_layernorm` / `post_attention_layernorm` gamma, **which is the same
  norm Qwen3.6's layer applies** — so the broadcast is not a guess, it is the right gamma.
- the 3 mixer fills per read (`down`, `up`) + 2 `block_inject` = 11 tensors, all zeros/ones.
- `q_proj [8192,2048]` needs no work: the engine already takes `NH*2*HD = 16*2*256 = 8192` and applies
  the gate (`mtp.cpp:1057`, `:1082`, `native_qsa_gate_apply`), and Whittle's `attn_output_gate` is True.
- the QSA indexer is **not read** by the MTP path (`mtp.cpp:373`: "dense attention: no indexer state is
  read"), so its 3 missing tensors cost nothing.
- the router takes the generic branch: `K = g.n_expert_used = 8` (`generate.cpp:4302`), so
  `router_top10(..., K=8, ...)` runs rather than the `n_expert==512 && K==10` fast path.

**Two layout traps already paid for** (memory: q36-mtp-sidecar-layout), both applied in the tool:
the expert tensors read back with the expert axis **already leading** (`[NE, FF, H]` / `[NE, H, FF]` — a
transpose breaks all three tensors to `[256,512,2048]`), and the checkpoint stores every RMSNorm gamma as
`w - 1` while the engine multiplies by `w`, so the 4 norms (and the two `hc_norm` copies) take `+1`.

**The blob size closes**: `cpu::BLOB = 3*(H*FF*BB/QK) = 3*(2048*512*18/64) = 884736` for Whittle's
geometry, and `blob_of()` produces exactly 884736 bytes with `O_GU_CODES=0 / O_D_CODES / O_GU_SCALES /
O_D_SCALES` in the order the engine's `expert.hpp:49-55` lays out. Verified locally on synthetic data.

**The source is already on the box**: `/home/Strata/.run/q36mtp/q36-mtp-f16.gguf` (1,689,304,512 B, built
2026-10-28 11:26 by `tools/q36_mtp_extract.py --from-checkpoint`, metadata `mtp.source = checkpoint-bf16`)
holds all 19 tensors as `blk.40.*`. So the MTP route needs **no network** — only a re-layout into the
Flash-Next loader's runtime directory, which is what the new tool does.

**It built.** `/home/Strata/mtp/qwen36/rt/`: `dense.bin` 73.9 MB (26 tensors), `experts.bin` 226.5 MB
(256 experts × 884736 B), `dense.txt` 26 lines. The tool asserts the name set at the end and prints
"all 23 names the loader reads are present"; an independent cross-check of its output against the 26 names
grepped out of `mtp.cpp` gives **0 missing, 0 extra**.

**Two build breaks on the way, both worth remembering:**
- `ExpertDispatch::fail` is `const char*` (`include/strata/core/expert_source.hpp:657`), **not** a
  `std::string`. Assigning a local `std::string` to it does not compile, and assigning `.c_str()` of a
  *local* would dangle. The parallel branch therefore keeps its per-helper messages in a function-static
  `std::array<std::string, 3>`: each worker writes only its own element (no race), and the storage outlives
  the `const char*` the caller keeps.
- The first version of the branch used `static thread_local std::string` per worker, which is worse than a
  race: each worker fills its own copy and the reading thread sees `""`, so every helper failure would have
  been reported as an empty message.
- `tools/mtp_rt_qwen36.py` initially used the checkpoint's `mtp.layers.0.*` names, but the sidecar is a
  GGUF, so its tensors are `blk.40.attn_q.weight` and friends — the rename table is llama.cpp's spelling
  on the left, the engine's on the right.

### 2026-10-28 21:20 — the MTP loads and runs, and its acceptance is ZERO

Binary `b624021caf32`, Whittle Q4_K_M, `--mtp /home/Strata/mtp/qwen36/rt --spec 4`, 275-token prompt,
64 new tokens, one run each arm:

```
strata mtp: draft layer loaded, 295 MiB of VRAM (experts 216, dense 70), files read in 0.31 s (915 MiB/s)
strata verify: captured the 4-token window (upload no error, sync no error)
strata mtp: 8.472 ms/round drafting (63 rounds), MTP prompt 7.6 ms, 295 MiB of VRAM
```

So the whole chain works: the layout tool's files load, the HC fill is accepted (no null dereference,
`fused_gr_read` did not `exit(1)`), the verify graphs capture, and the draft loop runs 63 rounds.

**But it accepts nothing.** 63 rounds for 64 tokens = **1.016 tokens/round** at a window of 4: essentially
every draft is rejected and each round advances one verified token. The cost of that is not free — 8.47 ms
of MTP forward per round buys nothing, so decode goes **backwards**:

| | no drafter (`--spec 1`) | MTP (`--spec 4`) |
|---|---:|---:|
| decode | **32.46 tok/s** (1971.7 ms / 64) | 17.89 tok/s (3578.2 ms / 64) |
| prefill | 227.22 tok/s | 228.00 tok/s |
| rounds | — | 63 (1.016 tok/round) |

Prefill is unaffected (as it should be — the drafter only lives in the decode loop).

**The acceptance, verbatim** (`--spec 4`, one run):

```
speculation              60 rounds of 6, drafts accepted 4 of 177 (0.023), 1.07 tokens per round
accepted per round       0:56 1:4 2:0 3:0 4:0 5:0
mtp                      8.471 ms/round drafting (59 rounds), MTP prompt 8.0 ms, 295 MiB of VRAM
```

177 drafts, 4 accepted, and **56 of 60 rounds accept nothing at all** — position 0 fails. That pattern is
not "a weak drafter", it is "the drafter predicts something else", so it is a layout question first.

**Found: three of the four norms were a silent overread.** `native_qsa_rms_norm_weighted` is called for
the hidden path with `n_cols = HC*N = 8192` (`src/core/mtp.cpp:954`) and the kernel checks the gamma
against `n_cols * 4` bytes (`src/kernels/cuda/native_qsa.cu:138`); `fused_gr`'s `w_norm` is
`(hc * n_embd) f32` (`include/strata/kernels/fused_gr.hpp:30`). The first two versions of the tool wrote
those as **[2048]** — so the kernel read 6144 floats past the end of each, straight into whatever tensor
followed it in `dense.bin`. No load error, no NaN, just wrong norms.

The fix is `hc_tile()`: the same gamma repeated HC times, which makes one RMS over the pooled `HC*N` row
scaled elementwise by the repeated gamma exactly "normalize the pooled vector with this gamma". Applied to
`pre_fc_norm_hidden` and all three `hc_norm` tensors; `pre_fc_norm_embedding` stays `[N]` because its call
passes `(int) N`. Rebuilt as `rt3`.

**The fix works and is not enough.** Three layouts, same binary `b624021caf32`, same prompt, `--spec 4`:

| layout | accepted | rate | rounds | tok/round | decode |
|---|---|---:|---:|---:|---:|
| no drafter (`--spec 1`) | — | — | — | — | **32.46 tok/s** |
| rt2 (norms `[N]`, overread) | 4 / 177 | 2.3 % | 60 | 1.07 | 19.08 tok/s |
| **rt3 (norms `[HC*N]`)** | **12 / 153** | **7.8 %** | 52 | 1.23 | 21.97 tok/s |

3.4× more acceptance for a real overread fix, so the layout work is right — and 7.8 % is still nowhere
near usable (DFlash2/MTP both measure 60-88 % on a matched target). `accepted per round 0:43 1:6 2:3 3:0
4:0 5:0` says positions 3-5 never survive: the drafter's first two tokens are sometimes close, then drift.
That is the signature of a head predicting from the wrong distribution, not of a layout bug.

**Why zero, and what it would take.** The block is a *correct* Qwen3.6 MTP layer re-laid-out into a
Flash-Next loader, but three things about it are borrowed rather than earned:
1. **The weights were never trained against Whittle's hidden states.** Whittle is a merge/fine-tune of
   Qwen3.6-35B-A3B with its own hyper-connection structure and its own layer contents; the MTP head
   predicts from the *base* model's hidden distribution. `#3457` measured exactly this failure mode for
   DFlash2 ("the gain comes from a drafter fine-tuned on the target's own features").
2. **The `hc_norm` gamma is a substitution.** The read is algebraically exact (see above), but the norm
   that Qwen3.6's head expects is over one stream, while Whittle's hidden lives in four; `mean_c(R_c·rs_c)`
   is the pooled reduction the engine offers, not necessarily the one the head was trained on.
3. **The head's final norm was missing, and that one was a bug in the tool, not a mismatch.** Flash-Next's
   block has no `mtp.norm.weight`; it lets `hyper_connection_mixer` build the head input. Qwen3.6's head
   input is `norm(R)` and its checkpoint carries that gamma, as `nextn.shared_head_norm.weight` in the
   GGUF's spelling — which the first version of the tool left out, feeding the head unnormalized values.
   The fix is the same construction as the two block mixers: put that gamma into the mixer's `hc_norm`
   and the zero down/up makes the mixer the pooled-RMS reduction. Rebuilt as `rt2`; the acceptance number
   above therefore belongs to the *broken* layout and the re-run is the one that counts.

Recorded as measured-negative for the item "MTP 要实现": the *mechanism* is now in place and measured, the
*weights* are the missing piece — and the only head that is not borrowed would have to be trained on
Whittle's own hidden states. The runtime directory, the tool and the loader change all stay, because they
are what makes such a head loadable the day it exists.

### 2026-10-28 22:50-23:05 — flag sweep: two survivors, one certain loss, and a harness defect

`R-flagsweep.sh`, 6 arms × 2 rounds, interleaved with `base`, 256-token generation, 275-token prompt,
binary `315469d836bf`. Decode ms:

| arm | round 1 | round 2 | mean | vs base |
|---|---:|---:|---:|---:|
| base | 5388.3 | 5368.4 | 5378.4 | — |
| **`STRATA_QFUSE=1`** | 5327.4 | 5318.6 | **5323.0** | **−1.03 %** |
| `STRATA_ADAPT_NOWAIT=1` | 5372.6 | 5409.8 | 5391.2 | +0.24 % |
| `STRATA_ADAPT_LAG=2` | 5367.3 | 5378.2 | 5372.8 | −0.10 % |
| **`STRATA_SIDE_MIXER=1`** | 5328.5 | 5299.3 | **5313.9** | **−1.20 %** |
| `STRATA_VERIFY_DEVICE_PLAN=1` | 5492.9 | 5493.3 | 5493.1 | **+2.13 %** |

Three findings:
1. **`QFUSE` and `SIDE_MIXER` each take ~1 % off decode, in both rounds, in the same direction.** That is
   the first decode gain found since the helper sweep. Both are *existing* switches the audit surfaced; nether
   is on by default on this arch (`QFUSE` is an sm120 default in `arch_defaults.cpp:78`, `SIDE_MIXER` needs
   sm120 or an explicit `=1` in `verify.cpp:252-273`). Round 3 to confirm them plus their combination is queued.
2. **`VERIFY_DEVICE_PLAN` is a certain loss: +2.13 % and +2.13 %** — the two rounds agree to 0.01 %, which
   is the tightest agreement any arm in this run has shown. Left off by default, now documented as measured.
3. **`ADAPT_NOWAIT` / `ADAPT_LAG` do nothing here** (−0.10 % to +0.24 %, sign flips between rounds). They are
   A4000/3060 defaults (`nv_defaults.cpp:26,40`), and this box's P100 does not take that path.

**Harness defect found, and it invalidates the token-hash check.** `base_1` and `base_2` are the same
configuration, yet their token streams diverge at token 18 and their `ram_reads` differ by 740
(60862 vs 61602). It is not sampling — serve requests default to `temperature 0`, hence greedy
(`generate.cpp:15577`). The same divergence appears in the earlier `twohelper` harness (`h3_1` vs `h3_2`:
61046 vs 60506 reads). So **under `--expert-cache auto` the expert residency path is host-timing
dependent, and the generated tokens are not reproducible on this box**. Two consequences, both recorded:

- `tok_md5` is **not** a validity check for any serve-mode arm in this run. It only ever proved the first
  17 tokens matched. Earlier uses of it as a "byte-identical" claim (e.g. for `HELPER_PARALLEL`) hold only
  for the `generate` mode runs, and those are still valid.
- The A/B noise floor has a real mechanism: different tokens → different routing → different expert hit
  rates. That is where part of the ±1.3 % spread comes from. A 1 % result therefore needs a same-direction
  repeat to be believed, and `VERIFY_DEVICE_PLAN`'s 0.01 % agreement across rounds is the exception, not
  the rule.

### 2026-10-29 — prefill 30k: the `combine` phase is the dominant cost, and it is a PCIe wait, not a kernel

`STRATA_PREFILL_TIMING=1` on a 30k prompt (`R-pftiming.sh`, binary `5f63a48e`, 2 helpers + `STRATA_PREFILL_REMOTE_PEERS=1`):

| phase | ms | % |
|---|---:|---:|
| **combine** | **21,230** | **28.2 %** |
| gemm gate/up | 12,567 | 16.7 % |
| gdn | 9,754 | 13.0 % |
| gemm down | 6,344 | 8.4 % |
| hc read | 5,683 | 7.6 % |
| qsa attn | 4,580 | 6.1 % |

**The helper cards are idle 77 % of the wall-clock** (`idle 58,132 ms` out of the 75,221 ms GPU timeline;
`moe in 6,245 + moe gemm 10,785 + moe out 27` = 17,057 ms busy). The prefill is entirely bound on the primary.

**Why `combine` is 28 %**: the `kPfCombine` mark at `prefill.cpp:5652` lands *before* the peer-return wait;
the `moe_combine` call at `:5691` comes after `cudaStreamWaitEvent(m.cs, P.ev_done, 0)` at `:5681`, so the
phase includes the helper-return window. On this box (no P2P) the helper caches force `sums=false`
(`prefill.cpp:2402`), so the return path is the **rows mode**: `copy_f32_wide(m.Dm + back_at*N,
P.host_rows, back_rows*N)` where `P.host_rows` is `cudaHostAllocMapped` pinned host memory. The P100 reads
`back_rows × N × 4` bytes of per-layer expert output through the PCIe mapped route (6 GB/s) inside the
combine window. At the default 512-token chunk, `back_rows ≈ 2560` per helper per layer × 2 helpers × 40
layers = 204,800 rows × 2048 × 4 B = 1.7 GB ÷ 6 GB/s ≈ 280 s — still far short of 21.2 s, so the
*wait* dominates, not the copy itself: the P100's compute stream blocks on `ev_done` while the P104 is
still running its `moe gemm` (10,785 ms per helper), and the P104's own input wait (`moe in`, 6,245 ms) is
the D2H upload of `T×N` activations (512×2048×4 = 4 MB per layer, trivial at 6 GB/s) plus the P104's
`f16_to_f32_wide` read from mapped host. The net effect: **the primary and each helper run in lockstep,
neither overlapping the other's PCIe window**, and the total is bounded by the sum of all sequential
PCIe waits, not by any single copy.

`R-pfenv30k2` (chunk 512/4096/8192/16384, 30k prompt, 2 helpers) settled the chunk-size question:

| chunk | prefill ms | tok/s | vs ch512 |
|---|---:|---:|---:|
| 512 | 79,544 | 377.2 | — |
| 4096 | 68,324 | 439.1 | −14.1 % |
| 8192 | 67,214 | 446.3 | −15.8 % |
| **16384** | **66,727** | **449.6** | **−16.2 %** |

The gain is front-loaded: 512→4096 is −14.1 %, the next two steps add only 2.1 % between them.
**ch16384 is the new prefill default at 449.6 tok/s** (was 399 at ch512). `R-pfenv30k3` (ch16384 with and
without `STRATA_DENSE_F16_ACC=1`, plus a timing breakdown at the winning chunk) is running to confirm
the combine share at ch16384 and whether the FP16-accumulate GEMM adds to it.

### 2026-10-28 23:04-23:10 — `--mtp-hnorm stream`: the Q36 port with upstream numbers, and it is NEUTRAL

The one Q36→Flash-Next port that had a pre-existing upstream measurement (PR #2227: acceptance 82 % → 86 % on an
RTX 4090D). Our `rt3` runtime directory tiles `pre_fc_norm_hidden` to `[HC*N] = 8192`, which is exactly what the
per-stream path indexes, so the arm was runnable as-is. Binary `315469d836bf`, 275-token prompt, `--spec 4`,
`--mtp /home/Strata/mtp/qwen36/rt3`, 128 new tokens:

| round | mode | accepted / offered | rate | tok/round | decode |
|---|---|---|---:|---:|---:|
| 1 | pooled (default) | 26 / 303 | 8.6 % | 1.25 | 22.82 tok/s |
| 1 | `stream` | 27 / 300 | 9.0 % | 1.27 | 22.89 tok/s |
| 2 | pooled | 27 / 300 | 9.0 % | 1.27 | 22.80 tok/s |
| 2 | `stream` | 28 / 297 | 9.4 % | 1.28 | (log) |

**Neutral.** Pooled reads 8.6 % and 9.0 %; stream reads 9.0 % and 9.4 %. The two modes overlap completely in
the run-to-run spread — the 8.6→9.0 movement between two *pooled* runs is as large as the pooled→stream gap
in round 1. The upstream's +4 points of acceptance came from a drafter trained on that target; ours is
borrowed, and acceptance sits at **8.6-9.4 %** either way. So the norm shape was never the problem — the
missing piece is the head's training distribution, which confirms the earlier conclusion rather than adding
a new lever. `stream` stays opt-in. `accepted per round 0:77 1:21 2:3 3:0 4:0 5:0` again: positions 3-5
never survive in any mode.

The Q36→Flash-Next port list is now closed for the decoder: **HELPER_ASYNC/MARSHAL** (refuted — the
per-card streams already overlap; `FINISH_PAR` +0.17 %), **`--mtp-hnorm stream`** (neutral, above),
**`HELPER_PARALLEL`** (implemented, −3.8 % after the pool rewrite, off). What did transfer, and pays,
is the **sm_60 default table** — which is not a Q36 port at all, it was already in the tree.

