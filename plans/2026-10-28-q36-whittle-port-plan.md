# Q36 → Whittle: the effective switches

The user request: 把 Q36 有效的优化开关同步过来. The cards covered five
dimensions of Q36's effective knobs. Both verifiers marked every card
correct on the effect claims; one detail in each of cards 1 and 5 is wrong
and is dropped here:

- Card 1 (ZC/ASYNC/MARSHAL) said the Whittle decode baseline "is 44 tok/s
  with remote-opt on." Wrong. MEASURED (plans/2026-10-28-whittle-summary.md:184,
  :177, :185-186; docs/WHITTLE45B.md:86-87): without remote-opt it is 44.2; with
  remote-opt 47.2; with `layer` placement 47.8. The ASYNC/MARSHAL ceiling is
  read off 47.2/47.8, not 44.
- Card 5 (Q8_SM60) said Whittle-Q4_K_M ships zero Q8_0 tensors and the
  registry would stay empty. Wrong. `.run/whittle_tensors.json`: type counts
  {12:515, 14:81, 6:81, 0:324, 8:20, 1:1}; all 20 type-8 tensors are
  `blk.{3..39 odd}.indexer.{q,k}_proj.weight` ([2048,512] x10 and
  [2048,128] x10), and with STRATA_Q8_SM60=1 they are eligible
  (n_in 2048, native_mmvq.cu:2751), packed (~16 MiB) and dispatched
  (native_mmvq.cu:2566-2572). Effect is still ~0 (20 tiny GEMVs against a
  22.58 ms/round decode); the static check counts ~20 "strata q8 sm60: packed"
  lines, not their absence.
- Also corrected from both verifiers: the Q36 helper ladder "24.3 -> 37.4 ->
  40.8 -> 60.3" conflates two chains — 24.3/37.4/40.8 are the 2026-10-18
  two-helper Ornith runs (plans/2026-10-17-q36-three-card-runbook.md:316-321);
  60.3 is a 2026-10-25 three-helper Qwen3.6-UD run with control 54.5
  (plans/2026-10-25-q36-decode-mtp-directions.md:25-27, :119-124).
- Corrected citation: the mask for a ported async helper must come from
  `RemoteExperts::holds()` at include/strata/core/remote_experts.hpp:60-61, not
  remote_experts.cpp:381-386 (that range is `owns()`).

Everything below cites only the card material and the tree spot-checks.

## 1. Switch on and measure

The knobs that already exist in this tree and only need the env var set on the
Whittle box (192.168.1.98, P100 + 3x P104, Whittle-45B Q4_K_M, 275-token
prompt bench, binary md5 9a4c410c9730 or later). Ordered by expected value.

1. `STRATA_DENSE_F16_ACC=1` — the only knob with a real measured delta here.
   Q36 number: +17.0% prompt (830.6 -> 972.0 tok/s pp512,
   docs/QWEN36.md:510-512). Whittle status: MEASURED neutral at T=275
   (prefill 213 vs 201 ms, decode 36.95 vs 36.87 tok/s, within noise;
   plans/2026-10-28-whittle-q36-opts.md:136; docs/WHITTLE45B.md:162). But that
   measurement was on the flag-off peer arm before the box moved 17% more
   expert rows to helpers; the plan says re-measure (plans/2026-10-28-whittle-
   remote-prefill-peers.md:87-89). Mechanism for the miss: the per-shape algo
   table kDenseAcc (src/prefill/gemm.cu:902-937) was tuned on Q36 shapes
   (8192x2048, 4096x2048, 2048x4096, 512x2048, 2048x512); a shape with no
   entry keeps the FP32-accumulate path (gemm.cu:928-937). Gate is cc>=60 &&
   cc<70 (gemm.cu:886-888), so the P104 helpers (sm_61) are eligible too.
   Bench arm: the 3-helper peer arm, flag off vs on, same day, same binary.
   Pass: prompt drops below ~790 ms (-5% over the noise floor), first-token
   logprob stays within 0.0027 nats of the flag-off value
   (plans/2026-10-28-whittle-remote-prefill-peers.md:154-164; the FP16-acc
   precision cost is 0.36-0.38% rms, docs/QWEN36.md:516). If flat, read the
   dense phases with STRATA_PREFILL_TIMING=1 (src/prefill/prefill.cpp:2848)
   and extend kDenseAcc — that is a port, section 2.
2. `STRATA_PF_F16_CUDA=1` — worth ~0 at the 275-token prompt, re-test only at
   long prompts. Q36 number: +48% prompt (589.6 -> 875.9 pp512,
   docs/QWEN36.md:499-505). Whittle status: MEASURED neutral on the short
   prompt (docs/WHITTLE45B.md:161: "the flag reaches the kernels but no
   measurable change"). Mechanism: the fused FP16 K-quant path only takes
   chunks with T >= stream_all_min() = 1024 (prefill.cpp:243, :2223,
   :4341; moe_fused_iq.cu:2099-2112), and the peer tier is MMQ-only
   (prefill.cpp:2226) — at 275 tokens nothing takes the fused kernel.
   Bench arm: the 3-helper peer arm at `--prefill 1024` (or a >=1024-token
   prompt), flag off vs on. Pass: prompt >3% over the ~5% run-to-run noise
   (plans/2026-10-28-whittle-remote-prefill-peers.md:134-138) and first-token
   logprob inside 0.0027 of flag-off.
3. `STRATA_MMQ_KQ=1` — no bench needed. It is already ON in every measured
   Whittle peer-prompt arm; without it the peer path refuses to run
   (prefill.cpp:2296-2301). Q36/Whittle number: 214.9 -> 241.8 tok/s (+12.5%,
   MEASURED, plans/2026-10-28-whittle-remote-prefill-peers.md:124-133).
   The gate: it opens strata_kq (src/prefill/moe_mmq.cu:44-54,
   mmq_kq_loader.cpp:59-94). Action: keep it set; the 241.8 -> 329.6 peer
   ladder is the acceptance record.
4. `STRATA_REMOTE_ZEROCOPY=0` — one env line, expected at most noise. Q36
   number: +48% decode with ZC off on the helper-ladder runbook
   (25.1-25.3 -> 36.1-37.8, plans/2026-10-17-q36-three-card-runbook.md:263-
   264). That +48% is NOT established to be the two PCIe copies
   (ASSUMED, flagged by the verifier: Q36 has no remote-opt, and on the tuned
   Q36 box ZC=1 crashes from VRAM exhaustion,
   plans/2026-10-20-q36-1p3h-optimal.md:81 — the gain is partly an
   unexplained PHB/pinned-access effect). Whittle status: the copy path is the
   pre-ZC default, byte-identical output (remote_experts.cpp:503, :565,
   :353-354). The up-leg is ~8 KiB/helper at T=1, ~5 us at the measured 6.1
   GB/s (DERIVED, not itself measured). Most of the return bytes remote-opt
   already removed (131.5 -> 16.4 MiB, plans/2026-10-28-whittle-summary.md:
   94-98). Bench arm: the 47.8 decode arm with and without the var; pass =
   decode unchanged-or-faster and text coherent. If flat, drop it from the
   serve env.

## 2. Port

Two items need code. Both are opt-in behind a new env var; the default path
stays byte-identical.

1. Deferred helper finish + worker-thread dispatch for the Flash-Next decode
   (the port of Q36's STRATA_Q36_HELPER_ASYNC / HELPER_PARALLEL_MARSHAL).
   Mechanism: today each remote helper's `begin()` runs strictly sequentially
   on the host and the host blocks on the helper's GPU finish between layers
   (src/core/expert_source.cpp:5013-5020 begin, :5214-5220 finish). Q36's
   version defers the eparts upload (finish + scatter + merge) to after the
   layer's local GPU work is issued, so local work overlaps helper compute
   (q36_engine.cu:5260-5278), and runs every helper's begin on its own worker
   thread so the dispatches overlap the main thread's primary work
   (q36_engine.cu:2858-2866, 2792-2798). Q36 number: async +40.8 ->
   60.3 tok/s (plans/2026-10-25-q36-decode-mtp-directions.md:119-124;
   control 54.5 on a differently flagged baseline, :25-27). Whittle target:
   the 1.9 ms/round host staging + 1.0 ms/round blocking wait of a 21.2 ms
   round (MEASURED, plans/2026-10-28-whittle-summary.md:221-228, which names
   HELPER_ASYNC as "written for exactly this, but engine-specific, not in
   this tree", :237-238). Files and size: ~150 lines in
   src/core/expert_source.cpp (expert_pool_dispatch_multi at :4812); no new
   kernel, no change to remote_experts.cpp. Not shared with the HIP build.
   Risks: (a) keep HELPER_RING == 1 — the remote_opt reduce is single-window
   (remote_experts.cpp:218-222); (b) the deferred finish scatters
   w.original_row_ into `out` while the CPU pool writes other rows — the
   disjoint-ownership guarantee comes from the mask, which must be computed
   from `RemoteExperts::holds()` (include/strata/core/remote_experts.hpp:
   60-61) BEFORE the workers start, not from owned_ (Q36 measured the owned_-
   read race: mask from owned_ lost 12.8% of entries, 60 -> 40 tok/s,
   plans/2026-10-25-q36-mixer-graph.md:88-91, q36_engine.cu:5219-5222);
   (c) worker threads need a DeviceScope per helper (q36_engine.cu:2810-
   2816). Decision measurement: serve-mode bench, flag off vs on, off arm
   byte-identical (the gate Q36 used, q36_engine.cu:5281-5284), and the
   ms_begin_/ms_wait_ counters (remote_experts.cpp) showing the 1.9 ms/round
   staging drop. Ceiling read off the right baseline: 47.8 tok/s (with remote-
   opt + layer placement), not 44.

2. kDenseAcc table rows for the Whittle shapes (src/prefill/gemm.cu:902-937).
   Only if item 1.1 above stays flat: the per-shape algo table was tuned on
   Q36 shapes; Whittle's dense projections (GDN attn_qkv 8192x2048,
   attn_gate 4096x2048, ssm_out 4096x2048, shared-expert 512x2048 /
   2048x512) partly have entries, but the T band at the measured chunk misses
   (DERIVED). Size: a few rows in one table, chosen by the gemmacc-rank probe
   on the P100 exactly as gemm.cu:869-872 describes; the cc 60-69 gate is
   already enforced (gemm.cu:886-888). No new kernel; HIP build also uses
   gemm.cu, but the table is CC-gated so HIP is untouched.

## 3. Does not apply here

- STRATA_Q36_ATTN_GQA / STRATA_Q36_QUEST — read only in
  src/qwen35/q36_engine.cu (attn_gqa at :2962, quest_k at :2985, dispatch at
  :4306, :569, :679); grep shows no reader in src/core, src/program,
  src/kernels, or serve/. The GQA "shared" shape Q36's =a flag buys is what
  the Flash-Next decode kernel already is the only shape: qsa_decode_attn.cu:
  24-25 hardcodes G = n_head/n_head_kv = 8 with a static_assert against the
  compiled geometry. Quest is already strictly weaker than what Whittle's
  indexer does: the selection budget is the model's idx_top_k
  (include/strata/core/geometry.hpp:51) shipped at 262144 = dense up to the
  32768 cap (docs/WHITTLE45B.md:188-189), so there is no "keep K chunks of a
  dense cache" to add — Q36's own record for Quest is +3.5% only at K=8 with
  the answer forking at token 5 at 64K (docs/QWEN36.md:614-620). Expected
  delta of setting the two vars: exactly 0. One imprecision to record: the
  cards said the vars are "not compiled into the Flash-Next path," but
  CMakeLists.txt:876 links strata_qwen35 into the strata binary and
  generate.cpp's family switch can route qwen35moe models into
  qwen35::run — they are readable, just never reached for a qwen4exp model.
  Suggested action: a docs/WHITTLE45B.md note that the two are src/qwen35
  knobs, to stop the flags from being re-tried.
- STRATA_Q36_MTP_VRAM / STRATA_Q36_MAPPED — model-conditional: both gate on
  m.has_mtp (q36_engine.cu:6908) and the Whittle file ships blk.0..blk.39
  only with no MTP block (docs/WHITTLE45B.md:147-148; geometry.hpp:66
  n_mtp_layers defaults to 0; the sidecar path refuses it, q36_engine.cu:
  1819-1822). The MTP_MAPPED mechanism (submit the whole draft chain before
  the host waits, picks through host-mapped memory) is already the Flash-Next
  native shape — src/core/mtp.cpp:1673-1707 is what q36_engine.cu:6583 ports
  FROM, so there is nothing to copy back. The Flash-Next drafter also needs
  tools/mtp_rt.py runtime files built from an MTP GGUF head (src/core/
  mtp.cpp:298-300), and no Whittle MTP head exists anywhere. Expected effect:
  zero. Do not chase a port of either flag.
- The Q36 READ_WINDOWS / CPU_POOL / FIXED_SLOTS / PROMPT_TIMING / VERIFY /
  OLD_KERNELS set (card 5) — each has a Flash-Next equivalent or is inert:
  - READ_WINDOWS: missing as a knob; the closest is `--short-read N`
    (src/program/generate.cpp:1008-1009, :3218), a different mechanism (read-
    window vs batched-prompt, not windows-per-chunk). Q36's gain was slot-
    keepalive in batched serve, not prompt speed (docs/QWEN36.md:300-303);
    the Whittle box is a single-session bench, so 0 tok/s. Skip.
  - CPU_POOL: a build gate (CMakeLists.txt:871), and the native CPU expert
    pool is always built in the main tree (`--pool-workers`, docs/DETAILS.
    md:150, :1090-1096; STRATA_POOL_CORES at generate.cpp:3453, :1143-1149).
    The Whittle box runs experts on GPU/helpers with remote-opt, so 0 effect.
  - FIXED_SLOTS: missing (the Q36 rebalance() pin, q36_engine.cu:6105-6107,
    has no counterpart in src/core/remote_experts.cpp). Zero speed effect;
    it is a determinism tool for benches. Port would be ~5 lines if bench
    pairs start disagreeing run-to-run; do not carry it into serving
    (docs/QWEN36.md:159, :295 — it was a test pin even in Q36).
  - PROMPT_TIMING: no port — STRATA_PREFILL_TIMING=1 is the equivalent
    (src/prefill/prefill.cpp:2848-2896; docs/OLDER_GPUS.md:274) and is what
    produced the 875 ms phase breakdown already.
  - VERIFY / OLD_KERNELS: no env-knob twin by design; the tree's equivalents
    are out-of-band harnesses: iq_multi_parity.cpp:244-247/:314-320,
    native_expert_parity.cpp:430-435, gdn_rec_parity.cu:5-8/:652, and the
    STRATA_VERIFY_* debug family (src/core/verify.cpp:104, :170, :313-323,
    :565). For the GDN path that dominates 21% of the Whittle prompt,
    gdn_rec_parity is the guard.
  - KV: not an env sync — the Flash-Next lever is the flag `--kv int8`
    (generate.cpp:561, :3738-3742; src/core/layer.cpp:205). Q36's +4%
    (59.3 -> 61.5, docs/QWEN36.md:459-460) was int8-over-fp32; Flash-Next's
    default store is already fp16 (generate.cpp:561), so `--kv int8` is 4x
    smaller K/V, not 2x. DERIVED: near-zero at the current config; maybe
    +1-2% only at 150K+ context. One A/B worth, not a sync.
- STRATA_Q8_SM60 / STRATA_Q8_SM60_BPS / STRATA_DP4A — already in the tree
  and already applicable to the Flash-Next native path (q8_sm60.cuh:30-83;
  native_mmvq.cu:2440-2453), but no lever on this model: the Q4_K_M pack's
  decode GEMVs are K-quants, and the only Q8_0 tensors are the 20 indexer
  projection matrices, 20 tiny GEMVs against a 22.58 ms/round decode
  (derived from the verifier's correction; `.run/whittle_tensors.json` type
  counts). Q36 number: +5.3% fp32 / +6.3% int8 decode on the Q36 box, bit-
  exact gate (plans/2026-10-20-q36-1p3h-optimal.md:89-95); the BPS scan
  found no gain on Q36 shapes (plans/2026-10-25-q36-decode-levers.md:149-
  188). Note STRATA_DP4A is not a knob at all — a compile-time macro in
  include/strata/kernels/dp4a.hpp, always compiled, selected per arch
  (sm_60 vmad fallback vs sm_61 __dp4a). Main GEMV site is native_mmvq.cu
  (34 sites), not iq_kernels.cu alone. Action: leave it off; setting it
  packs ~16 MiB and prints ~20 "strata q8 sm60: packed" lines (the static
  check is to count them, native_mmvq.cu:2800) — a misleading "it printed a
  banner, must be helping" is the only real risk. Do not repack Whittle to
  Q8_0: the Q4_K_M K-quant decode is what the box is tuned around, and the
  44.72 GiB Q8_0 figure is a size note, not a measured-slower claim
  (plans/2026-10-27-whittle-45b-q36-port.md:18).

## 4. Measured on the Whittle box

The numbers the plan leans on (192.168.1.98, P100 + 3x P104, Whittle-45B
Q4_K_M, 275-token prompt / 128-token decode):

- Prompt: 214.9 tok/s FP16 path -> 241.8 tok/s with MMQ_KQ (+12.5%); the peer-
  prefill ladder 238.7 -> 329.6 -> 333.6 tok/s, the last step moving 17% more
  expert rows to helpers (+1.2%, the prompt is not expert-bound) — MEASURED,
  plans/2026-10-28-whittle-remote-prefill-peers.md:124-133, :180-182.
  Peer arm with --remote-expert-opt: 834 ms / 329.6 tok/s.
- Decode: 44.2 tok/s without remote-opt; 47.2 with; 47.8 with remote-opt +
  `layer` placement (the +1.2% is the measured decode-table figure,
  docs/WHITTLE45B.md:86-87). MEASURED, plans/2026-10-28-whittle-summary.md:
  177, :184-186.
- Phase breakdown (875 ms prompt): expert GEMMs are 37% — gemm gate/up 215 +
  down 109 + router/shared 32; GDN recurrence 14 ms; one decode round is
  22.58 ms. MEASURED, plans/2026-10-28-whittle-remote-prefill-peers.md:
  203-218; docs/WHITTLE45B.md:170-183.
- DENSE_F16_ACC neutral: prefill 213 vs 201 ms, decode 36.95 vs 36.87, within
  noise. MEASURED, plans/2026-10-28-whittle-q36-opts.md:136; docs/
  WHITTLE45B.md:162.
- PF_F16_CUDA neutral at the short prompt: "the flag reaches the kernels but
  no measurable change." MEASURED, docs/WHITTLE45B.md:161.
- Helper host time: 1.9 ms/round begin staging + 1.0 ms/round blocking wait of
  a 21.2 ms round; HELPER_ASYNC named as "written for exactly this, but
  engine-specific, not in this tree." MEASURED, plans/2026-10-28-whittle-
  summary.md:221-228, :237-238.
- No drafter: 0 of 32 drafts accepted; one-row window when no drafter.
  MEASURED, docs/WHITTLE45B.md:147-152.
- Noise floor: ~5% run-to-run on prompt tok/s; 61.0-62.0 band on decode
  knobs. MEASURED, plans/2026-10-28-whittle-remote-prefill-peers.md:134-138;
  plans/2026-10-25-q36-decode-levers.md:149-168.

## 5. In this tree but never measured here

- STRATA_DENSE_F16_ACC — measured neutral at T=275 (above), but the 2026-
  10-28 measurement predates moving expert rows to helpers; never re-measured
  on the current peer arm. Q36 number: +17.0% prompt.
- STRATA_PF_F16_CUDA — measured neutral at the short prompt; never tried at a
  >=1024-token prompt on this box (the fused path's own trigger, prefill.
  cpp:243). Q36 number: +48% prompt.
- STRATA_REMOTE_ZEROCOPY=0 — the env switch lives at src/core/
  remote_experts.cpp:353; never armed on the Whittle box. Q36 number: +48%
  on the helper-ladder chain (25.1-25.3 -> 36.1-37.8, the 2026-10-18 two-
  helper runs).
- STRATA_Q8_SM60 / STRATA_Q8_SM60_BPS — exist and dispatch in this tree
  (native_mmvq.cu:2442, :2566-2572); never turned on for the Whittle model.
  Q36 number: +5.3% / +6.3% decode on a box whose dense matrices are Q8_0;
  here only the 20 indexer Q8_0 projections would pack.
- HELPER_ASYNC / MARSHAL (STRATA_Q36_HELPER_ASYNC=1,
  STRATA_Q36_HELPER_PARALLEL_MARSHAL=1) — missing as code; the host-side
  time they address IS measured on this box (1.9 + 1.0 ms/round, above).
  Q36 number: 40.8 -> 60.3 tok/s on the 2026-10-25 three-helper chain (control
  54.5, differently flagged baseline).
