# Whittle-DFlash2 drafter fine-tune — design spec

Goal: replace the borrowed incoai/Qwen3.6-35B-A3B-DFlash2 head with one trained on Whittle's own
hidden-state distribution, to push mean per-round acceptance from the measured 0.47–0.59 toward
0.70+ so DFlash2 becomes a clear net win on the 4-card fleet (currently a +3 % margin, E4 33.97
tok/s vs ~30.4 no-drafter).

Shapes fixed by the pack (`tools/dflash2_pack.py`): 6 layers, hidden 2048, heads 32 / kv 8 /
head-dim 128, inter 6144, 8 taps [1,6,11,16,22,27,32,37] (mean of the 4 hyper-connection
streams), rank-256 selector top-16, mask 248077, vocab 248320.

---

## 1. Weight init: init-from-incoai, two-phase

Start from `/home/Strata/df2/qwen36/model.safetensors` (1.05 GB). Whittle-Qwen-3.8-45B is merged
*from* the Qwen3.6-35B-A3B family, so its residual stream is close — the 6 layers, fc, and norms
are already a decent drafter for the right distribution; only the tap-mapping and the selector's
codebooks are mismatched. From-scratch would be slower and riskier (it is drafting, not a full
LM, so there is no self-supervised signal to bootstrap a head).

Phase A (train): freeze the two selector codebooks (248320×256 bf16 each = 254 MiB — too big to
waste grad memory on). Train fc + 6 layers + norms + conv + `candidate_selector.hidden_projection`
(~345 M params, fp32 ≈ 1.4 GB). Phase B (optional): unfreeze the codebooks for a short fine pass
only if the selector walk still dominates the loss.

---

## 2. Data: teacher-forced tap dumps, not the routing trace

`profile.bin` is a routing trace — it has no hidden states, so it cannot label the drafter. The
drafter's inputs are (a) the target's hidden states at the 8 taps and (b) the anchor token.
`tools/dflash2/dflash2_ref.py` + `run_eval.py` already walk a tap file of `(pos, token, taps)`
records.

Generate a corpus on the box with the existing hook (`src/core/verify.cpp:895`, `tap_file_`), one
pass per prompt with the engine resident:

    STRATA_TAP_FILE=/home/Strata/.run/df2/train/<p>.bin \
    STRATA_TAP_LAYERS=1,6,11,16,22,27,32,37 \
    strata generate --pack whittle45b --native Whittle-…-Q4_K_M.gguf \
      <prompt> --max-new 256

Each position's label = the greedy next-token in the same run (that is what "matching the target's
argmax" means). Size: one 275+256 run ≈ 531 rows × 8 × 2048 × 4 B ≈ 34 MB. A 10 k-prompt corpus is
~340 GB — too big to keep resident; sample **1–2 k prompts (~34–68 GB on disk)** and stream.
This is a Whittle-generated corpus, not just the 275-token bench line (that one is the *eval*,
kept out of the train set).

---

## 3. Training loop: PyTorch on the P100, two separate processes

Objective: drafter forward over the dumped taps → head logits over the 7 block positions →
cross-entropy against the target's argmax at each, plus the selector walk loss (the rank-256
bilinear score of each drafted token vs the next, `select_kernel` in `src/kernels/cuda/dflash2.cu`).
Embedding + head are frozen constants (Whittle's GGUF); only the drafter's own weights learn.

Framework: PyTorch on the P100, **not** a C++ CUDA loop — `tools/dflash2/dflash2_ref.py` is already
a faithful fp32 torch port, so training is a `requires_grad_()` on its `self.w` dict plus a CE/walk
loss. C++ would mean building backprop, autograd, the conv and ring from scratch; not worth it for
a one-off.

VRAM math (6 layers @ hidden 2048, inter 6144, batch-4 × 531-row sequences):
params ≈ 345 M fp32 = 1.4 GB; grad = 1.4 GB; AdamW states = 2.8 GB; activations at batch-4 ≈ 1–2 GB.
**Total ≈ 7 GB < the P100's 12 GB — fits.** The codebooks (254 MiB each) stay frozen in Phase A so
they are not in the optimizer.

**Co-residency is impossible:** Whittle's expert arena holds ~8 GiB of the P100, leaving ~562 MiB
free under serve load. So: (1) dump all taps in a *resident* engine pass, (2) kill the engine,
(3) train the drafter in a *separate* process reading the dumps. The drafter needs no target model
resident at train time — it only reads pre-dumped taps.

P100 is sm_60; the box has no torch (box-recon), so ship it: `pip install torch==<cu102/cu118
sm_60 wheel>` into a venv (~2.5 GB) or fall back to CPU (31 GB RAM, fp32, batch-1) which is slow
but works.

---

## 4. Acceptance target: 0.70 clears breakeven with margin; >0.80 is the real goal

From the accept-math: DFlash2 net = `41.6 ms` (8.7 drafter + 32.9 verify) buys `a×3` verified
tokens at the 30.8 ms/token fleet rate. Breakeven vs the 33.43 tok/s MTP-fallback needs **a ≥ 0.46**;
the borrowed head already sits at 0.47–0.59, which is why it is a *marginal* +3 % win — the verify
window (32.9 ms) is 3.8× the MTP's 8.47 ms, and depth-3 drafts never land (E4 accepts only to
depth-2, mean 1.03/round).

70 % is **enough** to flip DFlash2 from "marginal" to a comfortable net win:
mean/round `0.70×3 = 2.10` vs the 0.46 threshold is a 4.6× headroom, ≈ +8–10 % over the fleet
baseline — a real win, not a +1.5 % coin flip. >80 % (mean 2.40/round) is what makes depth-3 drafts
start to pay off and shrinks the amortization problem. Target **0.70 as the ship bar, 0.80 as the
stretch**; measure on the held-out 275-token bench line exactly as E4 did.

Note the sm_60 table (always-on P100-only) already *costs* DFlash2 6.4 % (0.590→0.473, R5 vs C2/D3);
any P100-only fleet box must run `STRATA_SM60_PERF=0`, so measure acceptance with the table in the
state the fleet will actually run it.

---

## 5. Decision: feasible on the box; here is the minimum experiment

**Feasible on 192.168.1.98, with two unconfirmed gates:** (a) the box has no torch — ship a
`torch==cu1xx` sm_60 wheel (~2.5 GB, pip) or use CPU; (b) disk capacity is unknown (box-recon found
no `df`); the 1–2 k-prompt tap corpus is ~34–68 GB plus the torch install. Both are checkable in
seconds before committing: `df -h` and one `pip install` smoke test.

Minimum viable experiment (a single decision, run on-box in order):

1. Dump a **1 k-prompt** tap corpus (~34 GB) with the engine-resident pass above, holding out the
   275-token bench line.
2. `pip install torch` (GPU sm_60 wheel if disk allows, else CPU). ~2.5 GB.
3. Init from incoai; train Phase A (codebooks frozen) for **~500 steps** at batch-4, fp32, on the
   P100 (~7 GB, engine not resident). 500 steps, not 100 — 100 steps over 1 k prompts is under-fit
   for a 345 M-param head; check loss plateau.
4. Pack with `tools/dflash2_pack.py` and run the **275-token bench line** exactly as the E4/R5 arms
   did (`--spec 3`, 256 decode, table in its fleet state). Ship bar: acceptance **≥ 0.70** and net
   tok/s > 33.43.

If disk < ~40 GB or the torch install fails, the box is the dump-only machine: ship the taps to a
bigger GPU (a single sm_70/80 card, ~10 GB, trains in hours) — the drafter is 7 GB and fits one
small card.
