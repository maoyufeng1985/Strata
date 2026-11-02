"""tools/whittle_mtp_train.py - retrain the MTP head on Whittle's final-R taps.

Trains the dense portion of the 19-tensor Qwen3.6 MTP head (init from
q36-mtp-f16.gguf) on Whittle-45B's final-residual taps dumped by
STRATA_TAP_FINAL_R, so fc_emb/fc_hid/hid_gamma align with Whittle's hidden
distribution instead of Qwen3.6 base's.

The drafter's engine head (mtp.cpp:1241-1265) runs the TARGET's frozen
output.weight - a [n_vocab, n_embd] matrix this dump does not carry. So this
train learns a small [n_vocab, H] proxy head on the drafter's residual h and
cross-entropy against the target's verified next-token labels. Only the dense
fc_emb/fc_hid/hid_gamma (the fused eh_proj + head norm) export to the engine;
the proxy head is a training signal, not shipped.

    python tools/whittle_mtp_train.py \
        --init /home/Strata/.run/q36mtp/q36-mtp-f16.gguf \
        --taps  /home/Strata/.run/whittle-mtp/train/runs \
        --out   /home/Strata/.run/whittle-mtp/train/whittle-mtp-head.gguf \
        --epochs 10 --batch 4 --device cpu

CPU fits: ~420 M params fp32 ~ 1.4 GB weights + grads + AdamW, well inside the
31 GB box. P100 has no CUDA torch on this box (pip has no cu build), so CPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def dequant_q6_k(raw: np.ndarray, n_vocab: int, n_embd: int) -> np.ndarray:
    """Dequantize Q6_K raw bytes to fp32 [n_vocab, n_embd].

    Q6_K layout (from iq_kernels.cu:3268-3285, block_q6_K):
      Per 256-element block (210 bytes):
        d:    1 x fp16 (2 bytes)       -- super-scale
        qs:   32 x uint8 (32 bytes)   -- low 4 bits of 128 q6 codes
        qh:   32 x uint8 (32 bytes)   -- high 2 bits of 128 q6 codes
        scales: 16 x int8 (16 bytes)  -- 16 per-group scales (the q6 sub-scales)

      Per row of 2048 elements = 8 blocks of 256 = 8 * 210 = 1680 bytes.
      The raw shape is [n_vocab, 1680] where each 1680-byte row covers 2048 values.
      Within a row: block b occupies raw[b*210 : (b+1)*210].
      Within a block: layout is d(2) + qs(32) + qh(32) + scales(16) = 82? No, 210 total.

    The actual block_q6_K layout (from ggml-quants.c):
      struct {
        float  d;          -- 4 bytes (NOT fp16, it's fp32 in this variant)
        uint8_t qs[128];  -- 128 bytes: 128 codes, low 4 bits each
        uint8_t qh[128];  -- 128 bytes: 128 codes, high 2 bits each
        int8_t  scales[16]; -- 16 bytes
      }
      Total = 4 + 128 + 128 + 16 = 276? No, that's not 210 either.

    Let me just compute: 256 values, 210 bytes.
      256 * 8 / 210 = 9.7 bits/value. That's not 6.
      256 * 6 / 8 = 192 bits = 24 bytes of codes. 210 - 24 = 186 bytes of scales/mins? No.

    The ggml Q6_K block (from ggml-quants.c, the standard one):
      256 values:
        16 x fp16 dmin  (32 bytes)
        16 x fp16 d     (32 bytes)
        32 x uint8 qs   (32 bytes)  -- 32 * 4 = 128 low bits
        32 x uint8 qh   (32 bytes)  -- 32 * 2 = 64 high bits
        32 x uint8 q2   (32 bytes)  -- the remaining 2 bits
      = 32+32+32+32+32 = 160 bytes. Still not 210.

    I'm going to just read the actual byte layout from the C code on the box.
    For now, implement a best-effort dequant that matches the engine's dq_q6_k
    at iq_kernels.cu:3268, which uses block_q6_K with .d (half), .ql (64 bytes),
    .qh (32 bytes), .scales (16 int8). That's 2+64+32+16 = 114 bytes per 256 block?
    No: x[ibs] is indexed by ibs = ib/2 (two 128-element sub-blocks per 256 block).
    So the struct is 256 values, 210 bytes:
      .d: half (2 bytes)
      .ql: 128 bytes (64*ip + il, 64*2 = 128)
      .qh: 32 bytes (32*ip + il, 32*2 = 64? no, qh[32*ip+il] with ip=0,1 -> 64 total)
      .scales: 16 int8
      Total = 2 + 128 + 64 + 16 = 210. Yes! That matches 210 bytes per 256 block.
    """
    raw = raw.astype(np.uint8)
    n_blocks = n_embd // 256
    out = np.zeros((n_vocab, n_embd), dtype=np.float32)

    # block_q6_K layout (ggml-common.h:362-367, 256 values, 210 bytes):
    #   ql[128]   at offset 0    (low 4 bits of 256 q6 codes, 64 per ip-half)
    #   qh[64]    at offset 128  (high 2 bits of 256 q6 codes, 32 per ip-half)
    #   scales[16] at offset 192  (16 per-group scales, 8 per 128-value half)
    #   d (fp16)   at offset 208  (super-block scale, 2 bytes)
    for b in range(n_blocks):
        base = b * 210
        ql = raw[:, base : base+128]      # [n_vocab, 128]
        qh = raw[:, base+128 : base+192]  # [n_vocab, 64]
        sc = raw[:, base+192 : base+208].astype(np.int32)  # [n_vocab, 16]
        # d: little-endian fp16 at the end of the block
        d16 = raw[:, base+208 : base+210].copy()  # [n_vocab, 2]
        d = (d16[:, 0].astype(np.uint16) | (d16[:, 1].astype(np.uint16) << 8)).view(np.float16).astype(np.float32)

        for ip in range(2):
            for il in range(32):
                # dq_q6_k (iq_kernels.cu:3268-3285):
                #   y = yy + 128*ip + il
                #   y[0]  = d * sc[0] * ((ql[0]  & 0xF) | (((qh >> 0) & 3) << 4)) - 32
                #   y[32] = d * sc[2] * ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32
                #   y[64] = d * sc[4] * ((ql[0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32
                #   y[96] = d * sc[6] * ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32
                # where ql_ptr = block.ql + 64*ip + il, qh = block.qh[32*ip+il],
                #       sc_ptr = block.scales + 8*ip + il//16
                for j in range(4):
                    # j=0: y[0],  j=1: y[32],  j=2: y[64],  j=3: y[96]
                    # ql: j even -> ql[0] (low nibble for j=0, high nibble for j=2)
                    #     j odd  -> ql[32] (low nibble for j=1, high nibble for j=3)
                    ql_offset = 32 * (j & 1)
                    ql_shift = 4 if j >= 2 else 0
                    ql_idx = 64 * ip + il + ql_offset
                    qh_idx = 32 * ip + il
                    sc_idx = 8 * ip + il // 16 + 2 * j
                    out_pos = b * 256 + 128 * ip + il + 32 * j

                    ql_raw = ql[:, ql_idx].astype(np.uint32)
                    ql_bits = (ql_raw >> ql_shift) & 0xF
                    qh_bits = ((qh[:, qh_idx].astype(np.uint32)) >> (2 * j)) & 3
                    s = sc[:, sc_idx].astype(np.float32)
                    q6 = ql_bits | (qh_bits << 4)
                    q_val = (q6.astype(np.int32) - 32).astype(np.float32)
                    out[:, out_pos] = d * s * q_val
    return out


def load_target_head_fp32(gguf_path: str, n_vocab: int, n_embd: int) -> np.ndarray:
    """Read the target's output.weight (Q6_K) and dequantize to fp32 [n_vocab, n_embd]."""
    import sys as _sys
    _sys.path.insert(0, "/home/Strata/build/_deps/strata_llamacpp-src/gguf-py")
    import gguf
    r = gguf.GGUFReader(gguf_path)
    for t in r.tensors:
        if t.name == "output.weight":
            raw = np.asarray(t.data)
            print(f"target output.weight: tensor_type={t.tensor_type}, raw shape={raw.shape}, dtype={raw.dtype}")
            if t.tensor_type == 0:  # F32
                return raw.astype(np.float32)
            if raw.dtype == np.float16:
                return raw.astype(np.float32)
            # Q6_K: dequantize
            # raw shape is [n_vocab, 1680] where 1680 = 8 blocks * 210 bytes
            # But we need to verify: is it [n_vocab, 1680] or [n_vocab * 1680]?
            if raw.ndim == 2 and raw.shape[0] == n_vocab:
                print(f"dequantizing Q6_K [{n_vocab}, {raw.shape[1]}] -> [{n_vocab}, {n_embd}]")
                return dequant_q6_k(raw, n_vocab, n_embd)
            # Fallback: flat array
            total = raw.size
            expected = n_vocab * (n_embd // 256) * 210
            if total == expected:
                raw2 = raw.reshape(n_vocab, -1)
                print(f"dequantizing Q6_K flat -> [{n_vocab}, {n_embd}]")
                return dequant_q6_k(raw2, n_vocab, n_embd)
            raise ValueError(f"output.weight raw shape {raw.shape} does not match expected Q6_K layout")
    sys.exit(f"{gguf_path}: no output.weight found")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="q36-mtp-f16.gguf (the Qwen3.6 rt3 head) - the init")
    ap.add_argument("--taps", required=True, help="directory of STRATA_TAP_FINAL_R corpus files (r1..r100)")
    ap.add_argument("--out", required=True, help="output GGUF with the retrained 19-tensor head")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-rows", type=int, default=0, help="cap rows (0 = all)")
    ap.add_argument("--target-head", default=None,
                    help="Whittle target GGUF with the real output.weight (Q6_K) - replace the proxy head_w "
                         "with the engine's actual head so fc trains against the real target logits")
    return ap.parse_args()


def load_init_numpy(gguf_path: str) -> dict:
    """Read the init tensors as a dict of plain numpy float32 arrays (no torch, no graph).

    eh_proj is the fused [H, 2H] projection (the engine's fc_embedding | fc_hidden split).
    post_attention_norm is the engine's pre_fc_norm_hidden gamma (mtp.cpp:954) - the norm
    applied to the residual branch right before fc_hidden. It is the one norm the training
    must align: the engine multiplies rmsnorm(pool) * gamma, and the dump's final-R is the
    engine's Rin_ itself, so gamma is the exact quantity the drafter uses."""
    import sys as _sys
    _sys.path.insert(0, "/home/Strata/build/_deps/strata_llamacpp-src/gguf-py")
    import gguf
    r = gguf.GGUFReader(gguf_path)
    tens = {t.name: t for t in r.tensors}
    blk = None
    for n in tens:
        if n.endswith("nextn.eh_proj.weight"):
            blk = n[: -len("nextn.eh_proj.weight")]
            break
    if blk is None:
        sys.exit(f"{gguf_path}: no nextn.eh_proj.weight - not a Qwen3.6-style MTP block")
    out = {}
    for suffix in ["nextn.eh_proj.weight", "post_attention_norm.weight"]:
        if blk + suffix not in tens:
            sys.exit(f"{gguf_path}: {blk + suffix} missing")
        t = tens[blk + suffix]
        data = np.asarray(t.data)
        if t.tensor_type == 0:
            arr = data.astype(np.float32)
        elif data.dtype == np.float16:
            arr = data.astype(np.float32)
        elif data.dtype == np.uint16:
            arr = (data.astype(np.uint32) << 16).view(np.float32)
        else:
            raise ValueError(f"{blk + suffix}: unsupported dtype")
        out[suffix] = np.ascontiguousarray(arr, dtype=np.float32)
    return out


def parse_tap_file(path: str) -> tuple:
    """One STRATA_TAP_FINAL_R file: magic, n_embd, hc, then rows of
    (int32 pos, int32 token, hc*n_embd floats). Returns (positions, tokens, finals)
    where finals is [n_rows, hc, n_embd] fp32."""
    import struct
    d = open(path, "rb").read()
    magic, = struct.unpack_from("<i", d, 0)
    if magic != 0x46455246:
        raise ValueError(f"{path}: bad magic 0x{magic:08x}")
    n_embd, hc = struct.unpack_from("<ii", d, 4)
    row_bytes = 8 + n_embd * hc * 4
    off = 8
    pos, tok, fin = [], [], []
    while off + row_bytes <= len(d):
        p, t = struct.unpack_from("<ii", d, off)
        fr = np.frombuffer(d, dtype=np.float32, count=hc * n_embd, offset=off + 8)
        pos.append(p); tok.append(t); fin.append(fr.reshape(hc, n_embd))
        off += row_bytes
    return np.array(pos), np.array(tok), np.stack(fin)


def main() -> int:
    a = parse_args()
    import torch
    device = a.device if (torch.cuda.is_available() and "cuda" in a.device) else "cpu"
    print(f"device: {device}")

    # 1. load the init dense head as plain numpy (no autograd graph at load time).
    H, HC = 2048, 4
    init_np = load_init_numpy(a.init)
    eh_np = init_np["nextn.eh_proj.weight"]          # [H, 2H]
    # The engine's hidden-branch norm (mtp.cpp:954): rmsnorm(pool) * pre_fc_norm_hidden, where
    # pre_fc_norm_hidden = post_attention_norm + 1 (the checkpoint stores w-1, mtp_rt_qwen36.py
    # norm1(); this gamma is the exact quantity the drafter multiplies the residual by).
    hid_np = init_np["post_attention_norm.weight"]   # [H]
    # Build fresh leaf tensors the optimizer can own. fc_emb/fc_hid are the two halves
    # of the fused eh_proj (mtp.cpp:949, 958, 960 applies both and adds).
    fc_emb = torch.tensor(eh_np[:, :H], dtype=torch.float32, device=device)
    fc_hid = torch.tensor(eh_np[:, H:], dtype=torch.float32, device=device).requires_grad_(True)
    hid_gamma = torch.tensor(hid_np + np.float32(1.0), dtype=torch.float32, device=device).requires_grad_(True)
    for nm, t in (("fc_emb", fc_emb), ("fc_hid", fc_hid), ("hid_gamma", hid_gamma)):
        if not t.is_leaf:
            sys.exit(f"{nm} is not a leaf tensor - cannot optimize")

    # 2. load the dumped corpus
    tap_dir = Path(a.taps)
    bins = [p for p in tap_dir.glob("r*")
            if p.suffix in ("", ".bin") and not p.name.endswith((".out", ".err"))]
    if not bins:
        sys.exit(f"no corpus files in {a.taps} (expected r1..rN)")
    bins.sort(key=lambda p: int("".join(c for c in p.name if c.isdigit()) or 0))
    finals, toks = [], []
    for bp in bins:
        p, t, f = parse_tap_file(str(bp))
        finals.append(f); toks.append(t)
    finals = np.concatenate(finals)   # [rows, HC, H]
    toks = np.concatenate(toks)       # [rows]
    n_files = len(bins)
    print(f"loaded {n_files} corpus files, {len(finals)} rows")
    if a.max_rows:
        keep = min(a.max_rows, len(finals))
        finals, toks = finals[:keep], toks[:keep]
        print(f"capped to {keep} rows")

    # 3. labels: each corpus file is one continuous target run. Row i's label is the input
    #    token of row i+1 (the target's verified-and-committed next token; the dump runs
    #    greedily, so the committed token IS the target's argmax - the exact recipe the
    #    spec wants, no STRATA_LOGPOS needed). The final row of each file has no next ->
    #    masked. Files are independent: a boundary row is never a label for the next file.
    allf, labels, valid = [], [], []
    off = 0
    for bp in bins:
        p, t, f = parse_tap_file(str(bp))
        nf = len(f)
        allf.append(f)
        lab = np.zeros(nf, dtype=np.int64)
        if nf > 1:
            lab[: nf - 1] = t[1:]
        lbl = np.ones(nf, dtype=bool)
        lbl[nf - 1] = False
        labels.append(lab); valid.append(lbl)
    finals = np.concatenate(allf)
    labels = np.concatenate(labels)
    valid = np.concatenate(valid)
    print(f"rows {len(finals)}, valid labels {valid.sum()}/{len(finals)}")
    if valid.sum() == 0:
        sys.exit("no valid labels - the corpus is empty or every file is a single row")

    n_rows = len(finals)
    f_tensor = torch.tensor(finals, dtype=torch.float32, device=device)
    l_tensor = torch.tensor(labels, dtype=torch.int64, device=device)
    v_mask = torch.tensor(valid, dtype=torch.bool, device=device)

    # 4. the forward + the LM head. The engine's MTP drafter (mtp.cpp:1241) runs the TARGET's
    #    frozen output.weight for the logits, NOT a proxy. So when --target-head is given,
    #    load the real target head (Q6_K -> fp32) and use it frozen; only fc_hid + hid_gamma
    #    are trained (fc_emb is frozen - the dump does not carry the token-embedding branch).
    #
    # The engine's true MTP layer forward (mtp.cpp:947-960, zero hyper-connection fills from
    # mtp_rt_qwen36.py, rs=1, q8_1 cancels):
    #   emb_out = fc_emb @ (rmsnorm(emb) * pre_fc_norm_embedding)
    #   hid_out = fc_hid @ (rmsnorm(pool) * pre_fc_norm_hidden)      <- trained here
    #   R = emb_out + hid_out + per-layer attention/mlp signals
    #
    # The dump's final-R IS the engine's Rin_ (verify.cpp final_R == mtp.cpp window_R, the
    # same pointer). The token-embedding branch and the per-layer attention/mlp signals are
    # not in the dump, so the train targets the hidden-branch mapping only:
    #   h ~ fc_hid @ (rmsnorm(pool(finals)) * pre_fc_norm_hidden)
    # fc_emb stays at its init value (frozen - no training signal for it); hid_gamma is
    # initialized to post_attention_norm + 1, the exact gamma the drafter multiplies by.
    def rmsnorm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def forward(finals: torch.Tensor) -> torch.Tensor:
        """finals [B, HC, H] -> pool [B, H] -> rmsnorm * hid_gamma -> fc_hid -> h [B, H].
        hid_gamma is pre_fc_norm_hidden (post_attention_norm + 1), the drafter's exact
        residual norm. fc_emb is frozen at init (the dump carries no embedding branch)."""
        pooled = finals.mean(1)
        x_normed = rmsnorm(pooled) * hid_gamma
        return x_normed @ fc_hid.T

    if a.target_head:
        # Use the real target LM head (Q6_K -> fp32), frozen. Only fc_hid + hid_gamma train
        # (fc_emb is frozen at init - the dump does not carry the token-embedding branch).
        print(f"loading target head from {a.target_head} ...")
        head_np = load_target_head_fp32(a.target_head, 248320, H)
        print(f"target head: shape={head_np.shape}, dtype={head_np.dtype}")
        head_w = torch.tensor(head_np, dtype=torch.float32, device=device)
        params = [fc_hid, hid_gamma]
        for i, p in enumerate(params):
            if not p.is_leaf:
                sys.exit(f"param[{i}] is not a leaf - cannot optimize")
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
        print("training with frozen target head (only fc_hid + hid_gamma learn)")
    else:
        # Fallback: the old proxy head (random [vocab, H], trained alongside fc).
        vocab = 248320
        head_w = torch.empty(vocab, H, dtype=torch.float32, device=device).normal_(0, 1e-3).requires_grad_(True)
        params = [fc_emb, fc_hid, hid_gamma, head_w]
        for i, p in enumerate(params):
            if not p.is_leaf:
                sys.exit(f"param[{i}] is not a leaf - cannot optimize")
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
        print("training with proxy head (fc + head_w learn; use --target-head for the real target head)")

    # 5. train
    print("training...")
    for ep in range(a.epochs):
        order = torch.randperm(n_rows)
        tot_loss, n_batch = 0.0, 0
        for b in range(0, n_rows, a.batch):
            idx = order[b: b + a.batch]
            sel = v_mask[idx]
            x = f_tensor[idx]
            lab = l_tensor[idx]
            h = forward(x)
            if int(sel.sum()) > 0:
                s = sel
                # Gather the label rows (the verified next-token ids) and their h rows,
                # then a single [n_sel, H] @ [H, vocab] GEMM -> [n_sel, vocab]. One 2D
                # mm + one CE over that 2D batch, not a per-row loop (which was O(vocab)
                # python work per row and made CPU training stall on the 248k-vocab GEMM).
                h_sel = h[s]
                logits = torch.mm(h_sel, head_w.T)
                loss = torch.nn.functional.cross_entropy(logits, lab[s], reduction="mean")
            else:
                loss = torch.zeros((), device=device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item(); n_batch += 1
            if n_batch % 100 == 0:
                print(f"  ep {ep} batch {n_batch} loss {tot_loss / n_batch:.4f}", flush=True)
        print(f"epoch {ep}: avg loss {tot_loss / n_batch:.4f}", flush=True)

    # 6. export: write the retrained dense head back into a 19-tensor GGUF in the
    #    q36-mtp-f16 layout (the engine's mtp_rt_qwen36.py reads this). fc_embedding /
    #    fc_hidden re-fuse into eh_proj; the head norm re-stores w-1 (the checkpoint
    #    convention, memory: q36-mtp-sidecar-layout). The 17 other tensors (attention,
    #    routed experts) are copied verbatim from the init - the Phase-A codebook is
    #    frozen and too expensive to retrain on this box.
    import sys as _sys
    _sys.path.insert(0, "/home/Strata/build/_deps/strata_llamacpp-src/gguf-py")
    import gguf
    from gguf import GGUFWriter
    # the pinned gguf-py exports the constants on GGMLQuantizationType, not as module names.
    QF32 = gguf.GGMLQuantizationType.F32
    QF16 = gguf.GGMLQuantizationType.F16
    QB16 = gguf.GGMLQuantizationType.BF16

    def to_f32(t):
        arr = np.asarray(t.data)
        if t.tensor_type == 0:
            return arr.astype(np.float32)
        if arr.dtype == np.float16:
            return arr.astype(np.float32)
        if arr.dtype == np.uint16:            # BF16 bits
            return (arr.astype(np.uint32) << 16).view(np.float32)
        raise ValueError(t.name)

    # keep the engine-read dtype: 0/2 (F32) -> F32, 1 (F16) -> F16, 3 (BF16) -> BF16.
    def qtype(t):
        tt = t.tensor_type
        if tt == 0:
            return QF32
        if tt == 1:
            return QF16
        if tt == 3:
            return QB16
        raise ValueError(t.name)

    init_r = gguf.GGUFReader(a.init)
    init_names = [t.name for t in init_r.tensors]
    init_blk = next(n for n in init_names if n.endswith("nextn.eh_proj.weight"))[: -len("nextn.eh_proj.weight")]
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(out_path), "qwen36_mtp_retrain")
    w.add_uint32("general.context_length", 4096)
    eh_re = torch.cat([fc_emb, fc_hid], dim=1).detach().cpu().numpy().astype(np.float32)
    w.add_tensor(init_blk + "nextn.eh_proj.weight", eh_re, raw_dtype=QF32)
    hid_g_re = hid_gamma.detach().cpu().numpy().astype(np.float32) - np.float32(1.0)  # back to w-1
    # The retrained norm goes to post_attention_norm.weight - the repack's real source for
    # pre_fc_norm_hidden (mtp_rt_qwen36.py RENAME:84). shared_head_norm feeds only the
    # mixer's hc_norm fill; leave it at its init value.
    w.add_tensor(init_blk + "post_attention_norm.weight", hid_g_re, raw_dtype=QF32)
    for t in init_r.tensors:
        if t.name in (init_blk + "nextn.eh_proj.weight", init_blk + "post_attention_norm.weight"):
            continue
        arr = to_f32(t)
        w.add_tensor(t.name, arr, raw_dtype=qtype(t))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.out} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("re-pack with: python tools/mtp_rt_qwen36.py --gguf", a.out,
          "--out /home/Strata/mtp/qwen36/rt3-retrain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
