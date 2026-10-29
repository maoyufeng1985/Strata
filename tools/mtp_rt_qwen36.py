"""tools/mtp_rt_qwen36.py - Qwen3.6-35B-A3B's MTP block as the Flash-Next MtpDrafter's runtime directory.

    python tools/mtp_rt_qwen36.py --gguf q36-mtp-f16.gguf --out rt [--experts q2_0|q4_0|q8_0]

Why this is not tools/mtp_rt.py.  That one reads Flash-Next's own MTP block: 31 tensors, 2560 wide, with a
QSA indexer and hyper-connection mixers (H, FF, NE = 2560, 640, 512).  Qwen3.6's MTP block is a plain
decoder layer: 19 tensors, 2048 wide, no indexer (src/core/mtp.cpp:373 - "dense attention: no indexer state
is read"), no hyper-connections (Qwen3.6's config has no hc_count).  Same engine, same draft loop, different
tensor list.

The hyper-connections, and why the missing ones are exact rather than an approximation.  The engine's MTP
path is HC-conditional only in the sense that it always wraps a layer in a hyper-connection read
(src/core/mtp.cpp:962-975 for attention, :1092 for the MLP).  With a few zero matrices that read becomes the
identity, which is what a non-HC layer's single stream needs:

  * w_down = 0  ->  lo = silu(w_down . xn / HC) = 0        (kernels/cuda/fused_gr.cu:186)
  * w_up   = 0  ->  sigmoid(w_up . lo) = 0.5               (:205)
  * so  mixed = (1/HC) * sum_c R_c * w_norm_c * rs_c * 0.5 (:204-209)
  * and w_norm = gamma, the same row broadcast to every stream, factors OUT of that mean, leaving
    mixed = 0.5 * gamma * mean_c(R_c * rs_c) - the pooled normalization Qwen3.6's own layer does.
  * w_inject = 0 -> inject = 0, and the residual write R += bo * 2 * sigmoid(inject / HC) is exactly + bo
    (:200: injectable = false), so the 0.5 above cancels exactly.
  * the same 0.5 then rides through native_quantize_q8_1, which divides by amax and multiplies back
    (d = amax/127, q = round(x/d), result q*d), so it cancels there too.

So the only tensor the engine needs that Qwen3.6 does not have is the 1-D gamma: `hc_norm`.  The three
mixers are broadcast to every stream because input_mix_weight_down/up have shape [hc_lr][HC*n_embd] /
[HC*n_embd][hc_lr] and a zero fill of those shapes is what makes lo and the gate zero.

`block_inject_weight` (attn :970, mlp :1114) is zero-filled and stays BF16 - w_inject = null and w_inject = 0
are the same result there (fused_gr.cu:163 skips the inject rows when w_inject is null).

What comes from where
  mtp.fc.weight [2048, 4096]         -> fc_embedding.weight [2048, 2048] (cols 0..2047)
                                       fc_hidden.weight    [2048, 2048] (cols 2048..4095)
                                       the engine applies both to a 2048-wide input and adds the results
                                       (src/core/mtp.cpp:949, :958, :960), so the column split is what the
                                       checkpoint's single fused projection means.
  mtp.layers.0.*                     -> self_attn.*, mlp.*, pre_fc_norm_*, mlp.gate.weight
  (nothing)                          -> the 4 hyper-connection tensors, built here
  mtp.norm.weight, mtp.layers.0.mlp.experts.gate_up_proj's gate half, mtp.layers.0.mlp.experts.down_proj
                                     -> used as they are
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
import gguf  # noqa: E402

# the engine's draft-geometry constants that this file has to hit (kernels/cpu/expert.hpp: BLOB = 3*(H*FF*BB/QK))
H, FF, NE, QK = 2048, 512, 256, 64
BB = 18                                  # Q2_0: one fp16 scale + 16 bytes of 2-bit codes per 64 values
BLOB = 3 * (H * FF * BB // QK)           # gate + up + down of one expert, Q2_0 coded
HC, HC_LR = 4, 320                       # Whittle's hc_count / hc_lowrank: what the engine's fused read is built for
N_EMBD = H

Q8 = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
      "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.shared_expert.gate_proj.weight",
      "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"}

# Source suffix (under the `blk.<n>.` prefix, llama.cpp's spelling) -> destination name in dense.txt (the engine's
# spelling).  The sidecar is a GGUF, so its tensors carry the GGUF names, not the checkpoint's `mtp.layers.0.*`.
RENAME = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "attn_norm.weight": "pre_fc_norm_embedding.weight",
    "post_attention_norm.weight": "pre_fc_norm_hidden.weight",
}


def q8_0(x: np.ndarray) -> bytes:
    """ggml quantize_row_q8_0_ref over rows of a 2-D float32 array: 32-value blocks, fp16 d = amax / 127."""
    b = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
    v = b * inv[:, None]
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)
    out = np.empty((b.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def q2_0(w: np.ndarray) -> np.ndarray:
    """[rows, n] float32 (n = H, a multiple of QK) -> Q2_0 blocks as [rows][n/QK][18]: fp16 d then 16 bytes of
    2-bit codes, with the MSE scale search of tools/mtp_pack.py (candidates span amax/2, where the +2 level
    reaches the max, to amax, ggml's reference).  The rows are the weight matrix's output rows, so a row's
    blocks are contiguous - the layout tools/mtp_rt.py:blob_of then interleaves gate and up on."""
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] % QK:
        raise ValueError("q2_0: expected [rows, multiple of %d], got %s" % (QK, w.shape))
    x = w.reshape(-1, QK)
    amax = np.abs(x).max(axis=1, keepdims=True)
    best_err = np.full((x.shape[0], 1), np.inf, dtype=np.float32)
    best_d = np.zeros_like(amax)
    for f in np.linspace(0.5, 1.0, 17, dtype=np.float32):
        d = amax * f
        inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
        q = np.clip(np.rint(x * inv), -1, 2)
        err = ((q * d - x) ** 2).sum(axis=1, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_d = np.where(better, d, best_d)
    d16 = best_d.astype(np.float16)
    d = d16.astype(np.float32)                                   # quantize against the STORED scale
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    codes = (np.clip(np.rint(x * inv), -1, 2) + 1).astype(np.uint8)          # 0..3
    c = codes.reshape(-1, 16, 4)
    packed = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)
    out = np.empty((x.shape[0], BB), dtype=np.uint8)
    out[:, :2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed
    return out.reshape(w.shape[0], w.shape[1] // QK, BB)


def blob_of(gate: np.ndarray, up: np.ndarray, dn: np.ndarray) -> bytes:
    """One expert's Q2_0 bytes in the engine's layout: gate/up rows interleaved (2r = gate r, 2r+1 = up r),
    then down; the codes plane of both, then the scales plane of both (tools/mtp_rt.py:blob_of)."""
    inter = np.empty((2 * FF, H // QK, BB), np.uint8)
    inter[0::2] = q2_0(gate)
    inter[1::2] = q2_0(up)
    dnb = q2_0(dn)
    parts = [inter[:, :, 2:].tobytes(), dnb[:, :, 2:].tobytes(), inter[:, :, :2].tobytes(), dnb[:, :, :2].tobytes()]
    out = b"".join(parts)
    assert len(out) == BLOB, (len(out), BLOB)
    return out



def to_f32(t) -> np.ndarray:
    """A GGUF tensor's values as float32 (F32 / F16 / BF16 all widened exactly)."""
    data = np.asarray(t.data)
    if t.tensor_type == 0:                      # F32
        return data.astype(np.float32)
    if data.dtype == np.float16:
        return data.astype(np.float32)
    if data.dtype == np.uint16:                 # BF16 bits
        return (data.astype(np.uint32) << 16).view(np.float32)
    raise ValueError("%s: unsupported dtype %s (type %s)" % (t.name, data.dtype, t.tensor_type))


# The checkpoint stores every RMSNorm weight as (w - 1); the engine multiplies by w directly, so a norm read from
# the source needs +1 (memory: q36-mtp-sidecar-layout, measured residual exactly 0).  Only the norms: not the
# router, not the shared-expert gate, and not the zero fills.
def norm1(t) -> np.ndarray:
    """A source RMSNorm gamma, shifted to the engine's convention."""
    a = to_f32(t)
    if a.ndim != 1:
        raise ValueError("%s: a norm should be 1-D, got %s" % (t.name, a.shape))
    return a + np.float32(1.0)


def hc_tile(g: np.ndarray, what: str) -> np.ndarray:
    """A [N] gamma as the [HC*N] one the engine's hyper-connection reads take.

    `native_qsa_rms_norm_weighted` is called with n_cols = HC*N and the kernel checks the gamma against
    n_cols * 4 bytes (kernels/cuda/native_qsa.cu:138), so a [N] tensor there is a 3/4-vector OVERREAD of
    whatever sits next in dense.bin - silently wrong norms, not a load error.  `fused_gr`'s w_norm is
    `(hc * n_embd) f32` for the same reason (kernels/fused_gr.hpp:30).

    The value is the same gamma for every stream: one RMS over the pooled HC*N row scaled elementwise by a
    gamma repeated HC times is exactly "normalize the pooled vector with this gamma"."""
    if g.shape != (N_EMBD,):
        raise ValueError("%s: expected a [%d] gamma, got %s" % (what, N_EMBD, g.shape))
    return np.tile(g, HC)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="q36-mtp-f16.gguf (tools/q36_mtp_extract.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--experts", choices=["q2_0"], default="q2_0",
                    help="the routed experts' format; only q2_0 is read by the engine's MTP pool")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    r = gguf.GGUFReader(a.gguf)
    tens = {t.name: t for t in r.tensors}
    blk = None
    for n in tens:
        if n.endswith("nextn.eh_proj.weight"):
            blk = n[: -len("nextn.eh_proj.weight")]
            break
    if blk is None:
        sys.exit("%s: no <blk>.nextn.eh_proj.weight - not a Qwen3.6-style MTP block" % a.gguf)
    print("source block prefix: %r" % blk)

    def get(suffix):
        t = tens.get(blk + suffix)
        if t is None:
            sys.exit("%s: missing %s" % (a.gguf, blk + suffix))
        return t

    # ---- the routed experts: one blob per expert, the engine's Q2_0 layout
    # gguf-py reports numpy shapes with the RAW dims reversed, so an expert tensor written as [E, rows, cols] reads
    # back as [E, rows, cols] with the expert already leading - no transpose (memory: q36-mtp-sidecar-layout).
    gate = to_f32(get("ffn_gate_exps.weight"))
    up = to_f32(get("ffn_up_exps.weight"))
    dn = to_f32(get("ffn_down_exps.weight"))
    if gate.shape != (NE, FF, H):
        sys.exit("ffn_gate_exps.weight is %s, expected [%d, %d, %d]" % (gate.shape, NE, FF, H))
    if up.shape != (NE, FF, H):
        sys.exit("ffn_up_exps.weight is %s, expected [%d, %d, %d]" % (up.shape, NE, FF, H))
    if dn.shape != (NE, H, FF):
        sys.exit("ffn_down_exps.weight is %s, expected [%d, %d, %d]" % (dn.shape, NE, H, FF))
    with open(out / "experts.bin", "wb") as f:
        for e in range(NE):
            f.write(blob_of(gate[e], up[e], dn[e]))
            if (e + 1) % 64 == 0:
                print("  %d/%d experts" % (e + 1, NE), flush=True)
    print("experts.bin %.1f MB (%d experts of %d B)" % (NE * BLOB / 1e6, NE, BLOB))

    # ---- the hyper-connection fills: exact, see the module docstring
    # The FINAL mixer (before the head) is the one place where Qwen3.6 does have a norm the engine would otherwise
    # skip: Flash-Next's block has no `mtp.norm.weight`, it lets `hyper_connection_mixer` produce the head input.
    # Qwen3.6's head input is `norm(R)`, so that norm's gamma goes into the mixer's hc_norm and the zero down/up
    # makes the mixer the pooled-RMS reduction above - the same construction as the two block mixers.
    z = {
        "hyper_connection_mixer.hc_norm.weight": ("f32", hc_tile(norm1(get("nextn.shared_head_norm.weight")), "hyper_connection_mixer.hc_norm")),
        "hyper_connection_mixer.input_mix_weight_down.weight": ("bf16", np.zeros((HC_LR, HC * N_EMBD), np.float32)),
        "hyper_connection_mixer.input_mix_weight_up.weight": ("bf16", np.zeros((HC * N_EMBD, HC_LR), np.float32)),
    }
    # The layer norms, already in the engine's convention: they are the source of both the layer's own RMSNorm
    # and (broadcast) the hyper-connection gamma, so the +1 lands once and both readers get it.
    # The four norms read from the source, and whether the engine indexes them over HC*N (tile) or N.
    #   pre_fc_norm_embedding: `native_qsa_rms_norm_weighted(..., (int) N, T, ...)`  -> N
    #   pre_fc_norm_hidden:    `native_qsa_rms_norm_weighted(..., (int)(HC*N), T, ...)` -> HC*N
    NORM_SRC = {
        "pre_fc_norm_embedding.weight": False,
        "pre_fc_norm_hidden.weight": True,
        "self_attn.q_norm.weight": False,
        "self_attn.k_norm.weight": False,
    }
    for side in ("attn", "mlp"):
        g = norm1(get("attn_norm.weight" if side == "attn" else "post_attention_norm.weight"))
        if g.shape != (N_EMBD,):
            sys.exit("%s_norm.weight is %s, expected [%d]" % (side, g.shape, N_EMBD))
        z["%s_hyper_connection.hc_norm.weight" % side] = ("f32", hc_tile(g, "%s_hyper_connection.hc_norm" % side))
        z["%s_hyper_connection.input_mix_weight_down.weight" % side] = ("bf16", np.zeros((HC_LR, HC * N_EMBD), np.float32))
        z["%s_hyper_connection.input_mix_weight_up.weight" % side] = ("bf16", np.zeros((HC * N_EMBD, HC_LR), np.float32))
        z["%s_hyper_connection.block_inject_weight.weight" % side] = ("bf16", np.zeros((HC, HC * N_EMBD), np.float32))

    # ---- fc: the checkpoint's one fused projection, split into the engine's two 2048-wide ones
    fc = to_f32(get("nextn.eh_proj.weight"))
    if fc.shape != (N_EMBD, 2 * N_EMBD):
        sys.exit("nextn.eh_proj.weight is %s, expected [%d, %d]" % (fc.shape, N_EMBD, 2 * N_EMBD))
    fc_emb, fc_hid = fc[:, :N_EMBD], fc[:, N_EMBD:]

    lines, off = [], 0
    with open(out / "dense.bin", "wb") as f:
        def emit(name, kind, arr):
            nonlocal off
            if kind == "f32":
                raw, rows, cols = np.ascontiguousarray(arr, np.float32).tobytes(), 1, arr.size
            elif kind == "q8_0":
                rows, cols = arr.shape
                raw = q8_0(arr)
            else:
                rows, cols = arr.shape
                raw = (np.ascontiguousarray(arr, np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
            pad = (-off) % 256
            f.write(b"\0" * pad)
            off += pad
            f.write(raw)
            lines.append("%s %s %d %d %d %d" % (name, kind, rows, cols, off, len(raw)))
            off += len(raw)

        emit("fc_embedding.weight", "q8_0", fc_emb)
        emit("fc_hidden.weight", "q8_0", fc_hid)
        done = {"fc_embedding.weight", "fc_hidden.weight"}
        for src, dst in RENAME.items():
            if dst in NORM_SRC:
                arr = norm1(get(src))
                if NORM_SRC[dst]:
                    arr = hc_tile(arr, dst)
            else:
                arr = to_f32(get(src))
            if arr.ndim == 1:
                emit(dst, "f32", arr)
            else:
                emit(dst, "q8_0" if dst in Q8 else "bf16", arr)
            done.add(dst)
        for name, (kind, arr) in z.items():
            emit(name, kind, arr)
            done.add(name)

    (out / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("dense.bin %.1f MB, %d tensors" % (off / 1e6, len(lines)))

    # ---- the load-time proof: the engine's required[] list, plus every f32/bf16 name it dereferences
    REQUIRED = ["fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.shared_expert.gate_proj.weight",
                "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight",
                "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", "mlp.shared_expert_gate.weight",
                "hyper_connection_mixer.hc_norm.weight", "hyper_connection_mixer.input_mix_weight_down.weight",
                "hyper_connection_mixer.input_mix_weight_up.weight",
                "attn_hyper_connection.hc_norm.weight", "attn_hyper_connection.input_mix_weight_down.weight",
                "attn_hyper_connection.input_mix_weight_up.weight", "attn_hyper_connection.block_inject_weight.weight",
                "mlp_hyper_connection.hc_norm.weight", "mlp_hyper_connection.input_mix_weight_down.weight",
                "mlp_hyper_connection.input_mix_weight_up.weight", "mlp_hyper_connection.block_inject_weight.weight"]
    missing = [n for n in REQUIRED if n not in done]
    if missing:
        sys.exit("BUG: the engine would dereference these and they are not written: %s" % missing)
    print("all %d names the loader reads are present" % len(REQUIRED))
    return 0


if __name__ == "__main__":
    sys.exit(main())
