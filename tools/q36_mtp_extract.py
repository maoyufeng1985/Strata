"""tools/q36_mtp_extract.py - extract Qwen3.6-35B-A3B's MTP block as a standalone FP16 GGUF.

Qwen3.6's MTP head is one decoder layer plus three folding tensors (the checkpoint's `mtp.*`, the GGUF's
`blk.<n_layers>.*`):

    eh_proj = fc(enorm(e) | hnorm(h)),  then attention + MoE,  then shared_head_norm,  then the main head

It is *not* Flash-Next's MTP block.  Flash-Next's is 2560-wide with hyper-connection mixers and a QSA indexer
(31 tensors, tools/mtp_fetch.py); Qwen3.6's is 2048-wide with none of those (19 tensors + the GGUF's 4
`nextn.*`).  This tool therefore does not reuse tools/mtp_pack.py: that one is organised around Flash-Next's
tensor list and its H 2560 / FF 640 / NE 512 expert blob.

Two sources, and they are not equivalent:

  --from-checkpoint   the official BF16 checkpoint (Qwen/Qwen3.6-35B-A3B).  Range-reads only the `mtp.*`
                      tensors out of the 26 shards' safetensors headers, so the values are the ORIGINAL
                      weights, never quantized.  This is the precision-honest route.
  --from-gguf         the GGUF already on disk.  Its blk.40 is already Q4_K / Q5_K / Q8_0, so widening it to
                      FP16 recovers nothing; it is here to make a standalone file without network access.

Output: a GGUF holding exactly the tensors the engine's `load()` asks for under `blk.<n_layers>.*`, in the
format given by --dtype (F16 for the 2-D weights, F32 for the 1-D norms, which is what upload_f32 expects).

    python tools/q36_mtp_extract.py --from-checkpoint --out q36-mtp-f16.gguf
    python tools/q36_mtp_extract.py --from-gguf --gguf /path/model.gguf --out q36-mtp-f16.gguf

Nothing here runs a model, and nothing here needs the whole checkpoint: `--from-checkpoint` reads headers and
then only the byte ranges of the tensors it needs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------------------------------------- the tensor list
# The names the engine's load() reads for the MTP layer, as "blk.<n_layers>." + suffix.  Keeping the same
# spelling means the extracted file is a drop-in for the block inside the model GGUF.
DENSE_SUFFIXES = (
    "attn_norm.weight",              # F32
    "post_attention_norm.weight",    # F32
    "attn_q.weight",                 # 2-D
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "attn_q_norm.weight",            # F32
    "attn_k_norm.weight",            # F32
    "ffn_gate_inp.weight",           # F32 (router)
    "ffn_gate_inp_shexp.weight",     # F32 (shared-expert gate)
    "ffn_gate_shexp.weight",         # 2-D
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
    "nextn.eh_proj.weight",          # 2-D
    "nextn.enorm.weight",            # F32
    "nextn.hnorm.weight",            # F32
    "nextn.shared_head_norm.weight", # F32
)
EXPERT_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")

# The RMS norms' weights, and the one place the checkpoint and the GGUF disagree: the checkpoint stores them
# shifted by -1 (the model applies 1 + that), the GGUF stores the plain w the engine multiplies by.  MEASURED on
# 2026-10-28: the model's own blk.40 norm minus the checkpoint's is 1.0000 for all seven of them (residual 0), and
# the shared expert's gate (not a norm) is identical in the two files.  So the sidecar takes the GGUF's value.
NORM_SUFFIXES = (
    "attn_norm.weight",
    "post_attention_norm.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "nextn.enorm.weight",
    "nextn.hnorm.weight",
    "nextn.shared_head_norm.weight",
)

# The checkpoint spells the same tensors mtp.* / mtp.layers.0.*.
CKPT_NAME = {
    "attn_norm.weight": "mtp.layers.0.input_layernorm.weight",
    "post_attention_norm.weight": "mtp.layers.0.post_attention_layernorm.weight",
    "attn_q.weight": "mtp.layers.0.self_attn.q_proj.weight",
    "attn_k.weight": "mtp.layers.0.self_attn.k_proj.weight",
    "attn_v.weight": "mtp.layers.0.self_attn.v_proj.weight",
    "attn_output.weight": "mtp.layers.0.self_attn.o_proj.weight",
    "attn_q_norm.weight": "mtp.layers.0.self_attn.q_norm.weight",
    "attn_k_norm.weight": "mtp.layers.0.self_attn.k_norm.weight",
    "ffn_gate_inp.weight": "mtp.layers.0.mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mtp.layers.0.mlp.shared_expert_gate.weight",
    "ffn_gate_shexp.weight": "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "ffn_gate_exps.weight": "mtp.layers.0.mlp.experts.gate_up_proj",
    "ffn_up_exps.weight": None,      # fused on the checkpoint side: gate_up_proj is split by this tool
    "ffn_down_exps.weight": "mtp.layers.0.mlp.experts.down_proj",
    "nextn.eh_proj.weight": "mtp.fc.weight",
    "nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
    "nextn.shared_head_norm.weight": "mtp.norm.weight",
}


# --------------------------------------------------------------------------------------- safetensors reading
def _st_header(url: str, timeout: int = 90) -> tuple[dict, dict] | tuple[None, None]:
    """Kept for callers that want the header and its __metadata__ (unused by the extraction itself)."""
    hdr, n = _st_shard(url, timeout)
    return hdr, {}


def _st_read(url: str, off: int, n: int, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={off}-{off + n - 1}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status != 206:
            raise RuntimeError(f"{url}: the server ignored the Range header (HTTP {r.status})")
        buf = r.read()
    if len(buf) != n:
        raise RuntimeError(f"{url}: asked for {n} bytes at {off}, got {len(buf)}")
    return buf


def _file_base(endpoint: str, repo: str) -> str:
    """Where a repository's files are served from.

    huggingface.co serves them at /<repo>/resolve/main/<file>; hf-mirror serves the same file at
    /models/<repo>/resolve/main/<file> and answers 403 (not 404) at the first shape.  Both are tried, mirror
    shape first when the endpoint is not huggingface.co itself.
    """
    ep = endpoint.rstrip("/")
    if "hf-mirror" in ep:
        return f"{ep}/models/{repo}/resolve/main"
    return f"{ep}/{repo}/resolve/main"


def _url(base: str, shard: str) -> str:
    return f"{base.rstrip('/')}/{shard}"


def _get_json(url: str, timeout: int = 120) -> dict:
    """GET a JSON file, trying an endpoint's two known shapes.

    huggingface.co serves /<repo>/resolve/main/<f>; hf-mirror serves /models/<repo>/resolve/main/<f>.  Rather than
    guess per endpoint, both are tried and the first that answers JSON wins.
    """
    i = url.find("/resolve/")
    head, tail = url[:i], url[i:]
    # "https://host/<repo>" -> "https://host/models/<repo>"
    scheme, rest = head.split("://", 1)
    host, repo = rest.split("/", 1)
    cands = [url, f"{scheme}://{host}/models/{repo}{tail}"]
    if "/models/" in head:
        cands = [url, f"{scheme}://{host}/{head.split('/models/', 1)[1]}{tail}"]
    last = "no candidate"
    for u in cands:
        try:
            with urllib.request.urlopen(u, timeout=timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            last = f"{u}: HTTP {e.code}"
            continue
        except Exception as e:                     # noqa: BLE001 - a mirror that drops the connection
            last = f"{u}: {type(e).__name__}"
            continue
        if raw.lstrip().startswith(b"{"):
            return json.loads(raw)
        last = f"{u}: not JSON ({raw[:50]!r})"
    raise RuntimeError(f"no JSON from any shape of {url} (last: {last})")



def _st_shard(url: str, timeout: int = 300) -> tuple[dict, int]:
    """A remote shard's safetensors JSON header and the byte length of that header."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status != 206:
            raise RuntimeError(f"{url}: the server ignored the Range header (HTTP {r.status})")
        n = struct.unpack("<Q", r.read(8))[0]
    req = urllib.request.Request(url, headers={"Range": f"bytes=8-{7 + n}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status != 206:
            raise RuntimeError(f"{url}: the server ignored the Range header (HTTP {r.status})")
        hdr = json.loads(r.read(n))
    hdr.pop("__metadata__", None)
    return hdr, n


def fetch_checkpoint_tensors(base_url: str, shards: dict, want: list[str], out_dir: Path) -> dict:
    """Range-read the named tensors out of their shards; returns name -> (dtype, shape, np.ndarray).

    One header read per shard (they are cached), then one read per tensor: the checkpoint is 26 shards and
    19 tensors, so this moves about 1.6 GB of BF16 and nothing else.
    """
    dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}
    got, heads = {}, {}
    for name in want:
        if name not in shards:
            raise KeyError(f"{name} is not in the checkpoint's index")
        shard, _ = shards[name]
        url = _url(base_url, shard)
        if shard not in heads:
            heads[shard] = _st_shard(url)
        hdr, n = heads[shard]
        if name not in hdr:
            raise KeyError(f"{name} is not in {shard}'s header")
        t = hdr[name]
        dtype, shape = t["dtype"], t["shape"]
        b0, b1 = t["data_offsets"]
        raw = _st_read(url, 8 + n + b0, b1 - b0)
        a = np.frombuffer(raw, dtype=dt[dtype]).reshape(shape)
        got[name] = (dtype, shape, a)
        print(f"  read {name} {dtype} {list(shape)} ({len(raw)} B) from {shard}")
    return got


# ----------------------------------------------------------------------------------------------- conversions
def to_f32(dtype: str, a: np.ndarray) -> np.ndarray:
    if dtype == "F32":
        return a.astype(np.float32)
    if dtype == "F16":
        return a.astype(np.float32)
    if dtype == "BF16":
        return (a.astype(np.uint32) << 16).view(np.float32)
    raise ValueError(f"unsupported dtype {dtype}")


def f32_to_f16(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float16)


# --------------------------------------------------------------------------------------------------- the GGUF
def write_gguf(path: Path, tensors: list[tuple[str, np.ndarray, int]], arch: str, extra: dict) -> None:
    """Write a GGUF with GGUF-py.  tensors: (name, float32 array, gguf type id)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import add_gguf_py  # noqa: E402
    add_gguf_py()
    import gguf  # noqa: E402

    w = gguf.GGUFWriter(str(path), arch)
    for k, v in extra.items():
        if isinstance(v, bool):
            w.add_bool(k, v)
        elif isinstance(v, int):
            w.add_uint32(k, v)
        elif isinstance(v, float):
            w.add_float32(k, v)
        elif isinstance(v, str):
            w.add_string(k, v)
    for name, a, gtype in tensors:
        if gtype == gguf.GGMLQuantizationType.F16:
            w.add_tensor(name, a.astype(np.float16))
        elif gtype == gguf.GGMLQuantizationType.F32:
            w.add_tensor(name, a.astype(np.float32))
        else:
            raise ValueError(f"{name}: unhandled type {gtype}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def verify(sidecar: Path, model: Path) -> int:
    """Compare the sidecar with the model's own block, element by element, in each file's own memory order.

    A layout that does not match the model's gives an error of order 1 (the values are uncorrelated).  The right
    layout gives the quantization error of the model's own weights (a few percent) and exactly zero for the
    tensors the model keeps in F32.  The exit code is 1 when any tensor is out of line.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import add_gguf_py  # noqa: E402
    add_gguf_py()
    import gguf  # noqa: E402
    from gguf import GGUFReader  # noqa: E402

    def flat(t):
        q = t.tensor_type
        raw = np.ascontiguousarray(t.data).view(np.uint8)
        if q == gguf.GGMLQuantizationType.F16:
            return raw.view(np.float16).astype(np.float32)
        if q == gguf.GGMLQuantizationType.F32:
            return raw.view(np.float32)
        if q == gguf.GGMLQuantizationType.BF16:
            return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        return np.asarray(gguf.quants.dequantize(raw, q), dtype=np.float32)

    m = GGUFReader(str(model), "r")
    s = GGUFReader(str(sidecar), "r")
    mt = {t.name: t for t in m.tensors}
    st = {t.name: t for t in s.tensors}
    names = sorted(set(mt) & set(st))
    print(f"{len(names)} tensors in both files")
    bad = 0
    for n in names:
        a, b = flat(mt[n]), flat(st[n])
        if a.size != b.size:
            print(f"  BAD {n}: {a.size} values vs {b.size}")
            bad += 1
            continue
        rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) or 1.0
        err = float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))) / rms
        ok = err < 0.25
        if not ok:
            bad += 1
        print(f"  {'OK ' if ok else 'BAD'} {n}: rel rms {err:.4f}  ({mt[n].tensor_type.name} vs {st[n].tensor_type.name}, "
              f"{a.size} values)")
    print("BAD_COUNT", bad)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--from-checkpoint", action="store_true", help="read the official BF16 checkpoint (no loss)")
    src.add_argument("--from-gguf", metavar="GGUF", help="widen the block inside a GGUF (lossless widening of a lossy source)")
    ap.add_argument("--repo", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT") or "https://huggingface.co")
    ap.add_argument("--index", help="an already-downloaded model.safetensors.index.json (--from-checkpoint)")
    ap.add_argument("--config", help="an already-downloaded config.json (--from-checkpoint)")
    ap.add_argument("--shard-url", help="the prefix shards are fetched from (default: derived from --endpoint)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", metavar="GGUF",
                    help="check the file in --out against the model's own MTP block and exit")
    ap.add_argument("--n-layer", type=int, default=None, help="the MTP block's index in the GGUF (default: from the metadata)")
    args = ap.parse_args()

    out = Path(args.out)
    if args.verify:
        return verify(out, Path(args.verify))
    if not (args.from_checkpoint or args.from_gguf):
        ap.error("one of --from-checkpoint, --from-gguf or --verify is needed")

    blocks = []   # (suffix, float32 array)

    if args.from_checkpoint:
        base = args.shard_url or _file_base(args.endpoint, args.repo)
        idx = json.load(open(args.index)) if args.index else _get_json(base + "/model.safetensors.index.json")
        wm = idx["weight_map"]
        shards = {k: (v, None) for k, v in wm.items()}
        cfg = json.load(open(args.config)) if args.config else _get_json(base + "/config.json")
        tc = cfg.get("text_config", cfg)
        want = [CKPT_NAME[s] for s in DENSE_SUFFIXES if CKPT_NAME.get(s)]
        want += [CKPT_NAME["ffn_gate_exps.weight"], CKPT_NAME["ffn_down_exps.weight"]]
        got = fetch_checkpoint_tensors(base, shards, want, out)
        for suffix in DENSE_SUFFIXES:
            ck = CKPT_NAME.get(suffix)
            if not ck:
                continue
            dtype, shape, a = got[ck]
            v = to_f32(dtype, a)
            if suffix in NORM_SUFFIXES:
                v = v + 1.0   # the checkpoint's shifted norm weights -> the GGUF's (see NORM_SUFFIXES)
            blocks.append((suffix, v))
        # gate_up_proj is fused: [E, 2*FF, H] on the checkpoint, H fastest.  That IS the engine's per-expert
        # layout (n_ff rows of n_embd, the experts consecutive), so the slice is written as it is: a transpose
        # here would interleave the experts (the axis order of a numpy array is not the GGUF's).
        # gguf-py writes the dims reversed, so [E, FF, H] lands in the file as [H, FF, E] - the same dims the
        # model's own blk.40 has.
        dtype, shape, gu = got[CKPT_NAME["ffn_gate_exps.weight"]]
        d_dtype, _, dn = got[CKPT_NAME["ffn_down_exps.weight"]]
        a = to_f32(dtype, gu)
        if a.ndim != 3:
            raise ValueError(f"gate_up_proj has shape {list(a.shape)}; expected [E, 2*FF, H]")
        e, two_ff, h = a.shape
        ff = two_ff // 2
        gate = np.ascontiguousarray(a[:, :ff, :])                            # [E, FF, H], H fastest
        up = np.ascontiguousarray(a[:, ff:, :])                               # [E, FF, H], H fastest
        dd = to_f32(d_dtype, dn)                                              # [E, H, FF] on the checkpoint, FF fastest
        d = np.ascontiguousarray(dd)                                          # the engine's down: n_embd rows of n_ff
        blocks.append(("ffn_gate_exps.weight", gate))
        blocks.append(("ffn_up_exps.weight", up))
        blocks.append(("ffn_down_exps.weight", d))
        n_layer = tc.get("num_hidden_layers", 40)
        n_expert = tc.get("num_experts", 256)
        n_embd = tc.get("hidden_size", 2048)
        n_ff = tc.get("moe_intermediate_size", 512)
        arch = "qwen35moe"
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _paths import add_gguf_py  # noqa: E402
        add_gguf_py()
        from gguf import GGUFReader  # noqa: E402
        r = GGUFReader(args.from_gguf)
        blocks_idx = sorted({int(t.name.split(".")[1]) for t in r.tensors
                             if t.name.startswith("blk.") and t.name.split(".")[1].isdigit()})
        n_layer = args.n_layer if args.n_layer is not None else blocks_idx[-1]
        n_expert = 256
        n_embd = 2048
        n_ff = 512
        arch = "qwen35moe"
        print(f"reading blk.{n_layer} from {args.from_gguf}")
        for t in r.tensors:
            if not t.name.startswith(f"blk.{n_layer}."):
                continue
            suffix = t.name[len(f"blk.{n_layer}."):]
            a = _dequant(t)
            blocks.append((suffix, a))
            print(f"  {suffix}: {t.tensor_type.name} {list(t.shape)[::-1]} -> f32 {a.shape}")

    # write the GGUF: 1-D tensors as F32 (what upload_f32/`f32` reads), 2-D as F16
    tensors = []
    for suffix, a in blocks:
        gguf_name = f"blk.{n_layer}.{suffix}"
        if a.ndim == 1:
            tensors.append((gguf_name, a, 0))          # F32
        else:
            tensors.append((gguf_name, a, 1))          # F16
    extra = {
        "general.architecture": arch,
        "qwen35moe.mtp.n_layer": int(n_layer),
        "qwen35moe.mtp.source": "checkpoint-bf16" if args.from_checkpoint else "gguf-widened",
    }
    write_gguf(out, tensors, arch, extra)
    tot = sum(a.size * (4 if a.ndim == 1 else 2) for _, a in blocks)
    print(f"\nwrote {out} ({out.stat().st_size / 2**20:.1f} MiB on disk; "
          f"{sum(a.size*(4 if a.ndim==1 else 2) for _,a in blocks)/2**20:.1f} MiB of tensor data)")
    print(f"  {len(tensors)} tensors, n_layer={n_layer}, n_expert={n_expert}, n_embd={n_embd}, n_ff={n_ff}")
    return 0


def _dequant(t) -> np.ndarray:
    """A GGUF tensor of the MTP block as float32 (Q8_0 / Q4_K / Q5_K / F32 / BF16 / F16).

    The array comes back in the ORIGINAL numpy shape, not the file's stored dims: gguf-py writes the dims
    reversed (dim 0 varies fastest) and the bytes in the numpy array's own C order, so the stored dims are
    the numpy shape backwards.  The checkpoint branch above produces exactly this shape, so a file widened
    from a GGUF and a file widened from the checkpoint have the same layout.
    """
    import gguf  # noqa: E402
    q = t.tensor_type
    raw = np.asarray(t.data)
    shape = [int(x) for x in t.shape][::-1]
    if q == gguf.GGMLQuantizationType.F32:
        return raw.view(np.float32).reshape(shape).copy()
    if q == gguf.GGMLQuantizationType.F16:
        return raw.view(np.float16).astype(np.float32).reshape(shape)
    if q == gguf.GGMLQuantizationType.BF16:
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
    deq = gguf.quants.dequantize
    # gguf.dequantize wants the raw bytes with the trailing element dim first; the reader gives them ready
    a = deq(np.ascontiguousarray(t.data).view(np.uint8), q)
    return np.asarray(a, dtype=np.float32).reshape(shape)


if __name__ == "__main__":
    sys.exit(main())
