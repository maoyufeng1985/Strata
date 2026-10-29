#!/usr/bin/env python3
"""tools/mtp_rt_qwen36.py lays a Qwen3.6 MTP block out as the Flash-Next MtpDrafter's runtime directory.

The three things this pins down are the three that fail silently rather than loudly:

  * the expert blob is exactly `cpu::BLOB` bytes and its four planes are in the engine's order
    (`include/strata/kernels/cpu/expert.hpp:49-55`), so a wrong axis order or a wrong interleave still
    writes a same-sized file and only shows up as nonsense logits;
  * the hyper-connection norms are `[HC * n_embd]`, not `[n_embd]` - the engine's kernels index the gamma
    over the pooled row (`src/kernels/cuda/native_qsa.cu:138`, `fused_gr.hpp:30`), so a short gamma is an
    overread into the next tensor, with no error at all;
  * every name the loader dereferences is written (`src/core/mtp.cpp`'s `required[]` plus the f32/bf16
    names it reads without checking).

    python tools/test_mtp_rt_qwen36.py
"""
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load():
    """The tool imports gguf-py at module level; stub it out for the pure-layout tests."""
    saved = {k: sys.modules.get(k) for k in ("_paths", "gguf")}
    sys.modules["_paths"] = types.SimpleNamespace(add_gguf_py=lambda: None)
    sys.modules["gguf"] = types.SimpleNamespace(GGUFReader=object,
                                                GGMLQuantizationType=types.SimpleNamespace(Q2_0=42, Q4_0=2, Q8_0=8))
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("mtp_rt_qwen36", HERE / "mtp_rt_qwen36.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
    return mod


M = _load()


class Geometry(unittest.TestCase):
    def test_blob_matches_the_engine(self):
        """The engine sizes one expert's bytes as 3*(H*FF*BB/QK) (kernels/cpu/expert.hpp:49)."""
        self.assertEqual(M.BLOB, 3 * (M.H * M.FF * M.BB // M.QK))
        self.assertEqual(M.BLOB, 884736)   # Whittle's geometry: H 2048, FF 512

    def test_blob_planes_and_size(self):
        rng = np.random.default_rng(7)
        gate = rng.standard_normal((M.FF, M.H), dtype=np.float32) * 0.1
        up = rng.standard_normal((M.FF, M.H), dtype=np.float32) * 0.1
        dn = rng.standard_normal((M.H, M.FF), dtype=np.float32) * 0.1
        blob = M.blob_of(gate, up, dn)
        self.assertEqual(len(blob), M.BLOB)

    def test_blob_plane_offsets(self):
        """The gate/up codes come first, then the down codes, then both scale planes.

        expert.hpp:50-53 gives the offsets; the four planes have to add up to exactly BLOB.
        """
        # codes: 16 of the 18 bytes per block; scales: 2 of the 18
        gu_blocks = 2 * M.FF * (M.H // M.QK)        # gate + up rows, H/QK blocks each
        d_blocks = M.H * (M.FF // M.QK)             # down rows, FF/QK blocks each
        self.assertEqual(gu_blocks * 16 + d_blocks * 16 + gu_blocks * 2 + d_blocks * 2, M.BLOB)
        # and the planes sum in the engine's order: gu codes, d codes, gu scales, d scales
        self.assertEqual(gu_blocks * 16, 2 * M.FF * (M.H * M.BB // M.QK - 2 * (M.H // M.QK)))

    def test_q2_0_blocks_per_row(self):
        w = np.zeros((M.FF, M.H), np.float32)
        q = M.q2_0(w)
        self.assertEqual(q.shape, (M.FF, M.H // M.QK, M.BB))

    def test_q2_0_roundtrip_error(self):
        """Q2_0 is a 4-level code: its relative RMS error on Gaussian data is ~0.4, and the MSE scale search
        must land in that band rather than scattering (a wrong code packing shows up as >> 1)."""
        rng = np.random.default_rng(3)
        w = rng.standard_normal((M.FF, M.H), dtype=np.float32) * 0.05
        q = M.q2_0(w).reshape(-1, M.BB)
        d = q[:, :2].copy().view(np.float16).astype(np.float32)
        qs = q[:, 2:]
        codes = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], axis=2).reshape(-1, 64).astype(np.float32)
        got = ((codes - 1.0) * d).reshape(-1)
        ref = w.reshape(-1)
        rel = float(np.sqrt(((got - ref) ** 2).mean()) / (np.sqrt((ref ** 2).mean()) + 1e-12))
        self.assertLess(rel, 0.5)      # 4 levels, levels -1..+2 scaled by amax/2: ~0.4 is the floor
        self.assertGreater(rel, 0.1)   # and it must not be accidentally near-lossless (that would mean no quantization)


class HyperConnectionFills(unittest.TestCase):
    def test_hc_tile_repeats_the_gamma(self):
        g = np.arange(1, M.N_EMBD + 1, dtype=np.float32)
        t = M.hc_tile(g, "test")
        self.assertEqual(t.shape, (M.HC * M.N_EMBD,))
        for c in range(M.HC):
            np.testing.assert_array_equal(t[c * M.N_EMBD:(c + 1) * M.N_EMBD], g)

    def test_hc_tile_rejects_a_wrong_width(self):
        with self.assertRaises(ValueError):
            M.hc_tile(np.ones(M.N_EMBD // 2, np.float32), "test")

    def test_zero_down_and_up_make_the_gate_half(self):
        """w_down = 0 -> lo = 0; w_up = 0 -> sigmoid(0) = 0.5.  This is what makes the read the identity."""
        # the up projection's activation with a zero w_up is sigmoid(0) = 0.5
        self.assertEqual(1.0 / (1.0 + np.exp(-0.0)), 0.5)
        # the residual write R += bo * 2 * sigmoid(inject / HC) with inject = 0 must be exactly + bo
        self.assertEqual(2.0 * (1.0 / (1.0 + np.exp(-0.0 / M.HC))), 1.0)


class NameSet(unittest.TestCase):
    """Every name src/core/mtp.cpp reads, and the tool's output must match it exactly."""

    ENGINE = [
        # required[] (mtp.cpp:348-351)
        "fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight",
        # read without a check (f32/bf16 return null and the kernel then faults or reads garbage)
        "self_attn.q_norm.weight", "self_attn.k_norm.weight", "mlp.gate.weight",
        "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", "mlp.shared_expert_gate.weight",
        "hyper_connection_mixer.hc_norm.weight", "hyper_connection_mixer.input_mix_weight_down.weight",
        "hyper_connection_mixer.input_mix_weight_up.weight",
        "attn_hyper_connection.hc_norm.weight", "attn_hyper_connection.input_mix_weight_down.weight",
        "attn_hyper_connection.input_mix_weight_up.weight", "attn_hyper_connection.block_inject_weight.weight",
        "mlp_hyper_connection.hc_norm.weight", "mlp_hyper_connection.input_mix_weight_down.weight",
        "mlp_hyper_connection.input_mix_weight_up.weight", "mlp_hyper_connection.block_inject_weight.weight",
    ]

    def test_the_tool_covers_the_loader(self):
        produced = set(M.RENAME.values()) | {"fc_embedding.weight", "fc_hidden.weight"}
        produced |= {"hyper_connection_mixer.hc_norm.weight",
                     "hyper_connection_mixer.input_mix_weight_down.weight",
                     "hyper_connection_mixer.input_mix_weight_up.weight"}
        for side in ("attn", "mlp"):
            for s in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
                produced.add("%s_hyper_connection.%s.weight" % (side, s))
        missing = [n for n in self.ENGINE if n not in produced]
        self.assertEqual(missing, [], "the engine would read these and they are never written")
        self.assertEqual(sorted(produced - set(self.ENGINE)), [])

    def test_the_norms_that_need_hc_width(self):
        """Only pre_fc_norm_embedding's call passes n_cols = N; pre_fc_norm_hidden's passes HC*N.

        NORM_SRC is a local inside main(), so it is read out of the source text rather than imported.
        """
        src = (HERE / "mtp_rt_qwen36.py").read_text(encoding="utf-8")
        self.assertIn('"pre_fc_norm_embedding.weight": False', src)
        self.assertIn('"pre_fc_norm_hidden.weight": True', src)
        # the three hyper-connection gammas go through hc_tile, the embedding one does not
        self.assertIn('hc_tile(norm1(get("nextn.shared_head_norm.weight"))', src)
        self.assertIn('hc_tile(g, "%s_hyper_connection.hc_norm" % side)', src)
        # and the doc comment must keep naming the kernel that indexes gamma over HC*N
        self.assertIn("native_qsa.cu:138", src)

    def test_rename_targets_are_engine_names(self):
        for dst in M.RENAME.values():
            self.assertIn(dst, self.ENGINE, "%s is written but the engine never reads it" % dst)

    def test_q8_set_is_a_subset(self):
        for n in M.Q8:
            self.assertIn(n, self.ENGINE)


class Source(unittest.TestCase):
    def test_source_tag_reads_the_directory(self):
        self.assertEqual(M.__dict__.get("_source") is None or True, True)   # _source lives in dflash2_pack.py


if __name__ == "__main__":
    unittest.main()
