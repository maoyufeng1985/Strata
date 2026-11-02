"""Phase A of the Whittle-DFlash2 fine-tune (spec: plans/2026-11-01-whittle-dflash2-finetune-spec.md).

Trains fc + 6 drafter layers + norms + the selector's hidden_projection (~345 M params, fp32) with
the two codebooks frozen, teacher-forced on the box's dumped tap files. The drafter math is a direct
port of tools/dflash2/dflash2_ref.py; the loss is the spec's: cross-entropy of the 7 drafted block
positions against the target's committed tokens (the tap dump's token field), plus the selector-walk
cross-entropy (the rank-256 bilinear scores against the next token, mirroring select_kernel in
src/kernels/cuda/dflash2.cu). 500 steps, batch 4, AdamW.

CPU-only (the box has no CUDA torch; the P100 cannot be used with a CPU wheel). The frozen
embedding/LM-head tables are fp16 constants from the dequantized GGUF sidecars
(work/train/embed_fp16.bin, head_fp16.bin); the drafter itself is fp32, so a batch-4 step is
~5-10 s on the box's 16 CPU cores.

Paths on the box:
  init   /home/Strata/df2/qwen36/{model.safetensors,config.json}
  dumps  /home/Strata/.run/df2/train/p_*.bin (+ prompt_*.txt int32 CSVs)
  out    /home/Strata/df2/work/train/{df2_whittle_head.safetensors,loss.json}
"""
import json, struct, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path("/home/Strata/df2")
DUMP = Path("/home/Strata/.run/df2/train")
H, L, NH, NKV, HD = 2048, 6, 32, 8, 128
VOCAB, MASK = 248320, 248077
BLOCK, TOPK, WIN = 8, 16, 2048
EPS = 1e-6
THREADS = 16
STEPS, BS, LR, MAXLEN, NBLK = 500, 4, 5e-5, 576, 3

INV_FREQ = 1.0 / (10000000.0 ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))


def rms(x, wt, eps=EPS):
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * wt


def rotate_half(x):
    h2 = x.shape[-1] // 2
    return torch.cat((-x[..., h2:], x[..., :h2]), dim=-1)


def cos_sin(pos):
    fr = pos.float()[:, None] * INV_FREQ[None, :]
    e = torch.cat([fr, fr], dim=-1)
    return e.cos(), e.sin()


class Drafter(nn.Module):
    def __init__(self, w, pred_cb, succ_cb):
        super().__init__()
        self.fc = nn.Parameter(w["fc.weight"].float().clone())
        self.hidden_norm = nn.Parameter(w["hidden_norm.weight"].float().clone())
        self.final_norm = nn.Parameter(w["norm.weight"].float().clone())
        self.pred_cb = pred_cb
        self.succ_cb = succ_cb
        self.sel_hidden = nn.Parameter(w["candidate_selector.hidden_projection.weight"].float().clone())
        self.ln_in = nn.Parameter(torch.stack([w["layers.%d.input_layernorm.weight" % l].float() for l in range(L)]).clone())
        self.ln_mid = nn.Parameter(torch.stack([w["layers.%d.post_attention_layernorm.weight" % l].float() for l in range(L)]).clone())
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, p, nn.Parameter(torch.stack([w["layers.%d.self_attn.%s.weight" % (l, p)].float() for l in range(L)]).clone()))
        for p in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, "mlp_" + p, nn.Parameter(torch.stack([w["layers.%d.mlp.%s.weight" % (l, p)].float() for l in range(L)]).clone()))
        self.q_norm = nn.Parameter(torch.stack([w["layers.%d.self_attn.q_norm.weight" % l].float() for l in range(L)]).clone())
        self.k_norm = nn.Parameter(torch.stack([w["layers.%d.self_attn.k_norm.weight" % l].float() for l in range(L)]).clone())
        for c in ("attention_conv", "mlp_conv"):
            setattr(self, c + "_kp", nn.Parameter(torch.stack([w["layers.%d.%s.kernel_projection.weight" % (l, c)].float() for l in range(L)]).clone()))
            setattr(self, c + "_bk", nn.Parameter(torch.stack([w["layers.%d.%s.base_kernel" % (l, c)].float() for l in range(L)]).clone()))

    def conv_pre(self, x, l, kp, bk):
        b, m, _ = x.shape
        g = H // 16
        dyn = (x @ kp[l].T).view(b, m, 2, 2, g)
        bl = x.view(b, m, g, 16)
        out = bl * bk[l][0].view(1, g, 16)
        out = out + dyn[:, :, 0, 0].unsqueeze(2) * bl
        return out.view(b, m, H), dyn[:, :, 1]

    def conv_fin(self, x, dyn, l, bk):
        b, m, _ = x.shape
        g = H // 16
        bl = x.view(b, m, g, 16)
        out = bl * bk[l][1].view(1, g, 16)
        out = out + dyn.unsqueeze(2) * bl
        return out.view(b, m, H)

    def forward(self, feats, ctx_tok, embed, head_t):
        """feats [b, nctx, 8*H] fp32 (the 8 tap streams); ctx_tok [b, nctx] the target's committed
        tokens. Teacher-forced: the block context at position s is ctx_tok[:, s+1:s+8] (the true
        sequence, per the spec's teacher-forcing note). NBLK full blocks are walked, anchored every
        BLOCK positions starting at 0; the context window K/V come from the tap features.
        Returns (logits [b, NBLK*7, V], cand, walk_scores [b, NBLK, 7, TOPK], walk_tgt [b, NBLK, 7]).
        """
        b, n = feats.shape[0], feats.shape[1]
        assert n >= NBLK * BLOCK
        h = rms(feats @ self.fc.T, self.hidden_norm)            # [b, n, H]
        Kc, Vc = [], []
        for l in range(L):
            kk = rms((h @ self.k_proj[l].T).view(b, n, NKV, HD), self.k_norm[l])
            vv = (h @ self.v_proj[l].T).view(b, n, NKV, HD)
            Kc.append(kk.permute(0, 2, 1, 3).contiguous())
            Vc.append(vv.permute(0, 2, 1, 3).contiguous())
        outs, cands, wss, wts = [], [], [], []
        for blk in range(NBLK):
            s = blk * BLOCK
            hblk = embed(ctx_tok[:, s + 1:s + BLOCK + 1])       # [b, 8, H] fp16
            cos, sin = cos_sin(torch.arange(s + 1, s + BLOCK + 1))
            for l in range(L):
                r = hblk
                x = rms(hblk, self.ln_in[l])
                x, kd = self.conv_pre(x, l, self.attn_conv_kp, self.attn_conv_bk)
                q = rms((x @ self.q_proj[l].T), self.q_norm[l]).view(b, 8, NH, HD)
                kn = rms((x @ self.k_proj[l].T), self.k_norm[l]).view(b, 8, NKV, HD)
                vn = (x @ self.v_proj[l].T).view(b, 8, NKV, HD)
                q = q * cos[:, None, None, :] + rotate_half(q) * sin[:, None, None, :]
                kn2 = kn * cos[:, None, None, :] + rotate_half(kn) * sin[:, None, None, :]
                lo = max(0, s - WIN + 1)
                K = torch.cat([Kc[l][:, lo:s + 0, :].permute(0, 1, 2), kn2.permute(0, 2, 1)], dim=2)
                V = torch.cat([Vc[l][:, lo:s + 0, :].permute(0, 1, 2), vn.permute(0, 2, 1)], dim=2)
                Kx = K.repeat_interleave(NH // NKV, dim=1)
                Vx = V.repeat_interleave(NH // NKV, dim=1)
                sc = torch.einsum("bhid,hqkd->bhq", q.reshape(b, NH, 8, HD), Kx) * (HD ** -0.5)
                cp = torch.arange(lo, s + 8)
                vis = (torch.arange(8)[:, None] + (s + 1) - cp[None, :]).abs() < WIN
                sc = sc.masked_fill((~vis)[None, None], float("-inf"))
                att = torch.softmax(sc, dim=-1)
                o = torch.einsum("bhq,hqkd->bhkd", att, Vx).reshape(b, 8, NH * HD)
                o = o @ self.o_proj[l].T
                o = self.conv_fin(o, kd, l, self.attn_conv_bk)
                hblk = r + o
                r = hblk
                x = rms(hblk, self.ln_mid[l])
                x, md = self.conv_pre(x, l, self.mlp_conv_kp, self.mlp_conv_bk)
                mm = F.silu(x @ self.mlp_gate_proj[l].T) * (x @ self.mlp_up_proj[l].T)
                mm = mm @ self.mlp_down_proj[l].T
                mm = self.conv_fin(mm, md, l, self.mlp_conv_bk)
                hblk = r + mm
            ln = rms(hblk, self.final_norm)[1:]                  # [b, 7, H] fp32
            lg = ln @ head_t.T                                    # [b, 7, V] fp16
            cand = torch.topk(lg, TOPK, dim=-1).indices
            un = torch.topk(lg, 1, dim=-1).values
            hp = ln @ self.sel_hidden.T                           # [b, 7, 256]
            A = self.pred_cb[ctx_tok[:, s + 1:s + 8]]            # [b, 7, 256]
            ws, wt = [], []
            for j in range(7):
                Bm = self.succ_cb[cand[:, j]].float()            # [b, TOPK, 256]
                ws.append(un[:, j, 0] + torch.einsum("bik,ik->bi", Bm, A[:, j] * hp[:, j]))
                wt.append(ctx_tok[:, s + 2 + j] if j < 6 else ctx_tok[:, s + 7])
            wss.append(torch.stack(ws, 1)); wts.append(torch.stack(wt, 1))
            outs.append(lg); cands.append(cand)
        return (torch.cat(outs, 1), torch.cat(cands, 1),
                torch.cat(wss, 1), torch.cat(wts, 1))


def read_rows(path):
    d = open(path, "rb").read()
    magic, ver, N, nt = struct.unpack("<4i", d[:16])
    assert magic == 0x50415453, hex(magic)
    data = d[16 + 4 * nt:]
    rec = 8 + 4 * nt * N
    rows = len(data) // rec
    pos = np.frombuffer(data, np.int32, rows * 2, 0).reshape(rows, 2)
    taps = np.frombuffer(data, np.float32, rows * nt * N, 8).reshape(rows, nt * N)
    return pos[:, 0], pos[:, 1].copy(), taps.copy()


def load_batch(files, bs, seed):
    rng = np.random.default_rng(seed)
    sel = rng.permutation(len(files))[:bs]
    feats, toks = [], []
    for i in sel:
        f = files[i]
        seq = np.fromfile(str(f.parent / ("prompt_%s.txt" % f.stem[2:])), dtype=np.int64)
        pos, t2, taps = read_rows(f)
        seqf = np.concatenate([seq, t2.astype(np.int64)])[:MAXLEN]
        n = int(pos.max()) + 1
        fe = np.zeros((n, taps.shape[1]), np.float32)
        for k in range(len(pos)):
            if pos[k] < MAXLEN:
                fe[pos[k]] = taps[k]
        fe[0] = fe[1]
        feats.append(fe[:MAXLEN])
        toks.append(seqf)
    F = torch.zeros((bs, MAXLEN, 8 * H))
    T = torch.zeros((bs, MAXLEN), dtype=torch.int64)
    for k in range(bs):
        F[k] = torch.from_numpy(feats[k])
        T[k] = torch.from_numpy(toks[k])
    return F, T


if __name__ == "__main__":
    t0 = time.time()
    torch.set_num_threads(THREADS)
    print("loading fp16 embed/head tables...", flush=True)
    emb = torch.from_numpy(np.fromfile(str(ROOT / "work/train/embed_fp16.bin"), np.float16).reshape(VOCAB, H)).half()
    head = torch.from_numpy(np.fromfile(str(ROOT / "work/train/head_fp16.bin"), np.float16).reshape(VOCAB, H)).half()
    print("loading incoai init safetensors...", flush=True)
    w = {}
    with safe_open(str(ROOT / "qwen36/model.safetensors"), "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            w[k] = t if "codebook" in k else t.float()
    model = Drafter(w, w["candidate_selector.predecessor_codebook"], w["candidate_selector.successor_codebook"])
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    files = sorted(DUMP.glob("p_*.bin"))
    print("trainable %.1f M params; %d dump files; %d steps batch-%d CPU(%d thr), %d s start" %
          (ntrain / 1e6, len(files), STEPS, BS, THREADS, time.time() - t0), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_hist = []
    for step in range(STEPS):
        F, T = load_batch(files, BS, step)
        lg, cd, ws, wt = model(F, T, lambda x: emb[x.long()], head.T)
        M = lg.shape[1]
        tg = torch.stack([T[:, s + 1:s + BLOCK] for s in range(0, M, BLOCK)], 1).reshape(-1)
        ce = F.cross_entropy(lg.reshape(-1, VOCAB), tg)
        idx = (cd.reshape(-1, TOPK) == wt.reshape(-1, 1).expand(-1, TOPK)).int().argmax(-1)
        wk = F.cross_entropy(ws.reshape(-1, TOPK).float(), idx)
        loss = ce + wk
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_hist.append(float(loss))
        el = time.time() - t0
        if step % 25 == 0 or step == 1:
            print("step %3d/%d loss ce=%.4f wk=%.4f  elapsed %.0f s (%.1f s/step, eta %d min)" %
                  (step + 1, STEPS, float(ce), float(wk), el, el / (step + 1),
                   int(el / (step + 1) * (STEPS - step - 1) / 60)), flush=True)
        if step in (99, 199, 399, 499):
            np.save(str(ROOT / "work/train/loss_step%d.npy" % (step + 1)), np.array(loss_hist[step * 25 + 1:step * 25 + 1]))
    out = ROOT / "work/train/df2_whittle_head.safetensors"
    import safetensors.torch
    sd = {k: v.detach().float() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    safetensors.torch.save_file(sd, str(out))
    json.dump({"loss": loss_hist, "steps": STEPS, "batch": BS, "lr": LR, "seconds_total": time.time() - t0},
              open(str(ROOT / "work/train/loss.json"), "w"))
    print("TRAIN_DONE total %.1f min -> %s" % ((time.time() - t0) / 60, out), flush=True)
