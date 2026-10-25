"""q36_peer_gate.py --exe E --args "..." --base-env "..." --peer-env "..." [--prompts F] [--n 64] [--width 256]
[--rounds 2] [--out F.json] [--golden G.json] : the peer prompt path's acceptance gate.

The peer prompt path (STRATA_Q36_PREFILL_PEERS=1) computes the prompt's experts on the helper GPUs, in a
different arithmetic from the default path's (q8_1 activations + native_expert_grouped on the helper, MMQ int8
on the primary).  So this gate does NOT ask for byte-identity with the default path: that is the condition
"may this become the default", and it does not hold (measured 2026-10-25 on 192.168.1.98: on a 216-token
prompt 9 of 32 dumped logits rows keep the same argmax, the largest difference is 34.47, and the FIRST row
already differs - plans/2026-10-25-q36-peer-prefill-today.md §3).  How big the gap is depends on the prompt:
the gate's own run the same day gave byte-identical tokens on 4 of its 6 wide prompts, and the other two
agreed for 20 and 29 tokens (plans/2026-10-25-q36-peer-acceptance-gate.md §3).

What the gate asks, and what each answer means:

  C1  the gate did not run empty.  The peer path only takes a prompt chunk WIDER than 128 tokens
      (peer_window_fits: n_ent > hybrid_t * top_k, q36_engine.cu:4453), so a narrow prompt would leave the
      peer arithmetic untouched and every other check would pass for free.  Each peer round's engine log must
      therefore say that the helpers served the prompt chunks.  Not said => FAIL, not pass.
  C2  the peer path is deterministic: two rounds give the same tokens, token for token.
  C3  the default path is untouched: the default arm's tokens equal the reference's (when --golden is given).
  C4  nothing crashed: no illegal memory access, no NaN, no ERR, every round ended normally.
  C5  the speed floor: the peer arm's prompt tok/s is not below the default arm's.
  C6  the gap, REPORTED and not judged: per prompt, the first token where the peer path and the default path
      differ, and how many leading tokens agree.  This is the number the "may it become the default" decision
      needs; the gate does not turn it into a pass or a fail.

Exit code 0 iff C1..C5 hold.

The prompts are the parity prompts joined into prompts of --width tokens, so no new data file is needed and
the ids are the ones the repo already keeps (parity_prompts.ids).  The output JSON holds the peer arm's
tokens, so two builds can be compared on it with q36_cmp_gate.py.

The box's GPU lock wraps the whole call (the engine is a child of this process, so a command prefix would not
cover it):  flock /data/gpu.lock -c 'python3 tools/qwen35/q36_peer_gate.py ...'
"""
from pathlib import Path
import argparse, json, os, re, shlex, sys

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from q36_harness import Eng, load_prompts

MIN_WIDTH = 129        # the peer path needs MORE than hybrid_t(128) tokens: 128 tokens = 1024 entries = the limit
IMA_RE = re.compile(r"illegal memory access", re.I)
NAN_RE = re.compile(r"\bnan\b", re.I)


def env_dict(s):
    d = {}
    for kv in shlex.split(s or ""):
        k, _, v = kv.partition("=")
        d[k] = v
    return d


def wide_prompts(src, width, count):
    """The parity prompts joined into prompts of `width` tokens.

    The width is the point: a prompt of 128 tokens or less never reaches the peer arithmetic, so a gate that
    ran on it would pass without ever testing the path."""
    base = load_prompts(src)
    if not base:
        raise SystemExit("no prompts in " + src)
    out = []
    for i in range(count):
        ids, j = [], i
        while len(ids) < width:
            ids += base[j % len(base)]
            j += 1
        out.append(ids[:width])
    return out


def run_arm(exe, args, env, prompts, n, log):
    """One engine, every prompt one request: the tokens and the prompt's time (the first token's wait)."""
    eng = Eng(exe, shlex.split(args), env=env, log=log)
    toks, ttfts = [], []
    for ids in prompts:
        t, done, ttft, wall = eng.solo(ids, n)
        toks.append(t)
        ttfts.append(ttft)
    eng.close()
    return toks, ttfts


def read_log(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def first_diff(a, b):
    """(the index of the first differing token, the number of leading tokens that agree)."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            return n, n
        n += 1
    if len(a) == len(b):
        return None, n
    return n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--args", default="")
    ap.add_argument("--base-env", default="", help="the flags both arms share (the helper mode's own flags)")
    ap.add_argument("--peer-env", default="STRATA_Q36_PREFILL_PEERS=1 STRATA_Q36_PREFILL_PEERS_HOST=1 "
                                          "STRATA_Q36_PEER_F16=1", help="the peer arm's own flags")
    ap.add_argument("--prompts", default=str(Path(__file__).resolve().parent / "parity_prompts.ids"))
    ap.add_argument("--count", type=int, default=6, help="how many wide prompts")
    ap.add_argument("--width", type=int, default=256, help="the prompts' width in tokens")
    ap.add_argument("--n", type=int, default=64, help="the greedy tokens per prompt")
    ap.add_argument("--rounds", type=int, default=2, help="the peer arm's rounds (2 = the determinism check)")
    ap.add_argument("--out", default=None, help="the JSON to write (the peer arm's tokens)")
    ap.add_argument("--golden", default=None, help="the reference JSON (the default path's tokens) to compare against")
    ap.add_argument("--logdir", default=None, help="where the engine logs go (default: next to --out)")
    a = ap.parse_args()

    if a.width < MIN_WIDTH:
        raise SystemExit("--width %d is too narrow: the peer path needs more than %d tokens (hybrid_t * top_k), "
                         "so the gate would run empty" % (a.width, MIN_WIDTH - 1))

    here = Path(a.out).resolve().parent if a.out else Path(".").resolve()
    logdir = Path(a.logdir).resolve() if a.logdir else here
    logdir.mkdir(parents=True, exist_ok=True)

    prompts = wide_prompts(a.prompts, a.width, a.count)
    base = env_dict(a.base_env)
    peer = dict(base)
    peer.update(env_dict(a.peer_env))
    # THE BASE ARM MUST BE THE DEFAULT PATH.  Eng() starts from this process's own environment, so a peer flag
    # that happens to be exported (a shell that had it set) would make the "default" arm the peer path - and the
    # gate would then compare the peer path with itself, which every check would pass.  The peer flags are
    # cleared here; the peer arm gets them from --peer-env, which is explicit.
    for k in ("STRATA_Q36_PREFILL_PEERS", "STRATA_Q36_PREFILL_PEERS_HOST", "STRATA_Q36_PEER_F16",
              "STRATA_Q36_PEER_KEEP_RESIDENT"):
        os.environ.pop(k, None)
    print("the prompts: %d of %d tokens (the peer path needs more than %d)" % (len(prompts), a.width, MIN_WIDTH - 1),
          flush=True)
    print("the peer arm's env: " + " ".join("%s=%s" % kv for kv in sorted(peer.items())), flush=True)

    fails, notes = [], []
    peer_toks, peer_ttfts, peer_served = [], [], []

    for r in range(a.rounds):
        log = str(logdir / ("peer%d.err" % r))
        print("===== the peer arm, round %d" % r, flush=True)
        try:
            toks, ttfts = run_arm(a.exe, a.args, peer, prompts, a.n, log)
        except SystemExit as e:
            fails.append("C4: the peer arm's round %d died: %s" % (r, e))
            break
        served = "serve prompt chunks" in read_log(log)
        peer_served.append(served)
        peer_toks.append(toks)
        peer_ttfts.append(ttfts)
        err = read_log(log)
        for i, (t, ttft) in enumerate(zip(toks, ttfts)):
            pps = len(prompts[i]) / max(1e-9, ttft or 0)
            print("p%02d %d tok  prompt %.1f tok/s  %s" % (i, len(t), pps, " ".join(map(str, t[:8]))), flush=True)
        print("  the helpers served the prompt chunks: %s" % ("yes" if served else "NO"), flush=True)
        if IMA_RE.search(err):
            fails.append("C4: the peer arm's round %d logged an illegal memory access" % r)
        if NAN_RE.search(err):
            fails.append("C4: the peer arm's round %d logged a NaN" % r)

    if not peer_toks:
        print("===== the verdict")
        print("  FAIL: the peer arm never produced a result")
        for f in fails:
            print("  " + f)
        return 1

    # C1: the gate ran empty?
    for r, served in enumerate(peer_served):
        if not served:
            fails.append("C1: round %d's engine log does not say the helpers served the prompt chunks: the peer "
                         "path did not run, so the rest of the gate would pass for free" % r)

    # C2: the peer path is deterministic.
    if len(peer_toks) > 1:
        for i in range(len(prompts)):
            n, _ = first_diff(peer_toks[0][i], peer_toks[1][i])
            if n is not None:
                fails.append("C2: prompt %d differs between the peer arm's two rounds at token %d" % (i, n))
    else:
        notes.append("C2: only one peer round ran, so the determinism was not checked")

    # the default arm: the same engine without the peer flags.
    dlog = str(logdir / "default.err")
    print("===== the default arm", flush=True)
    try:
        d_toks, d_ttfts = run_arm(a.exe, a.args, base, prompts, a.n, dlog)
    except SystemExit as e:
        fails.append("C4: the default arm died: %s" % e)
        d_toks, d_ttfts = [], []
    derr = read_log(dlog)
    if IMA_RE.search(derr):
        fails.append("C4: the default arm logged an illegal memory access")
    if NAN_RE.search(derr):
        fails.append("C4: the default arm logged a NaN")
    if "serve prompt chunks" in derr:
        fails.append("C1: the default arm's engine log says the helpers served the prompt chunks: the two arms "
                     "are not the two paths the gate compares")
    for i, (t, ttft) in enumerate(zip(d_toks, d_ttfts)):
        pps = len(prompts[i]) / max(1e-9, ttft or 0)
        print("p%02d %d tok  prompt %.1f tok/s  %s" % (i, len(t), pps, " ".join(map(str, t[:8]))), flush=True)

    # C3: the default path against the reference.
    if a.golden and d_toks:
        g = json.load(open(a.golden))
        # a reference written by this gate holds the default arm's tokens under "default_tokens"; one written by
        # q36_solo_gate.py (the default path's own gate) holds them under "tokens".
        ref = g["default_tokens"] if "default_tokens" in g else g["tokens"]
        if len(ref) != len(prompts):
            fails.append("C3: the reference holds %d prompts, the gate ran %d: the reference was made with "
                         "different prompts" % (len(ref), len(prompts)))
        else:
            bad = 0
            for i, (x, y) in enumerate(zip(ref, d_toks)):
                n, _ = first_diff(x, y)
                if n is not None:
                    bad += 1
                    fails.append("C3: the default path's prompt %d differs from the reference at token %s" % (i, n))
            if not bad:
                print("  the default path matches the reference on %d prompts" % len(ref), flush=True)
    elif not a.golden:
        notes.append("C3: no --golden given, so the default path's own reference was not checked")

    # C5: the speed floor.
    if d_ttfts and peer_ttfts:
        p_mean = sum(len(prompts[i]) / max(1e-9, peer_ttfts[0][i] or 0) for i in range(len(prompts))) / len(prompts)
        d_mean = sum(len(prompts[i]) / max(1e-9, d_ttfts[i] or 0) for i in range(len(prompts))) / len(prompts)
        print("  the prompt's speed: peer %.1f tok/s, default %.1f tok/s (%.2fx)" % (p_mean, d_mean,
                                                                                   p_mean / max(1e-9, d_mean)),
              flush=True)
        if p_mean < d_mean:
            fails.append("C5: the peer arm's prompt is slower than the default arm's (%.1f vs %.1f tok/s)"
                         % (p_mean, d_mean))

    # C6: the gap, reported.
    print("===== the gap to the default path (reported, not judged)", flush=True)
    if d_toks:
        for i in range(len(prompts)):
            n, same = first_diff(peer_toks[0][i], d_toks[i])
            if n is None:
                print("p%02d identical (%d tokens)" % (i, len(d_toks[i])), flush=True)
            else:
                a_tok = peer_toks[0][i][n] if n < len(peer_toks[0][i]) else None
                b_tok = d_toks[i][n] if n < len(d_toks[i]) else None
                print("p%02d differs at token %d of %d (peer %s vs default %s; the leading %d tokens agree)"
                      % (i, n, len(d_toks[i]), a_tok, b_tok, same), flush=True)
        print("  byte-identical to the default path: NO - the two paths compute the prompt's experts in "
              "different arithmetic (the helpers use q8_1 + native_expert_grouped, the primary MMQ int8)", flush=True)

    if a.out:
        json.dump({"args": a.args, "peer_env": a.peer_env, "width": a.width, "n": a.n,
                   "prompts": prompts, "tokens": peer_toks[0], "default_tokens": d_toks},
                  open(a.out, "w"))
        print("wrote " + a.out, flush=True)

    print("===== the verdict")
    for f in fails:
        print("  FAIL " + f)
    for n in notes:
        print("  note " + n)
    if not fails:
        print("  PASS: C1..C5 hold (the gap to the default path is reported above)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
