"""Check the two-phase WY form (the one tqLLM runs) against the sequential recurrence Q36's kernel runs.

WHY A SECOND CHECKER, when tools/q36_gdn_chunked_check.py already compares a chunked form against the
sequential one.  That one checks *our* form: the solve `(I+M) d = rhs` is done with the same fp64
substitution in both the reference and the comparison, so a mistake in how the solve is arranged would
cancel.  This one is the form the port is copying from tqLLM (`fn_gdn_wy_k` + `fn_gdn_seq_fast`,
src/models/qwen/qwen4exp/), where the inverse is formed as an explicit matrix in a parallel phase and the
sequential phase only applies an affine recurrence.  Written out independently, in the notation the CUDA
kernels use, so a translation slip shows up here and not on the box.

THE SEQUENTIAL FORM (k_gdn_seq_cols in src/qwen35/q36_engine.cu), with g = exp(gate):
    kv    = S^T k
    delta = (v - g * kv) * beta
    S     = g * S + k delta^T
    o     = (S^T q) * rsqrt(S)

THE TWO-PHASE FORM, exactly as fn_gdn_wy_k / fn_gdn_seq_fast order it.  Per chunk of C tokens, with
lg[t] = sum_{u<=t} log g_u (so exp(lg[t]) = D_t = prod_{u<=t} g_u), and ratio(t,j) = exp(lg[t] - lg[j]):
    phase 1 (independent per chunk)
        A[t][j]  = beta_t * ratio(t,j) * (k_t . k_j)       j <  t
        Mo[t][j] = ratio(t,j) * (q_t . k_j)                j <= t
        Tm        = (I + A)^-1                             by forward substitution, one column a lane
        U[t][d]  = sum_{j<=t} Tm[t][j] * beta_j * V[j][d]
        W[t][d]  = sum_{j<=t} Tm[t][j] * beta_j * D_j * K[j][d]
    phase 2 (walks the chunks in order, three barriers a chunk)
        E[t][v]  = U[t][v] - W[t] . S[:,v]
        o[t][v]  = D_t * (q_t . S[:,v]) + sum_{j<=t} Mo[t][j] * E[j][v]
        S'[:,v]  = D_{n-1} * S[:,v] + sum_j (D_{n-1}/D_j) * E[j][v] * k_j

Both are evaluated over random data at several chunk lengths and a ragged tail.  fp64 throughout, so the
only error is the arithmetic's own; the chunked form reorders the sums, so a small difference is expected
and the threshold is 1e-11, not 1e-15.
"""
import numpy as np

rng = np.random.default_rng(11)


def seq(k, q, v, gate, beta, S0):
    T, DK = k.shape
    DV = v.shape[1]
    state, o = S0.copy(), np.zeros((T, DV))
    for t in range(T):
        g = np.exp(gate[t])
        delta = (v[t] - g * (k[t] @ state)) * beta[t]        # (DV,): k_t . S over the KEY dimension
        state = g * state + np.outer(k[t], delta)           # (DK, DV)
        o[t] = (q[t] @ state) / np.sqrt(DK)
    return o, state


def wy(k, q, v, gate, beta, S0, C, DK):
    """DK is the KEY dimension; the state is (DK, DV) and the value dimension is inferred from v.

    The two are deliberately separate here even though Q36 has DK == DV == 128.  tqLLM's kernels take a single
    `dim` for both because Flash-Next has 128 of each, and that is exactly the kind of thing a port copies
    silently: with one symbol, a transposed state or a swapped index still produces the right SHAPE, so the
    mistake survives every shape check.  Keeping them apart makes a swap an error instead.
    """
    T, DV = v.shape
    S = DK
    state, o = S0.copy(), np.zeros((T, DV))
    assert state.shape == (DK, DV), state.shape
    for t0 in range(0, T, C):
        cn = min(C, T - t0)
        K, Q, V = k[t0:t0 + cn], q[t0:t0 + cn], v[t0:t0 + cn]
        b, gl = beta[t0:t0 + cn], gate[t0:t0 + cn]
        # `gate` IS log g in Q36 (the kernel does g = exp(gate)); tqLLM's `gates` holds exp() already, which is
        # why its kernel calls logf on it.  So the scan here is over gate itself, and every ratio below becomes
        # a subtraction - which is what keeps the D_t factor from being lost.
        lg = np.cumsum(gl)
        D = np.exp(lg)
        ratio = np.exp(lg[:, None] - lg[None, :])          # ratio(t,j) = prod_{u=j+1..t} g_u
        kk, qk = K @ K.T, Q @ K.T
        A = np.tril(b[:, None] * ratio * kk, -1)
        Mo = np.tril(ratio * qk)
        Tm = np.linalg.inv(np.eye(cn) + A)
        tb = Tm * b[None, :]
        U = tb @ V                                          # (cn, DV)
        W = (tb * D[None, :]) @ K                           # (cn, DK)
        E = U - W @ state                                   # (cn, DV): the step's own delta, in the tqLLM sign
        o[t0:t0 + cn] = (D[:, None] * (Q @ state) + Mo @ E) / np.sqrt(DK)
        r = np.exp(lg[cn - 1] - lg)                         # (D_{n-1}/D_j)
        state = D[cn - 1] * state + K.T @ (E * r[:, None])  # (DK, DV): sum_j k_j (r_j e_j)^T
    return o, state


def check(DK, DV, T, C, label=""):
    k = rng.standard_normal((T, DK)) * 0.5
    q = rng.standard_normal((T, DK)) * 0.5
    v = rng.standard_normal((T, DV)) * 0.5
    # THE GATE MUST BE NEGATIVE.  A gate with mean zero is a random walk whose g = exp(gate) exceeds 1 about half
    # the time, so the recurrence amplifies instead of decaying and 1024 tokens reach 1e157 - past fp64's range,
    # where exp() overflows and every comparison reads as a total mismatch.  Real GDN gates are a decay (g < 1).
    gate = -np.abs(rng.standard_normal(T)) * 0.3 - 0.02
    beta = 1.0 / (1.0 + np.exp(-rng.standard_normal(T)))
    S0 = rng.standard_normal((DK, DV)) * 0.3
    o1, s1 = seq(k, q, v, gate, beta, S0)
    o2, s2 = wy(k, q, v, gate, beta, S0, C, DK)
    eo = np.abs(o1 - o2).max() / np.abs(o1).max()
    es = np.abs(s1 - s2).max() / np.abs(s1).max()
    ok = max(eo, es) < 1e-11
    print("DK=%-3d DV=%-3d T=%-4d C=%-3d %-10s output %.2e  state %.2e  %s"
          % (DK, DV, T, C, label, eo, es, "OK" if ok else "MISMATCH"))
    return ok


def main():
    allok = True
    # DK != DV on purpose: a transposed state or a swapped k/v index must fail here, not on the box
    for DK, DV, T, C in [(6, 5, 12, 4), (8, 3, 32, 8), (4, 6, 7, 3), (8, 8, 40, 40),
                         (16, 12, 128, 32), (16, 12, 130, 32), (128, 128, 1024, 32)]:
        allok &= check(DK, DV, T, C, "(ragged)" if T % C else "")
    print()
    print("ALL OK" if allok else "FAILURES ABOVE")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
