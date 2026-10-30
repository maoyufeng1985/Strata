// mmq_kq_probe - the host-decidable core of the K-quant DSO's J probe cache (mmq_kq_lib.cu,
// STRATA_MMQ_KQ_J_PROBE=1): the nrows_x row buckets, the (type, bucket, cc) key, the candidate set
// (the upstream switch's J plus legal Js of the same or one more K-tile, under a caller-supplied
// legality predicate - in the DSO that is launch_j1's gate: the config is a table row and fits
// smpbo), the precedence (the KQ pin > the probe > the upstream switch), and the process-local
// winner cache.  The timed launches stay in the DSO; everything here is pure host code, which is
// what tests/cuda/mmq_kq_probe_test.cpp exercises without a GPU.
#pragma once

#include "ggml.h"   // ggml_type, GGML_TYPE_COUNT

#include <algorithm>
#include <climits>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::prefill::kqprobe {

// The rows side of the probe key: the first bound >= nrows_x, or OVER_512.  The bounds are the
// bench/prefill row counts the MMQ actually sees (a prompt is ~48 rows/expert; the bench goes wide).
constexpr int OVER_512 = 513;

inline int bucket(int64_t nrows_x) {
    static const int kBounds[8] = {16, 32, 48, 64, 96, 160, 256, 512};
    for (int b : kBounds)
        if (nrows_x <= b) return b;
    return OVER_512;
}

inline std::string bucket_label(int b) {
    return b == OVER_512 ? std::string(">512") : std::to_string(b);
}

struct ProbeKey {
    ggml_type type;
    int bucket;   // bucket(nrows_x)
    int cc;       // the card's compute capability

    bool operator==(const ProbeKey& o) const {
        return type == o.type && bucket == o.bucket && cc == o.cc;
    }
};

struct ProbeKeyHash {
    size_t operator()(const ProbeKey& k) const {
        return (size_t) k.type * 1000003u + (size_t) k.bucket * 991u + (size_t) k.cc;
    }
};

// The process-local winner cache: one entry per (type, rows-bucket, cc).  Not locked on its own -
// the DSO serializes it (g_jprobe_mu); the host test drives it single-threaded.
class Cache {
public:
    int hit(const ProbeKey& k) const {
        const auto it = m.find(k);
        return it == m.end() ? 0 : it->second;
    }

    void store(const ProbeKey& k, int J) { m[k] = J; }

    bool contains(const ProbeKey& k) const { return m.count(k) != 0; }

    size_t size() const { return m.size(); }

private:
    std::unordered_map<ProbeKey, int, ProbeKeyHash> m;
};

// Legality of one J: the DSO passes launch_j1's gate verbatim (the config row exists
// for (type, J, fallback, cc) and its shared memory fits smpbo); the host test passes a fake set.
using Legal = std::function<bool(int)>;

// The upstream switch's ntiles: ceil(ncols_opt / J), as mul_mat_q_switch_J computes it.
inline int ntiles(int ncols_opt, int J) {
    return (ncols_opt + J - 1) / J;
}

// A mirror of mul_mat_q_switch_J's scan over the legal Js: J += 8 from 8 to 128, the first (smallest
// J) that minimizes ntiles; 0 when nothing is legal.
inline int upstream_J(const Legal& legal, int ncols_opt) {
    int j_best = 0;
    int nt_best = INT_MAX;
    for (int J = 8; J <= 128 && nt_best > 1; J += 8) {
        if (!legal(J)) continue;
        const int nt = ntiles(ncols_opt, J);
        if (nt < nt_best) {
            j_best = J;
            nt_best = nt;
        }
    }
    return j_best;
}

// The probe's candidate set: the upstream switch's J, then the legal Js of the same ntiles or one
// more, ordered by fewer tiles then larger J; capped at `cap` total.  Every candidate passes the
// same gate as launch_j1, so a timed candidate cannot blow the shared-memory bound.
inline std::vector<int> candidates(int J_up, const Legal& legal, int ncols_opt, int cap = 3) {
    std::vector<int> out;
    if (J_up < 8 || J_up > 128 || J_up % 8 != 0 || cap < 1 || !legal(J_up)) return out;
    const int nt_up = ntiles(ncols_opt, J_up);
    out.push_back(J_up);
    struct Cand {
        int J;
        int nt;
    };
    std::vector<Cand> rest;
    for (int J = 8; J <= 128 && J != J_up; J += 8) {
        if (!legal(J)) continue;
        const int nt = ntiles(ncols_opt, J);
        if (nt == nt_up || nt == nt_up + 1) rest.push_back({J, nt});
    }
    std::sort(rest.begin(), rest.end(), [](const Cand& a, const Cand& b) {
        return a.nt < b.nt || (a.nt == b.nt && a.J > b.J);
    });
    for (const Cand& c : rest) {
        if ((int) out.size() >= cap) break;
        out.push_back(c.J);
    }
    return out;
}

// One decision for one MMQ call.  The precedence: the pinned J (STRATA_MMQ_KQ_J) wins outright;
// with the probe off nothing else applies and the upstream switch runs untouched; a cache hit
// reuses the probed winner; a cold miss hands back the candidate set to time.
struct Decision {
    enum class Kind { Upstream, Pinned, Cached, Probe };
    Kind kind = Kind::Upstream;
    int J = 0;                       // Pinned / Cached
    std::vector<int> candidates;     // Probe: [upstream J, ...], at most 3
};

inline Decision decide(int pin_J, bool probe_on, const ProbeKey& key, int ncols_opt, const Legal& legal,
                       const Cache& cache) {
    Decision d;
    if (pin_J != 0) {
        d.kind = Decision::Kind::Pinned;
        d.J = pin_J;
        return d;
    }
    if (!probe_on) return d;   // Upstream
    const int cached = cache.hit(key);
    if (cached != 0) {
        d.kind = Decision::Kind::Cached;
        d.J = cached;
        return d;
    }
    d.kind = Decision::Kind::Probe;
    d.candidates = candidates(upstream_J(legal, ncols_opt), legal, ncols_opt, 3);
    return d;
}

}  // namespace strata::prefill::kqprobe
