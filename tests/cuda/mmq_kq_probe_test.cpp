// mmq_kq_probe_test - the host-decidable core of the K-quant DSO's J probe cache (src/prefill/
// mmq_kq_lib.cu's STRATA_MMQ_KQ_J_PROBE, the plan's Task 3): the nrows_x bucket boundaries, the
// candidate-set construction (the upstream switch's J plus legal Js of the same or one more
// K-tile, the cap and the ordering), the KQ_J override precedence, and the cache-hit semantics
// (a second call for the same (type, bucket, cc) does no probe).  No GPU: the timed launches
// stay in the DSO - only the host-decidable parts (bucket math, candidate construction, the legal
// filter behind the launch_j1 gate, the precedence, the cache) are exercised here, against the
// real seam in src/prefill/mmq_kq_probe.hpp that the DSO itself calls.
#include "prefill/mmq_kq_probe.hpp"

#include "ggml.h"

#include <cstdio>
#include <functional>
#include <set>
#include <string>

namespace {
namespace kqprobe = strata::prefill::kqprobe;

int expect(bool ok, const char* what, const char* detail) {
    std::printf("%s %-40s %s\n", ok ? "ok  " : "FAIL", what, detail);
    return ok ? 0 : 1;
}

std::string join(const std::vector<int>& v) {
    std::string s;
    for (int x : v) s += std::to_string(x) + " ";
    return s;
}

// a fake legal set standing in for launch_j1's gate (a config row + the shared-memory bound)
std::function<bool(int)> legal_of(std::initializer_list<int> js) {
    std::set<int> s(js);
    return [s](int J) { return s.count(J) != 0; };
}

int bucket_checks() {
    int fails = 0;
    // the eight boundaries (nrows_x <= bound keeps the bound's bucket) and the >512 overflow
    const struct {
        int64_t n;
        int want;
    } rows[] = {
        {1, 16},      {16, 16},     {17, 32},     {32, 32},      {33, 48},     {48, 48},
        {49, 64},     {64, 64},     {65, 96},     {96, 96},      {97, 160},    {160, 160},
        {161, 256},   {256, 256},   {257, 512},   {512, 512},    {513, kqprobe::OVER_512},
        {514, kqprobe::OVER_512},   {1000000, kqprobe::OVER_512},
    };
    for (const auto& r : rows) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "bucket(%lld) = %d (want %d)", (long long) r.n, kqprobe::bucket(r.n), r.want);
        fails += expect(kqprobe::bucket(r.n) == r.want, "bucket boundary", buf);
    }
    fails += expect(kqprobe::bucket_label(kqprobe::OVER_512) == ">512", "bucket label overflow", "label(513) = >512");
    fails += expect(kqprobe::bucket_label(160) == "160", "bucket label", "label(160) = 160");
    return fails;
}

int candidate_checks() {
    int fails = 0;
    auto ck = [&](const char* what, std::initializer_list<int> js, int ncols, int cap, std::initializer_list<int> want) {
        const auto legal = legal_of(js);
        const auto got = kqprobe::candidates(kqprobe::upstream_J(legal, ncols), legal, ncols, cap);
        std::vector<int> w(want);
        char buf[160];
        std::snprintf(buf, sizeof buf, "ncols=%d cap=%d: got [%s] want [%s] (upstream J=%d)", ncols, cap,
                      join(got).c_str(), join(w).c_str(), kqprobe::upstream_J(legal, ncols));
        fails += expect(got == w, what, buf);
    };
    // the same-tile and one-more-tile branches: the +1 tile neighbor is in, the +2 one is out
    ck("plus-one tile in, two-more out", {80, 96, 128}, 512, 3, {128});
    ck("same ntiles, one-more", {96, 112}, 512, 3, {112, 96});
    // the cap: three legal Js qualify (64 same-tile, 56 one-more, 48 one-more), capped at cap
    ck("cap 3", {40, 48, 56, 64, 80, 128}, 160, 3, {80, 64, 56});   // same-tile first (64), then one-more (56, 48 capped out)
    ck("cap 2", {40, 48, 56, 64, 80, 128}, 160, 2, {80, 64});   // cap clips before the one-more 56
    // the legal filter: the same +1 neighbor that passed above drops out when its config is not legal
    ck("legal filter", {96, 112, 128}, 512, 3, {128, 112});
    // the early stop of the upstream scan (ntiles 1) and the empty-legal fallthrough
    ck("one tile", {8}, 8, 3, {8});
    ck("no legal J", {}, 512, 3, {});
    return fails;
}

int precedence_checks() {
    int fails = 0;
    kqprobe::Cache cache;
    const kqprobe::ProbeKey key{GGML_TYPE_Q4_K, kqprobe::bucket(160), 610};
    const auto legal = legal_of({96, 112});
    // the pin wins over the probe: no probe, no cache, even with a cache entry
    auto d = kqprobe::decide(40, true, key, 512, legal, cache);
    fails += expect(d.kind == kqprobe::Decision::Kind::Pinned && d.J == 40 && d.candidates.empty(), "pin > probe",
                    "decide(pin=40, probe=1) -> Pinned J=40, no candidates");
    cache.store(key, 96);
    d = kqprobe::decide(40, true, key, 512, legal, cache);
    fails += expect(d.kind == kqprobe::Decision::Kind::Pinned && d.J == 40, "pin > cache",
                    "decide(pin=40, cached=96) -> Pinned J=40");
    // the pin wins with the probe off too
    d = kqprobe::decide(40, false, key, 512, legal, cache);
    fails += expect(d.kind == kqprobe::Decision::Kind::Pinned && d.J == 40, "pin, probe off",
                    "decide(pin=40, probe=0) -> Pinned J=40");
    // with the probe off the cache never applies: the upstream switch runs, entry or not
    d = kqprobe::decide(0, false, key, 512, legal, cache);
    fails += expect(d.kind == kqprobe::Decision::Kind::Upstream && d.candidates.empty(), "probe off -> upstream",
                    "decide(pin=0, probe=0, cached=96) -> Upstream, no probe");
    return fails;
}

int cache_checks() {
    int fails = 0;
    kqprobe::Cache cache;
    const auto legal = legal_of({96, 112});
    const kqprobe::ProbeKey key{GGML_TYPE_Q4_K, kqprobe::bucket(160), 610};
    // a cold miss hands back the candidate set to time
    auto d = kqprobe::decide(0, true, key, 512, legal, cache);
    char buf[96];
    std::snprintf(buf, sizeof buf, "cold miss -> Probe, candidates [%s] (upstream first)", join(d.candidates).c_str());
    fails += expect(d.kind == kqprobe::Decision::Kind::Probe && d.candidates.size() == 2
                        && d.candidates[0] == 112 && d.candidates[1] == 96,
                    "cold miss probes", buf);
    // the probed winner: stored, and the second call for the key takes no probe
    cache.store(key, 96);
    d = kqprobe::decide(0, true, key, 512, legal, cache);
    fails += expect(d.kind == kqprobe::Decision::Kind::Cached && d.J == 96 && d.candidates.empty(),
                    "second call no probe", "cached winner 96 -> Cached, no candidates");
    fails += expect(cache.hit(key) == 96 && cache.size() == 1, "cache state", "hit(key) = 96, size = 1");
    // the key is (type, bucket, cc): any side moving is a miss.  bucket(200) = 256 != 160, so the key
    // stored at bucket 160 does not hit - the probe runs again.  A true same-bucket hit is rows 100
    // and 160, both bucketing to 160 (the key's bucket is the bound, not the row).
    const auto legal2 = legal_of({40, 48, 56, 64, 80, 128});
    auto miss = [&](const kqprobe::ProbeKey& k, int ncols, const char* what) {
        const auto d2 = kqprobe::decide(0, true, k, ncols, legal2, cache);
        fails += expect(d2.kind == kqprobe::Decision::Kind::Probe, what, "a different key still probes");
    };
    miss({GGML_TYPE_Q5_1, kqprobe::bucket(160), 610}, 160, "different type misses");
    miss({GGML_TYPE_Q4_K, kqprobe::bucket(49), 610}, 160, "different bucket misses");
    miss({GGML_TYPE_Q4_K, kqprobe::bucket(160), 600}, 160, "different cc misses");
    // rows 200 bucket to 256, not 160: a different key, so it still probes
    const auto d4 = kqprobe::decide(0, true, {GGML_TYPE_Q4_K, kqprobe::bucket(200), 610}, 160, legal2, cache);
    fails += expect(d4.kind == kqprobe::Decision::Kind::Probe, "rows-200 is bucket-256",
                    "bucket(200)=256 != 160, so the stored key does not hit");
    // rows 100 bucket to 160: the same key as the stored one, so it is a cache hit
    const auto d5 = kqprobe::decide(0, true, {GGML_TYPE_Q4_K, kqprobe::bucket(100), 610}, 160, legal2, cache);
    fails += expect(d5.kind == kqprobe::Decision::Kind::Cached && d5.J == 96, "same bucket hits",
                    "rows 100 bucket to 160 like the stored key -> Cached");
    return fails;
}
}  // namespace

int main() {
    int fails = 0;
    fails += bucket_checks();
    fails += candidate_checks();
    fails += precedence_checks();
    fails += cache_checks();
    std::printf("mmq_kq_probe_test: %d failures\n", fails);
    return fails ? 1 : 0;
}
