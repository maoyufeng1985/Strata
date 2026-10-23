// src/qwen35/q36_peer_prefill.cu - see src/qwen35/q36_peer_prefill.cuh.
//
// THREE PARTS, and the split is what keeps the hot call free of host work:
//   * open()    binds to one helper's VRAM cache, finds that helper's CUDA device from the blob addresses the cache
//               itself hands out, establishes the route the window crosses by (the peering link when the two GPUs have
//               one, pinned host memory when they do not), and allocates every buffer this executor owns - the
//               compute ones on the helper, the staging ones in host memory - all freed in close().
//   * prepare() stages the window's activations and routing on the helper and turns them into a ROW PLAN there:
//               which entry is rank r of this helper's run, which expert each activation row belongs to.  This is
//               the only place that reads a number back to the host (8 bytes), because MMQ's launches are sized on
//               the host.
//   * launch()  is the primary's MMQ branch, restricted to the entries this helper owns: q8_1 activations, gate+up
//               as one [2 FF] product read IN PLACE from the cache, SwiGLU, down, and the run back to the primary's
//               `eparts` - one peer copy, or one copy into host staging where there is no link.  No host
//               synchronisation.
//
// WHY THE DEVICE, NOT THE HOST, DECIDES THE ROW ORDER.  The row contract (see the header) is a prefix count over
// entry index, so it is a scan over `n_tok * k` entries - 4096 of them for a 512-token chunk.  Doing it on the
// helper keeps the routing on the device where it already is; only the two counts MMQ's host-side launch geometry
// needs come back.
#include "q36_peer_prefill.cuh"

#include "strata/artifact/gguf_reader.hpp"   // block_geometry: an expert step in whole blocks of each part's type
#include "strata/core/device.hpp"   // primary_device() (STRATA_PRIMARY_DEVICE): the device whose memory we copy from
#include "strata/core/remote_experts.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <cuda_fp16.h>

namespace strata::qwen35 {
namespace {

namespace MMQ = strata::prefill::mmq;

constexpr int kThreads = 256;

inline unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }

/// STRATA_Q36_PROMPT_TIMING=1: the opt-in measurement gate, the SAME exact-"1" parse q36_engine.cu's prompt probe
/// makes.  Read once, so every flag-off run does no clock work here.  Only prepare()'s two blocking sections key off it.
static bool prompt_timing() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_Q36_PROMPT_TIMING");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

/// STRATA_Q36_PEER_F16=1: the UP LEG (the helper's rows on their way back to the primary) travels as fp16 instead
/// of fp32.  The rows are packed to half on the helper, copied as half, and unpacked back to float on the primary, so
/// `eparts` stays fp32 and the row contract is untouched - only the bytes on the wire change.  Half is enough for
/// the transport because the values are then summed with the routing weights: the extra error is ~5e-4 relative,
/// while the values themselves are order 1.  OFF by default: with the variable unset not one byte and not one kernel
/// changes, so a default run is byte-identical to a run without this feature.
static bool peer_f16() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_Q36_PEER_F16");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

/// Switches the calling thread to `dev` and back.  The engine's own thread lives on the primary device, so every
/// CUDA call that must run on the helper is inside one of these.
struct On {
    int prev = 0;
    explicit On(int dev) {
        cudaGetDevice(&prev);
        if (prev != dev) cudaSetDevice(dev);
    }
    ~On() {
        int cur = 0;
        cudaGetDevice(&cur);
        if (cur != prev) cudaSetDevice(prev);
    }
};

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("q36 peer prefill: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

// ---- the row plan, on the helper ----
//
// `owned` is one byte per expert id: whether the helper's cache holds it (so an entry's routing id answers the whole
// "does this entry belong to me" question with one indexed load).  `pos` is the expert's position in the plan's
// expert order (-1 when the cache does not hold it), which is the order the products consume the activation rows in.

/// rank[e] = the number of owned entries BELOW e (the row contract's prefix count), for every owned e.  One block:
/// each thread counts a contiguous chunk, the block scans the chunk counts in shared memory, then each thread walks
/// its chunk again handing out ranks.  A chunk per thread and not a grid-stride loop, because the answer depends on
/// the order of e and only contiguous chunks compose into an ascending rank.
__global__ void k_peer_rank(const int32_t* __restrict__ ids, const uint8_t* __restrict__ owned, int n_ent,
                            int32_t* __restrict__ rank) {
    __shared__ int cnt[kThreads];
    __shared__ int off[kThreads];
    const int tid = threadIdx.x;
    const int chunk = (n_ent + kThreads - 1) / kThreads;
    const int lo = tid * chunk;
    const int hi = min(n_ent, lo + chunk);
    int c = 0;
    for (int e = lo; e < hi; ++e)
        if (owned[ids[e]]) ++c;
    cnt[tid] = c;
    __syncthreads();
    if (tid == 0) {
        int p = 0;
        for (int t = 0; t < kThreads; ++t) { off[t] = p; p += cnt[t]; }
    }
    __syncthreads();
    int r = off[tid];
    for (int e = lo; e < hi; ++e)
        if (owned[ids[e]]) rank[e] = r++;
}

__global__ void k_peer_count(const int32_t* __restrict__ ids, const uint8_t* __restrict__ owned,
                             const int32_t* __restrict__ pos, int n_ent, int32_t* __restrict__ cnt) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    if (!owned[ids[e]]) return;
    const int p = pos[ids[e]];
    if (p >= 0) atomicAdd(&cnt[p], 1);
}

/// bounds[pos] / cur[pos] from the counts, plus the widest expert (the products' launch grid) and the owned total.
__global__ void k_peer_scan(int32_t* __restrict__ cnt, int n_pos, int32_t* __restrict__ bounds,
                            int32_t* __restrict__ cur, int32_t* __restrict__ max_rows) {
    if (threadIdx.x != 0) return;
    int p = 0, mx = 0;
    for (int s = 0; s < n_pos; ++s) {
        bounds[s] = p;
        cur[s] = p;
        p += cnt[s];
        mx = max(mx, cnt[s]);
    }
    bounds[n_pos] = p;
    *max_rows = mx;
}

/// Each owned entry to a row: `src` = its token (the quantizer's gather), `dst` = its RANK (the product's output row
/// inside the helper's run).  The expert order decides which activation row it lands on; the rank decides where its
/// result goes - MMQ writes dst row `ids[activation row]`, so the run's order and the plan's order stay independent.
__global__ void k_peer_scatter(const int32_t* __restrict__ ids, const uint8_t* __restrict__ owned,
                               const int32_t* __restrict__ pos, const int32_t* __restrict__ rank, int n_ent, int k,
                               int32_t* __restrict__ cur, int32_t* __restrict__ src, int32_t* __restrict__ dst) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    if (!owned[ids[e]]) return;
    const int p = pos[ids[e]];
    if (p < 0) return;
    const int r = atomicAdd(&cur[p], 1);
    src[r] = e / k;
    dst[r] = rank[e];
}

/// STRATA_Q36_PEER_F16=1: the up leg's packing.  `src` (fp32) becomes `dst` (fp16), element for element: the row
/// count, the row order and every row's meaning are untouched - only the width of an element changes.
///
/// It also MEASURES the largest magnitude it saw, into `max_abs` (one unsigned slot, the float's own bit pattern,
/// so an integer max is a float max for the non-negative values).  That number is what says whether fp16 can carry
/// these rows at all: fp16 tops out at 65504, and a row that went past it would arrive as an infinity and spread.
/// Measuring it costs one atomic per element and turns "the values are probably small" into a fact.
///
/// A value that does not fit is CLAMPED, not silently turned into an infinity: the transport has to deliver
/// something finite, and the clamp is visible in the measured maximum.
__global__ void k_pack_f16(const float* __restrict__ src, __half* __restrict__ dst, int64_t n,
                            unsigned int* __restrict__ max_abs, unsigned int* __restrict__ max_rel) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = src[i];
    const float a = v < 0.0f ? -v : v;
    // The bit pattern of a non-negative float compares as an integer exactly like the float does.
    unsigned int bits;
    std::memcpy(&bits, &a, sizeof bits);
    atomicMax(max_abs, bits);
    const float lim = 65504.0f;   // the largest finite fp16
    const float c = a > lim ? (v < 0.0f ? -lim : lim) : v;
    const __half h = __float2half(c);
    dst[i] = h;
    // HOW MUCH THE PACKING COSTS, measured rather than assumed: the largest RELATIVE error of the round trip.
    // A relative error is non-negative, so its bit pattern compares as an integer the same way the magnitude's
    // does.  This is the number that says whether fp16 can carry these rows - not the tail of the logits.
    const float back = __half2float(h);
    const float babs = back < 0.0f ? -back : back;
    const float rel = a > 0.0f ? (babs - a) / a : 0.0f;
    unsigned int rbits;
    std::memcpy(&rbits, &rel, sizeof rbits);
    atomicMax(max_rel, rbits);
}

}  // namespace

struct PeerPrefill36::Impl {
    int device = -1;
    int primary = -1;   // the ordinal that OWNS the primary's memory: every peer copy names it, never a literal 0
    int64_t n_layers = 0, n_expert = 0, k = 0, n_embd = 0, n_ff = 0, cap = 0;
    std::vector<strata::kernels::NativeExpertLayout> layouts;   // the SAME table RemoteExperts::open() got
    const strata::core::RemoteExperts* helper = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t ev = nullptr;
    std::unique_ptr<MMQ::Context> mctx;   // per executor, created on the helper

    // WHICH ROUTE THE WINDOW CROSSES BY - decided once in open() and fixed for the life of the executor.  true: the
    // helper reads the primary's activations and writes the primary's eparts over the peering link.  false: the two
    // GPUs have no link (measured on 192.168.1.98, see open()), so the same bytes go through the pinned buffers below.
    bool p2p = true;
    // The host route's staging, pinned, allocated in open() ONLY when that route is taken - a box with a working link
    // allocates none of it, and its allocations are therefore what they were.
    float* h_x = nullptr;        // [cap][n_embd] the primary's activations, on their way down
    int32_t* h_ids = nullptr;    // [cap]          the primary's routing
    // WHERE THE UPLOAD READS FROM: this helper's own staging, or the caller's shared copy of the window (the
    // helpers of one layer are given the same activations and the same routing, so the primary can stage them once).
    const float* h_x_use = nullptr;
    const int32_t* h_ids_use = nullptr;
    float* h_out = nullptr;      // [cap][n_embd] this helper's run, on its way back
    // STRATA_Q36_PEER_F16=1: the same run as fp16.  d_out16 is the packed image on the helper, h_out16 the pinned
    // staging the D2H writes.  Both exist only when the flag is on, so a default run allocates exactly what it did.
    __half* d_out16 = nullptr;   // [cap][n_embd] the packed run, on the helper
    __half* h_out16 = nullptr;   // [cap][n_embd] the packed run, on its way back
    unsigned int* d_f16max = nullptr;   // [1] the largest magnitude the packing kernel saw
    unsigned int* h_f16max = nullptr;   // [1] its pinned copy, read back once per layer
    unsigned int* d_f16rel = nullptr;   // [1] the largest RELATIVE error of the fp16 round trip
    unsigned int* h_f16rel = nullptr;   // [1] its pinned copy
    // What a host-route launch() left in h_out, and where it belongs.  It is remembered rather than uploaded at once
    // because wait() is the only method the caller hands the PRIMARY's stream to, and the upload has to be ordered
    // against the primary's own work there.
    float* pending_eparts = nullptr;
    int64_t pending_base = 0, pending_n = 0;
    // STRATA_Q36_PEER_F16=1: whether the rows waiting to go back are the packed ones.  The up leg's shape is decided
    // in launch() (which route the rows took) and read back here.
    bool pending_f16 = false;

    // everything below is allocated on `device`
    float* d_x = nullptr;      // [cap][n_embd] the staged activations
    float* d_gu = nullptr;     // [cap][2 n_ff] the gate+up product
    float* d_h = nullptr;      // [cap][n_ff]   SwiGLU out
    float* d_out = nullptr;    // [cap][n_embd] this helper's run, before the one peer copy
    float* d_ys = nullptr;     // [cap]         NVFP4 w4a4 per-row activation scales (unused otherwise)
    void* d_xq = nullptr;      // q8_1 of the activations
    void* d_hq = nullptr;      // q8_1 of the SwiGLU out
    int32_t* d_ids = nullptr;      // [cap] the staged routing
    int32_t* d_ident = nullptr;    // [cap] identity row map: the products' "no reordering" table
    int32_t* d_src = nullptr;      // [cap] activation row -> token
    int32_t* d_map = nullptr;      // [cap] activation row -> output row of the run
    int32_t* d_rank = nullptr;     // [cap] entry -> its rank among owned entries
    int32_t* d_pos = nullptr;      // [n_expert] expert -> position in the plan's expert order
    int32_t* d_cnt = nullptr;      // [n_expert + 1] rows per position
    int32_t* d_cur = nullptr;      // [n_expert + 1] the scatter's cursors
    int32_t* d_bounds = nullptr;   // [n_expert + 1] the products' expert bounds
    int32_t* d_maxr = nullptr;     // [1]           the widest expert
    uint8_t* d_owned = nullptr;    // [n_expert]    the residency predicate

    // host scratch, reused per layer
    std::vector<uint8_t> owned_host;
    std::vector<int32_t> pos_host;
    std::vector<std::pair<uintptr_t, int32_t>> ptrs;   // (blob address, expert), resident only, sorted

    /// One MMQ::Product's worth of experts: `w + z * delta` is expert z's blob.  A layer's resident experts usually
    /// form ONE such run (the cache hands slots out in admission order and a layer's blobs all have that layer's
    /// size); a run splits where the gap between two blobs differs.  Splitting is not a correctness compromise: each
    /// run is still a contiguous slice of the plan's expert order, so the shared `bounds` array indexes straight
    /// into it.
    struct Group {
        const uint8_t* w = nullptr;
        size_t delta = 0;
        int first = 0;   // its first expert's position in the plan's expert order
        int n = 0;
    };
    std::vector<Group> groups;
    int64_t n_pos = 0;
    /// Whether the last prepared layer's resident experts are equidistant AT THE LAYOUT'S STEP, i.e. one product can
    /// address them all as `w + z * stride`.  It is an assertion, not a switch: a layer that is not is a fatal error
    /// in prepare() naming the layer, because by the time a layer is being prepared the engine has already routed its
    /// entries here - there is no per-layer fallback to take.  open() checks the same property for every layer before
    /// the first one runs, so a cache whose slot layout is not the engine's is a startup failure, not a wrong answer.
    bool uniform = true;

    int64_t prep_layer = -1;
    bool prepared = false;
    int64_t n_owned = 0, max_rows = 0;
    int64_t rows = 0, launches = 0;
    // The two numbers prepare_begin()'s row plan brings back.  They live here, not on the stack, because the read is
    // ENQUEUED in prepare_begin() and COLLECTED in prepare_end() - the two halves are separate calls so that several
    // helpers can have their transfers in flight at the same time.
    // The BUFFER is PINNED, and that is not a detail: a cudaMemcpyAsync into pageable host memory is a SYNCHRONOUS
    // copy (the driver stages it), so with a pageable target the read would block the host right there and the split
    // would buy nothing.  Pinned, the read rides the helper's stream like every other copy and prepare_begin()
    // returns without waiting.
    int32_t* h_plan = nullptr;   // [2] pinned: the owned-row count and the widest expert
    // The window's token count, kept for prepare_end()'s bound check (the row count cannot exceed n_tok * k).
    int64_t prep_n_tok = 0;

    // STRATA_Q36_PROMPT_TIMING=1 (opt-in, measurement only - see q36_engine.cu's probe): the host-side serial cost
    // prepare() adds, accumulated across the calls the engine made this run.  Written only under the gate, read via the
    // accessors, so a flag-off run never touches none of these execute and the values stay their zero init.
    double prep_d2h_ms = 0.0;   // the two blocking D2H copies of the host route (activations + routing)
    double prep_sync_ms = 0.0;  // the row-plan cudaStreamSynchronize that brings rows/maxr back to the host
    int64_t prep_calls = 0;
};

PeerPrefill36::PeerPrefill36() : impl_(new Impl()) {}
PeerPrefill36::~PeerPrefill36() { close(); }

bool PeerPrefill36::open(const strata::core::RemoteExperts& helper, int64_t n_layers, int64_t n_expert, int64_t k,
                         int64_t n_embd, int64_t n_ff, int64_t cap_entries,
                         const strata::kernels::NativeExpertLayout* layouts, std::string& err) {
    close();
    Impl& m = *impl_;
    if (layouts == nullptr) { err = "q36 peer prefill: no expert layout table"; return false; }
    if (n_layers <= 0 || n_expert <= 0 || k <= 0) { err = "q36 peer prefill: bad layer/expert/k geometry"; return false; }
    if (n_embd <= 0 || n_ff <= 0 || cap_entries <= 0) { err = "q36 peer prefill: bad expert width or window size"; return false; }
    if (cap_entries > 0x7fffffffLL) { err = "q36 peer prefill: window too wide for a 32-bit entry index"; return false; }
    if (!MMQ::built()) {
        err = "q36 peer prefill: this build has no MMQ (the helpers' prompt path is the primary's MMQ branch)";
        return false;
    }

    // THE HELPER'S DEVICE, FROM THE TIER.  RemoteExperts::device() is the accessor PeerExperts has had all along
    // (origin/main include/strata/core/peer_experts.hpp: `int device() const { return device_; }`); this class used to
    // have no such accessor, so the device was read back out of the first resident blob's pointer with
    // cudaPointerGetAttributes over an O(n_layers * n_expert) scan of slot_ptr().  Both are gone: the tier answers.
    // device() < 0 means open() never ran or failed, and an executor cannot bind to that.
    const int dev = helper.device();
    if (dev < 0) {
        err = "q36 peer prefill: the helper tier is not open (device() < 0), so there is no device to bind to";
        return false;
    }
    // THE PRIMARY'S DEVICE, ASKED RATHER THAN ASSUMED.  Every peer copy below names the device that owns the memory on
    // the other side, and the engine's own session may run on any visible ordinal (STRATA_PRIMARY_DEVICE; the engine's
    // peer_open asks this same function).  A literal 0 names a card that does not hold the activations or `eparts` on
    // such a box, so the peer-access request below fails for a GPU the model is not even running on and the whole
    // prompt path silently stays local.
    const int prim = strata::core::primary_device();
    if (prim < 0) {
        err = "q36 peer prefill: the primary device is not set (primary_device() < 0), so there is no device to copy from";
        return false;
    }
    if (prim == dev) {   // a helper cannot be the primary: every copy below would name one and the same card
        err = "q36 peer prefill: the helper is on CUDA" + std::to_string(dev) +
              ", which is the primary device; a helper cannot be the device the model runs on";
        return false;
    }
    // ONE consistency check, on one resident blob: the tier's own ordinal must agree with where its cache's memory
    // actually lives.  A GPU whose device() disagreed with its pointers would put every buffer this executor allocates
    // on the wrong card, and the failure would look like a wrong answer rather than a wrong device - so it is checked
    // once here and refused if it does not hold.  A tier holding nothing is also refused: there is nothing to compute
    // and the cache is filled at open time, so an empty one is a programming error, not a wait.
    const uint8_t* probe = nullptr;
    for (int64_t l = 0; l < n_layers && probe == nullptr; ++l)
        for (int64_t e = 0; e < n_expert && probe == nullptr; ++e) probe = helper.slot_ptr(l, (int32_t) e);
    if (probe == nullptr) {
        err = "q36 peer prefill: the helper's VRAM cache holds no expert, so there is nothing for it to compute";
        return false;
    }
    {
        cudaPointerAttributes attr;
        std::memset(&attr, 0, sizeof(attr));
        if (!ck(cudaPointerGetAttributes(&attr, probe), "cudaPointerGetAttributes", err)) return false;
        if (attr.type != cudaMemoryTypeDevice || attr.device != dev) {
            err = "q36 peer prefill: the helper tier reports CUDA" + std::to_string(dev) + " but its resident blobs are " +
                  (attr.type == cudaMemoryTypeDevice ? "on CUDA" + std::to_string(attr.device)
                                                     : std::string("not on a device at all"));
            return false;
        }
    }

    m.device = dev;
    m.primary = prim;
    m.helper = &helper;
    m.n_layers = n_layers;
    m.n_expert = n_expert;
    m.k = k;
    m.n_embd = n_embd;
    m.n_ff = n_ff;
    m.cap = cap_entries;
    m.layouts.assign(layouts, layouts + (size_t) n_layers);
    for (const auto& L : m.layouts) {
        if (L.gu_type < 0 || L.d_type < 0 || L.bytes == 0 || L.down_off >= L.bytes) {
            err = "q36 peer prefill: the expert layout table has an empty or inconsistent layer";
            close();
            return false;
        }
        if (!MMQ::supported(L.gu_type) || !MMQ::supported(L.d_type)) {
            err = "q36 peer prefill: MMQ has no kernel for this layer's expert types";
            close();
            return false;
        }
        if (!MMQ::fits(L.gu_type, 2 * n_ff) || !MMQ::fits(L.d_type, n_embd)) {
            err = "q36 peer prefill: MMQ has no tile that fits this layer's expert matrices on a visible GPU";
            close();
            return false;
        }
    }
    m.owned_host.assign((size_t) n_expert, 0);
    m.pos_host.assign((size_t) n_expert, -1);
    // THE SLOT GEOMETRY, CHECKED ONCE AT STARTUP RATHER THAN PER LAYER.  Every product below addresses a layer's
    // experts as `w + z * step`, and MMQ divides the step by the type's block size to find expert z - so a hard
    // requirement, not a tuning detail: consecutive experts of one layer must be exactly the layout's `stride`
    // apart, and the stride must be a whole number of both parts' blocks.  RemoteExperts::open() declares the slots
    // layer by layer precisely so this holds, and this asks the helper for three experts of a layer that holds at
    // least three and refuses the helper outright if it does not - which reaches the engine as a startup message
    // instead of NaN rows in the first answer.  A layer holding fewer than three resident experts is skipped: one
    // or two cannot show a step, and they are addressed one Product each anyway.
    for (int64_t l = 0; l < n_layers; ++l) {
        if (m.layouts[(size_t) l].stride == 0) {
            err = "q36 peer prefill: layer " + std::to_string((long long) l) + " has no expert stride";
            close();
            return false;
        }
        std::vector<const uint8_t*> at;
        for (int64_t e = 0; e < n_expert && at.size() < 3; ++e)
            if (const uint8_t* p = helper.slot_ptr(l, (int32_t) e)) at.push_back(p);
        if (at.size() >= 2) {
            const size_t step = m.layouts[(size_t) l].stride;
            for (size_t i = 1; i < at.size(); ++i) {
                const size_t gap = (size_t) (at[i] - at[0]);
                if (gap != step * i) {
                    err = "q36 peer prefill: layer " + std::to_string((long long) l) + ": " +
                          std::to_string((long long) i) + " consecutive resident experts are " +
                          std::to_string((long long) gap) + " bytes apart, but the expert step is " +
                          std::to_string((long long) step) +
                          " (the expert cache's slot layout is not the engine's; the MMQ prompt path cannot address it)";
                    close();
                    return false;
                }
            }
            // and the step itself has to divide into blocks, or MMQ's own division truncates
            int gu_el = 0, gu_b = 0, d_el = 0, d_b = 0;
            if (!strata::block_geometry((uint32_t) m.layouts[(size_t) l].gu_type, gu_el, gu_b) ||
                !strata::block_geometry((uint32_t) m.layouts[(size_t) l].d_type, d_el, d_b) ||
                gu_b <= 0 || d_b <= 0 || step % (size_t) gu_b != 0 || step % (size_t) d_b != 0) {
                err = "q36 peer prefill: layer " + std::to_string((long long) l) + ": an expert step of " +
                      std::to_string((long long) step) + " bytes is not a whole number of the expert types' blocks (" +
                      std::to_string(gu_b) + " and " + std::to_string(d_b) + ")";
                close();
                return false;
            }
        }
    }

    On on(m.device);
    // THE ROUTE.  The peering link is tried FIRST, and on a box that has one it is the only route ever taken.  The
    // point of asking for it is that the helper then reads the primary's activations and writes the primary's eparts
    // directly instead of the driver staging every one of those copies through host memory.
    //
    // But on the one box this feature runs on (192.168.1.98: a Tesla P100 as the primary, three P104-100 helpers)
    // there is no link at all: CanAccessPeer is 0 for all twelve ordered pairs, cudaDeviceEnablePeerAccess(0->1)
    // answers "peer access is not supported between these two devices", and cudaMemcpyPeer(0->1) returns success
    // having moved nothing (the driver stages it).  Refusing the helper there - what this did before - left the whole
    // prompt path dead on that box, so the route is now RECORDED instead: with no link the three copies go through
    // pinned host memory (prepare()'s down leg, launch()/wait()'s up leg), which is where the driver would have put
    // them anyway, only without a second staging copy.  Flash-Next's prompt path splits the same way (src/prefill/
    // prefill.cpp: Prefill::set_peer's P.p2p and the `!P.p2p` device->host then host->device copy pairs, with pinned
    // host_x / host_rows) and carries the same escape hatch: STRATA_Q36_PREFILL_PEERS_HOST=0 restores this class's old
    // answer - a helper without a link does not open, and the engine's clean local fallback (peer_open() simply does
    // not take the helper) applies.
    cudaError_t pca = cudaDeviceEnablePeerAccess(m.primary, 0);
    if (pca == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        pca = cudaSuccess;
    }
    // The capability query as a second witness, in the direction that matters: the helper reaching the primary's
    // memory.  A link the enable call accepts but this denies is the shape #2666 found (a copy that reports success
    // and carries nothing), and the host route is the safe answer there as well.
    int can = 0;
    const cudaError_t pcq = cudaDeviceCanAccessPeer(&can, m.primary, m.device);
    if (pcq != cudaSuccess) cudaGetLastError();
    std::string no_p2p;   // why the link is not usable: empty means it is
    if (pca != cudaSuccess) {
        no_p2p = cudaGetErrorString(pca);
    } else if (pcq != cudaSuccess) {
        no_p2p = "CUDA refused to ask whether the two GPUs have a peer path";
    } else if (can == 0) {
        no_p2p = "CUDA reports no peer path from this helper to the primary's memory";
    }
    if (!no_p2p.empty()) cudaGetLastError();   // a recorded route, not this run's error: drop the sticky code
    // OPT-IN, and the exact string "1" - the same parse q36_engine.cu's peer_host_route() makes, mirrored rather
    // than reached into.  The route is off for every other value, because it does not survive a decode step yet:
    // measured on 192.168.1.98, `--bench-pp 160 --bench-tg 0` is clean but any `--bench-tg > 0` after the prompt dies
    // with an illegal memory access, while the same run with the route off is clean (see peer_host_route()'s comment
    // in q36_engine.cu for the full matrix).  A default-on route would turn a working run into a crash.
    const bool host_route_ok = [] {
        const char* v = std::getenv("STRATA_Q36_PREFILL_PEERS_HOST");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    if (!no_p2p.empty() && !host_route_ok) {
        err = std::string("q36 peer prefill: no peer link between CUDA") + std::to_string(m.device) + " and CUDA" +
              std::to_string(m.primary) + ": " + no_p2p +
              " (the prompt path writes the primary's rows over P2P; set STRATA_Q36_PREFILL_PEERS_HOST=1 to stage "
              "them through host memory instead - experimental, it does not survive a decode step yet)";
        close();
        return false;
    }
    m.p2p = no_p2p.empty();
    if (!ck(cudaStreamCreateWithFlags(&m.stream, cudaStreamNonBlocking), "helper stream", err) ||
        !ck(cudaEventCreateWithFlags(&m.ev, cudaEventDisableTiming), "helper event", err)) {
        close();
        return false;
    }
    // MMQ's context holds a per-device pool, so it is created (and destroyed) on the helper, one per executor.
    m.mctx.reset(new MMQ::Context());

    const size_t nf = (size_t) cap_entries;
    const bool ok =
        ck(cudaMalloc((void**) &m.d_x, nf * (size_t) n_embd * sizeof(float)), "activation staging", err) &&
        ck(cudaMalloc((void**) &m.d_gu, nf * (size_t) (2 * n_ff) * sizeof(float)), "gate+up staging", err) &&
        ck(cudaMalloc((void**) &m.d_h, nf * (size_t) n_ff * sizeof(float)), "SwiGLU staging", err) &&
        ck(cudaMalloc((void**) &m.d_out, nf * (size_t) n_embd * sizeof(float)), "result run", err) &&
        ck(cudaMalloc((void**) &m.d_ys, nf * sizeof(float)), "activation scales", err) &&
        ck(cudaMalloc(&m.d_xq, MMQ::q8_bytes(cap_entries, n_embd)), "q8 activations", err) &&
        ck(cudaMalloc(&m.d_hq, MMQ::q8_bytes(cap_entries, n_ff)), "q8 SwiGLU", err) &&
        ck(cudaMalloc((void**) &m.d_ids, nf * sizeof(int32_t)), "routing staging", err) &&
        ck(cudaMalloc((void**) &m.d_ident, nf * sizeof(int32_t)), "identity rows", err) &&
        ck(cudaMalloc((void**) &m.d_src, nf * sizeof(int32_t)), "gather rows", err) &&
        ck(cudaMalloc((void**) &m.d_map, nf * sizeof(int32_t)), "output rows", err) &&
        ck(cudaMalloc((void**) &m.d_rank, nf * sizeof(int32_t)), "entry ranks", err) &&
        ck(cudaMalloc((void**) &m.d_pos, (size_t) (n_expert + 1) * sizeof(int32_t)), "expert order", err) &&
        ck(cudaMalloc((void**) &m.d_cnt, (size_t) (n_expert + 1) * sizeof(int32_t)), "expert counts", err) &&
        ck(cudaMalloc((void**) &m.d_cur, (size_t) (n_expert + 1) * sizeof(int32_t)), "expert cursors", err) &&
        ck(cudaMalloc((void**) &m.d_bounds, (size_t) (n_expert + 1) * sizeof(int32_t)), "expert bounds", err) &&
        ck(cudaMalloc((void**) &m.d_maxr, sizeof(int32_t)), "widest expert", err) &&
        ck(cudaMalloc((void**) &m.d_owned, (size_t) n_expert), "residency predicate", err);
    if (!ok) {
        close();
        return false;
    }
    // STRATA_Q36_PEER_F16=1: the fp16 up leg's buffers.  Allocated ONLY when the flag is on, so a default run's
    // allocation list is exactly the one above.  d_out16 is the packed image on the helper, d_f16max the slot the
    // packing kernel measures the largest magnitude into (one unsigned int).
    if (peer_f16()) {
        const bool fok =
            ck(cudaMalloc((void**) &m.d_out16, nf * (size_t) n_embd * sizeof(__half)), "packed result run", err) &&
            ck(cudaMalloc((void**) &m.d_f16max, sizeof(unsigned int)), "packed magnitude", err) &&
            ck(cudaMalloc((void**) &m.d_f16rel, sizeof(unsigned int)), "packed relative error", err);
        if (!fok) {
            close();
            return false;
        }
    }
    // THE HOST ROUTE'S STAGING, and only on that route: a box with a working link allocates nothing here, so its
    // allocation list is exactly what it was.  Pinned, because every copy that touches these buffers is a
    // cudaMemcpyAsync on a stream - a pageable buffer would make each one a synchronising copy.  Sized from the same
    // cap_entries the helper's GPU buffers are: h_x is [cap][n_embd] floats, which bounds n_tok * n_embd because a
    // window is capped at n_tok * k entries with k >= 1.  Allocated with the primary current: pinned memory is
    // host-wide, but the copies that read and write it name the primary.
    if (!m.p2p) {
        On onp(m.primary);
        const bool hok =
            ck(cudaHostAlloc((void**) &m.h_x, nf * (size_t) n_embd * sizeof(float), cudaHostAllocDefault),
               "pinned activations", err) &&
            ck(cudaHostAlloc((void**) &m.h_ids, nf * sizeof(int32_t), cudaHostAllocDefault), "pinned routing", err) &&
            ck(cudaHostAlloc((void**) &m.h_out, nf * (size_t) n_embd * sizeof(float), cudaHostAllocDefault),
               "pinned result rows", err);
        if (!hok) {   // a null staging buffer would turn every copy below into a silent no-op: refuse the helper
            close();
            return false;
        }
        // STRATA_Q36_PEER_F16=1: the pinned staging for the packed up leg, and the slot the packing kernel's
        // measurement comes back through.  Only when the flag is on, so a default run's host allocations are the
        // three above and nothing else.
        if (peer_f16()) {
            const bool fok =
                ck(cudaHostAlloc((void**) &m.h_out16, nf * (size_t) n_embd * sizeof(__half), cudaHostAllocDefault),
                   "pinned packed result rows", err) &&
                ck(cudaHostAlloc((void**) &m.h_f16max, sizeof(unsigned int), cudaHostAllocDefault),
                   "pinned packed magnitude", err) &&
                ck(cudaHostAlloc((void**) &m.h_f16rel, sizeof(unsigned int), cudaHostAllocDefault),
                   "pinned packed relative error", err);
            if (!fok) {
                close();
                return false;
            }
        }
    }
    // The row plan's two numbers, PINNED, on both routes: they are the target of a device-to-host copy that has to
    // ride the helper's stream (a pageable target would turn that copy into a synchronous one and undo the split of
    // prepare() into begin and end).  Eight bytes, allocated with the primary current like the rest of the staging.
    {
        On onp(m.primary);
        if (!ck(cudaHostAlloc((void**) &m.h_plan, 2 * sizeof(int32_t), cudaHostAllocDefault), "pinned row plan", err)) {
            close();
            return false;
        }
    }
    // The products' identity row map, built once: the gate+up product writes row r for activation row r, so `dst`
    // row == bounds row and the SwiGLU/down chain keeps one index space.  An all-zero buffer (what cudaMalloc gives)
    // would send every row to row 0, so this is filled before anything can read it, on the same stream.
    if (!ck(cudaMemsetAsync(m.d_ident, 0, nf * sizeof(int32_t), m.stream), "clear identity rows", err)) {
        close();
        return false;
    }
    MMQ::iota(m.d_ident, (int64_t) cap_entries, m.stream);
    if (!ck(cudaGetLastError(), "identity rows", err)) return false;
    // One line per executor, and only once it is open: the host route is a real change in how a layer's bytes travel,
    // so it is said out loud - naming the helper, the card on the other side and CUDA's own reason.  A box with a
    // working link prints nothing here, so its stderr is exactly what it was.
    if (!m.p2p)
        std::fprintf(stderr, "q36 peer prefill: CUDA%d's prompt rows cross through host staging (no peer link with "
                             "CUDA%d: %s)\n", m.device, m.primary, no_p2p.c_str());
    return true;
}

void PeerPrefill36::close() {
    Impl& m = *impl_;
    if (m.device < 0 && m.stream == nullptr) {
        m.helper = nullptr;
        return;
    }
    if (m.device >= 0) {
        On on(m.device);
        if (m.stream != nullptr) cudaStreamSynchronize(m.stream);
        m.mctx.reset();   // its pool belongs to the helper: destroy it there
        void* frees[] = {m.d_x,  m.d_gu, m.d_h,       m.d_out, m.d_ys,   m.d_xq,     m.d_hq,
                         m.d_ids, m.d_ident, m.d_src, m.d_map, m.d_rank, m.d_pos, m.d_cnt,   m.d_cur,
                         m.d_bounds, m.d_maxr, m.d_owned, m.d_out16, m.d_f16max, m.d_f16rel};
        for (void* p : frees)
            if (p != nullptr) cudaFree(p);
        if (m.ev != nullptr) cudaEventDestroy(m.ev);
        if (m.stream != nullptr) cudaStreamDestroy(m.stream);
        cudaGetLastError();
    }
    // The host route's staging, freed after the helper's stream is gone (its last copy into it was on that stream) and
    // after the PRIMARY has settled (wait() issued its H2D on a stream this class only ever sees as an argument, so
    // the one handle it can drain is the device's).  Both routes clear the pointers, so a close() of a link box leaves
    // nothing to confuse the next open().
    if (m.primary >= 0 && (m.h_x != nullptr || m.h_ids != nullptr || m.h_out != nullptr || m.h_plan != nullptr ||
                           m.h_out16 != nullptr || m.h_f16max != nullptr)) {
        On onp(m.primary);
        cudaDeviceSynchronize();
        if (m.h_x != nullptr) cudaFreeHost(m.h_x);
        if (m.h_ids != nullptr) cudaFreeHost(m.h_ids);
        if (m.h_out != nullptr) cudaFreeHost(m.h_out);
        if (m.h_plan != nullptr) cudaFreeHost(m.h_plan);
        if (m.h_out16 != nullptr) cudaFreeHost(m.h_out16);
        if (m.h_f16max != nullptr) cudaFreeHost(m.h_f16max);
        if (m.h_f16rel != nullptr) cudaFreeHost(m.h_f16rel);
        cudaGetLastError();
    }
    m.device = -1;
    m.primary = -1;
    m.p2p = true;
    m.h_x = nullptr;
    m.h_ids = nullptr;
    m.h_out = nullptr;
    m.h_plan = nullptr;
    m.h_out16 = nullptr;
    m.h_f16max = nullptr;
    m.h_f16rel = nullptr;
    m.pending_eparts = nullptr;
    m.pending_base = 0;
    m.pending_n = 0;
    m.stream = nullptr;
    m.ev = nullptr;
    m.d_x = m.d_gu = m.d_h = m.d_out = m.d_ys = nullptr;
    m.d_xq = m.d_hq = nullptr;
    m.d_ids = m.d_ident = m.d_src = m.d_map = m.d_rank = nullptr;
    m.d_pos = m.d_cnt = m.d_cur = m.d_bounds = m.d_maxr = nullptr;
    m.d_owned = nullptr;
    m.helper = nullptr;
    m.groups.clear();
    m.ptrs.clear();
    m.n_pos = 0;
    m.prepared = false;
    m.prep_layer = -1;
    m.n_owned = m.max_rows = 0;
    m.rows = m.launches = 0;
}

bool PeerPrefill36::valid() const { return impl_ && impl_->device >= 0 && impl_->stream != nullptr; }

bool PeerPrefill36::owns(int64_t layer, int32_t expert) const {
    const Impl& m = *impl_;
    if (m.device < 0 || m.helper == nullptr) return false;
    if (layer < 0 || layer >= m.n_layers) return false;
    if (expert < 0 || expert >= m.n_expert) return false;
    return m.helper->slot_ptr(layer, expert) != nullptr;
}

int64_t PeerPrefill36::cap_entries() const { return impl_->cap; }

bool PeerPrefill36::host_route() const { return !impl_->p2p; }

bool PeerPrefill36::prepare_begin(int64_t layer, const float* h_primary, const int32_t* ids_primary, int64_t n_tok,
                                   std::string& err, const float* h_staged, const int32_t* ids_staged) {
    Impl& m = *impl_;
    if (m.device < 0 || m.stream == nullptr) { err = "q36 peer prefill: prepare() on a closed executor"; return false; }
    if (h_primary == nullptr || ids_primary == nullptr) { err = "q36 peer prefill: null activations or routing"; return false; }
    if (layer < 0 || layer >= m.n_layers) { err = "q36 peer prefill: layer out of range"; return false; }
    if (n_tok <= 0) { err = "q36 peer prefill: empty window"; return false; }
    if (n_tok * m.k > m.cap) {
        err = "q36 peer prefill: a window of " + std::to_string(n_tok * m.k) + " entries exceeds the " +
              std::to_string(m.cap) + " open() allocated";
        return false;
    }
    m.prepared = false;
    const int n_ent = (int) (n_tok * m.k);
    const int64_t ne = m.n_expert;
    m.prep_n_tok = n_tok;   // prepare_end()'s bound check needs the window's token count
    // The row plan's two numbers start at zero: a layer this helper holds nothing of returns early below WITHOUT
    // reading them back, and prepare_end() must still see the zero rather than the previous layer's values.
    m.h_plan[0] = 0;
    m.h_plan[1] = 0;

    // ---- the layer's residency, and the expert order the products will consume, from the cache itself (host side,
    // no CUDA call): which experts this helper holds, where each blob lies, and the arithmetic runs that the MMQ
    // product's `w + z * expert_bytes` addressing can cover. ----
    m.ptrs.clear();
    std::fill(m.owned_host.begin(), m.owned_host.end(), (uint8_t) 0);
    for (int32_t e = 0; e < ne; ++e) {
        const uint8_t* p = m.helper->slot_ptr(layer, e);
        if (p == nullptr) continue;
        m.owned_host[(size_t) e] = 1;
        m.ptrs.emplace_back((uintptr_t) p, e);
    }
    std::sort(m.ptrs.begin(), m.ptrs.end());
    std::fill(m.pos_host.begin(), m.pos_host.end(), -1);
    m.groups.clear();
    // THE GROUPS, and the rule that changed: a layer's resident experts are ONE run whenever their blobs are
    // EQUIDISTANT, even when the first two happen to be the closest pair.  The old rule compared every gap with
    // the PREVIOUS group's delta, and the first group was created with delta = 0 - so the first expert was always
    // alone in a group of its own, and the second started a new group even when every gap was the same.  That
    // split is what broke the products: MMQ's `expert_bounds` (the `bounds` a Product carries) is indexed by the
    // product's OWN expert numbers, and the products read `xq` through `expert_bounds[z] .. expert_bounds[z + 1]`
    // (third_party/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:1014).  A group that starts at the plan's expert `first`
    // must therefore carry the bounds of THAT slice, and its `xq` rows must be the slice's rows - which is only
    // true when the slice is the whole plan.  One group over the whole layer is that case, and it is also the
    // case the plan's own order wants (the plan's expert order IS the address order).
    //
    // When the blobs are NOT equidistant the layer cannot be served by one product, and the plan must not claim
    // it: `uniform` says so, and the engine falls back to the local route for this layer (see
    // PeerPrefill36::layer_uniform).
    bool uniform = true;
    size_t delta = 0;
    for (size_t i = 0; i < m.ptrs.size(); ++i) {
        const uintptr_t p = m.ptrs[i].first;
        m.pos_host[(size_t) m.ptrs[i].second] = (int32_t) i;
        if (i == 0) continue;
        const size_t d = p - m.ptrs[i - 1].first;
        if (i == 1) delta = d;
        else if (d != delta) { uniform = false; break; }
    }
    // THE GAP MUST BE THE ENGINE'S STEP, not merely THE SAME GAP EVERYWHERE.  `delta` above is measured off the
    // blobs' addresses, so a cache that laid a layer's experts out at some other (perfectly regular) spacing yields a
    // uniform layer with the WRONG step: MMQ is then told to step `delta` while the layout's own step is `stride`,
    // and every expert after the first is read from bytes that hold no expert of this layer - NaN rows, not an
    // error.  That is exactly what a row-split helper used to do (two experts of one layer 40 slots apart, gap
    // 76722000 where the step is 2041200).  Only `stride` may be handed to MMQ: it is a whole number of both parts'
    // blocks, so MMQ's `expert_bytes / type_bytes` division is exact, while any other gap either truncates or points
    // at bytes that are not this layer's.  RemoteExperts::open() now declares the slots layer by layer so a layer's
    // experts really are one stride apart; this is the check that the premise still holds, and it refuses the layer
    // LOUDLY rather than computing rows from bytes nothing wrote.
    const size_t stride = m.layouts[(size_t) layer].stride;
    if (stride == 0)
        { err = "q36 peer prefill: layer " + std::to_string((long long) layer) + " has no expert stride"; return false; }
    if (uniform && !m.ptrs.empty() && m.ptrs.size() > 1 && delta != stride) {
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "q36 peer prefill: layer %lld's resident experts are %llu bytes apart but the expert step is "
                      "%llu (the cache did not lay this layer's slots out with the engine's step)",
                      (long long) layer, (unsigned long long) delta, (unsigned long long) stride);
        err = buf;
        return false;
    }
    m.uniform = uniform;
    if (uniform && !m.ptrs.empty())
        m.groups.push_back(Impl::Group{(const uint8_t*) m.ptrs[0].first, delta, 0, (int) m.ptrs.size()});
    else
        // Not equidistant: one single-expert group each.  `delta` is unused for n == 1, so this needs no step and is
        // always correct - it is the slow shape (one Product per expert), not a wrong one.
        for (size_t i = 0; i < m.ptrs.size(); ++i)
            m.groups.push_back(Impl::Group{(const uint8_t*) m.ptrs[i].first, 0, (int) i, 1});
    m.n_pos = (int64_t) m.ptrs.size();
    m.prep_layer = layer;
    m.n_owned = 0;
    m.max_rows = 0;
    m.rows = 0;

    // ---- THE HOST ROUTE'S DOWN LEG, and it runs BEFORE the helper's device scope below.  WHICH COPIES RUN HERE IS
    // THE ROUTE'S BUSINESS, so the whole window copy is in one place.
    //
    // Link: nothing happens here at all - the two cudaMemcpyPeerAsync below go straight off the primary's memory onto
    // the helper's stream, and nothing waits on the host.
    //
    // No link: two BLOCKING device-to-host copies into the pinned staging.  A blocking copy is correct here and is
    // NOT a new synchronisation point: the caller (q36's MMQ branch) has already synchronized the primary's stream
    // immediately before this call - it needs the routing on the host to build its own row plan - so the activations
    // are quiescent and there is nothing left for the copy to wait for.  It costs n_tok * (n_embd * 4 + k * 4) bytes
    // once per layer, and it buys the whole feature on a box whose two GPUs have no path between them.
    if (!m.p2p) {
        // STRATA_Q36_PROMPT_TIMING=1: the two blocking device-to-host copies of the host route.  On a box with a
        // working peer link this whole `if (!m.p2p)` block is skipped, so the probe does nothing there either;
        // on a host-route box these two copies are the serial cost the prompt timing probe suspects, and they are
        // counted separately from the row-plan sync below so each cost is visible.
        const auto t_d2h = prompt_timing() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (h_staged != nullptr && ids_staged != nullptr) {
            // THE CALLER ALREADY STAGED THE WINDOW.  Every helper in a layer is given the SAME activations and the
            // SAME routing, so the primary copies them out of its memory once and the helpers read that one copy.
            // Copying them per helper meant the primary moved the same 4 MiB out three times over.
            m.h_x_use = h_staged;
            m.h_ids_use = ids_staged;
        } else {
            m.h_x_use = m.h_x;
            m.h_ids_use = m.h_ids;
            On onp(m.primary);   // the two source pointers are the primary's: the copies run on its device
            if (!ck(cudaMemcpy(m.h_x, h_primary, (size_t) n_tok * (size_t) m.n_embd * sizeof(float),
                               cudaMemcpyDeviceToHost), "stage activations through host", err) ||
                !ck(cudaMemcpy(m.h_ids, ids_primary, (size_t) n_ent * sizeof(int32_t), cudaMemcpyDeviceToHost),
                    "stage routing through host", err))
                return false;
        }
        if (prompt_timing())
            m.prep_d2h_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_d2h).count();
    } else {
        m.h_x_use = nullptr;
        m.h_ids_use = nullptr;
    }

    On on(m.device);
    // ---- the window onto the helper: the primary's activations and routing.  On the link route the two peer copies
    // are the whole story; on the host route the pinned staging above was just filled, and only the upload is left.
    if (m.p2p) {
        if (!ck(cudaMemcpyPeerAsync(m.d_x, m.device, h_primary, m.primary,
                                    (size_t) n_tok * (size_t) m.n_embd * sizeof(float), m.stream), "stage activations",
                    err) ||
            !ck(cudaMemcpyPeerAsync(m.d_ids, m.device, ids_primary, m.primary, (size_t) n_ent * sizeof(int32_t),
                                    m.stream), "stage routing", err))
            return false;
    } else if (m.n_pos != 0) {
        // Skipped for a layer this helper holds nothing of: n_owned can only be 0 there, so launch() returns before any
        // kernel reads d_x - and an upload nobody waits for would still be queued on the helper's stream when the NEXT
        // layer's blocking D2H refills h_x, which arrives before that stream is ever synchronised.  Leaving it out is
        // cheaper than ordering around it.
        if (!ck(cudaMemcpyAsync(m.d_x, m.h_x_use, (size_t) n_tok * (size_t) m.n_embd * sizeof(float),
                                cudaMemcpyHostToDevice, m.stream), "upload activations", err) ||
            !ck(cudaMemcpyAsync(m.d_ids, m.h_ids_use, (size_t) n_ent * sizeof(int32_t), cudaMemcpyHostToDevice, m.stream),
                "upload routing", err))
            return false;
    }
    if (m.n_pos == 0) {   // nothing resident for this layer: no entry can be owned, and no kernel has work to do
        // The row plan has nothing to plan and the stream has nothing queued, so prepare_end() will find the row
        // count at its zero and mark the helper prepared.  The D2H above (if !p2p) was still a timed call, but the
        // call itself is counted by prepare_end(), which every prepared layer goes through.
        return true;
    }
    if (!ck(cudaMemcpyAsync(m.d_pos, m.pos_host.data(), (size_t) ne * sizeof(int32_t), cudaMemcpyHostToDevice, m.stream),
            "upload expert order", err) ||
        !ck(cudaMemcpyAsync(m.d_owned, m.owned_host.data(), (size_t) ne, cudaMemcpyHostToDevice, m.stream),
            "upload residency predicate", err) ||
        !ck(cudaMemsetAsync(m.d_cnt, 0, (size_t) (ne + 1) * sizeof(int32_t), m.stream), "clear expert counts", err))
        return false;
    k_peer_rank<<<1, kThreads, 0, m.stream>>>(m.d_ids, m.d_owned, n_ent, m.d_rank);
    k_peer_count<<<grid1(n_ent), kThreads, 0, m.stream>>>(m.d_ids, m.d_owned, m.d_pos, n_ent, m.d_cnt);
    k_peer_scan<<<1, 1, 0, m.stream>>>(m.d_cnt, (int) m.n_pos, m.d_bounds, m.d_cur, m.d_maxr);
    k_peer_scatter<<<grid1(n_ent), kThreads, 0, m.stream>>>(m.d_ids, m.d_owned, m.d_pos, m.d_rank, n_ent, (int) m.k,
                                                            m.d_cur, m.d_src, m.d_map);
    if (!ck(cudaGetLastError(), "row plan kernels", err)) return false;

    // THE TWO NUMBERS THE HOST NEEDS.  MMQ's launches are sized on the host (rows, the launch grid), so the row count
    // and the widest expert come back - 8 bytes, after work that is one window copy (peer, or staged through host)
    // and four small kernels.  Doing it here rather than in launch() is what keeps the hot call free of host
    // synchronisation.
    //
    // The READ is ENQUEUED here and COLLECTED in prepare_end().  Nothing waits in between: that is the point of the
    // split.  Several helpers can have their window copy and their row plan in flight at the same time - each on its
    // own stream and its own PCIe link - so the host's wait for one helper's rows no longer has to wait for the
    // others' transfers to be issued first.
    m.h_plan[0] = 0;
    m.h_plan[1] = 0;
    if (!ck(cudaMemcpyAsync(&m.h_plan[0], m.d_bounds + m.n_pos, sizeof(int32_t), cudaMemcpyDeviceToHost, m.stream),
            "read owned rows", err) ||
        !ck(cudaMemcpyAsync(&m.h_plan[1], m.d_maxr, sizeof(int32_t), cudaMemcpyDeviceToHost, m.stream),
            "read widest expert", err))
        return false;
    // STRATA_Q36_PROMPT_TIMING=1: the row-plan sync moved to prepare_end(); the two D2H reads above are the ones it
    // waits for.  The d2h timer above still counts the staging copies.
    return true;
}

bool PeerPrefill36::prepare_end(std::string& err) {
    Impl& m = *impl_;
    // THE ONE HOST-BLOCKING POINT ON THE HOST ROUTE: wait for this helper's stream, which now carries the window
    // copy, the row plan and the two reads.  The host needed the row count and the widest expert to size the
    // launches, and that is all this wait is for.
    // STRATA_Q36_PROMPT_TIMING=1: the row-plan sync.  It overlaps the primary's MMQ launch on the linked route, but
    // it is the one host-blocking point on the whole host route, so the probe counts it and the remainder the forward
    // total leaves over it is the primary's time.
    const auto t_sync = prompt_timing() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (!ck(cudaStreamSynchronize(m.stream), "row plan", err))
        return false;
    if (prompt_timing())
        m.prep_sync_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sync).count();
    const int32_t h_owned = m.h_plan[0], h_max = m.h_plan[1];
    if (h_owned < 0 || h_owned > (int) (m.prep_n_tok * m.k) || h_max < 0) {
        err = "q36 peer prefill: the row plan produced an impossible entry count";
        return false;
    }
    m.n_owned = h_owned;
    m.max_rows = h_max;
    m.prepared = true;
    // STRATA_Q36_PROMPT_TIMING=1: count a fully prepared layer.  The d2h/sync timers above are per-call accumulators,
    // so a per-call mean is the sums over this many calls.  Off the gate nothing here runs (the two timers never move).
    if (prompt_timing()) ++m.prep_calls;
    return true;
}

bool PeerPrefill36::prepare(int64_t layer, const float* h_primary, const int32_t* ids_primary, int64_t n_tok,
                            std::string& err) {
    if (!prepare_begin(layer, h_primary, ids_primary, n_tok, err)) return false;
    return prepare_end(err);
}

bool PeerPrefill36::launch(int64_t layer, float* eparts_primary, int64_t eparts_row_base, std::string& err) {
    Impl& m = *impl_;
    if (m.device < 0 || m.stream == nullptr) { err = "q36 peer prefill: launch() on a closed executor"; return false; }
    if (!m.prepared || layer != m.prep_layer) {
        err = "q36 peer prefill: launch() without a prepare() of this layer";
        return false;
    }
    if (eparts_primary == nullptr && m.n_owned > 0) { err = "q36 peer prefill: null eparts"; return false; }
    ++m.launches;
    m.rows = m.n_owned;
    if (m.n_owned == 0) return true;   // no entry of this window is this helper's: nothing to compute, nothing to send

    On on(m.device);
    const auto& L = m.layouts[(size_t) layer];
    const int64_t n = m.n_owned;

    // 1. the activations to q8_1, one row per owned entry, gathered in the plan's expert order (row r reads its
    //    entry's token).  One quantization per entry: two entries of the same token are quantized twice, which is
    //    what the primary's MMQ branch does too.  `yscale` is only read by NVFP4 in w4a4 mode (MMQ::fp4_activations),
    //    where the product needs the same buffer as Product::y_scale; everywhere else MMQ ignores it and null is what
    //    the primary passes.
    float* const ys_gu = MMQ::fp4_activations(L.gu_type) ? m.d_ys : nullptr;
    float* const ys_d = MMQ::fp4_activations(L.d_type) ? m.d_ys : nullptr;
    MMQ::quantize(m.d_x, m.d_src, m.d_xq, L.gu_type, m.n_embd, m.n_embd, n, m.stream, ys_gu);

    // 2. gate+up as one [2 n_ff] product per arithmetic run of experts, the weights read in place from the cache.
    //    Output rows stay in the PLAN's expert order (the identity table), because that is the index space `bounds`
    //    speaks: SwiGLU and the down product's activation rows are read through those bounds.  The run's own order
    //    is imposed one stage later, by the down product's `ids`.
    for (const Impl::Group& g : m.groups) {
        MMQ::Product gu;
        gu.w = g.w;
        gu.type = L.gu_type;
        gu.w_rows = 2 * m.n_ff;
        gu.w_cols = m.n_embd;
        gu.expert_bytes = g.delta;
        gu.n = g.n;
        gu.xq = m.d_xq;
        gu.bounds = m.d_bounds + g.first;   // the run is a contiguous slice of the plan's expert order
        gu.ids = m.d_ident;  // identity: gu row r IS activation row r (null would make MMQ ignore `bounds`)
        gu.total_rows = n;
        gu.max_rows = m.max_rows;
        gu.dst = m.d_gu;
        gu.ld_dst = 2 * m.n_ff;
        gu.y_scale = ys_gu;
        m.mctx->run(gu, m.stream);
    }
    MMQ::swiglu(m.d_gu, m.d_h, n, m.n_ff, false, m.stream);
    MMQ::quantize(m.d_h, nullptr, m.d_hq, L.d_type, m.n_ff, m.n_ff, n, m.stream, ys_d);

    // 3. down, from the same run's blobs at the layout's own down offset.  Here `ids` is NOT the identity: its row
    //    is the entry's RANK in this helper's run, so the product writes the final rows in the order the header's
    //    row contract demands, whatever order the experts were multiplied in.
    for (const Impl::Group& g : m.groups) {
        MMQ::Product dn;
        dn.w = g.w + L.down_off;
        dn.type = L.d_type;
        dn.w_rows = m.n_embd;
        dn.w_cols = m.n_ff;
        dn.expert_bytes = g.delta;
        dn.n = g.n;
        dn.xq = m.d_hq;
        dn.bounds = m.d_bounds + g.first;
        dn.ids = m.d_map;
        dn.total_rows = n;
        dn.max_rows = m.max_rows;
        dn.dst = m.d_out;
        dn.ld_dst = m.n_embd;
        dn.y_scale = ys_d;
        m.mctx->run(dn, m.stream);
    }

    // 4. the run, back to the primary.  Rank r's row lands in eparts_row_base + r, which is the run the header's row
    //    contract gives this helper.
    //
    //    Link: ONE peer copy on the helper's stream; the caller orders the primary against it with wait().
    //    No link: one device-to-host copy of the same rows into the pinned h_out, on the same helper stream, and the
    //    PRIMARY-side upload is left for wait() - the only method that is handed the primary's stream.  What it must
    //    write is remembered rather than passed along, because launch() has no primary stream to issue it on.
    if (m.p2p) {
        if (!ck(cudaMemcpyPeerAsync(eparts_primary + eparts_row_base * m.n_embd, m.primary, m.d_out, m.device,
                                    (size_t) n * (size_t) m.n_embd * sizeof(float), m.stream), "copy the run", err))
            return false;
    } else if (peer_f16()) {
        // STRATA_Q36_PEER_F16=1: the packed up leg.  The rows are packed to half on the helper (which also MEASURES
        // the largest magnitude it saw, so the question "can fp16 carry these values" is answered by a number and
        // not by an assumption), and the copy that follows moves half as many bytes.  The rows themselves are
        // unchanged: same count, same order, same meaning - only the width of an element on the wire.
        if (!ck(cudaMemsetAsync(m.d_f16max, 0, sizeof(unsigned int), m.stream), "clear packed magnitude", err) ||
            !ck(cudaMemsetAsync(m.d_f16rel, 0, sizeof(unsigned int), m.stream), "clear packed relative error", err))
            return false;
        k_pack_f16<<<grid1(n * m.n_embd), kThreads, 0, m.stream>>>(m.d_out, m.d_out16, n * m.n_embd, m.d_f16max,
                                                                m.d_f16rel);
        if (!ck(cudaGetLastError(), "pack the run", err)) return false;
        if (!ck(cudaMemcpyAsync(m.h_out16, m.d_out16, (size_t) n * (size_t) m.n_embd * sizeof(__half),
                                cudaMemcpyDeviceToHost, m.stream), "copy the packed run to host staging", err))
            return false;
        // The measurement comes back with the same copy: 4 bytes next to the rows, read on the helper's stream.
        if (!ck(cudaMemcpyAsync(m.h_f16max, m.d_f16max, sizeof(unsigned int), cudaMemcpyDeviceToHost, m.stream),
                "read the packed magnitude", err) ||
            !ck(cudaMemcpyAsync(m.h_f16rel, m.d_f16rel, sizeof(unsigned int), cudaMemcpyDeviceToHost, m.stream),
                "read the packed relative error", err))
            return false;
        m.pending_eparts = eparts_primary;
        m.pending_base = eparts_row_base;
        m.pending_n = n;
        m.pending_f16 = true;
    } else {
        if (!ck(cudaMemcpyAsync(m.h_out, m.d_out, (size_t) n * (size_t) m.n_embd * sizeof(float),
                                cudaMemcpyDeviceToHost, m.stream), "copy the run to host staging", err))
            return false;
        m.pending_eparts = eparts_primary;
        m.pending_base = eparts_row_base;
        m.pending_n = n;
        // The rows on their way back are floats: pending_f16 is false already (it starts false and the packed route
        // is the only one that sets it, and the flag cannot change under a running process).
    }
    return ck(cudaGetLastError(), "helper launches", err);
}

bool PeerPrefill36::wait(cudaStream_t st_primary, void* fp16_stage, std::string& err) {
    Impl& m = *impl_;
    if (m.device < 0 || m.stream == nullptr) { err = "q36 peer prefill: wait() on a closed executor"; return false; }
    if (st_primary == nullptr) { err = "q36 peer prefill: wait() without a primary stream"; return false; }
    On on(m.device);
    if (!ck(cudaEventRecord(m.ev, m.stream), "record helper completion", err) ||
        !ck(cudaStreamWaitEvent(st_primary, m.ev, 0), "make the primary wait", err))
        return false;
    // THE HOST ROUTE'S UP LEG.  launch() already left this helper's rows in the pinned h_out; here they go to where
    // the header's row contract says they belong, on the PRIMARY's stream.  What orders it, without a new
    // synchronisation: the primary's H2D above waits for `ev`, and `ev` is recorded on the helper's stream AFTER that
    // stream's device-to-host into h_out, so h_out is complete before the H2D reads it.  What orders the NEXT layer's
    // write into h_out (its launch(), on the same helper stream) after this read is the caller: it synchronises the
    // primary's stream before every plan (the same sync prepare()'s blocking D2H already stands on), which retires
    // this H2D on the device before the next launch() can be issued.  pending_n is cleared so a stray second wait()
    // posts no copy.
    if (!m.p2p && m.pending_n > 0) {
        const int64_t n = m.pending_n;
        m.pending_n = 0;
        On onp(m.primary);   // `eparts` and the stream are the primary's: the copy runs on its device
        if (m.pending_f16) {
            // STRATA_Q36_PEER_F16=1: the packed rows land in the PRIMARY's staging, NOT in eparts - the caller
            // unpacks them into eparts on the same stream, so `eparts` still holds floats when the combine reads it.
            // The staging is the caller's: it knows where the rows belong (the row base this helper was given).
            if (fp16_stage == nullptr) {
                err = "q36 peer prefill: the packed up leg needs a staging buffer on the primary";
                return false;
            }
            if (!ck(cudaMemcpyAsync(fp16_stage, m.h_out16, (size_t) n * (size_t) m.n_embd * sizeof(__half),
                                    cudaMemcpyHostToDevice, st_primary), "copy the packed run to the primary", err))
                return false;
        } else if (!ck(cudaMemcpyAsync(m.pending_eparts + m.pending_base * m.n_embd, m.h_out,
                                       (size_t) n * (size_t) m.n_embd * sizeof(float), cudaMemcpyHostToDevice, st_primary),
                       "copy the run into eparts", err))
            return false;
    }
    return ck(cudaGetLastError(), "helper completion", err);
}

int64_t PeerPrefill36::rows() const { return impl_->rows; }
int64_t PeerPrefill36::launches() const { return impl_->launches; }

double PeerPrefill36::last_prepare_d2h_ms() const { return impl_->prep_d2h_ms; }
// STRATA_Q36_PEER_F16=1: what the packing measured - the largest magnitude in the rows and the largest
// relative error of the fp16 round trip.  The second one is the answer to "can fp16 carry these rows";
// the first says whether anything came close to fp16's limit.
double PeerPrefill36::last_f16_max_abs() const {
    if (impl_->h_f16max == nullptr) return 0.0;
    unsigned int bits = *impl_->h_f16max;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return (double) f;
}
double PeerPrefill36::last_f16_max_rel() const {
    if (impl_->h_f16rel == nullptr) return 0.0;
    unsigned int bits = *impl_->h_f16rel;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return (double) f;
}
double PeerPrefill36::last_prepare_sync_ms() const { return impl_->prep_sync_ms; }
int64_t PeerPrefill36::last_prepare_calls() const { return impl_->prep_calls; }

}  // namespace strata::qwen35
