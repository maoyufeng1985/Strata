// src/prefill/mmq_kq_lib.cu - strata_kq, the shared library with llama.cpp's MMQ prompt kernels for the K-quants
// (Q4_K, Q5_K, Q5_1, Q6_K: the experts of Unsloth's UD-Q4/Q5/Q6_K_XL files).  MODEL-UPDATE Phase 1: the engine loads
// every kernel of its own binary when it starts (CUDA_MODULE_LOADING=EAGER, src/program/generate.cpp), so kernels in
// the engine cost every model VRAM; these four instances were ~3.8 MB of sm_60 code (3 expert slots of IQ3_XXS on a
// P100).  Here they load only when moe_mmq.cu opens this library - for a model with these experts, before the expert
// cache is sized - and never otherwise.  The library has its own copy of ggml_cuda_host.cu's symbols and of the static
// CUDA runtime; it works in the engine's context on the stream it is given.
#include "common.cuh"
#include "mmq.cuh"

#include "strata/prefill/mmq_kq_abi.hpp"
#include "mmq_kq_probe.hpp"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#if defined(_WIN32)
#define STRATA_KQ_EXPORT extern "C" __declspec(dllexport)
#else
#define STRATA_KQ_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
std::mutex g_mu;
ggml_backend_cuda_context* g_ctx[GGML_CUDA_MAX_DEVICES] = {};

ggml_backend_cuda_context* ctx_for(int dev) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_ctx[dev] == nullptr) g_ctx[dev] = new ggml_backend_cuda_context(dev);
    return g_ctx[dev];
}

// STRATA_MMQ_KQ_J=<8..128>: the MMQ's J (a tile's output columns, one template instantiation) pinned for the
// K-quant experts, bypassing upstream's minimum-tile switch - a per-card kernel tuning knob.  J changes how the
// product is tiled, not the K order or the float sums inside a tile, so its products are bitwise the switch's.
// Absent or out of range: the upstream switch, as before.
int kq_j() {
    static const int j = [] {
        const char* e = std::getenv("STRATA_MMQ_KQ_J");
        if (e == nullptr) return 0;
        const int v = std::atoi(e);
        if (v < 8 || v > 128 || v % 8 != 0) {
            std::fprintf(stderr, "strata: STRATA_MMQ_KQ_J=%s: expected a multiple of 8 in 8..128 - "
                                 "the upstream switch runs\n", e);
            return 0;
        }
        std::fprintf(stderr, "strata: STRATA_MMQ_KQ_J=%d: the K-quant experts' MMQ takes J=%d (bitwise the "
                             "switch's products)\n", v, v);
        return v;
    }();
    return j;
}

template <ggml_type T, bool Fallback>
bool launch_j1(ggml_backend_cuda_context& ctx, const mmq_args& a, int J, cudaStream_t s) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(T, J, Fallback, cc);
    if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, cc) > smpbo) return false;
    switch (J) {
        case 8:   launch_mul_mat_q<T, 8, Fallback>(ctx, a, s); break;
        case 16:  launch_mul_mat_q<T, 16, Fallback>(ctx, a, s); break;
        case 24:  launch_mul_mat_q<T, 24, Fallback>(ctx, a, s); break;
        case 32:  launch_mul_mat_q<T, 32, Fallback>(ctx, a, s); break;
        case 40:  launch_mul_mat_q<T, 40, Fallback>(ctx, a, s); break;
        case 48:  launch_mul_mat_q<T, 48, Fallback>(ctx, a, s); break;
        case 56:  launch_mul_mat_q<T, 56, Fallback>(ctx, a, s); break;
        case 64:  launch_mul_mat_q<T, 64, Fallback>(ctx, a, s); break;
        case 72:  launch_mul_mat_q<T, 72, Fallback>(ctx, a, s); break;
        case 80:  launch_mul_mat_q<T, 80, Fallback>(ctx, a, s); break;
        case 88:  launch_mul_mat_q<T, 88, Fallback>(ctx, a, s); break;
        case 96:  launch_mul_mat_q<T, 96, Fallback>(ctx, a, s); break;
        case 104: launch_mul_mat_q<T, 104, Fallback>(ctx, a, s); break;
        case 112: launch_mul_mat_q<T, 112, Fallback>(ctx, a, s); break;
        case 120: launch_mul_mat_q<T, 120, Fallback>(ctx, a, s); break;
        case 128: launch_mul_mat_q<T, 128, Fallback>(ctx, a, s); break;
    }
    return true;
}

template <ggml_type T>
bool launch_j(ggml_backend_cuda_context& ctx, const mmq_args& a, int J, cudaStream_t s) {
    // the same gate as mul_mat_q_case: the fallback template dimension is a compile-time constant
    if (a.nrows_x % 128 == 0) return launch_j1<T, false>(ctx, a, J, s);
    return launch_j1<T, true>(ctx, a, J, s);
}

// STRATA_MMQ_KQ_J_PROBE=1: the process-local J probe cache for these four types, on by that env alone.
// The pin (kq_j) overrides it; with neither set, strata_kq_run falls to mul_mat_q_case exactly as before.
bool kq_probe_on() {
    static const int on = [] {
        const char* e = std::getenv("STRATA_MMQ_KQ_J_PROBE");
        if (e == nullptr || e[0] == '\0') return 0;
        const int v = std::atoi(e);
        if (v != 0)
            std::fprintf(stderr, "strata: STRATA_MMQ_KQ_J_PROBE: the K-quant experts' MMQ times up to 3 "
                                 "candidate Js on a cold shape (the winner is cached process-locally per "
                                 "type, rows-bucket and cc)\n");
        return v;
    }();
    return on != 0;
}

namespace kqprobe = strata::prefill::kqprobe;

kqprobe::Cache g_jprobe;
std::mutex g_jprobe_mu;

// kq_j_probe takes the rows%128 split as its Fallback template dimension (like launch_j), so the
// callers below instantiate both split states before the probe decides.

// The cold-miss timing of kqprobe::decide's Probe: for each candidate one warmup launch and one
// event-timed launch, the very launch_mul_mat_q<T, J> through launch_j1's gate, on the caller's stream.
// The winner's timed launch already wrote the product (J retiles the product, not its K order or float
// sums), so no launch follows the cache store.  One miss costs up to 6 launches, ms-class, on the
// first prompt for that shape class; every later call for the key takes the Cached path.
template <ggml_type T, bool Fallback>
bool kq_j_probe(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    const kqprobe::ProbeKey key{T, kqprobe::bucket(a.nrows_x), cc};

    int cached = 0;
    {
        std::lock_guard<std::mutex> lk(g_jprobe_mu);
        cached = g_jprobe.hit(key);
    }
    if (cached != 0) return launch_j1<T, Fallback>(ctx, a, cached, s);

    // the gate copied from launch_j1: the config is a row of the pinned table for (T, J, Fallback, cc)
    // and fits the card's shared-memory bound
    const auto legal = [&](int J) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(T, J, Fallback, cc);
        return config.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(config, cc) <= smpbo;
    };
    const int J_up = kqprobe::upstream_J(legal, (int) a.ncols_opt);
    const std::vector<int> cands = kqprobe::candidates(J_up, legal, (int) a.ncols_opt, 3);
    if (cands.empty()) return false;   // nothing legal: the caller runs the upstream switch

    const int n = (int) cands.size();
    std::vector<float> ms(n, 1e30f);
    std::vector<bool> ok(n, false);
    for (int i = 0; i < n; ++i) {
        if (!(ok[i] = launch_j1<T, Fallback>(ctx, a, cands[i], s))) continue;   // the warmup launch
        cudaEvent_t e0 = nullptr, e1 = nullptr;
        const bool events = cudaEventCreate(&e0) == cudaSuccess && cudaEventCreate(&e1) == cudaSuccess;
        cudaEventRecord(e0, s);
        const bool launched = launch_j1<T, Fallback>(ctx, a, cands[i], s);      // the timed launch
        cudaEventRecord(e1, s);
        if (launched && events) {
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms[i], e0, e1);
        } else {
            ok[i] = false;
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }
    int w = -1;
    for (int i = 0; i < n; ++i)
        if (ok[i] && (w < 0 || ms[i] < ms[w])) w = i;
    if (w < 0) return false;   // every candidate failed: the caller runs the upstream switch

    {
        std::lock_guard<std::mutex> lk(g_jprobe_mu);
        g_jprobe.store(key, cands[w]);
    }
    // cands[0] is the upstream switch's J; the line names both picks so a bad one is visible
    const float ms_u = cands[w] == cands[0] ? ms[w] : ms[0];
    std::fprintf(stderr, "strata: mmq J-probe %s rows~%s cc=%d: pick J=%d (%.3f) over J=%d (%.3f)\n",
                 ggml_type_name(T), kqprobe::bucket_label(key.bucket).c_str(), cc, cands[w], ms[w], cands[0], ms_u);
    return true;
}
}  // namespace

// the interface version and the argument block's size: both sides are compiled from the same mmq.cuh
STRATA_KQ_EXPORT int strata_kq_abi() { return STRATA_KQ_ABI_VERSION * 100000 + (int) sizeof(mmq_args); }

// Starts this library's CUDA runtime in the engine's context on `device`: with CUDA_MODULE_LOADING=EAGER (the engine
// sets it before its first CUDA call) that loads every kernel here now, while the VRAM for it is still free.
// 0 = ready, else the cudaError_t.
STRATA_KQ_EXPORT int strata_kq_init(int device) {
    cudaError_t e = cudaSetDevice(device);
    if (e == cudaSuccess) e = cudaFree(nullptr);
    return (int) e;
}

// One MMQ product (moe_mmq.cu's Context::run for these types) on `device`, on the engine's `stream`.  0 = launched, -1 =
// a type this library does not have, else the cudaError_t of the launch.
// The J choice: the KQ pin (kq_j) if set; else the probe (kq_probe_on) on the K-quant DSO; else the upstream
// switch - that last path is byte-for-byte the pre-probe behavior.
STRATA_KQ_EXPORT int strata_kq_run(int device, int type, const void* args, void* stream) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return -1;
    if (cudaSetDevice(device) != cudaSuccess) return (int) cudaGetLastError();
    ggml_backend_cuda_context& ctx = *ctx_for(device);
    const mmq_args& a = *(const mmq_args*) args;
    const cudaStream_t s = (cudaStream_t) stream;
    const int J = kq_j();
    const bool probe = J == 0 && kq_probe_on();
    switch ((ggml_type) type) {
        case GGML_TYPE_Q4_K:
            if (J != 0) {
                if (launch_j<GGML_TYPE_Q4_K>(ctx, a, J, s)) break;
            } else if (probe) {
                if (a.nrows_x % 128 == 0) {
                    if (kq_j_probe<GGML_TYPE_Q4_K, false>(ctx, a, s)) break;
                } else if (kq_j_probe<GGML_TYPE_Q4_K, true>(ctx, a, s)) break;
            }
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, a, s);
            break;
        case GGML_TYPE_Q5_K:
            if (J != 0) {
                if (launch_j<GGML_TYPE_Q5_K>(ctx, a, J, s)) break;
            } else if (probe) {
                if (a.nrows_x % 128 == 0) {
                    if (kq_j_probe<GGML_TYPE_Q5_K, false>(ctx, a, s)) break;
                } else if (kq_j_probe<GGML_TYPE_Q5_K, true>(ctx, a, s)) break;
            }
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, a, s);
            break;
        case GGML_TYPE_Q5_1:
            if (J != 0) {
                if (launch_j<GGML_TYPE_Q5_1>(ctx, a, J, s)) break;
            } else if (probe) {
                if (a.nrows_x % 128 == 0) {
                    if (kq_j_probe<GGML_TYPE_Q5_1, false>(ctx, a, s)) break;
                } else if (kq_j_probe<GGML_TYPE_Q5_1, true>(ctx, a, s)) break;
            }
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, a, s);
            break;
        case GGML_TYPE_Q6_K:
            if (J != 0) {
                if (launch_j<GGML_TYPE_Q6_K>(ctx, a, J, s)) break;
            } else if (probe) {
                if (a.nrows_x % 128 == 0) {
                    if (kq_j_probe<GGML_TYPE_Q6_K, false>(ctx, a, s)) break;
                } else if (kq_j_probe<GGML_TYPE_Q6_K, true>(ctx, a, s)) break;
            }
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, a, s);
            break;
        default: return -1;
    }
    return (int) cudaGetLastError();
}
