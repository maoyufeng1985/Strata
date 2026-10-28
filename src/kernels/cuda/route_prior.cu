// src/kernels/cuda/route_prior.cu - STRATA_ROUTE_PRIOR=lambda: cache-conditioned routing (opt-in, changes the output).
//
// "Mixture of Cache-Conditional Experts" (arXiv 2412.00099), without training: when a token's top-k experts are picked,
// an expert already in VRAM gets a `lambda` bonus on its logit.  The SELECTION uses the biased logits; the WEIGHTS stay
// the router's (softmax over the unbiased logits), renormalised over the k picked like the native router.  Fewer missed
// experts to copy over PCIe or compute on the CPU.  Unlike STRATA_ROUTE_RESIDENT (a missed tail expert swapped for the
// best resident one within a logit margin), every rank can move, by an amount the logits decide.
//
// THE EXPERT COUNT AND THE TOP-K COME FROM THE CALLER, not from a constant.  They used to be 512 and 10 in the body of
// the kernel, which is Flash-Next's shape; a compiled geometry with 256 experts and top-8 (Whittle-Qwen-3.8-45B) then
// read 16 groups of 32 out of a 256-wide row and picked 10 ranks into an 8-wide array.  The caller passes them as launch
// arguments now, so the kernel is one text for every geometry and the arrays are sized by NE / K at the call site.
#include "strata/kernels/route_prior.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

namespace strata::kernels {
namespace {

__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m, 32));
    return v;
}
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m, 32);
    return v;
}

// one warp per token; the 32 lanes split the NE logits, so a lane holds ceil(NE/32) of them
constexpr int kMaxPerLane = 16;      // 512 experts (Flash-Next); asserted against the caller's NPL below
__global__ void route_prior_k(const float* __restrict__ logits, const int32_t* __restrict__ res, float lambda, int NE,
                              int K, int NPL, int32_t* __restrict__ ids, float* __restrict__ weights) {
    const float* lg = logits + (size_t) blockIdx.x * NE;
    int32_t* id = ids + (size_t) blockIdx.x * K;
    float* w = weights + (size_t) blockIdx.x * K;
    const int lane = threadIdx.x;
    float p[kMaxPerLane], s[kMaxPerLane];
    float mx = -INFINITY;
    for (int i = 0; i < NPL; ++i) {
        const int e = lane + 32 * i;
        p[i] = e < NE ? lg[e] : -INFINITY;
        mx = fmaxf(mx, p[i]);
    }
    mx = wmax(mx);
    float sum = 0.0f;
    for (int i = 0; i < NPL; ++i) {
        const int e = lane + 32 * i;
        s[i] = p[i] + (e < NE && res[e] >= 0 ? lambda : 0.0f);   // the selection score
        p[i] = e < NE ? expf(p[i] - mx) : 0.0f;                  // the router's probability (unnormalised)
        sum += p[i];
    }
    (void) wsum(sum);
    float sel_p = 0.0f, sel_sum = 0.0f;
    for (int rank = 0; rank < K; ++rank) {
        float best = -FLT_MAX, bp = 0.0f;
        int be = 1 << 30;
        for (int i = 0; i < NPL; ++i)
            if (s[i] > best) { best = s[i]; be = lane + 32 * i; bp = p[i]; }
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, m, 32);
            const int oe = __shfl_xor_sync(0xffffffffu, be, m, 32);
            const float op = __shfl_xor_sync(0xffffffffu, bp, m, 32);
            if (ob > best || (ob == best && oe < be)) { best = ob; be = oe; bp = op; }
        }
        if (be >= 0 && (be & 31) == lane) s[be >> 5] = -FLT_MAX;
        if (lane == rank) { id[rank] = be; sel_p = bp; }
        sel_sum += bp;                                      // the same on every lane
    }
    if (lane < K) w[lane] = sel_p / fmaxf(sel_sum, 1e-30f);
}

}  // namespace

void route_prior_topk(const float* logits, const int32_t* res_layer, float lambda, int32_t* ids, float* weights,
                      int n_tok, int n_expert, int k, void* stream) {
    if (n_tok <= 0 || n_expert <= 0 || k <= 0) return;
    const int npl = (n_expert + 31) / 32;
    if (npl > kMaxPerLane) return;          // 512 experts is the widest this kernel is written for
    route_prior_k<<<n_tok, 32, 0, (cudaStream_t) stream>>>(logits, res_layer, lambda, n_expert, k, npl, ids, weights);
}

}  // namespace strata::kernels
