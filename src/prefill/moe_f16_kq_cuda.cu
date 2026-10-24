// src/prefill/moe_f16_kq_cuda.cu - the fused FP16 K-quant prompt experts on CUDA.
//
// The kernel is moe_f16_kq_cuda.cuh's; this file the interface and the launch.
//
// The differences from the gfx11 path (moe_f16_kq.cu):
//   - NVIDIA: no F16 matrix cores on the P100, so the weights are dequantized into
//     registers and multiplied with HFMA2 (the gfx11 path stages the codes in LDS and
//     uses WMMA).  No FP16 copy of the weights either way.
//   - the geometry comes from the layer (NativeGeom): the gfx11 path hard-codes 2560/640,
//     this one takes n_embd and n_ff (UD-Q4_K_XL: 2048/512).
//
// The opt-in is STRATA_PF_F16_CUDA=1 (f16kqc::requested): the gfx11 path's STRATA_PF_F16
// keeps its own meaning.
#include "moe_f16_kq_cuda.hpp"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::fused::f16kqc {

bool requested() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_PF_F16_CUDA");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

}  // namespace strata::prefill::fused::f16kqc

// The HIP build compiles the gfx11 file (moe_f16_kq.cu) instead: the same symbols, one build
// or the other.  The switch above is compiled in both (it is plain host code).
#if !defined(__HIP_PLATFORM_AMD__)

#include "moe_f16_kq_cuda.cuh"

namespace strata::prefill::fused::f16kq {

static_assert(f16kqc::T_Q4_K == GGML_TYPE_Q4_K, "ggml's type ids moved");
static_assert(f16kqc::T_Q5_K == GGML_TYPE_Q5_K, "ggml's type ids moved");
static_assert(f16kqc::Q4_BLK == 144 && f16kqc::Q5_BLK == 176, "the block sizes moved");

bool covers(int gu_type) { return gu_type == GGML_TYPE_Q4_K; }

void gate_up(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t n,
             int n_expert, int64_t work_bound, const uint8_t* act, const int32_t* src, uint8_t* out, cudaStream_t s) {
    (void) n; (void) n_expert;
    const int n_embd = (int) g.n_embd, n_ff = (int) g.n_ff;
    if (b.e1 <= b.e0 || n_ff <= 0 || n_embd <= 0 || work_bound <= 0) return;
    // one block per (32 output indices, tile, 16 rows); the tile has at most kTileRows rows.  The tile count
    // is the host's (work_bound): ts lives on the device.
    const dim3 grid((n_ff + f16kqc::WIB - 1) / f16kqc::WIB, (unsigned) work_bound, kTileRows / f16kqc::RWB);
    f16kqc::gate_up_fp16<GGML_TYPE_Q4_K><<<grid, f16kqc::THREADS, 0, s>>>(b, ts, tiles, off, act, src, out, n_ff, n_embd,
                                                                          g.gu_row, g.up_off);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "moe_f16_kq_cuda: the gate/up launch failed: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

bool covers_down(int d_type) {
    (void) d_type;
    return false;   // the Q5_K down is not here yet
}

void down(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t n,
          int n_expert, int64_t work_bound, const uint8_t* h, float* dm, cudaStream_t s) {
    (void) b; (void) g; (void) ts; (void) tiles; (void) off; (void) n; (void) n_expert; (void) work_bound; (void) h;
    (void) dm; (void) s;
    std::fprintf(stderr, "moe_f16_kq_cuda: the down projection has no CUDA kernel yet\n");
    std::exit(1);
}

}  // namespace strata::prefill::fused::f16kq

#endif  // !__HIP_PLATFORM_AMD__
