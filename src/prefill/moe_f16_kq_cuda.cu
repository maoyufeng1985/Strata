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

namespace strata::prefill::fused::f16kqc {

/// The tiles of the experts' rows: one thread, a scan over the experts (256).
__global__ void k_tiles(const int32_t* __restrict__ bounds, int e0, int e1, int n_local, int2* __restrict__ tiles,
                        int max_tiles, int32_t* __restrict__ ts, int32_t* __restrict__ off) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int ntile = 0;
    for (int e = e0; e < e1; ++e) {
        const int first = bounds[e];
        const int end = min(bounds[e + 1], n_local);
        off[e] = min(first, n_local);
        ts[e] = ntile;
        for (int r = first; r < end; r += 64) {
            if (ntile < max_tiles) tiles[ntile] = make_int2(e, r);
            ++ntile;
        }
    }
    off[e1] = min(bounds[e1], n_local);
    ts[e1] = ntile;
}

void build_tiles(const int32_t* bounds, int e0, int e1, int n_local, int2* tiles, int max_tiles, int32_t* ts,
                 int32_t* off, cudaStream_t s) {
    if (e1 <= e0 || max_tiles <= 0) return;
    k_tiles<<<1, 32, 0, s>>>(bounds, e0, e1, n_local, tiles, max_tiles, ts, off);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "moe_f16_kq_cuda: the tile scan failed: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::prefill::fused::f16kqc

namespace strata::prefill::fused::f16kq {

static_assert(f16kqc::T_Q4_K == GGML_TYPE_Q4_K, "ggml's type ids moved");
static_assert(f16kqc::T_Q5_K == GGML_TYPE_Q5_K, "ggml's type ids moved");
static_assert(f16kqc::Q4_BLK == 144 && f16kqc::Q5_BLK == 176, "the block sizes moved");

// The K-quants with a kernel here: Q4_K and Q5_K (the q36 model's gate/up is Q4_K, some of its layers Q5_K).
bool covers(int gu_type) { return gu_type == GGML_TYPE_Q4_K || gu_type == GGML_TYPE_Q5_K; }

void gate_up(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t n,
             int n_expert, int64_t work_bound, const uint8_t* act, const int32_t* src, uint8_t* out, cudaStream_t s) {
    (void) n; (void) n_expert;
    const int n_embd = (int) g.n_embd, n_ff = (int) g.n_ff;
    if (b.e1 <= b.e0 || n_ff <= 0 || n_embd <= 0 || work_bound <= 0) return;
    // one block per (32 output indices, tile, 16 rows); the tile has at most kTileRows rows.  The tile count
    // is the host's (work_bound): ts lives on the device.
    const dim3 grid((n_ff + f16kqc::WIB - 1) / f16kqc::WIB, (unsigned) work_bound, kTileRows / f16kqc::RWB);
    // the template follows the layer's own type: the q36 model's gate/up is Q4_K on most layers, Q5_K on some
    if (g.gu_type == GGML_TYPE_Q5_K)
        f16kqc::gate_up_fp16<GGML_TYPE_Q5_K><<<grid, f16kqc::THREADS, 0, s>>>(b, ts, tiles, off, act, src, out, n_ff, n_embd,
                                                                               g.gu_row, g.up_off);
    else
        f16kqc::gate_up_fp16<GGML_TYPE_Q4_K><<<grid, f16kqc::THREADS, 0, s>>>(b, ts, tiles, off, act, src, out, n_ff, n_embd,
                                                                               g.gu_row, g.up_off);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "moe_f16_kq_cuda: the gate/up launch failed: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

// The down types with a kernel here: Q5_K (the q36 model) and Q4_K.
bool covers_down(int d_type) { return d_type == GGML_TYPE_Q4_K || d_type == GGML_TYPE_Q5_K; }

void down(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t n,
          int n_expert, int64_t work_bound, const uint8_t* h, float* dm, cudaStream_t s) {
    (void) n; (void) n_expert;
    const int n_out = (int) g.n_embd, n_in = (int) g.n_ff;
    if (b.e1 <= b.e0 || n_out <= 0 || n_in <= 0 || work_bound <= 0) return;
    // one block per (WARPS warps x 4 output columns, tile, 16 rows); the tile has at most kTileRows rows
    const dim3 grid((n_out + f16kqc::WARPS * 4 - 1) / (f16kqc::WARPS * 4), (unsigned) work_bound,
                    kTileRows / f16kqc::RWB);
    if (g.d_type == GGML_TYPE_Q4_K)
        f16kqc::down_fp16<GGML_TYPE_Q4_K><<<grid, f16kqc::THREADS, 0, s>>>(b, ts, tiles, off, h, dm, n_out, n_in, g.d_row,
                                                                            g.down_off);
    else
        f16kqc::down_fp16<GGML_TYPE_Q5_K><<<grid, f16kqc::THREADS, 0, s>>>(b, ts, tiles, off, h, dm, n_out, n_in, g.d_row,
                                                                            g.down_off);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "moe_f16_kq_cuda: the down launch failed: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::prefill::fused::f16kq

#endif  // !__HIP_PLATFORM_AMD__
