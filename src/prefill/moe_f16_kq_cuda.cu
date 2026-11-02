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

/// The down launch, with the row map.  `ids` is null for the identity (the row the kernel computed is
/// the destination row); with a map the value of the SORTED row goes to `ids[row]`.
static void launch_down(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off,
                        int64_t work_bound, const uint8_t* h, float* dm, const int32_t* ids, cudaStream_t s) {
    const int n_out = (int) g.n_embd, n_in = (int) g.n_ff;
    if (b.e1 <= b.e0 || n_out <= 0 || n_in <= 0 || work_bound <= 0) return;
    // one block per (WARPS warps x 4 output columns, tile, 16 rows); the tile has at most kTileRows rows
    const dim3 grid((n_out + f16kqc::WARPS * 4 - 1) / (f16kqc::WARPS * 4), (unsigned) work_bound,
                    kTileRows / f16kqc::RWB);
    if (g.d_type == GGML_TYPE_Q4_K)
        f16kqc::down_fp16<GGML_TYPE_Q4_K><<<grid, f16kqc::THREADS, 0, s>>>(b, ts, tiles, off, h, dm, n_out, n_in, g.d_row,
                                                                            g.down_off, ids);
    else if (g.d_type == GGML_TYPE_Q6_K)
        f16kqc::down_fp16<GGML_TYPE_Q6_K><<<grid, f16kqc::THREADS, 0, s>>>(b, ts, tiles, off, h, dm, n_out, n_in, g.d_row,
                                                                            g.down_off, ids);
    else
        f16kqc::down_fp16<GGML_TYPE_Q5_K><<<grid, f16kqc::THREADS, 0, s>>>(b, ts, tiles, off, h, dm, n_out, n_in, g.d_row,
                                                                            g.down_off, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "moe_f16_kq_cuda: the down launch failed: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

/// Down for the experts' rows, with the row map: the value of the SORTED row goes to `ids[row]`.
/// CUDA-only (the map is the peer path's), so the shared interface (moe_f16_kq.hpp) stays as it is
/// and the gfx11 build does not see this at all.
void down_rows(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t n,
               int n_expert, int64_t work_bound, const uint8_t* h, float* dm, const int32_t* ids, cudaStream_t s) {
    (void) n; (void) n_expert;
    launch_down(b, g, ts, tiles, off, work_bound, h, dm, ids, s);
}

}  // namespace strata::prefill::fused::f16kqc

namespace strata::prefill::fused::f16kq {

static_assert(f16kqc::T_Q4_K == GGML_TYPE_Q4_K, "ggml's type ids moved");
static_assert(f16kqc::T_Q5_K == GGML_TYPE_Q5_K, "ggml's type ids moved");
static_assert(f16kqc::T_Q6_K == GGML_TYPE_Q6_K, "ggml's type ids moved");
static_assert(f16kqc::Q4_BLK == 144 && f16kqc::Q5_BLK == 176 && f16kqc::Q6_BLK == 210, "the block sizes moved");

// The K-quants with a kernel here: Q4_K, Q5_K and Q6_K (the q36 model's gate/up is Q4_K, some of its layers Q5_K,
// the Whittle model's down projection is Q6_K).
bool covers(int gu_type) {
    return gu_type == GGML_TYPE_Q4_K || gu_type == GGML_TYPE_Q5_K || gu_type == GGML_TYPE_Q6_K;
}

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
    else if (g.gu_type == GGML_TYPE_Q6_K)
        f16kqc::gate_up_fp16<GGML_TYPE_Q6_K><<<grid, f16kqc::THREADS, 0, s>>>(b, ts, tiles, off, act, src, out, n_ff, n_embd,
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

// The down types with a kernel here: Q5_K (the q36 model), Q4_K and Q6_K (the Whittle model's down).
bool covers_down(int d_type) {
    return d_type == GGML_TYPE_Q4_K || d_type == GGML_TYPE_Q5_K || d_type == GGML_TYPE_Q6_K;
}

void down(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t n,
          int n_expert, int64_t work_bound, const uint8_t* h, float* dm, cudaStream_t s) {
    (void) n; (void) n_expert;
    f16kqc::launch_down(b, g, ts, tiles, off, work_bound, h, dm, nullptr, s);
}

}  // namespace strata::prefill::fused::f16kq

// STRATA_PF_PEER_I8: carry the activations to the helper cards, and the helpers' result rows
// back, in per-row INT8 instead of FP16 (or FP32): one 8-bit code and one FP32 scale per row,
// the grid of amax / 127 (254 intervals over [-amax, amax]).  One block per row, 256 threads, 8
// values a thread; the row's max over the threads is two warp reductions, the second over the
// first warps' values.  Rounding to the nearest level bounds the code's error by half a step
// (amax / 254, 0.394% of the row's peak); over a uniformly distributed residual the rms is
// step / sqrt(12) = 0.228% of the peak.  The residual is uncorrelated row to row, so it reads as
// noise in the expert's products - which are integer DP4A on the helpers' cards, not this code.
namespace strata::prefill::fused::i8q {

__global__ void k_f16_rows_to_i8(const uint64_t* __restrict__ in, int cols, int8_t* __restrict__ codes,
                                 float* __restrict__ scales) {
    const int64_t r = blockIdx.x;
    // a uint64_t carries FOUR halves, so a row of `cols` halves is `cols / 4` words wide and the
    // eight values a thread takes per pass are two words.  (An earlier version used `cols / 2` and
    // read eight halves out of one word, which addressed the wrong row and past the buffer.)
    const int64_t base = r * (int64_t) (cols / 4);
    __shared__ float s_w[8], s_scale;
    float amax = 0.f;
    for (int c = (int) threadIdx.x * 8; c < cols; c += 2048) {
        const int64_t w = base + c / 4;
        const uint64_t v0 = in[w], v1 = (w + 1 < base + (int64_t) (cols / 4)) ? in[w + 1] : 0ull;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint64_t v = i < 4 ? v0 : v1;
            if (c + i < cols) amax = fmaxf(amax, fabsf(__half2float(__ushort_as_half((uint16_t) (v >> ((i & 3) * 16))))));
        }
    }
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, o));
    if ((threadIdx.x & 31) == 0) s_w[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x < 32) {
        float a = (threadIdx.x < 8) ? s_w[threadIdx.x] : 0.f;
        // all 32 lanes take part: a mask that drops lane 7 also drops s_w[7]'s value from the row's max
        for (int o = 4; o > 0; o >>= 1) a = fmaxf(a, __shfl_down_sync(0xffffffff, a, o));
        if (threadIdx.x == 0) s_scale = a > 0.f ? a / 127.f : 1.f;
    }
    __syncthreads();
    const float sc = s_scale;
    for (int c = (int) threadIdx.x * 8; c < cols; c += 2048) {
        const int64_t w = base + c / 4;
        const uint64_t v0 = in[w], v1 = (w + 1 < base + (int64_t) (cols / 4)) ? in[w + 1] : 0ull;
        int64_t out = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint64_t v = i < 4 ? v0 : v1;
            int code = 0;
            if (c + i < cols) {
                const int q = (int) roundf(__half2float(__ushort_as_half((uint16_t) (v >> ((i & 3) * 16)))) / sc);
                code = min(127, max(-127, q));
            }
            out |= (int64_t) (uint8_t) (code & 0xff) << (i * 8);
        }
        ((int64_t*) (codes + r * cols))[c / 8] = out;
    }
    if (threadIdx.x == 0) scales[r] = sc;
}

__global__ void k_f32_rows_to_i8(const float4* __restrict__ in, int cols, int8_t* __restrict__ codes,
                                 float* __restrict__ scales) {
    const int64_t r = blockIdx.x;
    __shared__ float s_w[8], s_scale;
    float amax = 0.f;
    for (int c = (int) threadIdx.x * 4; c < cols; c += 1024) {
        const float4 v = in[r * (cols / 4) + c / 4];
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, o));
    if ((threadIdx.x & 31) == 0) s_w[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x < 32) {
        float a = (threadIdx.x < 8) ? s_w[threadIdx.x] : 0.f;
        for (int o = 4; o > 0; o >>= 1) a = fmaxf(a, __shfl_down_sync(0xffffffff, a, o));
        if (threadIdx.x == 0) s_scale = a > 0.f ? a / 127.f : 1.f;
    }
    __syncthreads();
    const float sc = s_scale;
    for (int c = (int) threadIdx.x * 4; c < cols; c += 1024) {
        const float4 v = in[r * (cols / 4) + c / 4];
        int32_t out = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float f = (i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w);
            const int q = (int) roundf(f / sc);
            const int code = min(127, max(-127, q));
            out |= (uint32_t) (uint8_t) (code & 0xff) << (i * 8);
        }
        ((uint32_t*) (codes + r * cols))[c / 4] = out;
    }
    if (threadIdx.x == 0) scales[r] = sc;
}

__global__ void k_i8_rows_to_f32(const int64_t* __restrict__ codes, const float* __restrict__ scales, int cols,
                                 float* __restrict__ out) {
    const int64_t r = blockIdx.x;
    const float sc = scales[r];
    for (int c = (int) threadIdx.x * 8; c < cols; c += 2048) {
        const int64_t v = codes[r * (cols / 8) + c / 8];
        float4 lo, hi;
        lo.x = ((int8_t) (v & 0xff)) * sc;
        lo.y = ((int8_t) ((v >> 8) & 0xff)) * sc;
        lo.z = ((int8_t) ((v >> 16) & 0xff)) * sc;
        lo.w = ((int8_t) ((v >> 24) & 0xff)) * sc;
        hi.x = ((int8_t) ((v >> 32) & 0xff)) * sc;
        hi.y = ((int8_t) ((v >> 40) & 0xff)) * sc;
        hi.z = ((int8_t) ((v >> 48) & 0xff)) * sc;
        hi.w = ((int8_t) ((v >> 56) & 0xff)) * sc;
        if (c + 8 <= cols) {
            ((float4*) (out + r * cols + c))[0] = lo;
            ((float4*) (out + r * cols + c))[1] = hi;
        } else {   // the row's tail, one value at a time
            float* fo = out + r * cols + c;
            for (int i = 0; i < 8 && c + i < cols; ++i) fo[i] = (int8_t) (v >> (i * 8)) * sc;
        }
    }
}

void f16_rows_to_i8(const uint16_t* in, int64_t rows, int64_t cols, int8_t* codes, float* scales, cudaStream_t s) {
    if (rows <= 0 || cols <= 0) return;
    k_f16_rows_to_i8<<<(unsigned) rows, 256, 0, s>>>((const uint64_t*) in, (int) cols, codes, scales);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "moe_f16_kq_cuda: the i8 quantize launch failed: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

void f32_rows_to_i8(const float* in, int64_t rows, int64_t cols, int8_t* codes, float* scales, cudaStream_t s) {
    if (rows <= 0 || cols <= 0) return;
    k_f32_rows_to_i8<<<(unsigned) rows, 256, 0, s>>>((const float4*) in, (int) cols, codes, scales);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "moe_f16_kq_cuda: the i8 quantize launch failed: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

void i8_rows_to_f32(const int8_t* codes, const float* scales, float* out, int64_t rows, int64_t cols, cudaStream_t s) {
    if (rows <= 0 || cols <= 0) return;
    k_i8_rows_to_f32<<<(unsigned) rows, 256, 0, s>>>((const int64_t*) codes, scales, (int) cols, out);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "moe_f16_kq_cuda: the i8 dequantize launch failed: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::prefill::fused::i8q

#endif  // !__HIP_PLATFORM_AMD__
