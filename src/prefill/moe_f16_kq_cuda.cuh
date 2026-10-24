// src/prefill/moe_f16_kq_cuda.cuh - the fused FP16 gate/up for the K-quants, on CUDA.
//
// One block: RWB routed rows x WIB output indices (WIB gate rows + WIB up rows), 8 warps.
//   warp w owns 8 weight rows: warps 0..3 the gate rows, warps 4..7 the up rows of the
//   same output indices, so the SwiGLU pairs them without leaving the block.
//   Per warp: RWB rows x 8 weight rows, the accumulators in registers (half2).
//
// Along K, one superblock (256 values) at a time:
//   - the activation (int8, one fp32 scale per 32 values) is staged into shared memory
//     as FP16 once per block, so the int8 -> FP16 conversion is paid once per row and
//     not once per column group;
//   - the K-quant codes of the warp's 8 weight rows are read and dequantized into
//     registers (the codes are never written back: no FP16 copy of the weights);
//   - the FMA is a half2 HFMA2 against the staged activation.
//
// The lane's K indices inside a superblock are the ones ggml's dequantize_row_q4_K /
// dequantize_row_q5_K use (low nibbles first): 64*c64 + j4 for the low nibbles and
// 64*c64 + 32 + j4 for the high ones, with the scale groups 2*c64 and 2*c64+1.
//
// The dequant subtracts 1024 before the packed FMA: the bias -(1024*d*s + dmin*m) is an
// fp16 of magnitude 1024*d*s, where the fp16 spacing is d*s (one quantization step), so
// that bias alone would carry half a step into the weight.
#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "strata/prefill/moe_fused.hpp"

namespace strata::prefill::fused::f16kqc {

/// ggml's type ids (checked against ggml.h in moe_f16_kq_cuda.cu).
constexpr int T_Q4_K = 12, T_Q5_K = 13;

constexpr int AB = 80;          ///< activation bytes per 64 values: 64 codes, {d0,d1,sum0,sum1}
constexpr int QK = 256;         ///< a K-quant superblock
constexpr int RWB = 16;         ///< routed rows per block
constexpr int WIB = 32;         ///< output indices per block (WIB gate rows + WIB up rows)
constexpr int WARPS = 8;        ///< warps per block, 8 weight rows each
constexpr int THREADS = 32 * WARPS;
constexpr int Q4_BLK = 144, Q5_BLK = 176;

__host__ __device__ __forceinline__ int blk_bytes(int qt) { return qt == T_Q4_K ? Q4_BLK : Q5_BLK; }
/// where the codes start in a block (Q4_K: right after the scales; Q5_K: after the 32 qh bytes)
__host__ __device__ __forceinline__ int code_off(int qt) { return qt == T_Q4_K ? 16 : 48; }

__device__ __forceinline__ uint32_t ldg32(const void* p) {
    uint32_t v; asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ uint4 ldg128(const void* p) {
    uint4 v; asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v;
}

/// ggml's get_scale_min_k4, branchless: the branch on j<4 is taken by lanes 0..15 and
/// not by 16..31 of the same warp (c64 = lane>>3), so the warp ran both paths.  `q` points
/// into a 16-byte array with the 12 scale bytes at q[0..11], so q[j-4] is always readable.
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    const uint8_t a0 = q[j];
    const uint8_t a1 = q[j + 4];
    const uint8_t a3 = q[j - 4];
    const bool hi = (j >= 4);
    *d = hi ? (uint8_t) ((a1 & 0xF) | ((a3 >> 6) << 4)) : (uint8_t) (a0 & 63);
    *m = hi ? (uint8_t) ((a1 >>  4) | ((a0 >> 6) << 4)) : (uint8_t) (a1 & 63);
}

/// two codes (one per byte) to two halves of (1024 + code); the caller subtracts 1024.
__device__ __forceinline__ __half2 pack2(uint32_t v0, uint32_t v1, int mask) {
    const uint32_t p = 0x64006400u | (v0 & mask) | (((v1 & mask)) << 16);
    return *(__half2*)&p;
}

/// two int8 codes and their scale to two halves.
__device__ __forceinline__ __half2 code2half2(uint32_t two, float d) {
    const int8_t a = (int8_t) (two & 0xFF);
    const int8_t b = (int8_t) ((two >> 8) & 0xFF);
    return __floats2half2_rn((float) a * d, (float) b * d);
}

/// The lane's 8 values of one 64-value block of an int8 activation row (64 codes, then
/// {d0,d1,sum0,sum1}: d0 for the values 0..31, d1 for 32..63), as FP16.
__device__ __forceinline__ void stage_block(const uint8_t* a, int j4, __half2 v[4]) {
    const uint32_t cl = ldg32(a + j4);
    const uint32_t ch = ldg32(a + 32 + j4);
    const float d0 = *(const float*) (a + 64);
    const float d1 = *(const float*) (a + 68);
    v[0] = code2half2(cl, d0);
    v[1] = code2half2(cl >> 16, d0);
    v[2] = code2half2(ch, d1);
    v[3] = code2half2(ch >> 16, d1);
}

/// The lane's 8 values of one superblock of one weight row, dequantized.
///
/// `wp` the block's start, `c64` the 64-value group (0..3), `j4` the lane's first byte
/// in the group (0, 4, ... 28).  wq[0..1] hold the low nibbles (the values 64*c64 + j4
/// + 0..3), wq[2..3] the high nibbles (the values 64*c64 + 32 + j4 + 0..3).
template <int QT>
__device__ __forceinline__ void dequant_block(const uint8_t* __restrict__ wp, int c64, int j4, __half2 wq[4]) {
    const int cmask = (QT == T_Q4_K) ? 0x0F : 0x1F;
    const int p = 2 * c64;                 // the fifth bit's place in qh[l] (Q5_K)
    const uint4 h = ldg128(wp);
    const __half d = __ushort_as_half((unsigned short) (h.x & 0xFFFFu));
    const __half dmin = __ushort_as_half((unsigned short) (h.x >> 16));
    uint8_t sc[16] = {};
    sc[4] = (uint8_t) h.y;         sc[5] = (uint8_t) (h.y >> 8);
    sc[6] = (uint8_t) (h.y >> 16); sc[7] = (uint8_t) (h.y >> 24);
    sc[8] = (uint8_t) h.z;         sc[9] = (uint8_t) (h.z >> 8);
    sc[10] = (uint8_t) (h.z >> 16); sc[11] = (uint8_t) (h.z >> 24);
    sc[12] = (uint8_t) h.w;        sc[13] = (uint8_t) (h.w >> 8);
    sc[14] = (uint8_t) (h.w >> 16); sc[15] = (uint8_t) (h.w >> 24);
    uint8_t s0, m0, s1, m1;
    get_scale_min_k4(2 * c64, sc + 4, &s0, &m0);
    get_scale_min_k4(2 * c64 + 1, sc + 4, &s1, &m1);
    const float df = __half2float(d), dmf = __half2float(dmin);
    const __half2 ds0 = __floats2half2_rn(df * (float) s0, df * (float) s0);
    const __half2 ds1 = __floats2half2_rn(df * (float) s1, df * (float) s1);
    const __half2 bs0 = __floats2half2_rn(-dmf * (float) m0, -dmf * (float) m0);
    const __half2 bs1 = __floats2half2_rn(-dmf * (float) m1, -dmf * (float) m1);
    const uint32_t raw = ldg32(wp + code_off(QT) + 32 * c64 + j4);
    uint32_t qlo = raw & 0x0f0f0f0fu;
    uint32_t qhi = (raw >> 4) & 0x0f0f0f0fu;
    if (QT == T_Q5_K) {
        const uint32_t qhb = ldg32(wp + 16 + j4);
        qlo |= ((qhb & (0x01010101u << p)) >> p) * 16u;
        qhi |= ((qhb & (0x01010101u << (p + 1))) >> (p + 1)) * 16u;
    }
    const __half2 c1024 = __floats2half2_rn(1024.0f, 1024.0f);
    wq[0] = __hfma2(__hsub2(pack2(qlo, qlo >> 8, cmask), c1024), ds0, bs0);
    wq[1] = __hfma2(__hsub2(pack2(qlo >> 16, qlo >> 24, cmask), c1024), ds0, bs0);
    wq[2] = __hfma2(__hsub2(pack2(qhi, qhi >> 8, cmask), c1024), ds1, bs1);
    wq[3] = __hfma2(__hsub2(pack2(qhi >> 16, qhi >> 24, cmask), c1024), ds1, bs1);
}

/// The fused FP16 gate/up + SwiGLU + the int8 H of one tile.
///
/// `tiles` is the grouping's tile list (expert, first row), `off` the experts' row
/// offsets, `act` the int8 activations (per token), `src` the routed row -> its token,
/// `out` H (int8, the activation layout).  The block's columns are the output indices
/// [blockIdx.x * WIB, +WIB); the rows are the tile's rows [row0 + blockIdx.z * RWB, +RWB).
/// `gu_row` the bytes of one gate/up weight row (the engine's gu_row) and `up_off` the up rows' offset in
/// the blob.  `ts` the tile table (the batch's first tile is ts[b.e0], read on the device).
template <int QT>
__global__ __launch_bounds__(THREADS) void gate_up_fp16(const Batch b, const int32_t* __restrict__ ts,
                                                        const int2* __restrict__ tiles,
                                                        const int32_t* __restrict__ off,
                                                        const uint8_t* __restrict__ act,
                                                        const int32_t* __restrict__ src,
                                                        uint8_t* __restrict__ out, int n_ff, int n_embd,
                                                        size_t gu_row, size_t up_off) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int c64 = lane >> 3;             // the 64-value group inside the superblock
    const int j4 = (lane & 7) * 4;         // this lane's 4 bytes in the group

    const int ntile = ts[b.e1] - ts[b.e0];       // the batch's tiles (on the device: the grid's y is an upper bound)
    if (blockIdx.y >= ntile) return;
    const int2 tl = tiles[ts[b.e0] + blockIdx.y];
    const int e = tl.x, row0 = tl.y;
    const int nrows = min(64, off[e + 1] - row0);
    const int rbase = row0 + blockIdx.z * RWB;
    if (rbase >= row0 + nrows) return;

    const uint8_t* blob = b.blob[e - b.e0];
    const int col0 = blockIdx.x * WIB;                 // the first output index
    const bool up = warp >= 4;                          // this warp's rows are the up rows
    const int wcol = col0 + (warp & 3) * 8;             // the first of the warp's 8 weight rows
    const uint8_t* wbase = blob + (up ? up_off : 0);

    const size_t act_row = (size_t) (n_embd / 64) * AB;
    const int nblk = n_embd / QK;
    const int last = row0 + nrows - 1;                  // the rows past the tile's end repeat its last one

    __shared__ __half sx[RWB][QK];       // the staged activation (FP16)
    __shared__ float shg[RWB][WIB];       // the gate results (then the SwiGLU results)
    __shared__ float shu[RWB][WIB];       // the up results
    __shared__ float sham[RWB];           // the H block's scale, per row
    __shared__ int shsum[RWB][4];         // the codes' sums, per row and per warp

    __half2 acc[RWB][8];
#pragma unroll
    for (int r = 0; r < RWB; ++r)
#pragma unroll
        for (int c = 0; c < 8; ++c) acc[r][c] = __float2half2_rn(0.f);

    for (int b = 0; b < nblk; ++b) {
        __syncthreads();
        // ---- stage this warp's two activation rows as FP16 ----
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const int r = warp * 2 + rr;
            if (r >= RWB) break;
            const int row = min(rbase + r, last);
            const uint8_t* a = act + (size_t) src[row] * act_row + (size_t) (b * 4 + c64) * AB;
            __half2 v[4];
            stage_block(a, j4, v);
            *(uint2*) &sx[r][64 * c64 + j4]      = *(const uint2*) &v[0];
            *(uint2*) &sx[r][64 * c64 + 32 + j4] = *(const uint2*) &v[2];
        }
        __syncthreads();

        // ---- dequantize the warp's 8 weight rows for this superblock ----
        __half2 wq[8][4];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const uint8_t* wp = wbase + (size_t) (wcol + c) * gu_row + (size_t) b * blk_bytes(QT);
            dequant_block<QT>(wp, c64, j4, wq[c]);
        }

        // ---- FMA ----
#pragma unroll
        for (int r = 0; r < RWB; ++r) {
            const __half2* sl = (const __half2*) &sx[r][64 * c64 + j4];
            const __half2* su = (const __half2*) &sx[r][64 * c64 + 32 + j4];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                acc[r][c] = __hfma2(wq[c][0], sl[0], acc[r][c]);
                acc[r][c] = __hfma2(wq[c][1], sl[1], acc[r][c]);
                acc[r][c] = __hfma2(wq[c][2], su[0], acc[r][c]);
                acc[r][c] = __hfma2(wq[c][3], su[1], acc[r][c]);
            }
        }
    }

    // ---- reduce the 32 lanes ----
#pragma unroll
    for (int r = 0; r < RWB; ++r) {
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            __half2 p = acc[r][c];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) p = __hadd2(p, __shfl_xor_sync(0xffffffffu, p, off));
            const float2 t = __half22float2(p);
            const float v = t.x + t.y;
            if (lane == 0) {
                if (up) shu[r][(warp - 4) * 8 + c] = v;
                else     shg[r][(warp & 3) * 8 + c] = v;
            }
        }
    }
    __syncthreads();

    // ---- SwiGLU: the H block's scale is the max over all 32 of its values, so the
    //      four gate warps need each other's values (shg holds the SwiGLU results) ----
    const int hh = (col0 / 32) & 1;                 // the half of the 64-value H block
    const int fb = col0 / 64;                       // the 64-value H block
    const int c0 = (warp & 3) * 8;
    if (warp < 4) {
        for (int r = 0; r < RWB; ++r) {
            if (rbase + r > last) break;
            for (int j = 0; j < 8; ++j) {
                const float gt = shg[r][c0 + j], upv = shu[r][c0 + j];
                shg[r][c0 + j] = gt / (1.0f + __expf(-gt)) * upv;
            }
        }
    }
    __syncthreads();
    if (warp < 4) {
        for (int r = 0; r < RWB; ++r) {
            if (rbase + r > last) break;
            float am = 0.0f;
#pragma unroll 8
            for (int j = 0; j < WIB; ++j) am = fmaxf(am, fabsf(shg[r][j]));
            const float inv = am > 0.0f ? 127.0f / am : 0.0f;
            uint8_t* ob = out + (size_t) (rbase + r) * (size_t) (n_ff / 64) * AB + (size_t) fb * AB;
            uint32_t wd[2] = {};
            int csum = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int c = __float2int_rn(shg[r][c0 + j] * inv);
                csum += c;
                wd[j >> 2] |= (uint32_t) (uint8_t) (int8_t) c << (8 * (j & 3));
            }
            *(uint2*) (ob + hh * 32 + c0) = make_uint2(wd[0], wd[1]);
            shsum[r][warp & 3] = csum;
            if ((warp & 3) == 0) sham[r] = am;
        }
    }
    __syncthreads();
    if (warp == 0) {
        for (int r = 0; r < RWB; ++r) {
            if (rbase + r > last) break;
            uint8_t* ob = out + (size_t) (rbase + r) * (size_t) (n_ff / 64) * AB + (size_t) fb * AB;
            *(float*) (ob + 64 + 4 * hh) = sham[r] / 127.0f;
            *(float*) (ob + 72 + 4 * hh) = (float) (shsum[r][0] + shsum[r][1] + shsum[r][2] + shsum[r][3]);
        }
    }
}

/// The fused FP16 down projection: the H (int8, the activation layout) and the K-quant weights
/// -> the routed rows' FP32 outputs.
///
/// The H's rows are the routed rows (one per routed row, in the tiles' order), so no token map is
/// needed here.  The block covers RWB rows x (WARPS * 8) output columns; the K dimension is the
/// H's width (n_in).
template <int QT>
__global__ __launch_bounds__(THREADS) void down_fp16(const Batch b, const int32_t* __restrict__ ts,
                                                     const int2* __restrict__ tiles,
                                                     const int32_t* __restrict__ off,
                                                     const uint8_t* __restrict__ h,
                                                     float* __restrict__ out, int n_out, int n_in,
                                                     size_t d_row, size_t down_off) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int c64 = lane >> 3;             // the 64-value group inside the superblock
    const int j4 = (lane & 7) * 4;         // this lane's 4 bytes in the group

    const int ntile = ts[b.e1] - ts[b.e0];       // the batch's tiles (on the device: the grid's y is an upper bound)
    if (blockIdx.y >= ntile) return;
    const int2 tl = tiles[ts[b.e0] + blockIdx.y];
    const int e = tl.x, row0 = tl.y;
    const int nrows = min(64, off[e + 1] - row0);
    const int rbase = row0 + blockIdx.z * RWB;
    if (rbase >= row0 + nrows) return;

    const uint8_t* blob = b.blob[e - b.e0] + down_off;
    const int col0 = blockIdx.x * (WARPS * 8);       // the first output column
    const int wcol = col0 + warp * 8;                // the first of the warp's 8 weight rows

    const size_t h_row = (size_t) (n_in / 64) * AB;  // the H's bytes per row
    const int nblk = n_in / QK;
    const int last = row0 + nrows - 1;

    __shared__ __half sx[RWB][QK];       // the staged H row (FP16)

    __half2 acc[RWB][8];
#pragma unroll
    for (int r = 0; r < RWB; ++r)
#pragma unroll
        for (int c = 0; c < 8; ++c) acc[r][c] = __float2half2_rn(0.f);

    for (int b = 0; b < nblk; ++b) {
        __syncthreads();
        // ---- stage this warp's two H rows as FP16 ----
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const int r = warp * 2 + rr;
            if (r >= RWB) break;
            const int row = min(rbase + r, last);
            const uint8_t* a = h + (size_t) row * h_row + (size_t) (b * 4 + c64) * AB;
            __half2 v[4];
            stage_block(a, j4, v);
            *(uint2*) &sx[r][64 * c64 + j4]      = *(const uint2*) &v[0];
            *(uint2*) &sx[r][64 * c64 + 32 + j4] = *(const uint2*) &v[2];
        }
        __syncthreads();

        // ---- dequantize the warp's 8 weight rows for this superblock ----
        __half2 wq[8][4];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const uint8_t* wp = blob + (size_t) (wcol + c) * d_row + (size_t) b * blk_bytes(QT);
            dequant_block<QT>(wp, c64, j4, wq[c]);
        }

        // ---- FMA ----
#pragma unroll
        for (int r = 0; r < RWB; ++r) {
            const __half2* sl = (const __half2*) &sx[r][64 * c64 + j4];
            const __half2* su = (const __half2*) &sx[r][64 * c64 + 32 + j4];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                acc[r][c] = __hfma2(wq[c][0], sl[0], acc[r][c]);
                acc[r][c] = __hfma2(wq[c][1], sl[1], acc[r][c]);
                acc[r][c] = __hfma2(wq[c][2], su[0], acc[r][c]);
                acc[r][c] = __hfma2(wq[c][3], su[1], acc[r][c]);
            }
        }
    }

    // ---- reduce the 32 lanes and write the rows ----
    const int col = col0 + warp * 8;
#pragma unroll
    for (int r = 0; r < RWB; ++r) {
        const int row = rbase + r;
        if (row > last) break;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            __half2 p = acc[r][c];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) p = __hadd2(p, __shfl_xor_sync(0xffffffffu, p, off));
            const float2 t = __half22float2(p);
            if (lane == 0) out[(size_t) row * n_out + col + c] = t.x + t.y;
        }
    }
}

}  // namespace strata::prefill::fused::f16kqc

