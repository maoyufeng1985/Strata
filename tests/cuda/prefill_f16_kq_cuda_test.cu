// prefill_f16_kq_cuda_test - the fused FP16 K-quant prompt experts on CUDA (moe_f16_kq_cuda.cu): the gate/up of a
// Q4_K expert against a double-precision reference.
//
// Two checks:
//   1. the weights: the kernel's dequantization (dequant_block) against ggml's dequantize_row_q4_K, value by
//      value.  The kernel computes in FP16 and ggml in FP32, so the error is not zero: the bound is in
//      quantization steps (the weight's own step, d * sc), and it is the FP16 rounding of the scale and the
//      minimum.
//   2. the int8 H (the codes, the scales, the codes' sums) against a double reference: the gate/up products, the
//      SwiGLU and the int8 H in the engine's native layout (per 32 values a scale and the codes' sum).
//
// The H codes are not bit-exact either: the kernel accumulates the products in FP16.  The SwiGLU's exponential
// part turns a small error in the gate value into a large one where the gate value is negative (silu(-8) is
// -0.0027: a 1% error in the gate value is an 8% error in the result), so the bounds are the measured ones
// (plans/2026-10-24-q36-fp16-cuda-landing-plan.md), not zero.
#include "strata/prefill/moe_fused_iq.hpp"

#include "../../src/prefill/moe_f16_kq.hpp"
#include "../../src/prefill/moe_f16_kq_cuda.cuh"

#include "ggml.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// ggml-quants.h (ggml-base): the scalar reference (the block's type is not needed: the bytes are cast).
extern "C" void dequantize_row_q4_K(const void* x, float* y, int64_t k);

namespace {

namespace f16kqc = strata::prefill::fused::f16kqc;
using strata::prefill::fused::Batch;
using strata::prefill::fused::NativeGeom;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

constexpr int N_EMBD = 2048, N_FF = 512;   // the UD-Q4_K_XL layer: gate/up Q4_K, K = n_embd
constexpr int ACT_ABS = 80;                 // bytes per 64 activation values: 64 codes, {d0,d1,sum0,sum1}

// The kernel's dequantization of one Q4_K block: one thread per (64-value group, byte in the group), the 8 values
// it owns.
__global__ void dump_blocks(const uint8_t* __restrict__ blob, size_t gu_row, int nsuper, float* __restrict__ out) {
    const int idx = blockIdx.x;                       // one block per (row, superblock)
    const int row = idx / nsuper, s = idx % nsuper;
    const uint8_t* blk = blob + (size_t) row * gu_row + (size_t) s * 144;
    const int t = threadIdx.x;                        // 0..31: (c64, j4)
    const int c64 = t >> 3, j4 = (t & 7) * 4;
    __half2 wq[4];
    f16kqc::dequant_block<f16kqc::T_Q4_K>(blk, c64, j4, wq);
    const float2 a = __half22float2(wq[0]), b = __half22float2(wq[1]);
    const float2 c = __half22float2(wq[2]), d = __half22float2(wq[3]);
    const int base = 64 * c64 + j4;
    float* o = out + (size_t) idx * 256;
    o[base] = a.x; o[base + 1] = a.y; o[base + 2] = b.x; o[base + 3] = b.y;
    o[base + 32] = c.x; o[base + 32 + 1] = c.y; o[base + 32 + 2] = d.x; o[base + 32 + 3] = d.y;
}

}  // namespace

int main() {
    // ---- the expert: the gate rows, then the up rows, quantized by ggml ----
    const size_t gu_row = (size_t) (N_EMBD / 256) * 144;
    const size_t up_off = (size_t) N_FF * gu_row;
    const int nsuper = N_EMBD / 256;
    const auto* traits = ggml_get_type_traits(GGML_TYPE_Q4_K);
    if (traits == nullptr || traits->from_float_ref == nullptr) {
        std::printf("prefill_f16_kq_cuda_test: ggml has no Q4_K quantizer\n");
        return 77;
    }
    std::vector<uint8_t> blob((size_t) (2 * N_FF) * gu_row);
    std::vector<float> row(N_EMBD);
    unsigned seed = 12345;
    auto rnd = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return (float)((int)((seed >> 8) & 0xFFFF) - 32768) * 0.001f;
    };
    for (int r = 0; r < 2 * N_FF; ++r) {
        for (int k = 0; k < N_EMBD; ++k) row[k] = rnd();
        traits->from_float_ref(row.data(), blob.data() + (size_t) r * gu_row, N_EMBD);
    }

    // ---- check 1: the weights, value by value, against ggml ----
    {
        Dev d_blob(blob.size()), d_out((size_t) 2 * N_FF * nsuper * 256 * 4);
        ck(cudaMemcpy(d_blob.p, blob.data(), blob.size(), cudaMemcpyHostToDevice), "copy");
        dump_blocks<<<2 * N_FF * nsuper, 32>>>((const uint8_t*) d_blob.p, gu_row, nsuper, (float*) d_out.p);
        ck(cudaDeviceSynchronize(), "launch");
        ck(cudaGetLastError(), "kernel");
        std::vector<float> got((size_t) 2 * N_FF * nsuper * 256);
        ck(cudaMemcpy(got.data(), d_out.p, got.size() * 4, cudaMemcpyDeviceToHost), "copy back");

        double worst_step = 0.0, worst_abs = 0.0;
        long bad = 0;
        std::vector<float> ref(256);
        for (int r = 0; r < 2 * N_FF; ++r) {
            for (int s = 0; s < nsuper; ++s) {
                const uint8_t* blk = blob.data() + (size_t) r * gu_row + (size_t) s * 144;
                dequantize_row_q4_K((const void*) blk, ref.data(), 256);
                // the step of this block's quantization: d * sc, the largest of the four scale groups
                const float d = __half2float(__ushort_as_half(*(const uint16_t*) blk));
                float step = 0.f;
                for (int g = 0; g < 4; ++g) step = fmaxf(step, d * (float) (blk[4 + 2 * g] & 63));
                for (int i = 0; i < 256; ++i) {
                    const double diff = fabs((double) got[((size_t) r * nsuper + s) * 256 + i] - ref[i]);
                    if (diff > 0.0) bad++;
                    worst_abs = fmax(worst_abs, diff);
                    worst_step = fmax(worst_step, diff / (step + 1e-30));
                }
            }
        }
        std::printf("weights: %d values, %ld differ from ggml, max |diff| %.3e, max diff %.4f steps\n",
                    2 * N_FF * nsuper * 256, bad, worst_abs, worst_step);
        if (worst_step > 0.05) {
            std::printf("prefill_f16_kq_cuda_test: the dequantization is off by %.4f steps (the bound is 0.05)\n",
                        worst_step);
            return 1;
        }
    }

    // ---- the activations in the engine's int8 form (quant_act_native) ----
    const int ntok = 64, nrows = 64;
    const size_t act_row = (size_t) (N_EMBD / 64) * ACT_ABS;
    std::vector<uint8_t> act((size_t) ntok * act_row);
    std::vector<float> xq((size_t) ntok * N_EMBD);
    {
        std::vector<float> x((size_t) ntok * N_EMBD);
        for (size_t i = 0; i < x.size(); ++i) x[i] = (float)((int)(i % 2001) - 1000) * 0.001f;
        for (int t = 0; t < ntok; ++t) {
            for (int b = 0; b < N_EMBD / 64; ++b) {
                uint8_t* o = &act[(size_t) t * act_row + (size_t) b * ACT_ABS];
                float a0 = 0.f, a1 = 0.f;
                for (int l = 0; l < 32; ++l) {
                    a0 = fmaxf(a0, fabsf(x[(size_t) t * N_EMBD + b * 64 + l]));
                    a1 = fmaxf(a1, fabsf(x[(size_t) t * N_EMBD + b * 64 + 32 + l]));
                }
                int s0 = 0, s1 = 0;
                for (int l = 0; l < 32; ++l) {
                    const int c0 = a0 > 0.f ? (int) lrintf(x[(size_t) t * N_EMBD + b * 64 + l] * (127.f / a0)) : 0;
                    const int c1 = a1 > 0.f ? (int) lrintf(x[(size_t) t * N_EMBD + b * 64 + 32 + l] * (127.f / a1)) : 0;
                    o[l] = (uint8_t) (int8_t) c0;
                    o[32 + l] = (uint8_t) (int8_t) c1;
                    s0 += c0;
                    s1 += c1;
                }
                *(float4*) (o + 64) = make_float4(a0 / 127.f, a1 / 127.f, (float) s0, (float) s1);
                for (int l = 0; l < 32; ++l) {
                    xq[(size_t) t * N_EMBD + b * 64 + l] = (float) (int8_t) o[l] * (a0 / 127.f);
                    xq[(size_t) t * N_EMBD + b * 64 + 32 + l] = (float) (int8_t) o[32 + l] * (a1 / 127.f);
                }
            }
        }
    }

    // ---- the tables: one expert, one tile of `nrows` rows ----
    std::vector<int32_t> hOff = {0, nrows}, hTs = {0, 1}, hSrc(nrows);
    std::vector<int2> hTiles = {{0, 0}};
    for (int r = 0; r < nrows; ++r) hSrc[r] = r % ntok;

    const size_t hbytes = (size_t) nrows * (N_FF / 64) * ACT_ABS;
    Dev d_act(act.size()), d_out(hbytes), d_off(hOff.size() * 4), d_tiles(hTiles.size() * 8), d_ts(hTs.size() * 4),
        d_src(hSrc.size() * 4);
    ck(cudaMemcpy(d_act.p, act.data(), act.size(), cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(d_off.p, hOff.data(), hOff.size() * 4, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(d_tiles.p, hTiles.data(), hTiles.size() * 8, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(d_ts.p, hTs.data(), hTs.size() * 4, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(d_src.p, hSrc.data(), hSrc.size() * 4, cudaMemcpyHostToDevice), "copy");

    Batch b;
    b.e0 = 0;
    b.e1 = 1;
    Dev d_blob(blob.size());
    ck(cudaMemcpy(d_blob.p, blob.data(), blob.size(), cudaMemcpyHostToDevice), "copy");
    b.blob[0] = (const uint8_t*) d_blob.p;   // the expert's blob (the gate rows, then the up rows)

    NativeGeom g;
    g.gu_type = GGML_TYPE_Q4_K;
    g.d_type = -1;
    g.gu_row = gu_row;
    g.up_off = up_off;
    g.n_embd = N_EMBD;
    g.n_ff = N_FF;

    strata::prefill::fused::f16kq::gate_up(b, g, (const int32_t*) d_ts.p, (const int2*) d_tiles.p,
                                            (const int32_t*) d_off.p, nrows, 1, 1, (const uint8_t*) d_act.p,
                                            (const int32_t*) d_src.p, (uint8_t*) d_out.p, 0);
    ck(cudaDeviceSynchronize(), "launch");
    ck(cudaGetLastError(), "kernel");
    std::vector<uint8_t> hOut(hbytes);
    ck(cudaMemcpy(hOut.data(), d_out.p, hbytes, cudaMemcpyDeviceToHost), "copy back");

    // ---- check 2: the H against a double reference ----
    {
        std::vector<float> w(N_EMBD), gv(N_FF), uv(N_FF);
        std::vector<float> ref(256);
        double worst_code = 0.0, worst_scale = 0.0, worst_sum = 0.0;
        long diff_codes = 0, total = 0;
        for (int r = 0; r < nrows; ++r) {
            const int tok = hSrc[r];
            for (int i = 0; i < N_FF; ++i) {
                const uint8_t* rw = blob.data() + (size_t) i * gu_row;
                for (int s = 0; s < nsuper; ++s) {
                    dequantize_row_q4_K((const void*) (rw + (size_t) s * 144), ref.data(), 256);
                    for (int k = 0; k < 256; ++k) w[(size_t) s * 256 + k] = ref[k];
                }
                double acc = 0.0;
                for (int k = 0; k < N_EMBD; ++k) acc += (double) w[k] * (double) xq[(size_t) tok * N_EMBD + k];
                gv[i] = (float) acc;
            }
            for (int i = 0; i < N_FF; ++i) {
                const uint8_t* rw = blob.data() + up_off + (size_t) i * gu_row;
                for (int s = 0; s < nsuper; ++s) {
                    dequantize_row_q4_K((const void*) (rw + (size_t) s * 144), ref.data(), 256);
                    for (int k = 0; k < 256; ++k) w[(size_t) s * 256 + k] = ref[k];
                }
                double acc = 0.0;
                for (int k = 0; k < N_EMBD; ++k) acc += (double) w[k] * (double) xq[(size_t) tok * N_EMBD + k];
                uv[i] = (float) acc;
            }
            for (int hb = 0; hb < N_FF / 64; ++hb) {
                for (int hh = 0; hh < 2; ++hh) {
                    const uint8_t* o = &hOut[(size_t) r * (N_FF / 64) * ACT_ABS + (size_t) hb * ACT_ABS];
                    float h[32], am = 0.f;
                    for (int j = 0; j < 32; ++j) {
                        const float gt = gv[hb * 64 + hh * 32 + j], up = uv[hb * 64 + hh * 32 + j];
                        h[j] = gt / (1.0f + expf(-gt)) * up;
                        am = fmaxf(am, fabsf(h[j]));
                    }
                    const float inv = am > 0.f ? 127.f / am : 0.f;
                    int csum = 0;
                    for (int j = 0; j < 32; ++j) {
                        const int c = (int) lrintf(h[j] * inv);
                        csum += c;
                        const int got = (int8_t) o[hh * 32 + j];
                        total++;
                        if (got != c) {
                            diff_codes++;
                            worst_code = fmax(worst_code, fabs((double) got - c));
                        }
                    }
                    const float dref = am / 127.f;
                    const float dgot = *(const float*) (o + 64 + 4 * hh);
                    worst_scale = fmax(worst_scale, fabs((double) (dgot - dref)) / (dref + 1e-30));
                    const int sgot = (int) *(const float*) (o + 72 + 4 * hh);
                    worst_sum = fmax(worst_sum, fabs((double) (sgot - csum)));
                }
            }
        }
        std::printf("H: %ld codes differ of %ld (max |diff| %.0f), scale rel err %.3e, code sum max |diff| %.0f\n",
                    diff_codes, total, worst_code, worst_scale, worst_sum);
        if (worst_code > 16.0 || worst_scale > 0.4 || worst_sum > 64.0) {
            std::printf("prefill_f16_kq_cuda_test: the H is out of bounds (code %.0f, scale %.3e, sum %.0f)\n",
                        worst_code, worst_scale, worst_sum);
            return 1;
        }
    }
    std::printf("prefill_f16_kq_cuda_test: ok\n");
    return 0;
}
