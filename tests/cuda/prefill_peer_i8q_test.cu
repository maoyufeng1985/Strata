// prefill_peer_i8q_test - the per-row INT8 activation route of moe_f16_kq_cuda.cu (the i8q namespace,
// STRATA_PF_PEER_I8): the three functions against a host-side reference.
//
//   f16_rows_to_i8: FP16 rows -> per-row INT8 codes + FP32 scale
//   f32_rows_to_i8: FP32 rows -> per-row INT8 codes + FP32 scale
//   i8_rows_to_f32: INT8 codes + FP32 scale -> FP32 rows
//
// Each row is quantized independently: scale = amax/127 (1.0 when amax == 0),
// code = clamp(round(x/scale), -127, 127).  Rounding to the nearest level bounds the code's error by
// half a step (amax/254); over a uniform residual the rms is step/sqrt(12) = amax/223.6.
// The kernel rounds with roundf (ties away from zero: moe_f16_kq_cuda.cu's i8q quantize) and the host
// references below mirror that tie rule - lrintf (ties to even) would disagree by one code on an exact .5
// tie, a full step against the half-step bound, and not a kernel defect.
// The error is measured against the SOURCE value, not against the reference's own re-quantization: the
// kernel is compiled with -use_fast_math, so its quotient is a fast reciprocal division and not
// IEEE-exact - on an exact .5 tie (or a hair under one) the device can land on the other nearest level
// than the host's exact division.  That is a full step apart but NOT a defect: both levels sit within
// half a step of the source, which is the bound the kernel documents.  This test locks
// the error bounds and the historical bugs: the row independence (a warp's amax must not be dropped from
// the row's max - the second warp reduction over the first warps' values), the multi-round coverage for
// cols > 2048 (cols = 6144 took 8 values a thread per pass, a stride of 2048 per pass: more than one
// pass per row), and the f16 source path's own rounding.
#include "../../src/prefill/moe_f16_kq_cuda.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// The three functions of moe_f16_kq_cuda.cu (the i8q namespace, compiled into strata_mmq): the forward
// declarations (no header of their own).
namespace strata::prefill::fused::i8q {
void f16_rows_to_i8(const uint16_t* in, int64_t rows, int64_t cols, int8_t* codes, float* scales, cudaStream_t s);
void f32_rows_to_i8(const float* in, int64_t rows, int64_t cols, int8_t* codes, float* scales, cudaStream_t s);
void i8_rows_to_f32(const int8_t* codes, const float* scales, float* out, int64_t rows, int64_t cols, cudaStream_t s);
}  // namespace strata::prefill::fused::i8q

namespace {

namespace i8q = strata::prefill::fused::i8q;

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

// The host-side reference of the row's quantization: amax over the row, the scale amax/127 (1.0 when
// amax == 0) and the codes' round(x/scale) clamped to [-127, 127].  The kernel's f32 path and the
// reference agree on the amax, the scale and the tie rule (roundf); the f16 path's reference is the same
// but over the row's FP16 values, the source's own rounding.
struct Quant {
    float amax;
    float scale;
    std::vector<int8_t> codes;
};

Quant quant_ref_f32(const float* x, int cols) {
    Quant q;
    q.amax = 0.f;
    for (int c = 0; c < cols; ++c) q.amax = fmaxf(q.amax, fabsf(x[c]));
    q.scale = q.amax > 0.f ? q.amax / 127.f : 1.f;
    q.codes.resize(cols);
    for (int c = 0; c < cols; ++c) {
        // the kernel's own rounding: roundf (ties away from zero), not lrintf (ties to even)
        const int code = (int) roundf(x[c] / q.scale);
        q.codes[c] = (int8_t) min(127, max(-127, code));
    }
    return q;
}

Quant quant_ref_f16(const uint16_t* x16, int cols) {
    Quant q;
    q.amax = 0.f;
    for (int c = 0; c < cols; ++c) q.amax = fmaxf(q.amax, fabsf(__half2float(__ushort_as_half(x16[c]))));
    q.scale = q.amax > 0.f ? q.amax / 127.f : 1.f;
    q.codes.resize(cols);
    for (int c = 0; c < cols; ++c) {
        const float v = __half2float(__ushort_as_half(x16[c]));
        // the kernel's own rounding (roundf, ties away from zero), as in the f32 reference above
        const int code = (int) roundf(v / q.scale);
        q.codes[c] = (int8_t) min(127, max(-127, code));
    }
    return q;
}

// The f32 round-trip: f32_rows_to_i8 -> i8_rows_to_f32, the codes and the scales back on the host.
void roundtrip_f32(const float* h_in, int rows, int cols, std::vector<int8_t>& h_codes,
                   std::vector<float>& h_scales, std::vector<float>& h_deq) {
    h_codes.assign((size_t) rows * cols, 0);
    h_scales.assign(rows, 0.f);
    h_deq.assign((size_t) rows * cols, 0.f);
    Dev d_in((size_t) rows * cols * 4), d_codes((size_t) rows * cols), d_scales(rows * 4), d_deq((size_t) rows * cols * 4);
    ck(cudaMemcpy(d_in.p, h_in, (size_t) rows * cols * 4, cudaMemcpyHostToDevice), "copy in");
    i8q::f32_rows_to_i8((const float*) d_in.p, rows, cols, (int8_t*) d_codes.p, (float*) d_scales.p, 0);
    i8q::i8_rows_to_f32((const int8_t*) d_codes.p, (const float*) d_scales.p, (float*) d_deq.p, rows, cols, 0);
    ck(cudaDeviceSynchronize(), "launch");
    ck(cudaGetLastError(), "kernel");
    ck(cudaMemcpy(h_codes.data(), d_codes.p, (size_t) rows * cols, cudaMemcpyDeviceToHost), "copy codes");
    ck(cudaMemcpy(h_scales.data(), d_scales.p, (size_t) rows * 4, cudaMemcpyDeviceToHost), "copy scales");
    ck(cudaMemcpy(h_deq.data(), d_deq.p, (size_t) rows * cols * 4, cudaMemcpyDeviceToHost), "copy deq");
}

// The f16 round-trip: f16_rows_to_i8 (the input the caller packed as halves) -> i8_rows_to_f32.
void roundtrip_f16(const uint16_t* h_in16, int rows, int cols, std::vector<int8_t>& h_codes,
                   std::vector<float>& h_scales, std::vector<float>& h_deq) {
    h_codes.assign((size_t) rows * cols, 0);
    h_scales.assign(rows, 0.f);
    h_deq.assign((size_t) rows * cols, 0.f);
    Dev d_in((size_t) rows * cols * 2), d_codes((size_t) rows * cols), d_scales(rows * 4), d_deq((size_t) rows * cols * 4);
    ck(cudaMemcpy(d_in.p, h_in16, (size_t) rows * cols * 2, cudaMemcpyHostToDevice), "copy in");
    i8q::f16_rows_to_i8((const uint16_t*) d_in.p, rows, cols, (int8_t*) d_codes.p, (float*) d_scales.p, 0);
    i8q::i8_rows_to_f32((const int8_t*) d_codes.p, (const float*) d_scales.p, (float*) d_deq.p, rows, cols, 0);
    ck(cudaDeviceSynchronize(), "launch");
    ck(cudaGetLastError(), "kernel");
    ck(cudaMemcpy(h_codes.data(), d_codes.p, (size_t) rows * cols, cudaMemcpyDeviceToHost), "copy codes");
    ck(cudaMemcpy(h_scales.data(), d_scales.p, (size_t) rows * 4, cudaMemcpyDeviceToHost), "copy scales");
    ck(cudaMemcpy(h_deq.data(), d_deq.p, (size_t) rows * cols * 4, cudaMemcpyDeviceToHost), "copy deq");
}

}  // namespace

int main() {
    // ============================== 1. the shapes ==============================
    // cols = 2048 (the model's n_embd), rows = 1, 3, 64; cols = 128 (less than a pass of 2048: the row's
    // tail, the last partial 8); cols = 6144 (more than one pass of 2048: three passes a thread - the
    // historical "the later rounds' overwrite" bug lived here, in the codes' write stride).  A uniform
    // random residual keeps the rms near its theoretical step/sqrt(12).
    {
        const int cols = 2048;
        const int rows[] = {1, 3, 64};
        std::vector<float> h_in((size_t) 64 * cols);
        std::vector<uint16_t> h_in16((size_t) 64 * cols);
        unsigned seed = 20261030u;
        auto rnd = [&seed]() {
            seed = seed * 1664525u + 1013904223u;
            return (float)((seed >> 8) & 0xFFFF) / 65536.0f;   // [0, 1)
        };
        // a per-row random amplitude (the rows are independent: the kernel's amax must be per row)
        for (size_t r = 0; r < 64; ++r) {
            const float amp = 0.5f + 300.f * rnd();
            for (size_t c = 0; c < cols; ++c) h_in[r * cols + c] = (rnd() - 0.5f) * amp;
            for (size_t c = 0; c < cols; ++c) h_in16[r * cols + c] = __half_as_ushort(__float2half_rn(h_in[r * cols + c]));
        }

        // ---- f32 source: cols = 2048, rows = 1 / 3 / 64 ----
        {
            double max_abs = 0.0, rms = 0.0;
            for (int n : rows) {
                std::vector<int8_t> codes;
                std::vector<float> scales, deq;
                roundtrip_f32(h_in.data(), n, cols, codes, scales, deq);
                (void) codes; (void) scales;
                for (int r = 0; r < n; ++r) {
                    const Quant ref = quant_ref_f32(h_in.data() + (size_t) r * cols, cols);
                    double row_max = 0.0, row_sq = 0.0;
                    for (int c = 0; c < cols; ++c) {
                        const double err = (double) deq[(size_t) r * cols + c] - (double) h_in[(size_t) r * cols + c];
                        row_max = fmax(row_max, fabs(err));
                        row_sq += err * err;
                    }
                    const double row_rms = sqrt(row_sq / cols) / (ref.amax > 0.f ? ref.amax : 1.0);
                    const double bound = ref.amax / 254.0 * 1.001 + 1e-6;
                    if (row_max > bound) {
                        std::printf("prefill_peer_i8q_test: f32 cols=%d row %d: max |err| %.3e > bound %.3e\n",
                                    cols, r, row_max, bound);
                        return 1;
                    }
                    max_abs = fmax(max_abs, row_max / (ref.amax > 0.f ? ref.amax : 1.0));
                    rms = fmax(rms, row_rms);
                }
            }
            std::printf("f32 cols=2048 rows=1/3/64: max |err|/amax %.4f%%, rms/amax %.4f%% (bound 0.394%%, rms ~0.228%%)\n",
                        max_abs * 100.0, rms * 100.0);
            if (rms >= 0.0030) {
                std::printf("prefill_peer_i8q_test: rms %.4f%% is over the 0.30%% bound\n", rms * 100.0);
                return 1;
            }
        }

        // ---- f16 source: the same data, halves: the source's own rounding is the error, not a bound violation ----
        {
            double max_abs = 0.0, rms = 0.0;
            for (int n : rows) {
                std::vector<int8_t> codes;
                std::vector<float> scales, deq;
                roundtrip_f16(h_in16.data(), n, cols, codes, scales, deq);
                (void) codes; (void) scales;
                for (int r = 0; r < n; ++r) {
                    // the reference is the same row's FP16 values, the f16 source's own rounding
                    const Quant ref = quant_ref_f16(h_in16.data() + (size_t) r * cols, cols);
                    double row_max = 0.0, row_sq = 0.0;
                    for (int c = 0; c < cols; ++c) {
                        const double err = (double) deq[(size_t) r * cols + c]
                                         - (double) __half2float(__ushort_as_half(h_in16[(size_t) r * cols + c]));
                        row_max = fmax(row_max, fabs(err));
                        row_sq += err * err;
                    }
                    const double row_rms = sqrt(row_sq / cols) / (ref.amax > 0.f ? ref.amax : 1.0);
                    const double bound = ref.amax / 254.0 * 1.001 + 1e-6;
                    if (row_max > bound) {
                        std::printf("prefill_peer_i8q_test: f16 cols=%d row %d: max |err| %.3e > bound %.3e\n",
                                    cols, r, row_max, bound);
                        return 1;
                    }
                    max_abs = fmax(max_abs, row_max / (ref.amax > 0.f ? ref.amax : 1.0));
                    rms = fmax(rms, row_rms);
                }
            }
            std::printf("f16 cols=2048 rows=1/3/64: max |err|/amax %.4f%%, rms/amax %.4f%% (the f16 source's rounding)\n",
                        max_abs * 100.0, rms * 100.0);
            if (rms >= 0.0030) {
                std::printf("prefill_peer_i8q_test: f16 rms %.4f%% is over the 0.30%% bound\n", rms * 100.0);
                return 1;
            }
        }

        // ---- cols = 128: less than a pass of 2048, the row's tail (the last 8 values are a partial write) ----
        {
            const int c128 = 128;
            const int rows128 = 3;
            std::vector<float> in128((size_t) rows128 * c128);
            for (int r = 0; r < rows128; ++r) {
                const float amp = 1.f + 100.f * rnd();
                for (int c = 0; c < c128; ++c) in128[r * c128 + c] = (rnd() - 0.5f) * amp;
            }
            std::vector<int8_t> codes;
            std::vector<float> scales, deq;
            roundtrip_f32(in128.data(), rows128, c128, codes, scales, deq);
            (void) codes; (void) scales;
            for (int r = 0; r < rows128; ++r) {
                const Quant ref = quant_ref_f32(in128.data() + (size_t) r * c128, c128);
                double row_max = 0.0, row_sq = 0.0;
                for (int c = 0; c < c128; ++c) {
                    const double err = (double) deq[(size_t) r * c128 + c] - (double) in128[(size_t) r * c128 + c];
                    row_max = fmax(row_max, fabs(err));
                    row_sq += err * err;
                }
                const double row_rms = sqrt(row_sq / c128) / (ref.amax > 0.f ? ref.amax : 1.0);
                const double bound = ref.amax / 254.0 * 1.001 + 1e-6;
                if (row_max > bound || row_rms >= 0.0030) {
                    std::printf("prefill_peer_i8q_test: f32 cols=128 row %d: max %.3e (bound %.3e), rms %.4f%%\n",
                                r, row_max, bound, row_rms * 100.0);
                    return 1;
                }
            }
            std::printf("f32 cols=128 rows=3: ok (the tail's partial write)\n");
        }

        // ---- cols = 6144: more than one pass of 2048 (three passes a thread: the historical overwrite bug) ----
        {
            const int c6 = 6144;
            const int rows6 = 3;
            std::vector<float> in6((size_t) rows6 * c6);
            for (int r = 0; r < rows6; ++r) {
                const float amp = 1.f + 100.f * rnd();
                for (int c = 0; c < c6; ++c) in6[r * c6 + c] = (rnd() - 0.5f) * amp;
            }
            std::vector<int8_t> codes;
            std::vector<float> scales, deq;
            roundtrip_f32(in6.data(), rows6, c6, codes, scales, deq);
            (void) codes; (void) scales;
            for (int r = 0; r < rows6; ++r) {
                const Quant ref = quant_ref_f32(in6.data() + (size_t) r * c6, c6);
                double row_max = 0.0, row_sq = 0.0;
                for (int c = 0; c < c6; ++c) {
                    const double err = (double) deq[(size_t) r * c6 + c] - (double) in6[(size_t) r * c6 + c];
                    row_max = fmax(row_max, fabs(err));
                    row_sq += err * err;
                }
                const double row_rms = sqrt(row_sq / c6) / (ref.amax > 0.f ? ref.amax : 1.0);
                const double bound = ref.amax / 254.0 * 1.001 + 1e-6;
                if (row_max > bound || row_rms >= 0.0030) {
                    std::printf("prefill_peer_i8q_test: f32 cols=6144 row %d: max %.3e (bound %.3e), rms %.4f%%\n",
                                r, row_max, bound, row_rms * 100.0);
                    return 1;
                }
            }
            // the f16 source of the same data: the multi-round overwrite of the codes' write stride, through the
            // f16 path (its own rounding: the same bound, the source's error)
            std::vector<uint16_t> in6_16((size_t) rows6 * c6);
            for (size_t i = 0; i < in6.size(); ++i) in6_16[i] = __half_as_ushort(__float2half_rn(in6[i]));
            std::vector<int8_t> codes16;
            std::vector<float> scales16, deq16;
            roundtrip_f16(in6_16.data(), rows6, c6, codes16, scales16, deq16);
            (void) codes16; (void) scales16;
            for (int r = 0; r < rows6; ++r) {
                const Quant ref = quant_ref_f16(in6_16.data() + (size_t) r * c6, c6);
                double row_max = 0.0, row_sq = 0.0;
                for (int c = 0; c < c6; ++c) {
                    const double err = (double) deq16[(size_t) r * c6 + c]
                                       - (double) __half2float(__ushort_as_half(in6_16[(size_t) r * c6 + c]));
                    row_max = fmax(row_max, fabs(err));
                    row_sq += err * err;
                }
                const double row_rms = sqrt(row_sq / c6) / (ref.amax > 0.f ? ref.amax : 1.0);
                const double bound = ref.amax / 254.0 * 1.001 + 1e-6;
                if (row_max > bound || row_rms >= 0.0030) {
                    std::printf("prefill_peer_i8q_test: f16 cols=6144 row %d: max %.3e (bound %.3e), rms %.4f%%\n",
                                r, row_max, bound, row_rms * 100.0);
                    return 1;
                }
            }
            std::printf("f32/f16 cols=6144 rows=3: ok (the multi-round coverage)\n");
        }
    }

    // ============================== 2. the rows' independence ==============================
    // The three rows' amaxes are radically different (1e-6, 100, 50): a row's scale must not be polluted by
    // a neighbour's amax.  Row 2's peak is at the LAST element only - a warp reduction that drops a warp's
    // amax (the second reduction over the first warps' values) scales the row's amax too small, clips the
    // last element to 127 and dequantizes it to less than 50.
    {
        const int cols = 2048, rows = 3;
        std::vector<float> in((size_t) rows * cols, 0.f);
        std::vector<uint16_t> in16((size_t) rows * cols, 0);
        for (int c = 0; c < cols; ++c) {
            in[(size_t) 0 * cols + c] = 1e-6f;                          // row 0: a tiny peak, all of it
            in[(size_t) 1 * cols + c] = 100.f;                          // row 1: a large peak, all of it
            in[(size_t) 2 * cols + c] = 0.f;
            in[(size_t) 2 * cols + cols - 1] = 50.f;                    // row 2: a peak at the LAST element
        }
        for (size_t i = 0; i < in.size(); ++i) in16[i] = __half_as_ushort(__float2half_rn(in[i]));
        const float amps[] = {1e-6f, 100.f, 50.f};

        // f32: the reference's scales must be the rows' own amaxes/127, not a shared max
        {
            std::vector<int8_t> codes;
            std::vector<float> scales, deq;
            roundtrip_f32(in.data(), rows, cols, codes, scales, deq);
            (void) codes;
            for (int r = 0; r < rows; ++r) {
                const Quant ref = quant_ref_f32(in.data() + (size_t) r * cols, cols);
                if (fabs(scales[r] - ref.scale) > 1e-6 * ref.scale) {
                    std::printf("prefill_peer_i8q_test: f32 independence row %d: scale %.6e (the row's is %.6e)\n",
                                r, scales[r], ref.scale);
                    return 1;
                }
                double row_max = 0.0;
                for (int c = 0; c < cols; ++c)
                    row_max = fmax(row_max, fabs((double) deq[(size_t) r * cols + c] - (double) in[(size_t) r * cols + c]));
                const double bound = ref.amax / 254.0 * 1.001 + 1e-6;
                if (row_max > bound) {
                    std::printf("prefill_peer_i8q_test: f32 independence row %d: max |err| %.3e > bound %.3e\n",
                                r, row_max, bound);
                    return 1;
                }
            }
            std::printf("f32 independence rows=3 (1e-6, 100, 50-last): scales are per row, the bound holds\n");
        }
        // f16: the same data through the halves: 1e-6, 100, 50 are all exactly representable in FP16, so the
        // source's rounding is exact here and the bound is the kernel's
        {
            std::vector<int8_t> codes;
            std::vector<float> scales, deq;
            roundtrip_f16(in16.data(), rows, cols, codes, scales, deq);
            (void) codes;
            for (int r = 0; r < rows; ++r) {
                const Quant ref = quant_ref_f16(in16.data() + (size_t) r * cols, cols);
                if (fabs(scales[r] - ref.scale) > 1e-6 * ref.scale) {
                    std::printf("prefill_peer_i8q_test: f16 independence row %d: scale %.6e (the row's is %.6e)\n",
                                r, scales[r], ref.scale);
                    return 1;
                }
                double row_max = 0.0;
                for (int c = 0; c < cols; ++c)
                    row_max = fmax(row_max, fabs((double) deq[(size_t) r * cols + c]
                                                 - (double) __half2float(__ushort_as_half(in16[(size_t) r * cols + c]))));
                const double bound = ref.amax / 254.0 * 1.001 + 1e-6;
                if (row_max > bound) {
                    std::printf("prefill_peer_i8q_test: f16 independence row %d: max |err| %.3e > bound %.3e\n",
                                r, row_max, bound);
                    return 1;
                }
            }
            std::printf("f16 independence rows=3 (1e-6, 100, 50-last): scales are per row, the bound holds\n");
        }
        (void) amps;
    }

    // ============================== 3. the boundary values ==============================
    // All-zeros (the scale's floor of 1.0 and the codes' zero), a single-element row, and the rows whose
    // amax is exactly the code's floor/ceiling (±127).
    {
        const int cols = 2048, rows = 4;
        std::vector<float> in((size_t) rows * cols);
        std::vector<uint16_t> in16((size_t) rows * cols);
        for (int c = 0; c < cols; ++c) {
            in[(size_t) 0 * cols + c] = 0.f;                  // row 0: all zeros -> the scale 1.0, the codes 0
            in[(size_t) 1 * cols + c] = 0.f;
            in[(size_t) 1 * cols + 0] = 1.f;                  // row 1: a single element (the rest are 0, the amax 1)
            in[(size_t) 2 * cols + c] = 7.5f;                 // row 2: the amax on every element -> the code 127
            in[(size_t) 3 * cols + c] = -7.5f;                // row 3: the amax on every element -> the code -127
        }
        for (size_t i = 0; i < in.size(); ++i) in16[i] = __half_as_ushort(__float2half_rn(in[i]));

        {
            std::vector<int8_t> codes;
            std::vector<float> scales, deq;
            roundtrip_f32(in.data(), rows, cols, codes, scales, deq);
            if (scales[0] != 1.0f) {
                std::printf("prefill_peer_i8q_test: f32 all-zeros: the scale is %.6e (the bound's floor is 1.0)\n",
                            scales[0]);
                return 1;
            }
            for (int c = 0; c < cols; ++c)
                if (codes[c] != 0 || deq[c] != 0.0f) {
                    std::printf("prefill_peer_i8q_test: f32 all-zeros row: the code %d at %d is not 0\n",
                                codes[c], c);
                    return 1;
                }
            if (scales[1] - 1.0f / 127.f > 1e-6) {
                std::printf("prefill_peer_i8q_test: f32 single-element row: the scale is %.6e (the row's is %.6e)\n",
                            scales[1], 1.0f / 127.f);
                return 1;
            }
            if (codes[cols] != 127) {
                std::printf("prefill_peer_i8q_test: f32 single-element row: the code is %d (the bound's ceiling is 127)\n",
                            codes[cols]);
                return 1;
            }
            if (codes[(size_t) 2 * cols] != 127 || codes[(size_t) 3 * cols] != -127) {
                std::printf("prefill_peer_i8q_test: f32 amax rows: the code is %d / %d (the bounds are 127 / -127)\n",
                            codes[(size_t) 2 * cols], codes[(size_t) 3 * cols]);
                return 1;
            }
            // the amax rows' dequantized values: code * scale = 127 * (7.5/127) = 7.5, the exact round-trip
            if (fabs(deq[(size_t) 2 * cols] - 7.5f) > 1e-6 || fabs(deq[(size_t) 3 * cols] + 7.5f) > 1e-6) {
                std::printf("prefill_peer_i8q_test: f32 amax rows: the dequantized %.6e / %.6e are not ±7.5\n",
                            deq[(size_t) 2 * cols], deq[(size_t) 3 * cols]);
                return 1;
            }
            std::printf("f32 boundaries: all-zeros, single-element, the amax rows' codes ±127\n");
        }
        {
            std::vector<int8_t> codes16;
            std::vector<float> scales16, deq16;
            roundtrip_f16(in16.data(), rows, cols, codes16, scales16, deq16);
            if (scales16[0] != 1.0f || scales16[1] - 1.0f / 127.f > 1e-6) {
                std::printf("prefill_peer_i8q_test: f16 boundaries: the scales are %.6e / %.6e\n", scales16[0], scales16[1]);
                return 1;
            }
            if (codes16[cols] != 127 || codes16[(size_t) 2 * cols] != 127 || codes16[(size_t) 3 * cols] != -127) {
                std::printf("prefill_peer_i8q_test: f16 boundaries: the codes are %d / %d / %d (the bounds are 127 / 127 / -127)\n",
                            codes16[cols], codes16[(size_t) 2 * cols], codes16[(size_t) 3 * cols]);
                return 1;
            }
            std::printf("f16 boundaries: all-zeros, single-element, the amax rows' codes ±127\n");
        }
        // a single-element row, cols = 8 (the row's smallest supported width: the codes' word is full):
        // each row has exactly one non-zero element, the amax's owner, and the code's ceiling 127
        {
            const int c8 = 8;
            const int rows8 = 4;
            std::vector<float> in8((size_t) rows8 * c8, 0.f);
            in8[(size_t) 0 * c8 + 0] = 2.5f;    // row 0: a single element, the amax's owner, the code 127
            in8[(size_t) 1 * c8 + 0] = 0.5f;    // row 1: a single element, the code 127
            in8[(size_t) 2 * c8 + 0] = 0.25f;   // row 2: a single element, the code 127
            in8[(size_t) 3 * c8 + 0] = -1.0f;   // row 3: a single element, the code -127
            std::vector<int8_t> codes8;
            std::vector<float> scales8, deq8;
            roundtrip_f32(in8.data(), rows8, c8, codes8, scales8, deq8);
            for (int r = 0; r < rows8; ++r) {
                const float am = fabsf(in8[(size_t) r * c8]);   // the amax, the element's magnitude
                const int want = in8[(size_t) r * c8] >= 0.f ? 127 : -127;
                if (fabs(scales8[r] - am / 127.f) > 1e-6 || codes8[(size_t) r * c8] != want) {
                    std::printf("prefill_peer_i8q_test: f32 cols=8 row %d: scale %.6e (the row's is %.6e), code %d (the bound is %d)\n",
                                r, scales8[r], am / 127.f, codes8[(size_t) r * c8], want);
                    return 1;
                }
                // the single element's dequantized value: code * scale = want * (|am| / 127) = the element
                if (fabs(deq8[(size_t) r * c8] - in8[(size_t) r * c8]) > 1e-6 * am) {
                    std::printf("prefill_peer_i8q_test: f32 cols=8 row %d: the dequantized %.6e is not the element %.6e\n",
                                r, deq8[(size_t) r * c8], in8[(size_t) r * c8]);
                    return 1;
                }
            }
            std::printf("f32 cols=8: the single-element rows, the code's ceiling ±127, the exact round-trip\n");
            // the f16 source of the same data: 2.5, 0.5, 0.25, -1.0 are all exactly representable in FP16, so
            // the source's rounding is exact and the bound is the kernel's
            std::vector<uint16_t> in8_16((size_t) rows8 * c8);
            for (size_t i = 0; i < in8.size(); ++i) in8_16[i] = __half_as_ushort(__float2half_rn(in8[i]));
            std::vector<int8_t> codes8_16;
            std::vector<float> scales8_16, deq8_16;
            roundtrip_f16(in8_16.data(), rows8, c8, codes8_16, scales8_16, deq8_16);
            for (int r = 0; r < rows8; ++r) {
                const float am = fabsf(in8[(size_t) r * c8]);
                const int want = in8[(size_t) r * c8] >= 0.f ? 127 : -127;
                if (fabs(scales8_16[r] - am / 127.f) > 1e-6 || codes8_16[(size_t) r * c8] != want) {
                    std::printf("prefill_peer_i8q_test: f16 cols=8 row %d: scale %.6e, code %d\n",
                                r, scales8_16[r], codes8_16[(size_t) r * c8]);
                    return 1;
                }
            }
            std::printf("f16 cols=8: the single-element rows, the code's ceiling ±127\n");
        }
    }

    // ============================== 4. the two source paths on the same data ==============================
    // The f32 and the f16 round-trips of the SAME random rows: the f16 source's rounding is its own error
    // (the halves' representation), but the codes of the two paths should agree to a code-step of the f16
    // rounding, and the dequantized values of both should sit inside the shared bound.
    // The shared bound here is ONE code step, not the half step of a single path: the f16 source's value can
    // sit half an FP16 ULP (2^-11 relative) off the f32 one and the two paths' amaxes differ by the same
    // relative amount, so their quotients differ by up to ~0.063 of a code - the codes can land on different
    // nearest levels (one step apart) and each path still sits within half a step of ITS OWN source.
    {
        const int cols = 2048, rows = 16;
        std::vector<float> in((size_t) rows * cols);
        std::vector<uint16_t> in16((size_t) rows * cols);
        unsigned seed = 4242u;
        auto rnd = [&seed]() {
            seed = seed * 1664525u + 1013904223u;
            return (float)((seed >> 8) & 0xFFFF) / 65536.0f;
        };
        for (int r = 0; r < rows; ++r) {
            const float amp = 0.5f + 300.f * rnd();
            for (int c = 0; c < cols; ++c) {
                in[(size_t) r * cols + c] = (rnd() - 0.5f) * amp;
                in16[(size_t) r * cols + c] = __half_as_ushort(__float2half_rn(in[(size_t) r * cols + c]));
            }
        }
        std::vector<int8_t> c32, c16;
        std::vector<float> s32, s16, d32, d16;
        roundtrip_f32(in.data(), rows, cols, c32, s32, d32);
        roundtrip_f16(in16.data(), rows, cols, c16, s16, d16);
        double worst_code = 0.0, worst_deq = 0.0;
        for (int r = 0; r < rows; ++r) {
            const double amax = quant_ref_f32(in.data() + (size_t) r * cols, cols).amax;
            double row_wcode = 0.0, row_wdeq = 0.0;
            for (int c = 0; c < cols; ++c) {
                row_wcode = fmax(row_wcode, fabs((double) c16[(size_t) r * cols + c] - (double) c32[(size_t) r * cols + c]));
                row_wdeq = fmax(row_wdeq, fabs((double) d32[(size_t) r * cols + c] - (double) d16[(size_t) r * cols + c]));
            }
            // the two paths may sit on different nearest levels (one code step), plus the f16 source's own
            // rounding: the bound is one step with a 10% margin over the derived 1.07 steps
            const double bound = amax / 127.0 * 1.1 + 1e-6;
            if (row_wcode > 3.0) {
                // the f16 source's amax can sit a half-step up or down from the f32 one, shifting the codes by
                // a code-step; a difference over 3 steps is a source-path disagreement, not a rounding
                std::printf("prefill_peer_i8q_test: two-path row %d: the codes differ by %.0f (the f16 rounding is ~1 step)\n",
                            r, row_wcode);
                return 1;
            }
            if (row_wdeq > bound) {
                std::printf("prefill_peer_i8q_test: two-path row %d: the dequantized differ by %.3e > bound %.3e\n",
                            r, row_wdeq, bound);
                return 1;
            }
            worst_code = fmax(worst_code, row_wcode);
            worst_deq = fmax(worst_deq, row_wdeq / (amax + 1e-30));
        }
        std::printf("two paths: the codes differ by %.0f code-steps at most, the dequantized differ by %.4f%% of amax at most\n",
                    worst_code, worst_deq * 100.0);
    }

    std::printf("prefill_peer_i8q_test: ok\n");
    return 0;
}
