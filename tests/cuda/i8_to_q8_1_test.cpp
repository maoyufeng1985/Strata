// i8_to_q8_1_test - the fused i8 -> q8_1 activation quantize (moe_mmq.cu's i8q::i8_rows_to_q8_1, the
// STRATA_PF_PEER_I8 helper route) against the two-step route it replaces (i8q::i8_rows_to_f32 to FP32,
// then mmq::quantize to q8_1): the SAME q8_1 bytes, whole buffer, for every q8_1 layout MMQ uses
// (D4: Q2_0 Q6_K, DS4: Q4_K Q5_K, D2S6: Q2_K), with a shuffled compact-row -> source-INT8-row map and
// chunk widths that cross the kernel's tile (the 128-value q8_1 block and the 512-value y-chunk).
//
// The invariant: the two-step route's FP32 value is (float)code * scale, exactly what the fused kernel
// computes in-register, so every per-32-subblock scale / sum / code is the same FP32 expression as
// llama.cpp's quantize_mmq_q8_1 - the buffers must agree byte for byte, not merely within a tolerance.
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// The i8q transport kernels (moe_f16_kq_cuda.cu's i8_rows_to_f32, moe_mmq.cu's i8_rows_to_q8_1, both in
// strata_mmq): the forward declarations (they have no header of their own, as in prefill_peer_i8q_test).
namespace strata::prefill::fused::i8q {
void i8_rows_to_f32(const int8_t* codes, const float* scales, float* out, int64_t rows, int64_t cols, cudaStream_t s);
void i8_rows_to_q8_1(const int8_t* codes, const float* scales, const int32_t* ids, void* y, int cols, int n_rows,
                     int type, cudaStream_t s);
}

namespace {
namespace mmq = strata::prefill::mmq;
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

int check(ggml_type t, int64_t rows, int64_t cols, uint32_t seed) {
    std::mt19937 rng(seed);
    // a per-row scale (the amax/127 grid) and the row's INT8 codes, a wide mix of magnitudes
    std::vector<float> h_scales((size_t) rows);
    std::vector<int8_t> h_codes((size_t) rows * cols);
    for (int64_t r = 0; r < rows; ++r) {
        const float amp = 0.5f + 400.f * ((float)((seed ^ (uint32_t) r) * 2654435761u) / 4294967295.0f);
        h_scales[(size_t) r] = amp / 127.0f;
        for (int64_t c = 0; c < cols; ++c)
            h_codes[(size_t) r * cols + (size_t) c] = (int8_t) (std::uniform_int_distribution<int>(-127, 127)(rng));
    }
    // the compact-row -> source-INT8-row map: a shuffled permutation (the row order is not the identity)
    std::vector<int32_t> h_ids((size_t) rows);
    for (int64_t r = 0; r < rows; ++r) h_ids[(size_t) r] = (int32_t) r;
    std::shuffle(h_ids.begin(), h_ids.end(), rng);

    const size_t qb = mmq::q8_bytes(rows, cols);
    Dev d_codes((size_t) rows * cols), d_scales((size_t) rows * 4), d_fp32((size_t) rows * cols * 4),
        d_ids((size_t) rows * 4), d_ref(qb), d_fused(qb);
    ck(cudaMemcpy(d_codes.p, h_codes.data(), h_codes.size(), cudaMemcpyHostToDevice), "codes");
    ck(cudaMemcpy(d_scales.p, h_scales.data(), h_scales.size() * 4, cudaMemcpyHostToDevice), "scales");
    ck(cudaMemcpy(d_ids.p, h_ids.data(), h_ids.size() * 4, cudaMemcpyHostToDevice), "ids");
    // both buffers start at 0xA5, so the q8_bytes' trailing pad (never written) agrees on its own
    ck(cudaMemset(d_ref.p, 0xA5, qb), "memset ref");
    ck(cudaMemset(d_fused.p, 0xA5, qb), "memset fused");

    // the two-step route: i8 -> FP32, then the FP32 -> q8_1 the MMQ does today
    i8q::i8_rows_to_f32((const int8_t*) d_codes.p, (const float*) d_scales.p, (float*) d_fp32.p, rows, cols, nullptr);
    mmq::quantize((const float*) d_fp32.p, (const int32_t*) d_ids.p, d_ref.p, (int) t, cols, cols, rows, nullptr);
    // the fused route: i8 -> q8_1 in one pass
    i8q::i8_rows_to_q8_1((const int8_t*) d_codes.p, (const float*) d_scales.p, (const int32_t*) d_ids.p, d_fused.p,
                         (int) cols, (int) rows, (int) t, nullptr);
    ck(cudaDeviceSynchronize(), "launch");

    std::vector<uint8_t> a(qb), b(qb);
    ck(cudaMemcpy(a.data(), d_ref.p, qb, cudaMemcpyDeviceToHost), "back ref");
    ck(cudaMemcpy(b.data(), d_fused.p, qb, cudaMemcpyDeviceToHost), "back fused");
    size_t diff = 0, first = 0;
    for (size_t i = 0; i < qb; ++i)
        if (a[i] != b[i]) {
            if (diff == 0) first = i;
            ++diff;
        }
    std::printf("%s %-7s rows=%-5lld cols=%-5lld %zu of %zu bytes differ%s\n", diff ? "FAIL" : "ok  ",
                ggml_type_name(t), (long long) rows, (long long) cols, diff, qb, diff ? "" : "");
    if (diff > 0) std::printf("     first difference at byte %zu (ref 0x%02X, fused 0x%02X)\n", first, a[first], b[first]);
    return diff ? 1 : 0;
}
}  // namespace

int main() {
    int fails = 0;
    try {
        // D4 (Q2_0, Q6_K), DS4 (Q4_K, Q5_K - the Whittle gate/up), D2S6 (Q2_K): every layout MMQ uses.
        const ggml_type types[] = {GGML_TYPE_Q2_0, GGML_TYPE_Q6_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q2_K};
        // the widths: 1024 / 2048 / 6144 are the model's n_embd and the multi-tile width (6144 spans 12 y-chunks
        // of 512 and 48 q8_1 blocks a row).  384 and 1280 are not 512-multiples: the MMQ's K-tile runs to
        // pad512, so the y-padded tail is zero-filled (the pad must agree byte for byte, not just the real part).
        const int64_t cols[] = {384, 1024, 1280, 2048, 6144};
        uint32_t seed = 20261030u;
        for (ggml_type t : types)
            for (int64_t c : cols)
                for (int64_t r : {1, 7, 64, 600}) fails += check(t, r, c, seed++);
    } catch (const std::exception& e) {
        std::printf("FAIL %s\n", e.what());
        return 1;
    }
    std::printf("i8_to_q8_1_test: %d failures\n", fails);
    return fails ? 1 : 0;
}
