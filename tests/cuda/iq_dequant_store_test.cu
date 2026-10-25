// tests/cuda/iq_dequant_store_test.cu - the dequantizer's wide stores.
//
// The engine's dequantizer writes the values of a group in one 8-byte store (iq_kernels.cu's put4)
// instead of one value at a time: the narrow stores of a warp cover a sector only half each time, so
// the same bytes cost several transactions (the Q4_K gate/up: 71 -> 35 us per expert on a P100).
// The wide store is guarded - a destination that is not 8-byte aligned falls back to the single
// stores - so this test checks that both paths give the same values, and that the values are
// ggml's.
//
//   iq_dequant_store_test
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "strata/kernels/iq_kernels.hpp"

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

uint32_t seed = 12345;
uint32_t rnd() { seed = seed * 1664525u + 1013904223u; return (seed >> 8) & 0xFFFFFF; }

// ggml's block_q4_K: {half2 dm; uint8_t scales[12]; uint8_t qs[128];}
struct blk_q4_k { __half2 dm; uint8_t scales[12]; uint8_t qs[128]; };
// ggml's block_q5_1: {half2 dm; uint8_t qh[4]; uint8_t qs[16];} - 24 B for 32 values
struct blk_q5_1 { __half2 dm; uint8_t qh[4]; uint8_t qs[16]; };

// A Q4_K row with the 6-bit scales packed the way ggml packs them.
void make_q4_k(uint8_t* row) {
    for (int i = 0; i < 144; ++i) row[i] = (uint8_t) (rnd() & 0xFF);
    const uint16_t d  = (uint16_t) (((8 + (rnd() % 4)) << 10) | (rnd() & 0x3FF));
    const uint16_t dm = (uint16_t) (((7 + (rnd() % 4)) << 10) | (rnd() & 0x3FF));
    memcpy(row + 0, &d, 2); memcpy(row + 2, &dm, 2);
    uint8_t sc[8], mi[8];
    for (int g = 0; g < 8; ++g) { sc[g] = (uint8_t) (1 + rnd() % 62); mi[g] = (uint8_t) (1 + rnd() % 62); }
    uint8_t q[12] = {0};
    for (int g = 0; g < 4; ++g) { q[g] = sc[g] & 0x3F; q[g + 4] = mi[g] & 0x3F; }
    for (int g = 4; g < 8; ++g) {
        q[g + 4] = (uint8_t) ((sc[g] & 0x0F) | ((mi[g] & 0x0F) << 4));
        q[g - 4] = (uint8_t) ((q[g - 4] & 0x3F) | ((sc[g] >> 4) << 6));
        q[g]     = (uint8_t) ((q[g]     & 0x3F) | ((mi[g] >> 4) << 6));
    }
    for (int t = 0; t < 12; ++t) row[4 + t] = q[t];
}

float half_to_float(uint16_t h) {
    __half x;
    memcpy(&x, &h, 2);
    return __half2float(x);
}

// ggml's dequantize_row_q4_K, in double (256 values)
void ref_q4_k(const uint8_t* blk, double* y) {
    const double d = half_to_float(*(const uint16_t*) (blk + 0));
    const double mn = half_to_float(*(const uint16_t*) (blk + 2));
    const uint8_t* q = blk + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m, sc2, m2;
        if (is < 4) { sc = blk[4 + is] & 63; m = blk[4 + is + 4] & 63; }
        else {
            sc = (uint8_t) ((blk[4+is+4] & 0xF) | ((blk[4+is-4] >> 6) << 4));
            m  = (uint8_t) ((blk[4+is+4] >>  4) | ((blk[4+is-0] >> 6) << 4));
        }
        const double d1 = d * sc, m1 = mn * m;
        if (is + 1 < 4) { sc2 = blk[4 + is + 1] & 63; m2 = blk[4 + is + 5] & 63; }
        else {
            sc2 = (uint8_t) ((blk[4+is+5] & 0xF) | ((blk[4+is-3] >> 6) << 4));
            m2  = (uint8_t) ((blk[4+is+5] >>  4) | ((blk[4+is+1] >> 6) << 4));
        }
        const double d2 = d * sc2, m2v = mn * m2;
        for (int l = 0; l < 32; ++l) y[j + l]    = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) y[j + l+32] = d2 * (q[l]  >> 4)  - m2v;
        q += 32; is += 2;
    }
}

// llama.cpp's dequantize_q5_1: the low nibble of the byte i is the value i, the high nibble the
// value i + 16, and the value is d * code + dmin.
void ref_q5_1(const uint8_t* blk, double* y) {
    const float2 dm = __half22float2(*(const __half2*) blk);
    uint32_t qh;
    memcpy(&qh, blk + 4, 4);
    for (int l = 0; l < 16; ++l) {
        const int xh_0 = (int) (((qh >> (l + 0)) << 4) & 0x10);
        const int xh_1 = (int) ((qh >> (l + 12)) & 0x10);
        y[l]      = (float) (blk[8 + l] & 0x0F | xh_0) * dm.x + dm.y;
        y[l + 16] = (float) (blk[8 + l] >> 4   | xh_1) * dm.x + dm.y;
    }
}

// The value in fp16 steps: the kernel rounds the fp32 product to fp16, the reference is in double,
// and the two may differ by one fp16 step (the kernel's multiply and subtract can be contracted
// into one fma).  Everything above one step is a bug.
double ulp16(double v) {
    const double s = fabs(v) * 0x0009765625;   // 2^-10: fp16's mantissa
    return s < 6.1e-5 ? 6.1e-5 : s;             // the smallest subnormal
}

int failures = 0;

void check(const char* what, bool ok, const char* detail = "") {
    std::printf("%-46s %s%s%s\n", what, ok ? "ok" : "FAIL", ok ? "" : "  ", detail);
    if (!ok) ++failures;
}

}  // namespace

int main() {
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");

    // ---- Q4_K: the gate/up's dequantizer ----
    {
        const int n = 256;                       // one superblock
        std::vector<uint8_t> row(144);
        make_q4_k(row.data());
        void* dBlob = nullptr;
        ck(cudaMalloc(&dBlob, 144), "m");
        ck(cudaMemcpy(dBlob, row.data(), 144, cudaMemcpyHostToDevice), "c");
        // an aligned destination and a separate one that is not 8-byte aligned (the fallback path)
        uint16_t* dOut = nullptr;
        ck(cudaMalloc(&dOut, (size_t) n * 2 + 64), "m");
        uint16_t* dOutOff = nullptr;
        ck(cudaMalloc(&dOutOff, (size_t) n * 2 + 64), "m");
        dOutOff = (uint16_t*) ((uint8_t*) dOutOff + 2);   // 2 bytes off: the wide store must not be used
        strata::kernels::iq_dequant_f16(12, dBlob, n, dOut, s);
        strata::kernels::iq_dequant_f16(12, dBlob, n, dOutOff, s);
        ck(cudaDeviceSynchronize(), "launch");
        std::vector<uint16_t> a(n), b(n);
        ck(cudaMemcpy(a.data(), dOut, (size_t) n * 2, cudaMemcpyDeviceToHost), "c");
        ck(cudaMemcpy(b.data(), dOutOff, (size_t) n * 2, cudaMemcpyDeviceToHost), "c");

        double w[256];
        ref_q4_k(row.data(), w);
        double worst = 0.0;
        for (int i = 0; i < n; ++i) {
            const double got = __half2float(__ushort_as_half(a[i]));
            worst = fmax(worst, fabs(got - w[i]) / ulp16(w[i]));
        }
        char buf[128];
        std::snprintf(buf, sizeof buf, " (%.2f fp16 steps vs ggml)", worst);
        check("Q4_K: the values are ggml's", worst <= 1.0, buf);
        bool same = memcmp(a.data(), b.data(), (size_t) n * 2) == 0;
        check("Q4_K: the aligned and unaligned stores agree", same);
        cudaFree(dBlob); cudaFree(dOut);
    }

    // ---- Q5_1: the down's dequantizer (24 B for 32 values) ----
    {
        const int n = 256;                       // eight blocks
        std::vector<uint8_t> row(24 * 8);
        for (size_t i = 0; i < row.size(); ++i) row[i] = (uint8_t) (rnd() & 0xFF);
        for (size_t i = 0; i < row.size(); i += 24) {
            const uint16_t d  = (uint16_t) (((8 + (rnd() % 4)) << 10) | (rnd() & 0x3FF));
            const uint16_t dm = (uint16_t) (((7 + (rnd() % 4)) << 10) | (rnd() & 0x3FF));
            memcpy(&row[i], &d, 2); memcpy(&row[i + 2], &dm, 2);
        }
        void* dBlob = nullptr;
        ck(cudaMalloc(&dBlob, row.size()), "m");
        ck(cudaMemcpy(dBlob, row.data(), row.size(), cudaMemcpyHostToDevice), "c");
        uint16_t* dOut = nullptr;
        ck(cudaMalloc(&dOut, (size_t) n * 2 + 64), "m");
        uint16_t* dOutOff = nullptr;
        ck(cudaMalloc(&dOutOff, (size_t) n * 2 + 64), "m");
        dOutOff = (uint16_t*) ((uint8_t*) dOutOff + 2);
        strata::kernels::iq_dequant_f16(7, dBlob, n, dOut, s);
        strata::kernels::iq_dequant_f16(7, dBlob, n, dOutOff, s);
        ck(cudaDeviceSynchronize(), "launch");
        std::vector<uint16_t> a(n), b(n);
        ck(cudaMemcpy(a.data(), dOut, (size_t) n * 2, cudaMemcpyDeviceToHost), "c");
        ck(cudaMemcpy(b.data(), dOutOff, (size_t) n * 2, cudaMemcpyDeviceToHost), "c");

        double worst = 0.0;
        for (int b_i = 0; b_i < 8; ++b_i) {
            double w[32];
            ref_q5_1(&row[(size_t) b_i * 24], w);
            for (int i = 0; i < 32; ++i) {
                const double got = __half2float(__ushort_as_half(a[b_i * 32 + i]));
                worst = fmax(worst, fabs(got - w[i]) / ulp16(w[i]));
            }
        }
        char buf[128];
        std::snprintf(buf, sizeof buf, " (%.2f fp16 steps vs llama.cpp)", worst);
        check("Q5_1: the values are llama.cpp's", worst <= 1.0, buf);
        check("Q5_1: the aligned and unaligned stores agree",
              memcmp(a.data(), b.data(), (size_t) n * 2) == 0);
        cudaFree(dBlob); cudaFree(dOut);
    }

    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
