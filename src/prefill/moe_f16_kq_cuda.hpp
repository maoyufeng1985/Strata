// src/prefill/moe_f16_kq_cuda.hpp - the CUDA side of the fused FP16 K-quant prompt experts.
//
// The same work as moe_f16_kq.hpp (the gfx11 kernels): the gate/up projection of the K-quant
// experts, with the codes dequantized into registers (no FP16 copy of the weights) and the
// int8 activations turned into FP16 in shared memory.
//
// The interface is the gfx11 one (moe_f16_kq.hpp), so a build compiles one of the two files
// (the HIP build moe_f16_kq.cu, the CUDA build moe_f16_kq_cuda.cu) and the call site in
// moe_fused_iq.cu is the same on both.  This header adds the CUDA-only switch.
#pragma once

#include "moe_f16_kq.hpp"

namespace strata::prefill::fused::f16kqc {

/// STRATA_PF_F16_CUDA=1: the K-quant prompt experts take this path.  The default prompt path
/// for these formats is MMQ (int8) and stays the default.
bool requested();

}  // namespace strata::prefill::fused::f16kqc
