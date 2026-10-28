// include/strata/kernels/route_prior.hpp - STRATA_ROUTE_PRIOR=lambda (opt-in, changes the output): cache-conditioned
// routing in verify windows.  See src/kernels/cuda/route_prior.cu.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Top-k per token over `logits` (n_tok x n_expert) where a resident expert (`res_layer[e] >= 0`, the layer's residency
/// table) gets `lambda` added to its logit for the SELECTION; the weights are the router's own probabilities
/// (softmax over the unbiased logits) renormalised over the k picked, as the native router does.
///
/// `n_expert` is the model's expert count and `k` its top-k, so one engine serves every geometry (Flash-Next's
/// 512 / 10 and Whittle's 256 / 8 both do).  `n_expert` must be 512 or less: the kernel keeps the logits in one warp's
/// registers, 16 a lane.
void route_prior_topk(const float* logits, const int32_t* res_layer, float lambda, int32_t* ids, float* weights,
                      int n_tok, int n_expert, int k, void* stream);

}  // namespace strata::kernels
