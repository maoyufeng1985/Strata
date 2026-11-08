#pragma once
// The q36 hybrid placement's slot arithmetic: how many experts of each layer stay in the primary's VRAM.  Header-only
// and free of CUDA, so src/qwen35/expert_plan_test.cpp can check it on any machine - the same reason tqLLM keeps its
// expert_bank_plan.hpp CUDA-free.  place_experts() in q36_engine.cu computed this inline, where checking it meant
// running a whole engine on a whole card.
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::qwen35 {

/// What place_experts() knows when it decides the slots.  `budget` is the expert budget after the weights; `free_bytes`
/// is the VRAM free at that moment, which can be the tighter of the two when something else has taken some since.
/// `per_slot` is one slot in every layer - the layers' strides summed - and must be positive; `mtp_stride` must be
/// positive whenever `mtp_only`, which place_experts() only sets for a model that has an MTP block.
struct SlotBudget {
    int64_t experts = 0;         // n_expert
    int64_t override_slots = 0;  // --expert-slots; <= 0 means "as many as the budget allows"
    size_t budget = 0;           // budget2 in place_experts()
    size_t free_bytes = 0;       // fr
    size_t staging = 0;          // the prompt's two staging buffers
    size_t per_slot = 0;         // one slot in every layer
    bool has_mtp = false;
    bool mtp_f16 = false;        // an F16 block: the CPU pool has no F16, so its experts cannot stream
    bool mtp_vram_flag = false;  // STRATA_Q36_MTP_VRAM
    size_t mtp_stride = 0;
    bool mtp_only = false;       // every main layer whole, the MTP block what is left
    size_t total_main = 0;       // the main layers' experts in device bytes
};

struct SlotPlan {
    int slots = 0;         // experts resident per layer
    bool mtp_all = false;  // the MTP block keeps every expert, so it never streams
    int64_t experts = 0;   // carried so layer_slots() need not take it back in
};

/// place_experts()'s slot count, in the order it decided them: the room left after the staging, the --expert-slots
/// override over that room, then the MTP block's share out of the same room, and mtp_only as a different placement of
/// the same budget (every main layer whole, the block on what is left) rather than a variant of the first.
inline SlotPlan plan_expert_slots(const SlotBudget& b) {
    const size_t ne = (size_t) std::max<int64_t>(0, b.experts);
    // a model always has layers, so per_slot is never 0 here; a caller that passes one would divide by it
    if (!b.per_slot) return SlotPlan{};
    // the room the slots get: the budget less the staging, but never more than the VRAM actually free - the second
    // clamp is what answers a caller that has already spent some of the budget elsewhere
    size_t slot_bytes = b.budget > b.staging ? b.budget - b.staging : 0;
    if (slot_bytes + b.staging + ((size_t) 256 << 20) > b.free_bytes)
        slot_bytes = b.free_bytes > b.staging + ((size_t) 512 << 20) ? b.free_bytes - b.staging - ((size_t) 512 << 20) : 0;

    SlotPlan p;
    p.experts = b.experts;
    p.slots = b.override_slots > 0
                  ? (int) std::min<int64_t>((int64_t) ne, b.override_slots)
                  : (int) std::min<size_t>(ne, slot_bytes / b.per_slot);
    // the MTP block whole comes out of the same room, and the main layers divide what is left
    p.mtp_all = b.has_mtp && !b.mtp_only && (b.mtp_vram_flag || b.mtp_f16);
    if (p.mtp_all) {
        const size_t mtp_bytes = b.mtp_stride * ne;
        const size_t main_per_slot = b.per_slot > b.mtp_stride ? b.per_slot - b.mtp_stride : 0;
        const size_t left = slot_bytes > mtp_bytes ? slot_bytes - mtp_bytes : 0;
        p.slots = (int) std::min<size_t>(ne, main_per_slot ? left / main_per_slot : 0);
    }
    if (b.mtp_only) {   // the main layers all, the MTP block what is left; mtp_stride is positive here
        const size_t left = b.budget > b.total_main ? b.budget - b.total_main : 0;
        p.slots = (int) std::min<size_t>(ne, left / b.mtp_stride);
    }
    return p;
}

/// How many experts one layer keeps.  `mtp_f16` is passed rather than read off the plan because mtp_only clears
/// mtp_all yet still keeps the block whole - the CPU pool cannot hold F16 either way.
inline int layer_slots(const SlotPlan& p, bool is_mtp, bool mtp_f16) {
    return is_mtp && (p.mtp_all || mtp_f16) ? (int) p.experts : p.slots;
}

}  // namespace strata::qwen35
