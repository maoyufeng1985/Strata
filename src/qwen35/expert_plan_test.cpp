// The extracted slot arithmetic against the inline version it replaced.  ref_slots() below is q36_engine.cu's
// place_experts() copied verbatim before the extraction, so this is an equivalence check rather than a restatement of
// what the new code happens to do.  Both run over the real numbers the engine meets (12 GB P100, 256 experts) and over
// the boundaries where a division, a clamp or the MTP share can change which branch wins.
#include "strata/qwen35/expert_plan.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace {

int checks = 0;

void check(bool ok, const char* label) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        std::exit(1);
    }
}

using strata::qwen35::layer_slots;
using strata::qwen35::plan_expert_slots;
using strata::qwen35::SlotBudget;
using strata::qwen35::SlotPlan;

/// q36_engine.cu:8956-8976 as it stood before the extraction, with the file's own locals named as inputs.  One
/// deliberate addition: the per_slot guard, which the original did not need because place_experts() already divides by
/// per_slot further up.  Without it this reference would crash on the "no layers" case, which is exactly the case where
/// the new header DOES differ from the original - so that case is documented as a known divergence, not a proof.
int ref_slots(const SlotBudget& in, bool& mtp_all) {
    if (!in.per_slot) { mtp_all = false; return 0; }
    size_t slot_bytes = in.budget > in.staging ? in.budget - in.staging : 0;
    if (slot_bytes + in.staging + ((size_t) 256 << 20) > in.free_bytes)
        slot_bytes = in.free_bytes > in.staging + ((size_t) 512 << 20) ? in.free_bytes - in.staging - ((size_t) 512 << 20) : 0;
    const int64_t ne = in.experts;
    const bool mtp_vram = in.has_mtp && !in.mtp_only && (in.mtp_vram_flag || in.mtp_f16);
    int S = in.override_slots > 0 ? (int) std::min<int64_t>(ne, in.override_slots)
                                  : (int) std::min<size_t>((size_t) ne, slot_bytes / in.per_slot);
    if (mtp_vram) {
        const size_t mtp_bytes = in.mtp_stride * (size_t) ne;
        const size_t main_per_slot = in.per_slot > in.mtp_stride ? in.per_slot - in.mtp_stride : 0;
        const size_t left = slot_bytes > mtp_bytes ? slot_bytes - mtp_bytes : 0;
        S = (int) std::min<size_t>((size_t) ne, main_per_slot ? left / main_per_slot : 0);
    }
    if (in.mtp_only) {
        const size_t left = in.budget > in.total_main ? in.budget - in.total_main : 0;
        S = (int) std::min<size_t>((size_t) ne, left / in.mtp_stride);
    }
    mtp_all = mtp_vram;
    return S;
}

const size_t MiB = (size_t) 1 << 20;
const size_t GiB = (size_t) 1 << 30;

/// One case: the same input through both paths, compared on the count and on the MTP decision.
void same(const SlotBudget& in, const char* label) {
    bool ref_mtp_all = false;
    const int ref = ref_slots(in, ref_mtp_all);
    const SlotPlan got = plan_expert_slots(in);
    check(got.slots == ref, label);
    check(got.mtp_all == ref_mtp_all, label);
    // the layer's count, which is what place_experts() stores: the block whole, or the same as the main layers
    for (int is_mtp = 0; is_mtp < 2; ++is_mtp) {
        const int want = is_mtp && (ref_mtp_all || in.mtp_f16) ? (int) in.experts : ref;
        check(layer_slots(got, is_mtp != 0, in.mtp_f16) == want, label);
    }
    ++checks;
}

SlotBudget base() {
    SlotBudget b;
    b.experts = 256;
    b.budget = (size_t) 9 * GiB;      // the 12 GB P100 after the weights and the context
    b.free_bytes = (size_t) 10 * GiB;
    b.staging = (size_t) 2 * GiB;
    b.per_slot = 40 * (size_t) 768 * 1024;   // 40 layers of a 768 KiB stride
    return b;
}

}  // namespace

int main() {
    // the plain hybrid mode: the budget less the staging, divided by one slot in every layer
    {
        SlotBudget b = base();
        same(b, "plain: budget / per_slot");
        const SlotPlan p = plan_expert_slots(b);
        check(p.slots == (int) (((size_t) 9 * GiB - (size_t) 2 * GiB) / b.per_slot), "plain: the count itself");
        check(p.slots > 0 && p.slots < (int) b.experts, "plain: a real split, not all and not none");
        check(!p.mtp_all, "plain: no MTP block here");
    }
    // --expert-slots wins over the budget, and is clamped at n_expert
    {
        SlotBudget b = base();
        b.override_slots = 8;
        same(b, "override: 8 a layer");
        check(plan_expert_slots(b).slots == 8, "override: 8");
        ++checks;
        b.override_slots = 100000;
        same(b, "override: clamped at n_expert");
        check(plan_expert_slots(b).slots == (int) b.experts, "override: clamped");
        ++checks;
    }
    // the free-VRAM clamp: the budget says one number and the card says a smaller one
    {
        SlotBudget b = base();
        b.free_bytes = (size_t) 4 * GiB;   // the staging plus a little, so the 256 MiB test trips
        same(b, "clamp: free is the tighter");
        SlotBudget t = base();
        t.free_bytes = t.staging + (size_t) 300 * MiB;   // inside the 512 MiB reserve the clamp wants
        same(t, "clamp: too tight to leave anything");
        check(plan_expert_slots(t).slots == 0, "clamp: zero slots");
        ++checks;
    }
    // a slot does not fit at all
    {
        SlotBudget b = base();
        b.budget = b.staging;
        same(b, "edge: no room left after the staging");
        check(plan_expert_slots(b).slots == 0, "edge: zero");
        ++checks;
    }
    // the MTP block whole, out of the same budget (STRATA_Q36_MTP_VRAM=1 or an F16 block)
    for (int f16 = 0; f16 < 2; ++f16) {
        SlotBudget b = base();
        b.has_mtp = true;
        b.mtp_stride = (size_t) 768 * 1024;
        b.mtp_f16 = f16 != 0;
        b.mtp_vram_flag = f16 == 0;
        same(b, f16 ? "mtp: an F16 block stays whole" : "mtp: the flag keeps the block whole");
        const SlotPlan p = plan_expert_slots(b);
        check(p.mtp_all, "mtp: the block is whole");
        check(layer_slots(p, true, b.mtp_f16) == (int) b.experts, "mtp: every expert of the block");
        check(layer_slots(p, false, b.mtp_f16) == p.slots, "mtp: the main layers on the share");
        ++checks;
    }
    // the block's bytes exceed the room: no main slots, and the block still whole
    {
        SlotBudget b = base();
        b.has_mtp = true;
        b.mtp_f16 = true;
        b.mtp_stride = (size_t) 1 << 30;   // 1 GiB a slot: 256 GiB of block, far past the budget
        same(b, "mtp: the block eats the whole room");
        check(plan_expert_slots(b).slots == 0, "mtp: no main slots left");
        ++checks;
    }
    // mtp_only: every main layer whole, the block on what is left
    {
        SlotBudget b = base();
        b.has_mtp = true;
        b.mtp_stride = (size_t) 768 * 1024;
        b.mtp_f16 = true;                    // cleared by mtp_only, but the block is still whole
        b.mtp_vram_flag = true;
        b.mtp_only = true;
        b.total_main = (size_t) 7 * GiB;
        same(b, "mtp_only: the block on what is left");
        const SlotPlan p = plan_expert_slots(b);
        check(!p.mtp_all, "mtp_only: the flag is cleared");
        // NOTE: layer_slots() is not asked about a MAIN layer here.  Under mtp_only the engine gives every main layer
        // ne directly (q36_engine.cu:9027) and only asks this function about the block, so asserting it for a main
        // layer would be testing a contract the engine never uses.
        check(layer_slots(p, true, b.mtp_f16) == (int) b.experts, "mtp_only: the block whole");
        check(p.slots == (int) std::min<size_t>((size_t) b.experts, ((size_t) 9 * GiB - (size_t) 7 * GiB) / b.mtp_stride),
              "mtp_only: the block's share");
        ++checks;
        b.total_main = (size_t) 9 * GiB;      // the main layers take the whole budget
        same(b, "mtp_only: nothing left");
        check(plan_expert_slots(b).slots == 0, "mtp_only: zero");
        ++checks;
        b.total_main = (size_t) 20 * GiB;     // more than the budget, which must not underflow
        same(b, "mtp_only: the main layers alone exceed the budget");
        check(plan_expert_slots(b).slots == 0, "mtp_only: zero, not a wrapped size_t");
        ++checks;
    }
    // a stride at or above the whole slot: per_slot - mtp_stride underflows and must clamp to 0, not wrap
    {
        SlotBudget b = base();
        b.has_mtp = true;
        b.mtp_f16 = true;
        b.mtp_stride = b.per_slot;
        b.free_bytes = b.staging;   // the clamp below zeroes slot_bytes, so this exercises the subtraction, not a division
        same(b, "mtp: the block's stride is the whole slot");
    }
    // n_expert of 0 and an unknown model: no division by a count of zero anywhere
    {
        SlotBudget b = base();
        b.experts = 0;
        b.override_slots = 3;
        same(b, "edge: no experts at all");
        check(plan_expert_slots(b).slots == 0, "edge: zero slots for zero experts");
        ++checks;
    }
    // a model with no layers: per_slot of 0.  KNOWN DIVERGENCE - the original would divide by zero here; the header
    // returns an empty plan instead.  place_experts() cannot reach it (it divides by per_slot at q36_engine.cu:8870,
    // before calling in), so this asserts the guard's own behaviour, not equivalence.
    {
        SlotBudget b = base();
        b.per_slot = 0;
        check(plan_expert_slots(b).slots == 0, "edge: zero slots without layers, no division");
        ++checks;
    }

    // The --expert-slots override is SILENTLY DISCARDED by both MTP paths.  That is the original's behaviour and the
    // most likely thing to get "fixed" by accident, so it is pinned here.
    {
        SlotBudget b = base();
        b.override_slots = 8;
        b.has_mtp = true;
        b.mtp_f16 = true;
        b.mtp_stride = (size_t) 768 * 1024;
        same(b, "override + mtp_all: the block's share wins over --expert-slots");
        check(!plan_expert_slots(b).slots == 0, "override + mtp_all: a real share");
        ++checks;
        SlotBudget c = base();
        c.override_slots = 8;
        c.has_mtp = true;
        c.mtp_f16 = false;      // the engine's mtp_only requires !mtp_f16 (q36_engine.cu:8898)
        c.mtp_stride = (size_t) 768 * 1024;
        c.mtp_only = true;
        c.total_main = (size_t) 7 * GiB;
        same(c, "override + mtp_only: the block's share wins over --expert-slots");
    }
    // the default MTP configuration the engine actually uses: a block, no mtp_only, no override - the block takes the
    // same count as the main layers and mtp_all stays false
    {
        SlotBudget b = base();
        b.has_mtp = true;
        b.mtp_stride = (size_t) 768 * 1024;
        same(b, "mtp: the default, block on the main count");
        const SlotPlan p = plan_expert_slots(b);
        check(!p.mtp_all, "mtp default: mtp_all is false");
        check(layer_slots(p, true, false) == p.slots, "mtp default: the block on the same count");
        ++checks;
    }
    std::printf("expert_plan: %d checks\n", checks);
    return 0;
}
