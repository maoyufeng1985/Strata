#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
class RemoteExperts;
class ExpertSource;

/// STRATA_Q36_HELPER_SUM: weighted per-token reduce on a helper (same arithmetic as RemoteExpertOpt::reduce).
/// `metadata` is row[cap] int32s then weight[cap] floats on the device; `write_mapped` writes the sum
/// straight into the mapped host buffer at `h_out` (no D2H).
bool helper_sum_reduce(void* stream, const float* parts, const void* metadata, float* d_sum,
                       float* h_out, int64_t n_embd, int64_t k, int64_t cap, int tokens,
                       bool write_mapped, std::string& err);

/// STRATA_REMOTE_FLAG_WAIT: publish a completion word into mapped pinned HOST memory at the end of a helper's
/// stream, so the primary can poll a host address instead of synchronising a CUDA event that lives on the
/// helper's device.  `d_flag` is the device alias of a cudaHostAlloc'd, mapped 4-byte word.
bool helper_flag_store(void* stream, unsigned int* d_flag, unsigned int value, std::string& err);

// Remote Expert Decode Optimization v1 on existing helper caches. One instance per serial engine.
class RemoteExpertOpt {
public:
    RemoteExpertOpt() = default;
    ~RemoteExpertOpt();
    RemoteExpertOpt(const RemoteExpertOpt&) = delete;
    RemoteExpertOpt& operator=(const RemoteExpertOpt&) = delete;

    void attach(RemoteExperts& remote); // before RemoteExperts::open
    bool init(std::string& err);        // after all helper caches are open
    bool owns(int64_t layer, int32_t expert) const;
    bool adapt(const std::vector<float>& usage, const std::vector<int32_t>& primary,
               const std::vector<std::pair<int32_t, int32_t>>& pending, int max_swaps,
               ExpertSource& source);

    // Bracket the existing pool callback; its signature and dispatch remain unchanged.
    void begin(const float* weights, int token_begin, int tokens);
    void end() { weights_ = nullptr; }
    void copy_rows(float* dst, const float* src, int token_begin, int tokens,
                   const int32_t* hit_rows, const int32_t* count, void* stream) const;
    void combine(float* dst, float* scratch, int token_begin, int tokens,
                 const uint32_t* skip, uint32_t ring, void* stream) const;

private:
    friend class RemoteExperts;
    /// The reduce metadata's size for the geometry init() read from the helpers: Flash-Next's
    /// 2560-wide rows with k = 10 give 640 B, Qwen3.6's 2048-wide rows with k = 8 give 512 B.
    size_t metadata_bytes() const;
    bool active() const { return weights_ != nullptr; }
    void prepare(const RemoteExperts& remote, void* metadata) const;
    bool reduce(RemoteExperts& remote, const void* metadata, std::string& err);
    void accumulate(const RemoteExperts& remote);

    struct Peer { RemoteExperts* remote; float* sum = nullptr; };
    std::vector<Peer> peers_;
    int64_t n_embd_ = 2560;  ///< the model's row width, set by init() from the helpers' open()
    int64_t k_ = 10;        ///< the experts routed per token, set by init() from the helpers' open()
    int64_t cap_ = 80;      ///< the window's entry count, MAXT * k_, set by init()
    float* h_sum_ = nullptr;
    float* m_sum_ = nullptr;
    int32_t* h_mask_ = nullptr;
    int32_t* m_mask_ = nullptr;
    const float* weights_ = nullptr;
    int token_begin_ = 0, tokens_ = 0;
};
} // namespace strata::core
