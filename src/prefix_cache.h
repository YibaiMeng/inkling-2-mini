#pragma once
#import <Metal/Metal.h>
#include "model_weights.h"
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct SequenceState;

struct PrefixCacheConfig {
    size_t ram_budget_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;   // 4 GiB default RAM budget
    size_t disk_budget_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL; // 16 GiB default NVMe offload budget (0 = disabled)
    std::string offload_base_dir;                                   // Default: $HOME/.cache/inkling
    int milestone_interval = 4096;                                  // Save intermediate checkpoints every 4K tokens
};

struct PrefixCacheStats {
    size_t ram_used_bytes = 0;
    size_t ram_budget_bytes = 0;
    size_t disk_used_bytes = 0;
    size_t disk_budget_bytes = 0;
    size_t checkpoints_ram = 0;
    size_t checkpoints_disk = 0;
    size_t blocks_ram = 0;
    size_t blocks_disk = 0;
    uint64_t hits_live = 0;
    uint64_t hits_ram = 0;
    uint64_t hits_disk = 0;
    uint64_t misses = 0;
    uint64_t saved_tokens_total = 0;
    uint64_t offloads_to_disk = 0;
    uint64_t loads_from_disk = 0;
    uint64_t evictions = 0;
    std::string offload_dir;
};

struct PrefixRestoreResult {
    int cached_tokens = 0;
    std::string source = "miss"; // "live", "ram", "nvme", "miss"
    double restore_ms = 0.0;
};

class PrefixCache {
public:
    static constexpr int kBlockTokens = 256;

    explicit PrefixCache(const PrefixCacheConfig& cfg = {});
    ~PrefixCache();

    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    void configure(const PrefixCacheConfig& cfg);
    bool enabled() const;
    void clear();

    // Find longest strict prefix match (comparing live sequence vs all cached RAM/NVMe checkpoints)
    // and restore it into `seq`. Caller must ensure `seq` has capacity >= returned cached_tokens
    // via `ensure_cap_fn(needed_tokens)` callback!
    PrefixRestoreResult restore_best_prefix(SequenceState& seq,
                                            const InklingWeights& weights,
                                            const std::vector<int32_t>& prompt_ids,
                                            bool need_draft,
                                            const std::function<bool(int)>& ensure_cap_fn,
                                            const std::function<void()>& flush_draft_fn,
                                            const std::function<void()>& reset_seq_fn);

    // Save a checkpoint at `seq.pos` (deduplicating all full 256-token Global KV blocks!).
    // `pinned` gives higher RAM retention priority (used for system+tools boundaries).
    void save_checkpoint(const SequenceState& seq,
                         const InklingWeights& weights,
                         bool include_draft,
                         bool pinned,
                         const std::string& cache_key = "");

    // Returns true if an exact checkpoint already exists for `tokens[0..pos-1]` with matching draft flag.
    bool has_exact_checkpoint(const std::vector<int32_t>& tokens, int pos, bool has_draft);

    PrefixCacheStats get_stats() const;
    std::string stats_json() const;

private:
    enum class Tier { Ram, Disk };

    struct Hash128 {
        uint64_t h1 = 0;
        uint64_t h2 = 0;
        bool operator==(const Hash128& o) const { return h1 == o.h1 && h2 == o.h2; }
    };

    struct BlockKey {
        Hash128 hash;
        bool has_draft = false;
        bool operator==(const BlockKey& o) const { return hash == o.hash && has_draft == o.has_draft; }
    };

    struct BlockKeyHash {
        size_t operator()(const BlockKey& k) const {
            uint64_t x = k.hash.h1 ^ (k.hash.h2 * 0x9e3779b97f4a7c15ULL) ^ (k.has_draft ? 1ULL : 0ULL);
            return static_cast<size_t>(x ^ (x >> 33));
        }
    };

    struct GlobalKVBlock {
        uint64_t id = 0;
        BlockKey key;
        std::vector<int32_t> tokens; // Exact 256 tokens for collision-free verification
        size_t payload_bytes = 0;
        uint32_t refcount = 0;
        Tier tier = Tier::Ram;
        std::vector<uint8_t> data;   // Packed [global_visits][2][8][256][128] BF16 (+ draft global if has_draft)
        std::string disk_path;
        uint64_t last_used_tick = 0;
    };

    struct Checkpoint {
        uint64_t id = 0;
        int pos = 0;
        bool has_draft = false;
        bool pinned = false;
        std::string cache_key;
        std::vector<int32_t> tokens;      // Exact prefix tokens [0..pos-1]
        std::vector<uint64_t> block_ids;  // Shared GlobalKVBlock IDs for full 256-token blocks
        Hash128 prefix_hash;              // Hash of full tokens[0..pos-1]
        size_t payload_bytes = 0;         // Size of packed recurrent + local rings + tail global KV
        Tier tier = Tier::Ram;
        std::vector<uint8_t> data;        // Packed: tail_global_kv + compact local rings + conv states + draft_last_feature
        std::string disk_path;
        uint64_t last_used_tick = 0;
    };

    mutable std::mutex mu_;
    PrefixCacheConfig cfg_;
    std::string session_dir_;
    bool session_dir_created_ = false;

    uint64_t next_id_ = 1;
    uint64_t tick_ = 1;

    std::unordered_map<uint64_t, GlobalKVBlock> blocks_;
    std::unordered_map<BlockKey, std::vector<uint64_t>, BlockKeyHash> block_index_;
    std::unordered_map<uint64_t, Checkpoint> checkpoints_;

    size_t ram_used_bytes_ = 0;
    size_t disk_used_bytes_ = 0;
    uint64_t hits_live_ = 0;
    uint64_t hits_ram_ = 0;
    uint64_t hits_disk_ = 0;
    uint64_t misses_ = 0;
    uint64_t saved_tokens_total_ = 0;
    uint64_t offloads_to_disk_ = 0;
    uint64_t loads_from_disk_ = 0;
    uint64_t evictions_ = 0;

    static Hash128 hash_seed();
    static Hash128 hash_extend(Hash128 cur, const int32_t* toks, int count);

    bool ensure_session_dir_locked();
    void cleanup_session_dir_locked();
    static bool write_file_atomic(const std::string& path, const std::vector<uint8_t>& data);
    static bool read_file_exact(const std::string& path, size_t expected_bytes, std::vector<uint8_t>& out);

    static void pack_global_kv_range(const SequenceState& seq,
                                     const InklingWeights& weights,
                                     int start_tok,
                                     int num_tok,
                                     bool include_draft,
                                     std::vector<uint8_t>& out);
    static void unpack_global_kv_range(SequenceState& seq,
                                       const InklingWeights& weights,
                                       int start_tok,
                                       int num_tok,
                                       bool include_draft,
                                       const uint8_t* src,
                                       size_t src_bytes);
    static void pack_checkpoint_payload(const SequenceState& seq,
                                        const InklingWeights& weights,
                                        int pos,
                                        int tail_start,
                                        int tail_len,
                                        bool include_draft,
                                        std::vector<uint8_t>& out);
    static void unpack_checkpoint_payload(SequenceState& seq,
                                          const InklingWeights& weights,
                                          int pos,
                                          int tail_start,
                                          int tail_len,
                                          bool include_draft,
                                          bool skip_tail_global,
                                          const uint8_t* src,
                                          size_t src_bytes);

    bool offload_block_locked(GlobalKVBlock& blk);
    bool offload_checkpoint_locked(Checkpoint& ckpt);
    bool load_block_to_ram_locked(GlobalKVBlock& blk);
    bool load_checkpoint_to_ram_locked(Checkpoint& ckpt);

    void delete_block_locked(uint64_t blk_id);
    void delete_checkpoint_locked(uint64_t ckpt_id);
    void enforce_budgets_locked(uint64_t protected_ckpt_id,
                                const std::unordered_set<uint64_t>& protected_blocks);
    void clear_locked();
};
