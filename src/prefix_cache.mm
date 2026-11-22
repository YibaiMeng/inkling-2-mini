#import <Foundation/Foundation.h>
#include "prefix_cache.h"
#include "engine.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace {

double cache_now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

bool mkdir_p(const std::string& dir) {
    if (dir.empty()) return false;
    std::string cur;
    size_t i = 0;
    if (dir[0] == '/') {
        cur = "/";
        i = 1;
    }
    while (i < dir.size()) {
        size_t slash = dir.find('/', i);
        std::string part = (slash == std::string::npos) ? dir.substr(i) : dir.substr(i, slash - i);
        if (!part.empty()) {
            if (!cur.empty() && cur.back() != '/') cur.push_back('/');
            cur += part;
            struct stat st {};
            if (stat(cur.c_str(), &st) != 0) {
                if (mkdir(cur.c_str(), 0700) != 0 && errno != EEXIST) {
                    return false;
                }
            }
        }
        if (slash == std::string::npos) break;
        i = slash + 1;
    }
    return true;
}

void append_bytes(std::vector<uint8_t>& out, const void* src, size_t len) {
    if (len == 0) return;
    size_t off = out.size();
    out.resize(off + len);
    std::memcpy(out.data() + off, src, len);
}

} // namespace

PrefixCache::PrefixCache(const PrefixCacheConfig& cfg) {
    configure(cfg);
}

PrefixCache::~PrefixCache() {
    std::lock_guard<std::mutex> lock(mu_);
    clear_locked();
    cleanup_session_dir_locked();
}

void PrefixCache::configure(const PrefixCacheConfig& cfg) {
    std::lock_guard<std::mutex> lock(mu_);
    clear_locked();
    cleanup_session_dir_locked();
    cfg_ = cfg;
    if (cfg_.offload_base_dir.empty()) {
        const char* home = getenv("HOME");
        if (home && home[0] != '\0') {
            cfg_.offload_base_dir = std::string(home) + "/.cache/inkling";
        } else {
            cfg_.offload_base_dir = ".inkling_cache";
        }
    }
    if (cfg_.milestone_interval < kBlockTokens) {
        cfg_.milestone_interval = 4096;
    }
    session_dir_ = cfg_.offload_base_dir + "/offload-" + std::to_string(static_cast<long>(getpid()));
    session_dir_created_ = false;
}

bool PrefixCache::enabled() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (cfg_.ram_budget_bytes > 0 || cfg_.disk_budget_bytes > 0);
}

void PrefixCache::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    clear_locked();
}

void PrefixCache::clear_locked() {
    for (const auto& kv : checkpoints_) {
        if (kv.second.tier == Tier::Disk && !kv.second.disk_path.empty()) {
            unlink(kv.second.disk_path.c_str());
        }
    }
    for (const auto& kv : blocks_) {
        if (kv.second.tier == Tier::Disk && !kv.second.disk_path.empty()) {
            unlink(kv.second.disk_path.c_str());
        }
    }
    checkpoints_.clear();
    blocks_.clear();
    block_index_.clear();
    ram_used_bytes_ = 0;
    disk_used_bytes_ = 0;
}

bool PrefixCache::ensure_session_dir_locked() {
    if (session_dir_created_) return true;
    if (mkdir_p(session_dir_)) {
        session_dir_created_ = true;
        return true;
    }
    return false;
}

void PrefixCache::cleanup_session_dir_locked() {
    if (session_dir_created_ && !session_dir_.empty()) {
        rmdir(session_dir_.c_str());
        session_dir_created_ = false;
    }
}

PrefixCache::Hash128 PrefixCache::hash_seed() {
    return {0x9e3779b97f4a7c15ULL, 0xbf58476d1ce4e5b9ULL};
}

PrefixCache::Hash128 PrefixCache::hash_extend(Hash128 cur, const int32_t* toks, int count) {
    uint64_t h1 = cur.h1;
    uint64_t h2 = cur.h2;
    for (int i = 0; i < count; ++i) {
        uint64_t v = static_cast<uint64_t>(static_cast<uint32_t>(toks[i]));
        h1 = splitmix64(h1 ^ (v + 0x9e3779b97f4a7c15ULL));
        h2 = splitmix64(h2 ^ ((v * 0x94d049bb133111ebULL) + h1));
    }
    return {h1, h2};
}

bool PrefixCache::write_file_atomic(const std::string& path, const std::vector<uint8_t>& data) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = true;
    if (!data.empty()) {
        ok = (fwrite(data.data(), 1, data.size(), f) == data.size());
    }
    if (fclose(f) != 0) ok = false;
    if (!ok) unlink(path.c_str());
    return ok;
}

bool PrefixCache::read_file_exact(const std::string& path, size_t expected_bytes, std::vector<uint8_t>& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    out.resize(expected_bytes);
    bool ok = true;
    if (expected_bytes > 0) {
        ok = (fread(out.data(), 1, expected_bytes, f) == expected_bytes);
    }
    fclose(f);
    if (!ok) {
        std::vector<uint8_t>().swap(out);
    }
    return ok;
}

void PrefixCache::pack_global_kv_range(const SequenceState& seq,
                                       const InklingWeights& weights,
                                       int start_tok,
                                       int num_tok,
                                       bool include_draft,
                                       std::vector<uint8_t>& out) {
    if (num_tok <= 0) return;
    const size_t row_bytes = 128u * 2u; // BF16 head_dim=128
    const size_t head_stride = static_cast<size_t>(seq.max_global_seq) * row_bytes;
    const size_t start_off = static_cast<size_t>(start_tok) * row_bytes;
    const size_t chunk_bytes = static_cast<size_t>(num_tok) * row_bytes;

    auto pack_buf = [&](id<MTLBuffer> buf) {
        const uint8_t* src = reinterpret_cast<const uint8_t*>(buf.contents);
        size_t dst_off = out.size();
        out.resize(dst_off + 8u * chunk_bytes);
        uint8_t* dst = out.data() + dst_off;
        for (size_t h = 0; h < 8u; ++h) {
            std::memcpy(dst + h * chunk_bytes, src + h * head_stride + start_off, chunk_bytes);
        }
    };

    for (size_t vi = 0; vi < weights.plan.size(); ++vi) {
        if (!weights.plan[vi].local) {
            pack_buf(seq.k_cache[vi]);
            pack_buf(seq.v_cache[vi]);
        }
    }
    if (include_draft && weights.draft.loaded) {
        for (int li = 0; li < weights.draft.num_layers; ++li) {
            if (!weights.draft.layers[li].is_local) {
                pack_buf(seq.draft_k_cache[li]);
                pack_buf(seq.draft_v_cache[li]);
            }
        }
    }
}

void PrefixCache::unpack_global_kv_range(SequenceState& seq,
                                         const InklingWeights& weights,
                                         int start_tok,
                                         int num_tok,
                                         bool include_draft,
                                         const uint8_t* src,
                                         size_t src_bytes) {
    if (num_tok <= 0) return;
    const size_t row_bytes = 128u * 2u;
    const size_t head_stride = static_cast<size_t>(seq.max_global_seq) * row_bytes;
    const size_t start_off = static_cast<size_t>(start_tok) * row_bytes;
    const size_t chunk_bytes = static_cast<size_t>(num_tok) * row_bytes;
    size_t cursor = 0;

    auto unpack_buf = [&](id<MTLBuffer> buf) {
        if (cursor + 8u * chunk_bytes > src_bytes) return;
        uint8_t* dst = reinterpret_cast<uint8_t*>(buf.contents);
        for (size_t h = 0; h < 8u; ++h) {
            std::memcpy(dst + h * head_stride + start_off, src + cursor + h * chunk_bytes, chunk_bytes);
        }
        cursor += 8u * chunk_bytes;
    };

    for (size_t vi = 0; vi < weights.plan.size(); ++vi) {
        if (!weights.plan[vi].local) {
            unpack_buf(seq.k_cache[vi]);
            unpack_buf(seq.v_cache[vi]);
        }
    }
    if (include_draft && weights.draft.loaded) {
        for (int li = 0; li < weights.draft.num_layers; ++li) {
            if (!weights.draft.layers[li].is_local) {
                unpack_buf(seq.draft_k_cache[li]);
                unpack_buf(seq.draft_v_cache[li]);
            }
        }
    }
}

void PrefixCache::pack_checkpoint_payload(const SequenceState& seq,
                                          const InklingWeights& weights,
                                          int pos,
                                          int tail_start,
                                          int tail_len,
                                          bool include_draft,
                                          std::vector<uint8_t>& out) {
    out.clear();
    const int win_target = weights.cfg.window; // 1024
    const int win_draft = weights.draft.loaded ? weights.draft.window : 1024;
    const int cap = seq.local_kv_cap; // 2048
    const size_t row_bytes = 128u * 2u;
    const size_t head_stride = static_cast<size_t>(cap) * row_bytes;

    // Estimate size upfront to avoid reallocations
    size_t t_win_len = static_cast<size_t>(std::min(pos, win_target));
    size_t d_win_len = static_cast<size_t>(std::min(pos, win_draft));
    size_t est = static_cast<size_t>(tail_len) * (include_draft ? 36864u : 32768u) +
                 40u * 2u * 8u * t_win_len * row_bytes +
                 (include_draft ? (2u * 2u * 8u * d_win_len * row_bytes + 4608u) : 0u) +
                 2097152u;
    out.reserve(est);

    // 1. Tail global KV (tokens [tail_start .. tail_start + tail_len - 1])
    if (tail_len > 0) {
        pack_global_kv_range(seq, weights, tail_start, tail_len, include_draft, out);
    }

    // Helper: pack active ring window [max(0, pos - win) .. pos - 1]
    auto pack_ring = [&](id<MTLBuffer> buf, int win_sz) {
        int w_start = (pos > win_sz) ? (pos - win_sz) : 0;
        int w_len = pos - w_start;
        if (w_len <= 0) return;
        int s0 = w_start % cap;
        int n1 = std::min(w_len, cap - s0);
        int n2 = w_len - n1;
        const uint8_t* src = reinterpret_cast<const uint8_t*>(buf.contents);
        size_t total_head_bytes = static_cast<size_t>(w_len) * row_bytes;
        size_t dst_off = out.size();
        out.resize(dst_off + 8u * total_head_bytes);
        uint8_t* dst = out.data() + dst_off;
        for (size_t h = 0; h < 8u; ++h) {
            const uint8_t* hs = src + h * head_stride;
            uint8_t* hd = dst + h * total_head_bytes;
            size_t b1 = static_cast<size_t>(n1) * row_bytes;
            std::memcpy(hd, hs + static_cast<size_t>(s0) * row_bytes, b1);
            if (n2 > 0) {
                std::memcpy(hd + b1, hs, static_cast<size_t>(n2) * row_bytes);
            }
        }
    };

    // 2. Target compact local KV rings + 4-tap ShortConv states
    for (size_t vi = 0; vi < weights.plan.size(); ++vi) {
        if (weights.plan[vi].local) {
            pack_ring(seq.k_cache[vi], win_target);
            pack_ring(seq.v_cache[vi], win_target);
        }
        append_bytes(out, seq.conv_k[vi].contents, seq.conv_k[vi].length);
        append_bytes(out, seq.conv_v[vi].contents, seq.conv_v[vi].length);
        append_bytes(out, seq.conv_attn[vi].contents, seq.conv_attn[vi].length);
        append_bytes(out, seq.conv_mlp[vi].contents, seq.conv_mlp[vi].length);
    }

    // 3. Draft compact local KV rings + draft_last_feature
    if (include_draft && weights.draft.loaded) {
        for (int li = 0; li < weights.draft.num_layers; ++li) {
            if (weights.draft.layers[li].is_local) {
                pack_ring(seq.draft_k_cache[li], win_draft);
                pack_ring(seq.draft_v_cache[li], win_draft);
            }
        }
        append_bytes(out, seq.draft_last_feature.contents, seq.draft_last_feature.length);
    }
}

void PrefixCache::unpack_checkpoint_payload(SequenceState& seq,
                                            const InklingWeights& weights,
                                            int pos,
                                            int tail_start,
                                            int tail_len,
                                            bool include_draft,
                                            bool skip_tail_global,
                                            const uint8_t* src,
                                            size_t src_bytes) {
    const int win_target = weights.cfg.window;
    const int win_draft = weights.draft.loaded ? weights.draft.window : 1024;
    const int cap = seq.local_kv_cap;
    const size_t row_bytes = 128u * 2u;
    const size_t head_stride = static_cast<size_t>(cap) * row_bytes;

    size_t num_global_visits = 0;
    for (const auto& v : weights.plan) {
        if (!v.local) num_global_visits++;
    }
    size_t num_draft_global = 0;
    if (include_draft && weights.draft.loaded) {
        for (int li = 0; li < weights.draft.num_layers; ++li) {
            if (!weights.draft.layers[li].is_local) num_draft_global++;
        }
    }
    size_t tail_bytes = static_cast<size_t>(tail_len) * (num_global_visits + num_draft_global) * 2u * 8u * row_bytes;
    size_t cursor = 0;

    // 1. Tail global KV
    if (tail_len > 0) {
        if (!skip_tail_global && cursor + tail_bytes <= src_bytes) {
            unpack_global_kv_range(seq, weights, tail_start, tail_len, include_draft, src + cursor, tail_bytes);
        }
        cursor += tail_bytes;
    }

    auto unpack_ring = [&](id<MTLBuffer> buf, int win_sz) {
        int w_start = (pos > win_sz) ? (pos - win_sz) : 0;
        int w_len = pos - w_start;
        if (w_len <= 0) return;
        int s0 = w_start % cap;
        int n1 = std::min(w_len, cap - s0);
        int n2 = w_len - n1;
        size_t total_head_bytes = static_cast<size_t>(w_len) * row_bytes;
        if (cursor + 8u * total_head_bytes > src_bytes) return;
        uint8_t* dst = reinterpret_cast<uint8_t*>(buf.contents);
        for (size_t h = 0; h < 8u; ++h) {
            const uint8_t* hs = src + cursor + h * total_head_bytes;
            uint8_t* hd = dst + h * head_stride;
            size_t b1 = static_cast<size_t>(n1) * row_bytes;
            std::memcpy(hd + static_cast<size_t>(s0) * row_bytes, hs, b1);
            if (n2 > 0) {
                std::memcpy(hd, hs + b1, static_cast<size_t>(n2) * row_bytes);
            }
        }
        cursor += 8u * total_head_bytes;
    };

    auto unpack_raw = [&](id<MTLBuffer> buf) {
        size_t len = buf.length;
        if (cursor + len > src_bytes) return;
        std::memcpy(buf.contents, src + cursor, len);
        cursor += len;
    };

    // 2. Target compact local rings + ShortConv states
    for (size_t vi = 0; vi < weights.plan.size(); ++vi) {
        if (weights.plan[vi].local) {
            unpack_ring(seq.k_cache[vi], win_target);
            unpack_ring(seq.v_cache[vi], win_target);
        }
        unpack_raw(seq.conv_k[vi]);
        unpack_raw(seq.conv_v[vi]);
        unpack_raw(seq.conv_attn[vi]);
        unpack_raw(seq.conv_mlp[vi]);
    }

    // 3. Draft compact local rings + draft_last_feature
    if (include_draft && weights.draft.loaded) {
        for (int li = 0; li < weights.draft.num_layers; ++li) {
            if (weights.draft.layers[li].is_local) {
                unpack_ring(seq.draft_k_cache[li], win_draft);
                unpack_ring(seq.draft_v_cache[li], win_draft);
            }
        }
        unpack_raw(seq.draft_last_feature);
    }
}

bool PrefixCache::offload_block_locked(GlobalKVBlock& blk) {
    if (blk.tier != Tier::Ram || cfg_.disk_budget_bytes == 0) return false;
    if (!ensure_session_dir_locked()) return false;
    std::string path = session_dir_ + "/blk_" + std::to_string(blk.id) + ".bin";
    if (!write_file_atomic(path, blk.data)) return false;
    std::vector<uint8_t>().swap(blk.data);
    blk.disk_path = std::move(path);
    blk.tier = Tier::Disk;
    ram_used_bytes_ -= blk.payload_bytes;
    disk_used_bytes_ += blk.payload_bytes;
    offloads_to_disk_++;
    return true;
}

bool PrefixCache::offload_checkpoint_locked(Checkpoint& ckpt) {
    if (ckpt.tier != Tier::Ram || cfg_.disk_budget_bytes == 0) return false;
    if (!ensure_session_dir_locked()) return false;
    std::string path = session_dir_ + "/ckpt_" + std::to_string(ckpt.id) + ".bin";
    if (!write_file_atomic(path, ckpt.data)) return false;
    std::vector<uint8_t>().swap(ckpt.data);
    ckpt.disk_path = std::move(path);
    ckpt.tier = Tier::Disk;
    ram_used_bytes_ -= ckpt.payload_bytes;
    disk_used_bytes_ += ckpt.payload_bytes;
    offloads_to_disk_++;
    return true;
}

bool PrefixCache::load_block_to_ram_locked(GlobalKVBlock& blk) {
    if (blk.tier == Tier::Ram) return true;
    std::vector<uint8_t> loaded;
    if (!read_file_exact(blk.disk_path, blk.payload_bytes, loaded)) return false;
    unlink(blk.disk_path.c_str());
    blk.disk_path.clear();
    blk.data = std::move(loaded);
    blk.tier = Tier::Ram;
    disk_used_bytes_ -= blk.payload_bytes;
    ram_used_bytes_ += blk.payload_bytes;
    loads_from_disk_++;
    return true;
}

bool PrefixCache::load_checkpoint_to_ram_locked(Checkpoint& ckpt) {
    if (ckpt.tier == Tier::Ram) return true;
    std::vector<uint8_t> loaded;
    if (!read_file_exact(ckpt.disk_path, ckpt.payload_bytes, loaded)) return false;
    unlink(ckpt.disk_path.c_str());
    ckpt.disk_path.clear();
    ckpt.data = std::move(loaded);
    ckpt.tier = Tier::Ram;
    disk_used_bytes_ -= ckpt.payload_bytes;
    ram_used_bytes_ += ckpt.payload_bytes;
    loads_from_disk_++;
    return true;
}

void PrefixCache::delete_block_locked(uint64_t blk_id) {
    auto it = blocks_.find(blk_id);
    if (it == blocks_.end()) return;
    GlobalKVBlock& blk = it->second;
    if (blk.tier == Tier::Ram) {
        ram_used_bytes_ -= blk.payload_bytes;
    } else {
        if (!blk.disk_path.empty()) unlink(blk.disk_path.c_str());
        disk_used_bytes_ -= blk.payload_bytes;
    }
    auto idx_it = block_index_.find(blk.key);
    if (idx_it != block_index_.end()) {
        auto& vec = idx_it->second;
        vec.erase(std::remove(vec.begin(), vec.end(), blk_id), vec.end());
        if (vec.empty()) block_index_.erase(idx_it);
    }
    blocks_.erase(it);
    evictions_++;
}

void PrefixCache::delete_checkpoint_locked(uint64_t ckpt_id) {
    auto it = checkpoints_.find(ckpt_id);
    if (it == checkpoints_.end()) return;
    Checkpoint& ckpt = it->second;
    if (ckpt.tier == Tier::Ram) {
        ram_used_bytes_ -= ckpt.payload_bytes;
    } else {
        if (!ckpt.disk_path.empty()) unlink(ckpt.disk_path.c_str());
        disk_used_bytes_ -= ckpt.payload_bytes;
    }
    std::vector<uint64_t> refs = std::move(ckpt.block_ids);
    checkpoints_.erase(it);
    evictions_++;

    for (uint64_t bid : refs) {
        auto bit = blocks_.find(bid);
        if (bit != blocks_.end()) {
            if (bit->second.refcount > 0) bit->second.refcount--;
            if (bit->second.refcount == 0) {
                delete_block_locked(bid);
            }
        }
    }
}

void PrefixCache::enforce_budgets_locked(uint64_t protected_ckpt_id,
                                         const std::unordered_set<uint64_t>& protected_blocks) {
    // Phase 1: Enforce RAM budget (offload to NVMe if enabled, else permanently evict coldest checkpoint)
    while (ram_used_bytes_ > cfg_.ram_budget_bytes) {
        bool did_work = false;

        if (cfg_.disk_budget_bytes > 0) {
            // Find coldest unprotected RAM item (unpinned checkpoints + blocks first; pinned checkpoints last)
            bool found = false;
            bool victim_is_ckpt = false;
            uint64_t victim_id = 0;
            uint64_t best_tick = UINT64_MAX;

            // Pass A: unpinned checkpoints and all unprotected RAM blocks
            for (const auto& kv : checkpoints_) {
                const auto& c = kv.second;
                if (c.tier != Tier::Ram || c.id == protected_ckpt_id || c.pinned) continue;
                if (c.last_used_tick < best_tick) {
                    best_tick = c.last_used_tick;
                    victim_is_ckpt = true;
                    victim_id = c.id;
                    found = true;
                }
            }
            for (const auto& kv : blocks_) {
                const auto& b = kv.second;
                if (b.tier != Tier::Ram || protected_blocks.count(b.id) > 0) continue;
                if (b.last_used_tick < best_tick) {
                    best_tick = b.last_used_tick;
                    victim_is_ckpt = false;
                    victim_id = b.id;
                    found = true;
                }
            }
            // Pass B: pinned checkpoints (only if no unpinned RAM items left!)
            if (!found) {
                for (const auto& kv : checkpoints_) {
                    const auto& c = kv.second;
                    if (c.tier != Tier::Ram || c.id == protected_ckpt_id) continue;
                    if (c.last_used_tick < best_tick) {
                        best_tick = c.last_used_tick;
                        victim_is_ckpt = true;
                        victim_id = c.id;
                        found = true;
                    }
                }
            }

            if (found) {
                bool ok = victim_is_ckpt ? offload_checkpoint_locked(checkpoints_.at(victim_id))
                                         : offload_block_locked(blocks_.at(victim_id));
                if (ok) {
                    continue;
                }
            }
        }

        // Fallback (disk disabled or disk write failed): permanently evict coldest unprotected checkpoint
        // that frees RAM (either checkpoint itself or at least one of its blocks is in RAM)
        uint64_t evict_id = 0;
        uint64_t best_tick = UINT64_MAX;
        for (int pass = 0; pass < 2 && evict_id == 0; ++pass) {
            bool allow_pinned = (pass == 1);
            for (const auto& kv : checkpoints_) {
                const auto& c = kv.second;
                if (c.id == protected_ckpt_id) continue;
                if (!allow_pinned && c.pinned) continue;
                bool holds_ram = (c.tier == Tier::Ram);
                if (!holds_ram) {
                    for (uint64_t bid : c.block_ids) {
                        auto bit = blocks_.find(bid);
                        if (bit != blocks_.end() && bit->second.tier == Tier::Ram && bit->second.refcount == 1) {
                            holds_ram = true;
                            break;
                        }
                    }
                }
                if (holds_ram && c.last_used_tick < best_tick) {
                    best_tick = c.last_used_tick;
                    evict_id = c.id;
                }
            }
        }
        if (evict_id != 0) {
            delete_checkpoint_locked(evict_id);
            did_work = true;
        }
        if (!did_work) break;
    }

    // Phase 2: Enforce NVMe disk offload budget (evict coldest checkpoints using disk space)
    while (disk_used_bytes_ > cfg_.disk_budget_bytes) {
        uint64_t evict_id = 0;
        uint64_t best_tick = UINT64_MAX;
        for (int pass = 0; pass < 2 && evict_id == 0; ++pass) {
            bool allow_pinned = (pass == 1);
            for (const auto& kv : checkpoints_) {
                const auto& c = kv.second;
                if (c.id == protected_ckpt_id) continue;
                if (!allow_pinned && c.pinned) continue;
                bool uses_disk = (c.tier == Tier::Disk);
                if (!uses_disk) {
                    for (uint64_t bid : c.block_ids) {
                        auto bit = blocks_.find(bid);
                        if (bit != blocks_.end() && bit->second.tier == Tier::Disk) {
                            uses_disk = true;
                            break;
                        }
                    }
                }
                if (uses_disk && c.last_used_tick < best_tick) {
                    best_tick = c.last_used_tick;
                    evict_id = c.id;
                }
            }
        }
        if (evict_id != 0) {
            delete_checkpoint_locked(evict_id);
        } else {
            break;
        }
    }
}

bool PrefixCache::has_exact_checkpoint(const std::vector<int32_t>& tokens, int pos, bool has_draft) {
    if (pos <= 0 || static_cast<int>(tokens.size()) < pos) return false;
    Hash128 target_hash = hash_extend(hash_seed(), tokens.data(), pos);
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& kv : checkpoints_) {
        auto& c = kv.second;
        if (c.pos == pos && c.has_draft == has_draft && c.prefix_hash == target_hash &&
            std::equal(c.tokens.begin(), c.tokens.end(), tokens.begin())) {
            c.last_used_tick = ++tick_;
            for (uint64_t bid : c.block_ids) {
                auto bit = blocks_.find(bid);
                if (bit != blocks_.end()) bit->second.last_used_tick = tick_;
            }
            return true;
        }
    }
    return false;
}

void PrefixCache::save_checkpoint(const SequenceState& seq,
                                  const InklingWeights& weights,
                                  bool include_draft,
                                  bool pinned,
                                  const std::string& cache_key) {
    const int pos = seq.pos;
    if (pos < 16 || static_cast<int>(seq.cached_tokens.size()) < pos) return;
    const bool has_draft = include_draft && weights.draft.loaded;

    std::lock_guard<std::mutex> lock(mu_);
    const size_t total_budget = cfg_.ram_budget_bytes + cfg_.disk_budget_bytes;
    if (total_budget == 0) return;

    Hash128 full_hash = hash_extend(hash_seed(), seq.cached_tokens.data(), pos);

    // Check if an exact checkpoint already exists (deduplicate + upgrade pinned/LRU if needed)
    for (auto& kv : checkpoints_) {
        auto& c = kv.second;
        if (c.pos == pos && c.has_draft == has_draft && c.prefix_hash == full_hash &&
            std::equal(c.tokens.begin(), c.tokens.end(), seq.cached_tokens.begin())) {
            if (pinned) c.pinned = true;
            if (!cache_key.empty()) c.cache_key = cache_key;
            c.last_used_tick = ++tick_;
            for (uint64_t bid : c.block_ids) {
                auto bit = blocks_.find(bid);
                if (bit != blocks_.end()) bit->second.last_used_tick = tick_;
            }
            return;
        }
    }

    // 1. Deduplicate / insert all full 256-token Global KV blocks
    const int nblocks = pos / kBlockTokens;
    std::vector<uint64_t> block_ids;
    block_ids.reserve(static_cast<size_t>(nblocks));
    std::unordered_set<uint64_t> protected_blocks;
    protected_blocks.reserve(static_cast<size_t>(nblocks));

    Hash128 run_hash = hash_seed();
    uint64_t now_tick = ++tick_;
    for (int b = 0; b < nblocks; ++b) {
        const int start_tok = b * kBlockTokens;
        const int32_t* btoks = seq.cached_tokens.data() + start_tok;
        run_hash = hash_extend(run_hash, btoks, kBlockTokens);
        BlockKey bkey{run_hash, has_draft};

        uint64_t found_id = 0;
        auto idx_it = block_index_.find(bkey);
        if (idx_it != block_index_.end()) {
            for (uint64_t cand_id : idx_it->second) {
                auto bit = blocks_.find(cand_id);
                if (bit != blocks_.end() &&
                    static_cast<int>(bit->second.tokens.size()) == kBlockTokens &&
                    std::equal(bit->second.tokens.begin(), bit->second.tokens.end(), btoks)) {
                    found_id = cand_id;
                    break;
                }
            }
        }

        if (found_id != 0) {
            auto& blk = blocks_.at(found_id);
            blk.refcount++;
            blk.last_used_tick = now_tick;
            block_ids.push_back(found_id);
            protected_blocks.insert(found_id);
        } else {
            GlobalKVBlock blk;
            blk.id = next_id_++;
            blk.key = bkey;
            blk.tokens.assign(btoks, btoks + kBlockTokens);
            pack_global_kv_range(seq, weights, start_tok, kBlockTokens, has_draft, blk.data);
            blk.payload_bytes = blk.data.size();
            blk.refcount = 1;
            blk.tier = Tier::Ram;
            blk.last_used_tick = now_tick;
            ram_used_bytes_ += blk.payload_bytes;

            uint64_t bid = blk.id;
            blocks_.emplace(bid, std::move(blk));
            block_index_[bkey].push_back(bid);
            block_ids.push_back(bid);
            protected_blocks.insert(bid);
        }
    }

    // 2. Pack checkpoint recurrent state + compact local rings + tail global KV
    const int tail_start = nblocks * kBlockTokens;
    const int tail_len = pos - tail_start;

    Checkpoint ckpt;
    ckpt.id = next_id_++;
    ckpt.pos = pos;
    ckpt.has_draft = has_draft;
    ckpt.pinned = pinned;
    ckpt.cache_key = cache_key;
    ckpt.tokens.assign(seq.cached_tokens.begin(), seq.cached_tokens.begin() + pos);
    ckpt.block_ids = std::move(block_ids);
    ckpt.prefix_hash = full_hash;
    pack_checkpoint_payload(seq, weights, pos, tail_start, tail_len, has_draft, ckpt.data);
    ckpt.payload_bytes = ckpt.data.size();
    ckpt.tier = Tier::Ram;
    ckpt.last_used_tick = now_tick;

    // Guard: if a single checkpoint's own payload exceeds the total RAM+Disk budget, abort cleanly
    if (ckpt.payload_bytes > total_budget) {
        for (uint64_t bid : ckpt.block_ids) {
            auto bit = blocks_.find(bid);
            if (bit != blocks_.end()) {
                if (bit->second.refcount > 0) bit->second.refcount--;
                if (bit->second.refcount == 0) delete_block_locked(bid);
            }
        }
        return;
    }

    ram_used_bytes_ += ckpt.payload_bytes;
    uint64_t ckpt_id = ckpt.id;
    checkpoints_.emplace(ckpt_id, std::move(ckpt));

    // 3. Enforce RAM & NVMe offload budgets
    enforce_budgets_locked(ckpt_id, protected_blocks);

    // If RAM budget is 0 (pure NVMe offload mode), offload this checkpoint + blocks too
    if (cfg_.ram_budget_bytes == 0 && cfg_.disk_budget_bytes > 0) {
        auto cit = checkpoints_.find(ckpt_id);
        if (cit != checkpoints_.end()) {
            offload_checkpoint_locked(cit->second);
            for (uint64_t bid : cit->second.block_ids) {
                auto bit = blocks_.find(bid);
                if (bit != blocks_.end() && bit->second.tier == Tier::Ram) {
                    offload_block_locked(bit->second);
                }
            }
        }
        enforce_budgets_locked(ckpt_id, protected_blocks);
    }
}

PrefixRestoreResult PrefixCache::restore_best_prefix(SequenceState& seq,
                                                     const InklingWeights& weights,
                                                     const std::vector<int32_t>& prompt_ids,
                                                     bool need_draft,
                                                     const std::function<bool(int)>& ensure_cap_fn,
                                                     const std::function<void()>& flush_draft_fn,
                                                     const std::function<void()>& reset_seq_fn) {
    PrefixRestoreResult res;
    const int plen = static_cast<int>(prompt_ids.size());
    if (plen <= 1) {
        reset_seq_fn();
        std::lock_guard<std::mutex> lock(mu_);
        misses_++;
        return res;
    }

    double t0 = cache_now_ms();

    // 1. Check live sequence strict-prefix match (0 ms zero-copy!)
    int live_match = 0;
    if (seq.pos > 0 && seq.pos == static_cast<int>(seq.cached_tokens.size()) &&
        seq.pos < plen && (!need_draft || seq.live_has_draft) &&
        std::equal(seq.cached_tokens.begin(), seq.cached_tokens.end(), prompt_ids.begin())) {
        live_match = seq.pos;
    }

    std::lock_guard<std::mutex> lock(mu_);

    // 2. Find deepest matching checkpoint across RAM + NVMe offload
    uint64_t best_ckpt_id = 0;
    int best_pos = live_match;
    for (const auto& kv : checkpoints_) {
        const auto& c = kv.second;
        if (c.pos <= best_pos || c.pos >= plen) continue;
        if (need_draft && !c.has_draft) continue;
        // Fast edge-token filter before full comparison
        if (c.tokens.front() != prompt_ids.front() || c.tokens.back() != prompt_ids[static_cast<size_t>(c.pos - 1)]) {
            continue;
        }
        if (std::equal(c.tokens.begin(), c.tokens.end(), prompt_ids.begin())) {
            best_pos = c.pos;
            best_ckpt_id = c.id;
        }
    }

    // Fast path: live sequence is already at the deepest prefix match!
    if (best_ckpt_id == 0 && live_match > 0) {
        if (need_draft && seq.pending_count > 0) {
            flush_draft_fn();
        }
        hits_live_++;
        saved_tokens_total_ += static_cast<uint64_t>(live_match);
        res.cached_tokens = live_match;
        res.source = "live";
        res.restore_ms = cache_now_ms() - t0;
        return res;
    }

    // Cache miss
    if (best_ckpt_id == 0) {
        reset_seq_fn();
        misses_++;
        res.cached_tokens = 0;
        res.source = "miss";
        res.restore_ms = cache_now_ms() - t0;
        return res;
    }

    // 3. Restore best checkpoint into `seq`!
    Checkpoint& ckpt = checkpoints_.at(best_ckpt_id);
    if (!ensure_cap_fn(ckpt.pos + 1)) {
        reset_seq_fn();
        misses_++;
        return res;
    }

    // Check how many leading tokens of `seq`'s Global KV already match `ckpt.tokens`!
    // (Allows skipping redundant Global KV copies on same-session rollback / turn continuation!)
    int common_live_global = 0;
    if (seq.pos > 0 && static_cast<int>(seq.cached_tokens.size()) >= seq.pos &&
        (!ckpt.has_draft || seq.live_has_draft)) {
        int max_cmp = std::min(seq.pos, ckpt.pos);
        if (ckpt.has_draft) {
            max_cmp = std::min(max_cmp, seq.draft_ctx_pos);
        }
        while (common_live_global < max_cmp &&
               seq.cached_tokens[static_cast<size_t>(common_live_global)] ==
                   ckpt.tokens[static_cast<size_t>(common_live_global)]) {
            common_live_global++;
        }
    }

    uint64_t now_tick = ++tick_;
    ckpt.last_used_tick = now_tick;
    std::unordered_set<uint64_t> protected_blocks;
    protected_blocks.reserve(ckpt.block_ids.size());
    bool touched_disk = false;

    // 3a. Restore Global KV blocks that aren't already resident in `seq`
    const int nblocks = static_cast<int>(ckpt.block_ids.size());
    for (int b = 0; b < nblocks; ++b) {
        uint64_t bid = ckpt.block_ids[static_cast<size_t>(b)];
        auto bit = blocks_.find(bid);
        if (bit == blocks_.end()) {
            // Should never happen; gracefully fall back to full reset
            reset_seq_fn();
            misses_++;
            return res;
        }
        GlobalKVBlock& blk = bit->second;
        blk.last_used_tick = now_tick;
        protected_blocks.insert(bid);

        int b_start = b * kBlockTokens;
        if (b_start + kBlockTokens <= common_live_global) {
            continue; // Already exact in seq.k_cache/v_cache!
        }

        if (blk.tier == Tier::Disk) {
            touched_disk = true;
            if (cfg_.ram_budget_bytes >= blk.payload_bytes) {
                if (!load_block_to_ram_locked(blk)) {
                    reset_seq_fn();
                    misses_++;
                    return res;
                }
                unpack_global_kv_range(seq, weights, b_start, kBlockTokens, ckpt.has_draft,
                                       blk.data.data(), blk.data.size());
            } else {
                // Direct stream from NVMe without keeping in RAM
                std::vector<uint8_t> tmp;
                if (!read_file_exact(blk.disk_path, blk.payload_bytes, tmp)) {
                    reset_seq_fn();
                    misses_++;
                    return res;
                }
                loads_from_disk_++;
                unpack_global_kv_range(seq, weights, b_start, kBlockTokens, ckpt.has_draft,
                                       tmp.data(), tmp.size());
            }
        } else {
            unpack_global_kv_range(seq, weights, b_start, kBlockTokens, ckpt.has_draft,
                                   blk.data.data(), blk.data.size());
        }
    }

    // 3b. Restore checkpoint recurrent state + compact local rings + tail global KV
    const int tail_start = nblocks * kBlockTokens;
    const int tail_len = ckpt.pos - tail_start;
    const bool skip_tail_global = (tail_start + tail_len <= common_live_global);

    if (ckpt.tier == Tier::Disk) {
        touched_disk = true;
        if (cfg_.ram_budget_bytes >= ckpt.payload_bytes) {
            if (!load_checkpoint_to_ram_locked(ckpt)) {
                reset_seq_fn();
                misses_++;
                return res;
            }
            unpack_checkpoint_payload(seq, weights, ckpt.pos, tail_start, tail_len,
                                      ckpt.has_draft, skip_tail_global,
                                      ckpt.data.data(), ckpt.data.size());
        } else {
            std::vector<uint8_t> tmp;
            if (!read_file_exact(ckpt.disk_path, ckpt.payload_bytes, tmp)) {
                reset_seq_fn();
                misses_++;
                return res;
            }
            loads_from_disk_++;
            unpack_checkpoint_payload(seq, weights, ckpt.pos, tail_start, tail_len,
                                      ckpt.has_draft, skip_tail_global,
                                      tmp.data(), tmp.size());
        }
    } else {
        unpack_checkpoint_payload(seq, weights, ckpt.pos, tail_start, tail_len,
                                  ckpt.has_draft, skip_tail_global,
                                  ckpt.data.data(), ckpt.data.size());
    }

    // Update SequenceState metadata
    seq.pos = ckpt.pos;
    seq.draft_ctx_pos = (ckpt.has_draft && weights.draft.loaded) ? ckpt.pos : 0;
    seq.pending_taps.clear();
    seq.pending_count = 0;
    seq.live_has_draft = ckpt.has_draft;
    seq.cached_tokens = ckpt.tokens;

    // Enforce budgets after any Disk -> RAM promotions
    if (touched_disk) {
        enforce_budgets_locked(best_ckpt_id, protected_blocks);
        hits_disk_++;
        res.source = "nvme";
    } else {
        hits_ram_++;
        res.source = "ram";
    }
    saved_tokens_total_ += static_cast<uint64_t>(ckpt.pos);
    res.cached_tokens = ckpt.pos;
    res.restore_ms = cache_now_ms() - t0;
    return res;
}

PrefixCacheStats PrefixCache::get_stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    PrefixCacheStats s;
    s.ram_used_bytes = ram_used_bytes_;
    s.ram_budget_bytes = cfg_.ram_budget_bytes;
    s.disk_used_bytes = disk_used_bytes_;
    s.disk_budget_bytes = cfg_.disk_budget_bytes;
    for (const auto& kv : checkpoints_) {
        if (kv.second.tier == Tier::Ram) s.checkpoints_ram++;
        else s.checkpoints_disk++;
    }
    for (const auto& kv : blocks_) {
        if (kv.second.tier == Tier::Ram) s.blocks_ram++;
        else s.blocks_disk++;
    }
    s.hits_live = hits_live_;
    s.hits_ram = hits_ram_;
    s.hits_disk = hits_disk_;
    s.misses = misses_;
    s.saved_tokens_total = saved_tokens_total_;
    s.offloads_to_disk = offloads_to_disk_;
    s.loads_from_disk = loads_from_disk_;
    s.evictions = evictions_;
    s.offload_dir = session_dir_;
    return s;
}

std::string PrefixCache::stats_json() const {
    PrefixCacheStats s = get_stats();
    std::ostringstream oss;
    oss << "{\"ram_used_bytes\":" << s.ram_used_bytes
        << ",\"ram_budget_bytes\":" << s.ram_budget_bytes
        << ",\"disk_used_bytes\":" << s.disk_used_bytes
        << ",\"disk_budget_bytes\":" << s.disk_budget_bytes
        << ",\"checkpoints_ram\":" << s.checkpoints_ram
        << ",\"checkpoints_disk\":" << s.checkpoints_disk
        << ",\"blocks_ram\":" << s.blocks_ram
        << ",\"blocks_disk\":" << s.blocks_disk
        << ",\"hits_live\":" << s.hits_live
        << ",\"hits_ram\":" << s.hits_ram
        << ",\"hits_disk\":" << s.hits_disk
        << ",\"misses\":" << s.misses
        << ",\"saved_tokens_total\":" << s.saved_tokens_total
        << ",\"offloads_to_disk\":" << s.offloads_to_disk
        << ",\"loads_from_disk\":" << s.loads_from_disk
        << ",\"evictions\":" << s.evictions << "}";
    return oss.str();
}
