#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "model_weights.h"
#include "json_lite.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

struct RawTensorInfo {
    std::string blob_path;
    size_t data_offset = 0;
    size_t nbytes = 0;
    std::vector<int64_t> shape;
    std::string dtype;
};

static inline float bf16_to_f32(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

InklingWeights::~InklingWeights() {
    owned_buffers_.clear();
    for (auto& mb : mapped_blobs_) {
        mb.buf = nil;
        if (mb.ptr && mb.ptr != MAP_FAILED) {
            munmap(mb.ptr, mb.mapped_len);
        }
    }
    mapped_blobs_.clear();
}

static bool parse_safetensors_header(const std::string& path,
                                     std::unordered_map<std::string, RawTensorInfo>& out) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    uint64_t hlen = 0;
    if (pread(fd, &hlen, 8, 0) != 8) { close(fd); return false; }
    std::string hdr(hlen, '\0');
    if (pread(fd, hdr.data(), hlen, 8) != static_cast<ssize_t>(hlen)) { close(fd); return false; }
    close(fd);

    json_lite::Value root = json_lite::parse(hdr);
    for (const auto& kv : root.as_object()) {
        if (kv.first == "__metadata__") continue;
        const auto& obj = kv.second;
        RawTensorInfo info;
        info.blob_path = path;
        info.dtype = obj.at("dtype").as_string();
        for (const auto& s : obj.at("shape").as_array()) info.shape.push_back(s.as_i64());
        const auto& offs = obj.at("data_offsets").as_array();
        size_t start = static_cast<size_t>(offs[0].as_i64());
        size_t end = static_cast<size_t>(offs[1].as_i64());
        info.data_offset = 8 + static_cast<size_t>(hlen) + start;
        info.nbytes = end - start;
        out.emplace(kv.first, std::move(info));
    }
    return true;
}

static bool read_raw_bytes(const RawTensorInfo& info, void* dst) {
    int fd = open(info.blob_path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    size_t done = 0;
    while (done < info.nbytes) {
        ssize_t n = pread(fd, static_cast<char*>(dst) + done, info.nbytes - done, info.data_offset + done);
        if (n <= 0) { close(fd); return false; }
        done += static_cast<size_t>(n);
    }
    close(fd);
    return true;
}

bool InklingWeights::load_from_ollama_manifest(id<MTLDevice> device,
                                               const std::string& manifest_path,
                                               const std::string& blobs_dir,
                                               bool load_draft) {
    std::ifstream ifs(manifest_path, std::ios::binary);
    if (!ifs) {
        fprintf(stderr, "Failed to open manifest: %s\n", manifest_path.c_str());
        return false;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    json_lite::Value manifest = json_lite::parse(oss.str());

    std::unordered_map<std::string, std::string> config_blobs;
    std::unordered_map<std::string, RawTensorInfo> raw_tensors;
    std::vector<std::string> tensor_blobs_to_parse;

    for (const auto& layer : manifest.at("layers").as_array()) {
        const std::string& mt = layer.at("mediaType").as_string();
        std::string digest = layer.at("digest").as_string();
        for (char& c : digest) if (c == ':') c = '-';
        std::string bpath = blobs_dir + "/" + digest;
        if (mt == "application/vnd.ollama.image.json") {
            const auto* name = layer.get("name");
            if (name) config_blobs[name->as_string()] = bpath;
        } else if (mt == "application/vnd.ollama.image.tensor") {
            const auto* name = layer.get("name");
            if (!name) continue;
            const std::string& n = name->as_string();
            // Skip vision encoder tensors (not needed for text inference)
            if (n.rfind("model.visual.", 0) == 0) continue;
            if (!load_draft && n.rfind("draft.", 0) == 0) continue;
            tensor_blobs_to_parse.push_back(bpath);
        }
    }

    if (config_blobs.count("tokenizer.json")) {
        tokenizer_path = config_blobs["tokenizer.json"];
    }

    // Parse config.json
    if (config_blobs.count("config.json")) {
        std::ifstream cf(config_blobs["config.json"]);
        std::ostringstream coss; coss << cf.rdbuf();
        json_lite::Value cj = json_lite::parse(coss.str());
        const auto& tc = cj.at("text_config");
        cfg.num_layers = tc.at("num_hidden_layers").as_int();
        cfg.hidden_size = tc.at("hidden_size").as_int();
        cfg.vocab_size = tc.at("vocab_size").as_int();
        cfg.unpadded_vocab = tc.at("unpadded_vocab_size").as_int();
        if (const auto* ml = tc.get("model_max_length"); ml && ml->is_number()) {
            cfg.model_max_length = ml->as_int();
        } else if (const auto* mp = tc.get("max_position_embeddings"); mp && mp->is_number()) {
            cfg.model_max_length = mp->as_int();
        }
        cfg.head_dim = tc.at("head_dim").as_int();
        cfg.heads = tc.at("num_attention_heads").as_int();
        cfg.kv_heads = tc.at("num_key_value_heads").as_int();
        cfg.rel_dim = tc.at("d_rel").as_int();
        cfg.rel_range = tc.at("rel_extent").as_int();
        cfg.window = tc.at("sliding_window_size").as_int();
        cfg.conv_taps = tc.at("sconv_kernel_size").as_int();
        cfg.experts = tc.at("n_routed_experts").as_int();
        cfg.top_k = tc.at("num_experts_per_tok").as_int();
        cfg.expert_width = tc.at("intermediate_size").as_int();
        cfg.residual_block_size = tc.at("n_layers_per_attn_res_block").as_int();
        const auto& lc = tc.at("loop_config");
        cfg.loop_start = lc.at("core_start").as_int();
        cfg.loop_end = lc.at("core_end").as_int();
        cfg.loop_passes = lc.at("num_passes").as_int();
        cfg.gate_rank = lc.at("gate_rank").as_int();
        cfg.norm_eps = tc.at("rms_norm_eps").as_float();
        cfg.qk_clip = tc.at("qk_l2_clip_threshold").as_float();
        cfg.rel_clip = tc.at("rel_l2_clip_threshold").as_float();
        cfg.log_floor = tc.at("log_scaling_n_floor").as_float();
        cfg.log_alpha = tc.at("log_scaling_alpha").as_float();
        cfg.route_scale = tc.at("route_scale").as_float();
        cfg.gate_cap = tc.at("mlp_gate_softcap").as_float();
        cfg.up_cap = tc.at("mlp_up_softcap").as_float();
        cfg.logit_cap = tc.at("final_logit_softcapping").as_float();
    }

    // Build local layer bitmap & 48-visit execution plan
    std::vector<bool> is_local(cfg.num_layers, true);
    is_local[5] = false; is_local[11] = false; is_local[17] = false; is_local[23] = false;
    plan.clear();
    for (int l = 0; l < cfg.loop_end; ++l) plan.push_back({l, 0, is_local[l]});
    for (int p = 1; p < cfg.loop_passes; ++p)
        for (int l = cfg.loop_start; l < cfg.loop_end; ++l) plan.push_back({l, p, is_local[l]});
    for (int l = cfg.loop_end; l < cfg.num_layers; ++l) plan.push_back({l, 0, is_local[l]});

    // Parse all safetensors headers
    raw_tensors.reserve(1024);
    for (const auto& bp : tensor_blobs_to_parse) {
        if (!parse_safetensors_header(bp, raw_tensors)) {
            fprintf(stderr, "Failed to parse safetensors: %s\n", bp.c_str());
            return false;
        }
    }

    const size_t page_sz = static_cast<size_t>(getpagesize());
    std::unordered_map<std::string, MappedBlob*> blob_cache;

    auto mmap_blob = [&](const std::string& path) -> MappedBlob* {
        auto it = blob_cache.find(path);
        if (it != blob_cache.end()) return it->second;
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return nullptr;
        struct stat st;
        fstat(fd, &st);
        size_t fsz = static_cast<size_t>(st.st_size);
        size_t mlen = (fsz + page_sz - 1) & ~(page_sz - 1);
        void* ptr = mmap(nullptr, mlen, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd); // POSIX: mapping stays valid after close! Avoids FD limit!
        if (ptr == MAP_FAILED) return nullptr;
        id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:ptr
                                                      length:mlen
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
        if (!buf) {
            munmap(ptr, mlen);
            return nullptr;
        }
        mapped_blobs_.push_back({ptr, mlen, buf});
        MappedBlob* mb = &mapped_blobs_.back();
        blob_cache[path] = mb;
        return mb;
    };

    auto get_raw = [&](const std::string& name) -> const RawTensorInfo& {
        auto it = raw_tensors.find(name);
        if (it == raw_tensors.end()) {
            fprintf(stderr, "Missing tensor: %s\n", name.c_str());
            std::abort();
        }
        return it->second;
    };

    // Zero-copy mmap tensor ref (used for large tensors: experts, embed, unembed, wo, etc.)
    auto load_mmap_tensor = [&](const std::string& name) -> TensorRef {
        const auto& info = get_raw(name);
        MappedBlob* mb = mmap_blob(info.blob_path);
        if (!mb) { fprintf(stderr, "mmap failed for %s\n", name.c_str()); std::abort(); }
        return {mb->buf, info.data_offset, info.nbytes, info.shape, info.dtype};
    };

    // Arena buffer for small packed tensors (norms, queries, convs, fused projections)
    // Allocate a clean 2 GB arena or separate packed buffers per layer so offsets are always 256-byte aligned!
    auto alloc_aligned_buf = [&](size_t nbytes) -> id<MTLBuffer> {
        size_t sz = (nbytes + 255) & ~size_t(255);
        id<MTLBuffer> b = [device newBufferWithLength:sz options:MTLResourceStorageModeShared];
        owned_buffers_.push_back(b);
        return b;
    };

    // Pack multiple 2D row-major tensors along dim 0 into one contiguous MTLBuffer
    auto pack_concat_rows = [&](const std::vector<std::string>& names) -> TensorRef {
        size_t total = 0;
        int64_t cols = 0;
        int64_t total_rows = 0;
        std::string dt;
        for (const auto& n : names) {
            const auto& info = get_raw(n);
            total += info.nbytes;
            total_rows += info.shape[0];
            cols = info.shape[1];
            dt = info.dtype;
        }
        id<MTLBuffer> b = alloc_aligned_buf(total);
        char* dst = static_cast<char*>(b.contents);
        size_t off = 0;
        for (const auto& n : names) {
            const auto& info = get_raw(n);
            read_raw_bytes(info, dst + off);
            off += info.nbytes;
        }
        return {b, 0, total, {total_rows, cols}, dt};
    };

    // Copy single small tensor into a small shared arena
    id<MTLBuffer> small_arena = alloc_aligned_buf(64 * 1024 * 1024); // 64 MB
    size_t small_off = 0;
    auto alloc_small = [&](size_t nbytes) -> std::pair<id<MTLBuffer>, size_t> {
        size_t aligned = (nbytes + 255) & ~size_t(255);
        if (small_off + aligned > small_arena.length) {
            small_arena = alloc_aligned_buf(64 * 1024 * 1024);
            small_off = 0;
        }
        size_t o = small_off;
        small_off += aligned;
        return {small_arena, o};
    };

    auto load_small = [&](const std::string& name) -> TensorRef {
        const auto& info = get_raw(name);
        auto [b, o] = alloc_small(info.nbytes);
        read_raw_bytes(info, static_cast<char*>(b.contents) + o);
        return {b, o, info.nbytes, info.shape, info.dtype};
    };

    // Load short conv weight [C, 1, 4] BF16 -> transposed [4, C] BF16
    auto load_static_conv = [&](const std::string& name) -> TensorRef {
        const auto& info = get_raw(name);
        int64_t C = info.shape[0];
        int64_t T = info.shape[2]; // 4
        std::vector<uint16_t> tmp(C * T);
        read_raw_bytes(info, tmp.data());
        auto [b, o] = alloc_small(info.nbytes);
        uint16_t* dst = reinterpret_cast<uint16_t*>(static_cast<char*>(b.contents) + o);
        for (int64_t c = 0; c < C; ++c)
            for (int64_t t = 0; t < T; ++t)
                dst[t * C + c] = tmp[c * T + t];
        return {b, o, info.nbytes, {T, C}, "BF16"};
    };

    // Load global target tensors
    embed = load_mmap_tensor("model.llm.embed.weight");
    unembed = load_mmap_tensor("model.llm.unembed.weight");
    embed_norm = load_small("model.llm.embed_norm.weight");
    output_norm = load_small("model.llm.norm.weight");
    output_query = load_small("model.llm.attn_res.output_query");

    // Load cross-pass queries for loop core
    std::vector<float> cp_queries, cp_kv_queries;
    {
        const auto& qi = get_raw("model.llm.middle_cycle_cross_pass_queries");
        const auto& kvi = get_raw("model.llm.middle_cycle_cross_pass_kv_queries");
        cp_queries.resize(qi.nbytes / 4);
        cp_kv_queries.resize(kvi.nbytes / 4);
        read_raw_bytes(qi, cp_queries.data());
        read_raw_bytes(kvi, cp_kv_queries.data());
    }

    layers.resize(cfg.num_layers);
    for (int i = 0; i < cfg.num_layers; ++i) {
        auto& lw = layers[i];
        lw.is_local = is_local[i];
        std::string p = "model.llm.layers." + std::to_string(i) + ".";
        std::string ap = p + "attn.";

        if (lw.is_local) {
            TensorRef n = load_small(p + "attn_norm.weight");
            lw.attn_norm[0] = lw.attn_norm[1] = lw.attn_norm[2] = n;
            // Fuse Q(3072)+R(384)+Gate(3072)+Sink(24)+K(1024)+V(1024) -> [8600, 2304] BF16
            // Layout matches global [0..6552) = Q+R+Gate+Sink, plus K at 6552 and V at 7576!
            lw.fused_q_proj = pack_concat_rows({
                ap + "wq_du.weight",
                ap + "wr_du.weight",
                ap + "gate_attn.weight",
                ap + "attn_sink_proj.weight",
                ap + "wk_dv.weight",
                ap + "wv_dv.weight"
            });
            lw.k_sconv_w = load_static_conv(ap + "k_sconv.weight");
            lw.v_sconv_w = load_static_conv(ap + "v_sconv.weight");
        } else {
            lw.attn_norm[0] = load_small(p + "q_input_norm.weight");
            lw.attn_norm[1] = load_small(p + "k_input_norm.weight");
            lw.attn_norm[2] = load_small(p + "v_input_norm.weight");
            // Fuse Q+R+Gate+Sink = [6552, 2304], K+k_dyn = [1056, 2304], V+v_dyn = [1056, 2304]
            lw.fused_q_proj = pack_concat_rows({
                ap + "wq_du.weight",
                ap + "wr_du.weight",
                ap + "gate_attn.weight",
                ap + "attn_sink_proj.weight"
            });
            lw.fused_k_proj = pack_concat_rows({
                ap + "wk_dv.weight",
                ap + "k_sconv.weight_proj"
            });
            lw.fused_v_proj = pack_concat_rows({
                ap + "wv_dv.weight",
                ap + "v_sconv.weight_proj"
            });
            lw.k_sconv_w = load_small(ap + "k_sconv.static_w");
            lw.v_sconv_w = load_small(ap + "v_sconv.static_w");
        }

        lw.wo = load_mmap_tensor(ap + "wo_ud.weight");
        lw.q_scale = load_small(ap + "q_l2_clip.scale");
        lw.rel_scale = load_small(ap + "rel_l2_clip.scale");
        lw.rel_proj = load_small(ap + "rel_logits_proj.proj");
        lw.attn_sconv_w = load_static_conv(p + "attn_sconv.weight");
        lw.mlp_sconv_w = load_static_conv(p + "mlp_sconv.weight");
        lw.mlp_norm = load_small(p + "mlp_norm.weight");

        if (i > 0) {
            lw.attn_query = load_small("model.llm.attn_res.attn_queries." + std::to_string(i - 1));
        }
        lw.mlp_query = load_small("model.llm.attn_res.mlp_queries." + std::to_string(i));

        lw.router_w = load_small(p + "mlp.gate.weight");
        lw.router_bias = load_small(p + "mlp.gate.bias");

        auto load_bank = [&](const std::string& wname) -> ExpertBankWeights {
            ExpertBankWeights eb;
            eb.weight = load_mmap_tensor(wname);
            eb.scales = load_mmap_tensor(wname + "_scale");
            if (raw_tensors.count(wname + "_amax")) {
                eb.is_nvfp4 = true;
                eb.amax = load_small(wname + "_amax");
                const auto& di = get_raw(wname + "_outer_scale_denominator");
                read_raw_bytes(di, &eb.outer_denom);
            } else {
                eb.is_nvfp4 = false;
            }
            return eb;
        };
        lw.w13 = load_bank(p + "mlp.experts.w13_weight");
        lw.w2 = load_bank(p + "mlp.experts.w2_weight");

        // Loop controllers for core layers [6..17]
        if (i >= cfg.loop_start && i < cfg.loop_end) {
            lw.has_controllers = true;
            int row = 2 * (i - cfg.loop_start);
            size_t qbytes = 4 * cfg.hidden_size * sizeof(float);
            auto [qb, qo] = alloc_small(qbytes);
            float* qdst = reinterpret_cast<float*>(static_cast<char*>(qb.contents) + qo);
            std::memcpy(qdst + 0 * cfg.hidden_size, cp_queries.data() + row * cfg.hidden_size, cfg.hidden_size * 4);
            std::memcpy(qdst + 1 * cfg.hidden_size, cp_kv_queries.data() + row * cfg.hidden_size, cfg.hidden_size * 4);
            std::memcpy(qdst + 2 * cfg.hidden_size, cp_kv_queries.data() + (row + 1) * cfg.hidden_size, cfg.hidden_size * 4);
            std::memcpy(qdst + 3 * cfg.hidden_size, cp_queries.data() + (row + 1) * cfg.hidden_size, cfg.hidden_size * 4);
            lw.msg_queries = {qb, qo, qbytes, {4, cfg.hidden_size}, "F32"};

            const char* cnames[4] = {"attn", "k", "v", "mlp"};
            for (int c = 0; c < 4; ++c) {
                std::string cp = p + "controllers." + cnames[c] + ".";
                lw.ctrl[c].down = load_small(cp + "down_weight");
                lw.ctrl[c].gain_forget = pack_concat_rows({cp + "gain_weight", cp + "forget_weight"});
            }
        }
    }

    // Load DSpark / DFlash2 draft model weights
    if (load_draft && config_blobs.count("draft/config.json")) {
        std::ifstream df(config_blobs["draft/config.json"]);
        std::ostringstream doss; doss << df.rdbuf();
        json_lite::Value dj = json_lite::parse(doss.str());
        draft.num_layers = dj.at("num_hidden_layers").as_int();
        draft.block_size = dj.at("block_size").as_int();
        draft.state_width = dj.at("markov_rank").as_int();
        draft.window = dj.at("sliding_window_size").as_int();
        draft.rope_theta = dj.at("rope_theta").as_float();
        draft.taps.clear();
        for (const auto& t : dj.at("target_layer_ids").as_array()) draft.taps.push_back(t.as_int());

        // Fuse weights [4, 4] BF16 -> softmaxed F32
        {
            const auto& fi = get_raw("draft.fuse_weight");
            uint16_t raw_fw[16];
            read_raw_bytes(fi, raw_fw);
            for (int r = 0; r < 4; ++r) {
                float mx = -1e30f;
                float vals[4];
                for (int c = 0; c < 4; ++c) {
                    vals[c] = bf16_to_f32(raw_fw[r * 4 + c]);
                    if (vals[c] > mx) mx = vals[c];
                }
                float sum = 0.0f;
                for (int c = 0; c < 4; ++c) { vals[c] = std::exp(vals[c] - mx); sum += vals[c]; }
                for (int c = 0; c < 4; ++c) draft.fuse_weights[r][c] = vals[c] / sum;
            }
        }

        draft.mask_embed = load_small("draft.mask_embed");
        draft.norm = load_small("draft.norm.weight");
        draft.recurrent_norm = load_small("draft.markov_head.dflare_norm.weight");
        draft.markov_w1 = load_mmap_tensor("draft.markov_head.markov_w1.weight");
        draft.rnn_proj_w = load_small("draft.markov_head.rnn_proj.weight");
        draft.rnn_proj_b = load_small("draft.markov_head.rnn_proj.bias");

        // Pre-fold draft.markov_head.hidden_out into unembed:
        // folded_w [200058, 256] = unembed[200058, 2304] @ hidden_out.weight[2304, 256]
        // folded_b [200058]      = unembed[200058, 2304] @ hidden_out.bias[2304]
        // Compute on GPU right after Metal init, OR store raw hidden_out and fold on GPU in 5ms!
        // For simplicity here, let's store raw hidden_out temporarily or fold in parallel on CPU/GPU.
        // Better: expose hidden_out_w and hidden_out_b so engine.mm folds on GPU in ~3ms!
        draft.folded_w = load_small("draft.markov_head.hidden_out.weight"); // [2304, 256] BF16 (replaced by GPU-folded [200058, 256])
        draft.folded_b = load_small("draft.markov_head.hidden_out.bias");   // [2304] BF16      (replaced by GPU-folded [200058])

        draft.layers.resize(draft.num_layers);
        for (int i = 0; i < draft.num_layers; ++i) {
            auto& dl = draft.layers[i];
            dl.is_local = (i == 0 || i == 2);
            std::string lp = "draft.layers." + std::to_string(i) + ".";
            std::string ap = lp + "self_attn.";
            dl.input_norm = load_small(lp + "input_layernorm.weight");
            dl.ff_norm = load_small(lp + "post_attention_layernorm.weight");
            dl.ctx_norm = load_small(lp + "dflare_norm.weight");
            dl.q_norm = load_small(ap + "q_norm.weight");
            dl.k_norm = load_small(ap + "k_norm.weight");
            dl.k_norm_ctx = load_small(ap + "k_norm_ctx.weight");
            dl.qkv_proj = pack_concat_rows({ap + "q_proj.weight", ap + "k_proj.weight", ap + "v_proj.weight"});
            dl.ctx_kv_proj = pack_concat_rows({ap + "k_proj_ctx.weight", ap + "v_proj_ctx.weight"});
            dl.o_proj = load_mmap_tensor(ap + "o_proj.weight");
            dl.gate_up_proj = pack_concat_rows({lp + "mlp.gate_proj.weight", lp + "mlp.up_proj.weight"});
            dl.down_proj = load_mmap_tensor(lp + "mlp.down_proj.weight");
        }
        draft.loaded = true;
    }

    return true;
}
