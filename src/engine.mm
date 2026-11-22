#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace {

struct BlockMixParams {
    uint32_t num_sources;
    uint32_t num_queries;
    uint32_t f32_mask;
    uint32_t apply_norm;
    float eps;
};

struct LinearParams {
    uint32_t M, N, K;
    uint32_t out_f32, has_bias, x_is_f32;
    float softcap;
};

struct ShortConvParams {
    uint32_t M, C;
    uint32_t x_stride, x_offset;
    uint32_t is_dynamic, dyn_stride, dyn_offset;
    uint32_t save_win;
};

struct AttnPrepParams {
    uint32_t M, pos, kv_capacity, is_ring;
    uint32_t q_stride, r_offset;
    float qk_clip, rel_clip, eps, log_floor, log_alpha;
};

struct RelAttnParams {
    uint32_t M, pos, kv_capacity, is_ring, window, parts;
    uint32_t q_stride, gate_offset, sink_offset;
    float scale;
};

struct RouterParams {
    uint32_t M, w13_grid_x, w2_grid_x;
    float route_scale;
};

struct MoEQuantParams {
    uint32_t K, N;
    float outer_denom, gate_cap, up_cap;
};

struct MoEReduceConvParams {
    uint32_t M;
    uint32_t partial_is_f32;
    uint32_t save_win;
};

struct DraftRopeParams {
    uint32_t M, heads, stride, offset, pos;
    float rope_theta, eps;
    uint32_t write_cache, kv_cap, is_ring;
};

struct DraftAttnParams {
    uint32_t M, start, end, kv_cap, is_ring;
    float scale;
};

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

} // namespace

static id<MTLComputePipelineState> make_pso(id<MTLDevice> dev, id<MTLLibrary> lib, const char* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
    if (!fn) {
        fprintf(stderr, "Missing Metal function: %s\n", name);
        std::abort();
    }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "PSO error for %s: %s\n", name, err.localizedDescription.UTF8String);
        std::abort();
    }
    return pso;
}

bool InklingEngine::init(const std::string& manifest_path,
                         const std::string& blobs_dir,
                         const std::string& metal_path,
                         bool load_draft,
                         int max_seq) {
    double t0 = now_ms();
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Metal device unavailable!\n");
        return false;
    }
    queue = [device newCommandQueue];

    // Compile Metal shaders
    std::ifstream mf(metal_path);
    if (!mf) {
        fprintf(stderr, "Failed to open Metal shader file: %s\n", metal_path.c_str());
        return false;
    }
    std::ostringstream moss;
    moss << mf.rdbuf();
    std::string msrc = moss.str();

    NSError* err = nil;
    // Compile two shader libraries:
    // - lib_fast (MTLMathModeFast): max throughput for ALU-heavy quant/MoE/attention/draft kernels
    // - lib_safe (MTLMathModeSafe): bit-exact FP accumulation order for linear_bf16_m1..m8 across M=1..8
    MTLCompileOptions* opts_fast = [MTLCompileOptions new];
    opts_fast.languageVersion = MTLLanguageVersion3_2;
    opts_fast.mathMode = MTLMathModeFast;
    library = [device newLibraryWithSource:@(msrc.c_str()) options:opts_fast error:&err];
    if (!library) {
        fprintf(stderr, "Metal compile error (fast):\n%s\n", err.localizedDescription.UTF8String);
        return false;
    }
    MTLCompileOptions* opts_safe = [MTLCompileOptions new];
    opts_safe.languageVersion = MTLLanguageVersion3_2;
    opts_safe.mathMode = MTLMathModeSafe;
    id<MTLLibrary> lib_safe = [device newLibraryWithSource:@(msrc.c_str()) options:opts_safe error:&err];
    if (!lib_safe) {
        fprintf(stderr, "Metal compile error (safe):\n%s\n", err.localizedDescription.UTF8String);
        return false;
    }

    pso_copy_bf16_          = make_pso(device, library, "copy_bf16");
    pso_embed_rms_          = make_pso(device, library, "embed_rms_norm");
    pso_rms_bf16_           = make_pso(device, library, "rms_norm_bf16");
    pso_rms_f32_            = make_pso(device, library, "rms_norm_f32_to_bf16");
    pso_block_mix_          = make_pso(device, library, "block_residual_mix");
    pso_ctrl_norm_concat_   = make_pso(device, library, "controller_norm_concat");
    pso_ctrl_mix_rms_       = make_pso(device, library, "controller_mix_rmsnorm");
    for (int m = 1; m <= 8; ++m) {
        char fname[32];
        snprintf(fname, sizeof(fname), "linear_bf16_m%d", m);
        pso_linear_m_[m] = make_pso(device, library, fname);
    }
    pso_linear_k64_         = make_pso(device, library, "linear_bf16_k64");
    pso_linear_tiled_       = make_pso(device, library, "linear_bf16_tiled");
    pso_short_conv_         = make_pso(device, library, "short_conv_4tap");
    pso_short_conv_add_f32_ = make_pso(device, library, "short_conv_4tap_add_f32");
    pso_short_conv_rollback_= make_pso(device, library, "short_conv_rollback");
    pso_attn_prep_qr_       = make_pso(device, library, "attn_prep_qr");
    pso_attn_prep_kv_       = make_pso(device, library, "attn_prep_kv_cache");
    pso_attn_rel_table_     = make_pso(device, library, "attn_rel_table");
    pso_rel_attn_split_     = make_pso(device, library, "rel_attn_split");
    pso_rel_attn_merge_gate_= make_pso(device, library, "rel_attn_merge_gate");
    pso_rel_attn_prefill_amx_ = make_pso(device, library, "rel_attn_prefill_amx");
    pso_add_bf16_f32_       = make_pso(device, library, "add_bf16_to_f32");
    pso_add_f32_bf16_bf16_  = make_pso(device, library, "add_f32_bf16_to_bf16");
    pso_add_bf16_bf16_bf16_ = make_pso(device, library, "add_bf16_bf16_to_bf16");
    pso_moe_topk_           = make_pso(device, library, "moe_topk_and_group");
    pso_nvfp4_quant_act_    = make_pso(device, library, "nvfp4_quant_act_4over6");
    pso_mxfp8_quant_act_    = make_pso(device, library, "mxfp8_quant_act");
    pso_moe_w13_nvfp4_m1_   = make_pso(device, lib_safe, "moe_w13_nvfp4_m1");
    pso_moe_w13_nvfp4_      = make_pso(device, lib_safe, "moe_w13_nvfp4_coalesced");
    pso_moe_w2_nvfp4_       = make_pso(device, lib_safe, "moe_w2_nvfp4_coalesced");
    pso_moe_w13_mxfp8_      = make_pso(device, lib_safe, "moe_w13_mxfp8_coalesced");
    pso_moe_w2_mxfp8_       = make_pso(device, lib_safe, "moe_w2_mxfp8_coalesced");
    pso_moe_w13_nvfp4_prefill_ = make_pso(device, library, "moe_w13_nvfp4_prefill");
    pso_moe_w2_nvfp4_prefill_  = make_pso(device, library, "moe_w2_nvfp4_prefill");
    pso_moe_w13_mxfp8_prefill_ = make_pso(device, library, "moe_w13_mxfp8_prefill");
    pso_moe_w2_mxfp8_prefill_  = make_pso(device, library, "moe_w2_mxfp8_prefill");
    pso_moe_reduce_         = make_pso(device, library, "moe_reduce_sum");
    pso_moe_reduce_conv_add_= make_pso(device, library, "moe_reduce_conv_add");
    pso_argmax_             = make_pso(device, library, "argmax_f32");

    pso_draft_fuse_taps_    = make_pso(device, library, "draft_fuse_taps");
    pso_draft_qk_rope_      = make_pso(device, library, "draft_qk_norm_rope");
    pso_draft_v_cache_      = make_pso(device, library, "draft_v_to_cache");
    pso_draft_bidir_attn_   = make_pso(device, library, "draft_bidir_attn");
    pso_draft_swiglu_       = make_pso(device, library, "draft_swiglu");
    pso_draft_res_add_      = make_pso(device, library, "draft_res_add_f32");
    pso_draft_markov_pack_  = make_pso(device, library, "draft_markov_pack");
    pso_draft_markov_update_= make_pso(device, library, "draft_markov_update");
    pso_draft_folded_logits_= make_pso(device, library, "draft_folded_logits");
    pso_precompute_folded_  = make_pso(device, library, "precompute_draft_folded");

    double t_metal = now_ms();

    if (!weights.load_from_ollama_manifest(device, manifest_path, blobs_dir, load_draft)) {
        return false;
    }
    double t_weights = now_ms();

    if (!tokenizer.load_from_json_file(weights.tokenizer_path)) {
        fprintf(stderr, "Failed to load tokenizer: %s\n", weights.tokenizer_path.c_str());
        return false;
    }
    double t_tok = now_ms();

    // Allocate static scratch buffers (shared storage on Apple Silicon unified memory)
    auto alloc = [&](size_t bytes) -> id<MTLBuffer> {
        return [device newBufferWithLength:std::max<size_t>(bytes, 256)
                                   options:MTLResourceStorageModeShared];
    };

    const size_t D = weights.cfg.hidden_size; // 2304
    const size_t V = weights.cfg.unpadded_vocab; // 200058

    buf_tok_ids_ = alloc(MAX_M * sizeof(int32_t));
    buf_argmax_  = alloc(MAX_M * sizeof(int32_t));

    for (int i = 0; i < 5; ++i) buf_res_blocks_[i] = alloc(MAX_M * D * 2);
    buf_partial_bf16_    = alloc(MAX_M * D * 2);
    buf_partial_f32_     = alloc(MAX_M * D * 4);
    buf_prelude_partial_ = alloc(MAX_M * D * 2);
    for (int i = 0; i < 2; ++i) buf_prev_blocks_[i] = alloc(MAX_M * D * 2);
    for (int i = 0; i < 4; ++i) buf_taps_[i] = alloc(MAX_M * D * 2);

    buf_mix_f32_     = alloc(MAX_M * 4 * D * 4);
    buf_msg_f32_     = alloc(MAX_M * 4 * D * 4);
    for (int i = 0; i < 3; ++i) buf_norm_in_[i] = alloc(MAX_M * D * 2);
    buf_ffn_in_      = alloc(MAX_M * D * 2);
    buf_ctrl_concat_ = alloc(MAX_M * 4608 * 2);
    buf_ctrl_down_   = alloc(MAX_M * 64 * 4);
    buf_ctrl_gf_     = alloc(MAX_M * 4608 * 4);

    buf_q_proj_      = alloc(MAX_M * 8600 * 2);
    buf_k_proj_      = alloc(MAX_M * 1056 * 2);
    buf_v_proj_      = alloc(MAX_M * 1056 * 2);
    buf_k_conv_      = alloc(MAX_M * 1024 * 2);
    buf_v_conv_      = alloc(MAX_M * 1024 * 2);
    buf_q_f32_       = alloc(MAX_M * 24 * 128 * 4);
    buf_r_bf16_      = alloc(MAX_M * 24 * 16 * 2);
    buf_rel_table_   = alloc(MAX_M * 24 * 1024 * 2);
    buf_attn_part_out_ = alloc(8 * 24 * 16 * 128 * 4);
    buf_attn_part_st_  = alloc(8 * 24 * 16 * 8);
    buf_attn_out_    = alloc(MAX_M * 3072 * 2);
    buf_wo_out_      = alloc(MAX_M * D * 2);
    buf_sconv_out_   = alloc(MAX_M * D * 2);

    buf_router_logits_ = alloc(MAX_M * 128 * 4);
    buf_route_table_ = alloc(1048576);
    buf_ind_w13_     = alloc(64);
    buf_ind_w2_      = alloc(64);
    buf_act_q_       = alloc(MAX_M * D * 4);
    buf_act_amax_    = alloc(MAX_M * 4);
    buf_moe_mid_     = alloc(MAX_M * 8 * 1152 * 2);
    buf_mid_q_       = alloc(MAX_M * 8 * 1152 * 4);
    buf_mid_amax_    = alloc(MAX_M * 8 * 4);
    buf_moe_down_    = alloc(MAX_M * 8 * D * 4);
    buf_ffn_out_     = alloc(MAX_M * D * 2);

    buf_logits_      = alloc(MAX_LOGITS_M * V * 4);

    if (weights.draft.loaded) {
        buf_draft_folded_w_ = alloc(V * 256 * 2);
        buf_draft_folded_b_ = alloc(V * 2);
        for (int i = 0; i < 4; ++i) buf_draft_ctx_taps_[i] = alloc(64 * D * 2);
        buf_draft_ctx_mix_  = alloc(64 * D * 2);
        buf_draft_ctx_kv_   = alloc(64 * 2048 * 2);
        buf_draft_x_f32_    = alloc(8 * D * 4);
        buf_draft_x_bf16_   = alloc(8 * D * 2);
        buf_draft_norm_     = alloc(8 * D * 2);
        buf_draft_qkv_      = alloc(8 * 5120 * 2);
        buf_draft_q_        = alloc(8 * 24 * 128 * 2);
        buf_draft_attn_     = alloc(8 * 3072 * 2);
        buf_draft_o_        = alloc(8 * D * 2);
        buf_draft_gate_up_  = alloc(8 * 18432 * 2);
        buf_draft_swiglu_   = alloc(8 * 9216 * 2);
        buf_draft_base_logits_ = alloc(7 * V * 2);
        buf_draft_step_logits_ = alloc(V * 4);
        buf_draft_latent_   = alloc(256 * 2);
        buf_draft_rnn_in_   = alloc(5120 * 2);
        buf_draft_rnn_gates_= alloc(768 * 2);
        buf_draft_tok_      = alloc(8 * 4);
        precompute_draft_folded();
    }

    double t_end = now_ms();
    fprintf(stderr, "[init] Metal: %.0fms | Weights: %.0fms | Tokenizer: %.0fms | Scratch+DraftFold: %.0fms | Total: %.2fs\n",
            t_metal - t0, t_weights - t_metal, t_tok - t_weights, t_end - t_tok, (t_end - t0) / 1000.0);
    return true;
}

void InklingEngine::precompute_draft_folded() {
    uint32_t V = weights.cfg.unpadded_vocab;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso_precompute_folded_];
    [enc setBuffer:weights.unembed.buf offset:weights.unembed.offset atIndex:0];
    [enc setBuffer:weights.draft.folded_w.buf offset:weights.draft.folded_w.offset atIndex:1];
    [enc setBuffer:weights.draft.folded_b.buf offset:weights.draft.folded_b.offset atIndex:2];
    [enc setBuffer:buf_draft_folded_w_ offset:0 atIndex:3];
    [enc setBuffer:buf_draft_folded_b_ offset:0 atIndex:4];
    [enc setBytes:&V length:sizeof(V) atIndex:5];
    // Grid: (257, V, 1), TG: (32, 8, 1)
    [enc dispatchThreads:MTLSizeMake(257, V, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

SequenceState InklingEngine::create_sequence(int max_global_seq) {
    SequenceState s;
    s.pos = 0;
    s.max_seq_limit = std::max(max_global_seq, 1024);
    // Start at <= 4096 tokens (~144 MB global KV) and dynamically grow up to max_seq_limit (256K = 9.0 GB) on demand!
    s.max_global_seq = std::min(s.max_seq_limit, 4096);
    s.local_kv_cap = 1088;
    const size_t nvisits = weights.plan.size(); // 48
    s.k_cache.resize(nvisits);
    s.v_cache.resize(nvisits);
    s.conv_k.resize(nvisits);
    s.conv_v.resize(nvisits);
    s.conv_attn.resize(nvisits);
    s.conv_mlp.resize(nvisits);
    s.win_k.resize(nvisits);
    s.win_v.resize(nvisits);
    s.win_attn.resize(nvisits);
    s.win_mlp.resize(nvisits);

    auto alloc_zero = [&](size_t bytes) -> id<MTLBuffer> {
        id<MTLBuffer> b = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        std::memset(b.contents, 0, bytes);
        return b;
    };

    for (size_t i = 0; i < nvisits; ++i) {
        size_t cap = weights.plan[i].local ? static_cast<size_t>(s.local_kv_cap)
                                           : static_cast<size_t>(s.max_global_seq);
        size_t kv_bytes = 8 * cap * 128 * 2; // [8, cap, 128] BF16
        s.k_cache[i] = alloc_zero(kv_bytes);
        s.v_cache[i] = alloc_zero(kv_bytes);
        s.conv_k[i]    = alloc_zero(3 * 1024 * 2);
        s.conv_v[i]    = alloc_zero(3 * 1024 * 2);
        s.conv_attn[i] = alloc_zero(3 * 2304 * 2);
        s.conv_mlp[i]  = alloc_zero(3 * 2304 * 2);
        s.win_k[i]     = alloc_zero(11 * 1024 * 2);
        s.win_v[i]     = alloc_zero(11 * 1024 * 2);
        s.win_attn[i]  = alloc_zero(11 * 2304 * 2);
        s.win_mlp[i]   = alloc_zero(11 * 2304 * 2);
    }

    if (weights.draft.loaded) {
        s.draft_k_cache.resize(weights.draft.num_layers);
        s.draft_v_cache.resize(weights.draft.num_layers);
        for (int i = 0; i < weights.draft.num_layers; ++i) {
            size_t cap = weights.draft.layers[i].is_local ? static_cast<size_t>(s.local_kv_cap)
                                                          : static_cast<size_t>(s.max_global_seq);
            size_t kv_bytes = 8 * cap * 128 * 2;
            s.draft_k_cache[i] = alloc_zero(kv_bytes);
            s.draft_v_cache[i] = alloc_zero(kv_bytes);
        }
        s.draft_last_feature = alloc_zero(2304 * 2);
        s.draft_rnn_state = alloc_zero(256 * 2);
    }
    return s;
}

bool InklingEngine::ensure_global_capacity(SequenceState& s, int needed_tokens) {
    if (needed_tokens <= s.max_global_seq) return true;
    if (needed_tokens > s.max_seq_limit) return false;

    int old_cap = s.max_global_seq;
    int new_cap = std::max(old_cap, 4096);
    while (new_cap < needed_tokens && new_cap < s.max_seq_limit) {
        new_cap = std::min(s.max_seq_limit, new_cap * 2);
    }
    if (new_cap < needed_tokens) new_cap = needed_tokens;

    auto grow_kv = [&](__strong id<MTLBuffer>& buf, int used_rows) {
        size_t new_bytes = size_t(8) * static_cast<size_t>(new_cap) * 128u * 2u;
        id<MTLBuffer> nb = [device newBufferWithLength:new_bytes options:MTLResourceStorageModeShared];
        int copy_rows = std::min(old_cap, std::max(0, used_rows));
        if (copy_rows > 0) {
            const uint8_t* src = reinterpret_cast<const uint8_t*>(buf.contents);
            uint8_t* dst = reinterpret_cast<uint8_t*>(nb.contents);
            size_t old_stride = static_cast<size_t>(old_cap) * 128u * 2u;
            size_t new_stride = static_cast<size_t>(new_cap) * 128u * 2u;
            size_t row_bytes = static_cast<size_t>(copy_rows) * 128u * 2u;
            for (size_t h = 0; h < 8; ++h) {
                std::memcpy(dst + h * new_stride, src + h * old_stride, row_bytes);
            }
        }
        buf = nb;
    };

    for (size_t i = 0; i < weights.plan.size(); ++i) {
        if (!weights.plan[i].local) {
            grow_kv(s.k_cache[i], s.pos);
            grow_kv(s.v_cache[i], s.pos);
        }
    }
    if (weights.draft.loaded) {
        for (int i = 0; i < weights.draft.num_layers; ++i) {
            if (!weights.draft.layers[i].is_local) {
                grow_kv(s.draft_k_cache[i], s.draft_ctx_pos);
                grow_kv(s.draft_v_cache[i], s.draft_ctx_pos);
            }
        }
    }
    s.max_global_seq = new_cap;
    return true;
}

void InklingEngine::reset_sequence(SequenceState& s) {
    s.pos = 0;
    s.draft_ctx_pos = 0;
    s.pending_taps.clear();
    s.pending_count = 0;
    s.live_has_draft = false;
    s.cached_tokens.clear();
    for (size_t i = 0; i < s.conv_k.size(); ++i) {
        std::memset(s.conv_k[i].contents, 0, s.conv_k[i].length);
        std::memset(s.conv_v[i].contents, 0, s.conv_v[i].length);
        std::memset(s.conv_attn[i].contents, 0, s.conv_attn[i].length);
        std::memset(s.conv_mlp[i].contents, 0, s.conv_mlp[i].length);
    }
}

PrefixRestoreResult InklingEngine::prepare_prefix_cache(SequenceState& seq,
                                                        const std::vector<int32_t>& prompt_ids,
                                                        bool need_draft) {
    return prefix_cache.restore_best_prefix(
        seq, weights, prompt_ids, need_draft,
        [&](int needed) { return ensure_global_capacity(seq, needed); },
        [&]() { draft_flush_context(seq); },
        [&]() { reset_sequence(seq); });
}

void InklingEngine::encode_linear(id<MTLComputeCommandEncoder> enc,
                                  id<MTLBuffer> x, size_t x_off, bool x_is_f32,
                                  const TensorRef& w,
                                  id<MTLBuffer> bias, size_t bias_off, bool has_bias,
                                  id<MTLBuffer> y, size_t y_off, bool out_f32,
                                  uint32_t M, uint32_t N, uint32_t K, float softcap) {
    LinearParams p{M, N, K, out_f32 ? 1u : 0u, has_bias ? 1u : 0u, x_is_f32 ? 1u : 0u, softcap};
    if (K == 64 && x_is_f32 && out_f32 && !has_bias && M <= 512) {
        [enc setComputePipelineState:pso_linear_k64_];
        [enc setBuffer:x offset:x_off atIndex:0];
        [enc setBuffer:w.buf offset:w.offset atIndex:1];
        [enc setBuffer:y offset:y_off atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        uint32_t tgs = (N + 15u) / 16u;
        [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return;
    }
    if (M >= 1 && M <= 8 && (K % 256 == 0)) {
        [enc setComputePipelineState:pso_linear_m_[M]];
        [enc setBuffer:x offset:x_off atIndex:0];
        [enc setBuffer:w.buf offset:w.offset atIndex:1];
        [enc setBuffer:(has_bias ? bias : x) offset:(has_bias ? bias_off : 0) atIndex:2];
        [enc setBuffer:y offset:y_off atIndex:3];
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        uint32_t rows_per_tg = 8u;
        uint32_t tgs = (N + rows_per_tg - 1u) / rows_per_tg;
        [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        [enc setComputePipelineState:pso_linear_tiled_];
        [enc setBuffer:x offset:x_off atIndex:0];
        [enc setBuffer:w.buf offset:w.offset atIndex:1];
        [enc setBuffer:(has_bias ? bias : x) offset:(has_bias ? bias_off : 0) atIndex:2];
        [enc setBuffer:y offset:y_off atIndex:3];
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((N + 63u) / 64u, (M + 31u) / 32u, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
}

void InklingEngine::encode_short_conv(id<MTLComputeCommandEncoder> enc,
                                      id<MTLBuffer> x, size_t x_off, uint32_t x_stride, uint32_t x_el_off,
                                      const TensorRef& w,
                                      id<MTLBuffer> state,
                                      id<MTLBuffer> win, bool save_win,
                                      id<MTLBuffer> out, size_t out_off,
                                      uint32_t M, uint32_t C,
                                      bool is_dynamic, uint32_t dyn_stride, uint32_t dyn_el_off) {
    ShortConvParams p{M, C, x_stride, x_el_off, is_dynamic ? 1u : 0u, dyn_stride, dyn_el_off, save_win ? 1u : 0u};
    [enc setComputePipelineState:pso_short_conv_];
    [enc setBuffer:x offset:x_off atIndex:0];
    [enc setBuffer:w.buf offset:w.offset atIndex:1];
    [enc setBuffer:state offset:0 atIndex:2];
    [enc setBuffer:out offset:out_off atIndex:3];
    [enc setBuffer:(win ? win : state) offset:0 atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(C, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void InklingEngine::rollback_target(SequenceState& seq, int prev_pos, int accepted, int proposed_M) {
    seq.pos = prev_pos + accepted;
    if (static_cast<int>(seq.cached_tokens.size()) > seq.pos) {
        seq.cached_tokens.resize(static_cast<size_t>(seq.pos));
    }
    if (accepted >= proposed_M) return;
    uint32_t clen = static_cast<uint32_t>(accepted);
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso_short_conv_rollback_];
        auto rb = [&](id<MTLBuffer> win, id<MTLBuffer> st, uint32_t C) {
            [enc setBuffer:win offset:0 atIndex:0];
            [enc setBuffer:st offset:0 atIndex:1];
            [enc setBytes:&C length:4 atIndex:2];
            [enc setBytes:&clen length:4 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(C, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };
        for (size_t vi = 0; vi < weights.plan.size(); ++vi) {
            rb(seq.win_k[vi], seq.conv_k[vi], 1024u);
            rb(seq.win_v[vi], seq.conv_v[vi], 1024u);
            rb(seq.win_attn[vi], seq.conv_attn[vi], 2304u);
            rb(seq.win_mlp[vi], seq.conv_mlp[vi], 2304u);
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

void InklingEngine::encode_controller(id<MTLComputeCommandEncoder> enc,
                                      const ControllerWeights& ctrl,
                                      id<MTLBuffer> inp, uint32_t inp_stride, uint32_t inp_off,
                                      id<MTLBuffer> msg, uint32_t msg_off,
                                      const TensorRef& norm_w,
                                      id<MTLBuffer> out_bf16,
                                      uint32_t M) {
    float eps = weights.cfg.norm_eps;
    // 1. RMSNorm(inp) || RMSNorm(msg) -> buf_ctrl_concat_ [M, 4608] BF16
    [enc setComputePipelineState:pso_ctrl_norm_concat_];
    [enc setBuffer:inp offset:0 atIndex:0];
    [enc setBuffer:msg offset:0 atIndex:1];
    [enc setBuffer:buf_ctrl_concat_ offset:0 atIndex:2];
    [enc setBytes:&inp_stride length:4 atIndex:3];
    [enc setBytes:&inp_off length:4 atIndex:4];
    [enc setBytes:&msg_off length:4 atIndex:5];
    [enc setBytes:&eps length:4 atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(288, 1, 1)];

    // 2. down [64, 4608] -> buf_ctrl_down_ [M, 64] F32
    encode_linear(enc, buf_ctrl_concat_, 0, false, ctrl.down, nil, 0, false,
                  buf_ctrl_down_, 0, true, M, 64, 4608);

    // 3. gain_forget [4608, 64] -> buf_ctrl_gf_ [M, 4608] F32
    encode_linear(enc, buf_ctrl_down_, 0, true, ctrl.gain_forget, nil, 0, false,
                  buf_ctrl_gf_, 0, true, M, 4608, 64);

    // 4. Fused mix + RMSNorm -> out_bf16 [M, 2304] BF16
    [enc setComputePipelineState:pso_ctrl_mix_rms_];
    [enc setBuffer:inp offset:0 atIndex:0];
    [enc setBuffer:msg offset:0 atIndex:1];
    [enc setBuffer:buf_ctrl_gf_ offset:0 atIndex:2];
    [enc setBuffer:norm_w.buf offset:norm_w.offset atIndex:3];
    [enc setBuffer:out_bf16 offset:0 atIndex:4];
    [enc setBytes:&inp_stride length:4 atIndex:5];
    [enc setBytes:&inp_off length:4 atIndex:6];
    [enc setBytes:&msg_off length:4 atIndex:7];
    [enc setBytes:&eps length:4 atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(288, 1, 1)];
}

void InklingEngine::encode_moe(id<MTLComputeCommandEncoder> enc,
                               const LayerWeights& lw,
                               id<MTLBuffer> ffn_in,
                               id<MTLBuffer> ffn_out,
                               uint32_t M,
                               bool do_reduce) {
    // 1a. Parallel Router GEMV/GEMM [128, 2304] -> buf_router_logits_ [M, 128] F32 (~3 us!)
    encode_linear(enc, ffn_in, 0, false, lw.router_w, nil, 0, false,
                  buf_router_logits_, 0, true, M, 128, 2304);

    // 1b. Fast Top-8 + unique expert bucketing
    // Prefill (M > 8): AMX simdgroup_bfloat8x8 kernels -> grid_x = 36 (64 rows/TG)
    // Decode/Verify (M <= 8): zero-barrier GEMV kernels -> NVFP4 (288, 144), MXFP8 (288, 288)
    const bool use_amx_prefill = (M > 8u);
    uint32_t w13_gx = use_amx_prefill ? 36u : 288u;
    uint32_t w2_gx  = use_amx_prefill ? 36u : (lw.w2.is_nvfp4 ? 144u : 288u);
    RouterParams rp{M, w13_gx, w2_gx, weights.cfg.route_scale};
    [enc setComputePipelineState:pso_moe_topk_];
    [enc setBuffer:buf_router_logits_ offset:0 atIndex:0];
    [enc setBuffer:lw.router_bias.buf offset:lw.router_bias.offset atIndex:1];
    [enc setBuffer:buf_route_table_ offset:0 atIndex:2];
    [enc setBuffer:buf_ind_w13_ offset:0 atIndex:3];
    [enc setBuffer:buf_ind_w2_ offset:0 atIndex:4];
    [enc setBytes:&rp length:sizeof(rp) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    if (lw.w13.is_nvfp4) {
        // 2a. Quantize input activations (exact NVFP4 4-over-6 -> half), W=2304 -> 144 blocks -> 160 threads
        uint32_t W1 = 2304u;
        [enc setComputePipelineState:pso_nvfp4_quant_act_];
        [enc setBuffer:ffn_in offset:0 atIndex:0];
        [enc setBuffer:buf_act_q_ offset:0 atIndex:1];
        [enc setBuffer:buf_act_amax_ offset:0 atIndex:2];
        [enc setBytes:&W1 length:4 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(160, 1, 1)];

        // 2b. Coalesced W13 NVFP4 + Fused Weighted SwiGLU
        MoEQuantParams qp1{2304u, 2304u, lw.w13.outer_denom, weights.cfg.gate_cap, weights.cfg.up_cap};
        id<MTLComputePipelineState> w13_pso = use_amx_prefill ? pso_moe_w13_nvfp4_prefill_
                                                              : (M == 1u ? pso_moe_w13_nvfp4_m1_ : pso_moe_w13_nvfp4_);
        [enc setComputePipelineState:w13_pso];
        [enc setBuffer:buf_act_q_ offset:0 atIndex:0];
        [enc setBuffer:buf_act_amax_ offset:0 atIndex:1];
        [enc setBuffer:lw.w13.weight.buf offset:lw.w13.weight.offset atIndex:2];
        [enc setBuffer:lw.w13.scales.buf offset:lw.w13.scales.offset atIndex:3];
        [enc setBuffer:lw.w13.amax.buf offset:lw.w13.amax.offset atIndex:4];
        [enc setBuffer:buf_route_table_ offset:0 atIndex:5];
        [enc setBuffer:buf_moe_mid_ offset:0 atIndex:6];
        [enc setBytes:&qp1 length:sizeof(qp1) atIndex:7];
        [enc dispatchThreadgroupsWithIndirectBuffer:buf_ind_w13_
                               indirectBufferOffset:0
                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        // 3a. Quantize mid activations (W=1152 -> 72 blocks -> 96 threads, M*8 rows)
        uint32_t W2 = 1152u;
        [enc setComputePipelineState:pso_nvfp4_quant_act_];
        [enc setBuffer:buf_moe_mid_ offset:0 atIndex:0];
        [enc setBuffer:buf_mid_q_ offset:0 atIndex:1];
        [enc setBuffer:buf_mid_amax_ offset:0 atIndex:2];
        [enc setBytes:&W2 length:4 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(M * 8u, 1, 1) threadsPerThreadgroup:MTLSizeMake(96, 1, 1)];

        // 3b. Coalesced W2 NVFP4
        MoEQuantParams qp2{1152u, 2304u, lw.w2.outer_denom, 0.0f, 0.0f};
        [enc setComputePipelineState:(use_amx_prefill ? pso_moe_w2_nvfp4_prefill_ : pso_moe_w2_nvfp4_)];
        [enc setBuffer:buf_mid_q_ offset:0 atIndex:0];
        [enc setBuffer:buf_mid_amax_ offset:0 atIndex:1];
        [enc setBuffer:lw.w2.weight.buf offset:lw.w2.weight.offset atIndex:2];
        [enc setBuffer:lw.w2.scales.buf offset:lw.w2.scales.offset atIndex:3];
        [enc setBuffer:lw.w2.amax.buf offset:lw.w2.amax.offset atIndex:4];
        [enc setBuffer:buf_route_table_ offset:0 atIndex:5];
        [enc setBuffer:buf_moe_down_ offset:0 atIndex:6];
        [enc setBytes:&qp2 length:sizeof(qp2) atIndex:7];
        [enc dispatchThreadgroupsWithIndirectBuffer:buf_ind_w2_
                               indirectBufferOffset:0
                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        // MXFP8 layers (20..23)
        uint32_t W1 = 2304u;
        [enc setComputePipelineState:pso_mxfp8_quant_act_];
        [enc setBuffer:ffn_in offset:0 atIndex:0];
        [enc setBuffer:buf_act_q_ offset:0 atIndex:1];
        [enc setBytes:&W1 length:4 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(W1 / 32u, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        MoEQuantParams qp1{2304u, 2304u, 1.0f, weights.cfg.gate_cap, weights.cfg.up_cap};
        [enc setComputePipelineState:(use_amx_prefill ? pso_moe_w13_mxfp8_prefill_ : pso_moe_w13_mxfp8_)];
        [enc setBuffer:buf_act_q_ offset:0 atIndex:0];
        [enc setBuffer:lw.w13.weight.buf offset:lw.w13.weight.offset atIndex:1];
        [enc setBuffer:lw.w13.scales.buf offset:lw.w13.scales.offset atIndex:2];
        [enc setBuffer:buf_route_table_ offset:0 atIndex:3];
        [enc setBuffer:buf_moe_mid_ offset:0 atIndex:4];
        [enc setBytes:&qp1 length:sizeof(qp1) atIndex:5];
        [enc dispatchThreadgroupsWithIndirectBuffer:buf_ind_w13_
                               indirectBufferOffset:0
                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        uint32_t W2 = 1152u;
        [enc setComputePipelineState:pso_mxfp8_quant_act_];
        [enc setBuffer:buf_moe_mid_ offset:0 atIndex:0];
        [enc setBuffer:buf_mid_q_ offset:0 atIndex:1];
        [enc setBytes:&W2 length:4 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(W2 / 32u, M * 8u, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        [enc setComputePipelineState:(use_amx_prefill ? pso_moe_w2_mxfp8_prefill_ : pso_moe_w2_mxfp8_)];
        [enc setBuffer:buf_mid_q_ offset:0 atIndex:0];
        [enc setBuffer:lw.w2.weight.buf offset:lw.w2.weight.offset atIndex:1];
        [enc setBuffer:lw.w2.scales.buf offset:lw.w2.scales.offset atIndex:2];
        [enc setBuffer:buf_route_table_ offset:0 atIndex:3];
        [enc setBuffer:buf_moe_down_ offset:0 atIndex:4];
        [enc dispatchThreadgroupsWithIndirectBuffer:buf_ind_w2_
                               indirectBufferOffset:0
                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    // 4. Optional standalone reduce (skipped when fused with mlp_sconv + residual add)
    if (do_reduce) {
        [enc setComputePipelineState:pso_moe_reduce_];
        [enc setBuffer:buf_moe_down_ offset:0 atIndex:0];
        [enc setBuffer:ffn_out offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(2304, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}

int32_t InklingEngine::sample_logits_cpu(const float* logits,
                                         const SamplingParams& sampling,
                                         uint64_t& rng_state) const {
    const uint32_t V = weights.cfg.unpadded_vocab;
    if (sampling.temperature <= 0.0f) {
        uint32_t best_i = 0;
        float best_v = logits[0];
        for (uint32_t i = 1; i < V; ++i) {
            if (logits[i] > best_v) { best_v = logits[i]; best_i = i; }
        }
        return static_cast<int32_t>(best_i);
    }

    // xorshift64* PRNG
    auto next_u01 = [&]() -> float {
        rng_state ^= rng_state >> 12;
        rng_state ^= rng_state << 25;
        rng_state ^= rng_state >> 27;
        uint64_t r = rng_state * 2685821657736338717ULL;
        return static_cast<float>((r >> 11) * (1.0 / 9007199254740992.0));
    };

    const float inv_t = 1.0f / sampling.temperature;
    float max_v = logits[0];
    for (uint32_t i = 1; i < V; ++i) {
        if (logits[i] > max_v) max_v = logits[i];
    }

    // Fast path: filter top candidates above threshold (exp(x - max) > 1e-6 -> diff > -13.8)
    struct Cand { uint32_t id; float p; };
    std::vector<Cand> cands;
    cands.reserve(256);
    float sum_p = 0.0f;
    for (uint32_t i = 0; i < V; ++i) {
        float scaled = (logits[i] - max_v) * inv_t;
        if (scaled > -14.0f) {
            float p = std::exp(scaled);
            cands.push_back({i, p});
            sum_p += p;
        }
    }
    if (cands.empty()) return 0;

    if (sampling.top_p < 1.0f && cands.size() > 1) {
        std::sort(cands.begin(), cands.end(),
                  [](const Cand& a, const Cand& b) { return a.p > b.p; });
        float cum = 0.0f;
        size_t keep = 0;
        const float cutoff = sampling.top_p * sum_p;
        for (; keep < cands.size(); ++keep) {
            cum += cands[keep].p;
            if (cum >= cutoff) { ++keep; break; }
        }
        cands.resize(std::max<size_t>(1, keep));
        sum_p = cum;
    }

    float target = next_u01() * sum_p;
    float acc = 0.0f;
    for (const auto& c : cands) {
        acc += c.p;
        if (acc >= target) return static_cast<int32_t>(c.id);
    }
    return static_cast<int32_t>(cands.back().id);
}

// ============================================================================
// Single-command-buffer 48-visit Target Forward (1 <= M <= 32)
// ============================================================================
std::vector<int32_t> InklingEngine::forward_target_single(SequenceState& seq,
                                                          const std::vector<int32_t>& tokens,
                                                          bool capture_taps,
                                                          bool save_conv_windows,
                                                          std::vector<float>* last_logits_out,
                                                          LogitsMode logits_mode) {
    const uint32_t M = static_cast<uint32_t>(tokens.size());
    const uint32_t D = weights.cfg.hidden_size; // 2304
    const float eps = weights.cfg.norm_eps;
    const uint32_t start_pos = static_cast<uint32_t>(seq.pos);
    ensure_global_capacity(seq, seq.pos + static_cast<int>(M));

    std::memcpy(buf_tok_ids_.contents, tokens.data(), M * sizeof(int32_t));

    // If saving conv snapshots for speculative rollback, stash pre-step states on CPU (tiny: 1.9MB)
    // Better: allocate once per SequenceState if needed. We'll copy to CPU vector only when save_conv_windows is true!
    // Wait: copying 48*4 tiny conv buffers on CPU takes < 0.08ms! Let's actually save in SequenceState.win_* via blit encoder!
    // Even simpler: save_conv_windows stores the exact M-token pre-conv inputs? No, just keep a small snapshot buffer!
    // Wait: during speculative verify we might accept 1..M tokens, so we need the conv state at EVERY step 1..M!
    // Let's handle rollback cleanly by recording the pre-step conv_state AND running short_conv on the accepted tokens on rollback, OR re-stepping!
    // Actually: if we stash the pre-step conv_state (1.9 MB memcpy = 0.05 ms) and the pre-conv inputs (48*4 * M * C * 2 = ~5 MB), wait — we can do rollback even simpler!
    // Wait: if we just save the pre-step conv state right before each speculative verify, and if accepted < M, we only need to update the conv states! Let's check: how big is the 3-row conv state? 48 * (2*3*1024 + 2*3*2304) * 2 = 1.92 MB!
    // Even better: add a small kernel or keep a rollback snapshot of the 11-row window `[s0, s1, s2, x0..x_{M-1}]` per visit inside `short_conv_4tap`!
    // For Milestone 1 (exact target verification), let's first encode the clean, blazing-fast target forward!

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        // 0. Embed + EmbedNorm -> buf_partial_bf16_ [M, 2304]
        [enc setComputePipelineState:pso_embed_rms_];
        [enc setBuffer:buf_tok_ids_ offset:0 atIndex:0];
        [enc setBuffer:weights.embed.buf offset:weights.embed.offset atIndex:1];
        [enc setBuffer:weights.embed_norm.buf offset:weights.embed_norm.offset atIndex:2];
        [enc setBuffer:buf_partial_bf16_ offset:0 atIndex:3];
        [enc setBytes:&eps length:4 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(288, 1, 1)];

        // Helper to copy [M, 2304] BF16 between GPU buffers via blit (wait: can't blit inside compute encoder!
        // Use a tiny compute copy or keep ping-pong buffer pointers!)
        // Even better: ZERO copies for blocks! Just track which buffer holds each block!
        // Wait: at boundary (layer % 6 == 0):
        //   visit 0 (L0): saves partial -> block[0]
        //   visit 6 (L6, pass 0): saves prelude_partial = partial (block[0] stays block[0]!), then boundary saves partial -> block[1]
        //   visit 12 (L12, pass 0): boundary saves partial -> block[2]
        //   visit 18 (L6, pass 1): previous = {block[2], partial}; restore state = {block[0], prelude_partial}!
        //                          then boundary saves prelude_partial -> block[1]
        //   visit 24 (L12, pass 1): boundary saves partial -> block[2]
        //   visit 30 (L6, pass 2): previous = {block[2], partial}; restore state = {block[0], prelude_partial}!
        //                          then boundary saves prelude_partial -> block[1]
        //   visit 36 (L12, pass 2): boundary saves partial -> block[2]
        //   visit 42 (L18, pass 0): boundary saves partial -> block[3]
        //   end: blocks = {block[0..3]}, partial = partial (5 sources total!)
        //
        // LOOK HOW CLEAN THAT IS!! We only need a 1-line copy kernel `copy_bf16` (or `add_bf16_bf16` with 0) —
        // actually let's just end/start blit encoder? No, ending compute encoder breaks nothing and blit on 36KB is 0.5us!
        // Actually we can do a super fast copy via `buf_res_blocks_`! Let's use blit encoder at the 8 boundaries + 3 taps = 11 tiny blits (total < 10us across all 48 layers!).

        auto blit_copy = [&](id<MTLBuffer> src, id<MTLBuffer> dst, size_t nbytes) {
            uint32_t n4 = static_cast<uint32_t>(nbytes / 8); // bfloat4 = 8 bytes
            [enc setComputePipelineState:pso_copy_bf16_];
            [enc setBuffer:src offset:0 atIndex:0];
            [enc setBuffer:dst offset:0 atIndex:1];
            [enc setBytes:&n4 length:4 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(n4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };

        const size_t act_bytes = M * D * 2;
        int num_blocks = 0; // how many blocks currently in buf_res_blocks_[0..num_blocks-1]
        int prelude_num_blocks = 0;
        int tap_idx = 0;

        auto dispatch_mix = [&](__strong id<MTLBuffer>* srcs, uint32_t nsrc, uint32_t f32_mask,
                                const TensorRef& query, uint32_t nq,
                                bool apply_norm,
                                const TensorRef* n0, const TensorRef* n1, const TensorRef* n2,
                                id<MTLBuffer> out_f32, id<MTLBuffer> out_bf16) {
            BlockMixParams p{nsrc, nq, f32_mask, apply_norm ? 1u : 0u, eps};
            [enc setComputePipelineState:pso_block_mix_];
            for (uint32_t s = 0; s < 5; ++s) {
                [enc setBuffer:srcs[std::min(s, nsrc - 1)] offset:0 atIndex:s];
            }
            [enc setBuffer:query.buf offset:query.offset atIndex:5];
            [enc setBuffer:(n0 ? n0->buf : buf_partial_bf16_) offset:(n0 ? n0->offset : 0) atIndex:6];
            [enc setBuffer:(n1 ? n1->buf : buf_partial_bf16_) offset:(n1 ? n1->offset : 0) atIndex:7];
            [enc setBuffer:(n2 ? n2->buf : buf_partial_bf16_) offset:(n2 ? n2->offset : 0) atIndex:8];
            [enc setBuffer:out_f32 offset:0 atIndex:9];
            [enc setBuffer:out_bf16 offset:0 atIndex:10];
            [enc setBytes:&p length:sizeof(p) atIndex:11];
            [enc dispatchThreadgroups:MTLSizeMake(1, nq, M) threadsPerThreadgroup:MTLSizeMake(288, 1, 1)];
        };

        for (size_t vi = 0; vi < weights.plan.size(); ++vi) {
            const auto& visit = weights.plan[vi];
            const auto& lw = weights.layers[visit.layer];
            bool boundary = (visit.layer % weights.cfg.residual_block_size == 0);

            // Handle loop core entry (layer == 6)
            if (visit.layer == weights.cfg.loop_start) {
                if (visit.pass == 0) {
                    prelude_num_blocks = num_blocks; // 1
                    blit_copy(buf_partial_bf16_, buf_prelude_partial_, act_bytes);
                } else {
                    // previous = {blocks[prelude_num_blocks+1 .. num_blocks-1], partial} = {blocks[2], partial}
                    blit_copy(buf_res_blocks_[prelude_num_blocks + 1], buf_prev_blocks_[0], act_bytes);
                    blit_copy(buf_partial_bf16_, buf_prev_blocks_[1], act_bytes);
                    // restore state = prelude
                    num_blocks = prelude_num_blocks;
                    blit_copy(buf_prelude_partial_, buf_partial_bf16_, act_bytes);
                }
            }

            // Capture draft tap if requested (taps at visits 5, 23, 41)
            if (capture_taps && tap_idx < 3 && static_cast<int>(vi) == weights.draft.taps[tap_idx]) {
                blit_copy(buf_partial_bf16_, buf_taps_[tap_idx], act_bytes);
                tap_idx++;
            }

            // Compute cross-pass messages if pass > 0
            bool has_msg = (visit.pass > 0);
            if (has_msg) {
                id<MTLBuffer> psrcs[5] = {buf_prev_blocks_[0], buf_prev_blocks_[1], buf_prev_blocks_[1], buf_prev_blocks_[1], buf_prev_blocks_[1]};
                dispatch_mix(psrcs, 2, 0u, lw.msg_queries, 4, false, nullptr, nullptr, nullptr,
                             buf_msg_f32_, buf_norm_in_[0]);
            }

            // ----- Attention Input Preparation -----
            if (num_blocks == 0) {
                // Visit 0 (layer 0, local, pass 0): no mix, just RMSNorm(partial) -> buf_norm_in_[0]
                [enc setComputePipelineState:pso_rms_bf16_];
                [enc setBuffer:buf_partial_bf16_ offset:0 atIndex:0];
                [enc setBuffer:lw.attn_norm[0].buf offset:lw.attn_norm[0].offset atIndex:1];
                [enc setBuffer:buf_norm_in_[0] offset:0 atIndex:2];
                [enc setBytes:&eps length:4 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(288, 1, 1)];
            } else {
                id<MTLBuffer> srcs[5];
                for (int b = 0; b < num_blocks; ++b) srcs[b] = buf_res_blocks_[b];
                srcs[num_blocks] = buf_partial_bf16_;
                uint32_t nsrc = static_cast<uint32_t>(num_blocks + 1);
                uint32_t nq = lw.is_local ? 1u : 3u;

                if (!has_msg) {
                    if (lw.is_local) {
                        // Fused BlockResidualMix + RMSNorm -> buf_norm_in_[0]!
                        dispatch_mix(srcs, nsrc, 0u, lw.attn_query, 1, true,
                                     &lw.attn_norm[0], nullptr, nullptr, buf_mix_f32_, buf_norm_in_[0]);
                    } else {
                        // Global layer: mix produces [M, 3, 2304] F32; then 3 norms? Wait: block_residual_mix
                        // with apply_norm=1 writes [M, 3, 2304] BF16 interleaved by token!
                        // For M=1..8, separating into buf_norm_in_[0..2] is easiest if we mix -> buf_mix_f32_,
                        // OR even better: for global layers (only 8 of 48 visits!), mix -> buf_mix_f32_ (or if M==1 they are contiguous!).
                        // Let's check: when M > 1, [M, 3, 2304] has stride 3*2304 between tokens for q/k/v.
                        // Using mix -> buf_mix_f32_ then 3 tiny RMSNorms? Wait: let's just write to buf_mix_f32_ and run
                        // RMSNorm? Actually our rms_norm_f32_to_bf16 assumes stride 2304!
                        // So let's call dispatch_mix 3 times with query offset (0, 2304*4, 4608*4) and nq=1, apply_norm=1!
                        // That writes DIRECTLY into contiguous [M, 2304] buf_norm_in_[0], [1], [2] with FUSED RMSNorm!!
                        for (int q_i = 0; q_i < 3; ++q_i) {
                            TensorRef qsub = lw.attn_query;
                            qsub.offset += size_t(q_i) * D * sizeof(float);
                            dispatch_mix(srcs, nsrc, 0u, qsub, 1, true,
                                         &lw.attn_norm[q_i], nullptr, nullptr, buf_mix_f32_, buf_norm_in_[q_i]);
                        }
                    }
                } else {
                    // pass > 0: mix to F32, then apply loop controllers + fused RMSNorm -> buf_norm_in_[0..2]
                    dispatch_mix(srcs, nsrc, 0u, lw.attn_query, nq, false,
                                 nullptr, nullptr, nullptr, buf_mix_f32_, buf_norm_in_[0]);
                    uint32_t istride = nq * D;
                    for (int c_i = 0; c_i < 3; ++c_i) {
                        uint32_t ioff = lw.is_local ? 0u : (static_cast<uint32_t>(c_i) * D);
                        encode_controller(enc, lw.ctrl[c_i],
                                          buf_mix_f32_, istride, ioff,
                                          buf_msg_f32_, static_cast<uint32_t>(c_i) * D,
                                          lw.attn_norm[c_i], buf_norm_in_[c_i], M);
                    }
                }
            }

            // ----- Attention Projections + ShortConv -----
            uint32_t q_stride, r_offset, gate_offset, sink_offset;
            if (lw.is_local && !has_msg) {
                // Single fused projection [8600, 2304] for Q(3072)+R(384)+Gate(3072)+Sink(24)+K(1024)+V(1024)!
                encode_linear(enc, buf_norm_in_[0], 0, false, lw.fused_q_proj, nil, 0, false,
                              buf_q_proj_, 0, false, M, 8600, D);
                encode_short_conv(enc, buf_q_proj_, 0, 8600, 6552, lw.k_sconv_w, seq.conv_k[vi], seq.win_k[vi], save_conv_windows, buf_k_conv_, 0, M, 1024);
                encode_short_conv(enc, buf_q_proj_, 0, 8600, 7576, lw.v_sconv_w, seq.conv_v[vi], seq.win_v[vi], save_conv_windows, buf_v_conv_, 0, M, 1024);
                q_stride = 8600; r_offset = 3072; gate_offset = 3456; sink_offset = 6528;
            } else if (lw.is_local && has_msg) {
                encode_linear(enc, buf_norm_in_[0], 0, false, lw.fused_q_proj, nil, 0, false,
                              buf_q_proj_, 0, false, M, 6552, D);
                TensorRef wk_ref = lw.fused_q_proj; wk_ref.offset += size_t(6552) * D * 2;
                TensorRef wv_ref = lw.fused_q_proj; wv_ref.offset += size_t(7576) * D * 2;
                encode_linear(enc, buf_norm_in_[1], 0, false, wk_ref, nil, 0, false, buf_k_proj_, 0, false, M, 1024, D);
                encode_linear(enc, buf_norm_in_[2], 0, false, wv_ref, nil, 0, false, buf_v_proj_, 0, false, M, 1024, D);
                encode_short_conv(enc, buf_k_proj_, 0, 1024, 0, lw.k_sconv_w, seq.conv_k[vi], seq.win_k[vi], save_conv_windows, buf_k_conv_, 0, M, 1024);
                encode_short_conv(enc, buf_v_proj_, 0, 1024, 0, lw.v_sconv_w, seq.conv_v[vi], seq.win_v[vi], save_conv_windows, buf_v_conv_, 0, M, 1024);
                q_stride = 6552; r_offset = 3072; gate_offset = 3456; sink_offset = 6528;
            } else {
                // Global layer (pass 0, 1, or 2): Q+R+Gate+Sink (6552), K+k_dyn (1056), V+v_dyn (1056)
                encode_linear(enc, buf_norm_in_[0], 0, false, lw.fused_q_proj, nil, 0, false, buf_q_proj_, 0, false, M, 6552, D);
                encode_linear(enc, buf_norm_in_[1], 0, false, lw.fused_k_proj, nil, 0, false, buf_k_proj_, 0, false, M, 1056, D);
                encode_linear(enc, buf_norm_in_[2], 0, false, lw.fused_v_proj, nil, 0, false, buf_v_proj_, 0, false, M, 1056, D);
                encode_short_conv(enc, buf_k_proj_, 0, 1056, 0, lw.k_sconv_w, seq.conv_k[vi], seq.win_k[vi], save_conv_windows, buf_k_conv_, 0, M, 1024, true, 1056, 1024);
                encode_short_conv(enc, buf_v_proj_, 0, 1056, 0, lw.v_sconv_w, seq.conv_v[vi], seq.win_v[vi], save_conv_windows, buf_v_conv_, 0, M, 1024, true, 1056, 1024);
                q_stride = 6552; r_offset = 3072; gate_offset = 3456; sink_offset = 6528;
            }

            // (Wait: fix pass-0 local branch above to also use the new [Q,R,Gate,Sink,K,V] packing!)
            // We'll make sure model_weights.mm packs local in order [wq, wr, gate, sink, wk, wv]!

            // ----- Attention Prep (ClipL2 + V RMSNorm + KV Cache Write + RelTable) -----
            uint32_t kv_cap = visit.local ? static_cast<uint32_t>(seq.local_kv_cap) : static_cast<uint32_t>(seq.max_global_seq);
            AttnPrepParams app{
                M, start_pos, kv_cap, visit.local ? 1u : 0u,
                q_stride, r_offset,
                weights.cfg.qk_clip, weights.cfg.rel_clip, eps,
                visit.local ? 0.0f : weights.cfg.log_floor,
                visit.local ? 0.0f : weights.cfg.log_alpha
            };
            [enc setComputePipelineState:pso_attn_prep_qr_];
            [enc setBuffer:buf_q_proj_ offset:0 atIndex:0];
            [enc setBuffer:lw.q_scale.buf offset:lw.q_scale.offset atIndex:1];
            [enc setBuffer:lw.rel_scale.buf offset:lw.rel_scale.offset atIndex:2];
            [enc setBuffer:buf_q_f32_ offset:0 atIndex:3];
            [enc setBuffer:buf_r_bf16_ offset:0 atIndex:4];
            [enc setBytes:&app length:sizeof(app) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(24, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            [enc setComputePipelineState:pso_attn_prep_kv_];
            [enc setBuffer:buf_k_conv_ offset:0 atIndex:0];
            [enc setBuffer:buf_v_conv_ offset:0 atIndex:1];
            [enc setBuffer:seq.k_cache[vi] offset:0 atIndex:2];
            [enc setBuffer:seq.v_cache[vi] offset:0 atIndex:3];
            [enc setBytes:&app length:sizeof(app) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(8, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            [enc setComputePipelineState:pso_attn_rel_table_];
            [enc setBuffer:buf_r_bf16_ offset:0 atIndex:0];
            [enc setBuffer:lw.rel_proj.buf offset:lw.rel_proj.offset atIndex:1];
            [enc setBuffer:buf_rel_table_ offset:0 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(32, M * 24, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            // ----- Relative-Position Attention + Fused Output Gate -----
            // Prefill (M > 8): AMX FlashAttention-2 (BM=16, BN=32, 16x less KV bandwidth, zero partial_out!)
            // Decode/Verify (M <= 8): Fixed parts=16 Split-KV preserves 256/256 bit-exact FP reduction tree!
            uint32_t parts = 16u;
            RelAttnParams rap{
                M, start_pos, kv_cap, visit.local ? 1u : 0u,
                visit.local ? static_cast<uint32_t>(weights.cfg.window) : 0u,
                parts, q_stride, gate_offset, sink_offset,
                1.0f / std::sqrt(128.0f)
            };
            if (M > 8u) {
                [enc setComputePipelineState:pso_rel_attn_prefill_amx_];
                [enc setBuffer:buf_q_f32_ offset:0 atIndex:0];
                [enc setBuffer:seq.k_cache[vi] offset:0 atIndex:1];
                [enc setBuffer:seq.v_cache[vi] offset:0 atIndex:2];
                [enc setBuffer:buf_rel_table_ offset:0 atIndex:3];
                [enc setBuffer:buf_q_proj_ offset:0 atIndex:4];
                [enc setBuffer:buf_attn_out_ offset:0 atIndex:5];
                [enc setBytes:&rap length:sizeof(rap) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake((M + 15u) / 16u, 24, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                [enc setComputePipelineState:pso_rel_attn_split_];
                [enc setBuffer:buf_q_f32_ offset:0 atIndex:0];
                [enc setBuffer:seq.k_cache[vi] offset:0 atIndex:1];
                [enc setBuffer:seq.v_cache[vi] offset:0 atIndex:2];
                [enc setBuffer:buf_rel_table_ offset:0 atIndex:3];
                [enc setBuffer:buf_q_proj_ offset:0 atIndex:4];
                [enc setBuffer:buf_attn_part_out_ offset:0 atIndex:5];
                [enc setBuffer:buf_attn_part_st_ offset:0 atIndex:6];
                [enc setBytes:&rap length:sizeof(rap) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(parts, M * 24, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                [enc setComputePipelineState:pso_rel_attn_merge_gate_];
                [enc setBuffer:buf_attn_part_out_ offset:0 atIndex:0];
                [enc setBuffer:buf_attn_part_st_ offset:0 atIndex:1];
                [enc setBuffer:buf_q_proj_ offset:0 atIndex:2];
                [enc setBuffer:buf_attn_out_ offset:0 atIndex:3];
                [enc setBytes:&rap length:sizeof(rap) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, M * 24, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }

            // ----- Output Projection + Fused Attention ShortConv + Residual Update -----
            encode_linear(enc, buf_attn_out_, 0, false, lw.wo, nil, 0, false, buf_wo_out_, 0, false, M, D, 3072);

            bool partial_is_f32 = false;
            if (boundary) {
                blit_copy(buf_partial_bf16_, buf_res_blocks_[num_blocks], act_bytes);
                num_blocks++;
                // Write ShortConv output DIRECTLY into buf_partial_bf16_ (zero extra copy!)
                encode_short_conv(enc, buf_wo_out_, 0, D, 0, lw.attn_sconv_w, seq.conv_attn[vi],
                                  seq.win_attn[vi], save_conv_windows, buf_partial_bf16_, 0, M, D);
                partial_is_f32 = false;
            } else {
                // Fused ShortConv + Residual Add into buf_partial_f32_ (40 fewer dispatches!)
                ShortConvParams scp{M, D, D, 0u, 0u, 0u, 0u, save_conv_windows ? 1u : 0u};
                [enc setComputePipelineState:pso_short_conv_add_f32_];
                [enc setBuffer:buf_wo_out_ offset:0 atIndex:0];
                [enc setBuffer:lw.attn_sconv_w.buf offset:lw.attn_sconv_w.offset atIndex:1];
                [enc setBuffer:seq.conv_attn[vi] offset:0 atIndex:2];
                [enc setBuffer:buf_partial_bf16_ offset:0 atIndex:3];
                [enc setBuffer:buf_partial_f32_ offset:0 atIndex:4];
                [enc setBuffer:seq.win_attn[vi] offset:0 atIndex:5];
                [enc setBytes:&scp length:sizeof(scp) atIndex:6];
                [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                partial_is_f32 = true;
            }

            // ----- FFN Input Preparation -----
            {
                id<MTLBuffer> srcs[5];
                for (int b = 0; b < num_blocks; ++b) srcs[b] = buf_res_blocks_[b];
                srcs[num_blocks] = partial_is_f32 ? buf_partial_f32_ : buf_partial_bf16_;
                uint32_t nsrc = static_cast<uint32_t>(num_blocks + 1);
                uint32_t f32_mask = partial_is_f32 ? (1u << num_blocks) : 0u;

                if (!has_msg) {
                    dispatch_mix(srcs, nsrc, f32_mask, lw.mlp_query, 1, true,
                                 &lw.mlp_norm, nullptr, nullptr, buf_mix_f32_, buf_ffn_in_);
                } else {
                    dispatch_mix(srcs, nsrc, f32_mask, lw.mlp_query, 1, false,
                                 nullptr, nullptr, nullptr, buf_mix_f32_, buf_ffn_in_);
                    encode_controller(enc, lw.ctrl[3],
                                      buf_mix_f32_, D, 0,
                                      buf_msg_f32_, 3u * D,
                                      lw.mlp_norm, buf_ffn_in_, M);
                }
            }

            // ----- Coalesced Multi-Token MoE + Fused Reduce + MLP ShortConv + Residual Add! -----
            encode_moe(enc, lw, buf_ffn_in_, buf_ffn_out_, M, false);
            {
                MoEReduceConvParams rcp{M, partial_is_f32 ? 1u : 0u, save_conv_windows ? 1u : 0u};
                [enc setComputePipelineState:pso_moe_reduce_conv_add_];
                [enc setBuffer:buf_moe_down_ offset:0 atIndex:0];
                [enc setBuffer:lw.mlp_sconv_w.buf offset:lw.mlp_sconv_w.offset atIndex:1];
                [enc setBuffer:seq.conv_mlp[vi] offset:0 atIndex:2];
                [enc setBuffer:seq.win_mlp[vi] offset:0 atIndex:3];
                [enc setBuffer:(partial_is_f32 ? buf_partial_f32_ : buf_partial_bf16_) offset:0 atIndex:4];
                [enc setBuffer:buf_partial_bf16_ offset:0 atIndex:5];
                [enc setBytes:&rcp length:sizeof(rcp) atIndex:6];
                [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }

        // ----- Final Output Mix + Norm -> buf_ffn_in_ [M, 2304] BF16 -----
        if (capture_taps || logits_mode != LogitsMode::None) {
            id<MTLBuffer> srcs[5];
            for (int b = 0; b < num_blocks; ++b) srcs[b] = buf_res_blocks_[b];
            srcs[num_blocks] = buf_partial_bf16_;
            dispatch_mix(srcs, static_cast<uint32_t>(num_blocks + 1), 0u, weights.output_query, 1, true,
                         &weights.output_norm, nullptr, nullptr, buf_mix_f32_, buf_ffn_in_);
        }

        if (capture_taps) {
            blit_copy(buf_ffn_in_, buf_taps_[3], act_bytes);
        }

        // ----- Unembed [200058, 2304] BF16 + Fused Softcap(30.0) + GPU Argmax -----
        uint32_t V = weights.cfg.unpadded_vocab;
        if (logits_mode == LogitsMode::LastOnly) {
            encode_linear(enc, buf_ffn_in_, size_t(M - 1) * D * 2, false, weights.unembed, nil, 0, false,
                          buf_logits_, 0, true, 1, V, D, weights.cfg.logit_cap);
            [enc setComputePipelineState:pso_argmax_];
            [enc setBuffer:buf_logits_ offset:0 atIndex:0];
            [enc setBuffer:buf_argmax_ offset:0 atIndex:1];
            [enc setBytes:&V length:4 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        } else if (logits_mode == LogitsMode::All) {
            encode_linear(enc, buf_ffn_in_, 0, false, weights.unembed, nil, 0, false,
                          buf_logits_, 0, true, M, V, D, weights.cfg.logit_cap);
            [enc setComputePipelineState:pso_argmax_];
            [enc setBuffer:buf_logits_ offset:0 atIndex:0];
            [enc setBuffer:buf_argmax_ offset:0 atIndex:1];
            [enc setBytes:&V length:4 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    seq.pos += static_cast<int>(M);
    seq.cached_tokens.insert(seq.cached_tokens.end(), tokens.begin(), tokens.end());

    // Save captured taps for draft model if requested
    if (capture_taps && weights.draft.loaded) {
        size_t old_sz = seq.pending_taps.size();
        seq.pending_taps.resize(old_sz + M * 4 * D);
        for (uint32_t t = 0; t < M; ++t) {
            for (int tp = 0; tp < 4; ++tp) {
                const uint16_t* src = reinterpret_cast<const uint16_t*>(buf_taps_[tp].contents) + t * D;
                uint16_t* dst = seq.pending_taps.data() + old_sz + (t * 4 + tp) * D;
                std::memcpy(dst, src, D * 2);
            }
        }
        seq.pending_count += static_cast<int>(M);
    }

    if (logits_mode == LogitsMode::None) {
        return {};
    }

    const uint32_t out_rows = (logits_mode == LogitsMode::LastOnly) ? 1u : M;
    std::vector<int32_t> out(out_rows);
    std::memcpy(out.data(), buf_argmax_.contents, out_rows * sizeof(int32_t));

    if (last_logits_out) {
        uint32_t V = weights.cfg.unpadded_vocab;
        last_logits_out->resize(V);
        const float* lp = reinterpret_cast<const float*>(buf_logits_.contents) + size_t(out_rows - 1) * V;
        std::memcpy(last_logits_out->data(), lp, V * sizeof(float));
    }

    return out;
}

// ============================================================================
// Batched Low-Batch Decode (B = 1..8 concurrent sequences, 1 token each!)
// Shares ALL dense projections, loop controllers, MoE expert banks, and Unembed!
// ============================================================================
std::vector<int32_t> InklingEngine::forward_target_batch(std::vector<SequenceState*>& seqs,
                                                         const std::vector<int32_t>& tokens,
                                                         const SamplingParams* sampling,
                                                         uint64_t* rng_state) {
    const uint32_t B = static_cast<uint32_t>(seqs.size());
    if (B == 1) {
        auto out = forward_target_single(*seqs[0], tokens, false, false, nullptr, LogitsMode::All);
        if (sampling && sampling->temperature > 0.0f && rng_state) {
            out[0] = sample_logits_cpu(reinterpret_cast<const float*>(buf_logits_.contents), *sampling, *rng_state);
        }
        return out;
    }
    const uint32_t D = weights.cfg.hidden_size;
    const float eps = weights.cfg.norm_eps;
    for (auto* s : seqs) ensure_global_capacity(*s, s->pos + 1);
    std::memcpy(buf_tok_ids_.contents, tokens.data(), B * sizeof(int32_t));

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:pso_embed_rms_];
        [enc setBuffer:buf_tok_ids_ offset:0 atIndex:0];
        [enc setBuffer:weights.embed.buf offset:weights.embed.offset atIndex:1];
        [enc setBuffer:weights.embed_norm.buf offset:weights.embed_norm.offset atIndex:2];
        [enc setBuffer:buf_partial_bf16_ offset:0 atIndex:3];
        [enc setBytes:&eps length:4 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(288, 1, 1)];

        auto blit_copy = [&](id<MTLBuffer> src, id<MTLBuffer> dst, size_t nbytes) {
            uint32_t n4 = static_cast<uint32_t>(nbytes / 8); // bfloat4 = 8 bytes
            [enc setComputePipelineState:pso_copy_bf16_];
            [enc setBuffer:src offset:0 atIndex:0];
            [enc setBuffer:dst offset:0 atIndex:1];
            [enc setBytes:&n4 length:4 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(n4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };

        const size_t act_bytes = B * D * 2;
        int num_blocks = 0;
        int prelude_num_blocks = 0;

        auto dispatch_mix = [&](__strong id<MTLBuffer>* srcs, uint32_t nsrc, uint32_t f32_mask,
                                const TensorRef& query, uint32_t nq,
                                bool apply_norm,
                                const TensorRef* n0, const TensorRef* n1, const TensorRef* n2,
                                id<MTLBuffer> out_f32, id<MTLBuffer> out_bf16) {
            BlockMixParams p{nsrc, nq, f32_mask, apply_norm ? 1u : 0u, eps};
            [enc setComputePipelineState:pso_block_mix_];
            for (uint32_t s = 0; s < 5; ++s) [enc setBuffer:srcs[std::min(s, nsrc - 1)] offset:0 atIndex:s];
            [enc setBuffer:query.buf offset:query.offset atIndex:5];
            [enc setBuffer:(n0 ? n0->buf : buf_partial_bf16_) offset:(n0 ? n0->offset : 0) atIndex:6];
            [enc setBuffer:(n1 ? n1->buf : buf_partial_bf16_) offset:(n1 ? n1->offset : 0) atIndex:7];
            [enc setBuffer:(n2 ? n2->buf : buf_partial_bf16_) offset:(n2 ? n2->offset : 0) atIndex:8];
            [enc setBuffer:out_f32 offset:0 atIndex:9];
            [enc setBuffer:out_bf16 offset:0 atIndex:10];
            [enc setBytes:&p length:sizeof(p) atIndex:11];
            [enc dispatchThreadgroups:MTLSizeMake(1, nq, B) threadsPerThreadgroup:MTLSizeMake(288, 1, 1)];
        };

        for (size_t vi = 0; vi < weights.plan.size(); ++vi) {
            const auto& visit = weights.plan[vi];
            const auto& lw = weights.layers[visit.layer];
            bool boundary = (visit.layer % weights.cfg.residual_block_size == 0);

            if (visit.layer == weights.cfg.loop_start) {
                if (visit.pass == 0) {
                    prelude_num_blocks = num_blocks;
                    blit_copy(buf_partial_bf16_, buf_prelude_partial_, act_bytes);
                } else {
                    blit_copy(buf_res_blocks_[prelude_num_blocks + 1], buf_prev_blocks_[0], act_bytes);
                    blit_copy(buf_partial_bf16_, buf_prev_blocks_[1], act_bytes);
                    num_blocks = prelude_num_blocks;
                    blit_copy(buf_prelude_partial_, buf_partial_bf16_, act_bytes);
                }
            }

            bool has_msg = (visit.pass > 0);
            if (has_msg) {
                id<MTLBuffer> psrcs[5] = {buf_prev_blocks_[0], buf_prev_blocks_[1], buf_prev_blocks_[1], buf_prev_blocks_[1], buf_prev_blocks_[1]};
                dispatch_mix(psrcs, 2, 0u, lw.msg_queries, 4, false, nullptr, nullptr, nullptr, buf_msg_f32_, buf_norm_in_[0]);
            }

            if (num_blocks == 0) {
                [enc setComputePipelineState:pso_rms_bf16_];
                [enc setBuffer:buf_partial_bf16_ offset:0 atIndex:0];
                [enc setBuffer:lw.attn_norm[0].buf offset:lw.attn_norm[0].offset atIndex:1];
                [enc setBuffer:buf_norm_in_[0] offset:0 atIndex:2];
                [enc setBytes:&eps length:4 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(288, 1, 1)];
            } else {
                id<MTLBuffer> srcs[5];
                for (int b = 0; b < num_blocks; ++b) srcs[b] = buf_res_blocks_[b];
                srcs[num_blocks] = buf_partial_bf16_;
                uint32_t nsrc = static_cast<uint32_t>(num_blocks + 1);
                uint32_t nq = lw.is_local ? 1u : 3u;
                if (!has_msg) {
                    if (lw.is_local) {
                        dispatch_mix(srcs, nsrc, 0u, lw.attn_query, 1, true, &lw.attn_norm[0], nullptr, nullptr, buf_mix_f32_, buf_norm_in_[0]);
                    } else {
                        for (int q_i = 0; q_i < 3; ++q_i) {
                            TensorRef qsub = lw.attn_query; qsub.offset += size_t(q_i) * D * 4;
                            dispatch_mix(srcs, nsrc, 0u, qsub, 1, true, &lw.attn_norm[q_i], nullptr, nullptr, buf_mix_f32_, buf_norm_in_[q_i]);
                        }
                    }
                } else {
                    dispatch_mix(srcs, nsrc, 0u, lw.attn_query, nq, false, nullptr, nullptr, nullptr, buf_mix_f32_, buf_norm_in_[0]);
                    uint32_t istride = nq * D;
                    for (int c_i = 0; c_i < 3; ++c_i) {
                        uint32_t ioff = lw.is_local ? 0u : (static_cast<uint32_t>(c_i) * D);
                        encode_controller(enc, lw.ctrl[c_i], buf_mix_f32_, istride, ioff, buf_msg_f32_, uint32_t(c_i) * D, lw.attn_norm[c_i], buf_norm_in_[c_i], B);
                    }
                }
            }

            // Shared-weight batched projections across all B sequences!
            uint32_t q_stride;
            if (lw.is_local && !has_msg) {
                encode_linear(enc, buf_norm_in_[0], 0, false, lw.fused_q_proj, nil, 0, false, buf_q_proj_, 0, false, B, 8600, D);
                q_stride = 8600;
            } else if (lw.is_local && has_msg) {
                encode_linear(enc, buf_norm_in_[0], 0, false, lw.fused_q_proj, nil, 0, false, buf_q_proj_, 0, false, B, 6552, D);
                TensorRef wk_ref = lw.fused_q_proj; wk_ref.offset += size_t(6552) * D * 2;
                TensorRef wv_ref = lw.fused_q_proj; wv_ref.offset += size_t(7576) * D * 2;
                encode_linear(enc, buf_norm_in_[1], 0, false, wk_ref, nil, 0, false, buf_k_proj_, 0, false, B, 1024, D);
                encode_linear(enc, buf_norm_in_[2], 0, false, wv_ref, nil, 0, false, buf_v_proj_, 0, false, B, 1024, D);
                q_stride = 6552;
            } else {
                encode_linear(enc, buf_norm_in_[0], 0, false, lw.fused_q_proj, nil, 0, false, buf_q_proj_, 0, false, B, 6552, D);
                encode_linear(enc, buf_norm_in_[1], 0, false, lw.fused_k_proj, nil, 0, false, buf_k_proj_, 0, false, B, 1056, D);
                encode_linear(enc, buf_norm_in_[2], 0, false, lw.fused_v_proj, nil, 0, false, buf_v_proj_, 0, false, B, 1056, D);
                q_stride = 6552;
            }

            // Per-sequence conv + attention (each seq has its own KV & conv state)
            for (uint32_t bi = 0; bi < B; ++bi) {
                SequenceState& seq = *seqs[bi];
                uint32_t spos = static_cast<uint32_t>(seq.pos);
                uint32_t kv_cap = visit.local ? uint32_t(seq.local_kv_cap) : uint32_t(seq.max_global_seq);
                if (lw.is_local && !has_msg) {
                    encode_short_conv(enc, buf_q_proj_, size_t(bi) * 8600 * 2, 8600, 6552, lw.k_sconv_w, seq.conv_k[vi], nil, false, buf_k_conv_, size_t(bi) * 1024 * 2, 1, 1024);
                    encode_short_conv(enc, buf_q_proj_, size_t(bi) * 8600 * 2, 8600, 7576, lw.v_sconv_w, seq.conv_v[vi], nil, false, buf_v_conv_, size_t(bi) * 1024 * 2, 1, 1024);
                } else if (lw.is_local && has_msg) {
                    encode_short_conv(enc, buf_k_proj_, size_t(bi) * 1024 * 2, 1024, 0, lw.k_sconv_w, seq.conv_k[vi], nil, false, buf_k_conv_, size_t(bi) * 1024 * 2, 1, 1024);
                    encode_short_conv(enc, buf_v_proj_, size_t(bi) * 1024 * 2, 1024, 0, lw.v_sconv_w, seq.conv_v[vi], nil, false, buf_v_conv_, size_t(bi) * 1024 * 2, 1, 1024);
                } else {
                    encode_short_conv(enc, buf_k_proj_, size_t(bi) * 1056 * 2, 1056, 0, lw.k_sconv_w, seq.conv_k[vi], nil, false, buf_k_conv_, size_t(bi) * 1024 * 2, 1, 1024, true, 1056, 1024);
                    encode_short_conv(enc, buf_v_proj_, size_t(bi) * 1056 * 2, 1056, 0, lw.v_sconv_w, seq.conv_v[vi], nil, false, buf_v_conv_, size_t(bi) * 1024 * 2, 1, 1024, true, 1056, 1024);
                }

                AttnPrepParams app{1, spos, kv_cap, visit.local ? 1u : 0u, q_stride, 3072u,
                                   weights.cfg.qk_clip, weights.cfg.rel_clip, eps,
                                   visit.local ? 0.0f : weights.cfg.log_floor,
                                   visit.local ? 0.0f : weights.cfg.log_alpha};
                [enc setComputePipelineState:pso_attn_prep_qr_];
                [enc setBuffer:buf_q_proj_ offset:size_t(bi) * q_stride * 2 atIndex:0];
                [enc setBuffer:lw.q_scale.buf offset:lw.q_scale.offset atIndex:1];
                [enc setBuffer:lw.rel_scale.buf offset:lw.rel_scale.offset atIndex:2];
                [enc setBuffer:buf_q_f32_ offset:size_t(bi) * 24 * 128 * 4 atIndex:3];
                [enc setBuffer:buf_r_bf16_ offset:size_t(bi) * 24 * 16 * 2 atIndex:4];
                [enc setBytes:&app length:sizeof(app) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(24, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                [enc setComputePipelineState:pso_attn_prep_kv_];
                [enc setBuffer:buf_k_conv_ offset:size_t(bi) * 1024 * 2 atIndex:0];
                [enc setBuffer:buf_v_conv_ offset:size_t(bi) * 1024 * 2 atIndex:1];
                [enc setBuffer:seq.k_cache[vi] offset:0 atIndex:2];
                [enc setBuffer:seq.v_cache[vi] offset:0 atIndex:3];
                [enc setBytes:&app length:sizeof(app) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                [enc setComputePipelineState:pso_attn_rel_table_];
                [enc setBuffer:buf_r_bf16_ offset:size_t(bi) * 24 * 16 * 2 atIndex:0];
                [enc setBuffer:lw.rel_proj.buf offset:lw.rel_proj.offset atIndex:1];
                [enc setBuffer:buf_rel_table_ offset:size_t(bi) * 24 * 1024 * 2 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(32, 24, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                uint32_t parts = 16u;
                RelAttnParams rap{1, spos, kv_cap, visit.local ? 1u : 0u,
                                  visit.local ? uint32_t(weights.cfg.window) : 0u,
                                  parts, q_stride, 3456u, 6528u, 1.0f / std::sqrt(128.0f)};
                [enc setComputePipelineState:pso_rel_attn_split_];
                [enc setBuffer:buf_q_f32_ offset:size_t(bi) * 24 * 128 * 4 atIndex:0];
                [enc setBuffer:seq.k_cache[vi] offset:0 atIndex:1];
                [enc setBuffer:seq.v_cache[vi] offset:0 atIndex:2];
                [enc setBuffer:buf_rel_table_ offset:size_t(bi) * 24 * 1024 * 2 atIndex:3];
                [enc setBuffer:buf_q_proj_ offset:size_t(bi) * q_stride * 2 atIndex:4];
                [enc setBuffer:buf_attn_part_out_ offset:0 atIndex:5];
                [enc setBuffer:buf_attn_part_st_ offset:0 atIndex:6];
                [enc setBytes:&rap length:sizeof(rap) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(parts, 24, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                [enc setComputePipelineState:pso_rel_attn_merge_gate_];
                [enc setBuffer:buf_attn_part_out_ offset:0 atIndex:0];
                [enc setBuffer:buf_attn_part_st_ offset:0 atIndex:1];
                [enc setBuffer:buf_q_proj_ offset:size_t(bi) * q_stride * 2 atIndex:2];
                [enc setBuffer:buf_attn_out_ offset:size_t(bi) * 3072 * 2 atIndex:3];
                [enc setBytes:&rap length:sizeof(rap) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, 24, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }

            // Batched WO + Fused per-seq attn_sconv + residual update
            encode_linear(enc, buf_attn_out_, 0, false, lw.wo, nil, 0, false, buf_wo_out_, 0, false, B, D, 3072);
            bool partial_is_f32 = false;
            if (boundary) {
                blit_copy(buf_partial_bf16_, buf_res_blocks_[num_blocks], act_bytes);
                num_blocks++;
                for (uint32_t bi = 0; bi < B; ++bi) {
                    encode_short_conv(enc, buf_wo_out_, size_t(bi) * D * 2, D, 0, lw.attn_sconv_w,
                                      seqs[bi]->conv_attn[vi], nil, false,
                                      buf_partial_bf16_, size_t(bi) * D * 2, 1, D);
                }
            } else {
                ShortConvParams scp{1u, D, D, 0u, 0u, 0u, 0u, 0u};
                [enc setComputePipelineState:pso_short_conv_add_f32_];
                [enc setBuffer:lw.attn_sconv_w.buf offset:lw.attn_sconv_w.offset atIndex:1];
                [enc setBytes:&scp length:sizeof(scp) atIndex:6];
                for (uint32_t bi = 0; bi < B; ++bi) {
                    [enc setBuffer:buf_wo_out_ offset:size_t(bi) * D * 2 atIndex:0];
                    [enc setBuffer:seqs[bi]->conv_attn[vi] offset:0 atIndex:2];
                    [enc setBuffer:buf_partial_bf16_ offset:size_t(bi) * D * 2 atIndex:3];
                    [enc setBuffer:buf_partial_f32_ offset:size_t(bi) * D * 4 atIndex:4];
                    [enc setBuffer:seqs[bi]->conv_attn[vi] offset:0 atIndex:5];
                    [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                partial_is_f32 = true;
            }

            {
                id<MTLBuffer> srcs[5];
                for (int b = 0; b < num_blocks; ++b) srcs[b] = buf_res_blocks_[b];
                srcs[num_blocks] = partial_is_f32 ? buf_partial_f32_ : buf_partial_bf16_;
                uint32_t nsrc = uint32_t(num_blocks + 1);
                uint32_t f32_mask = partial_is_f32 ? (1u << num_blocks) : 0u;
                if (!has_msg) {
                    dispatch_mix(srcs, nsrc, f32_mask, lw.mlp_query, 1, true, &lw.mlp_norm, nullptr, nullptr, buf_mix_f32_, buf_ffn_in_);
                } else {
                    dispatch_mix(srcs, nsrc, f32_mask, lw.mlp_query, 1, false, nullptr, nullptr, nullptr, buf_mix_f32_, buf_ffn_in_);
                    encode_controller(enc, lw.ctrl[3], buf_mix_f32_, D, 0, buf_msg_f32_, 3u * D, lw.mlp_norm, buf_ffn_in_, B);
                }
            }

            // Coalesced MoE across ALL B sequences + Fused Reduce + MLP ShortConv + Residual Add!
            encode_moe(enc, lw, buf_ffn_in_, buf_ffn_out_, B, false);
            {
                MoEReduceConvParams rcp{1u, partial_is_f32 ? 1u : 0u, 0u};
                [enc setComputePipelineState:pso_moe_reduce_conv_add_];
                [enc setBuffer:lw.mlp_sconv_w.buf offset:lw.mlp_sconv_w.offset atIndex:1];
                [enc setBytes:&rcp length:sizeof(rcp) atIndex:6];
                for (uint32_t bi = 0; bi < B; ++bi) {
                    [enc setBuffer:buf_moe_down_ offset:size_t(bi) * 8 * D * 4 atIndex:0];
                    [enc setBuffer:seqs[bi]->conv_mlp[vi] offset:0 atIndex:2];
                    [enc setBuffer:seqs[bi]->conv_mlp[vi] offset:0 atIndex:3];
                    [enc setBuffer:(partial_is_f32 ? buf_partial_f32_ : buf_partial_bf16_)
                            offset:size_t(bi) * D * (partial_is_f32 ? 4 : 2) atIndex:4];
                    [enc setBuffer:buf_partial_bf16_ offset:size_t(bi) * D * 2 atIndex:5];
                    [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
            }
        }

        {
            id<MTLBuffer> srcs[5];
            for (int b = 0; b < num_blocks; ++b) srcs[b] = buf_res_blocks_[b];
            srcs[num_blocks] = buf_partial_bf16_;
            dispatch_mix(srcs, uint32_t(num_blocks + 1), 0u, weights.output_query, 1, true, &weights.output_norm, nullptr, nullptr, buf_mix_f32_, buf_ffn_in_);
        }

        uint32_t V = weights.cfg.unpadded_vocab;
        encode_linear(enc, buf_ffn_in_, 0, false, weights.unembed, nil, 0, false, buf_logits_, 0, true, B, V, D, weights.cfg.logit_cap);
        [enc setComputePipelineState:pso_argmax_];
        [enc setBuffer:buf_logits_ offset:0 atIndex:0];
        [enc setBuffer:buf_argmax_ offset:0 atIndex:1];
        [enc setBytes:&V length:4 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    for (auto* s : seqs) s->pos += 1;
    std::vector<int32_t> out(B);
    if (sampling && sampling->temperature > 0.0f && rng_state) {
        const uint32_t V = weights.cfg.unpadded_vocab;
        const float* base = reinterpret_cast<const float*>(buf_logits_.contents);
        for (uint32_t bi = 0; bi < B; ++bi) {
            out[bi] = sample_logits_cpu(base + size_t(bi) * V, *sampling, *rng_state);
        }
    } else {
        std::memcpy(out.data(), buf_argmax_.contents, B * sizeof(int32_t));
    }
    return out;
}

// ============================================================================
// DSpark / DFlash2 Draft Model: Context Flush + Block Proposal
// ============================================================================
void InklingEngine::draft_flush_context(SequenceState& seq) {
    if (!weights.draft.loaded || seq.pending_count == 0) return;
    ensure_global_capacity(seq, seq.draft_ctx_pos + seq.pending_count + 16);
    const uint32_t D = weights.cfg.hidden_size;
    const float eps = weights.cfg.norm_eps;
    int offset = 0;

    while (offset < seq.pending_count) {
        uint32_t chunk = static_cast<uint32_t>(std::min(64, seq.pending_count - offset));
        for (uint32_t t = 0; t < chunk; ++t) {
            for (int tp = 0; tp < 4; ++tp) {
                const uint16_t* src = seq.pending_taps.data() + ((offset + t) * 4 + tp) * D;
                uint16_t* dst = reinterpret_cast<uint16_t*>(buf_draft_ctx_taps_[tp].contents) + t * D;
                std::memcpy(dst, src, D * 2);
            }
        }

        @autoreleasepool {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

            // Compute last committed feature for markov_head (row 3 of fuse_weights)
            if (offset + static_cast<int>(chunk) == seq.pending_count) {
                float fw3[4] = {weights.draft.fuse_weights[3][0], weights.draft.fuse_weights[3][1],
                                weights.draft.fuse_weights[3][2], weights.draft.fuse_weights[3][3]};
                [enc setComputePipelineState:pso_draft_fuse_taps_];
                for (int tp = 0; tp < 4; ++tp)
                    [enc setBuffer:buf_draft_ctx_taps_[tp] offset:size_t(chunk - 1) * D * 2 atIndex:tp];
                [enc setBuffer:weights.draft.recurrent_norm.buf offset:weights.draft.recurrent_norm.offset atIndex:4];
                [enc setBuffer:seq.draft_last_feature offset:0 atIndex:5];
                [enc setBytes:fw3 length:sizeof(fw3) atIndex:6];
                [enc setBytes:&eps length:4 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(288, 1, 1)];
            }

            // Project and cache context KV for all 3 draft layers
            for (int li = 0; li < weights.draft.num_layers; ++li) {
                const auto& dl = weights.draft.layers[li];
                float fw[4] = {weights.draft.fuse_weights[li][0], weights.draft.fuse_weights[li][1],
                               weights.draft.fuse_weights[li][2], weights.draft.fuse_weights[li][3]};
                [enc setComputePipelineState:pso_draft_fuse_taps_];
                for (int tp = 0; tp < 4; ++tp) [enc setBuffer:buf_draft_ctx_taps_[tp] offset:0 atIndex:tp];
                [enc setBuffer:dl.ctx_norm.buf offset:dl.ctx_norm.offset atIndex:4];
                [enc setBuffer:buf_draft_ctx_mix_ offset:0 atIndex:5];
                [enc setBytes:fw length:sizeof(fw) atIndex:6];
                [enc setBytes:&eps length:4 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(chunk, 1, 1) threadsPerThreadgroup:MTLSizeMake(288, 1, 1)];

                encode_linear(enc, buf_draft_ctx_mix_, 0, false, dl.ctx_kv_proj, nil, 0, false,
                              buf_draft_ctx_kv_, 0, false, chunk, 2048, D);

                uint32_t kv_cap = dl.is_local ? uint32_t(seq.local_kv_cap) : uint32_t(seq.max_global_seq);
                DraftRopeParams kp{chunk, 8, 2048, 0, uint32_t(seq.draft_ctx_pos),
                                   weights.draft.rope_theta, eps, 1u, kv_cap, dl.is_local ? 1u : 0u};
                [enc setComputePipelineState:pso_draft_qk_rope_];
                [enc setBuffer:buf_draft_ctx_kv_ offset:0 atIndex:0];
                [enc setBuffer:dl.k_norm_ctx.buf offset:dl.k_norm_ctx.offset atIndex:1];
                [enc setBuffer:seq.draft_k_cache[li] offset:0 atIndex:2];
                [enc setBytes:&kp length:sizeof(kp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(8, chunk, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                DraftRopeParams vp = kp; vp.offset = 1024;
                [enc setComputePipelineState:pso_draft_v_cache_];
                [enc setBuffer:buf_draft_ctx_kv_ offset:0 atIndex:0];
                [enc setBuffer:seq.draft_v_cache[li] offset:0 atIndex:1];
                [enc setBytes:&vp length:sizeof(vp) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(8, chunk, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }

            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        seq.draft_ctx_pos += static_cast<int>(chunk);
        offset += static_cast<int>(chunk);
    }

    seq.pending_taps.clear();
    seq.pending_count = 0;
}

std::vector<int32_t> InklingEngine::draft_propose(SequenceState& seq, int32_t anchor_token, int draft_len) {
    ensure_global_capacity(seq, seq.draft_ctx_pos + seq.pending_count + 16);
    if (seq.pending_count > 64) {
        draft_flush_context(seq);
    }
    const uint32_t block_len = 8; // Always run full 8-token bidirectional block (anchor + 7 masks)
    const uint32_t D = weights.cfg.hidden_size;
    const uint32_t V = weights.cfg.unpadded_vocab;
    const float eps = weights.cfg.norm_eps;

    uint32_t flush_chunk = static_cast<uint32_t>(seq.pending_count);
    uint32_t flush_start_pos = static_cast<uint32_t>(seq.draft_ctx_pos);
    if (flush_chunk > 0) {
        for (uint32_t t = 0; t < flush_chunk; ++t) {
            for (int tp = 0; tp < 4; ++tp) {
                const uint16_t* src = seq.pending_taps.data() + (t * 4 + tp) * D;
                uint16_t* dst = reinterpret_cast<uint16_t*>(buf_draft_ctx_taps_[tp].contents) + t * D;
                std::memcpy(dst, src, D * 2);
            }
        }
        seq.draft_ctx_pos += static_cast<int>(flush_chunk);
        seq.pending_taps.clear();
        seq.pending_count = 0;
    }

    const uint32_t ctx_pos = static_cast<uint32_t>(seq.draft_ctx_pos);

    std::memcpy(buf_draft_tok_.contents, &anchor_token, 4);
    std::memset(seq.draft_rnn_state.contents, 0, 256 * 2);

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        // Fused context flush (in the SAME command buffer — zero extra CPU-GPU sync!)
        if (flush_chunk > 0) {
            float fw3[4] = {weights.draft.fuse_weights[3][0], weights.draft.fuse_weights[3][1],
                            weights.draft.fuse_weights[3][2], weights.draft.fuse_weights[3][3]};
            [enc setComputePipelineState:pso_draft_fuse_taps_];
            for (int tp = 0; tp < 4; ++tp)
                [enc setBuffer:buf_draft_ctx_taps_[tp] offset:size_t(flush_chunk - 1) * D * 2 atIndex:tp];
            [enc setBuffer:weights.draft.recurrent_norm.buf offset:weights.draft.recurrent_norm.offset atIndex:4];
            [enc setBuffer:seq.draft_last_feature offset:0 atIndex:5];
            [enc setBytes:fw3 length:sizeof(fw3) atIndex:6];
            [enc setBytes:&eps length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(288, 1, 1)];

            for (int li = 0; li < weights.draft.num_layers; ++li) {
                const auto& dl = weights.draft.layers[li];
                float fw[4] = {weights.draft.fuse_weights[li][0], weights.draft.fuse_weights[li][1],
                               weights.draft.fuse_weights[li][2], weights.draft.fuse_weights[li][3]};
                [enc setComputePipelineState:pso_draft_fuse_taps_];
                for (int tp = 0; tp < 4; ++tp) [enc setBuffer:buf_draft_ctx_taps_[tp] offset:0 atIndex:tp];
                [enc setBuffer:dl.ctx_norm.buf offset:dl.ctx_norm.offset atIndex:4];
                [enc setBuffer:buf_draft_ctx_mix_ offset:0 atIndex:5];
                [enc setBytes:fw length:sizeof(fw) atIndex:6];
                [enc setBytes:&eps length:4 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(flush_chunk, 1, 1) threadsPerThreadgroup:MTLSizeMake(288, 1, 1)];

                encode_linear(enc, buf_draft_ctx_mix_, 0, false, dl.ctx_kv_proj, nil, 0, false,
                              buf_draft_ctx_kv_, 0, false, flush_chunk, 2048, D);

                uint32_t kv_cap = dl.is_local ? uint32_t(seq.local_kv_cap) : uint32_t(seq.max_global_seq);
                DraftRopeParams kp{flush_chunk, 8, 2048, 0, flush_start_pos,
                                   weights.draft.rope_theta, eps, 1u, kv_cap, dl.is_local ? 1u : 0u};
                [enc setComputePipelineState:pso_draft_qk_rope_];
                [enc setBuffer:buf_draft_ctx_kv_ offset:0 atIndex:0];
                [enc setBuffer:dl.k_norm_ctx.buf offset:dl.k_norm_ctx.offset atIndex:1];
                [enc setBuffer:seq.draft_k_cache[li] offset:0 atIndex:2];
                [enc setBytes:&kp length:sizeof(kp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(8, flush_chunk, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                DraftRopeParams vp = kp; vp.offset = 1024;
                [enc setComputePipelineState:pso_draft_v_cache_];
                [enc setBuffer:buf_draft_ctx_kv_ offset:0 atIndex:0];
                [enc setBuffer:seq.draft_v_cache[li] offset:0 atIndex:1];
                [enc setBytes:&vp length:sizeof(vp) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(8, flush_chunk, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
        }

        // Row 0 = embed_rms(anchor_token) -> buf_draft_x_bf16_[0]
        [enc setComputePipelineState:pso_embed_rms_];
        [enc setBuffer:buf_draft_tok_ offset:0 atIndex:0];
        [enc setBuffer:weights.embed.buf offset:weights.embed.offset atIndex:1];
        [enc setBuffer:weights.embed_norm.buf offset:weights.embed_norm.offset atIndex:2];
        [enc setBuffer:buf_draft_x_bf16_ offset:0 atIndex:3];
        [enc setBytes:&eps length:4 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(288, 1, 1)];

        // Copy mask_embed to rows 1..7 via copy_bf16 (stay in ONE compute encoder!)
        uint32_t d4 = D / 4u;
        [enc setComputePipelineState:pso_copy_bf16_];
        for (uint32_t r = 1; r < block_len; ++r) {
            [enc setBuffer:weights.draft.mask_embed.buf offset:weights.draft.mask_embed.offset atIndex:0];
            [enc setBuffer:buf_draft_x_bf16_ offset:size_t(r) * D * 2 atIndex:1];
            [enc setBytes:&d4 length:4 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(d4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        bool x_is_f32 = false;
        for (int li = 0; li < weights.draft.num_layers; ++li) {
            const auto& dl = weights.draft.layers[li];
            // 1. Input RMSNorm -> buf_draft_norm_ [8, 2304] BF16
            [enc setComputePipelineState:(x_is_f32 ? pso_rms_f32_ : pso_rms_bf16_)];
            [enc setBuffer:(x_is_f32 ? buf_draft_x_f32_ : buf_draft_x_bf16_) offset:0 atIndex:0];
            [enc setBuffer:dl.input_norm.buf offset:dl.input_norm.offset atIndex:1];
            [enc setBuffer:buf_draft_norm_ offset:0 atIndex:2];
            [enc setBytes:&eps length:4 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(block_len, 1, 1) threadsPerThreadgroup:MTLSizeMake(288, 1, 1)];

            // 2. QKV [5120, 2304]
            encode_linear(enc, buf_draft_norm_, 0, false, dl.qkv_proj, nil, 0, false,
                          buf_draft_qkv_, 0, false, block_len, 5120, D);

            uint32_t kv_cap = dl.is_local ? uint32_t(seq.local_kv_cap) : uint32_t(seq.max_global_seq);
            // Q norm + RoPE -> buf_draft_q_
            DraftRopeParams qp{block_len, 24, 5120, 0, ctx_pos, weights.draft.rope_theta, eps, 0u, kv_cap, 0u};
            [enc setComputePipelineState:pso_draft_qk_rope_];
            [enc setBuffer:buf_draft_qkv_ offset:0 atIndex:0];
            [enc setBuffer:dl.q_norm.buf offset:dl.q_norm.offset atIndex:1];
            [enc setBuffer:buf_draft_q_ offset:0 atIndex:2];
            [enc setBytes:&qp length:sizeof(qp) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(24, block_len, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            // K norm + RoPE -> draft_k_cache[ctx_pos .. ctx_pos+7]
            DraftRopeParams kp{block_len, 8, 5120, 3072, ctx_pos, weights.draft.rope_theta, eps, 1u, kv_cap, dl.is_local ? 1u : 0u};
            [enc setComputePipelineState:pso_draft_qk_rope_];
            [enc setBuffer:buf_draft_qkv_ offset:0 atIndex:0];
            [enc setBuffer:dl.k_norm.buf offset:dl.k_norm.offset atIndex:1];
            [enc setBuffer:seq.draft_k_cache[li] offset:0 atIndex:2];
            [enc setBytes:&kp length:sizeof(kp) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(8, block_len, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            // V -> draft_v_cache[ctx_pos .. ctx_pos+7]
            DraftRopeParams vp = kp; vp.offset = 4096;
            [enc setComputePipelineState:pso_draft_v_cache_];
            [enc setBuffer:buf_draft_qkv_ offset:0 atIndex:0];
            [enc setBuffer:seq.draft_v_cache[li] offset:0 atIndex:1];
            [enc setBytes:&vp length:sizeof(vp) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(8, block_len, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            // 3. Bidirectional attention over [start, ctx_pos + 8)
            uint32_t end_pos = ctx_pos + block_len;
            uint32_t start_pos = (dl.is_local && ctx_pos > uint32_t(weights.draft.window))
                                     ? (ctx_pos - uint32_t(weights.draft.window)) : 0u;
            DraftAttnParams ap{block_len, start_pos, end_pos, kv_cap, dl.is_local ? 1u : 0u, 1.0f / std::sqrt(128.0f)};
            [enc setComputePipelineState:pso_draft_bidir_attn_];
            [enc setBuffer:buf_draft_q_ offset:0 atIndex:0];
            [enc setBuffer:seq.draft_k_cache[li] offset:0 atIndex:1];
            [enc setBuffer:seq.draft_v_cache[li] offset:0 atIndex:2];
            [enc setBuffer:buf_draft_attn_ offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(24, block_len, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            // 4. O proj + residual sum (draftResidualSum)
            encode_linear(enc, buf_draft_attn_, 0, false, dl.o_proj, nil, 0, false, buf_draft_o_, 0, false, block_len, D, 3072);
            uint32_t n4 = (block_len * D) / 4u;
            if (!x_is_f32) {
                [enc setComputePipelineState:pso_add_bf16_f32_];
                [enc setBuffer:buf_draft_x_bf16_ offset:0 atIndex:0];
                [enc setBuffer:buf_draft_o_ offset:0 atIndex:1];
                [enc setBuffer:buf_draft_x_f32_ offset:0 atIndex:2];
                [enc setBytes:&n4 length:4 atIndex:3];
                [enc dispatchThreads:MTLSizeMake(n4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                x_is_f32 = true;
            } else {
                [enc setComputePipelineState:pso_draft_res_add_];
                [enc setBuffer:buf_draft_x_f32_ offset:0 atIndex:0];
                [enc setBuffer:buf_draft_o_ offset:0 atIndex:1];
                [enc setBuffer:buf_draft_x_f32_ offset:0 atIndex:2];
                [enc setBytes:&n4 length:4 atIndex:3];
                [enc dispatchThreads:MTLSizeMake(n4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            // 5. FFN norm + gate_up + SwiGLU + down + residual sum
            [enc setComputePipelineState:pso_rms_f32_];
            [enc setBuffer:buf_draft_x_f32_ offset:0 atIndex:0];
            [enc setBuffer:dl.ff_norm.buf offset:dl.ff_norm.offset atIndex:1];
            [enc setBuffer:buf_draft_norm_ offset:0 atIndex:2];
            [enc setBytes:&eps length:4 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(block_len, 1, 1) threadsPerThreadgroup:MTLSizeMake(288, 1, 1)];

            encode_linear(enc, buf_draft_norm_, 0, false, dl.gate_up_proj, nil, 0, false,
                          buf_draft_gate_up_, 0, false, block_len, 18432, D);
            [enc setComputePipelineState:pso_draft_swiglu_];
            [enc setBuffer:buf_draft_gate_up_ offset:0 atIndex:0];
            [enc setBuffer:buf_draft_swiglu_ offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(9216, block_len, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            encode_linear(enc, buf_draft_swiglu_, 0, false, dl.down_proj, nil, 0, false,
                          buf_draft_o_, 0, false, block_len, D, 9216);
            [enc setComputePipelineState:pso_draft_res_add_];
            [enc setBuffer:buf_draft_x_f32_ offset:0 atIndex:0];
            [enc setBuffer:buf_draft_o_ offset:0 atIndex:1];
            [enc setBuffer:buf_draft_x_f32_ offset:0 atIndex:2];
            [enc setBytes:&n4 length:4 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(n4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        // Final draft norm -> buf_draft_x_bf16_ [8, 2304]
        [enc setComputePipelineState:pso_rms_f32_];
        [enc setBuffer:buf_draft_x_f32_ offset:0 atIndex:0];
        [enc setBuffer:weights.draft.norm.buf offset:weights.draft.norm.offset atIndex:1];
        [enc setBuffer:buf_draft_x_bf16_ offset:0 atIndex:2];
        [enc setBytes:&eps length:4 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(block_len, 1, 1) threadsPerThreadgroup:MTLSizeMake(288, 1, 1)];

        // Base logits for rows 1..draft_len (7 rows) -> buf_draft_base_logits_ [draft_len, V] BF16
        uint32_t dlen = static_cast<uint32_t>(std::clamp(draft_len, 1, 7));
        encode_linear(enc, buf_draft_x_bf16_, D * 2, false, weights.unembed, nil, 0, false,
                      buf_draft_base_logits_, 0, false, dlen, V, D, 0.0f);

        // Markov RNN sequential selection for i = 0 .. dlen-1 (parallelized across 96 TGs!)
        float lcap = weights.cfg.logit_cap;
        for (uint32_t i = 0; i < dlen; ++i) {
            // 1. Pack concat(state, markov_w1[tok], hidden[i+1], feature) -> [5120] BF16
            [enc setComputePipelineState:pso_draft_markov_pack_];
            [enc setBuffer:buf_draft_tok_ offset:size_t(i) * 4 atIndex:0];
            [enc setBuffer:weights.draft.markov_w1.buf offset:weights.draft.markov_w1.offset atIndex:1];
            [enc setBuffer:buf_draft_x_bf16_ offset:size_t(i + 1) * D * 2 atIndex:2];
            [enc setBuffer:seq.draft_last_feature offset:0 atIndex:3];
            [enc setBuffer:seq.draft_rnn_state offset:0 atIndex:4];
            [enc setBuffer:buf_draft_rnn_in_ offset:0 atIndex:5];
            [enc dispatchThreads:MTLSizeMake(5120, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            // 2. Parallel rnn_proj [768, 5120] + bias -> buf_draft_rnn_gates_ [768] BF16
            encode_linear(enc, buf_draft_rnn_in_, 0, false, weights.draft.rnn_proj_w,
                          weights.draft.rnn_proj_b.buf, weights.draft.rnn_proj_b.offset, true,
                          buf_draft_rnn_gates_, 0, false, 1, 768, 5120);

            // 3. Update state & write latent
            [enc setComputePipelineState:pso_draft_markov_update_];
            [enc setBuffer:buf_draft_rnn_gates_ offset:0 atIndex:0];
            [enc setBuffer:seq.draft_rnn_state offset:0 atIndex:1];
            [enc setBuffer:buf_draft_latent_ offset:0 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            // Folded head correction + softcap -> buf_draft_step_logits_ [V] F32
            [enc setComputePipelineState:pso_draft_folded_logits_];
            [enc setBuffer:buf_draft_base_logits_ offset:size_t(i) * V * 2 atIndex:0];
            [enc setBuffer:buf_draft_latent_ offset:0 atIndex:1];
            [enc setBuffer:buf_draft_folded_w_ offset:0 atIndex:2];
            [enc setBuffer:buf_draft_folded_b_ offset:0 atIndex:3];
            [enc setBuffer:buf_draft_step_logits_ offset:0 atIndex:4];
            [enc setBytes:&V length:4 atIndex:5];
            [enc setBytes:&lcap length:4 atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((V + 15u) / 16u, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            // Argmax -> buf_draft_tok_[i+1]
            [enc setComputePipelineState:pso_argmax_];
            [enc setBuffer:buf_draft_step_logits_ offset:0 atIndex:0];
            [enc setBuffer:buf_draft_tok_ offset:size_t(i + 1) * 4 atIndex:1];
            [enc setBytes:&V length:4 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    std::vector<int32_t> proposed(draft_len);
    const int32_t* tp = reinterpret_cast<const int32_t*>(buf_draft_tok_.contents);
    for (int i = 0; i < draft_len; ++i) proposed[i] = tp[i + 1];
    return proposed;
}

// ============================================================================
// High-level Generation (Plain B=1, Adaptive/Fixed Speculative B=1, Batch B=1..8)
// ============================================================================
std::vector<int32_t> InklingEngine::generate_plain(SequenceState& seq,
                                                   const std::vector<int32_t>& prompt_ids,
                                                   int max_new_tokens,
                                                   GenStats& stats,
                                                   bool stream_stdout,
                                                   const SamplingParams& sampling,
                                                   const TokenCallback& on_token,
                                                   bool reuse_prefix,
                                                   int sys_prefix_len,
                                                   const PrefillCallback& on_prefill,
                                                   const std::string& cache_key) {
    stats = {};
    stats.prompt_tokens = static_cast<int>(prompt_ids.size());
    uint64_t rng = sampling.seed ? sampling.seed : 42ULL;

    double t0 = now_ms();
    int cached = 0;
    if (reuse_prefix && prefix_cache.enabled()) {
        auto r = prepare_prefix_cache(seq, prompt_ids, false);
        cached = r.cached_tokens;
        stats.cache_hit_source = r.source;
        stats.cache_restore_ms = r.restore_ms;
    } else {
        reset_sequence(seq);
    }
    stats.cached_tokens = cached;
    seq.live_has_draft = false;

    const bool do_cache = reuse_prefix && prefix_cache.enabled();
    constexpr int kMilestoneStep = 4096;
    int last_saved_pos = cached;
    int next_milestone = ((cached / kMilestoneStep) + 1) * kMilestoneStep;

    int32_t next_tok = 0;
    size_t off = static_cast<size_t>(cached);
    while (off < prompt_ids.size()) {
        size_t len = std::min<size_t>(MAX_M, prompt_ids.size() - off);
        if (do_cache && sys_prefix_len > static_cast<int>(off) &&
            sys_prefix_len < static_cast<int>(prompt_ids.size()) &&
            off + len > static_cast<size_t>(sys_prefix_len)) {
            len = static_cast<size_t>(sys_prefix_len) - off;
        }
        bool is_last = (off + len == prompt_ids.size());
        std::vector<int32_t> chunk(prompt_ids.begin() + off, prompt_ids.begin() + off + len);
        auto preds = forward_target_single(seq, chunk, false, false, nullptr,
                                           is_last ? LogitsMode::LastOnly : LogitsMode::None);
        off += len;
        int cur_off = static_cast<int>(off);
        if (do_cache && sys_prefix_len > 0 && cur_off == sys_prefix_len) {
            prefix_cache.save_checkpoint(seq, weights, false, true, cache_key);
            last_saved_pos = cur_off;
        } else if (do_cache && !is_last && cur_off >= next_milestone) {
            if (cur_off - last_saved_pos >= 2048 &&
                cur_off + MAX_M < static_cast<int>(prompt_ids.size())) {
                prefix_cache.save_checkpoint(seq, weights, false, false, cache_key);
                last_saved_pos = cur_off;
            }
            next_milestone = ((cur_off / kMilestoneStep) + 1) * kMilestoneStep;
        }
        if (is_last) {
            next_tok = (sampling.temperature > 0.0f)
                           ? sample_logits_cpu(reinterpret_cast<const float*>(buf_logits_.contents), sampling, rng)
                           : preds[0];
        } else if (on_prefill && !on_prefill(cur_off, static_cast<int>(prompt_ids.size()))) {
            if (do_cache && cur_off - last_saved_pos >= 512) {
                prefix_cache.save_checkpoint(seq, weights, false, false, cache_key);
            }
            stats.prefill_ms = now_ms() - t0;
            return {};
        }
    }
    if (do_cache && static_cast<int>(prompt_ids.size()) > last_saved_pos) {
        prefix_cache.save_checkpoint(seq, weights, false, false, cache_key);
    }
    double t1 = now_ms();
    stats.prefill_ms = t1 - t0;

    std::vector<int32_t> generated;
    generated.reserve(max_new_tokens);
    if (max_new_tokens <= 0) return generated;

    generated.push_back(next_tok);
    if (stream_stdout) { printf("%s", tokenizer.decode_token(next_tok).c_str()); fflush(stdout); }
    bool keep_going = (!on_token || on_token(next_tok));

    double td0 = now_ms();
    for (int step = 1; keep_going && step < max_new_tokens; ++step) {
        if (next_tok == tokenizer.eos_id()) break;
        auto preds = forward_target_single(seq, {next_tok}, false, false, nullptr, LogitsMode::All);
        next_tok = (sampling.temperature > 0.0f)
                       ? sample_logits_cpu(reinterpret_cast<const float*>(buf_logits_.contents), sampling, rng)
                       : preds[0];
        generated.push_back(next_tok);
        if (stream_stdout) { printf("%s", tokenizer.decode_token(next_tok).c_str()); fflush(stdout); }
        if (on_token && !on_token(next_tok)) break;
    }
    double td1 = now_ms();
    stats.decode_ms = td1 - td0;
    stats.gen_tokens = static_cast<int>(generated.size());
    return generated;
}

std::vector<int32_t> InklingEngine::generate_speculative(SequenceState& seq,
                                                         const std::vector<int32_t>& prompt_ids,
                                                         int max_new_tokens,
                                                         int draft_depth,
                                                         GenStats& stats,
                                                         bool stream_stdout,
                                                         const TokenCallback& on_token,
                                                         bool reuse_prefix,
                                                         int sys_prefix_len,
                                                         const PrefillCallback& on_prefill,
                                                         const std::string& cache_key) {
    stats = {};
    stats.prompt_tokens = static_cast<int>(prompt_ids.size());
    const bool adaptive = (draft_depth <= 0);
    const int fixed_depth = std::clamp(draft_depth, 1, 7);

    // M4 Pro empirical round cost model (target_verify(1+d) + draft_propose(d), ms)
    static constexpr double kPlainMs = 31.0;
    static constexpr double kRoundMs[8] = {31.0, 43.5, 49.2, 55.0, 61.3, 67.9, 73.5, 79.1};
    // Per-position conditional acceptance EMAs P(accept_k | accept_{<k}) for k = 0..6
    // Captures DSpark's non-geometric position decay without over-drafting!
    double pos_succ[7]   = {3.6, 3.2, 2.8, 2.3, 1.9, 1.5, 1.2};
    double pos_trials[7] = {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0};
    int plain_streak = 0;

    auto choose_depth = [&](int rem) -> int {
        int max_d = std::min(7, std::max(1, rem - 1));
        if (!adaptive) return std::min(fixed_depth, max_d);
        double best_tps = 1000.0 / kPlainMs;
        int best_d = (plain_streak >= 3) ? 1 : 0; // Periodically re-probe if in plain fallback
        if (best_d == 1) best_tps = 0.0;
        double surv = 1.0;
        double exp_toks = 1.0;
        for (int d = 1; d <= max_d; ++d) {
            double pk = std::clamp(pos_succ[d - 1] / std::max(pos_trials[d - 1], 1e-6), 0.05, 0.98);
            surv *= pk;
            exp_toks += surv;
            double tps = (exp_toks * 1000.0) / kRoundMs[d];
            if (tps > best_tps) {
                best_tps = tps;
                best_d = d;
            }
        }
        return best_d;
    };

    // Prefill with 2-tier RAM+NVMe prefix cache reuse + tap capture (512-token AMX chunks + unembed ONLY last token!)
    double t0 = now_ms();
    int cached = 0;
    if (reuse_prefix && prefix_cache.enabled()) {
        auto r = prepare_prefix_cache(seq, prompt_ids, true);
        cached = r.cached_tokens;
        stats.cache_hit_source = r.source;
        stats.cache_restore_ms = r.restore_ms;
    } else {
        reset_sequence(seq);
    }
    stats.cached_tokens = cached;
    seq.live_has_draft = true;

    const bool do_cache = reuse_prefix && prefix_cache.enabled();
    constexpr int kMilestoneStep = 4096;
    int last_saved_pos = cached;
    int next_milestone = ((cached / kMilestoneStep) + 1) * kMilestoneStep;

    int32_t cur_tok = 0;
    size_t off = static_cast<size_t>(cached);
    while (off < prompt_ids.size()) {
        size_t len = std::min<size_t>(MAX_M, prompt_ids.size() - off);
        if (do_cache && sys_prefix_len > static_cast<int>(off) &&
            sys_prefix_len < static_cast<int>(prompt_ids.size()) &&
            off + len > static_cast<size_t>(sys_prefix_len)) {
            len = static_cast<size_t>(sys_prefix_len) - off;
        }
        bool is_last = (off + len == prompt_ids.size());
        std::vector<int32_t> chunk(prompt_ids.begin() + off, prompt_ids.begin() + off + len);
        auto preds = forward_target_single(seq, chunk, true, false, nullptr,
                                           is_last ? LogitsMode::LastOnly : LogitsMode::None);
        off += len;
        int cur_off = static_cast<int>(off);
        if (seq.pending_count >= 512) {
            draft_flush_context(seq);
        }
        if (do_cache && sys_prefix_len > 0 && cur_off == sys_prefix_len) {
            draft_flush_context(seq);
            prefix_cache.save_checkpoint(seq, weights, true, true, cache_key);
            last_saved_pos = cur_off;
        } else if (do_cache && !is_last && cur_off >= next_milestone) {
            if (cur_off - last_saved_pos >= 2048 &&
                cur_off + MAX_M < static_cast<int>(prompt_ids.size())) {
                draft_flush_context(seq);
                prefix_cache.save_checkpoint(seq, weights, true, false, cache_key);
                last_saved_pos = cur_off;
            }
            next_milestone = ((cur_off / kMilestoneStep) + 1) * kMilestoneStep;
        }
        if (is_last) {
            cur_tok = preds[0];
        } else if (on_prefill && !on_prefill(cur_off, static_cast<int>(prompt_ids.size()))) {
            draft_flush_context(seq);
            if (do_cache && cur_off - last_saved_pos >= 512) {
                prefix_cache.save_checkpoint(seq, weights, true, false, cache_key);
            }
            stats.prefill_ms = now_ms() - t0;
            return {};
        }
    }
    draft_flush_context(seq);
    if (do_cache && static_cast<int>(prompt_ids.size()) > last_saved_pos) {
        prefix_cache.save_checkpoint(seq, weights, true, false, cache_key);
    }
    double t1 = now_ms();
    stats.prefill_ms = t1 - t0;

    std::vector<int32_t> generated;
    generated.reserve(max_new_tokens);
    if (max_new_tokens <= 0) return generated;
    generated.push_back(cur_tok);
    if (stream_stdout) { printf("%s", tokenizer.decode_token(cur_tok).c_str()); fflush(stdout); }
    bool keep_going = (!on_token || on_token(cur_tok));

    double td0 = now_ms();
    while (keep_going && static_cast<int>(generated.size()) < max_new_tokens && cur_tok != tokenizer.eos_id()) {
        int rem = max_new_tokens - static_cast<int>(generated.size());
        int d = (rem <= 1) ? 0 : choose_depth(rem);
        if (d == 0) {
            auto preds = forward_target_single(seq, {cur_tok}, true, false, nullptr, LogitsMode::All);
            cur_tok = preds[0];
            generated.push_back(cur_tok);
            if (stream_stdout) { printf("%s", tokenizer.decode_token(cur_tok).c_str()); fflush(stdout); }
            if (on_token && !on_token(cur_tok)) break;
            plain_streak++;
            continue;
        }
        plain_streak = 0;

        // 1. Draft d tokens
        auto drafts = draft_propose(seq, cur_tok, d);

        // 2. Build verify block: [cur_tok, drafts[0..d-1]]
        std::vector<int32_t> block;
        block.reserve(1 + d);
        block.push_back(cur_tok);
        for (int i = 0; i < d; ++i) block.push_back(drafts[i]);

        int prev_pos = seq.pos;
        int prev_pending = seq.pending_count;

        // 3. Target verify forward on (1 + d) tokens with conv window capture for zero-cost rollback!
        auto target_preds = forward_target_single(seq, block, true, true, nullptr, LogitsMode::All);

        // 4. Greedy verification: compare target_preds[i] vs drafts[i]
        int accepted = 0;
        while (accepted < d && target_preds[accepted] == drafts[accepted]) {
            accepted++;
        }
        int32_t bonus_tok = target_preds[accepted];

        // Update per-position conditional acceptance EMAs (exact censoring at rejection!)
        if (adaptive) {
            constexpr double decay = 0.88;
            for (int k = 0; k < accepted; ++k) {
                pos_succ[k]   = pos_succ[k] * decay + 1.0;
                pos_trials[k] = pos_trials[k] * decay + 1.0;
            }
            if (accepted < d) {
                pos_succ[accepted]   = pos_succ[accepted] * decay;
                pos_trials[accepted] = pos_trials[accepted] * decay + 1.0;
            }
        }

        // Emit accepted draft tokens + bonus token
        bool hit_eos = false;
        for (int i = 0; i < accepted; ++i) {
            if (static_cast<int>(generated.size()) < max_new_tokens) {
                generated.push_back(drafts[i]);
                if (stream_stdout) { printf("%s", tokenizer.decode_token(drafts[i]).c_str()); fflush(stdout); }
                if (on_token && !on_token(drafts[i])) { hit_eos = true; break; }
                if (drafts[i] == tokenizer.eos_id()) { hit_eos = true; break; }
            }
        }
        if (!hit_eos && static_cast<int>(generated.size()) < max_new_tokens) {
            generated.push_back(bonus_tok);
            if (stream_stdout) { printf("%s", tokenizer.decode_token(bonus_tok).c_str()); fflush(stdout); }
            if (on_token && !on_token(bonus_tok)) hit_eos = true;
            if (bonus_tok == tokenizer.eos_id()) hit_eos = true;
        }

        stats.spec_rounds++;
        stats.spec_drafted += d;
        stats.spec_accepted += accepted;

        // 5. Commit exactly (accepted + 1) tokens of target state via GPU conv rollback!
        int committed_len = accepted + 1;
        if (committed_len < 1 + d) {
            seq.pending_count = prev_pending + committed_len;
            seq.pending_taps.resize(size_t(seq.pending_count) * 4 * weights.cfg.hidden_size);
            rollback_target(seq, prev_pos, committed_len, 1 + d);
        }

        cur_tok = bonus_tok;
        if (hit_eos) break;
    }
    double td1 = now_ms();
    stats.decode_ms = td1 - td0;
    stats.gen_tokens = static_cast<int>(generated.size());
    return generated;
}

std::vector<std::vector<int32_t>> InklingEngine::generate_batch(const std::vector<std::vector<int32_t>>& prompts,
                                                                int max_new_tokens,
                                                                GenStats& stats,
                                                                const SamplingParams& sampling) {
    const size_t B = prompts.size();
    std::vector<SequenceState> seqs;
    seqs.reserve(B);
    for (size_t b = 0; b < B; ++b) seqs.push_back(create_sequence(262144));
    uint64_t rng = sampling.seed ? sampling.seed : 42ULL;

    stats = {};
    double t0 = now_ms();
    std::vector<int32_t> cur_toks(B, 0);
    for (size_t b = 0; b < B; ++b) {
        stats.prompt_tokens += static_cast<int>(prompts[b].size());
        for (size_t off = 0; off < prompts[b].size(); off += MAX_M) {
            size_t len = std::min<size_t>(MAX_M, prompts[b].size() - off);
            bool is_last = (off + len == prompts[b].size());
            std::vector<int32_t> chunk(prompts[b].begin() + off, prompts[b].begin() + off + len);
            auto preds = forward_target_single(seqs[b], chunk, false, false, nullptr,
                                               is_last ? LogitsMode::LastOnly : LogitsMode::None);
            if (is_last) {
                cur_toks[b] = (sampling.temperature > 0.0f)
                                  ? sample_logits_cpu(reinterpret_cast<const float*>(buf_logits_.contents), sampling, rng)
                                  : preds[0];
            }
        }
    }
    double t1 = now_ms();
    stats.prefill_ms = t1 - t0;

    std::vector<std::vector<int32_t>> outputs(B);
    std::vector<SequenceState*> ptrs(B);
    for (size_t b = 0; b < B; ++b) {
        ptrs[b] = &seqs[b];
        if (max_new_tokens > 0) outputs[b].push_back(cur_toks[b]);
    }

    double td0 = now_ms();
    for (int step = 1; step < max_new_tokens; ++step) {
        cur_toks = forward_target_batch(ptrs, cur_toks, &sampling, &rng);
        for (size_t b = 0; b < B; ++b) outputs[b].push_back(cur_toks[b]);
    }
    double td1 = now_ms();
    stats.decode_ms = td1 - td0;
    stats.gen_tokens = static_cast<int>(B * max_new_tokens);
    return outputs;
}
