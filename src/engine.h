#pragma once
#import <Metal/Metal.h>
#include "model_weights.h"
#include "tokenizer.h"
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct SequenceState {
    int pos = 0;
    int max_global_seq = 0;     // Currently allocated global KV capacity (grows dynamically)
    int max_seq_limit = 262144; // Hard maximum context length (default 256K)
    int local_kv_cap = 1088;    // 1024 + 64 headroom for speculative rollback

    // Per-visit (48) caches:
    // Local visits: K/V [8, 1088, 128] BF16
    // Global visits: K/V [8, max_global_seq, 128] BF16
    std::vector<id<MTLBuffer>> k_cache;
    std::vector<id<MTLBuffer>> v_cache;

    // 4 conv states per visit: k_sconv[3,1024], v_sconv[3,1024], attn_sconv[3,2304], mlp_sconv[3,2304]
    // Plus 4 conv windows [11, C] for speculative rollback!
    std::vector<id<MTLBuffer>> conv_k, conv_v, conv_attn, conv_mlp;
    std::vector<id<MTLBuffer>> win_k, win_v, win_attn, win_mlp;

    // Draft KV caches (3 layers: L0 & L2 ring 1088, L1 linear max_global_seq)
    int draft_ctx_pos = 0;
    std::vector<id<MTLBuffer>> draft_k_cache;
    std::vector<id<MTLBuffer>> draft_v_cache;
    id<MTLBuffer> draft_last_feature = nil; // [2304] BF16 (last committed target feature for markov head)
    id<MTLBuffer> draft_rnn_state = nil;    // [256] BF16

    // Buffered target tap features awaiting flush to draft KV cache
    // Each entry: 4 taps [4, 2304] BF16
    std::vector<uint16_t> pending_taps;
    int pending_count = 0;
};

struct GenStats {
    int prompt_tokens = 0;
    int gen_tokens = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    int spec_rounds = 0;
    int spec_drafted = 0;
    int spec_accepted = 0;
    double avg_unique_experts = 0.0;
};

enum class LogitsMode {
    None,     // Prefill intermediate chunk: skip unembed + argmax
    LastOnly, // Prefill final chunk: unembed only last token (M=1)
    All       // Decode / speculative verify (M <= 8): unembed all M tokens
};

struct SamplingParams {
    float temperature = 0.0f; // 0.0 = greedy (GPU argmax)
    float top_p = 1.0f;
    uint64_t seed = 42;
};

class InklingEngine {
public:
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;

    InklingWeights weights;
    InklingTokenizer tokenizer;

    bool init(const std::string& manifest_path,
              const std::string& blobs_dir,
              const std::string& metal_path,
              bool load_draft = true,
              int max_seq = 262144);

    SequenceState create_sequence(int max_global_seq = 262144);
    bool ensure_global_capacity(SequenceState& seq, int needed_tokens);
    void reset_sequence(SequenceState& seq);

    // Run target model forward on M tokens (1 <= M <= 32) for ONE sequence starting at seq.pos.
    // Advances seq.pos by M (caller can rollback with rollback_target).
    // Returns argmax token IDs according to logits_mode (All -> M ids, LastOnly -> 1 id, None -> empty).
    std::vector<int32_t> forward_target_single(SequenceState& seq,
                                               const std::vector<int32_t>& tokens,
                                               bool capture_taps = false,
                                               bool save_conv_windows = false,
                                               std::vector<float>* last_logits_out = nullptr,
                                               LogitsMode logits_mode = LogitsMode::All);

    // Roll back speculative target state after accepting `accepted` tokens out of `proposed_M` (1 <= accepted <= proposed_M).
    void rollback_target(SequenceState& seq, int prev_pos, int accepted, int proposed_M);

    // Batched decode: B sequences (1 <= B <= 8), 1 token per sequence, sharing ALL weight reads!
    std::vector<int32_t> forward_target_batch(std::vector<SequenceState*>& seqs,
                                              const std::vector<int32_t>& tokens,
                                              const SamplingParams* sampling = nullptr,
                                              uint64_t* rng_state = nullptr);

    // Flush buffered target taps into draft KV cache
    void draft_flush_context(SequenceState& seq);

    // Propose `draft_len` (1..7) speculative tokens given anchor token ID
    std::vector<int32_t> draft_propose(SequenceState& seq, int32_t anchor_token, int draft_len = 7);

    // Optional per-token streaming callback; return false to stop generation early.
    using TokenCallback = std::function<bool(int32_t)>;

    // High-level generation helpers
    std::vector<int32_t> generate_plain(SequenceState& seq,
                                        const std::vector<int32_t>& prompt_ids,
                                        int max_new_tokens,
                                        GenStats& stats,
                                        bool stream_stdout = false,
                                        const SamplingParams& sampling = {},
                                        const TokenCallback& on_token = nullptr);

    // draft_depth == 0 enables online adaptive speculative depth controller; 1..7 fixes depth.
    std::vector<int32_t> generate_speculative(SequenceState& seq,
                                              const std::vector<int32_t>& prompt_ids,
                                              int max_new_tokens,
                                              int draft_depth,
                                              GenStats& stats,
                                              bool stream_stdout = false,
                                              const TokenCallback& on_token = nullptr);

    // Multi-sequence parallel low-batch generation (B = 1..8)
    std::vector<std::vector<int32_t>> generate_batch(const std::vector<std::vector<int32_t>>& prompts,
                                                     int max_new_tokens,
                                                     GenStats& stats,
                                                     const SamplingParams& sampling = {});

private:
    // Pipeline states
    id<MTLComputePipelineState> pso_copy_bf16_ = nil;
    id<MTLComputePipelineState> pso_embed_rms_ = nil;
    id<MTLComputePipelineState> pso_rms_bf16_ = nil;
    id<MTLComputePipelineState> pso_rms_f32_ = nil;
    id<MTLComputePipelineState> pso_block_mix_ = nil;
    id<MTLComputePipelineState> pso_ctrl_norm_concat_ = nil;
    id<MTLComputePipelineState> pso_ctrl_mix_rms_ = nil;
    id<MTLComputePipelineState> pso_linear_m_[9] = {}; // 1..8 specialized
    id<MTLComputePipelineState> pso_linear_k64_ = nil;
    id<MTLComputePipelineState> pso_linear_tiled_ = nil;
    id<MTLComputePipelineState> pso_short_conv_ = nil;
    id<MTLComputePipelineState> pso_short_conv_add_f32_ = nil;
    id<MTLComputePipelineState> pso_short_conv_rollback_ = nil;
    id<MTLComputePipelineState> pso_attn_prep_qr_ = nil;
    id<MTLComputePipelineState> pso_attn_prep_kv_ = nil;
    id<MTLComputePipelineState> pso_attn_rel_table_ = nil;
    id<MTLComputePipelineState> pso_rel_attn_split_ = nil;
    id<MTLComputePipelineState> pso_rel_attn_merge_gate_ = nil;
    id<MTLComputePipelineState> pso_add_bf16_f32_ = nil;
    id<MTLComputePipelineState> pso_add_f32_bf16_bf16_ = nil;
    id<MTLComputePipelineState> pso_add_bf16_bf16_bf16_ = nil;
    id<MTLComputePipelineState> pso_moe_topk_ = nil;
    id<MTLComputePipelineState> pso_nvfp4_quant_act_ = nil;
    id<MTLComputePipelineState> pso_mxfp8_quant_act_ = nil;
    id<MTLComputePipelineState> pso_moe_w13_nvfp4_m1_ = nil;
    id<MTLComputePipelineState> pso_moe_w13_nvfp4_ = nil;
    id<MTLComputePipelineState> pso_moe_w2_nvfp4_ = nil;
    id<MTLComputePipelineState> pso_moe_w13_mxfp8_ = nil;
    id<MTLComputePipelineState> pso_moe_w2_mxfp8_ = nil;
    id<MTLComputePipelineState> pso_moe_reduce_ = nil;
    id<MTLComputePipelineState> pso_moe_reduce_conv_add_ = nil;
    id<MTLComputePipelineState> pso_argmax_ = nil;

    // Draft pipelines
    id<MTLComputePipelineState> pso_draft_fuse_taps_ = nil;
    id<MTLComputePipelineState> pso_draft_qk_rope_ = nil;
    id<MTLComputePipelineState> pso_draft_v_cache_ = nil;
    id<MTLComputePipelineState> pso_draft_bidir_attn_ = nil;
    id<MTLComputePipelineState> pso_draft_swiglu_ = nil;
    id<MTLComputePipelineState> pso_draft_res_add_ = nil;
    id<MTLComputePipelineState> pso_draft_markov_pack_ = nil;
    id<MTLComputePipelineState> pso_draft_markov_update_ = nil;
    id<MTLComputePipelineState> pso_draft_folded_logits_ = nil;
    id<MTLComputePipelineState> pso_precompute_folded_ = nil;

    // Reusable scratch GPU buffers (sized for max M=32 prefill, decode/verify M<=8!)
    static constexpr int MAX_M = 32;
    static constexpr int MAX_LOGITS_M = 8;
    id<MTLBuffer> buf_tok_ids_ = nil;
    id<MTLBuffer> buf_argmax_ = nil;

    // Residual stream slots: up to 5 blocks + partial + prelude_partial + prev[2] + f32_partial
    id<MTLBuffer> buf_res_blocks_[5] = {};  // each [MAX_M, 2304] BF16
    id<MTLBuffer> buf_partial_bf16_ = nil;  // [MAX_M, 2304] BF16
    id<MTLBuffer> buf_partial_f32_ = nil;   // [MAX_M, 2304] F32
    id<MTLBuffer> buf_prelude_partial_ = nil; // [MAX_M, 2304] BF16
    id<MTLBuffer> buf_prev_blocks_[2] = {}; // [MAX_M, 2304] BF16
    id<MTLBuffer> buf_taps_[4] = {};        // [MAX_M, 2304] BF16

    // Layer scratch
    id<MTLBuffer> buf_mix_f32_ = nil;       // [MAX_M, 4, 2304] F32
    id<MTLBuffer> buf_msg_f32_ = nil;       // [MAX_M, 4, 2304] F32
    id<MTLBuffer> buf_norm_in_[3] = {};     // [MAX_M, 2304] BF16 (q_in, k_in, v_in)
    id<MTLBuffer> buf_ffn_in_ = nil;        // [MAX_M, 2304] BF16
    id<MTLBuffer> buf_ctrl_concat_ = nil;   // [MAX_M, 4608] BF16
    id<MTLBuffer> buf_ctrl_down_ = nil;     // [MAX_M, 64] F32
    id<MTLBuffer> buf_ctrl_gf_ = nil;       // [MAX_M, 4608] F32

    id<MTLBuffer> buf_q_proj_ = nil;        // [MAX_M, 8592] BF16
    id<MTLBuffer> buf_k_proj_ = nil;        // [MAX_M, 1056] BF16
    id<MTLBuffer> buf_v_proj_ = nil;        // [MAX_M, 1056] BF16
    id<MTLBuffer> buf_k_conv_ = nil;        // [MAX_M, 1024] BF16
    id<MTLBuffer> buf_v_conv_ = nil;        // [MAX_M, 1024] BF16
    id<MTLBuffer> buf_q_f32_ = nil;         // [MAX_M, 24, 128] F32
    id<MTLBuffer> buf_r_bf16_ = nil;        // [MAX_M, 24, 16] BF16
    id<MTLBuffer> buf_rel_table_ = nil;     // [MAX_M, 24, 1024] BF16
    id<MTLBuffer> buf_attn_part_out_ = nil; // [MAX_M * 24, 64, 128] F32
    id<MTLBuffer> buf_attn_part_st_ = nil;  // [MAX_M * 24, 64, 2] F32
    id<MTLBuffer> buf_attn_out_ = nil;      // [MAX_M, 3072] BF16
    id<MTLBuffer> buf_wo_out_ = nil;        // [MAX_M, 2304] BF16
    id<MTLBuffer> buf_sconv_out_ = nil;     // [MAX_M, 2304] BF16

    // MoE scratch
    id<MTLBuffer> buf_router_logits_ = nil; // [MAX_M, 128] F32
    id<MTLBuffer> buf_route_table_ = nil;
    id<MTLBuffer> buf_ind_w13_ = nil;
    id<MTLBuffer> buf_ind_w2_ = nil;
    id<MTLBuffer> buf_act_q_ = nil;         // [MAX_M, 2304] F32
    id<MTLBuffer> buf_act_amax_ = nil;      // [MAX_M] F32
    id<MTLBuffer> buf_moe_mid_ = nil;       // [MAX_M * 8, 1152] BF16
    id<MTLBuffer> buf_mid_q_ = nil;         // [MAX_M * 8, 1152] F32
    id<MTLBuffer> buf_mid_amax_ = nil;      // [MAX_M * 8] F32
    id<MTLBuffer> buf_moe_down_ = nil;      // [MAX_M * 8, 2304] F32
    id<MTLBuffer> buf_ffn_out_ = nil;       // [MAX_M, 2304] BF16

    // Output logits
    id<MTLBuffer> buf_logits_ = nil;        // [MAX_M, 200058] F32

    // Draft scratch
    id<MTLBuffer> buf_draft_folded_w_ = nil; // [200058, 256] BF16
    id<MTLBuffer> buf_draft_folded_b_ = nil; // [200058] BF16
    id<MTLBuffer> buf_draft_ctx_taps_[4] = {}; // [64, 2304] BF16 for context flush
    id<MTLBuffer> buf_draft_ctx_mix_ = nil;    // [64, 2304] BF16
    id<MTLBuffer> buf_draft_ctx_kv_ = nil;     // [64, 2048] BF16
    id<MTLBuffer> buf_draft_x_f32_ = nil;      // [8, 2304] F32
    id<MTLBuffer> buf_draft_x_bf16_ = nil;     // [8, 2304] BF16
    id<MTLBuffer> buf_draft_norm_ = nil;       // [8, 2304] BF16
    id<MTLBuffer> buf_draft_qkv_ = nil;        // [8, 5120] BF16
    id<MTLBuffer> buf_draft_q_ = nil;          // [8, 24, 128] BF16
    id<MTLBuffer> buf_draft_attn_ = nil;       // [8, 3072] BF16
    id<MTLBuffer> buf_draft_o_ = nil;          // [8, 2304] BF16
    id<MTLBuffer> buf_draft_gate_up_ = nil;    // [8, 18432] BF16
    id<MTLBuffer> buf_draft_swiglu_ = nil;     // [8, 9216] BF16
    id<MTLBuffer> buf_draft_base_logits_ = nil; // [7, 200058] BF16
    id<MTLBuffer> buf_draft_step_logits_ = nil; // [200058] F32
    id<MTLBuffer> buf_draft_latent_ = nil;      // [256] BF16
    id<MTLBuffer> buf_draft_rnn_in_ = nil;      // [5120] BF16
    id<MTLBuffer> buf_draft_rnn_gates_ = nil;   // [768] BF16
    id<MTLBuffer> buf_draft_tok_ = nil;         // [8] int32

    void encode_linear(id<MTLComputeCommandEncoder> enc,
                       id<MTLBuffer> x, size_t x_off, bool x_is_f32,
                       const TensorRef& w,
                       id<MTLBuffer> bias, size_t bias_off, bool has_bias,
                       id<MTLBuffer> y, size_t y_off, bool out_f32,
                       uint32_t M, uint32_t N, uint32_t K, float softcap = 0.0f);

    void encode_short_conv(id<MTLComputeCommandEncoder> enc,
                           id<MTLBuffer> x, size_t x_off, uint32_t x_stride, uint32_t x_el_off,
                           const TensorRef& w,
                           id<MTLBuffer> state,
                           id<MTLBuffer> win, bool save_win,
                           id<MTLBuffer> out, size_t out_off,
                           uint32_t M, uint32_t C,
                           bool is_dynamic = false, uint32_t dyn_stride = 0, uint32_t dyn_el_off = 0);

    void encode_controller(id<MTLComputeCommandEncoder> enc,
                           const ControllerWeights& ctrl,
                           id<MTLBuffer> inp, uint32_t inp_stride, uint32_t inp_off,
                           id<MTLBuffer> msg, uint32_t msg_off,
                           const TensorRef& norm_w,
                           id<MTLBuffer> out_bf16,
                           uint32_t M);

    void encode_moe(id<MTLComputeCommandEncoder> enc,
                    const LayerWeights& lw,
                    id<MTLBuffer> ffn_in,
                    id<MTLBuffer> ffn_out,
                    uint32_t M,
                    bool do_reduce = true);

    int32_t sample_logits_cpu(const float* logits,
                              const SamplingParams& sampling,
                              uint64_t& rng_state) const;

    void precompute_draft_folded();
};
