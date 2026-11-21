#pragma once
#import <Metal/Metal.h>
#include <cstdint>
#include <string>
#include <vector>

struct TensorRef {
    id<MTLBuffer> buf = nil;
    size_t offset = 0;
    size_t nbytes = 0;
    std::vector<int64_t> shape;
    std::string dtype;
};

struct MappedBlob {
    void* ptr = nullptr;
    size_t mapped_len = 0;
    id<MTLBuffer> buf = nil;
};

struct ExpertBankWeights {
    bool is_nvfp4 = true; // false = mxfp8
    TensorRef weight;      // U32 [128, rows, cols/8] (nvfp4) or [128, rows, cols/4] (mxfp8)
    TensorRef scales;      // U8  [128, rows, cols/16] (nvfp4) or [128, rows, cols/32] (mxfp8)
    TensorRef amax;        // F32 [128] (nvfp4 only)
    float outer_denom = 2688.0f; // F32 scalar (nvfp4 only)
};

struct ControllerWeights {
    TensorRef down;        // BF16 [64, 4608]
    TensorRef gain_forget; // Fused BF16 [4608, 64] (gain[2304,64] stacked with forget[2304,64])
};

struct LayerWeights {
    bool is_local = true;

    // Norms
    TensorRef attn_norm[3];   // BF16 [2304] (local: all 3 point to same attn_norm)
    TensorRef mlp_norm;       // BF16 [2304]

    // Residual attention queries
    TensorRef attn_query;     // F32 [2304] (local, layer>0) or [3, 2304] (global)
    TensorRef mlp_query;      // F32 [2304]

    // Fused Attention input projections:
    // Local:  qkv_r_g_s = [wq(3072), wk(1024), wv(1024), wr(384), gate(3072), sink(24)] -> [8592, 2304] BF16
    // Global: q_r_g_s   = [wq(3072), wr(384), gate(3072), sink(24)] -> [6552, 2304] BF16
    //         k_proj    = [wk(1024), k_sconv_dyn(32)] -> [1056, 2304] BF16
    //         v_proj    = [wv(1024), v_sconv_dyn(32)] -> [1056, 2304] BF16
    TensorRef fused_q_proj;
    TensorRef fused_k_proj;   // global only
    TensorRef fused_v_proj;   // global only

    TensorRef wo;             // BF16 [2304, 3072]
    TensorRef q_scale;        // BF16 [24, 128]
    TensorRef rel_scale;      // BF16 [24, 16]
    TensorRef rel_proj;       // BF16 [16, 1024]

    // Short conv weights (always [4, C] BF16)
    TensorRef k_sconv_w;      // BF16 [4, 1024]
    TensorRef v_sconv_w;      // BF16 [4, 1024]
    TensorRef attn_sconv_w;   // BF16 [4, 2304]
    TensorRef mlp_sconv_w;    // BF16 [4, 2304]

    // MoE
    TensorRef router_w;       // BF16 [128, 2304]
    TensorRef router_bias;    // F32  [128]
    ExpertBankWeights w13;
    ExpertBankWeights w2;

    // Loop core (layers 6..17 only)
    bool has_controllers = false;
    TensorRef msg_queries;         // F32 [4, 2304] ([q, k, v, mlp])
    ControllerWeights ctrl[4];     // [attn, k, v, mlp]
};

struct DraftLayerWeights {
    bool is_local = true;
    TensorRef input_norm;     // BF16 [2304]
    TensorRef ff_norm;        // BF16 [2304]
    TensorRef ctx_norm;       // BF16 [2304]
    TensorRef q_norm;         // BF16 [128]
    TensorRef k_norm;         // BF16 [128]
    TensorRef k_norm_ctx;     // BF16 [128]
    TensorRef qkv_proj;       // Fused BF16 [5120, 2304] (q:3072, k:1024, v:1024)
    TensorRef ctx_kv_proj;    // Fused BF16 [2048, 2304] (k_ctx:1024, v_ctx:1024)
    TensorRef o_proj;         // BF16 [2304, 3072]
    TensorRef gate_up_proj;   // Fused BF16 [18432, 2304] (gate:9216, up:9216)
    TensorRef down_proj;      // BF16 [2304, 9216]
};

struct DraftWeights {
    bool loaded = false;
    int num_layers = 3;
    int block_size = 8;
    int state_width = 256;
    int window = 1024;
    float rope_theta = 1000000.0f;
    std::vector<int> taps; // [5, 23, 41, 48]

    float fuse_weights[4][4]; // softmaxed F32 [4, 4]
    TensorRef mask_embed;     // BF16 [2304]
    TensorRef norm;           // BF16 [2304]
    TensorRef recurrent_norm; // BF16 [2304]
    TensorRef markov_w1;      // BF16 [201024, 256]
    TensorRef rnn_proj_w;     // BF16 [768, 5120]
    TensorRef rnn_proj_b;     // BF16 [768]
    TensorRef folded_w;       // BF16 [200058, 256] = unembed[200058,2304] @ hidden_out_w[2304,256]
    TensorRef folded_b;       // BF16 [200058]      = unembed[200058,2304] @ hidden_out_b[2304]
    std::vector<DraftLayerWeights> layers;
};

struct ModelConfig {
    int num_layers = 24;
    int hidden_size = 2304;
    int vocab_size = 201024;
    int unpadded_vocab = 200058;
    int model_max_length = 1048576;
    int head_dim = 128;
    int heads = 24;
    int kv_heads = 8;
    int rel_dim = 16;
    int rel_range = 1024;
    int window = 1024;
    int conv_taps = 4;
    int experts = 128;
    int top_k = 8;
    int expert_width = 1152;
    int residual_block_size = 6;
    int loop_start = 6;
    int loop_end = 18;
    int loop_passes = 3;
    int gate_rank = 64;
    float norm_eps = 1e-6f;
    float qk_clip = 22.627417f;
    float rel_clip = 8.0f;
    float log_floor = 8192.0f;
    float log_alpha = 0.1f;
    float route_scale = 8.0f;
    float gate_cap = 4.0f;
    float up_cap = 25.0f;
    float logit_cap = 30.0f;
};

struct LayerVisit {
    int layer;
    int pass;
    bool local;
};

class InklingWeights {
public:
    ModelConfig cfg;
    std::vector<LayerVisit> plan; // 48 visits
    std::string tokenizer_path;

    TensorRef embed;        // BF16 [201024, 2304]
    TensorRef embed_norm;   // BF16 [2304]
    TensorRef output_norm;  // BF16 [2304]
    TensorRef output_query; // F32  [2304]
    TensorRef unembed;      // BF16 [201024, 2304] (first 200058 rows used)

    std::vector<LayerWeights> layers; // 24 physical layers
    DraftWeights draft;

    bool load_from_ollama_manifest(id<MTLDevice> device,
                                   const std::string& manifest_path,
                                   const std::string& blobs_dir,
                                   bool load_draft = true);
    ~InklingWeights();

private:
    std::vector<MappedBlob> mapped_blobs_;
    std::vector<id<MTLBuffer>> owned_buffers_;
};
