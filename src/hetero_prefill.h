#pragma once
#import <Metal/Metal.h>
#include "model_weights.h"
#include <cstdint>

struct MoEWorkTileHost {
    uint16_t u;
    uint16_t base;
    uint16_t batch;
    uint16_t pad;
};

struct MoERouteTableHost {
    uint32_t num_unique;
    uint32_t unique_experts[128];
    uint32_t expert_counts[128];
    uint32_t expert_tokens[128 * 1024];
    uint32_t expert_slots[128 * 1024];
    float expert_weights[128 * 1024];
    uint8_t top_ids[1024 * 8];
    float top_ws[1024 * 8];
    uint32_t num_work_tiles;
    MoEWorkTileHost work_tiles[512];
    uint32_t gpu_tiles_w13;
    uint32_t gpu_tiles_w2;
};

struct HeteroTimingStats {
    double qkv_wait_ms = 0.0, qkv_sme_ms = 0.0;
    double wo_wait_ms  = 0.0, wo_sme_ms  = 0.0;
    double w13_wait_ms = 0.0, w13_sme_ms = 0.0;
    double w2_wait_ms  = 0.0, w2_sme_ms  = 0.0;
    double final_gpu_wait_ms = 0.0;
    void reset() { *this = HeteroTimingStats{}; }
};

// Pure User-Space Heterogeneous Prefill Accelerator (Apple M4 Pro Metal GPU + ARMv9 SME2):
//   - Zero CoreML, zero IOSurface, zero extra weight RAM (reads MTLBuffer.contents directly in unified memory!)
//   1. Dense BF16 QKV & WO (M >= 256): Metal GPU AMX + native BF16->FP32 SME2 (svmopa_za32_bf16_m)
//   2. NVFP4 MoE W13 & W2  (M >= 256): Metal GPU AMX + FP16->FP32 SME2 (svmopa_za32_f16_m + NEON NVFP4 unpack)
class HeteroPrefill {
public:
    HeteroPrefill();
    ~HeteroPrefill();
    HeteroPrefill(const HeteroPrefill&) = delete;
    HeteroPrefill& operator=(const HeteroPrefill&) = delete;

    bool init(bool enable_dense_sme2 = true, bool enable_moe_sme2 = true);
    void disable() { ready_ = false; dense_ready_ = false; moe_ready_ = false; }

    bool ready() const { return ready_; }
    bool dense_ready() const { return dense_ready_; }
    bool moe_ready() const { return moe_ready_; }

    // Dense BF16 GEMM: Y[M, N] = X[M, K] @ W[N, K]^T on 10-core ARMv9 SME2 (BF16->FP32->BF16).
    void run_sme2_bf16_gemm(const uint16_t* x_bf16,
                            const uint16_t* w_bf16,
                            uint16_t* y_bf16,
                            uint32_t M, uint32_t N, uint32_t K) const;

    // NVFP4 MoE W13 work_tiles[tile_start .. tile_end) on 10-core ARMv9 SME2 CPU (parallel via GCD).
    void run_sme2_moe_w13(const LayerWeights& lw,
                          const void* act_h, const float* act_amax,
                          const MoERouteTableHost* tbl,
                          uint32_t tile_start, uint32_t tile_end,
                          uint16_t* mid_out,
                          float gate_cap, float up_cap) const;

    // NVFP4 MoE W2 work_tiles[tile_start .. tile_end) on 10-core ARMv9 SME2 CPU (parallel via GCD).
    void run_sme2_moe_w2(const LayerWeights& lw,
                         const void* mid_h, const float* mid_amax,
                         const MoERouteTableHost* tbl,
                         uint32_t tile_start, uint32_t tile_end,
                         float* down_out) const;

    int dense_qkv_gpu_pct = 62; // % of QKV tokens on Metal GPU (rest on SME2 BF16)
    int dense_wo_gpu_pct  = 70; // % of WO tokens on Metal GPU (rest on SME2 BF16)
    int moe_w13_gpu_pct   = 66; // % of MoE W13 token-work on Metal GPU (rest on SME2 NVFP4)
    int moe_w2_gpu_pct    = 62; // % of MoE W2 token-work on Metal GPU (rest on SME2 NVFP4)
    mutable HeteroTimingStats stats;

private:
    bool ready_ = false;
    bool dense_ready_ = false;
    bool moe_ready_ = false;
    uint16_t* xp_scratch_ = nullptr; // Reusable 6 MB packed-X scratch (allocated once on init)
};
