#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#include "hetero_prefill.h"

#include <arm_neon.h>
#if defined(__ARM_FEATURE_SME)
#include <arm_sme.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

static inline uint16_t f32_to_bf16_s(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

static inline float bf16_to_f32_s(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Fast 4x4 uint32 transpose (4 rows x 4 uint32 pairs of 16-bit elems -> 4 K2-steps x 4 rows)
static inline void transpose4x4_u32(uint32x4_t r0, uint32x4_t r1, uint32x4_t r2, uint32x4_t r3,
                                    uint32x4_t& c0, uint32x4_t& c1, uint32x4_t& c2, uint32x4_t& c3) {
    uint32x4_t t0 = vtrn1q_u32(r0, r1);
    uint32x4_t t1 = vtrn2q_u32(r0, r1);
    uint32x4_t t2 = vtrn1q_u32(r2, r3);
    uint32x4_t t3 = vtrn2q_u32(r2, r3);
    c0 = vreinterpretq_u32_u64(vzip1q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
    c1 = vreinterpretq_u32_u64(vzip1q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
    c2 = vreinterpretq_u32_u64(vzip2q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
    c3 = vreinterpretq_u32_u64(vzip2q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
}

// Pack a [valid_rows, bk] column slice starting at col k0 (stride K) into SME2 panel [bk/2, 2, 16, 2].
static void pack_32_16b_slice(const uint16_t* __restrict src, uint32_t valid_rows,
                              uint32_t K, uint32_t k0, uint32_t bk,
                              uint16_t* __restrict dst) {
    uint32_t* dst32 = reinterpret_cast<uint32_t*>(dst);
    const uint32_t k0_2 = k0 >> 1;
    const uint32_t bk2 = bk >> 1;

    if (valid_rows == 32u) {
        for (uint32_t half_id = 0; half_id < 2u; ++half_id) {
            for (uint32_t g = 0; g < 4u; ++g) {
                uint32_t r0 = half_id * 16u + g * 4u;
                const uint32_t* p0 = reinterpret_cast<const uint32_t*>(src + size_t(r0 + 0u) * K) + k0_2;
                const uint32_t* p1 = reinterpret_cast<const uint32_t*>(src + size_t(r0 + 1u) * K) + k0_2;
                const uint32_t* p2 = reinterpret_cast<const uint32_t*>(src + size_t(r0 + 2u) * K) + k0_2;
                const uint32_t* p3 = reinterpret_cast<const uint32_t*>(src + size_t(r0 + 3u) * K) + k0_2;
                uint32_t* d = dst32 + half_id * 16u + g * 4u;
                uint32_t k2 = 0;
                for (; k2 + 4u <= bk2; k2 += 4u) {
                    uint32x4_t c0, c1, c2, c3;
                    transpose4x4_u32(vld1q_u32(p0 + k2), vld1q_u32(p1 + k2),
                                     vld1q_u32(p2 + k2), vld1q_u32(p3 + k2),
                                     c0, c1, c2, c3);
                    vst1q_u32(d + (k2 + 0u) * 32u, c0);
                    vst1q_u32(d + (k2 + 1u) * 32u, c1);
                    vst1q_u32(d + (k2 + 2u) * 32u, c2);
                    vst1q_u32(d + (k2 + 3u) * 32u, c3);
                }
                for (; k2 < bk2; ++k2) {
                    d[k2 * 32u + 0u] = p0[k2];
                    d[k2 * 32u + 1u] = p1[k2];
                    d[k2 * 32u + 2u] = p2[k2];
                    d[k2 * 32u + 3u] = p3[k2];
                }
            }
        }
        return;
    }

    for (uint32_t r = 0; r < 32u; ++r) {
        uint32_t half_id = r >> 4;
        uint32_t lr = r & 15u;
        uint32_t* d = dst32 + half_id * 16u + lr;
        if (r < valid_rows) {
            const uint32_t* p = reinterpret_cast<const uint32_t*>(src + size_t(r) * K) + k0_2;
            for (uint32_t k2 = 0; k2 < bk2; ++k2) d[k2 * 32u] = p[k2];
        } else {
            for (uint32_t k2 = 0; k2 < bk2; ++k2) d[k2 * 32u] = 0u;
        }
    }
}

static void pack_32_16b_rows(const uint16_t* __restrict src, uint32_t valid_rows, uint32_t K,
                             uint16_t* __restrict dst) {
    pack_32_16b_slice(src, valid_rows, K, 0u, K, dst);
}

// ============================================================================
// ARMv9 SME2 Microkernels (Native BF16->FP32 & FP16->FP32 on Apple M4 Pro!)
// ============================================================================
#if defined(__ARM_FEATURE_SME)

__arm_new("za") __arm_locally_streaming
static void sme2_bf16_tile32x32_acc(const __bf16* __restrict ap, const __bf16* __restrict bp,
                                    float* __restrict c32, int BK, bool zero_init) {
    svbool_t pg = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    if (zero_init) {
        svzero_za();
    } else {
        for (int r = 0; r < 16; ++r) {
            svld1_hor_za32(0, r, pg32, c32 + r * 32);
            svld1_hor_za32(1, r, pg32, c32 + r * 32 + 16);
            svld1_hor_za32(2, r, pg32, c32 + (16 + r) * 32);
            svld1_hor_za32(3, r, pg32, c32 + (16 + r) * 32 + 16);
        }
    }
    for (int k = 0; k < BK; k += 4) {
        svbfloat16_t a0 = svld1_bf16(pg, ap);
        svbfloat16_t a1 = svld1_bf16(pg, ap + 32);
        svbfloat16_t b0 = svld1_bf16(pg, bp);
        svbfloat16_t b1 = svld1_bf16(pg, bp + 32);
        svmopa_za32_bf16_m(0, pg, pg, a0, b0);
        svmopa_za32_bf16_m(1, pg, pg, a0, b1);
        svmopa_za32_bf16_m(2, pg, pg, a1, b0);
        svmopa_za32_bf16_m(3, pg, pg, a1, b1);

        svbfloat16_t a2 = svld1_bf16(pg, ap + 64);
        svbfloat16_t a3 = svld1_bf16(pg, ap + 96);
        svbfloat16_t b2 = svld1_bf16(pg, bp + 64);
        svbfloat16_t b3 = svld1_bf16(pg, bp + 96);
        svmopa_za32_bf16_m(0, pg, pg, a2, b2);
        svmopa_za32_bf16_m(1, pg, pg, a2, b3);
        svmopa_za32_bf16_m(2, pg, pg, a3, b2);
        svmopa_za32_bf16_m(3, pg, pg, a3, b3);
        ap += 128;
        bp += 128;
    }
    for (int r = 0; r < 16; ++r) {
        svst1_hor_za32(0, r, pg32, c32 + r * 32);
        svst1_hor_za32(1, r, pg32, c32 + r * 32 + 16);
        svst1_hor_za32(2, r, pg32, c32 + (16 + r) * 32);
        svst1_hor_za32(3, r, pg32, c32 + (16 + r) * 32 + 16);
    }
}

__arm_new("za") __arm_locally_streaming
static void sme2_f16_tile32x32_acc(const __fp16* __restrict ap, const __fp16* __restrict bp,
                                   float* __restrict c32, int BK, bool zero_init) {
    svbool_t pg = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    if (zero_init) {
        svzero_za();
    } else {
        for (int r = 0; r < 16; ++r) {
            svld1_hor_za32(0, r, pg32, c32 + r * 32);
            svld1_hor_za32(1, r, pg32, c32 + r * 32 + 16);
            svld1_hor_za32(2, r, pg32, c32 + (16 + r) * 32);
            svld1_hor_za32(3, r, pg32, c32 + (16 + r) * 32 + 16);
        }
    }
    for (int k = 0; k < BK; k += 4) {
        svfloat16_t a0 = svld1_f16(pg, ap);
        svfloat16_t a1 = svld1_f16(pg, ap + 32);
        svfloat16_t b0 = svld1_f16(pg, bp);
        svfloat16_t b1 = svld1_f16(pg, bp + 32);
        svmopa_za32_f16_m(0, pg, pg, a0, b0);
        svmopa_za32_f16_m(1, pg, pg, a0, b1);
        svmopa_za32_f16_m(2, pg, pg, a1, b0);
        svmopa_za32_f16_m(3, pg, pg, a1, b1);

        svfloat16_t a2 = svld1_f16(pg, ap + 64);
        svfloat16_t a3 = svld1_f16(pg, ap + 96);
        svfloat16_t b2 = svld1_f16(pg, bp + 64);
        svfloat16_t b3 = svld1_f16(pg, bp + 96);
        svmopa_za32_f16_m(0, pg, pg, a2, b2);
        svmopa_za32_f16_m(1, pg, pg, a2, b3);
        svmopa_za32_f16_m(2, pg, pg, a3, b2);
        svmopa_za32_f16_m(3, pg, pg, a3, b3);
        ap += 128;
        bp += 128;
    }
    for (int r = 0; r < 16; ++r) {
        svst1_hor_za32(0, r, pg32, c32 + r * 32);
        svst1_hor_za32(1, r, pg32, c32 + r * 32 + 16);
        svst1_hor_za32(2, r, pg32, c32 + (16 + r) * 32);
        svst1_hor_za32(3, r, pg32, c32 + (16 + r) * 32 + 16);
    }
}

#else
static void sme2_bf16_tile32x32_acc(const __bf16*, const __bf16*, float*, int, bool) {}
static void sme2_f16_tile32x32_acc(const __fp16*, const __fp16*, float*, int, bool) {}
#endif

// ============================================================================
// ARMv9 SME2 NVFP4 MoE Microkernels (NEON 4x4 Transpose Unpack -> Coalesced!)
// ============================================================================
static inline float fp8_e4m3_cpu(uint8_t bits) {
    uint32_t mag = bits & 127u;
    uint32_t exp = mag >> 3;
    uint32_t mant = mag & 7u;
    float v = (exp == 0u) ? (float(mant) * (1.0f / 512.0f))
                          : std::ldexp(1.0f + float(mant) * 0.125f, int(exp) - 7);
    return (bits & 128u) ? -v : v;
}

static const int8_t kE2M1_x2[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12
};
static __fp16 kScaleHalf[128];
static bool kLutReady = false;

static void init_nvfp4_lut() {
    if (kLutReady) return;
    for (int s = 0; s < 128; ++s) {
        kScaleHalf[s] = (__fp16)(fp8_e4m3_cpu(uint8_t(s)) * 0.5f);
    }
    kLutReady = true;
}

static inline void unpack_16_nvfp4_reg(uint64_t qw, __fp16 sc_half, int8x16_t tbl,
                                       uint32x4_t& lo_out, uint32x4_t& hi_out) {
    uint8x8_t raw = vcreate_u8(qw);
    uint8x8_t lo = vand_u8(raw, vdup_n_u8(0x0F));
    uint8x8_t hi = vshr_n_u8(raw, 4);
    uint8x8x2_t z = vzip_u8(lo, hi);
    uint8x16_t nibs = vcombine_u8(z.val[0], z.val[1]);
    int8x16_t q8 = vqtbl1q_s8(tbl, nibs);
    float16x8_t f_lo = vmulq_n_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(q8))), sc_half);
    float16x8_t f_hi = vmulq_n_f16(vcvtq_f16_s16(vmovl_high_s8(q8)), sc_half);
    lo_out = vreinterpretq_u32_f16(f_lo);
    hi_out = vreinterpretq_u32_f16(f_hi);
}

template<int K_DIM, bool IS_W2>
static void run_sme2_nvfp4_tile(
    const __fp16* act_h, const float* act_amax,
    const uint32_t* w_packed, const uint8_t* w_scales, const float* w_amax,
    const MoERouteTableHost* tbl, const MoEWorkTileHost& wt,
    void* out_raw, float outer_denom, float gate_cap, float up_cap)
{
    constexpr int N = 2304;
    constexpr int NBLKS = K_DIM / 16;
    constexpr int W_U32_STRIDE = K_DIM / 8;
    constexpr int K2 = K_DIM / 2;

    const uint32_t expert = wt.u;
    const uint32_t base = wt.base;
    const uint32_t batch = wt.batch;
    if (batch == 0u) return;

    constexpr int BK_BLKS = 32; // 32 NVFP4 blocks = 512 K-elems -> 32 KB Bp (100% L1D resident!)
    alignas(64) __fp16 Ap[2][K_DIM * 32];
    alignas(64) __fp16 Bp[512 * 32];
    alignas(64) float C0[32 * 32], C1[32 * 32];

    const uint32_t* toks = &tbl->expert_tokens[size_t(expert) * 1024u + base];
    const uint32_t* slots = &tbl->expert_slots[size_t(expert) * 1024u + base];
    const float* rws = &tbl->expert_weights[size_t(expert) * 1024u + base];

    // Fast NEON 4x4 transpose pack of Ap[0..npass-1]
    const int npass = (batch > 32u) ? 2 : 1;
    for (int p = 0; p < npass; ++p) {
        uint32_t* dst32 = reinterpret_cast<uint32_t*>(Ap[p]);
        const uint32_t p_base = uint32_t(p * 32);
        const uint32_t p_valid = (batch > p_base) ? std::min<uint32_t>(32u, batch - p_base) : 0u;
        const uint32_t* row_ptrs[32];
        for (uint32_t r = 0; r < 32u; ++r) {
            if (r < p_valid) {
                uint32_t j = p_base + r;
                uint32_t row_idx = IS_W2 ? (toks[j] * 8u + slots[j]) : toks[j];
                row_ptrs[r] = reinterpret_cast<const uint32_t*>(act_h + size_t(row_idx) * K_DIM);
            } else {
                row_ptrs[r] = nullptr;
            }
        }
        uint32x4_t z4 = vdupq_n_u32(0u);
        for (uint32_t half_id = 0; half_id < 2u; ++half_id) {
            for (uint32_t g = 0; g < 4u; ++g) {
                uint32_t r0 = half_id * 16u + g * 4u;
                const uint32_t* p0 = row_ptrs[r0 + 0u];
                const uint32_t* p1 = row_ptrs[r0 + 1u];
                const uint32_t* p2 = row_ptrs[r0 + 2u];
                const uint32_t* p3 = row_ptrs[r0 + 3u];
                uint32_t* d = dst32 + half_id * 16u + g * 4u;
                if (p0 && p1 && p2 && p3) {
                    for (int k2 = 0; k2 < K2; k2 += 4) {
                        uint32x4_t c0, c1, c2, c3;
                        transpose4x4_u32(vld1q_u32(p0 + k2), vld1q_u32(p1 + k2),
                                         vld1q_u32(p2 + k2), vld1q_u32(p3 + k2),
                                         c0, c1, c2, c3);
                        vst1q_u32(d + (k2 + 0) * 32, c0);
                        vst1q_u32(d + (k2 + 1) * 32, c1);
                        vst1q_u32(d + (k2 + 2) * 32, c2);
                        vst1q_u32(d + (k2 + 3) * 32, c3);
                    }
                } else {
                    for (int k2 = 0; k2 < K2; k2 += 4) {
                        uint32x4_t c0, c1, c2, c3;
                        transpose4x4_u32(p0 ? vld1q_u32(p0 + k2) : z4,
                                         p1 ? vld1q_u32(p1 + k2) : z4,
                                         p2 ? vld1q_u32(p2 + k2) : z4,
                                         p3 ? vld1q_u32(p3 + k2) : z4,
                                         c0, c1, c2, c3);
                        vst1q_u32(d + (k2 + 0) * 32, c0);
                        vst1q_u32(d + (k2 + 1) * 32, c1);
                        vst1q_u32(d + (k2 + 2) * 32, c2);
                        vst1q_u32(d + (k2 + 3) * 32, c3);
                    }
                }
            }
        }
    }

    float wmax = w_amax[expert];
    float inv_denom = 1.0f / (outer_denom * 1536.0f);
    float outers[64];
    for (uint32_t j = 0; j < batch; ++j) {
        uint32_t ri = IS_W2 ? (toks[j] * 8u + slots[j]) : toks[j];
        outers[j] = act_amax[ri] * wmax * inv_denom;
    }

    const int8x16_t e2m1_tbl = vld1q_s8(kE2M1_x2);
    uint32_t* bp32 = reinterpret_cast<uint32_t*>(Bp);

    for (int n0 = 0; n0 < N; n0 += 32) {
        for (int blk0 = 0; blk0 < NBLKS; blk0 += BK_BLKS) {
            int nblk = std::min(BK_BLKS, NBLKS - blk0);
            int bk = nblk * 16;
            bool first_k = (blk0 == 0);
            // Coalesced NEON 4-row-at-a-time NVFP4 unpack + 4x4 transpose into L1D-resident 32 KB Bp!
            for (int half_id = 0; half_id < 2; ++half_id) {
                for (int g = 0; g < 4; ++g) {
                    int r0 = half_id * 16 + g * 4;
                    const uint64_t* wr0 = reinterpret_cast<const uint64_t*>(w_packed + (size_t(expert) * N + n0 + r0 + 0) * W_U32_STRIDE) + blk0;
                    const uint64_t* wr1 = reinterpret_cast<const uint64_t*>(w_packed + (size_t(expert) * N + n0 + r0 + 1) * W_U32_STRIDE) + blk0;
                    const uint64_t* wr2 = reinterpret_cast<const uint64_t*>(w_packed + (size_t(expert) * N + n0 + r0 + 2) * W_U32_STRIDE) + blk0;
                    const uint64_t* wr3 = reinterpret_cast<const uint64_t*>(w_packed + (size_t(expert) * N + n0 + r0 + 3) * W_U32_STRIDE) + blk0;
                    const uint8_t* sr0 = w_scales + (size_t(expert) * N + n0 + r0 + 0) * NBLKS + blk0;
                    const uint8_t* sr1 = w_scales + (size_t(expert) * N + n0 + r0 + 1) * NBLKS + blk0;
                    const uint8_t* sr2 = w_scales + (size_t(expert) * N + n0 + r0 + 2) * NBLKS + blk0;
                    const uint8_t* sr3 = w_scales + (size_t(expert) * N + n0 + r0 + 3) * NBLKS + blk0;
                    uint32_t* d = bp32 + half_id * 16 + g * 4;

                    for (int b = 0; b < nblk; ++b) {
                        uint32x4_t l0, h0, l1, h1, l2, h2, l3, h3;
                        unpack_16_nvfp4_reg(wr0[b], kScaleHalf[sr0[b] & 127u], e2m1_tbl, l0, h0);
                        unpack_16_nvfp4_reg(wr1[b], kScaleHalf[sr1[b] & 127u], e2m1_tbl, l1, h1);
                        unpack_16_nvfp4_reg(wr2[b], kScaleHalf[sr2[b] & 127u], e2m1_tbl, l2, h2);
                        unpack_16_nvfp4_reg(wr3[b], kScaleHalf[sr3[b] & 127u], e2m1_tbl, l3, h3);

                        uint32x4_t c0, c1, c2, c3;
                        transpose4x4_u32(l0, l1, l2, l3, c0, c1, c2, c3);
                        int kb = b * 8;
                        vst1q_u32(d + (kb + 0) * 32, c0);
                        vst1q_u32(d + (kb + 1) * 32, c1);
                        vst1q_u32(d + (kb + 2) * 32, c2);
                        vst1q_u32(d + (kb + 3) * 32, c3);

                        transpose4x4_u32(h0, h1, h2, h3, c0, c1, c2, c3);
                        vst1q_u32(d + (kb + 4) * 32, c0);
                        vst1q_u32(d + (kb + 5) * 32, c1);
                        vst1q_u32(d + (kb + 6) * 32, c2);
                        vst1q_u32(d + (kb + 7) * 32, c3);
                    }
                }
            }
            size_t ap_off = size_t(blk0) * 16u * 32u;
            sme2_f16_tile32x32_acc(Ap[0] + ap_off, Bp, C0, bk, first_k);
            if (npass == 2) sme2_f16_tile32x32_acc(Ap[1] + ap_off, Bp, C1, bk, first_k);
        }

        if constexpr (!IS_W2) {
            uint16_t* mid_out = reinterpret_cast<uint16_t*>(out_raw);
            int pair_base = n0 / 2;
            for (uint32_t j = 0; j < batch; ++j) {
                const float* cr = (j < 32u) ? (C0 + j * 32) : (C1 + (j - 32u) * 32);
                uint16_t* dst = mid_out + size_t(toks[j] * 8u + slots[j]) * 1152u + pair_base;
                float sc = outers[j], rw = rws[j];
                for (int pl = 0; pl < 16; ++pl) {
                    float gv = bf16_to_f32_s(f32_to_bf16_s(cr[pl * 2 + 0] * sc));
                    float uv = bf16_to_f32_s(f32_to_bf16_s(cr[pl * 2 + 1] * sc));
                    float av = (gate_cap * std::tanh(gv / gate_cap)) * (1.0f / (1.0f + std::exp(-gv))) *
                               (up_cap * std::tanh(uv / up_cap)) * rw;
                    dst[pl] = f32_to_bf16_s(av);
                }
            }
        } else {
            float* down_out = reinterpret_cast<float*>(out_raw);
            for (uint32_t j = 0; j < batch; ++j) {
                const float* cr = (j < 32u) ? (C0 + j * 32) : (C1 + (j - 32u) * 32);
                uint32_t ri = toks[j] * 8u + slots[j];
                float* dst = down_out + size_t(ri) * 2304u + n0;
                float sc = outers[j];
                for (int cl = 0; cl < 32; ++cl) {
                    dst[cl] = bf16_to_f32_s(f32_to_bf16_s(cr[cl] * sc));
                }
            }
        }
    }
}

} // namespace

HeteroPrefill::HeteroPrefill() = default;
HeteroPrefill::~HeteroPrefill() {
    free(xp_scratch_);
    xp_scratch_ = nullptr;
}

bool HeteroPrefill::init(bool enable_dense_sme2, bool enable_moe_sme2) {
    init_nvfp4_lut();
#if defined(__ARM_FEATURE_SME)
    dense_ready_ = enable_dense_sme2;
    moe_ready_ = enable_moe_sme2;
#else
    (void)enable_dense_sme2;
    (void)enable_moe_sme2;
    dense_ready_ = false;
    moe_ready_ = false;
#endif
    if (dense_ready_ && !xp_scratch_) {
        // Reusable 64B-aligned 6 MB buffer for packed X (up to 1024 x 3072 BF16) — zero per-call malloc!
        if (posix_memalign(reinterpret_cast<void**>(&xp_scratch_), 64, size_t(1024u) * 3072u * sizeof(uint16_t)) != 0) {
            xp_scratch_ = nullptr;
            dense_ready_ = false;
        }
    }
    ready_ = (dense_ready_ || moe_ready_);
    fprintf(stderr, "[hetero] Pure User-Space ARMv9 SME2 ready | Dense BF16=%s (QKV=%d%%, WO=%d%%) | MoE NVFP4=%s (W13=%d%%, W2=%d%%)\n",
            dense_ready_ ? "ON" : "OFF", dense_qkv_gpu_pct, dense_wo_gpu_pct,
            moe_ready_ ? "ON" : "OFF", moe_w13_gpu_pct, moe_w2_gpu_pct);
    return ready_;
}

void HeteroPrefill::run_sme2_bf16_gemm(const uint16_t* x_bf16,
                                       const uint16_t* w_bf16,
                                       uint16_t* y_bf16,
                                       uint32_t M, uint32_t N, uint32_t K) const {
    if (!dense_ready_ || !xp_scratch_ || M == 0u || N == 0u || K == 0u) return;

    const uint32_t m_tiles = (M + 31u) / 32u;
    const uint32_t n_tiles = (N + 31u) / 32u;

    uint16_t* xp_ptr = xp_scratch_;
    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    dispatch_apply(m_tiles, q, ^(size_t mt) {
        uint32_t m0 = static_cast<uint32_t>(mt) * 32u;
        uint32_t vm = std::min<uint32_t>(32u, M - m0);
        pack_32_16b_rows(x_bf16 + size_t(m0) * K, vm, K, xp_ptr + mt * 32u * K);
    });

    // Sweep N-tiles in parallel with L1D-resident BK=512 (32 KB Bp stays 100% hot in L1D across all m_tiles!)
    constexpr uint32_t BK = 512u;
    const uint32_t n_chunks = (n_tiles + 1u) / 2u;
    dispatch_apply(n_chunks, q, ^(size_t nc) {
        alignas(64) uint16_t bp_buf[32u * BK];
        alignas(64) float C_all[16u * 32u * 32u];
        uint32_t nt_start = static_cast<uint32_t>(nc) * 2u;
        uint32_t nt_end = std::min<uint32_t>(nt_start + 2u, n_tiles);

        for (uint32_t nt = nt_start; nt < nt_end; ++nt) {
            uint32_t n0 = nt * 32u;
            uint32_t vn = std::min<uint32_t>(32u, N - n0);
            const uint16_t* w_base = w_bf16 + size_t(n0) * K;

            for (uint32_t mt_base = 0; mt_base < m_tiles; mt_base += 16u) {
                uint32_t mt_cnt = std::min<uint32_t>(16u, m_tiles - mt_base);
                for (uint32_t k0 = 0; k0 < K; k0 += BK) {
                    uint32_t bk = std::min<uint32_t>(BK, K - k0);
                    pack_32_16b_slice(w_base, vn, K, k0, bk, bp_buf);
                    const __bf16* bp = reinterpret_cast<const __bf16*>(bp_buf);
                    bool first_k = (k0 == 0u);
                    for (uint32_t j = 0; j < mt_cnt; ++j) {
                        const __bf16* ap = reinterpret_cast<const __bf16*>(
                            xp_ptr + size_t(mt_base + j) * 32u * K + size_t(k0) * 32u);
                        sme2_bf16_tile32x32_acc(ap, bp, C_all + j * 1024u, static_cast<int>(bk), first_k);
                    }
                }

                uint32x4_t rnd_bias = vdupq_n_u32(0x7FFFu);
                uint32x4_t one = vdupq_n_u32(1u);
                for (uint32_t j = 0; j < mt_cnt; ++j) {
                    uint32_t m0 = (mt_base + j) * 32u;
                    uint32_t vm = std::min<uint32_t>(32u, M - m0);
                    const float* C = C_all + j * 1024u;
                    if (vn == 32u) {
                        for (uint32_t r = 0; r < vm; ++r) {
                            const float* cr = C + r * 32u;
                            uint16_t* dst = y_bf16 + size_t(m0 + r) * N + n0;
                            for (uint32_t c = 0; c < 32u; c += 8u) {
                                uint32x4_t u0 = vreinterpretq_u32_f32(vld1q_f32(cr + c));
                                uint32x4_t u1 = vreinterpretq_u32_f32(vld1q_f32(cr + c + 4u));
                                u0 = vaddq_u32(u0, vaddq_u32(rnd_bias, vandq_u32(vshrq_n_u32(u0, 16), one)));
                                u1 = vaddq_u32(u1, vaddq_u32(rnd_bias, vandq_u32(vshrq_n_u32(u1, 16), one)));
                                vst1q_u16(dst + c, vcombine_u16(vshrn_n_u32(u0, 16), vshrn_n_u32(u1, 16)));
                            }
                        }
                    } else {
                        for (uint32_t r = 0; r < vm; ++r) {
                            const float* cr = C + r * 32u;
                            uint16_t* dst = y_bf16 + size_t(m0 + r) * N + n0;
                            for (uint32_t c = 0; c < vn; ++c) dst[c] = f32_to_bf16_s(cr[c]);
                        }
                    }
                }
            }
        }
    });
}

void HeteroPrefill::run_sme2_moe_w13(const LayerWeights& lw,
                                     const void* act_h, const float* act_amax,
                                     const MoERouteTableHost* tbl,
                                     uint32_t tile_start, uint32_t tile_end,
                                     uint16_t* mid_out,
                                     float gate_cap, float up_cap) const {
    if (!moe_ready_ || tile_start >= tile_end) return;
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const uint8_t*>(lw.w13.weight.buf.contents) + lw.w13.weight.offset);
    const uint8_t* ws = reinterpret_cast<const uint8_t*>(lw.w13.scales.buf.contents) + lw.w13.scales.offset;
    const float* wa = reinterpret_cast<const float*>(
        reinterpret_cast<const uint8_t*>(lw.w13.amax.buf.contents) + lw.w13.amax.offset);
    const float od = lw.w13.outer_denom;
    const __fp16* ah = reinterpret_cast<const __fp16*>(act_h);
    uint32_t cnt = tile_end - tile_start;
    dispatch_apply(cnt, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t idx) {
        run_sme2_nvfp4_tile<2304, false>(ah, act_amax, wp, ws, wa, tbl, tbl->work_tiles[tile_start + idx],
                                         mid_out, od, gate_cap, up_cap);
    });
}

void HeteroPrefill::run_sme2_moe_w2(const LayerWeights& lw,
                                    const void* mid_h, const float* mid_amax,
                                    const MoERouteTableHost* tbl,
                                    uint32_t tile_start, uint32_t tile_end,
                                    float* down_out) const {
    if (!moe_ready_ || tile_start >= tile_end) return;
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const uint8_t*>(lw.w2.weight.buf.contents) + lw.w2.weight.offset);
    const uint8_t* ws = reinterpret_cast<const uint8_t*>(lw.w2.scales.buf.contents) + lw.w2.scales.offset;
    const float* wa = reinterpret_cast<const float*>(
        reinterpret_cast<const uint8_t*>(lw.w2.amax.buf.contents) + lw.w2.amax.offset);
    const float od = lw.w2.outer_denom;
    const __fp16* mh = reinterpret_cast<const __fp16*>(mid_h);
    uint32_t cnt = tile_end - tile_start;
    dispatch_apply(cnt, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t idx) {
        run_sme2_nvfp4_tile<1152, true>(mh, mid_amax, wp, ws, wa, tbl, tbl->work_tiles[tile_start + idx],
                                        down_out, od, 0.0f, 0.0f);
    });
}
