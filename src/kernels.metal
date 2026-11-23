#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ============================================================================
// Inkling 2 Mini — M4 Pro (applegpu_g16s) Specialized Metal Kernels (v2)
// Target: 240+ GB/s sustained DRAM bandwidth at low batch sizes (M = 1..8)
// ============================================================================

// Pure SIMD ALU E2M1 decode (ZERO LUT, ZERO branches, exact bit-pattern in half4 scaled by 2^-14!)
// Caller folds * 16384.0f into the block scale ONCE per 16 weights!
inline float4 decode4_e2m1_scaled(uint w16) {
    ushort4 nibs = ushort4(w16 & 0xFu, (w16 >> 4) & 0xFu, (w16 >> 8) & 0xFu, (w16 >> 12) & 0xFu);
    ushort4 fp16 = ((nibs & ushort4(8u)) << 12) | ((nibs & ushort4(7u)) << 9);
    return float4(as_type<half4>(fp16));
}

// Pure FP16 unscaled E2M1 decode (exact ±0, 0.5, 1, 1.5, 2, 3, 4, 6 in half4; no subnormals, fast-math safe!)
inline half4 decode4_e2m1_unscaled_h(uint w16) {
    ushort4 nibs = ushort4(w16 & 0xFu, (w16 >> 4) & 0xFu, (w16 >> 8) & 0xFu, (w16 >> 12) & 0xFu);
    ushort4 mag = nibs & ushort4(7u);
    ushort4 sgn = (nibs & ushort4(8u)) << 12;
    ushort4 body = select(ushort4(0x3800u) + select(mag << 9, ushort4(0u), mag == ushort4(1u)),
                          ushort4(0u), mag == ushort4(0u));
    return as_type<half4>(sgn | body);
}

// Pure SIMD ALU FP8 E4M3FN decode (ZERO branches, scaled by 2^-8; fold * 256.0f into scale ONCE!)
inline float4 decode4_fp8_scaled(uint w) {
    ushort4 b = ushort4(w & 0xFFu, (w >> 8) & 0xFFu, (w >> 16) & 0xFFu, (w >> 24) & 0xFFu);
    ushort4 fp16 = ((b & ushort4(128u)) << 8) | ((b & ushort4(127u)) << 7);
    return float4(as_type<half4>(fp16));
}

inline float fp8_e4m3(uchar bits) {
    uint mag = bits & 127u;
    ushort sign = ushort(bits & 128u) << 8;
    if (mag == 127u) return float(as_type<half>(ushort(sign | 0x7f80u)));
    return float(as_type<half>(ushort(sign | (mag << 7)))) * 256.0f;
}

inline float mxfp8_e8m0(uchar bits) {
    if (bits == 0u) return as_type<float>(0x00800000u) * 0.5f;
    return as_type<float>(uint(bits) << 23);
}

inline float softcap_fn(float x, float cap) { return cap * tanh(x / cap); }
inline float sigmoid_fn(float x) { return 1.0f / (1.0f + exp(-x)); }

inline float round_e4m3(float x) {
    uint exponent = (as_type<uint>(x) >> 23) & 255u;
    float spacing = as_type<float>((max(exponent, 121u) - 3u) << 23);
    return rint(x / spacing) * spacing;
}

inline float round_e2m1(float x) {
    float m = abs(x);
    float q = m > 0.25f ? 0.5f : 0.0f;
    if (m >= 0.75f) q = 1.0f;
    if (m > 1.25f) q = 1.5f;
    if (m >= 1.75f) q = 2.0f;
    if (m > 2.5f) q = 3.0f;
    if (m >= 3.5f) q = 4.0f;
    if (m > 5.0f) q = 6.0f;
    return x < 0.0f ? -q : q;
}

inline uchar to_fp8_e4m3_signed(float x) {
    uint bits = as_type<uint>(x);
    uchar sign = uchar((bits >> 24) & 0x80u);
    float a = min(abs(x), 448.0f);
    if (a == 0.0f) return sign;
    uint abits = as_type<uint>(a);
    int exp = int((abits >> 23) & 0xFFu) - 127;
    uint mant = abits & 0x7FFFFFu;
    if (exp < -6) {
        uint m = min(uint(rint(a * 512.0f)), 7u);
        return sign | uchar(m);
    }
    int e4 = exp + 7;
    uint top = mant >> 20;
    uint rem = mant & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (top & 1u))) {
        top += 1;
        if (top == 8u) { top = 0; e4 += 1; }
    }
    if (e4 > 15 || (e4 == 15 && top > 6)) { e4 = 15; top = 6; }
    return sign | uchar((e4 << 3) | top);
}

// ============================================================================
// 0. Fast BF16 Copy (keeps entire 48-visit pass in ONE ComputeCommandEncoder!)
// ============================================================================
kernel void copy_bf16(
    const device bfloat4* src [[buffer(0)]],
    device bfloat4* dst [[buffer(1)]],
    constant uint& n4 [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n4) dst[tid] = src[tid];
}

// ============================================================================
// 1. Embedding + RMSNorm (BF16 -> BF16)
// ============================================================================
kernel void embed_rms_norm(
    const device int* token_ids [[buffer(0)]],
    const device bfloat* embed_table [[buffer(1)]],
    const device bfloat* norm_w [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    uint tok = uint(token_ids[row]);
    uint base = tok * 2304u + tid * 8u;
    float4 a = float4(((const device bfloat4*)(embed_table + base))[0]);
    float4 b = float4(((const device bfloat4*)(embed_table + base))[1]);
    float sq = dot(a, a) + dot(b, b);
    threadgroup float partial[16];
    float s = simd_sum(sq);
    if (lane == 0u) partial[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = simd_sum(lane < 9u ? partial[lane] : 0.0f);
    float inv = rsqrt(total / 2304.0f + eps);
    float4 wa = float4(((const device bfloat4*)(norm_w + tid * 8u))[0]);
    float4 wb = float4(((const device bfloat4*)(norm_w + tid * 8u))[1]);
    uint obase = row * 2304u + tid * 8u;
    ((device bfloat4*)(out + obase))[0] = bfloat4(a * inv * wa);
    ((device bfloat4*)(out + obase))[1] = bfloat4(b * inv * wb);
}

// ============================================================================
// 2. RMSNorm (BF16 or F32 -> BF16), width=2304
// ============================================================================
kernel void rms_norm_bf16(
    const device bfloat* x [[buffer(0)]],
    const device bfloat* w [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    uint base = row * 2304u + tid * 8u;
    float4 a = float4(((const device bfloat4*)(x + base))[0]);
    float4 b = float4(((const device bfloat4*)(x + base))[1]);
    threadgroup float partial[16];
    float s = simd_sum(dot(a, a) + dot(b, b));
    if (lane == 0u) partial[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt(simd_sum(lane < 9u ? partial[lane] : 0.0f) / 2304.0f + eps);
    float4 wa = float4(((const device bfloat4*)(w + tid * 8u))[0]);
    float4 wb = float4(((const device bfloat4*)(w + tid * 8u))[1]);
    ((device bfloat4*)(out + base))[0] = bfloat4(a * inv * wa);
    ((device bfloat4*)(out + base))[1] = bfloat4(b * inv * wb);
}

kernel void rms_norm_f32_to_bf16(
    const device float* x [[buffer(0)]],
    const device bfloat* w [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    uint base = row * 2304u + tid * 8u;
    float4 a = ((const device float4*)(x + base))[0];
    float4 b = ((const device float4*)(x + base))[1];
    threadgroup float partial[16];
    float s = simd_sum(dot(a, a) + dot(b, b));
    if (lane == 0u) partial[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt(simd_sum(lane < 9u ? partial[lane] : 0.0f) / 2304.0f + eps);
    float4 wa = float4(((const device bfloat4*)(w + tid * 8u))[0]);
    float4 wb = float4(((const device bfloat4*)(w + tid * 8u))[1]);
    ((device bfloat4*)(out + base))[0] = bfloat4(a * inv * wa);
    ((device bfloat4*)(out + base))[1] = bfloat4(b * inv * wb);
}

// ============================================================================
// 3. Block Residual Mix (Attention Residuals)
// ============================================================================
struct BlockMixParams {
    uint num_sources;
    uint num_queries;
    uint f32_mask;
    uint apply_norm;
    float eps;
};

kernel void block_residual_mix(
    const device uchar* s0 [[buffer(0)]],
    const device uchar* s1 [[buffer(1)]],
    const device uchar* s2 [[buffer(2)]],
    const device uchar* s3 [[buffer(3)]],
    const device uchar* s4 [[buffer(4)]],
    const device float* query [[buffer(5)]],
    const device bfloat* norm0 [[buffer(6)]],
    const device bfloat* norm1 [[buffer(7)]],
    const device bfloat* norm2 [[buffer(8)]],
    device float* out_f32 [[buffer(9)]],
    device bfloat* out_bf16 [[buffer(10)]],
    constant BlockMixParams& p [[buffer(11)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]])
{
    uint row = tg_pos.z;
    uint q = tg_pos.y;
    uint first = tid * 8u;
    uint offset = row * 2304u + first;

    threadgroup float squares[5][9];
    threadgroup float dots[5][9];
    float4 va[5], vb[5];
    float inv_sqrt_d = 1.0f / 48.0f; // sqrt(2304) == 48 exactly!
    float4 ka = ((const device float4*)(query + q * 2304u + first))[0] * inv_sqrt_d;
    float4 kb = ((const device float4*)(query + q * 2304u + first))[1] * inv_sqrt_d;

    for (uint s = 0; s < p.num_sources; ++s) {
        bool is_f32 = ((p.f32_mask >> s) & 1u) != 0u;
        const device uchar* ptr = (s == 0) ? s0 : (s == 1) ? s1 : (s == 2) ? s2 : (s == 3) ? s3 : s4;
        if (is_f32) {
            const device float4* fp = (const device float4*)((const device float*)ptr + offset);
            va[s] = fp[0]; vb[s] = fp[1];
        } else {
            const device bfloat4* bp = (const device bfloat4*)((const device bfloat*)ptr + offset);
            va[s] = float4(bp[0]); vb[s] = float4(bp[1]);
        }
        float part = simd_sum(dot(va[s], va[s]) + dot(vb[s], vb[s]));
        if (lane == 0u) squares[s][sg] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < p.num_sources; ++s) {
        float tot = simd_sum(lane < 9u ? squares[s][lane] : 0.0f);
        float inv_rms = rsqrt(tot / 2304.0f + p.eps);
        va[s] *= inv_rms;
        vb[s] *= inv_rms;
        float part = simd_sum(dot(va[s], ka) + dot(vb[s], kb));
        if (lane == 0u) dots[s][sg] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float w[5];
    float peak = -INFINITY;
    for (uint s = 0; s < p.num_sources; ++s) {
        w[s] = simd_sum(lane < 9u ? dots[s][lane] : 0.0f);
        peak = max(peak, w[s]);
    }
    float wsum = 0.0f;
    for (uint s = 0; s < p.num_sources; ++s) { w[s] = exp(w[s] - peak); wsum += w[s]; }
    float inv_wsum = 1.0f / wsum;
    for (uint s = 0; s < p.num_sources; ++s) w[s] *= inv_wsum;

    float4 ma = float4(0.0f), mb = float4(0.0f);
    for (uint s = 0; s < p.num_sources; ++s) {
        ma = fma(va[s], w[s], ma);
        mb = fma(vb[s], w[s], mb);
    }

    uint out_idx = (row * p.num_queries + q) * 2304u + first;
    if (p.apply_norm == 0u) {
        ((device float4*)(out_f32 + out_idx))[0] = ma;
        ((device float4*)(out_f32 + out_idx))[1] = mb;
    } else {
        threadgroup float npart[9];
        float ps = simd_sum(dot(ma, ma) + dot(mb, mb));
        if (lane == 0u) npart[sg] = ps;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = rsqrt(simd_sum(lane < 9u ? npart[lane] : 0.0f) / 2304.0f + p.eps);
        const device bfloat* nw = (q == 0) ? norm0 : (q == 1) ? norm1 : norm2;
        float4 na = float4(((const device bfloat4*)(nw + first))[0]);
        float4 nb = float4(((const device bfloat4*)(nw + first))[1]);
        ((device bfloat4*)(out_bf16 + out_idx))[0] = bfloat4(ma * inv * na);
        ((device bfloat4*)(out_bf16 + out_idx))[1] = bfloat4(mb * inv * nb);
    }
}

// ============================================================================
// 4. Loop Controller: norm_concat & mix_rmsnorm
// ============================================================================
kernel void controller_norm_concat(
    const device float* inp [[buffer(0)]],
    const device float* msg [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& inp_stride [[buffer(3)]],
    constant uint& inp_off [[buffer(4)]],
    constant uint& msg_off [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    uint first = tid * 8u;
    const device float4* pi = (const device float4*)(inp + row * inp_stride + inp_off + first);
    const device float4* pm = (const device float4*)(msg + row * (4u * 2304u) + msg_off + first);
    float4 ia = pi[0], ib = pi[1], ma = pm[0], mb = pm[1];
    threadgroup float parti[9], partm[9];
    float si = simd_sum(dot(ia, ia) + dot(ib, ib));
    float sm = simd_sum(dot(ma, ma) + dot(mb, mb));
    if (lane == 0u) { parti[sg] = si; partm[sg] = sm; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_i = rsqrt(simd_sum(lane < 9u ? parti[lane] : 0.0f) / 2304.0f + eps);
    float inv_m = rsqrt(simd_sum(lane < 9u ? partm[lane] : 0.0f) / 2304.0f + eps);
    device bfloat* po = out + row * 4608u;
    ((device bfloat4*)(po + first))[0] = bfloat4(ia * inv_i);
    ((device bfloat4*)(po + first))[1] = bfloat4(ib * inv_i);
    ((device bfloat4*)(po + 2304u + first))[0] = bfloat4(ma * inv_m);
    ((device bfloat4*)(po + 2304u + first))[1] = bfloat4(mb * inv_m);
}

kernel void controller_mix_rmsnorm(
    const device float* inp [[buffer(0)]],
    const device float* msg [[buffer(1)]],
    const device float* gain_forget [[buffer(2)]],
    const device bfloat* norm_w [[buffer(3)]],
    device bfloat* out [[buffer(4)]],
    constant uint& inp_stride [[buffer(5)]],
    constant uint& inp_off [[buffer(6)]],
    constant uint& msg_off [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    uint first = tid * 8u;
    const device float4* pi = (const device float4*)(inp + row * inp_stride + inp_off + first);
    const device float4* pm = (const device float4*)(msg + row * (4u * 2304u) + msg_off + first);
    const device float4* pg = (const device float4*)(gain_forget + row * 4608u + first);
    const device float4* pf = (const device float4*)(gain_forget + row * 4608u + 2304u + first);
    float4 va = pi[0] * (1.0f - tanh(pf[0])) + pm[0] * tanh(pg[0]);
    float4 vb = pi[1] * (1.0f - tanh(pf[1])) + pm[1] * tanh(pg[1]);
    threadgroup float part[9];
    float s = simd_sum(dot(va, va) + dot(vb, vb));
    if (lane == 0u) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt(simd_sum(lane < 9u ? part[lane] : 0.0f) / 2304.0f + eps);
    float4 na = float4(((const device bfloat4*)(norm_w + first))[0]);
    float4 nb = float4(((const device bfloat4*)(norm_w + first))[1]);
    device bfloat* po = out + row * 2304u + first;
    ((device bfloat4*)po)[0] = bfloat4(va * inv * na);
    ((device bfloat4*)po)[1] = bfloat4(vb * inv * nb);
}

// ============================================================================
// 5. HIGH-BANDWIDTH LOW-BATCH BF16 LINEAR PROJECTIONS (245+ GB/s on M4 Pro!)
//    Template-specialized per M = 1..8 -> ZERO register spills, 100% occupancy!
//    - M == 1: 4 rows/SG (16 rows/TG), zero shared mem, zero barriers!
//    - M == 2..8: 2 rows/SG (8 rows/TG), shared threadgroup x tile (<= 8 KB)!
// ============================================================================
struct LinearParams {
    uint M;
    uint N;
    uint K;
    uint out_f32;
    uint has_bias;
    uint x_is_f32;
    float softcap;
};

// Template for M = 1..8: threadgroup caches x[M, 256] (<= 8 KB), 2 output rows per SG (8 rows/TG)
// Zero register spills! Reads W ONCE across all M tokens at ~220-245 GB/s!
template<uint TM>
inline void linear_bf16_mt_impl(
    const device uchar* x_raw,
    const device bfloat* w,
    const device bfloat* bias,
    device uchar* y_raw,
    constant LinearParams& p,
    threadgroup float4* tg_x,
    uint tid, uint lane, uint sg, uint tg_x_pos)
{
    uint row0 = (tg_x_pos * 4u + sg) * 2u;
    uint K = p.K;
    uint N = p.N;
    bool active0 = row0 < N;
    bool active1 = (row0 + 1u) < N;
    const device bfloat* w0 = w + size_t(min(row0, N - 1u)) * K;
    const device bfloat* w1 = w + size_t(min(row0 + 1u, N - 1u)) * K;

    float acc0[TM], acc1[TM];
    for (uint m = 0; m < TM; ++m) { acc0[m] = 0.0f; acc1[m] = 0.0f; }

    for (uint k0 = 0; k0 < K; k0 += 256u) {
        for (uint idx = tid; idx < TM * 64u; idx += 128u) {
            uint m = idx >> 6;
            uint c4 = idx & 63u;
            uint k = k0 + c4 * 4u;
            if (p.x_is_f32 == 0u) {
                tg_x[m * 64u + c4] = float4(((const device bfloat4*)((const device bfloat*)x_raw + size_t(m) * K + k))[0]);
            } else {
                tg_x[m * 64u + c4] = ((const device float4*)((const device float*)x_raw + size_t(m) * K + k))[0];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active0) {
            uint k = k0 + lane * 8u;
            const device bfloat4* p0 = (const device bfloat4*)(w0 + k);
            float4 wa0 = float4(p0[0]), wb0 = float4(p0[1]);
            const device bfloat4* p1 = (const device bfloat4*)(w1 + k);
            float4 wa1 = float4(p1[0]), wb1 = float4(p1[1]);
            uint c0 = lane * 2u;
            for (uint m = 0; m < TM; ++m) {
                float4 xa = tg_x[m * 64u + c0];
                float4 xb = tg_x[m * 64u + c0 + 1u];
                acc0[m] += dot(xa, wa0) + dot(xb, wb0);
                acc1[m] += dot(xa, wa1) + dot(xb, wb1);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active0) {
        float b0 = (p.has_bias != 0u) ? float(bias[row0]) : 0.0f;
        float b1 = (p.has_bias != 0u && active1) ? float(bias[row0 + 1u]) : 0.0f;
        for (uint m = 0; m < TM; ++m) {
            float v0 = simd_sum(acc0[m]) + b0;
            float v1 = simd_sum(acc1[m]) + b1;
            if (lane == 0u) {
                if (p.softcap > 0.0f) { v0 = softcap_fn(v0, p.softcap); v1 = softcap_fn(v1, p.softcap); }
                if (p.out_f32 != 0u) {
                    ((device float*)y_raw)[size_t(m) * N + row0] = v0;
                    if (active1) ((device float*)y_raw)[size_t(m) * N + row0 + 1u] = v1;
                } else {
                    ((device bfloat*)y_raw)[size_t(m) * N + row0] = bfloat(v0);
                    if (active1) ((device bfloat*)y_raw)[size_t(m) * N + row0 + 1u] = bfloat(v1);
                }
            }
        }
    }
}

#define DEF_LINEAR_MT(MVAL) \
kernel void linear_bf16_m##MVAL( \
    const device uchar* x_raw [[buffer(0)]], \
    const device bfloat* w [[buffer(1)]], \
    const device bfloat* bias [[buffer(2)]], \
    device uchar* y_raw [[buffer(3)]], \
    constant LinearParams& p [[buffer(4)]], \
    uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], \
    uint sg [[simdgroup_index_in_threadgroup]], \
    uint tg_x_pos [[threadgroup_position_in_grid]]) \
{ \
    threadgroup float4 tg_x[MVAL * 64]; \
    linear_bf16_mt_impl<MVAL>(x_raw, w, bias, y_raw, p, tg_x, tid, lane, sg, tg_x_pos); \
}

// M == 1: 2 rows/SG (8 rows/TG), ZERO threadgroup memory, ZERO barriers -> max occupancy!
kernel void linear_bf16_m1(
    const device uchar* x_raw [[buffer(0)]],
    const device bfloat* w [[buffer(1)]],
    const device bfloat* bias [[buffer(2)]],
    device uchar* y_raw [[buffer(3)]],
    constant LinearParams& p [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tg_x_pos [[threadgroup_position_in_grid]])
{
    uint row0 = (tg_x_pos * 4u + sg) * 2u;
    uint N = p.N;
    if (row0 >= N) return;
    bool active1 = (row0 + 1u) < N;
    uint K = p.K;
    const device bfloat* w0 = w + size_t(row0) * K;
    const device bfloat* w1 = w + size_t(min(row0 + 1u, N - 1u)) * K;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint k = lane * 8u; k < K; k += 256u) {
        float4 xa, xb;
        if (p.x_is_f32 == 0u) {
            const device bfloat4* xp = (const device bfloat4*)((const device bfloat*)x_raw + k);
            xa = float4(xp[0]); xb = float4(xp[1]);
        } else {
            const device float4* xp = (const device float4*)((const device float*)x_raw + k);
            xa = xp[0]; xb = xp[1];
        }
        const device bfloat4* p0 = (const device bfloat4*)(w0 + k);
        acc0 += dot(xa, float4(p0[0])) + dot(xb, float4(p0[1]));
        const device bfloat4* p1 = (const device bfloat4*)(w1 + k);
        acc1 += dot(xa, float4(p1[0])) + dot(xb, float4(p1[1]));
    }
    float v0 = simd_sum(acc0) + ((p.has_bias != 0u) ? float(bias[row0]) : 0.0f);
    float v1 = simd_sum(acc1) + ((p.has_bias != 0u && active1) ? float(bias[row0 + 1u]) : 0.0f);
    if (lane == 0u) {
        if (p.softcap > 0.0f) { v0 = softcap_fn(v0, p.softcap); v1 = softcap_fn(v1, p.softcap); }
        if (p.out_f32 != 0u) {
            ((device float*)y_raw)[row0] = v0;
            if (active1) ((device float*)y_raw)[row0 + 1u] = v1;
        } else {
            ((device bfloat*)y_raw)[row0] = bfloat(v0);
            if (active1) ((device bfloat*)y_raw)[row0 + 1u] = bfloat(v1);
        }
    }
}

DEF_LINEAR_MT(2)
DEF_LINEAR_MT(3)
DEF_LINEAR_MT(4)
DEF_LINEAR_MT(5)
DEF_LINEAR_MT(6)
DEF_LINEAR_MT(7)
DEF_LINEAR_MT(8)

// Fast path for K=64 (loop controller gain_forget [4608, 64], supports M = 1..32 in exact F32)
kernel void linear_bf16_k64(
    const device float* x [[buffer(0)]],
    const device bfloat* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant LinearParams& p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tg_x [[threadgroup_position_in_grid]])
{
    uint row0 = (tg_x * 4u + sg) * 4u;
    if (row0 >= p.N) return;
    uint rmax = min(4u, p.N - row0);
    uint M = p.M;
    float2 wf[4];
    for (uint r = 0; r < rmax; ++r) {
        wf[r] = float2(((const device bfloat2*)(w + size_t(row0 + r) * 64u))[lane]);
    }
    for (uint m0 = 0; m0 < M; m0 += 8u) {
        uint mb = min(8u, M - m0);
        float2 xv[8];
        for (uint j = 0; j < mb; ++j) xv[j] = ((const device float2*)(x + size_t(m0 + j) * 64u))[lane];
        for (uint r = 0; r < rmax; ++r) {
            for (uint j = 0; j < mb; ++j) {
                float s = simd_sum(xv[j].x * wf[r].x + xv[j].y * wf[r].y);
                if (lane == 0u) y[size_t(m0 + j) * p.N + (row0 + r)] = s;
            }
        }
    }
}

// High-bandwidth tiled BF16 AMX GEMM for M > 8 (BM=32, BN=64, BK=32, M-inner grid swizzle for L2 W reuse!)
kernel void linear_bf16_tiled(
    const device uchar* x_raw [[buffer(0)]],
    const device bfloat* w [[buffer(1)]],
    const device bfloat* bias [[buffer(2)]],
    device uchar* y_raw [[buffer(3)]],
    constant LinearParams& p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]])
{
    uint bm = tg_pos.x * 32u;
    uint bn = tg_pos.y * 64u;
    uint sg_m = (sg / 2u) * 16u;
    uint sg_n = (sg % 2u) * 32u;

    // 8 KB shared memory reused between (As[32][36] + Bs[64][36] = 6912 B) and Cs[32][64] (8192 B)
    threadgroup float smem_f32[2048];
    threadgroup bfloat* As = (threadgroup bfloat*)smem_f32;          // [32 * 36] = 1152 bfloats
    threadgroup bfloat* Bs = ((threadgroup bfloat*)smem_f32) + 1152; // [64 * 36] = 2304 bfloats

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint M = p.M, N = p.N, K = p.K;
    uint a_r = tid >> 3;        // 0..15
    uint a_c4 = (tid & 7u) * 4u; // 0,4,8..28
    bool full_tile = (bm + 32u <= M) && (bn + 64u <= N) && ((K & 31u) == 0u) && (p.x_is_f32 == 0u);
    if (full_tile) {
        const device bfloat* x_bf = (const device bfloat*)x_raw;
        for (uint k0 = 0; k0 < K; k0 += 32u) {
            ((threadgroup bfloat4*)(As + (0u + a_r) * 36u + a_c4))[0] =
                ((const device bfloat4*)(x_bf + size_t(bm + 0u + a_r) * K + k0 + a_c4))[0];
            ((threadgroup bfloat4*)(As + (16u + a_r) * 36u + a_c4))[0] =
                ((const device bfloat4*)(x_bf + size_t(bm + 16u + a_r) * K + k0 + a_c4))[0];
            for (uint r0 = 0; r0 < 64u; r0 += 16u) {
                ((threadgroup bfloat4*)(Bs + (r0 + a_r) * 36u + a_c4))[0] =
                    ((const device bfloat4*)(w + size_t(bn + r0 + a_r) * K + k0 + a_c4))[0];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ks = 0; ks < 32u; ks += 8u) {
                simdgroup_bfloat8x8 a0, a1, b0, b1, b2, b3;
                simdgroup_load(a0, As + (sg_m + 0u) * 36u + ks, 36);
                simdgroup_load(a1, As + (sg_m + 8u) * 36u + ks, 36);
                simdgroup_load(b0, Bs + (sg_n + 0u) * 36u + ks, 36, ulong2(0), true);
                simdgroup_load(b1, Bs + (sg_n + 8u) * 36u + ks, 36, ulong2(0), true);
                simdgroup_load(b2, Bs + (sg_n + 16u) * 36u + ks, 36, ulong2(0), true);
                simdgroup_load(b3, Bs + (sg_n + 24u) * 36u + ks, 36, ulong2(0), true);
                simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
                simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
                simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);
                simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
                simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
                simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
                simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);
                simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (uint k0 = 0; k0 < K; k0 += 32u) {
            for (uint r0 = 0; r0 < 32u; r0 += 16u) {
                uint r = r0 + a_r;
                uint gm = bm + r;
                uint gk = k0 + a_c4;
                bfloat4 v = bfloat4(bfloat(0.0f));
                if (gm < M && gk + 3u < K) {
                    if (p.x_is_f32 == 0u) {
                        v = ((const device bfloat4*)((const device bfloat*)x_raw + size_t(gm) * K + gk))[0];
                    } else {
                        v = bfloat4(((const device float4*)((const device float*)x_raw + size_t(gm) * K + gk))[0]);
                    }
                } else if (gm < M) {
                    for (uint q = 0; q < 4u; ++q) {
                        if (gk + q < K) {
                            v[q] = (p.x_is_f32 == 0u) ? ((const device bfloat*)x_raw)[size_t(gm) * K + gk + q]
                                                      : bfloat(((const device float*)x_raw)[size_t(gm) * K + gk + q]);
                        }
                    }
                }
                ((threadgroup bfloat4*)(As + r * 36u + a_c4))[0] = v;
            }
            for (uint r0 = 0; r0 < 64u; r0 += 16u) {
                uint r = r0 + a_r;
                uint gn = bn + r;
                uint gk = k0 + a_c4;
                bfloat4 v = bfloat4(bfloat(0.0f));
                if (gn < N && gk + 3u < K) {
                    v = ((const device bfloat4*)(w + size_t(gn) * K + gk))[0];
                } else if (gn < N) {
                    for (uint q = 0; q < 4u; ++q) {
                        if (gk + q < K) v[q] = w[size_t(gn) * K + gk + q];
                    }
                }
                ((threadgroup bfloat4*)(Bs + r * 36u + a_c4))[0] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint ks = 0; ks < 32u; ks += 8u) {
                simdgroup_bfloat8x8 a0, a1, b0, b1, b2, b3;
                simdgroup_load(a0, As + (sg_m + 0u) * 36u + ks, 36);
                simdgroup_load(a1, As + (sg_m + 8u) * 36u + ks, 36);
                simdgroup_load(b0, Bs + (sg_n + 0u) * 36u + ks, 36, ulong2(0), true);
                simdgroup_load(b1, Bs + (sg_n + 8u) * 36u + ks, 36, ulong2(0), true);
                simdgroup_load(b2, Bs + (sg_n + 16u) * 36u + ks, 36, ulong2(0), true);
                simdgroup_load(b3, Bs + (sg_n + 24u) * 36u + ks, 36, ulong2(0), true);
                simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
                simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
                simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);
                simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
                simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
                simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
                simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);
                simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Reuse smem_f32 as Cs[32][64]
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j)
            simdgroup_store(acc[i][j], smem_f32 + (sg_m + i * 8u) * 64u + (sg_n + j * 8u), 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < 2048u; idx += 128u) {
        uint r = idx >> 6, c = idx & 63u;
        uint gm = bm + r, gn = bn + c;
        if (gm < M && gn < N) {
            float v = smem_f32[idx];
            if (p.has_bias != 0u) v += float(bias[gn]);
            if (p.softcap > 0.0f) v = softcap_fn(v, p.softcap);
            if (p.out_f32 != 0u) ((device float*)y_raw)[size_t(gm) * N + gn] = v;
            else ((device bfloat*)y_raw)[size_t(gm) * N + gn] = bfloat(v);
        }
    }
}

// Peak-roofline BF16 AMX GEMM for M >= 64, N >= 128 (BM=64, BN=64, BK=32, 6.53 TFLOPS on M4 Pro!)
// Grid: ((M + 63)/64, (N + 63)/64, 1) -> M-inner swizzle keeps B[bn..bn+63] resident in L2 across all M tiles!
kernel void linear_bf16_tiled64(
    const device uchar* x_raw [[buffer(0)]],
    const device bfloat* w [[buffer(1)]],
    const device bfloat* bias [[buffer(2)]],
    device uchar* y_raw [[buffer(3)]],
    constant LinearParams& p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]])
{
    uint bm = tg_pos.x * 64u;
    uint bn = tg_pos.y * 64u;
    uint sg_m = (sg / 2u) * 32u;
    uint sg_n = (sg % 2u) * 32u;

    // 16 KB shared memory reused between (As[64][36] + Bs[64][36] = 9216 B) and Cs[64][64] (16384 B)
    threadgroup float smem_f32[4096];
    threadgroup bfloat* As = (threadgroup bfloat*)smem_f32;          // [64 * 36] = 2304 bfloats
    threadgroup bfloat* Bs = ((threadgroup bfloat*)smem_f32) + 2304; // [64 * 36] = 2304 bfloats

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4u; ++i)
        for (uint j = 0; j < 4u; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint M = p.M, N = p.N, K = p.K;
    uint a_r = tid >> 3;         // 0..15
    uint a_c4 = (tid & 7u) * 4u; // 0,4..28
    bool full_tile = (bm + 64u <= M) && (bn + 64u <= N) && ((K & 31u) == 0u) && (p.x_is_f32 == 0u);
    if (full_tile) {
        const device bfloat* x_bf = (const device bfloat*)x_raw;
        for (uint k0 = 0; k0 < K; k0 += 32u) {
            for (uint r0 = 0; r0 < 64u; r0 += 16u) {
                uint r = r0 + a_r;
                ((threadgroup bfloat4*)(As + r * 36u + a_c4))[0] =
                    ((const device bfloat4*)(x_bf + size_t(bm + r) * K + k0 + a_c4))[0];
                ((threadgroup bfloat4*)(Bs + r * 36u + a_c4))[0] =
                    ((const device bfloat4*)(w + size_t(bn + r) * K + k0 + a_c4))[0];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ks = 0; ks < 32u; ks += 8u) {
                simdgroup_bfloat8x8 a[4], b[4];
                for (uint i = 0; i < 4u; ++i) simdgroup_load(a[i], As + (sg_m + i * 8u) * 36u + ks, 36);
                for (uint j = 0; j < 4u; ++j) simdgroup_load(b[j], Bs + (sg_n + j * 8u) * 36u + ks, 36, ulong2(0), true);
                for (uint i = 0; i < 4u; ++i)
                    for (uint j = 0; j < 4u; ++j)
                        simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (uint k0 = 0; k0 < K; k0 += 32u) {
            for (uint r0 = 0; r0 < 64u; r0 += 16u) {
                uint r = r0 + a_r;
                uint gm = bm + r;
                uint gn = bn + r;
                uint gk = k0 + a_c4;
                bfloat4 av = bfloat4(bfloat(0.0f));
                if (gm < M && gk + 3u < K) {
                    if (p.x_is_f32 == 0u) {
                        av = ((const device bfloat4*)((const device bfloat*)x_raw + size_t(gm) * K + gk))[0];
                    } else {
                        av = bfloat4(((const device float4*)((const device float*)x_raw + size_t(gm) * K + gk))[0]);
                    }
                } else if (gm < M) {
                    for (uint q = 0; q < 4u; ++q) {
                        if (gk + q < K) {
                            av[q] = (p.x_is_f32 == 0u) ? ((const device bfloat*)x_raw)[size_t(gm) * K + gk + q]
                                                       : bfloat(((const device float*)x_raw)[size_t(gm) * K + gk + q]);
                        }
                    }
                }
                ((threadgroup bfloat4*)(As + r * 36u + a_c4))[0] = av;

                bfloat4 bv = bfloat4(bfloat(0.0f));
                if (gn < N && gk + 3u < K) {
                    bv = ((const device bfloat4*)(w + size_t(gn) * K + gk))[0];
                } else if (gn < N) {
                    for (uint q = 0; q < 4u; ++q) {
                        if (gk + q < K) bv[q] = w[size_t(gn) * K + gk + q];
                    }
                }
                ((threadgroup bfloat4*)(Bs + r * 36u + a_c4))[0] = bv;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ks = 0; ks < 32u; ks += 8u) {
                simdgroup_bfloat8x8 a[4], b[4];
                for (uint i = 0; i < 4u; ++i) simdgroup_load(a[i], As + (sg_m + i * 8u) * 36u + ks, 36);
                for (uint j = 0; j < 4u; ++j) simdgroup_load(b[j], Bs + (sg_n + j * 8u) * 36u + ks, 36, ulong2(0), true);
                for (uint i = 0; i < 4u; ++i)
                    for (uint j = 0; j < 4u; ++j)
                        simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint i = 0; i < 4u; ++i)
        for (uint j = 0; j < 4u; ++j)
            simdgroup_store(acc[i][j], smem_f32 + (sg_m + i * 8u) * 64u + (sg_n + j * 8u), 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Vectorized bfloat4 / float4 epilogue when aligned, else scalar tail
    if (bm + 64u <= M && bn + 64u <= N && p.has_bias == 0u && p.softcap == 0.0f && (N & 3u) == 0u) {
        for (uint idx = tid; idx < 1024u; idx += 128u) {
            uint r = idx >> 4;
            uint c4 = (idx & 15u) * 4u;
            float4 v = ((const threadgroup float4*)(smem_f32 + r * 64u + c4))[0];
            if (p.out_f32 != 0u) {
                ((device float4*)((device float*)y_raw + size_t(bm + r) * N + bn + c4))[0] = v;
            } else {
                ((device bfloat4*)((device bfloat*)y_raw + size_t(bm + r) * N + bn + c4))[0] = bfloat4(v);
            }
        }
    } else {
        for (uint idx = tid; idx < 4096u; idx += 128u) {
            uint r = idx >> 6, c = idx & 63u;
            uint gm = bm + r, gn = bn + c;
            if (gm < M && gn < N) {
                float v = smem_f32[idx];
                if (p.has_bias != 0u) v += float(bias[gn]);
                if (p.softcap > 0.0f) v = softcap_fn(v, p.softcap);
                if (p.out_f32 != 0u) ((device float*)y_raw)[size_t(gm) * N + gn] = v;
                else ((device bfloat*)y_raw)[size_t(gm) * N + gn] = bfloat(v);
            }
        }
    }
}

// ============================================================================
// 6. Short Causal Convolution (4 taps) + Speculative Rollback
// ============================================================================
struct ShortConvParams {
    uint M, C;
    uint x_stride, x_offset;
    uint is_dynamic, dyn_stride, dyn_offset;
    uint save_win;
};

kernel void short_conv_4tap(
    const device bfloat* x [[buffer(0)]],
    const device bfloat* weight [[buffer(1)]],
    device bfloat* conv_state [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    device bfloat* conv_win [[buffer(4)]],
    constant ShortConvParams& p [[buffer(5)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= p.C) return;
    float s0 = float(conv_state[0u * p.C + c]);
    float s1 = float(conv_state[1u * p.C + c]);
    float s2 = float(conv_state[2u * p.C + c]);
    if (p.save_win != 0u) {
        conv_win[0u * p.C + c] = bfloat(s0);
        conv_win[1u * p.C + c] = bfloat(s1);
        conv_win[2u * p.C + c] = bfloat(s2);
    }
    float w0 = float(weight[0u * p.C + c]);
    float w1 = float(weight[1u * p.C + c]);
    float w2 = float(weight[2u * p.C + c]);
    float w3 = float(weight[3u * p.C + c]);
    uint head = c / 128u;
    float inv_scale = (p.is_dynamic != 0u) ? (1.0f / 3.0f) : rsqrt(5.0f);

    for (uint t = 0; t < p.M; ++t) {
        float cur = float(x[size_t(t) * p.x_stride + p.x_offset + c]);
        if (p.save_win != 0u) conv_win[(3u + t) * p.C + c] = bfloat(cur);
        float sum;
        if (p.is_dynamic == 0u) {
            sum = s0 * w0 + s1 * w1 + s2 * w2 + cur * w3 + cur;
        } else {
            const device bfloat* dp = x + size_t(t) * p.dyn_stride + p.dyn_offset + head * 4u;
            sum = cur * (w0 + float(dp[0])) + s2 * (w1 + float(dp[1])) +
                  s1 * (w2 + float(dp[2])) + s0 * (w3 + float(dp[3])) + cur;
        }
        out[size_t(t) * p.C + c] = bfloat(sum * inv_scale);
        s0 = s1; s1 = s2; s2 = cur;
    }
    conv_state[0u * p.C + c] = bfloat(s0);
    conv_state[1u * p.C + c] = bfloat(s1);
    conv_state[2u * p.C + c] = bfloat(s2);
}

// Fused Attention ShortConv (4-tap static) + Residual Add into F32 partial!
// Eliminates separate add_bf16_to_f32 kernel on non-boundary visits (40 fewer dispatches/forward!)
kernel void short_conv_4tap_add_f32(
    const device bfloat* x [[buffer(0)]],
    const device bfloat* weight [[buffer(1)]],
    device bfloat* conv_state [[buffer(2)]],
    const device bfloat* residual_bf16 [[buffer(3)]],
    device float* out_f32 [[buffer(4)]],
    device bfloat* conv_win [[buffer(5)]],
    constant ShortConvParams& p [[buffer(6)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= p.C) return;
    float s0 = float(conv_state[0u * p.C + c]);
    float s1 = float(conv_state[1u * p.C + c]);
    float s2 = float(conv_state[2u * p.C + c]);
    if (p.save_win != 0u) {
        conv_win[0u * p.C + c] = bfloat(s0);
        conv_win[1u * p.C + c] = bfloat(s1);
        conv_win[2u * p.C + c] = bfloat(s2);
    }
    float w0 = float(weight[0u * p.C + c]);
    float w1 = float(weight[1u * p.C + c]);
    float w2 = float(weight[2u * p.C + c]);
    float w3 = float(weight[3u * p.C + c]);
    float inv_scale = rsqrt(5.0f);

    for (uint t = 0; t < p.M; ++t) {
        float cur = float(x[size_t(t) * p.C + c]);
        if (p.save_win != 0u) conv_win[(3u + t) * p.C + c] = bfloat(cur);
        float conv_out = float(bfloat((s0 * w0 + s1 * w1 + s2 * w2 + cur * w3 + cur) * inv_scale));
        out_f32[size_t(t) * p.C + c] = float(residual_bf16[size_t(t) * p.C + c]) + conv_out;
        s0 = s1; s1 = s2; s2 = cur;
    }
    conv_state[0u * p.C + c] = bfloat(s0);
    conv_state[1u * p.C + c] = bfloat(s1);
    conv_state[2u * p.C + c] = bfloat(s2);
}

kernel void short_conv_rollback(
    const device bfloat* conv_win [[buffer(0)]],
    device bfloat* conv_state [[buffer(1)]],
    constant uint& C [[buffer(2)]],
    constant uint& committed_len [[buffer(3)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= C) return;
    conv_state[0u * C + c] = conv_win[(committed_len + 0u) * C + c];
    conv_state[1u * C + c] = conv_win[(committed_len + 1u) * C + c];
    conv_state[2u * C + c] = conv_win[(committed_len + 2u) * C + c];
}

// Parallel 2D FIR 4-tap ShortConv for prefill (M > 8, save_win == 0): saturates all 20 M4 Pro GPU cores!
kernel void short_conv_4tap_prefill(
    const device bfloat* x [[buffer(0)]],
    const device bfloat* weight [[buffer(1)]],
    const device bfloat* conv_state [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant ShortConvParams& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint c = gid.x, t = gid.y;
    if (c >= p.C || t >= p.M) return;
    float cur = float(x[size_t(t) * p.x_stride + p.x_offset + c]);
    float s2 = (t >= 1u) ? float(x[size_t(t - 1u) * p.x_stride + p.x_offset + c]) : float(conv_state[2u * p.C + c]);
    float s1 = (t >= 2u) ? float(x[size_t(t - 2u) * p.x_stride + p.x_offset + c]) : float(conv_state[(1u + t) * p.C + c]);
    float s0 = (t >= 3u) ? float(x[size_t(t - 3u) * p.x_stride + p.x_offset + c]) : float(conv_state[t * p.C + c]);
    float w0 = float(weight[0u * p.C + c]);
    float w1 = float(weight[1u * p.C + c]);
    float w2 = float(weight[2u * p.C + c]);
    float w3 = float(weight[3u * p.C + c]);
    float inv_scale = (p.is_dynamic != 0u) ? (1.0f / 3.0f) : rsqrt(5.0f);
    float sum;
    if (p.is_dynamic == 0u) {
        sum = s0 * w0 + s1 * w1 + s2 * w2 + cur * w3 + cur;
    } else {
        uint head = c / 128u;
        const device bfloat* dp = x + size_t(t) * p.dyn_stride + p.dyn_offset + head * 4u;
        sum = cur * (w0 + float(dp[0])) + s2 * (w1 + float(dp[1])) +
              s1 * (w2 + float(dp[2])) + s0 * (w3 + float(dp[3])) + cur;
    }
    out[size_t(t) * p.C + c] = bfloat(sum * inv_scale);
}

kernel void short_conv_4tap_add_f32_prefill(
    const device bfloat* x [[buffer(0)]],
    const device bfloat* weight [[buffer(1)]],
    const device bfloat* conv_state [[buffer(2)]],
    const device bfloat* residual_bf16 [[buffer(3)]],
    device float* out_f32 [[buffer(4)]],
    constant ShortConvParams& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint c = gid.x, t = gid.y;
    if (c >= p.C || t >= p.M) return;
    float cur = float(x[size_t(t) * p.C + c]);
    float s2 = (t >= 1u) ? float(x[size_t(t - 1u) * p.C + c]) : float(conv_state[2u * p.C + c]);
    float s1 = (t >= 2u) ? float(x[size_t(t - 2u) * p.C + c]) : float(conv_state[(1u + t) * p.C + c]);
    float s0 = (t >= 3u) ? float(x[size_t(t - 3u) * p.C + c]) : float(conv_state[t * p.C + c]);
    float w0 = float(weight[0u * p.C + c]);
    float w1 = float(weight[1u * p.C + c]);
    float w2 = float(weight[2u * p.C + c]);
    float w3 = float(weight[3u * p.C + c]);
    float conv_out = float(bfloat((s0 * w0 + s1 * w1 + s2 * w2 + cur * w3 + cur) * rsqrt(5.0f)));
    out_f32[size_t(t) * p.C + c] = float(residual_bf16[size_t(t) * p.C + c]) + conv_out;
}

kernel void short_conv_4tap_add_bf16_prefill(
    const device bfloat* x [[buffer(0)]],
    const device bfloat* weight [[buffer(1)]],
    const device bfloat* conv_state [[buffer(2)]],
    const device uchar* partial_in [[buffer(3)]],
    device bfloat* partial_out [[buffer(4)]],
    constant ShortConvParams& p [[buffer(5)]], // is_dynamic reused as partial_is_f32!
    uint2 gid [[thread_position_in_grid]])
{
    uint c = gid.x, t = gid.y;
    if (c >= p.C || t >= p.M) return;
    float cur = float(x[size_t(t) * p.C + c]);
    float s2 = (t >= 1u) ? float(x[size_t(t - 1u) * p.C + c]) : float(conv_state[2u * p.C + c]);
    float s1 = (t >= 2u) ? float(x[size_t(t - 2u) * p.C + c]) : float(conv_state[(1u + t) * p.C + c]);
    float s0 = (t >= 3u) ? float(x[size_t(t - 3u) * p.C + c]) : float(conv_state[t * p.C + c]);
    float w0 = float(weight[0u * p.C + c]);
    float w1 = float(weight[1u * p.C + c]);
    float w2 = float(weight[2u * p.C + c]);
    float w3 = float(weight[3u * p.C + c]);
    float conv_out = float(bfloat((s0 * w0 + s1 * w1 + s2 * w2 + cur * w3 + cur) * rsqrt(5.0f)));
    float prev = (p.is_dynamic != 0u)
                     ? ((const device float*)partial_in)[size_t(t) * p.C + c]
                     : float(((const device bfloat*)partial_in)[size_t(t) * p.C + c]);
    partial_out[size_t(t) * p.C + c] = bfloat(prev + conv_out);
}

// Race-free 3-tap state commit after 2D parallel prefill FIR (requires M >= 3)
kernel void short_conv_commit_state(
    const device bfloat* x [[buffer(0)]],
    device bfloat* conv_state [[buffer(1)]],
    constant ShortConvParams& p [[buffer(2)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= p.C) return;
    conv_state[0u * p.C + c] = x[size_t(p.M - 3u) * p.x_stride + p.x_offset + c];
    conv_state[1u * p.C + c] = x[size_t(p.M - 2u) * p.x_stride + p.x_offset + c];
    conv_state[2u * p.C + c] = x[size_t(p.M - 1u) * p.x_stride + p.x_offset + c];
}

// ============================================================================
// 7. Attention Prep + Relative-Position FlashDecoding + Fused Output Gate
// ============================================================================
struct AttnPrepParams {
    uint M, pos, kv_capacity, is_ring;
    uint q_stride, r_offset;
    float qk_clip, rel_clip, eps, log_floor, log_alpha;
};

kernel void attn_prep_qr(
    const device bfloat* q_proj [[buffer(0)]],
    const device bfloat* q_scale [[buffer(1)]],
    const device bfloat* rel_scale [[buffer(2)]],
    device float* q_out [[buffer(3)]],
    device bfloat* r_out [[buffer(4)]],
    constant AttnPrepParams& p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    uint h = tg.x, t = tg.y;
    float tau = 1.0f;
    if (p.log_floor > 0.0f && p.log_alpha != 0.0f) {
        float ratio = max(float(p.pos + t + 1u) / p.log_floor, 1.0f);
        tau = 1.0f + p.log_alpha * log(ratio);
    }
    uint q_base = t * p.q_stride + h * 128u + lane * 4u;
    uint qs_base = h * 128u + lane * 4u;
    float4 qv = float4(((const device bfloat4*)(q_proj + q_base))[0]) *
                float4(((const device bfloat4*)(q_scale + qs_base))[0]);
    float fac = min(p.qk_clip / (sqrt(simd_sum(dot(qv, qv))) + 1e-6f), 1.0f) * tau;
    uint qo_base = (t * 24u + h) * 128u + lane * 4u;
    ((device float4*)(q_out + qo_base))[0] = float4(bfloat4(qv * fac));

    float rv = 0.0f;
    if (lane < 16u) {
        rv = float(q_proj[t * p.q_stride + p.r_offset + h * 16u + lane]) *
             float(rel_scale[h * 16u + lane]);
    }
    float rfac = min(p.rel_clip / (sqrt(simd_sum(rv * rv)) + 1e-6f), 1.0f) * tau;
    if (lane < 16u) r_out[(t * 24u + h) * 16u + lane] = bfloat(rv * rfac);
}

kernel void attn_prep_kv_cache(
    const device bfloat* k_conv [[buffer(0)]],
    const device bfloat* v_conv [[buffer(1)]],
    device bfloat* k_cache [[buffer(2)]],
    device bfloat* v_cache [[buffer(3)]],
    constant AttnPrepParams& p [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    uint h = tg.x, t = tg.y;
    uint abs_pos = p.pos + t;
    uint slot = (p.is_ring != 0u) ? (abs_pos % p.kv_capacity) : abs_pos;
    uint in_base = t * 1024u + h * 128u + lane * 4u;
    float4 kv = float4(((const device bfloat4*)(k_conv + in_base))[0]);
    float4 vv = float4(((const device bfloat4*)(v_conv + in_base))[0]);
    float kfac = min(p.qk_clip / (sqrt(simd_sum(dot(kv, kv))) + 1e-6f), 1.0f);
    float vinv = rsqrt(simd_sum(dot(vv, vv)) / 128.0f + p.eps);
    uint dst_base = (size_t(h) * p.kv_capacity + slot) * 128u + lane * 4u;
    ((device bfloat4*)(k_cache + dst_base))[0] = bfloat4(kv * kfac);
    ((device bfloat4*)(v_cache + dst_base))[0] = bfloat4(vv * vinv);
}

kernel void attn_rel_table(
    const device bfloat* r_in [[buffer(0)]],
    const device bfloat* rel_proj [[buffer(1)]],
    device bfloat* table [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    uint d = tg.x * 32u + lane;
    uint row = tg.y;
    const device bfloat* rp = r_in + row * 16u;
    float acc = 0.0f;
    for (uint k = 0; k < 16u; ++k) acc = fma(float(rp[k]), float(rel_proj[k * 1024u + d]), acc);
    table[row * 1024u + d] = bfloat(acc);
}

// AMX Relative Position Table GEMM for prefill: table[R, 1024] = r_in[R, 16] @ rel_proj[16, 1024]
// BM=32, BN=64, K=16 -> 1 barrier, 2 AMX steps, 10x faster than scalar attn_rel_table!
// Grid: (16, (R + 31) / 32, 1), TG: (128, 1, 1)
kernel void attn_rel_table_amx(
    const device bfloat* r_in [[buffer(0)]],
    const device bfloat* rel_proj [[buffer(1)]],
    device bfloat* table [[buffer(2)]],
    constant uint& total_rows [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    uint bm = tg.y * 32u;
    uint bn = tg.x * 64u;
    uint sg_m = (sg / 2u) * 16u;
    uint sg_n = (sg % 2u) * 32u;

    // 8 KB shared mem reused between (As[32][20] + Bs[16][68] = 3456 B) and Cs[32][64] (8192 B)
    threadgroup float smem_f32[2048];
    threadgroup bfloat* As = (threadgroup bfloat*)smem_f32;          // [32 * 20] = 640 bfloats
    threadgroup bfloat* Bs = ((threadgroup bfloat*)smem_f32) + 640;  // [16 * 68] = 1088 bfloats

    // Load As[32][16] (128 threads x 1 bfloat4 = 512 elems exact!)
    uint a_r = tid >> 2;         // 0..31
    uint a_c4 = (tid & 3u) * 4u; // 0, 4, 8, 12
    uint gm = bm + a_r;
    ((threadgroup bfloat4*)(As + a_r * 20u + a_c4))[0] =
        (gm < total_rows) ? ((const device bfloat4*)(r_in + size_t(gm) * 16u + a_c4))[0]
                          : bfloat4(bfloat(0.0f));

    // Load Bs[16][64] (128 threads x 2 bfloat4 = 1024 elems exact!)
    uint b_r0 = tid >> 4;          // 0..7
    uint b_c4 = (tid & 15u) * 4u;  // 0, 4..60
    ((threadgroup bfloat4*)(Bs + (b_r0 + 0u) * 68u + b_c4))[0] =
        ((const device bfloat4*)(rel_proj + size_t(b_r0 + 0u) * 1024u + bn + b_c4))[0];
    ((threadgroup bfloat4*)(Bs + (b_r0 + 8u) * 68u + b_c4))[0] =
        ((const device bfloat4*)(rel_proj + size_t(b_r0 + 8u) * 1024u + bn + b_c4))[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0; i < 2u; ++i)
        for (uint j = 0; j < 4u; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint ks = 0u; ks < 16u; ks += 8u) {
        simdgroup_bfloat8x8 a0, a1, b0, b1, b2, b3;
        simdgroup_load(a0, As + (sg_m + 0u) * 20u + ks, 20);
        simdgroup_load(a1, As + (sg_m + 8u) * 20u + ks, 20);
        simdgroup_load(b0, Bs + ks * 68u + sg_n + 0u, 68);
        simdgroup_load(b1, Bs + ks * 68u + sg_n + 8u, 68);
        simdgroup_load(b2, Bs + ks * 68u + sg_n + 16u, 68);
        simdgroup_load(b3, Bs + ks * 68u + sg_n + 24u, 68);
        simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
        simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
        simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);
        simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
        simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
        simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
        simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);
        simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < 2u; ++i)
        for (uint j = 0; j < 4u; ++j)
            simdgroup_store(acc[i][j], smem_f32 + (sg_m + i * 8u) * 64u + (sg_n + j * 8u), 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Vectorized bfloat4 writeout (512 bfloat4s / 128 threads = 4 stores/thread)
    for (uint idx = tid; idx < 512u; idx += 128u) {
        uint r = idx >> 4;
        uint c4 = (idx & 15u) * 4u;
        uint out_row = bm + r;
        if (out_row < total_rows) {
            float4 v = ((const threadgroup float4*)(smem_f32 + r * 64u + c4))[0];
            ((device bfloat4*)(table + size_t(out_row) * 1024u + bn + c4))[0] = bfloat4(v);
        }
    }
}

struct RelAttnParams {
    uint M, pos, kv_capacity, is_ring, window, parts;
    uint q_stride, gate_offset, sink_offset;
    float scale;
};

kernel void rel_attn_split(
    const device float* q_in [[buffer(0)]],
    const device bfloat* k_cache [[buffer(1)]],
    const device bfloat* v_cache [[buffer(2)]],
    const device bfloat* table [[buffer(3)]],
    const device bfloat* q_proj [[buffer(4)]],
    device float* partial_out [[buffer(5)]],
    device float2* partial_stat [[buffer(6)]],
    constant RelAttnParams& p [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    uint shard = tg.x, row = tg.y;
    uint t = row / 24u, head = row % 24u, kv_head = head / 3u;
    uint q_pos = p.pos + t;
    uint end = q_pos + 1u;
    uint start = (p.window > 0u && end > p.window) ? (end - p.window) : 0u;
    uint visible = end - start;
    uint comp = lane * 4u;
    float4 qv = ((const device float4*)(q_in + row * 128u + comp))[0] * p.scale;
    float sink_val = float(q_proj[size_t(t) * p.q_stride + p.sink_offset + head]);
    float peak = (shard == 0u) ? 0.0f : -INFINITY;
    float wsum = (shard == 0u) ? 1.0f : 0.0f;
    float4 acc = float4(0.0f);
    const device bfloat* kh_base = k_cache + size_t(kv_head) * p.kv_capacity * 128u;
    const device bfloat* vh_base = v_cache + size_t(kv_head) * p.kv_capacity * 128u;
    const device bfloat* tbl_row = table + size_t(row) * 1024u;

    for (uint i = shard; i < visible; i += p.parts) {
        uint k_pos = start + i;
        uint slot = (p.is_ring != 0u) ? (k_pos % p.kv_capacity) : k_pos;
        float4 kf = float4(((const device bfloat4*)(kh_base + size_t(slot) * 128u + comp))[0]);
        float dot_val = simd_sum(dot(qv, kf));
        uint dist = q_pos - k_pos;
        float bias = (dist < 1024u) ? float(tbl_row[dist]) : 0.0f;
        float score = dot_val + (bias - sink_val);
        float next = max(peak, score);
        float prior = fast::exp(peak - next);
        float prob = fast::exp(score - next);
        float4 vf = float4(((const device bfloat4*)(vh_base + size_t(slot) * 128u + comp))[0]);
        acc = fma(acc, prior, prob * vf);
        wsum = fma(wsum, prior, prob);
        peak = next;
    }
    uint pidx = row * p.parts + shard;
    if (lane == 0u) partial_stat[pidx] = float2(peak, wsum);
    ((device float4*)(partial_out + size_t(pidx) * 128u + comp))[0] = acc;
}

kernel void rel_attn_merge_gate(
    const device float* partial_out [[buffer(0)]],
    const device float2* partial_stat [[buffer(1)]],
    const device bfloat* q_proj [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant RelAttnParams& p [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    uint row = tg.y, t = row / 24u, head = row % 24u;
    uint first = row * p.parts;
    float peak = -INFINITY;
    for (uint s = lane; s < p.parts; s += 32u) peak = max(peak, partial_stat[first + s].x);
    peak = simd_max(peak);
    float z = 0.0f;
    float4 acc = float4(0.0f);
    uint comp = lane * 4u;
    for (uint s = 0; s < p.parts; ++s) {
        float2 st = partial_stat[first + s];
        float fac = fast::exp(st.x - peak);
        z = fma(st.y, fac, z);
        acc = fma(((const device float4*)(partial_out + size_t(first + s) * 128u + comp))[0], fac, acc);
    }
    float inv_z = 1.0f / z;
    uint gbase = size_t(t) * p.q_stride + p.gate_offset + head * 128u + comp;
    float4 gv = float4(((const device bfloat4*)(q_proj + gbase))[0]);
    float4 res = float4((acc.x * inv_z) * sigmoid_fn(gv.x),
                        (acc.y * inv_z) * sigmoid_fn(gv.y),
                        (acc.z * inv_z) * sigmoid_fn(gv.z),
                        (acc.w * inv_z) * sigmoid_fn(gv.w));
    ((device bfloat4*)(out + (size_t(t) * 24u + head) * 128u + comp))[0] = bfloat4(res);
}

// ============================================================================
// 7b. AMX FlashAttention-2 Prefill Kernel (BM=16, BN=32, D=128, 15.25 KB smem -> 2 TGs/CU!)
//     - 16x fewer K/V DRAM reads + Apple AMX simdgroup_bfloat8x8 for Q@K^T & P@V
//     - Fast unmasked/no-bias interior path for long-context global attention (dist >= 1024)
//     - Exact Inkling 2 relative position bias, virtual sink token, sliding window,
//       causal mask, and fused sigmoid output gate (zero partial_out roundtrip!)
//     - Grid: ((M + 15) / 16, 24, 1), TG: (128, 1, 1)
// ============================================================================
kernel void rel_attn_prefill_amx(
    const device float* q_in [[buffer(0)]],
    const device bfloat* k_cache [[buffer(1)]],
    const device bfloat* v_cache [[buffer(2)]],
    const device bfloat* table [[buffer(3)]],
    const device bfloat* q_proj [[buffer(4)]],
    device bfloat* out [[buffer(5)]],
    constant RelAttnParams& p [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    uint t0 = tg.x * 16u;
    if (t0 >= p.M) return;
    uint head = tg.y;
    uint kv_head = head / 3u;
    uint batch_q = min(16u, p.M - t0);

    // 15.25 KB shared memory (<= 16 KB -> 2 TGs/CU on M4 Pro!)
    threadgroup bfloat Qs[16 * 128];    // 4096 B
    threadgroup float kv_f32[16 * 128]; // 8192 B (aliased as KVs[32 * 128] bfloat & pv_f32[16 * 128] float!)
    threadgroup bfloat* KVs = (threadgroup bfloat*)kv_f32;
    threadgroup float Ss[16 * 32];      // 2048 B
    threadgroup bfloat Ps[16 * 32];     // 1024 B
    threadgroup float row_peak[16], row_wsum[16], row_prior[16], row_sink[16]; // 256 B

    uint c4 = (tid & 31u) * 4u;
    for (uint pass = 0u; pass < 4u; ++pass) {
        uint r = pass * 4u + sg;
        bfloat4 qv = bfloat4(bfloat(0.0f));
        if (r < batch_q) {
            uint row = (t0 + r) * 24u + head;
            qv = bfloat4(((const device float4*)(q_in + size_t(row) * 128u + c4))[0]);
        }
        ((threadgroup bfloat4*)(Qs + r * 128u + c4))[0] = qv;
    }
    if (tid < 16u) {
        row_peak[tid] = 0.0f;
        row_wsum[tid] = 1.0f;
        row_prior[tid] = 1.0f;
        row_sink[tid] = (tid < batch_q)
                            ? float(q_proj[size_t(t0 + tid) * p.q_stride + p.sink_offset + head])
                            : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint out_r = tid >> 3;          // 0..15
    uint out_c0 = (tid & 7u) * 16u; // 0, 16, ..., 112
    float O_reg[16];
    for (uint k = 0u; k < 16u; ++k) O_reg[k] = 0.0f;

    uint first_q_pos = p.pos + t0;
    uint last_q_pos = p.pos + t0 + batch_q - 1u;
    uint max_end = last_q_pos + 1u;
    uint min_start = (p.window > 0u && (first_q_pos + 1u) > p.window) ? (first_q_pos + 1u - p.window) : 0u;
    uint max_q_start = (p.window > 0u && (last_q_pos + 1u) > p.window) ? (last_q_pos + 1u - p.window) : 0u;

    const device bfloat* kh_base = k_cache + size_t(kv_head) * p.kv_capacity * 128u;
    const device bfloat* vh_base = v_cache + size_t(kv_head) * p.kv_capacity * 128u;

    uint qk_sg_m = (sg / 2u) * 8u;
    uint qk_sg_n = (sg % 2u) * 16u;
    uint pv_sg_m = (sg / 2u) * 8u;
    uint pv_sg_n = (sg % 2u) * 64u;
    uint kv_r0 = tid >> 4;          // 0..7 (8 rows per 16-byte load pass!)
    uint kv_c8 = (tid & 15u) * 8u;  // 0, 8, ..., 120 (8 bfloats = 16 bytes/thread)

    for (uint k_start = min_start; k_start < max_end; k_start += 32u) {
        for (uint pass = 0u; pass < 4u; ++pass) {
            uint r = pass * 8u + kv_r0;
            uint k_pos = k_start + r;
            uint4 kv = uint4(0u);
            if (k_pos < max_end) {
                uint slot = (p.is_ring != 0u) ? ((p.kv_capacity == 2048u) ? (k_pos & 2047u) : (k_pos % p.kv_capacity)) : k_pos;
                kv = ((const device uint4*)(kh_base + size_t(slot) * 128u + kv_c8))[0];
            }
            ((threadgroup uint4*)(KVs + r * 128u + kv_c8))[0] = kv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 s1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint d = 0u; d < 128u; d += 8u) {
            simdgroup_bfloat8x8 qm, km0, km1;
            simdgroup_load(qm, Qs + qk_sg_m * 128u + d, 128);
            simdgroup_load(km0, KVs + (qk_sg_n + 0u) * 128u + d, 128, ulong2(0), true);
            simdgroup_load(km1, KVs + (qk_sg_n + 8u) * 128u + d, 128, ulong2(0), true);
            simdgroup_multiply_accumulate(s0, qm, km0, s0);
            simdgroup_multiply_accumulate(s1, qm, km1, s1);
        }
        simdgroup_store(s0, Ss + qk_sg_m * 32u + (qk_sg_n + 0u), 32);
        simdgroup_store(s1, Ss + qk_sg_m * 32u + (qk_sg_n + 8u), 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool no_rel_bias = (batch_q == 16u) && (k_start >= max_q_start) &&
                           (k_start + 31u + 1024u <= first_q_pos);
        uint k_pos_lane = k_start + lane;
        if (no_rel_bias) {
            for (uint rr = 0u; rr < 4u; ++rr) {
                uint r = sg * 4u + rr;
                float score = Ss[r * 32u + lane] * p.scale - row_sink[r];
                float tile_max = simd_max(score);
                float prev_peak = row_peak[r];
                float next_peak = max(prev_peak, tile_max);
                float prior = fast::exp(prev_peak - next_peak);
                float prob = fast::exp(score - next_peak);
                float tile_sum = simd_sum(prob);
                Ps[r * 32u + lane] = bfloat(prob);
                if (lane == 0u) {
                    row_peak[r] = next_peak;
                    row_wsum[r] = fma(row_wsum[r], prior, tile_sum);
                    row_prior[r] = prior;
                }
            }
        } else {
            for (uint rr = 0u; rr < 4u; ++rr) {
                uint r = sg * 4u + rr;
                float score = -INFINITY;
                bool valid = false;
                if (r < batch_q) {
                    uint q_pos = p.pos + t0 + r;
                    uint q_start = (p.window > 0u && (q_pos + 1u) > p.window) ? (q_pos + 1u - p.window) : 0u;
                    valid = (k_pos_lane >= q_start) && (k_pos_lane <= q_pos);
                    if (valid) {
                        uint dist = q_pos - k_pos_lane;
                        const device bfloat* tbl_row = table + size_t((t0 + r) * 24u + head) * 1024u;
                        float bias = (dist < 1024u) ? float(tbl_row[dist]) : 0.0f;
                        score = Ss[r * 32u + lane] * p.scale + (bias - row_sink[r]);
                    }
                }
                float tile_max = simd_max(score);
                float prev_peak = row_peak[r];
                float next_peak = max(prev_peak, tile_max);
                float prior = fast::exp(prev_peak - next_peak);
                float prob = valid ? fast::exp(score - next_peak) : 0.0f;
                float tile_sum = simd_sum(prob);
                Ps[r * 32u + lane] = bfloat(prob);
                if (lane == 0u) {
                    row_peak[r] = next_peak;
                    row_wsum[r] = fma(row_wsum[r], prior, tile_sum);
                    row_prior[r] = prior;
                }
            }
        }

        for (uint pass = 0u; pass < 4u; ++pass) {
            uint r = pass * 8u + kv_r0;
            uint k_pos = k_start + r;
            uint4 vv = uint4(0u);
            if (k_pos < max_end) {
                uint slot = (p.is_ring != 0u) ? ((p.kv_capacity == 2048u) ? (k_pos & 2047u) : (k_pos % p.kv_capacity)) : k_pos;
                vv = ((const device uint4*)(vh_base + size_t(slot) * 128u + kv_c8))[0];
            }
            ((threadgroup uint4*)(KVs + r * 128u + kv_c8))[0] = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 pv[8];
        for (uint j = 0u; j < 8u; ++j) pv[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint ks = 0u; ks < 32u; ks += 8u) {
            simdgroup_bfloat8x8 pm;
            simdgroup_load(pm, Ps + pv_sg_m * 32u + ks, 32);
            for (uint j = 0u; j < 8u; ++j) {
                simdgroup_bfloat8x8 vm;
                simdgroup_load(vm, KVs + ks * 128u + pv_sg_n + j * 8u, 128);
                simdgroup_multiply_accumulate(pv[j], pm, vm, pv[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0u; j < 8u; ++j) {
            simdgroup_store(pv[j], kv_f32 + pv_sg_m * 128u + pv_sg_n + j * 8u, 128);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float prior = row_prior[out_r];
        const threadgroup float* pvr = kv_f32 + out_r * 128u + out_c0;
        for (uint k = 0u; k < 16u; ++k) {
            O_reg[k] = fma(O_reg[k], prior, pvr[k]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (out_r < batch_q) {
        uint t = t0 + out_r;
        float inv_z = 1.0f / row_wsum[out_r];
        uint gbase = size_t(t) * p.q_stride + p.gate_offset + head * 128u + out_c0;
        uint obase = (size_t(t) * 24u + head) * 128u + out_c0;
        for (uint q = 0u; q < 4u; ++q) {
            float4 gv = float4(((const device bfloat4*)(q_proj + gbase + q * 4u))[0]);
            float4 res = float4((O_reg[q * 4u + 0u] * inv_z) * sigmoid_fn(gv.x),
                                (O_reg[q * 4u + 1u] * inv_z) * sigmoid_fn(gv.y),
                                (O_reg[q * 4u + 2u] * inv_z) * sigmoid_fn(gv.z),
                                (O_reg[q * 4u + 3u] * inv_z) * sigmoid_fn(gv.w));
            ((device bfloat4*)(out + obase + q * 4u))[0] = bfloat4(res);
        }
    }
}

// ============================================================================
// 8. Residual Stream Updates
// ============================================================================
kernel void add_bf16_to_f32(
    const device bfloat4* a [[buffer(0)]],
    const device bfloat4* b [[buffer(1)]],
    device float4* out [[buffer(2)]],
    constant uint& n4 [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{ if (tid < n4) out[tid] = float4(a[tid]) + float4(b[tid]); }

kernel void add_f32_bf16_to_bf16(
    const device float4* a [[buffer(0)]],
    const device bfloat4* b [[buffer(1)]],
    device bfloat4* out [[buffer(2)]],
    constant uint& n4 [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{ if (tid < n4) out[tid] = bfloat4(a[tid] + float4(b[tid])); }

kernel void add_bf16_bf16_to_bf16(
    const device bfloat4* a [[buffer(0)]],
    const device bfloat4* b [[buffer(1)]],
    device bfloat4* out [[buffer(2)]],
    constant uint& n4 [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{ if (tid < n4) out[tid] = bfloat4(float4(a[tid]) + float4(b[tid])); }

// ============================================================================
// 9. Fast MoE Top-K + Parallel Bucketing + 64-Token Worklist (M = 1..1024!)
// ============================================================================
struct IndirectDispatchArgs { uint x, y, z; };

struct MoEWorkTile {
    ushort u;
    ushort base;
    ushort batch;
    ushort pad;
};

struct MoERouteTable {
    uint num_unique;
    uint unique_experts[128];
    uint expert_counts[128];
    uint expert_tokens[128 * 1024];
    uint expert_slots[128 * 1024];
    float expert_weights[128 * 1024];
    uchar top_ids[1024 * 8];
    float top_ws[1024 * 8];
    uint num_work_tiles;
    MoEWorkTile work_tiles[512];
    uint gpu_tiles_w13;
    uint gpu_tiles_w2;
};

struct RouterParams {
    uint M;
    uint w13_grid_x;
    uint w2_grid_x;
    float route_scale;
    uint w13_gpu_pct;
    uint w2_gpu_pct;
};

// Parallel top-8 selection across all M tokens (for prefill M > 8): 1 thread per token!
kernel void moe_topk_parallel(
    const device float* router_logits [[buffer(0)]],
    const device float* router_bias [[buffer(1)]],
    device MoERouteTable* table [[buffer(2)]],
    constant RouterParams& p [[buffer(3)]],
    uint m [[thread_position_in_grid]])
{
    if (m >= p.M) return;
    const device float* lp = router_logits + size_t(m) * 128u;
    uint chosen_id[8];
    float chosen_sel[8];
    for (uint k = 0; k < 8u; ++k) { chosen_id[k] = 0u; chosen_sel[k] = -INFINITY; }
    for (uint ex = 0; ex < 128u; ++ex) {
        float s = sigmoid_fn(lp[ex]) + router_bias[ex];
        if (s > chosen_sel[7]) {
            uint pos = 7u;
            while (pos > 0u && s > chosen_sel[pos - 1u]) {
                chosen_sel[pos] = chosen_sel[pos - 1u];
                chosen_id[pos] = chosen_id[pos - 1u];
                pos--;
            }
            chosen_sel[pos] = s;
            chosen_id[pos] = ex;
        }
    }
    float lw[8];
    float mx = -INFINITY;
    for (uint k = 0; k < 8u; ++k) {
        float lg = lp[chosen_id[k]];
        float l = (lg >= 0.0f) ? -log(1.0f + exp(-lg)) : (lg - log(1.0f + exp(lg)));
        lw[k] = l;
        mx = max(mx, l);
    }
    float wsum = 0.0f;
    for (uint k = 0; k < 8u; ++k) { lw[k] = exp(lw[k] - mx); wsum += lw[k]; }
    float scale = p.route_scale / wsum;
    for (uint k = 0; k < 8u; ++k) {
        table->top_ids[m * 8u + k] = uchar(chosen_id[k]);
        table->top_ws[m * 8u + k] = lw[k] * scale;
    }
}

// Pass 1: 128-TG parallel expert bucketing (1 TG per expert e, 128 threads scan M tokens in <=8 steps!)
// Writes directly to expert-indexed table->expert_tokens[e * 1024 + j] and table->expert_counts[e].
kernel void moe_group_bucket_parallel(
    device MoERouteTable* table [[buffer(0)]],
    constant RouterParams& p [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint e [[threadgroup_position_in_grid]])
{
    uint M = p.M;
    ushort my_tok[8];
    uchar my_slot[8];
    uint my_cnt = 0u;
    for (uint m = tid; m < M; m += 128u) {
        uint base8 = m * 8u;
        for (uint k = 0; k < 8u; ++k) {
            if (uint(table->top_ids[base8 + k]) == e) {
                my_tok[my_cnt] = ushort(m);
                my_slot[my_cnt] = uchar(k);
                my_cnt++;
                break;
            }
        }
    }
    uint sg_off = simd_prefix_exclusive_sum(my_cnt);
    uint sg_sum = simd_sum(my_cnt);
    threadgroup uint sg_totals[4];
    if (lane == 0u) sg_totals[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base_off = 0u;
    for (uint s = 0u; s < sg; ++s) base_off += sg_totals[s];
    uint out_idx = base_off + sg_off;
    uint e_base = e * 1024u;
    for (uint i = 0u; i < my_cnt; ++i) {
        uint m = uint(my_tok[i]);
        uint k = uint(my_slot[i]);
        table->expert_tokens[e_base + out_idx + i] = m;
        table->expert_slots[e_base + out_idx + i] = k;
        table->expert_weights[e_base + out_idx + i] = table->top_ws[m * 8u + k];
    }
    if (tid == 0u) {
        table->expert_counts[e] = sg_totals[0] + sg_totals[1] + sg_totals[2] + sg_totals[3];
    }
}

// Pass 2: 1-TG 64-token load-balanced worklist builder (splits hot experts >64 into balanced BM=64 tiles!)
kernel void moe_build_worklist_64(
    device MoERouteTable* table [[buffer(0)]],
    device IndirectDispatchArgs* ind_w13 [[buffer(1)]],
    device IndirectDispatchArgs* ind_w2 [[buffer(2)]],
    constant RouterParams& p [[buffer(3)]],
    uint e [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    uint cnt = table->expert_counts[e];
    uint nt = (cnt + 63u) >> 6;
    uint sg_off = simd_prefix_exclusive_sum(nt);
    uint sg_sum = simd_sum(nt);
    threadgroup uint sg_totals[4];
    if (lane == 0u) sg_totals[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base_off = 0u;
    for (uint s = 0u; s < sg; ++s) base_off += sg_totals[s];
    uint tile_off = base_off + sg_off;
    for (uint t = 0u; t < nt; ++t) {
        uint b = t * 64u;
        table->work_tiles[tile_off + t] = MoEWorkTile{ushort(e), ushort(b), ushort(min(64u, cnt - b)), 0u};
    }
    if (e == 0u) {
        uint total_tiles = sg_totals[0] + sg_totals[1] + sg_totals[2] + sg_totals[3];
        table->num_work_tiles = total_tiles;

        // When splitting with CPU SME2, partition small tiles (batch < 48) first for GPU's 4-bucket AMX
        // and fat tiles (batch >= 48, mostly 64) last for SME2 -> maximizes SME2 weight-unpack reuse!
        if (p.w13_gpu_pct < 100u || p.w2_gpu_pct < 100u) {
            uint write_pos = 0u;
            for (uint ti = 0u; ti < total_tiles; ++ti) {
                if (table->work_tiles[ti].batch < 48u) {
                    if (ti != write_pos) {
                        MoEWorkTile tmp = table->work_tiles[write_pos];
                        table->work_tiles[write_pos] = table->work_tiles[ti];
                        table->work_tiles[ti] = tmp;
                    }
                    write_pos++;
                }
            }
        }

        uint gt13 = total_tiles;
        uint gt2  = total_tiles;
        if (p.w13_gpu_pct < 100u) {
            uint target13 = (p.M * 8u * p.w13_gpu_pct) / 100u;
            uint acc = 0u;
            gt13 = 0u;
            for (uint ti = 0u; ti < total_tiles; ++ti) {
                if (acc >= target13) break;
                acc += uint(table->work_tiles[ti].batch);
                gt13 = ti + 1u;
            }
        }
        if (p.w2_gpu_pct < 100u) {
            uint target2 = (p.M * 8u * p.w2_gpu_pct) / 100u;
            uint acc = 0u;
            gt2 = 0u;
            for (uint ti = 0u; ti < total_tiles; ++ti) {
                if (acc >= target2) break;
                acc += uint(table->work_tiles[ti].batch);
                gt2 = ti + 1u;
            }
        }
        table->gpu_tiles_w13 = gt13;
        table->gpu_tiles_w2  = gt2;
        ind_w13->x = p.w13_grid_x; ind_w13->y = gt13; ind_w13->z = 1u;
        ind_w2->x  = p.w2_grid_x;  ind_w2->y  = gt2;  ind_w2->z  = 1u;
    }
}

kernel void moe_topk_and_group(
    const device float* router_logits [[buffer(0)]], // [M, 128] F32 from parallel linear
    const device float* router_bias [[buffer(1)]],   // [128] F32
    device MoERouteTable* table [[buffer(2)]],
    device IndirectDispatchArgs* ind_w13 [[buffer(3)]],
    device IndirectDispatchArgs* ind_w2 [[buffer(4)]],
    constant RouterParams& p [[buffer(5)]],
    uint e [[thread_index_in_threadgroup]])
{
    uint M = p.M; // 1..8 (decode/verify)
    threadgroup float tg_bias[128];
    tg_bias[e] = router_bias[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup uchar top_ids[16][8];
    threadgroup float top_ws[16][8];
    for (uint m = e; m < M; m += 128u) {
        const device float* lp = router_logits + size_t(m) * 128u;
        uint chosen_id[8];
        float chosen_sel[8];
        for (uint k = 0; k < 8u; ++k) { chosen_id[k] = 0u; chosen_sel[k] = -INFINITY; }
        for (uint ex = 0; ex < 128u; ++ex) {
            float s = sigmoid_fn(lp[ex]) + tg_bias[ex];
            if (s > chosen_sel[7]) {
                uint pos = 7u;
                while (pos > 0u && s > chosen_sel[pos - 1u]) {
                    chosen_sel[pos] = chosen_sel[pos - 1u];
                    chosen_id[pos] = chosen_id[pos - 1u];
                    pos--;
                }
                chosen_sel[pos] = s;
                chosen_id[pos] = ex;
            }
        }
        float lw[8];
        float mx = -INFINITY;
        for (uint k = 0; k < 8u; ++k) {
            float lg = lp[chosen_id[k]];
            float l = (lg >= 0.0f) ? -log(1.0f + exp(-lg)) : (lg - log(1.0f + exp(lg)));
            lw[k] = l;
            mx = max(mx, l);
        }
        float wsum = 0.0f;
        for (uint k = 0; k < 8u; ++k) { lw[k] = exp(lw[k] - mx); wsum += lw[k]; }
        float scale = p.route_scale / wsum;
        for (uint k = 0; k < 8u; ++k) { top_ids[m][k] = uchar(chosen_id[k]); top_ws[m][k] = lw[k] * scale; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cnt = 0u;
    for (uint m = 0; m < M; ++m) {
        for (uint k = 0; k < 8u; ++k) {
            if (uint(top_ids[m][k]) == e) cnt++;
        }
    }

    threadgroup uint active_flag[128], prefix[128];
    active_flag[e] = (cnt > 0u) ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (e == 0u) {
        uint total = 0u;
        for (uint i = 0; i < 128u; ++i) { prefix[i] = total; total += active_flag[i]; }
        table->num_unique = total;
        ind_w13->x = p.w13_grid_x; ind_w13->y = total; ind_w13->z = 1u;
        ind_w2->x  = p.w2_grid_x;  ind_w2->y  = total; ind_w2->z  = 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (cnt > 0u) {
        uint u = prefix[e];
        table->unique_experts[u] = e;
        table->expert_counts[u] = cnt;
        uint j = 0u;
        for (uint m = 0; m < M; ++m) {
            for (uint k = 0; k < 8u; ++k) {
                if (uint(top_ids[m][k]) == e) {
                    table->expert_tokens[u * 1024u + j] = m;
                    table->expert_slots[u * 1024u + j] = k;
                    table->expert_weights[u * 1024u + j] = top_ws[m][k];
                    j++;
                }
            }
        }
    }
}

// ============================================================================
// 10. Activation Quantization (Exact NVFP4 4-over-6 & Exact MXFP8)
// ============================================================================
// Exact NVFP4 4-over-6 activation quantizer -> writes `half` (100% bit-exact for E2M1 x E4M3FN!)
kernel void nvfp4_quant_act_4over6(
    const device bfloat* x [[buffer(0)]],
    device half* inner [[buffer(1)]],
    device float* amax [[buffer(2)]],
    constant uint& W [[buffer(3)]],
    uint worker [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    uint nblocks = W / 16u;
    bool active = worker < nblocks;
    uint start = row * W + worker * 16u;
    float data[16];
    float block_max = 0.0f;
    for (uint i = 0; i < 16u; ++i) {
        data[i] = active ? float(x[start + i]) : 0.0f;
        block_max = max(block_max, abs(data[i]));
    }
    threadgroup float partial[8];
    float sm = simd_max(block_max);
    if (lane == 0u) partial[sg] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsgs = (nblocks + 31u) / 32u;
    float row_max = simd_max(lane < nsgs ? partial[lane] : 0.0f);
    if (worker == 0u) amax[row] = row_max;
    if (!active) return;

    float normalizer = min(1536.0f / row_max, FLT_MAX);
    float decoder = 1.0f / normalizer;
    float outer = row_max * (1.0f / 1536.0f);
    float scale_a = round_e4m3(min(block_max * (normalizer * (1.0f / 6.0f)), 448.0f));
    float scale_b = round_e4m3(min(block_max * (normalizer * 0.25f), 448.0f));
    float inv_a = min(1.0f / (scale_a * decoder), FLT_MAX);
    float inv_b = min(1.0f / (scale_b * decoder), FLT_MAX);
    float cand_a[16], cand_b[16];
    float err_a = 0.0f, err_b = 0.0f;
    for (uint i = 0; i < 16u; ++i) {
        cand_a[i] = round_e2m1(data[i] * inv_a) * scale_a;
        cand_b[i] = round_e2m1(data[i] * inv_b) * scale_b;
        err_a += abs(data[i] - cand_a[i] * outer);
        err_b += abs(data[i] - cand_b[i] * outer);
    }
    for (uint i = 0; i < 16u; ++i) inner[start + i] = half((err_b < err_a) ? cand_b[i] : cand_a[i]);
}

kernel void mxfp8_quant_act(
    const device bfloat* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& W [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    uint start = tg.y * W + tg.x * 32u;
    float v = float(x[start + lane]);
    float bmax = simd_max(abs(v));
    float raw_scale = bmax * (1.0f / 448.0f);
    uint bits = as_type<uint>(raw_scale);
    int exponent = int(bits >> 23);
    if ((bits & 0x7FFFFFu) > 0u) exponent += 1;
    exponent = min(exponent, 254);
    float minimum = as_type<float>(0x00800000u) * 0.5f;
    if (raw_scale <= minimum) exponent = 0;
    float scale = (exponent == 0) ? minimum : as_type<float>(uint(exponent) << 23);
    float inv = (exponent == 0) ? (1.0f / minimum) : (1.0f / scale);
    uchar q = to_fp8_e4m3_signed(clamp(v * inv, -448.0f, 448.0f));
    out[start + lane] = fp8_e4m3(q) * scale;
}

// ============================================================================
// 11. HIGH-BANDWIDTH COALESCED NVFP4 MoE KERNELS (v4 — 218 GB/s on M4 Pro!)
//     - ZERO threadgroup memory, ZERO barriers -> MAX GPU occupancy!
//     - half4 bit-exact NVFP4 activations -> 2x less L1 traffic!
//     - W13: 1 pair/SG (grid_x=288), uint2 (16 elems/lane); dedicated M=1 kernel!
//     - W2:  4 rows/SG (grid_x=144), uint1 (8 elems/lane) -> 205 GB/s!
// ============================================================================
struct MoEQuantParams {
    uint K, N;
    float outer_denom, gate_cap, up_cap;
};

// Dedicated M == 1 W13 NVFP4 kernel: 2 pairs/SG (grid_x=144), ZERO barriers, ZERO tg mem, 2x L1 act reuse!
kernel void moe_w13_nvfp4_m1(
    const device half* act_h [[buffer(0)]],
    const device float* act_amax [[buffer(1)]],
    const device uint* w_packed [[buffer(2)]],
    const device uchar* w_scales [[buffer(3)]],
    const device float* w_amax [[buffer(4)]],
    const device MoERouteTable* table [[buffer(5)]],
    device bfloat* mid_out [[buffer(6)]],
    constant MoEQuantParams& p [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    uint u = tg.y;
    uint expert = table->unique_experts[u];
    uint pair0 = (tg.x * 4u + sg) * 2u;
    if (pair0 >= 1152u) return;

    uint slot = table->expert_slots[u * 1024u];
    float rw = table->expert_weights[u * 1024u];
    float outer = act_amax[0] * w_amax[expert] * (16384.0f / (p.outer_denom * 1536.0f));

    uint gr = pair0 * 2u;
    const device uint2* gw0 = (const device uint2*)(w_packed + (size_t(expert) * 2304u + gr) * 288u);
    const device uchar* gs0 = w_scales + (size_t(expert) * 2304u + gr) * 144u;
    const device uint2* uw0 = gw0 + 144u;
    const device uchar* us0 = gs0 + 144u;
    const device uint2* gw1 = gw0 + 288u;
    const device uchar* gs1 = gs0 + 288u;
    const device uint2* uw1 = gw0 + 432u;
    const device uchar* us1 = gs0 + 432u;
    const device half4* xp = (const device half4*)act_h;

    float gsum0 = 0.0f, usum0 = 0.0f, gsum1 = 0.0f, usum1 = 0.0f;
    for (uint blk = lane; blk < 144u; blk += 32u) {
        float4 x0 = float4(xp[blk * 4u + 0u]);
        float4 x1 = float4(xp[blk * 4u + 1u]);
        float4 x2 = float4(xp[blk * 4u + 2u]);
        float4 x3 = float4(xp[blk * 4u + 3u]);
        uint2 w = gw0[blk];
        gsum0 = fma(dot(x0, decode4_e2m1_scaled(w.x)) + dot(x1, decode4_e2m1_scaled(w.x >> 16)) +
                    dot(x2, decode4_e2m1_scaled(w.y)) + dot(x3, decode4_e2m1_scaled(w.y >> 16)),
                    fp8_e4m3(gs0[blk]), gsum0);
        w = uw0[blk];
        usum0 = fma(dot(x0, decode4_e2m1_scaled(w.x)) + dot(x1, decode4_e2m1_scaled(w.x >> 16)) +
                    dot(x2, decode4_e2m1_scaled(w.y)) + dot(x3, decode4_e2m1_scaled(w.y >> 16)),
                    fp8_e4m3(us0[blk]), usum0);
        w = gw1[blk];
        gsum1 = fma(dot(x0, decode4_e2m1_scaled(w.x)) + dot(x1, decode4_e2m1_scaled(w.x >> 16)) +
                    dot(x2, decode4_e2m1_scaled(w.y)) + dot(x3, decode4_e2m1_scaled(w.y >> 16)),
                    fp8_e4m3(gs1[blk]), gsum1);
        w = uw1[blk];
        usum1 = fma(dot(x0, decode4_e2m1_scaled(w.x)) + dot(x1, decode4_e2m1_scaled(w.x >> 16)) +
                    dot(x2, decode4_e2m1_scaled(w.y)) + dot(x3, decode4_e2m1_scaled(w.y >> 16)),
                    fp8_e4m3(us1[blk]), usum1);
    }
    float gv0 = simd_sum(gsum0) * outer;
    float uv0 = simd_sum(usum0) * outer;
    float gv1 = simd_sum(gsum1) * outer;
    float uv1 = simd_sum(usum1) * outer;
    if (lane == 0u) {
        gv0 = float(bfloat(gv0)); uv0 = float(bfloat(uv0));
        mid_out[slot * 1152u + pair0]      = bfloat(softcap_fn(gv0, p.gate_cap) * sigmoid_fn(gv0) * softcap_fn(uv0, p.up_cap) * rw);
        gv1 = float(bfloat(gv1)); uv1 = float(bfloat(uv1));
        mid_out[slot * 1152u + pair0 + 1u] = bfloat(softcap_fn(gv1, p.gate_cap) * sigmoid_fn(gv1) * softcap_fn(uv1, p.up_cap) * rw);
    }
}

// Multi-token W13 NVFP4 v4 (vD winner: 218 GB/s, ZERO barriers, half4 L1 act, uint2 weights!)
kernel void moe_w13_nvfp4_coalesced(
    const device half* act_h [[buffer(0)]],
    const device float* act_amax [[buffer(1)]],
    const device uint* w_packed [[buffer(2)]],
    const device uchar* w_scales [[buffer(3)]],
    const device float* w_amax [[buffer(4)]],
    const device MoERouteTable* table [[buffer(5)]],
    device bfloat* mid_out [[buffer(6)]],
    constant MoEQuantParams& p [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    uint u = tg.y;
    uint expert = table->unique_experts[u];
    uint cnt = table->expert_counts[u];
    uint pair = tg.x * 4u + sg;
    if (pair >= 1152u) return;

    uint gr = pair * 2u;
    const device uint2* gw = (const device uint2*)(w_packed + (size_t(expert) * 2304u + gr) * 288u);
    const device uchar* gs = w_scales + (size_t(expert) * 2304u + gr) * 144u;
    const device uint2* uw = gw + 144u;
    const device uchar* us = gs + 144u;

    float wmax = w_amax[expert];
    float inv_denom = 16384.0f / (p.outer_denom * 1536.0f);

    if (cnt == 1u) {
        uint tok = table->expert_tokens[u * 1024u];
        uint slot = table->expert_slots[u * 1024u];
        float rw = table->expert_weights[u * 1024u];
        float outer = act_amax[tok] * wmax * inv_denom;
        const device half4* xp = (const device half4*)(act_h + size_t(tok) * 2304u);
        float gsum = 0.0f, usum = 0.0f;
        for (uint blk = lane; blk < 144u; blk += 32u) {
            float4 x0 = float4(xp[blk * 4u + 0u]);
            float4 x1 = float4(xp[blk * 4u + 1u]);
            float4 x2 = float4(xp[blk * 4u + 2u]);
            float4 x3 = float4(xp[blk * 4u + 3u]);
            uint2 gww = gw[blk];
            gsum = fma(dot(x0, decode4_e2m1_scaled(gww.x)) + dot(x1, decode4_e2m1_scaled(gww.x >> 16)) +
                       dot(x2, decode4_e2m1_scaled(gww.y)) + dot(x3, decode4_e2m1_scaled(gww.y >> 16)),
                       fp8_e4m3(gs[blk]), gsum);
            uint2 uww = uw[blk];
            usum = fma(dot(x0, decode4_e2m1_scaled(uww.x)) + dot(x1, decode4_e2m1_scaled(uww.x >> 16)) +
                       dot(x2, decode4_e2m1_scaled(uww.y)) + dot(x3, decode4_e2m1_scaled(uww.y >> 16)),
                       fp8_e4m3(us[blk]), usum);
        }
        float gv = simd_sum(gsum) * outer;
        float uv = simd_sum(usum) * outer;
        if (lane == 0u) {
            gv = float(bfloat(gv)); uv = float(bfloat(uv));
            mid_out[(tok * 8u + slot) * 1152u + pair] = bfloat(softcap_fn(gv, p.gate_cap) * sigmoid_fn(gv) * softcap_fn(uv, p.up_cap) * rw);
        }
        return;
    }

    for (uint base = 0u; base < cnt; base += 4u) {
        uint batch = min(4u, cnt - base);
        uint toks[4], slots[4];
        float rw[4], outer[4];
        const device half4* xp[4];
        float gsum[4] = {0,0,0,0}, usum[4] = {0,0,0,0};
        for (uint j = 0; j < batch; ++j) {
            toks[j] = table->expert_tokens[u * 1024u + base + j];
            slots[j] = table->expert_slots[u * 1024u + base + j];
            rw[j] = table->expert_weights[u * 1024u + base + j];
            outer[j] = act_amax[toks[j]] * wmax * inv_denom;
            xp[j] = (const device half4*)(act_h + size_t(toks[j]) * 2304u);
        }
        for (uint blk = lane; blk < 144u; blk += 32u) {
            uint2 gww = gw[blk]; float gsc = fp8_e4m3(gs[blk]);
            float4 g0 = decode4_e2m1_scaled(gww.x), g1 = decode4_e2m1_scaled(gww.x >> 16);
            float4 g2 = decode4_e2m1_scaled(gww.y), g3 = decode4_e2m1_scaled(gww.y >> 16);
            uint2 uww = uw[blk]; float usc = fp8_e4m3(us[blk]);
            float4 u0 = decode4_e2m1_scaled(uww.x), u1 = decode4_e2m1_scaled(uww.x >> 16);
            float4 u2 = decode4_e2m1_scaled(uww.y), u3 = decode4_e2m1_scaled(uww.y >> 16);
            for (uint j = 0; j < batch; ++j) {
                float4 x0 = float4(xp[j][blk * 4u + 0u]), x1 = float4(xp[j][blk * 4u + 1u]);
                float4 x2 = float4(xp[j][blk * 4u + 2u]), x3 = float4(xp[j][blk * 4u + 3u]);
                gsum[j] = fma(dot(x0, g0) + dot(x1, g1) + dot(x2, g2) + dot(x3, g3), gsc, gsum[j]);
                usum[j] = fma(dot(x0, u0) + dot(x1, u1) + dot(x2, u2) + dot(x3, u3), usc, usum[j]);
            }
        }
        for (uint j = 0; j < batch; ++j) {
            float gv = simd_sum(gsum[j]) * outer[j];
            float uv = simd_sum(usum[j]) * outer[j];
            if (lane == 0u) {
                gv = float(bfloat(gv)); uv = float(bfloat(uv));
                mid_out[(toks[j] * 8u + slots[j]) * 1152u + pair] = bfloat(softcap_fn(gv, p.gate_cap) * sigmoid_fn(gv) * softcap_fn(uv, p.up_cap) * rw[j]);
            }
        }
    }
}

// W2 NVFP4 v4 (vF winner: 205 GB/s! 4 rows/SG -> grid_x = 144, uint1 = 8 elems/lane, half4 L1 act, ZERO barriers!)
kernel void moe_w2_nvfp4_coalesced(
    const device half* mid_h [[buffer(0)]],
    const device float* mid_amax [[buffer(1)]],
    const device uint* w_packed [[buffer(2)]],
    const device uchar* w_scales [[buffer(3)]],
    const device float* w_amax [[buffer(4)]],
    const device MoERouteTable* table [[buffer(5)]],
    device float* down_out [[buffer(6)]],
    constant MoEQuantParams& p [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    uint u = tg.y;
    uint expert = table->unique_experts[u];
    uint cnt = table->expert_counts[u];
    uint out0 = (tg.x * 4u + sg) * 4u;
    if (out0 >= 2304u) return;

    const device uint* wb = w_packed + (size_t(expert) * 2304u + out0) * 144u;
    const device uchar* sb_base = w_scales + (size_t(expert) * 2304u + out0) * 72u;
    float wmax = w_amax[expert];
    float inv_denom = 16384.0f / (p.outer_denom * 1536.0f);

    if (cnt == 1u) {
        uint ri = table->expert_tokens[u * 1024u] * 8u + table->expert_slots[u * 1024u];
        float outer = mid_amax[ri] * wmax * inv_denom;
        const device half4* xp = (const device half4*)(mid_h + size_t(ri) * 1152u);
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (uint w = lane; w < 144u; w += 32u) {
            float4 x0 = float4(xp[w * 2u]), x1 = float4(xp[w * 2u + 1u]);
            uint sb = w >> 1;
            uint q;
            q = wb[0u*144u + w]; a0 = fma(dot(x0, decode4_e2m1_scaled(q)) + dot(x1, decode4_e2m1_scaled(q >> 16)), fp8_e4m3(sb_base[0u*72u + sb]), a0);
            q = wb[1u*144u + w]; a1 = fma(dot(x0, decode4_e2m1_scaled(q)) + dot(x1, decode4_e2m1_scaled(q >> 16)), fp8_e4m3(sb_base[1u*72u + sb]), a1);
            q = wb[2u*144u + w]; a2 = fma(dot(x0, decode4_e2m1_scaled(q)) + dot(x1, decode4_e2m1_scaled(q >> 16)), fp8_e4m3(sb_base[2u*72u + sb]), a2);
            q = wb[3u*144u + w]; a3 = fma(dot(x0, decode4_e2m1_scaled(q)) + dot(x1, decode4_e2m1_scaled(q >> 16)), fp8_e4m3(sb_base[3u*72u + sb]), a3);
        }
        float v0 = simd_sum(a0) * outer, v1 = simd_sum(a1) * outer;
        float v2 = simd_sum(a2) * outer, v3 = simd_sum(a3) * outer;
        if (lane == 0u) {
            down_out[size_t(ri) * 2304u + out0 + 0u] = float(bfloat(v0));
            down_out[size_t(ri) * 2304u + out0 + 1u] = float(bfloat(v1));
            down_out[size_t(ri) * 2304u + out0 + 2u] = float(bfloat(v2));
            down_out[size_t(ri) * 2304u + out0 + 3u] = float(bfloat(v3));
        }
        return;
    }

    for (uint base = 0u; base < cnt; base += 2u) {
        bool has1 = (base + 1u) < cnt;
        uint ri0 = table->expert_tokens[u * 1024u + base] * 8u + table->expert_slots[u * 1024u + base];
        uint ri1 = table->expert_tokens[u * 1024u + min(base + 1u, cnt - 1u)] * 8u + table->expert_slots[u * 1024u + min(base + 1u, cnt - 1u)];
        float out_sc0 = mid_amax[ri0] * wmax * inv_denom;
        float out_sc1 = mid_amax[ri1] * wmax * inv_denom;
        const device half4* xp0 = (const device half4*)(mid_h + size_t(ri0) * 1152u);
        const device half4* xp1 = (const device half4*)(mid_h + size_t(ri1) * 1152u);
        float a0[4] = {0,0,0,0}, a1[4] = {0,0,0,0};
        for (uint w = lane; w < 144u; w += 32u) {
            uint sb = w >> 1;
            float4 x0 = float4(xp0[w * 2u]), x1 = float4(xp0[w * 2u + 1u]);
            float4 y0 = has1 ? float4(xp1[w * 2u]) : float4(0), y1 = has1 ? float4(xp1[w * 2u + 1u]) : float4(0);
            for (uint r = 0; r < 4u; ++r) {
                uint q = wb[r * 144u + w]; float sc = fp8_e4m3(sb_base[r * 72u + sb]);
                float4 wa = decode4_e2m1_scaled(q), wbv = decode4_e2m1_scaled(q >> 16);
                a0[r] = fma(dot(x0, wa) + dot(x1, wbv), sc, a0[r]);
                if (has1) a1[r] = fma(dot(y0, wa) + dot(y1, wbv), sc, a1[r]);
            }
        }
        for (uint r = 0; r < 4u; ++r) {
            float v0 = simd_sum(a0[r]) * out_sc0;
            if (lane == 0u) down_out[size_t(ri0) * 2304u + out0 + r] = float(bfloat(v0));
            if (has1) {
                float v1 = simd_sum(a1[r]) * out_sc1;
                if (lane == 0u) down_out[size_t(ri1) * 2304u + out0 + r] = float(bfloat(v1));
            }
        }
    }
}

// ============================================================================
// 12. HIGH-BANDWIDTH COALESCED MXFP8 MoE KERNELS (L20..23 — v4: ZERO barriers, ZERO tg_act!)
// ============================================================================
kernel void moe_w13_mxfp8_coalesced(
    const device float* act [[buffer(0)]],
    const device uint* w_packed [[buffer(1)]],
    const device uchar* w_scales [[buffer(2)]],
    const device MoERouteTable* table [[buffer(3)]],
    device bfloat* mid_out [[buffer(4)]],
    constant MoEQuantParams& p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    uint u = tg.y;
    uint expert = table->unique_experts[u];
    uint cnt = table->expert_counts[u];
    uint pair = tg.x * 4u + sg;
    if (pair >= 1152u) return;

    uint gr = pair * 2u, ur = gr + 1u;
    const device uint2* gw = (const device uint2*)(w_packed + (size_t(expert) * 2304u + gr) * 576u);
    const device uchar* gs = w_scales + (size_t(expert) * 2304u + gr) * 72u;
    const device uint2* uw = (const device uint2*)(w_packed + (size_t(expert) * 2304u + ur) * 576u);
    const device uchar* us = w_scales + (size_t(expert) * 2304u + ur) * 72u;

    if (cnt == 1u) {
        uint tok = table->expert_tokens[u * 1024u];
        uint slot = table->expert_slots[u * 1024u];
        float rw = table->expert_weights[u * 1024u];
        const device float4* xp = (const device float4*)(act + size_t(tok) * 2304u);
        float gsum = 0.0f, usum = 0.0f;
        for (uint blk = lane; blk < 288u; blk += 32u) {
            uint sb = blk >> 2;
            float4 x0 = xp[blk * 2u + 0u];
            float4 x1 = xp[blk * 2u + 1u];
            uint2 gww = gw[blk];
            gsum = fma(dot(x0, decode4_fp8_scaled(gww.x)) + dot(x1, decode4_fp8_scaled(gww.y)),
                       mxfp8_e8m0(gs[sb]) * 256.0f, gsum);
            uint2 uww = uw[blk];
            usum = fma(dot(x0, decode4_fp8_scaled(uww.x)) + dot(x1, decode4_fp8_scaled(uww.y)),
                       mxfp8_e8m0(us[sb]) * 256.0f, usum);
        }
        float gv = float(bfloat(simd_sum(gsum)));
        float uv = float(bfloat(simd_sum(usum)));
        if (lane == 0u) {
            float av = softcap_fn(gv, p.gate_cap) * sigmoid_fn(gv) * softcap_fn(uv, p.up_cap) * rw;
            mid_out[(tok * 8u + slot) * 1152u + pair] = bfloat(av);
        }
        return;
    }

    for (uint base = 0u; base < cnt; base += 4u) {
        uint batch = min(4u, cnt - base);
        uint toks[4], slots[4];
        const device float4* xp[4];
        float rw[4], gsum[4] = {0,0,0,0}, usum[4] = {0,0,0,0};
        for (uint j = 0; j < batch; ++j) {
            toks[j] = table->expert_tokens[u * 1024u + base + j];
            slots[j] = table->expert_slots[u * 1024u + base + j];
            rw[j] = table->expert_weights[u * 1024u + base + j];
            xp[j] = (const device float4*)(act + size_t(toks[j]) * 2304u);
        }
        for (uint blk = lane; blk < 288u; blk += 32u) {
            uint sb = blk >> 2;
            uint2 gww = gw[blk]; float gsc = mxfp8_e8m0(gs[sb]) * 256.0f;
            float4 g0 = decode4_fp8_scaled(gww.x), g1 = decode4_fp8_scaled(gww.y);
            uint2 uww = uw[blk]; float usc = mxfp8_e8m0(us[sb]) * 256.0f;
            float4 u0 = decode4_fp8_scaled(uww.x), u1 = decode4_fp8_scaled(uww.y);
            for (uint j = 0; j < batch; ++j) {
                float4 x0 = xp[j][blk * 2u + 0u];
                float4 x1 = xp[j][blk * 2u + 1u];
                gsum[j] = fma(dot(x0, g0) + dot(x1, g1), gsc, gsum[j]);
                usum[j] = fma(dot(x0, u0) + dot(x1, u1), usc, usum[j]);
            }
        }
        for (uint j = 0; j < batch; ++j) {
            float gv = float(bfloat(simd_sum(gsum[j])));
            float uv = float(bfloat(simd_sum(usum[j])));
            if (lane == 0u) {
                float av = softcap_fn(gv, p.gate_cap) * sigmoid_fn(gv) * softcap_fn(uv, p.up_cap) * rw[j];
                mid_out[(toks[j] * 8u + slots[j]) * 1152u + pair] = bfloat(av);
            }
        }
    }
}

kernel void moe_w2_mxfp8_coalesced(
    const device float* mid_act [[buffer(0)]],
    const device uint* w_packed [[buffer(1)]],
    const device uchar* w_scales [[buffer(2)]],
    const device MoERouteTable* table [[buffer(3)]],
    device float* down_out [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    uint u = tg.y;
    uint expert = table->unique_experts[u];
    uint cnt = table->expert_counts[u];
    uint out0 = (tg.x * 4u + sg) * 2u;
    if (out0 >= 2304u) return;

    const device uint2* w0 = (const device uint2*)(w_packed + (size_t(expert) * 2304u + out0) * 288u);
    const device uchar* s0 = w_scales + (size_t(expert) * 2304u + out0) * 36u;
    const device uint2* w1 = (const device uint2*)(w_packed + (size_t(expert) * 2304u + out0 + 1u) * 288u);
    const device uchar* s1 = w_scales + (size_t(expert) * 2304u + out0 + 1u) * 36u;

    if (cnt == 1u) {
        uint ri = table->expert_tokens[u * 1024u] * 8u + table->expert_slots[u * 1024u];
        const device float4* xp = (const device float4*)(mid_act + size_t(ri) * 1152u);
        float a0 = 0.0f, a1 = 0.0f;
        for (uint blk = lane; blk < 144u; blk += 32u) {
            uint sb = blk >> 2;
            float4 x0 = xp[blk * 2u + 0u];
            float4 x1 = xp[blk * 2u + 1u];
            uint2 ww0 = w0[blk];
            a0 = fma(dot(x0, decode4_fp8_scaled(ww0.x)) + dot(x1, decode4_fp8_scaled(ww0.y)),
                     mxfp8_e8m0(s0[sb]) * 256.0f, a0);
            uint2 ww1 = w1[blk];
            a1 = fma(dot(x0, decode4_fp8_scaled(ww1.x)) + dot(x1, decode4_fp8_scaled(ww1.y)),
                     mxfp8_e8m0(s1[sb]) * 256.0f, a1);
        }
        float v0 = simd_sum(a0), v1 = simd_sum(a1);
        if (lane == 0u) {
            down_out[size_t(ri) * 2304u + out0] = float(bfloat(v0));
            down_out[size_t(ri) * 2304u + out0 + 1u] = float(bfloat(v1));
        }
        return;
    }

    for (uint base = 0u; base < cnt; base += 4u) {
        uint batch = min(4u, cnt - base);
        uint ris[4];
        const device float4* xp[4];
        float a0[4] = {0,0,0,0}, a1[4] = {0,0,0,0};
        for (uint j = 0; j < batch; ++j) {
            ris[j] = table->expert_tokens[u * 1024u + base + j] * 8u + table->expert_slots[u * 1024u + base + j];
            xp[j] = (const device float4*)(mid_act + size_t(ris[j]) * 1152u);
        }
        for (uint blk = lane; blk < 144u; blk += 32u) {
            uint sb = blk >> 2;
            uint2 ww0 = w0[blk]; float sc0 = mxfp8_e8m0(s0[sb]) * 256.0f;
            float4 r0 = decode4_fp8_scaled(ww0.x), r1 = decode4_fp8_scaled(ww0.y);
            uint2 ww1 = w1[blk]; float sc1 = mxfp8_e8m0(s1[sb]) * 256.0f;
            float4 q0 = decode4_fp8_scaled(ww1.x), q1 = decode4_fp8_scaled(ww1.y);
            for (uint j = 0; j < batch; ++j) {
                float4 x0 = xp[j][blk * 2u + 0u];
                float4 x1 = xp[j][blk * 2u + 1u];
                a0[j] = fma(dot(x0, r0) + dot(x1, r1), sc0, a0[j]);
                a1[j] = fma(dot(x0, q0) + dot(x1, q1), sc1, a1[j]);
            }
        }
        for (uint j = 0; j < batch; ++j) {
            float v0 = simd_sum(a0[j]), v1 = simd_sum(a1[j]);
            if (lane == 0u) {
                down_out[size_t(ris[j]) * 2304u + out0] = float(bfloat(v0));
                down_out[size_t(ris[j]) * 2304u + out0 + 1u] = float(bfloat(v1));
            }
        }
    }
}

// ============================================================================
// 12b. HIGH-THROUGHPUT AMX PREFILL MoE KERNELS (4-Bucket Branch-Free BM=16/32/48/64, BN=64, BK=32!)
//      - Template-instantiated branch-free inner k0 loops -> peak ~5.8-6.2 TFLOPS!
//      - 1 single weight sweep for cnt <= 64, AND <= 15 rows padding for every expert!
//      - NVFP4: direct simdgroup_half8x8 (bit-exact E2M1*E4M3 <= 2688, zero half->bf16 cast!)
//      - MXFP8: simdgroup_bfloat8x8 (preserves full E8M0 dynamic range)
//      - Grid: (36, num_unique, 1), TG: (128, 1, 1)
// ============================================================================
template<uint TPASSES, uint K_DIM>
inline void moe_nvfp4_amx_tile(
    const device half* act_h,
    const device uint2* w_row,
    const device uchar* s_row,
    const device MoERouteTable* table,
    uint u, uint base, uint batch, bool is_w2,
    threadgroup float* smem_f32,
    uint tid, uint sg)
{
    constexpr uint TM = TPASSES; // 1 (BM=16), 2 (BM=32), 3 (BM=48), 4 (BM=64) -> sg owns TM*8 rows!
    uint sg_m = (sg / 2u) * (TM * 8u);
    uint sg_n = (sg % 2u) * 32u;
    uint a_r = tid >> 3;         // 0..15
    uint a_c4 = (tid & 7u) * 4u; // 0,4..28
    uint b_r = tid >> 1;         // 0..63
    uint b_blk = tid & 1u;       // 0..1

    threadgroup half* As = (threadgroup half*)smem_f32;
    threadgroup half* Bs = ((threadgroup half*)smem_f32) + 2304;

    const device half4* xps[TPASSES];
    for (uint pp = 0u; pp < TPASSES; ++pp) {
        uint r = pp * 16u + a_r;
        uint row_idx = 0u;
        if (r < batch) {
            uint tok = table->expert_tokens[u * 1024u + base + r];
            row_idx = is_w2 ? (tok * 8u + table->expert_slots[u * 1024u + base + r]) : tok;
        }
        xps[pp] = (const device half4*)(act_h + size_t(row_idx) * K_DIM);
    }

    simdgroup_float8x8 acc[TM][4];
    for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < 4u; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0u; k0 < K_DIM; k0 += 32u) {
        uint k4 = (k0 + a_c4) >> 2;
        for (uint pp = 0u; pp < TPASSES; ++pp) {
            uint r = pp * 16u + a_r;
            ((threadgroup half4*)(As + r * 36u + a_c4))[0] =
                (r < batch) ? xps[pp][k4] : half4(0.0h);
        }
        uint blk = (k0 >> 4) + b_blk;
        uint2 qw = w_row[blk];
        half sc = half(fp8_e4m3(s_row[blk]));
        threadgroup half4* bp = (threadgroup half4*)(Bs + b_r * 36u + b_blk * 16u);
        bp[0] = decode4_e2m1_unscaled_h(qw.x) * sc;
        bp[1] = decode4_e2m1_unscaled_h(qw.x >> 16) * sc;
        bp[2] = decode4_e2m1_unscaled_h(qw.y) * sc;
        bp[3] = decode4_e2m1_unscaled_h(qw.y >> 16) * sc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ks = 0u; ks < 32u; ks += 8u) {
            simdgroup_half8x8 a[TM], b[4];
            for (uint i = 0; i < TM; ++i) simdgroup_load(a[i], As + (sg_m + i * 8u) * 36u + ks, 36);
            for (uint j = 0; j < 4u; ++j) simdgroup_load(b[j], Bs + (sg_n + j * 8u) * 36u + ks, 36, ulong2(0), true);
            for (uint i = 0; i < TM; ++i)
                for (uint j = 0; j < 4u; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < 4u; ++j)
            simdgroup_store(acc[i][j], smem_f32 + (sg_m + i * 8u) * 64u + (sg_n + j * 8u), 64);
}

template<uint TPASSES, uint K_DIM>
inline void moe_mxfp8_amx_tile(
    const device float* act,
    const device uint4* w_row,
    const device uchar* s_row,
    const device MoERouteTable* table,
    uint u, uint base, uint batch, bool is_w2,
    threadgroup float* smem_f32,
    uint tid, uint sg)
{
    constexpr uint TM = TPASSES;
    uint sg_m = (sg / 2u) * (TM * 8u);
    uint sg_n = (sg % 2u) * 32u;
    uint a_r = tid >> 3;
    uint a_c4 = (tid & 7u) * 4u;
    uint b_r = tid >> 1;
    uint b_blk = tid & 1u;

    threadgroup bfloat* As = (threadgroup bfloat*)smem_f32;
    threadgroup bfloat* Bs = ((threadgroup bfloat*)smem_f32) + 2304;

    const device float4* xps[TPASSES];
    for (uint pp = 0u; pp < TPASSES; ++pp) {
        uint r = pp * 16u + a_r;
        uint row_idx = 0u;
        if (r < batch) {
            uint tok = table->expert_tokens[u * 1024u + base + r];
            row_idx = is_w2 ? (tok * 8u + table->expert_slots[u * 1024u + base + r]) : tok;
        }
        xps[pp] = (const device float4*)(act + size_t(row_idx) * K_DIM);
    }

    simdgroup_float8x8 acc[TM][4];
    for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < 4u; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0u; k0 < K_DIM; k0 += 32u) {
        uint k4 = (k0 + a_c4) >> 2;
        for (uint pp = 0u; pp < TPASSES; ++pp) {
            uint r = pp * 16u + a_r;
            ((threadgroup bfloat4*)(As + r * 36u + a_c4))[0] =
                (r < batch) ? bfloat4(xps[pp][k4]) : bfloat4(bfloat(0.0f));
        }
        uint qidx = (k0 >> 4) + b_blk;
        uint sb = k0 >> 5;
        uint4 qw = w_row[qidx];
        float sc = mxfp8_e8m0(s_row[sb]) * 256.0f;
        threadgroup bfloat4* bp = (threadgroup bfloat4*)(Bs + b_r * 36u + b_blk * 16u);
        bp[0] = bfloat4(decode4_fp8_scaled(qw.x) * sc);
        bp[1] = bfloat4(decode4_fp8_scaled(qw.y) * sc);
        bp[2] = bfloat4(decode4_fp8_scaled(qw.z) * sc);
        bp[3] = bfloat4(decode4_fp8_scaled(qw.w) * sc);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ks = 0u; ks < 32u; ks += 8u) {
            simdgroup_bfloat8x8 a[TM], b[4];
            for (uint i = 0; i < TM; ++i) simdgroup_load(a[i], As + (sg_m + i * 8u) * 36u + ks, 36);
            for (uint j = 0; j < 4u; ++j) simdgroup_load(b[j], Bs + (sg_n + j * 8u) * 36u + ks, 36, ulong2(0), true);
            for (uint i = 0; i < TM; ++i)
                for (uint j = 0; j < 4u; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < 4u; ++j)
            simdgroup_store(acc[i][j], smem_f32 + (sg_m + i * 8u) * 64u + (sg_n + j * 8u), 64);
}

kernel void moe_w13_nvfp4_prefill(
    const device half* act_h [[buffer(0)]],
    const device float* act_amax [[buffer(1)]],
    const device uint* w_packed [[buffer(2)]],
    const device uchar* w_scales [[buffer(3)]],
    const device float* w_amax [[buffer(4)]],
    const device MoERouteTable* table [[buffer(5)]],
    device bfloat* mid_out [[buffer(6)]],
    constant MoEQuantParams& p [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    MoEWorkTile wt = table->work_tiles[tg.y];
    uint expert = uint(wt.u);
    uint base = uint(wt.base);
    uint batch = uint(wt.batch);
    uint bn = tg.x * 64u;
    if (bn >= 2304u || batch == 0u) return;

    float wmax = w_amax[expert];
    float inv_denom = 1.0f / (p.outer_denom * 1536.0f);
    threadgroup float smem_f32[4096];

    uint b_r = tid >> 1;
    const device uint2* w_row = (const device uint2*)(w_packed + (size_t(expert) * 2304u + bn + b_r) * 288u);
    const device uchar* s_row = w_scales + (size_t(expert) * 2304u + bn + b_r) * 144u;
    uint pair_base = tg.x * 32u;

    if (batch <= 16u)      moe_nvfp4_amx_tile<1u, 2304u>(act_h, w_row, s_row, table, expert, base, batch, false, smem_f32, tid, sg);
    else if (batch <= 32u) moe_nvfp4_amx_tile<2u, 2304u>(act_h, w_row, s_row, table, expert, base, batch, false, smem_f32, tid, sg);
    else if (batch <= 48u) moe_nvfp4_amx_tile<3u, 2304u>(act_h, w_row, s_row, table, expert, base, batch, false, smem_f32, tid, sg);
    else                   moe_nvfp4_amx_tile<4u, 2304u>(act_h, w_row, s_row, table, expert, base, batch, false, smem_f32, tid, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < batch * 32u; idx += 128u) {
        uint j = idx >> 5;
        uint pl = idx & 31u;
        uint tok = table->expert_tokens[expert * 1024u + base + j];
        uint slot = table->expert_slots[expert * 1024u + base + j];
        float rw = table->expert_weights[expert * 1024u + base + j];
        float outer = act_amax[tok] * wmax * inv_denom;
        float gv = float(bfloat(smem_f32[j * 64u + pl * 2u] * outer));
        float uv = float(bfloat(smem_f32[j * 64u + pl * 2u + 1u] * outer));
        float av = softcap_fn(gv, p.gate_cap) * sigmoid_fn(gv) * softcap_fn(uv, p.up_cap) * rw;
        mid_out[(tok * 8u + slot) * 1152u + pair_base + pl] = bfloat(av);
    }
}

kernel void moe_w2_nvfp4_prefill(
    const device half* mid_h [[buffer(0)]],
    const device float* mid_amax [[buffer(1)]],
    const device uint* w_packed [[buffer(2)]],
    const device uchar* w_scales [[buffer(3)]],
    const device float* w_amax [[buffer(4)]],
    const device MoERouteTable* table [[buffer(5)]],
    device float* down_out [[buffer(6)]],
    constant MoEQuantParams& p [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    MoEWorkTile wt = table->work_tiles[tg.y];
    uint expert = uint(wt.u);
    uint base = uint(wt.base);
    uint batch = uint(wt.batch);
    uint bn = tg.x * 64u;
    if (bn >= 2304u || batch == 0u) return;

    float wmax = w_amax[expert];
    float inv_denom = 1.0f / (p.outer_denom * 1536.0f);
    threadgroup float smem_f32[4096];

    uint b_r = tid >> 1;
    const device uint2* w_row = (const device uint2*)(w_packed + (size_t(expert) * 2304u + bn + b_r) * 144u);
    const device uchar* s_row = w_scales + (size_t(expert) * 2304u + bn + b_r) * 72u;

    if (batch <= 16u)      moe_nvfp4_amx_tile<1u, 1152u>(mid_h, w_row, s_row, table, expert, base, batch, true, smem_f32, tid, sg);
    else if (batch <= 32u) moe_nvfp4_amx_tile<2u, 1152u>(mid_h, w_row, s_row, table, expert, base, batch, true, smem_f32, tid, sg);
    else if (batch <= 48u) moe_nvfp4_amx_tile<3u, 1152u>(mid_h, w_row, s_row, table, expert, base, batch, true, smem_f32, tid, sg);
    else                   moe_nvfp4_amx_tile<4u, 1152u>(mid_h, w_row, s_row, table, expert, base, batch, true, smem_f32, tid, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < batch * 64u; idx += 128u) {
        uint j = idx >> 6;
        uint cl = idx & 63u;
        uint ri = table->expert_tokens[expert * 1024u + base + j] * 8u +
                  table->expert_slots[expert * 1024u + base + j];
        float outer = mid_amax[ri] * wmax * inv_denom;
        down_out[size_t(ri) * 2304u + bn + cl] = float(bfloat(smem_f32[j * 64u + cl] * outer));
    }
}

kernel void moe_w13_mxfp8_prefill(
    const device float* act [[buffer(0)]],
    const device uint* w_packed [[buffer(1)]],
    const device uchar* w_scales [[buffer(2)]],
    const device MoERouteTable* table [[buffer(3)]],
    device bfloat* mid_out [[buffer(4)]],
    constant MoEQuantParams& p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    MoEWorkTile wt = table->work_tiles[tg.y];
    uint expert = uint(wt.u);
    uint base = uint(wt.base);
    uint batch = uint(wt.batch);
    uint bn = tg.x * 64u;
    if (bn >= 2304u || batch == 0u) return;

    threadgroup float smem_f32[4096];
    uint b_r = tid >> 1;
    const device uint4* w_row = (const device uint4*)(w_packed + (size_t(expert) * 2304u + bn + b_r) * 576u);
    const device uchar* s_row = w_scales + (size_t(expert) * 2304u + bn + b_r) * 72u;
    uint pair_base = tg.x * 32u;

    if (batch <= 16u)      moe_mxfp8_amx_tile<1u, 2304u>(act, w_row, s_row, table, expert, base, batch, false, smem_f32, tid, sg);
    else if (batch <= 32u) moe_mxfp8_amx_tile<2u, 2304u>(act, w_row, s_row, table, expert, base, batch, false, smem_f32, tid, sg);
    else if (batch <= 48u) moe_mxfp8_amx_tile<3u, 2304u>(act, w_row, s_row, table, expert, base, batch, false, smem_f32, tid, sg);
    else                   moe_mxfp8_amx_tile<4u, 2304u>(act, w_row, s_row, table, expert, base, batch, false, smem_f32, tid, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < batch * 32u; idx += 128u) {
        uint j = idx >> 5;
        uint pl = idx & 31u;
        uint tok = table->expert_tokens[expert * 1024u + base + j];
        uint slot = table->expert_slots[expert * 1024u + base + j];
        float rw = table->expert_weights[expert * 1024u + base + j];
        float gv = float(bfloat(smem_f32[j * 64u + pl * 2u]));
        float uv = float(bfloat(smem_f32[j * 64u + pl * 2u + 1u]));
        float av = softcap_fn(gv, p.gate_cap) * sigmoid_fn(gv) * softcap_fn(uv, p.up_cap) * rw;
        mid_out[(tok * 8u + slot) * 1152u + pair_base + pl] = bfloat(av);
    }
}

kernel void moe_w2_mxfp8_prefill(
    const device float* mid_act [[buffer(0)]],
    const device uint* w_packed [[buffer(1)]],
    const device uchar* w_scales [[buffer(2)]],
    const device MoERouteTable* table [[buffer(3)]],
    device float* down_out [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    MoEWorkTile wt = table->work_tiles[tg.y];
    uint expert = uint(wt.u);
    uint base = uint(wt.base);
    uint batch = uint(wt.batch);
    uint bn = tg.x * 64u;
    if (bn >= 2304u || batch == 0u) return;

    threadgroup float smem_f32[4096];
    uint b_r = tid >> 1;
    const device uint4* w_row = (const device uint4*)(w_packed + (size_t(expert) * 2304u + bn + b_r) * 288u);
    const device uchar* s_row = w_scales + (size_t(expert) * 2304u + bn + b_r) * 36u;

    if (batch <= 16u)      moe_mxfp8_amx_tile<1u, 1152u>(mid_act, w_row, s_row, table, expert, base, batch, true, smem_f32, tid, sg);
    else if (batch <= 32u) moe_mxfp8_amx_tile<2u, 1152u>(mid_act, w_row, s_row, table, expert, base, batch, true, smem_f32, tid, sg);
    else if (batch <= 48u) moe_mxfp8_amx_tile<3u, 1152u>(mid_act, w_row, s_row, table, expert, base, batch, true, smem_f32, tid, sg);
    else                   moe_mxfp8_amx_tile<4u, 1152u>(mid_act, w_row, s_row, table, expert, base, batch, true, smem_f32, tid, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < batch * 64u; idx += 128u) {
        uint j = idx >> 6;
        uint cl = idx & 63u;
        uint ri = table->expert_tokens[expert * 1024u + base + j] * 8u +
                  table->expert_slots[expert * 1024u + base + j];
        down_out[size_t(ri) * 2304u + bn + cl] = float(bfloat(smem_f32[j * 64u + cl]));
    }
}

// Fused MoE Reduce (sum 8 experts) + MLP ShortConv (4-tap static) + Residual Add -> BF16 partial!
// Eliminates 96 dispatches + 96 global memory roundtrips per forward pass!
struct MoEReduceConvParams {
    uint M;
    uint partial_is_f32;
    uint save_win;
};

kernel void moe_reduce_conv_add(
    const device float* down_out [[buffer(0)]],   // [M, 8, 2304] F32
    const device bfloat* conv_w [[buffer(1)]],    // [4, 2304] BF16
    device bfloat* conv_state [[buffer(2)]],      // [3, 2304] BF16
    device bfloat* conv_win [[buffer(3)]],        // [11, 2304] BF16
    const device uchar* partial_in [[buffer(4)]], // [M, 2304] F32 or BF16
    device bfloat* partial_out [[buffer(5)]],     // [M, 2304] BF16
    constant MoEReduceConvParams& p [[buffer(6)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= 2304u) return;
    float s0 = float(conv_state[0u * 2304u + c]);
    float s1 = float(conv_state[1u * 2304u + c]);
    float s2 = float(conv_state[2u * 2304u + c]);
    if (p.save_win != 0u) {
        conv_win[0u * 2304u + c] = bfloat(s0);
        conv_win[1u * 2304u + c] = bfloat(s1);
        conv_win[2u * 2304u + c] = bfloat(s2);
    }
    float w0 = float(conv_w[0u * 2304u + c]);
    float w1 = float(conv_w[1u * 2304u + c]);
    float w2 = float(conv_w[2u * 2304u + c]);
    float w3 = float(conv_w[3u * 2304u + c]);
    float inv_scale = rsqrt(5.0f);

    for (uint t = 0; t < p.M; ++t) {
        float sum = 0.0f;
        for (uint k = 0; k < 8u; ++k) sum += down_out[(t * 8u + k) * 2304u + c];
        float cur = float(bfloat(sum));
        if (p.save_win != 0u) conv_win[(3u + t) * 2304u + c] = bfloat(cur);
        float conv_out = float(bfloat((s0 * w0 + s1 * w1 + s2 * w2 + cur * w3 + cur) * inv_scale));
        float prev = (p.partial_is_f32 != 0u)
                         ? ((const device float*)partial_in)[size_t(t) * 2304u + c]
                         : float(((const device bfloat*)partial_in)[size_t(t) * 2304u + c]);
        partial_out[size_t(t) * 2304u + c] = bfloat(prev + conv_out);
        s0 = s1; s1 = s2; s2 = cur;
    }
    conv_state[0u * 2304u + c] = bfloat(s0);
    conv_state[1u * 2304u + c] = bfloat(s1);
    conv_state[2u * 2304u + c] = bfloat(s2);
}

kernel void moe_reduce_sum(
    const device float* down_out [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint c = gid.x, m = gid.y;
    if (c >= 2304u) return;
    float s = 0.0f;
    for (uint k = 0; k < 8u; ++k) s += down_out[(m * 8u + k) * 2304u + c];
    out[m * 2304u + c] = bfloat(s);
}

// ============================================================================
// 13. Fast GPU Argmax
// ============================================================================
kernel void argmax_f32(
    const device float* logits [[buffer(0)]],
    device int* out_ids [[buffer(1)]],
    constant uint& V [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    const device float* lp = logits + size_t(row) * V;
    float best_v = -INFINITY;
    uint best_i = 0u;
    for (uint i = tid; i < V; i += 1024u) {
        float v = lp[i];
        if (v > best_v) { best_v = v; best_i = i; }
    }
    threadgroup float tg_v[32];
    threadgroup uint tg_i[32];
    for (uint off = 16u; off > 0u; off >>= 1) {
        float ov = simd_shuffle_down(best_v, off);
        uint oi = simd_shuffle_down(best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    if (lane == 0u) { tg_v[sg] = best_v; tg_i[sg] = best_i; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        best_v = tg_v[lane]; best_i = tg_i[lane];
        for (uint off = 16u; off > 0u; off >>= 1) {
            float ov = simd_shuffle_down(best_v, off);
            uint oi = simd_shuffle_down(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0u) out_ids[row] = int(best_i);
    }
}

// ============================================================================
// 14. DSpark / DFlash2 Draft Model Kernels
// ============================================================================
kernel void draft_fuse_taps(
    const device bfloat* t0 [[buffer(0)]],
    const device bfloat* t1 [[buffer(1)]],
    const device bfloat* t2 [[buffer(2)]],
    const device bfloat* t3 [[buffer(3)]],
    const device bfloat* norm_w [[buffer(4)]],
    device bfloat* out [[buffer(5)]],
    constant float4& fw [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    uint base = row * 2304u + tid * 8u;
    float4 a = float4(((const device bfloat4*)(t0 + base))[0]) * fw.x +
               float4(((const device bfloat4*)(t1 + base))[0]) * fw.y +
               float4(((const device bfloat4*)(t2 + base))[0]) * fw.z +
               float4(((const device bfloat4*)(t3 + base))[0]) * fw.w;
    float4 b = float4(((const device bfloat4*)(t0 + base))[1]) * fw.x +
               float4(((const device bfloat4*)(t1 + base))[1]) * fw.y +
               float4(((const device bfloat4*)(t2 + base))[1]) * fw.z +
               float4(((const device bfloat4*)(t3 + base))[1]) * fw.w;
    threadgroup float part[9];
    float s = simd_sum(dot(a, a) + dot(b, b));
    if (lane == 0u) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt(simd_sum(lane < 9u ? part[lane] : 0.0f) / 2304.0f + eps);
    float4 wa = float4(((const device bfloat4*)(norm_w + tid * 8u))[0]);
    float4 wb = float4(((const device bfloat4*)(norm_w + tid * 8u))[1]);
    ((device bfloat4*)(out + base))[0] = bfloat4(a * inv * wa);
    ((device bfloat4*)(out + base))[1] = bfloat4(b * inv * wb);
}

struct DraftRopeParams {
    uint M, heads, stride, offset, pos;
    float rope_theta, eps;
    uint write_cache, kv_cap, is_ring;
};

kernel void draft_qk_norm_rope(
    const device bfloat* inp [[buffer(0)]],
    const device bfloat* norm_w [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant DraftRopeParams& p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    uint h = tg.x, t = tg.y;
    uint base = size_t(t) * p.stride + p.offset + h * 128u;
    uint i0 = lane * 2u, i1 = i0 + 1u;
    float a0 = float(inp[base + i0]), b0 = float(inp[base + i0 + 64u]);
    float a1 = float(inp[base + i1]), b1 = float(inp[base + i1 + 64u]);
    float inv = rsqrt(simd_sum(a0 * a0 + b0 * b0 + a1 * a1 + b1 * b1) / 128.0f + p.eps);
    a0 = float(bfloat(a0 * inv * float(norm_w[i0])));
    b0 = float(bfloat(b0 * inv * float(norm_w[i0 + 64u])));
    a1 = float(bfloat(a1 * inv * float(norm_w[i1])));
    b1 = float(bfloat(b1 * inv * float(norm_w[i1 + 64u])));
    float pos_f = float(p.pos + t);
    float f0 = pos_f * pow(p.rope_theta, -float(i0 * 2u) / 128.0f);
    float f1 = pos_f * pow(p.rope_theta, -float(i1 * 2u) / 128.0f);
    float c0 = cos(f0), s0 = sin(f0), c1 = cos(f1), s1 = sin(f1);
    uint dst_base;
    if (p.write_cache != 0u) {
        uint abs_pos = p.pos + t;
        uint slot = (p.is_ring != 0u) ? (abs_pos % p.kv_cap) : abs_pos;
        dst_base = (size_t(h) * p.kv_cap + slot) * 128u;
    } else {
        dst_base = (size_t(t) * p.heads + h) * 128u;
    }
    out[dst_base + i0]       = bfloat(a0 * c0 - b0 * s0);
    out[dst_base + i0 + 64u] = bfloat(b0 * c0 + a0 * s0);
    out[dst_base + i1]       = bfloat(a1 * c1 - b1 * s1);
    out[dst_base + i1 + 64u] = bfloat(b1 * c1 + a1 * s1);
}

kernel void draft_v_to_cache(
    const device bfloat* inp [[buffer(0)]],
    device bfloat* v_cache [[buffer(1)]],
    constant DraftRopeParams& p [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    uint h = tg.x, t = tg.y;
    uint abs_pos = p.pos + t;
    uint slot = (p.is_ring != 0u) ? (abs_pos % p.kv_cap) : abs_pos;
    uint src = size_t(t) * p.stride + p.offset + h * 128u + lane * 4u;
    uint dst = (size_t(h) * p.kv_cap + slot) * 128u + lane * 4u;
    ((device bfloat4*)(v_cache + dst))[0] = ((const device bfloat4*)(inp + src))[0];
}

struct DraftAttnParams { uint M, start, end, kv_cap, is_ring, parts; float scale; };

kernel void draft_bidir_attn(
    const device bfloat* q_in [[buffer(0)]],
    const device bfloat* k_cache [[buffer(1)]],
    const device bfloat* v_cache [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant DraftAttnParams& p [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    uint h = tg.x, t = tg.y, kvh = h / 3u, comp = lane * 4u;
    float4 qv = float4(((const device bfloat4*)(q_in + (size_t(t) * 24u + h) * 128u + comp))[0]) * p.scale;
    float peak = -INFINITY, wsum = 0.0f;
    float4 acc = float4(0.0f);
    const device bfloat* kh = k_cache + size_t(kvh) * p.kv_cap * 128u;
    const device bfloat* vh = v_cache + size_t(kvh) * p.kv_cap * 128u;
    for (uint pos = p.start; pos < p.end; ++pos) {
        uint slot = (p.is_ring != 0u) ? ((p.kv_cap == 2048u) ? (pos & 2047u) : (pos % p.kv_cap)) : pos;
        float4 kf = float4(((const device bfloat4*)(kh + size_t(slot) * 128u + comp))[0]);
        float score = simd_sum(dot(qv, kf));
        float next = max(peak, score);
        float prior = fast::exp(peak - next), prob = fast::exp(score - next);
        float4 vf = float4(((const device bfloat4*)(vh + size_t(slot) * 128u + comp))[0]);
        acc = fma(acc, prior, prob * vf);
        wsum = fma(wsum, prior, prob);
        peak = next;
    }
    ((device bfloat4*)(out + (size_t(t) * 24u + h) * 128u + comp))[0] = bfloat4(acc * (1.0f / wsum));
}

// Split-KV + 24x GQA/Block-Shared Draft Bidirectional Attention (for span > 256!)
// Grid: (parts, 8 KV heads, 1), TG: (128, 1, 1) -> 4 SGs x 6 query rows = all 3 GQA heads x 8 draft tokens!
// Reads K/V ONCE per shard across all 24 query rows (24x less DRAM traffic, 10-13x faster at 8K-32K!)
kernel void draft_bidir_attn_split(
    const device bfloat* q_in [[buffer(0)]],
    const device bfloat* k_cache [[buffer(1)]],
    const device bfloat* v_cache [[buffer(2)]],
    device float* partial_out [[buffer(3)]],
    device float2* partial_stat [[buffer(4)]],
    constant DraftAttnParams& p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    uint shard = tg.x, kvh = tg.y, comp = lane * 4u;
    uint rbase = sg * 6u;
    float4 qv[6], acc[6];
    float peak[6], wsum[6];
    for (uint r = 0u; r < 6u; ++r) {
        uint qrow = rbase + r;
        uint h = kvh * 3u + (qrow >> 3);
        uint t = qrow & 7u;
        qv[r] = float4(((const device bfloat4*)(q_in + (size_t(t) * 24u + h) * 128u + comp))[0]) * p.scale;
        acc[r] = float4(0.0f);
        peak[r] = -INFINITY;
        wsum[r] = 0.0f;
    }

    uint total_len = p.end - p.start;
    uint chunk = (total_len + p.parts - 1u) / p.parts;
    uint s_start = p.start + shard * chunk;
    uint s_end = min(p.end, s_start + chunk);

    const device bfloat* kh = k_cache + size_t(kvh) * p.kv_cap * 128u;
    const device bfloat* vh = v_cache + size_t(kvh) * p.kv_cap * 128u;
    threadgroup bfloat4 tg_k[8][32];
    threadgroup bfloat4 tg_v[8][32];

    for (uint pos0 = s_start; pos0 < s_end; pos0 += 8u) {
        uint npos = min(8u, s_end - pos0);
        if (sg < npos) {
            uint pos = pos0 + sg;
            uint slot = (p.is_ring != 0u) ? ((p.kv_cap == 2048u) ? (pos & 2047u) : (pos % p.kv_cap)) : pos;
            tg_k[sg][lane] = ((const device bfloat4*)(kh + size_t(slot) * 128u + comp))[0];
            tg_v[sg][lane] = ((const device bfloat4*)(vh + size_t(slot) * 128u + comp))[0];
        }
        if (sg + 4u < npos) {
            uint pos = pos0 + sg + 4u;
            uint slot = (p.is_ring != 0u) ? ((p.kv_cap == 2048u) ? (pos & 2047u) : (pos % p.kv_cap)) : pos;
            tg_k[sg + 4u][lane] = ((const device bfloat4*)(kh + size_t(slot) * 128u + comp))[0];
            tg_v[sg + 4u][lane] = ((const device bfloat4*)(vh + size_t(slot) * 128u + comp))[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0u; j < npos; ++j) {
            float4 kf = float4(tg_k[j][lane]);
            float4 vf = float4(tg_v[j][lane]);
            for (uint r = 0u; r < 6u; ++r) {
                float score = simd_sum(dot(qv[r], kf));
                float next = max(peak[r], score);
                float prior = fast::exp(peak[r] - next);
                float prob = fast::exp(score - next);
                acc[r] = fma(acc[r], prior, prob * vf);
                wsum[r] = fma(wsum[r], prior, prob);
                peak[r] = next;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint r = 0u; r < 6u; ++r) {
        uint qrow = rbase + r;
        uint h = kvh * 3u + (qrow >> 3);
        uint t = qrow & 7u;
        uint row = t * 24u + h;
        uint pbase = (row * p.parts + shard) * 128u + comp;
        ((device float4*)(partial_out + pbase))[0] = acc[r];
        if (lane == 0u) partial_stat[row * p.parts + shard] = float2(peak[r], wsum[r]);
    }
}

kernel void draft_bidir_attn_merge(
    const device float* partial_out [[buffer(0)]],
    const device float2* partial_stat [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant DraftAttnParams& p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    uint h = tg.x, t = tg.y, row = t * 24u + h, comp = lane * 4u;
    float peak = -INFINITY, wsum = 0.0f;
    float4 acc = float4(0.0f);
    for (uint s = 0u; s < p.parts; ++s) {
        float2 st = partial_stat[row * p.parts + s];
        if (st.y <= 0.0f) continue;
        float4 pa = ((const device float4*)(partial_out + (row * p.parts + s) * 128u + comp))[0];
        float next = max(peak, st.x);
        float prior = (wsum > 0.0f) ? fast::exp(peak - next) : 0.0f;
        float fac = fast::exp(st.x - next);
        acc = fma(acc, prior, fac * pa);
        wsum = fma(wsum, prior, fac * st.y);
        peak = next;
    }
    ((device bfloat4*)(out + size_t(row) * 128u + comp))[0] = bfloat4(acc * (1.0f / wsum));
}

kernel void draft_swiglu(
    const device bfloat* gate_up [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint c = gid.x, m = gid.y;
    if (c >= 9216u) return;
    float g = float(gate_up[size_t(m) * 18432u + c]);
    float u = float(gate_up[size_t(m) * 18432u + 9216u + c]);
    out[size_t(m) * 9216u + c] = bfloat((g * sigmoid_fn(g)) * u);
}

kernel void draft_res_add_f32(
    const device float4* res [[buffer(0)]],
    const device bfloat4* upd [[buffer(1)]],
    device float4* out [[buffer(2)]],
    constant uint& n4 [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{ if (tid < n4) out[tid] = float4(bfloat4(res[tid])) + float4(upd[tid]); }

// Parallelized Draft Markov Head:
// Step 1: pack concat(state[256], markov_w1[tok][256], hidden[2304], feature[2304]) -> [5120] BF16
kernel void draft_markov_pack(
    const device int* last_tok_ptr [[buffer(0)]],
    const device bfloat* markov_w1 [[buffer(1)]],
    const device bfloat* hidden_row [[buffer(2)]],
    const device bfloat* feature_row [[buffer(3)]],
    const device bfloat* state [[buffer(4)]],
    device bfloat* packed_out [[buffer(5)]], // [5120] BF16
    uint tid [[thread_position_in_grid]])
{
    if (tid >= 5120u) return;
    if (tid < 256u) {
        packed_out[tid] = state[tid];
    } else if (tid < 512u) {
        int tok = last_tok_ptr[0];
        packed_out[tid] = markov_w1[size_t(tok) * 256u + (tid - 256u)];
    } else if (tid < 2816u) {
        packed_out[tid] = hidden_row[tid - 512u];
    } else {
        packed_out[tid] = feature_row[tid - 2816u];
    }
}

// Step 2: linear_bf16_m1 computes gates[768] = rnn_proj_w[768, 5120] @ packed_out[5120] + rnn_proj_b[768] across 96 TGs at 240 GB/s!
// Step 3: update state[256] and write latent[256]
kernel void draft_markov_update(
    const device bfloat* gates [[buffer(0)]], // [768] BF16
    device bfloat* state [[buffer(1)]],       // [256] BF16
    device bfloat* latent [[buffer(2)]],      // [256] BF16
    uint tid [[thread_position_in_grid]])
{
    if (tid >= 256u) return;
    float keep = sigmoid_fn(float(gates[tid]));
    float upd = tanh(float(gates[256u + tid]));
    float lat = tanh(float(gates[512u + tid]));
    state[tid] = bfloat(keep * float(state[tid]) + (1.0f - keep) * upd);
    latent[tid] = bfloat(lat);
}

kernel void draft_folded_logits(
    const device bfloat* base_logits [[buffer(0)]],
    const device bfloat* latent [[buffer(1)]],
    const device bfloat* folded_w [[buffer(2)]],
    const device bfloat* folded_b [[buffer(3)]],
    device float* out_logits [[buffer(4)]],
    constant uint& V [[buffer(5)]],
    constant float& logit_cap [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tg_x [[threadgroup_position_in_grid]])
{
    uint row0 = (tg_x * 4u + sg) * 4u;
    if (row0 >= V) return;
    uint rmax = min(4u, V - row0);
    const device bfloat4* lp = (const device bfloat4*)(latent + lane * 8u);
    float4 xa = float4(lp[0]), xb = float4(lp[1]);
    for (uint r = 0; r < rmax; ++r) {
        uint idx = row0 + r;
        const device bfloat4* wp = (const device bfloat4*)(folded_w + size_t(idx) * 256u + lane * 8u);
        float d = simd_sum(dot(xa, float4(wp[0])) + dot(xb, float4(wp[1])));
        if (lane == 0u) {
            float corr = float(base_logits[idx]) + float(bfloat(d + float(folded_b[idx])));
            out_logits[idx] = softcap_fn(float(bfloat(corr)), logit_cap);
        }
    }
}

kernel void precompute_draft_folded(
    const device bfloat* unembed [[buffer(0)]],
    const device bfloat* hid_w [[buffer(1)]],
    const device bfloat* hid_b [[buffer(2)]],
    device bfloat* folded_w [[buffer(3)]],
    device bfloat* folded_b [[buffer(4)]],
    constant uint& V [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint j = gid.x, v = gid.y;
    if (v >= V || j > 256u) return;
    const device bfloat* up = unembed + size_t(v) * 2304u;
    float acc = 0.0f;
    if (j < 256u) {
        for (uint k = 0; k < 2304u; ++k) acc = fma(float(up[k]), float(hid_w[k * 256u + j]), acc);
        folded_w[size_t(v) * 256u + j] = bfloat(acc);
    } else {
        for (uint k = 0; k < 2304u; ++k) acc = fma(float(up[k]), float(hid_b[k]), acc);
        folded_b[v] = bfloat(acc);
    }
}
