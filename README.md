# Inkling 2 Mini — Specialized Apple M4 Pro Metal Inference Engine

A standalone, zero-dependency **C++20 / Objective-C++ + direct Metal (MSL 3.2)** inference engine built specifically for **Inkling 2 Mini (`26b-a5b-nvfp4`)** on **Apple M4 Pro (20-core GPU, 48 GB unified memory)** and specialized for **low batch sizes** (`B=1` plain & DSpark speculative decoding, and concurrent `B=2..8` decode).

## Performance (Apple M4 Pro, Warm, 256-Token Greedy)

Benchmark prompt: *"Write a short Python function to compute Fibonacci numbers iteratively and explain its time and space complexity."* (`think=0`, exact same prompt & weights as Ollama).

| Engine / Mode | Init Time | Decode Throughput | Speedup vs Ollama | Token Exactness vs Plain B=1 |
| :--- | :---: | :---: | :---: | :---: |
| Ollama (`mlxrunner`, DSpark spec) | 7.23 s | 30.94 tok/s | 1.00× | Baseline |
| **inkling** — Plain `B=1` (no spec) | 0.79 s | **34.37 tok/s** (29.1 ms/tok) | 1.11× | 256 / 256 |
| **inkling** — Speculative `B=1` (`--spec`, adaptive) | 1.62 s | **49.7–50.8 tok/s** | **1.61×** | **256 / 256** |
| **inkling** — Long-Context 8K Plain / Spec `B=1` | 1.62 s | **28.2 / 33.8 tok/s** | — | **128 / 128** |
| **inkling** — Concurrent Batch `B=2` | 0.96 s | **63.49 agg tok/s** (31.7 tok/s/seq) | **2.05×** | **128 / 128** |
| **inkling** — Concurrent Batch `B=4` | 0.96 s | **100.84 agg tok/s** (25.2 tok/s/seq) | **3.26×** | **128 / 128** |
| **inkling** — Concurrent Batch `B=8` | 0.96 s | **132.01 agg tok/s** (16.5 tok/s/seq) | **4.27×** | **128 / 128** |

### Prefill Throughput & Hardware Roofline (Apple M4 Pro: 6.55 TFLOPS AMX Peak, 254 GB/s DRAM)

| Prompt Length (Diverse Code) | Prefill Latency | Prefill Throughput | % of 711 tok/s Theoretical Roofline |
| :--- | :---: | :---: | :---: |
| 512 tokens | 1.06 s | **483.5 tok/s** | 68.0% |
| 1,024 tokens | 1.89 s | **541.4 tok/s** | **76.1%** |
| 4,096 tokens | 7.96 s | **514.3 tok/s** | 72.3% |
| 8,192 tokens | 16.35 s | **501.0 tok/s** | 70.5% |
| 11,118 tokens | 22.71 s | **489.5 tok/s** | 72.6% (of 674 tok/s @ 11K) |

## Architecture & M4 Pro Specializations

- **Zero-Copy Weight Loading + Load-Time Fusion ([`model_weights.mm`](src/model_weights.mm)):**
  - Directly `mmap`s Ollama's local safetensors blobs (`newBufferWithBytesNoCopy:`, closing FDs immediately to stay under `ulimit -n`).
  - Pre-packs per-layer BF16 projections at load time (local layers fused into `[8600, 2304]` `[Q, R, Gate, Sink, K, V]`; loop controllers fused into `[4608, 64]`; draft QKV/FFN fused; Markov head folded on GPU into `[200058, 256]` BF16).
- **Single-Command-Buffer 48-Visit Target Forward ([`engine.mm`](src/engine.mm), [`kernels.metal`](src/kernels.metal)):**
  - Executes the full looped MoE schedule (`L0..5`, `L6..17 × 3` with cross-pass residual attention & rank-64 loop controllers, `L18..23`) inside a single Metal compute command encoder.
  - Fused `BlockResidualMix + RMSNorm`, `AttnPrep (ClipL2 + V-RMSNorm + KV-Write)`, `RelativePositionAttention + OutputGate`, `ShortConv + ResidualAdd`, and `MoEReduce + MLPShortConv + ResidualAdd` (eliminating 144 dispatches/forward).
- **Coalesced Indirect-Dispatch MoE Kernels (NVFP4 L0–19 & MXFP8 L20–23):**
  - GPU-side top-8 routing + unique-expert bucketing (`moe_topk_and_group`) drives `dispatchThreadgroupsWithIndirectBuffer`, reading each unique active expert **once** across all tokens in a speculative verify block or concurrent batch.
  - Pure SIMD ALU NVFP4/FP8 decoders (`decode4_e2m1_scaled`, `decode4_fp8_scaled`) with zero constant-cache LUT stalls, bit-exact `half4` quantized activations, and zero-barrier / zero-shared-memory layouts hitting **215–223 GB/s** on M4 Pro (NVFP4 W13 2 pairs/SG + no-barrier MXFP8).
- **Bit-Exact Speculative & Batched Decoding:**
  - Hybrid Metal shader compilation (`MTLMathModeFast` for bandwidth/ALU kernels, `MTLMathModeSafe` for the 5 MoE matmul kernels) + fixed 16-part split-KV attention reduction guarantees **256/256 bit-exact token agreement** between plain, speculative, and batched decoding.
- **Native DSpark / DFlash2 Speculative Decoding:**
  - Fuses draft context KV flush (up to 1024-tok chunks) + 3-layer bidirectional draft block + parallel Markov RNN head into a single command buffer (**7.1–9.8 ms**, ~235 GB/s = 93% of DRAM bandwidth).
  - Long-context Split-KV 24× GQA/block-shared `draft_bidir_attn_split` cuts 8K draft attention from 4.21 ms -> **0.74 ms** (5.7× faster; 32K in **2.51 ms**).
  - GPU-side `short_conv_rollback` over 11-row conv windows and power-of-2 2048-slot ring KV caches.
  - Online per-position conditional acceptance adaptive depth controller (`--draft-depth 0`, default).

## Build, Lint & Run

Requires only macOS Xcode Command Line Tools (`clang++`, Metal framework). No third-party packages are needed.

```bash
# Build (strict -Wall -Wextra -Wshadow -Werror)
make

# Static analysis (Clang Static Analyzer)
make lint

# Format (optional; requires `brew install clang-format`)
make format
```

### Usage Examples

```bash
# Adaptive DSpark speculative decoding (fastest B=1, ~50 tok/s)
./build/inkling --spec --stream --max-tokens 256 \
  --prompt "Write a short Python function to compute Fibonacci numbers iteratively."

# Fixed speculative draft depth (1..7)
./build/inkling --spec --draft-depth 3 --max-tokens 256

# Plain B=1 with temperature / top-p sampling
./build/inkling --temp 0.6 --top-p 0.9 --seed 123 --max-tokens 256

# Concurrent low-batch decode (B=1..8, shares all dense/MoE/unembed weight reads)
./build/inkling --batch 4 --max-tokens 128

# OpenAI Chat Completions-compatible HTTP server (DSpark speculative on by default!)
./build/inkling --serve --host 127.0.0.1 --port 8080

# Heterogeneous Metal GPU + ARMv9 SME2 prefill (+13-16% faster prefill, +0 MB extra RAM)
./build/inkling --spec --hetero --max-tokens 256 --prompt "..."

# Load directly from a standard Hugging Face model directory (config.json, tokenizer.json, *.safetensors)
./build/inkling --model-dir /path/to/inkling-2-mini-hf --spec --max-tokens 256
```

### OpenAI-Compatible API (`--serve`)

- **Roofline-Tuned AMX Prefill (`M=1024` MoE + FlashAttention-2 + Dense Tiled64):** Uses `BM=64, BN=64, BK=32` BF16 AMX GEMMs (`linear_bf16_tiled64`, **6.33 TFLOPS = 97% of M4 Pro peak**), 4-bucket branch-free `BM=16/32/48/64` FP16-AMX NVFP4 & BF16-AMX MXFP8 MoE kernels driven by 128-TG parallel routing + 64-token load-balanced worklists (`1.34 ms` across 48 visits), AMX relative-position table GEMM (23.6× faster), parallel 2D FIR short-conv prefill (3.9× faster), and `BM=16, BN=32` 16-byte-vectorized FlashAttention-2 (with `no_rel_bias` unmasked global interior fast path + power-of-2 2048-slot ring cache) reaching **489–541 tok/s** sustained prefill from 1K to 11K+ tokens (**76% of the 711 tok/s theoretical M4 Pro roofline**, 4.75× faster than decode-kernel prefill) with live progress logs and SSE keepalives.
- **2-Tier RAM + NVMe Offload Prefix Cache (`--serve`):** Content-addressed `256`-token Global KV block deduplication + compacted sliding-window/conv checkpoints (`min(pos, 1024)` slots) across multiple concurrent sessions/subagents, mid-conversation branches/retries, periodic 4K milestones, and aborted prefills. Strictly bounded by `--cache-ram-mb` (default `4096`) and `--cache-disk-mb` (default `16384`, `--cache-dir` default `~/.cache/inkling`), automatically offloading cold entries to NVMe SSD (~50–70 ms reload) and cleaning up its session offload directory on shutdown. Inspect/clear via `GET /v1/cache` / `DELETE /v1/cache` and `GET /health`; reports `cached_tokens` in OpenAI `usage`.
- **OpenAI-Compatible API (`--serve`):** Multi-threaded HTTP/1.1 server (with `SO_RCVTIMEO`, `Transfer-Encoding: chunked`, and non-blocking `GET /health`, `GET /props`, `GET /v1/models`, `GET /v1/cache`) supporting `POST /v1/chat/completions` and `POST /v1/responses` (sync & SSE streaming, `reasoning_content` / `reasoning` + `content` / `output_text`, native Inkling 2 tool calling via `tools` / `tool_calls` / `function_call` / `function_call_output`, `prompt_cache_key`, `max_tokens`/`max_output_tokens`, `temperature`, `top_p`, `seed`, `stop`, and `reasoning_effort`). Default context is **262,144** with dynamic grow-on-demand global KV caches (configurable via `--max-seq`).

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inkling-2-mini",
    "messages": [{"role": "user", "content": "Write an iterative Fibonacci function in Python."}],
    "max_tokens": 256,
    "stream": true
  }'
```

By default, weights are loaded directly from `~/.ollama/models/manifests/registry.ollama.ai/library/inkling-2-mini/26b-a5b-nvfp4` and `~/.ollama/models/blobs` (override with `--manifest` and `--blobs`), or pass `--model-dir <dir>` to load directly from a standard Hugging Face directory (`config.json`, `tokenizer.json`, `model.safetensors.index.json` / `*.safetensors`, and optional `draft/` subfolder).
