# Inkling 2 Mini — Specialized Apple M4 Pro Metal Inference Engine

A standalone, zero-dependency **C++20 / Objective-C++ + direct Metal (MSL 3.2)** inference engine built specifically for **Inkling 2 Mini (`26b-a5b-nvfp4`)** on **Apple M4 Pro (20-core GPU, 48 GB unified memory)** and specialized for **low batch sizes** (`B=1` plain & DSpark speculative decoding, and concurrent `B=2..8` decode).

## Performance (Apple M4 Pro, Warm, 256-Token Greedy)

Benchmark prompt: *"Write a short Python function to compute Fibonacci numbers iteratively and explain its time and space complexity."* (`think=0`, exact same prompt & weights as Ollama).

| Engine / Mode | Init Time | Decode Throughput | Speedup vs Ollama | Token Exactness vs Plain B=1 |
| :--- | :---: | :---: | :---: | :---: |
| Ollama (`mlxrunner`, DSpark spec) | 7.23 s | 30.94 tok/s | 1.00× | Baseline |
| **inkling** — Plain `B=1` (no spec) | 0.79 s | **32.51–32.80 tok/s** | 1.06× | 256 / 256 |
| **inkling** — Speculative `B=1` (`--spec`, adaptive) | 1.62 s | **49.69 tok/s** | **1.61×** | **256 / 256** |
| **inkling** — Speculative `B=1` (`--spec --draft-depth 3`) | 1.62 s | **49.80–50.21 tok/s** | **1.62×** | **256 / 256** |
| **inkling** — Concurrent Batch `B=2` | 0.96 s | **57.89 agg tok/s** (28.94 tok/s/seq) | **1.87×** | **128 / 128** |
| **inkling** — Concurrent Batch `B=4` | 0.96 s | **88.88–89.63 agg tok/s** (22.41 tok/s/seq) | **2.90×** | **128 / 128** |

## Architecture & M4 Pro Specializations

- **Zero-Copy Weight Loading + Load-Time Fusion ([`model_weights.mm`](src/model_weights.mm)):**
  - Directly `mmap`s Ollama's local safetensors blobs (`newBufferWithBytesNoCopy:`, closing FDs immediately to stay under `ulimit -n`).
  - Pre-packs per-layer BF16 projections at load time (local layers fused into `[8600, 2304]` `[Q, R, Gate, Sink, K, V]`; loop controllers fused into `[4608, 64]`; draft QKV/FFN fused; Markov head folded on GPU into `[200058, 256]` BF16).
- **Single-Command-Buffer 48-Visit Target Forward ([`engine.mm`](src/engine.mm), [`kernels.metal`](src/kernels.metal)):**
  - Executes the full looped MoE schedule (`L0..5`, `L6..17 × 3` with cross-pass residual attention & rank-64 loop controllers, `L18..23`) inside a single Metal compute command encoder.
  - Fused `BlockResidualMix + RMSNorm`, `AttnPrep (ClipL2 + V-RMSNorm + KV-Write)`, `RelativePositionAttention + OutputGate`, `ShortConv + ResidualAdd`, and `MoEReduce + MLPShortConv + ResidualAdd` (eliminating 144 dispatches/forward).
- **Coalesced Indirect-Dispatch MoE Kernels (NVFP4 L0–19 & MXFP8 L20–23):**
  - GPU-side top-8 routing + unique-expert bucketing (`moe_topk_and_group`) drives `dispatchThreadgroupsWithIndirectBuffer`, reading each unique active expert **once** across all tokens in a speculative verify block or concurrent batch.
  - Pure SIMD ALU NVFP4/FP8 decoders (`decode4_e2m1_scaled`, `decode4_fp8_scaled`) with zero constant-cache LUT stalls, bit-exact `half4` quantized activations, and zero-barrier / zero-shared-memory layouts hitting **205–218 GB/s** on M4 Pro.
- **Bit-Exact Speculative & Batched Decoding:**
  - Hybrid Metal shader compilation (`MTLMathModeFast` for bandwidth/ALU kernels, `MTLMathModeSafe` for the 5 MoE matmul kernels) + fixed 16-part split-KV attention reduction guarantees **256/256 bit-exact token agreement** between plain, speculative, and batched decoding.
- **Native DSpark / DFlash2 Speculative Decoding:**
  - Fuses draft context KV flush + 3-layer bidirectional draft block + parallel Markov RNN head into a single command buffer (**7.5–9.4 ms**, ~235 GB/s = 93% of DRAM bandwidth).
  - GPU-side `short_conv_rollback` over 11-row conv windows and 1088-slot ring KV caches.
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
```

### OpenAI-Compatible API (`--serve`)

Supports `GET /health`, `GET /props`, `GET /v1/models` (exposing `context_length` / `max_model_len`, default **262,144** with dynamic grow-on-demand global KV caches; configurable via `--max-seq`), `POST /v1/chat/completions`, and `POST /v1/responses` (both non-streaming JSON and SSE streaming, `reasoning_content` / `reasoning` + `content` / `output_text`, **native Inkling 2 tool calling** via `tools` / `tool_calls` / `function_call` / `function_call_output`, multi-turn messages, `max_tokens`/`max_output_tokens`, `temperature`, `top_p`, `seed`, `stop`, and `reasoning_effort`).

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

By default, weights are loaded directly from `~/.ollama/models/manifests/registry.ollama.ai/library/inkling-2-mini/26b-a5b-nvfp4` and `~/.ollama/models/blobs` (override with `--manifest` and `--blobs`).
