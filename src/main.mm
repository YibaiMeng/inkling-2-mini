#import <Foundation/Foundation.h>
#include "engine.h"
#include "http_server.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n"
           "  --serve               Run OpenAI-compatible HTTP server (enables --spec by default)\n"
           "  --host <addr>         Server bind address (default: 127.0.0.1)\n"
           "  --port <port>         Server listen port (default: 8080)\n"
           "  --model-name <name>   Model ID reported by /v1/models (default: inkling-2-mini)\n"
           "  --prompt <text>       User prompt (rendered with Inkling 2 chat template)\n"
           "  --raw-prompt <text>   Raw prompt string (no chat template)\n"
           "  --think <effort>      Thinking effort (0, 0.1, 0.2, 0.7, 0.9, 0.99; default: 0)\n"
           "  --max-tokens <N>      Max generated tokens (default: 128)\n"
           "  --batch <B>           Concurrent batch size 1..8 (default: 1)\n"
           "  --spec                Enable DSpark speculative decoding (B=1, greedy)\n"
           "  --no-spec             Disable speculative decoding (e.g., with --serve)\n"
           "  --draft-depth <D>     Speculative draft depth: 0=adaptive (default), 1..7=fixed\n"
           "  --temp <T>            Sampling temperature (default: 0.0 greedy)\n"
           "  --top-p <P>           Nucleus sampling top-p (default: 1.0)\n"
           "  --seed <S>            PRNG seed for sampling (default: 42)\n"
           "  --manifest <path>     Ollama manifest path\n"
           "  --blobs <dir>         Ollama blobs directory\n"
           "  --metal <path>        Path to kernels.metal\n"
           "  --stream              Stream tokens as generated\n"
           "  --dump-ids            Print prompt and generated token IDs\n",
           prog);
}

int main(int argc, char** argv) {
    const char* home_env = getenv("HOME");
    std::string home = home_env ? home_env : "";
    std::string manifest = home + "/.ollama/models/manifests/registry.ollama.ai/library/inkling-2-mini/26b-a5b-nvfp4";
    std::string blobs = home + "/.ollama/models/blobs";
    std::string metal_path = "src/kernels.metal";
    std::string prompt = "Write a short Python function to compute Fibonacci numbers iteratively and explain its time and space complexity.";
    std::string think = "0";
    bool raw_prompt = false;
    int max_tokens = 128;
    int batch_size = 1;
    bool use_spec = false;
    bool spec_explicit = false;
    int draft_depth = 0; // adaptive by default
    bool serve_mode = false;
    ServerConfig srv_cfg;
    SamplingParams sampling;
    bool stream = false;
    bool dump_ids = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--serve") { serve_mode = true; }
        else if (a == "--host" && i + 1 < argc) { srv_cfg.host = argv[++i]; }
        else if (a == "--port" && i + 1 < argc) { srv_cfg.port = atoi(argv[++i]); }
        else if (a == "--model-name" && i + 1 < argc) { srv_cfg.model_name = argv[++i]; }
        else if (a == "--prompt" && i + 1 < argc) { prompt = argv[++i]; raw_prompt = false; }
        else if (a == "--raw-prompt" && i + 1 < argc) { prompt = argv[++i]; raw_prompt = true; }
        else if (a == "--think" && i + 1 < argc) { think = argv[++i]; }
        else if (a == "--max-tokens" && i + 1 < argc) { max_tokens = atoi(argv[++i]); }
        else if (a == "--batch" && i + 1 < argc) { batch_size = atoi(argv[++i]); }
        else if (a == "--spec") { use_spec = true; spec_explicit = true; }
        else if (a == "--no-spec") { use_spec = false; spec_explicit = true; }
        else if (a == "--draft-depth" && i + 1 < argc) { draft_depth = atoi(argv[++i]); }
        else if (a == "--temp" && i + 1 < argc) { sampling.temperature = static_cast<float>(atof(argv[++i])); }
        else if (a == "--top-p" && i + 1 < argc) { sampling.top_p = static_cast<float>(atof(argv[++i])); }
        else if (a == "--seed" && i + 1 < argc) { sampling.seed = static_cast<uint64_t>(strtoull(argv[++i], nullptr, 10)); }
        else if (a == "--manifest" && i + 1 < argc) { manifest = argv[++i]; }
        else if (a == "--blobs" && i + 1 < argc) { blobs = argv[++i]; }
        else if (a == "--metal" && i + 1 < argc) { metal_path = argv[++i]; }
        else if (a == "--stream") { stream = true; }
        else if (a == "--dump-ids") { dump_ids = true; }
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
    }

    if (serve_mode && !spec_explicit) {
        use_spec = true; // Fast DSpark speculative mode by default when serving
    }

    if (!serve_mode && use_spec && sampling.temperature > 0.0f) {
        fprintf(stderr, "[warn] Speculative decoding uses exact greedy verification; ignoring --temp %.2f.\n",
                sampling.temperature);
    }

    InklingEngine engine;
    if (!engine.init(manifest, blobs, metal_path, use_spec, 4096)) {
        fprintf(stderr, "Engine initialization failed.\n");
        return 1;
    }

    if (serve_mode) {
        srv_cfg.use_spec = use_spec;
        srv_cfg.draft_depth = draft_depth;
        srv_cfg.default_think = think;
        srv_cfg.max_seq = 4096;
        return run_openai_server(engine, srv_cfg);
    }

    std::string rendered = raw_prompt ? prompt : InklingTokenizer::render_prompt(prompt, think);
    auto prompt_ids = engine.tokenizer.encode(rendered);
    fprintf(stderr, "[prompt] %zu tokens\n", prompt_ids.size());
    if (dump_ids) {
        fprintf(stderr, "prompt_ids:");
        for (int32_t id : prompt_ids) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
    }

    GenStats stats;
    if (batch_size > 1) {
        std::vector<std::vector<int32_t>> prompts(batch_size, prompt_ids);
        auto outs = engine.generate_batch(prompts, max_tokens, stats, sampling);
        double dec_tps = (stats.gen_tokens - batch_size) / (stats.decode_ms / 1000.0);
        printf("=== Batch B=%d Output (seq 0) ===\n%s\n", batch_size, engine.tokenizer.decode(outs[0]).c_str());
        fprintf(stderr, "[bench B=%d] prefill=%d tok in %.1fms (%.1f tok/s) | decode=%d tok in %.1fms (%.2f agg tok/s, %.2f tok/s/seq)\n",
                batch_size, stats.prompt_tokens, stats.prefill_ms,
                stats.prompt_tokens / (stats.prefill_ms / 1000.0),
                stats.gen_tokens - batch_size, stats.decode_ms, dec_tps, dec_tps / batch_size);
    } else {
        auto seq = engine.create_sequence(4096);
        std::vector<int32_t> out;
        if (use_spec) {
            out = engine.generate_speculative(seq, prompt_ids, max_tokens, draft_depth, stats, stream);
        } else {
            out = engine.generate_plain(seq, prompt_ids, max_tokens, stats, stream, sampling);
        }
        if (!stream) {
            printf("%s\n", engine.tokenizer.decode(out).c_str());
        } else {
            printf("\n");
        }
        if (dump_ids) {
            fprintf(stderr, "gen_ids:");
            for (int32_t id : out) fprintf(stderr, " %d", id);
            fprintf(stderr, "\n");
        }
        int dec_toks = std::max(0, stats.gen_tokens - 1);
        double dec_tps = stats.decode_ms > 0 ? dec_toks / (stats.decode_ms / 1000.0) : 0.0;
        if (use_spec) {
            char dlabel[16];
            if (draft_depth <= 0) snprintf(dlabel, sizeof(dlabel), "auto");
            else snprintf(dlabel, sizeof(dlabel), "%d", draft_depth);
            fprintf(stderr, "[bench spec d=%s] prefill=%d tok in %.1fms (%.1f tok/s) | decode=%d tok in %.1fms (%.2f tok/s) | rounds=%d drafted=%d accepted=%d (%.1f%%, %.2f tok/round)\n",
                    dlabel, stats.prompt_tokens, stats.prefill_ms,
                    stats.prompt_tokens / (stats.prefill_ms / 1000.0),
                    dec_toks, stats.decode_ms, dec_tps,
                    stats.spec_rounds, stats.spec_drafted, stats.spec_accepted,
                    stats.spec_drafted ? 100.0 * stats.spec_accepted / stats.spec_drafted : 0.0,
                    stats.spec_rounds ? double(stats.spec_accepted + stats.spec_rounds) / stats.spec_rounds : 0.0);
        } else {
            fprintf(stderr, "[bench plain B=1] prefill=%d tok in %.1fms (%.1f tok/s) | decode=%d tok in %.1fms (%.2f tok/s)\n",
                    stats.prompt_tokens, stats.prefill_ms,
                    stats.prompt_tokens / (stats.prefill_ms / 1000.0),
                    dec_toks, stats.decode_ms, dec_tps);
        }
    }
    return 0;
}
