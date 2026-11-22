#pragma once
#include "engine.h"
#include <string>

struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    bool use_spec = true;
    int draft_depth = 0; // 0 = adaptive
    std::string default_think = "0";
    std::string model_name = "inkling-2-mini";
    int max_seq = 262144;
    size_t cache_ram_mb = 4096;
    size_t cache_disk_mb = 16384;
    std::string cache_dir;
};

int run_openai_server(InklingEngine& engine, const ServerConfig& cfg);
