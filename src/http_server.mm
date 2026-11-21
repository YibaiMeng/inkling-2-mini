#import <Foundation/Foundation.h>
#include "http_server.h"
#include "json_lite.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::atomic<bool> g_stop_server{false};

void handle_signal(int) {
    g_stop_server.store(true);
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    return out;
}

bool send_all(int fd, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, data + sent, len - sent, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

bool send_str(int fd, const std::string& s) {
    return send_all(fd, s.data(), s.size());
}

void send_http_json(int fd, int status, const char* status_text, const std::string& body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << " " << status_text << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    send_str(fd, oss.str());
}

void send_openai_error(int fd, int status, const char* status_text, const std::string& msg,
                       const char* type = "invalid_request_error") {
    std::string body = "{\"error\":{\"message\":\"" + json_escape(msg) +
                       "\",\"type\":\"" + std::string(type) + "\",\"code\":null}}";
    send_http_json(fd, status, status_text, body);
}

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
};

std::string to_lower_ascii(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool read_http_request(int fd, HttpRequest& req) {
    constexpr size_t kMaxHeaderBytes = 64 * 1024;
    constexpr size_t kMaxBodyBytes = 8 * 1024 * 1024;
    std::string buf;
    buf.reserve(4096);
    char tmp[4096];

    size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        buf.append(tmp, static_cast<size_t>(n));
        header_end = buf.find("\r\n\r\n");
        if (buf.size() > kMaxHeaderBytes && header_end == std::string::npos) return false;
    }

    std::string header_block = buf.substr(0, header_end);
    size_t line_end = header_block.find("\r\n");
    std::string request_line = (line_end == std::string::npos) ? header_block : header_block.substr(0, line_end);
    std::istringstream rl(request_line);
    std::string version;
    if (!(rl >> req.method >> req.path >> version)) return false;

    // Strip query string from path
    size_t qpos = req.path.find('?');
    if (qpos != std::string::npos) req.path.resize(qpos);

    size_t content_length = 0;
    bool expect_continue = false;
    size_t pos = (line_end == std::string::npos) ? header_block.size() : line_end + 2;
    while (pos < header_block.size()) {
        size_t next = header_block.find("\r\n", pos);
        std::string line = (next == std::string::npos) ? header_block.substr(pos) : header_block.substr(pos, next - pos);
        size_t colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = to_lower_ascii(line.substr(0, colon));
            std::string val = line.substr(colon + 1);
            size_t vs = val.find_first_not_of(" \t");
            val = (vs == std::string::npos) ? "" : val.substr(vs);
            if (key == "content-length") {
                content_length = static_cast<size_t>(std::strtoull(val.c_str(), nullptr, 10));
            } else if (key == "expect" && to_lower_ascii(val).find("100-continue") != std::string::npos) {
                expect_continue = true;
            }
        }
        if (next == std::string::npos) break;
        pos = next + 2;
    }

    if (content_length > kMaxBodyBytes) return false;
    if (expect_continue) {
        send_str(fd, "HTTP/1.1 100 Continue\r\n\r\n");
    }

    req.body = buf.substr(header_end + 4);
    while (req.body.size() < content_length) {
        size_t want = std::min(sizeof(tmp), content_length - req.body.size());
        ssize_t n = recv(fd, tmp, want, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        req.body.append(tmp, static_cast<size_t>(n));
    }
    if (req.body.size() > content_length) req.body.resize(content_length);
    return true;
}

// Map OpenAI reasoning_effort / think parameter to Inkling 2 effort string
std::string map_think_effort(const json_lite::Value* val, const std::string& fallback) {
    if (!val || val->is_null()) return fallback;
    if (val->is_bool()) return val->as_bool() ? "0.9" : "0";
    if (val->is_number()) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%g", val->as_double());
        return std::string(buf);
    }
    if (val->is_string()) {
        std::string s = to_lower_ascii(val->as_string());
        if (s == "none" || s == "off" || s == "false") return "0";
        if (s == "minimal") return "0.1";
        if (s == "low") return "0.2";
        if (s == "medium") return "0.7";
        if (s == "high" || s == "true") return "0.9";
        if (s == "xhigh" || s == "max") return "0.99";
        return val->as_string();
    }
    return fallback;
}

// Extract text from OpenAI content (either string or [{"type":"text","text":"..."}])
std::string extract_content_string(const json_lite::Value* val) {
    if (!val || val->is_null()) return "";
    if (val->is_string()) return val->as_string();
    if (val->is_array()) {
        std::string out;
        for (const auto& item : val->as_array()) {
            if (item.is_object()) {
                const auto* tv = item.get("text");
                if (tv && tv->is_string()) out += tv->as_string();
            } else if (item.is_string()) {
                out += item.as_string();
            }
        }
        return out;
    }
    return "";
}

// Returns the longest prefix length of `s` that ends on a complete UTF-8 codepoint boundary.
size_t valid_utf8_prefix_len(const std::string& s) {
    if (s.empty()) return 0;
    size_t n = s.size();
    // Walk back up to 3 bytes to inspect the last leading byte
    size_t i = n - 1;
    size_t back = 0;
    while (back < 3 && i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) {
        --i;
        ++back;
    }
    unsigned char lead = static_cast<unsigned char>(s[i]);
    size_t expected = 1;
    if ((lead & 0x80) == 0x00) expected = 1;
    else if ((lead & 0xE0) == 0xC0) expected = 2;
    else if ((lead & 0xF0) == 0xE0) expected = 3;
    else if ((lead & 0xF8) == 0xF0) expected = 4;
    else return n; // invalid byte, flush as-is

    size_t avail = n - i;
    return (avail >= expected) ? n : i;
}

struct StreamChunkBuilder {
    std::string id;
    int64_t created;
    std::string model;

    std::string make_chunk(const std::string& delta_json, const char* finish_reason,
                           const std::string& usage_json = "") const {
        std::ostringstream oss;
        oss << "data: {\"id\":\"" << id
            << "\",\"object\":\"chat.completion.chunk\",\"created\":" << created
            << ",\"model\":\"" << json_escape(model)
            << "\",\"choices\":[{\"index\":0,\"delta\":" << delta_json
            << ",\"logprobs\":null,\"finish_reason\":";
        if (finish_reason) oss << "\"" << finish_reason << "\"";
        else oss << "null";
        oss << "}]";
        if (!usage_json.empty()) oss << ",\"usage\":" << usage_json;
        oss << "}\n\n";
        return oss.str();
    }
};

struct OutputChannelParser {
    enum class Mode { Idle, Thinking, Text };

    int32_t tok_msg_model = -1;
    int32_t tok_thinking = -1;
    int32_t tok_text = -1;
    int32_t tok_end_msg = -1;
    int32_t tok_eos = -1;

    Mode mode = Mode::Idle;
    std::string utf8_buf;
    std::string text_pending;
    std::string reasoning_acc;
    std::string content_acc;
    std::vector<std::string> stops;
    size_t max_stop_len = 0;
    bool hit_stop = false;

    void init(const InklingTokenizer& tok, const std::vector<std::string>& stop_list) {
        tok_msg_model = tok.token_id("<|message_model|>");
        tok_thinking  = tok.token_id("<|content_thinking|>");
        tok_text      = tok.token_id("<|content_text|>");
        tok_end_msg   = tok.token_id("<|end_message|>");
        tok_eos       = tok.eos_id();
        stops = stop_list;
        max_stop_len = 0;
        for (const auto& st : stops) max_stop_len = std::max(max_stop_len, st.size());
    }

    // Feed one token ID. Emits ready text via `emit(is_reasoning, chunk)`.
    // Returns false if generation should stop (EOS, stop sequence, or emit failed).
    template <typename EmitFn>
    bool feed(const InklingTokenizer& tok, int32_t id, EmitFn&& emit) {
        if (id == tok_eos) {
            flush_all(emit);
            return false;
        }
        if (id == tok_thinking) {
            flush_utf8(emit);
            mode = Mode::Thinking;
            return true;
        }
        if (id == tok_text) {
            flush_utf8(emit);
            mode = Mode::Text;
            return true;
        }
        if (id == tok_msg_model || id == tok_end_msg) {
            flush_utf8(emit);
            mode = Mode::Idle;
            return true;
        }
        if (tok.is_special(id)) {
            return true;
        }

        std::string piece = tok.decode_token(id);
        if (piece.empty()) return true;
        utf8_buf += piece;

        size_t ready = valid_utf8_prefix_len(utf8_buf);
        if (ready == 0) return true;
        std::string valid = utf8_buf.substr(0, ready);
        utf8_buf.erase(0, ready);

        if (mode == Mode::Thinking) {
            reasoning_acc += valid;
            return emit(true, valid);
        }

        // Text mode (or Idle fallback)
        text_pending += valid;
        if (!stops.empty()) {
            // Check if any stop string appears inside text_pending
            size_t earliest = std::string::npos;
            for (const auto& st : stops) {
                if (st.empty()) continue;
                size_t p = text_pending.find(st);
                if (p != std::string::npos && p < earliest) earliest = p;
            }
            if (earliest != std::string::npos) {
                std::string before = text_pending.substr(0, earliest);
                text_pending.clear();
                hit_stop = true;
                if (!before.empty()) {
                    content_acc += before;
                    emit(false, before);
                }
                return false;
            }
            // Hold back up to max_stop_len - 1 bytes in case a stop sequence spans tokens
            if (text_pending.size() >= max_stop_len) {
                size_t emit_len = text_pending.size() - (max_stop_len - 1);
                emit_len = valid_utf8_prefix_len(text_pending.substr(0, emit_len));
                if (emit_len > 0) {
                    std::string chunk = text_pending.substr(0, emit_len);
                    text_pending.erase(0, emit_len);
                    content_acc += chunk;
                    return emit(false, chunk);
                }
            }
            return true;
        } else {
            std::string chunk;
            chunk.swap(text_pending);
            content_acc += chunk;
            return emit(false, chunk);
        }
    }

    template <typename EmitFn>
    void flush_utf8(EmitFn&& emit) {
        if (!utf8_buf.empty()) {
            std::string rem;
            rem.swap(utf8_buf);
            if (mode == Mode::Thinking) {
                reasoning_acc += rem;
                emit(true, rem);
            } else if (mode == Mode::Text) {
                text_pending += rem;
            }
        }
    }

    template <typename EmitFn>
    void flush_all(EmitFn&& emit) {
        flush_utf8(emit);
        if (!text_pending.empty()) {
            std::string rem;
            rem.swap(text_pending);
            content_acc += rem;
            emit(false, rem);
        }
    }
};

} // namespace

int run_openai_server(InklingEngine& engine, const ServerConfig& cfg) {
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        fprintf(stderr, "[server] socket() failed: %s\n", strerror(errno));
        return 1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(cfg.port));
    if (inet_pton(AF_INET, cfg.host.c_str(), &addr.sin_addr) <= 0) {
        fprintf(stderr, "[server] Invalid bind host: %s\n", cfg.host.c_str());
        close(server_fd);
        return 1;
    }

    if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        fprintf(stderr, "[server] bind(%s:%d) failed: %s\n", cfg.host.c_str(), cfg.port, strerror(errno));
        close(server_fd);
        return 1;
    }
    if (listen(server_fd, 16) < 0) {
        fprintf(stderr, "[server] listen() failed: %s\n", strerror(errno));
        close(server_fd);
        return 1;
    }

    // Allocate one persistent reusable SequenceState so KV/conv Metal buffers are never reallocated!
    SequenceState seq = engine.create_sequence(cfg.max_seq);
    uint64_t req_counter = 0;

    fprintf(stderr,
            "[server] OpenAI-compatible API listening on http://%s:%d\n"
            "[server] Endpoints: GET /v1/models | POST /v1/chat/completions | GET /health\n"
            "[server] Model: %s | Speculative: %s (draft_depth=%s) | Default think: %s\n",
            cfg.host.c_str(), cfg.port, cfg.model_name.c_str(),
            cfg.use_spec ? "on" : "off",
            cfg.draft_depth <= 0 ? "auto" : std::to_string(cfg.draft_depth).c_str(),
            cfg.default_think.c_str());

    while (!g_stop_server.load()) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(server_fd, &rfds);
        timeval tv{0, 250000}; // 250ms poll for clean SIGINT handling
        int sel = select(server_fd + 1, &rfds, nullptr, nullptr, &tv);
        if (sel <= 0) continue;

        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) continue;

        HttpRequest req;
        if (!read_http_request(client_fd, req)) {
            close(client_fd);
            continue;
        }

        if (req.method == "OPTIONS") {
            send_str(client_fd,
                     "HTTP/1.1 204 No Content\r\n"
                     "Access-Control-Allow-Origin: *\r\n"
                     "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                     "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
                     "Connection: close\r\n\r\n");
            close(client_fd);
            continue;
        }

        if (req.method == "GET" && req.path == "/health") {
            std::string body = "{\"status\":\"ok\",\"model\":\"" + json_escape(cfg.model_name) +
                               "\",\"speculative\":" + (cfg.use_spec ? "true" : "false") + "}";
            send_http_json(client_fd, 200, "OK", body);
            close(client_fd);
            continue;
        }

        if (req.method == "GET" && (req.path == "/v1/models" || req.path == "/models")) {
            int64_t now_ts = static_cast<int64_t>(std::time(nullptr));
            std::ostringstream oss;
            oss << "{\"object\":\"list\",\"data\":[{\"id\":\"" << json_escape(cfg.model_name)
                << "\",\"object\":\"model\",\"created\":" << now_ts
                << ",\"owned_by\":\"inkling-2-mini-metal\"}]}";
            send_http_json(client_fd, 200, "OK", oss.str());
            close(client_fd);
            continue;
        }

        if (req.path != "/v1/chat/completions" && req.path != "/chat/completions") {
            send_openai_error(client_fd, 404, "Not Found", "Unknown endpoint: " + req.path, "not_found_error");
            close(client_fd);
            continue;
        }
        if (req.method != "POST") {
            send_openai_error(client_fd, 405, "Method Not Allowed", "Expected POST for " + req.path);
            close(client_fd);
            continue;
        }

        // Parse OpenAI Chat Completions JSON body
        json_lite::Value root;
        try {
            root = json_lite::parse(req.body);
        } catch (const std::exception& e) {
            send_openai_error(client_fd, 400, "Bad Request", std::string("Invalid JSON: ") + e.what());
            close(client_fd);
            continue;
        }

        if (!root.is_object()) {
            send_openai_error(client_fd, 400, "Bad Request", "Request body must be a JSON object");
            close(client_fd);
            continue;
        }

        const auto* msgs_val = root.get("messages");
        if (!msgs_val || !msgs_val->is_array() || msgs_val->as_array().empty()) {
            send_openai_error(client_fd, 400, "Bad Request", "Missing or empty 'messages' array");
            close(client_fd);
            continue;
        }

        std::vector<ChatMessage> messages;
        messages.reserve(msgs_val->as_array().size());
        for (const auto& m : msgs_val->as_array()) {
            if (!m.is_object()) continue;
            ChatMessage cm;
            const auto* rv = m.get("role");
            cm.role = (rv && rv->is_string()) ? rv->as_string() : "user";
            cm.content = extract_content_string(m.get("content"));
            const auto* rcv = m.get("reasoning_content");
            if (rcv && rcv->is_string()) cm.reasoning_content = rcv->as_string();
            messages.push_back(std::move(cm));
        }

        std::string req_model = cfg.model_name;
        if (const auto* mv = root.get("model"); mv && mv->is_string() && !mv->as_string().empty()) {
            req_model = mv->as_string();
        }

        int max_tokens = 512;
        if (const auto* mt = root.get("max_completion_tokens"); mt && mt->is_number()) {
            max_tokens = std::max(1, mt->as_int());
        } else if (const auto* mt2 = root.get("max_tokens"); mt2 && mt2->is_number()) {
            max_tokens = std::max(1, mt2->as_int());
        }

        SamplingParams sampling;
        sampling.temperature = 0.0f;
        sampling.top_p = 1.0f;
        sampling.seed = static_cast<uint64_t>(std::time(nullptr)) ^ (++req_counter * 2654435761ULL);
        if (const auto* tv_val = root.get("temperature"); tv_val && tv_val->is_number()) {
            sampling.temperature = std::max(0.0f, tv_val->as_float());
        }
        if (const auto* pv = root.get("top_p"); pv && pv->is_number()) {
            sampling.top_p = std::clamp(pv->as_float(), 0.01f, 1.0f);
        }
        if (const auto* sv = root.get("seed"); sv && sv->is_number()) {
            sampling.seed = static_cast<uint64_t>(sv->as_i64());
        }

        bool stream = false;
        if (const auto* stv = root.get("stream"); stv && stv->is_bool()) {
            stream = stv->as_bool();
        }

        bool include_usage = false;
        if (const auto* sov = root.get("stream_options"); sov && sov->is_object()) {
            if (const auto* iuv = sov->get("include_usage"); iuv && iuv->is_bool()) {
                include_usage = iuv->as_bool();
            }
        }

        std::vector<std::string> stops;
        if (const auto* stopv = root.get("stop")) {
            if (stopv->is_string() && !stopv->as_string().empty()) {
                stops.push_back(stopv->as_string());
            } else if (stopv->is_array()) {
                for (const auto& item : stopv->as_array()) {
                    if (item.is_string() && !item.as_string().empty() && stops.size() < 16) {
                        stops.push_back(item.as_string());
                    }
                }
            }
        }

        std::string think_effort = cfg.default_think;
        if (const auto* rv = root.get("reasoning_effort")) {
            think_effort = map_think_effort(rv, think_effort);
        } else if (const auto* thv = root.get("think")) {
            think_effort = map_think_effort(thv, think_effort);
        }

        std::string rendered = InklingTokenizer::render_messages(messages, think_effort);
        std::vector<int32_t> prompt_ids = engine.tokenizer.encode(rendered);
        if (static_cast<int>(prompt_ids.size()) + 1 >= cfg.max_seq) {
            send_openai_error(client_fd, 400, "Bad Request",
                              "Prompt exceeds max context length (" + std::to_string(cfg.max_seq) + ")");
            close(client_fd);
            continue;
        }
        int clamped_max_tokens = std::min(max_tokens, cfg.max_seq - static_cast<int>(prompt_ids.size()) - 1);

        int64_t created_ts = static_cast<int64_t>(std::time(nullptr));
        char id_buf[64];
        snprintf(id_buf, sizeof(id_buf), "chatcmpl-%lld-%llu",
                 static_cast<long long>(created_ts), static_cast<unsigned long long>(req_counter));
        std::string comp_id(id_buf);

        // Speculative decoding is used when enabled and greedy (temperature == 0);
        // non-zero temperature automatically falls back to exact plain sampling.
        bool run_spec = cfg.use_spec && engine.weights.draft.loaded && sampling.temperature <= 0.0f;

        OutputChannelParser parser;
        parser.init(engine.tokenizer, stops);
        StreamChunkBuilder scb{comp_id, created_ts, req_model};

        bool client_ok = true;
        if (stream) {
            std::string sse_hdr =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Cache-Control: no-cache\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n\r\n";
            client_ok = send_str(client_fd, sse_hdr);
            if (client_ok) {
                client_ok = send_str(client_fd, scb.make_chunk("{\"role\":\"assistant\",\"content\":\"\"}", nullptr));
            }
        }

        auto emit_piece = [&](bool is_reasoning, const std::string& text) -> bool {
            if (!stream || text.empty()) return true;
            if (!client_ok) return false;
            std::string delta = is_reasoning
                                    ? ("{\"reasoning_content\":\"" + json_escape(text) + "\"}")
                                    : ("{\"content\":\"" + json_escape(text) + "\"}");
            client_ok = send_str(client_fd, scb.make_chunk(delta, nullptr));
            return client_ok;
        };

        auto on_token = [&](int32_t tok_id) -> bool {
            return parser.feed(engine.tokenizer, tok_id, emit_piece);
        };

        GenStats stats;
        std::vector<int32_t> out_ids;
        if (client_ok) {
            if (run_spec) {
                out_ids = engine.generate_speculative(seq, prompt_ids, clamped_max_tokens,
                                                      cfg.draft_depth, stats, false, on_token);
            } else {
                out_ids = engine.generate_plain(seq, prompt_ids, clamped_max_tokens,
                                                stats, false, sampling, on_token);
            }
        }

        // Flush any remaining held-back bytes if generation hit max_tokens or EOS
        if (client_ok) {
            parser.flush_all(emit_piece);
        }

        bool hit_eos = (!out_ids.empty() && out_ids.back() == engine.tokenizer.eos_id());
        const char* finish_reason = (hit_eos || parser.hit_stop) ? "stop" : "length";
        int prompt_toks = stats.prompt_tokens;
        int comp_toks = stats.gen_tokens;

        if (stream) {
            if (client_ok) {
                send_str(client_fd, scb.make_chunk("{}", finish_reason));
                if (include_usage) {
                    std::ostringstream uoss;
                    uoss << "{\"prompt_tokens\":" << prompt_toks
                         << ",\"completion_tokens\":" << comp_toks
                         << ",\"total_tokens\":" << (prompt_toks + comp_toks) << "}";
                    std::ostringstream coss;
                    coss << "data: {\"id\":\"" << comp_id
                         << "\",\"object\":\"chat.completion.chunk\",\"created\":" << created_ts
                         << ",\"model\":\"" << json_escape(req_model)
                         << "\",\"choices\":[],\"usage\":" << uoss.str() << "}\n\n";
                    send_str(client_fd, coss.str());
                }
                send_str(client_fd, "data: [DONE]\n\n");
            }
        } else {
            std::ostringstream body;
            body << "{\"id\":\"" << comp_id
                 << "\",\"object\":\"chat.completion\",\"created\":" << created_ts
                 << ",\"model\":\"" << json_escape(req_model)
                 << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\""
                 << json_escape(parser.content_acc) << "\"";
            if (!parser.reasoning_acc.empty()) {
                body << ",\"reasoning_content\":\"" << json_escape(parser.reasoning_acc) << "\"";
            }
            body << "},\"logprobs\":null,\"finish_reason\":\"" << finish_reason << "\"}]"
                 << ",\"usage\":{\"prompt_tokens\":" << prompt_toks
                 << ",\"completion_tokens\":" << comp_toks
                 << ",\"total_tokens\":" << (prompt_toks + comp_toks) << "}}";
            send_http_json(client_fd, 200, "OK", body.str());
        }

        close(client_fd);

        int dec_toks = std::max(0, stats.gen_tokens - 1);
        double dec_tps = stats.decode_ms > 0.0 ? dec_toks / (stats.decode_ms / 1000.0) : 0.0;
        fprintf(stderr,
                "[req #%llu] %s stream=%d spec=%d prompt=%d (%.0fms) gen=%d (%.1fms, %.2f tok/s) finish=%s\n",
                static_cast<unsigned long long>(req_counter), req.path.c_str(), stream ? 1 : 0,
                run_spec ? 1 : 0, prompt_toks, stats.prefill_ms, comp_toks, stats.decode_ms, dec_tps,
                finish_reason);
    }

    close(server_fd);
    fprintf(stderr, "[server] Shutdown cleanly.\n");
    return 0;
}
