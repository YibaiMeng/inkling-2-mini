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
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
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
    constexpr size_t kMaxBodyBytes = 64 * 1024 * 1024;

    // Prevent idle/half-open TCP connections from blocking forever
    timeval rtv{15, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rtv, sizeof(rtv));

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
    bool is_chunked = false;
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
            } else if (key == "transfer-encoding" && to_lower_ascii(val).find("chunked") != std::string::npos) {
                is_chunked = true;
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

    std::string raw = buf.substr(header_end + 4);
    if (is_chunked) {
        req.body.clear();
        size_t cursor = 0;
        auto fill_more = [&]() -> bool {
            ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
            if (n < 0) return errno == EINTR;
            if (n == 0) return false;
            raw.append(tmp, static_cast<size_t>(n));
            return raw.size() <= kMaxBodyBytes + 65536;
        };
        while (true) {
            size_t crlf = std::string::npos;
            while ((crlf = raw.find("\r\n", cursor)) == std::string::npos) {
                if (!fill_more()) return false;
            }
            std::string sz_line = raw.substr(cursor, crlf - cursor);
            size_t semi = sz_line.find(';');
            if (semi != std::string::npos) sz_line.resize(semi);
            size_t chunk_len = static_cast<size_t>(std::strtoull(sz_line.c_str(), nullptr, 16));
            cursor = crlf + 2;
            if (chunk_len == 0) break;
            if (req.body.size() + chunk_len > kMaxBodyBytes) return false;
            while (raw.size() < cursor + chunk_len + 2) {
                if (!fill_more()) return false;
            }
            req.body.append(raw.data() + cursor, chunk_len);
            cursor += chunk_len + 2;
        }
        return true;
    }

    req.body = std::move(raw);
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

// Serialize OpenAI `tools` array (either Chat Completions or Responses format) into Inkling 2's
// compact tool_declare JSON array matching ollama/model/renderers/inkling2.go.
std::string build_tools_declare_json(const json_lite::Value* tools_val) {
    if (!tools_val || !tools_val->is_array() || tools_val->as_array().empty()) return "";
    json_lite::Array specs;
    for (const auto& t : tools_val->as_array()) {
        if (!t.is_object()) continue;
        std::string kind = "function";
        if (const auto* kv = t.get("type"); kv && kv->is_string() && !kv->as_string().empty()) {
            kind = kv->as_string();
        }
        const json_lite::Value* fn_obj = t.get("function");
        const json_lite::Value& src = (fn_obj && fn_obj->is_object()) ? *fn_obj : t;

        const auto* name_v = src.get("name");
        if (!name_v || !name_v->is_string() || name_v->as_string().empty()) continue;

        json_lite::Object spec;
        spec["name"] = json_lite::Value(name_v->as_string());
        spec["type"] = json_lite::Value(kind);
        if (const auto* desc_v = src.get("description"); desc_v && desc_v->is_string()) {
            spec["description"] = json_lite::Value(desc_v->as_string());
        } else {
            spec["description"] = json_lite::Value(std::string(""));
        }
        if (const auto* params_v = src.get("parameters"); params_v && params_v->is_object()) {
            spec["parameters"] = *params_v;
        } else {
            spec["parameters"] = json_lite::Value(json_lite::Object{});
        }
        specs.emplace_back(std::move(spec));
    }
    if (specs.empty()) return "";
    return json_lite::dump_compact(json_lite::Value(std::move(specs)));
}

struct OutputChannelParser {
    enum class Mode { Idle, ModelHeader, Thinking, Text, ToolJson, ToolIgnore };

    int32_t tok_msg_model = -1;
    int32_t tok_thinking = -1;
    int32_t tok_text = -1;
    int32_t tok_tool_json = -1;
    int32_t tok_tool_text = -1;
    int32_t tok_tool_error = -1;
    int32_t tok_end_msg = -1;
    int32_t tok_eos = -1;

    Mode mode = Mode::Idle;
    std::string utf8_buf;
    std::string header_name;
    std::string tool_json_buf;
    std::string text_pending;
    std::string reasoning_acc;
    std::string content_acc;
    std::vector<ChatToolCall> tool_calls;
    std::string call_id_prefix;
    std::vector<std::string> stops;
    size_t max_stop_len = 0;
    bool hit_stop = false;

    void init(const InklingTokenizer& tok, const std::vector<std::string>& stop_list,
              const std::string& id_prefix) {
        tok_msg_model  = tok.token_id("<|message_model|>");
        tok_thinking   = tok.token_id("<|content_thinking|>");
        tok_text       = tok.token_id("<|content_text|>");
        tok_tool_json  = tok.token_id("<|content_invoke_tool_json|>");
        tok_tool_text  = tok.token_id("<|content_invoke_tool_text|>");
        tok_tool_error = tok.token_id("<|content_tool_error|>");
        tok_end_msg    = tok.token_id("<|end_message|>");
        tok_eos        = tok.eos_id();
        call_id_prefix = id_prefix;
        stops = stop_list;
        max_stop_len = 0;
        for (const auto& st : stops) max_stop_len = std::max(max_stop_len, st.size());
    }

    template <typename ToolFn>
    bool finalize_tool_call(ToolFn&& on_tool) {
        if (tool_json_buf.empty()) return true;
        std::string raw;
        raw.swap(tool_json_buf);
        std::string name = header_name;
        std::string args_json = "{}";
        try {
            json_lite::Value v = json_lite::parse(raw);
            if (v.is_object()) {
                if (const auto* nv = v.get("name"); nv && nv->is_string() && !nv->as_string().empty()) {
                    name = nv->as_string();
                }
                if (const auto* av = v.get("args")) {
                    args_json = json_lite::dump_compact(*av);
                } else if (const auto* av2 = v.get("arguments")) {
                    args_json = av2->is_string() ? av2->as_string() : json_lite::dump_compact(*av2);
                }
            }
        } catch (...) {
            // Best-effort fallback if truncated near max_tokens
            if (name.empty()) return true;
        }
        if (name.empty()) return true;

        size_t idx = tool_calls.size();
        ChatToolCall tc;
        tc.id = call_id_prefix + "_" + std::to_string(idx);
        tc.name = std::move(name);
        tc.arguments_json = std::move(args_json);
        tool_calls.push_back(tc);
        return on_tool(tc, idx);
    }

    // Feed one token ID. Emits ready text via `emit(is_reasoning, chunk)` and tool calls via `on_tool(tc, idx)`.
    // Returns false if generation should stop (EOS, stop sequence, or emit failed).
    template <typename EmitFn, typename ToolFn>
    bool feed(const InklingTokenizer& tok, int32_t id, EmitFn&& emit, ToolFn&& on_tool) {
        if (id == tok_eos) {
            flush_all(emit, on_tool);
            return false;
        }
        if (id == tok_msg_model) {
            flush_utf8(emit);
            if (mode == Mode::ToolJson && !finalize_tool_call(on_tool)) return false;
            mode = Mode::ModelHeader;
            header_name.clear();
            return true;
        }
        if (id == tok_thinking) {
            flush_utf8(emit);
            mode = Mode::Thinking;
            header_name.clear();
            return true;
        }
        if (id == tok_text) {
            flush_utf8(emit);
            mode = Mode::Text;
            header_name.clear();
            return true;
        }
        if (id == tok_tool_json) {
            flush_utf8(emit);
            mode = Mode::ToolJson;
            tool_json_buf.clear();
            return true;
        }
        if (id == tok_tool_text || id == tok_tool_error) {
            flush_utf8(emit);
            mode = Mode::ToolIgnore;
            return true;
        }
        if (id == tok_end_msg) {
            flush_utf8(emit);
            if (mode == Mode::ToolJson && !finalize_tool_call(on_tool)) return false;
            mode = Mode::Idle;
            header_name.clear();
            return true;
        }
        if (tok.is_special(id)) {
            return true;
        }

        std::string piece = tok.decode_token(id);
        if (piece.empty()) return true;

        if (mode == Mode::ModelHeader) {
            header_name += piece;
            return true;
        }
        if (mode == Mode::ToolJson) {
            tool_json_buf += piece;
            return true;
        }
        if (mode == Mode::ToolIgnore) {
            return true;
        }

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

    template <typename EmitFn, typename ToolFn>
    void flush_all(EmitFn&& emit, ToolFn&& on_tool) {
        flush_utf8(emit);
        if (mode == Mode::ToolJson) {
            finalize_tool_call(on_tool);
            mode = Mode::Idle;
        }
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
    std::mutex gpu_mutex;
    std::atomic<uint64_t> req_counter{0};
    std::atomic<int> active_workers{0};

    fprintf(stderr,
            "[server] OpenAI-compatible API listening on http://%s:%d\n"
            "[server] Endpoints: GET /v1/models | POST /v1/chat/completions | POST /v1/responses | GET /health | GET /props\n"
            "[server] Model: %s | Max context: %d | Speculative: %s (draft_depth=%s) | Default think: %s\n",
            cfg.host.c_str(), cfg.port, cfg.model_name.c_str(), cfg.max_seq,
            cfg.use_spec ? "on" : "off",
            cfg.draft_depth <= 0 ? "auto" : std::to_string(cfg.draft_depth).c_str(),
            cfg.default_think.c_str());

    auto handle_client = [&](int client_fd) {
        struct Guard {
            int fd;
            std::atomic<int>& cnt;
            ~Guard() {
                close(fd);
                cnt.fetch_sub(1);
            }
        } guard{client_fd, active_workers};

        @autoreleasepool {
        HttpRequest req;
        if (!read_http_request(client_fd, req)) {
            return;
        }
        fprintf(stderr, "[http] %s %s (body=%zu B)\n", req.method.c_str(), req.path.c_str(), req.body.size());

        if (req.method == "OPTIONS") {
            send_str(client_fd,
                     "HTTP/1.1 204 No Content\r\n"
                     "Access-Control-Allow-Origin: *\r\n"
                     "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                     "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
                     "Connection: close\r\n\r\n");
            return;
        }

        if ((req.method == "GET" || req.method == "HEAD") && req.path == "/health") {
            std::string body = "{\"status\":\"ok\",\"model\":\"" + json_escape(cfg.model_name) +
                               "\",\"context_length\":" + std::to_string(cfg.max_seq) +
                               ",\"speculative\":" + (cfg.use_spec ? "true" : "false") + "}";
            send_http_json(client_fd, 200, "OK", req.method == "HEAD" ? "" : body);
            return;
        }

        if ((req.method == "GET" || req.method == "HEAD") && req.path == "/props") {
            std::ostringstream oss;
            oss << "{\"default_generation_settings\":{\"n_ctx\":" << cfg.max_seq
                << "},\"total_slots\":1,\"model_path\":\"" << json_escape(cfg.model_name) << "\"}";
            send_http_json(client_fd, 200, "OK", req.method == "HEAD" ? "" : oss.str());
            return;
        }

        if ((req.method == "GET" || req.method == "HEAD") && (req.path == "/v1/models" || req.path == "/models")) {
            int64_t now_ts = static_cast<int64_t>(std::time(nullptr));
            std::ostringstream oss;
            oss << "{\"object\":\"list\",\"data\":[{\"id\":\"" << json_escape(cfg.model_name)
                << "\",\"object\":\"model\",\"created\":" << now_ts
                << ",\"owned_by\":\"inkling-2-mini-metal\""
                << ",\"max_model_len\":" << cfg.max_seq
                << ",\"context_length\":" << cfg.max_seq
                << ",\"max_context_length\":" << cfg.max_seq
                << ",\"max_input_tokens\":" << cfg.max_seq
                << ",\"meta\":{\"n_ctx_train\":" << engine.weights.cfg.model_max_length << "}}]}";
            send_http_json(client_fd, 200, "OK", req.method == "HEAD" ? "" : oss.str());
            return;
        }

        bool is_chat = (req.path == "/v1/chat/completions" || req.path == "/chat/completions");
        bool is_responses = (req.path == "/v1/responses" || req.path == "/responses");
        if (!is_chat && !is_responses) {
            send_openai_error(client_fd, 404, "Not Found", "Unknown endpoint: " + req.path, "not_found_error");
            return;
        }
        if (req.method != "POST") {
            send_openai_error(client_fd, 405, "Method Not Allowed", "Expected POST for " + req.path);
            return;
        }

        // Parse JSON body (supports both /v1/chat/completions and /v1/responses)
        json_lite::Value root;
        try {
            root = json_lite::parse(req.body);
        } catch (const std::exception& e) {
            send_openai_error(client_fd, 400, "Bad Request", std::string("Invalid JSON: ") + e.what());
            return;
        }

        if (!root.is_object()) {
            send_openai_error(client_fd, 400, "Bad Request", "Request body must be a JSON object");
            return;
        }

        std::vector<ChatMessage> messages;
        if (is_responses) {
            if (const auto* inst = root.get("instructions"); inst && inst->is_string() && !inst->as_string().empty()) {
                ChatMessage sm;
                sm.role = "system";
                sm.content = inst->as_string();
                messages.push_back(std::move(sm));
            }
            const auto* inp = root.get("input");
            if (!inp) inp = root.get("messages");
            if (!inp || inp->is_null()) {
                send_openai_error(client_fd, 400, "Bad Request", "Missing 'input' field");
                return;
            }
            if (inp->is_string()) {
                ChatMessage um;
                um.role = "user";
                um.content = inp->as_string();
                messages.push_back(std::move(um));
            } else if (inp->is_array()) {
                for (const auto& item : inp->as_array()) {
                    if (item.is_string()) {
                        ChatMessage um;
                        um.role = "user";
                        um.content = item.as_string();
                        messages.push_back(std::move(um));
                        continue;
                    }
                    if (!item.is_object()) continue;
                    std::string itype;
                    if (const auto* type_val = item.get("type"); type_val && type_val->is_string()) {
                        itype = type_val->as_string();
                    }
                    if (itype == "reasoning") {
                        continue;
                    }
                    if (itype == "function_call_output") {
                        ChatMessage tm;
                        tm.role = "tool";
                        tm.content = extract_content_string(item.get("output"));
                        if (const auto* cid = item.get("call_id"); cid && cid->is_string()) {
                            tm.tool_call_id = cid->as_string();
                        }
                        if (const auto* nv = item.get("name"); nv && nv->is_string()) {
                            tm.tool_name = nv->as_string();
                        }
                        messages.push_back(std::move(tm));
                        continue;
                    }
                    if (itype == "function_call") {
                        ChatToolCall tc;
                        if (const auto* cid = item.get("call_id"); cid && cid->is_string()) tc.id = cid->as_string();
                        else if (const auto* idv = item.get("id"); idv && idv->is_string()) tc.id = idv->as_string();
                        if (const auto* nv = item.get("name"); nv && nv->is_string()) tc.name = nv->as_string();
                        if (const auto* av = item.get("arguments")) {
                            tc.arguments_json = av->is_string() ? av->as_string() : json_lite::dump_compact(*av);
                        }
                        if (!messages.empty() && messages.back().role == "assistant") {
                            messages.back().tool_calls.push_back(std::move(tc));
                        } else {
                            ChatMessage am;
                            am.role = "assistant";
                            am.tool_calls.push_back(std::move(tc));
                            messages.push_back(std::move(am));
                        }
                        continue;
                    }
                    ChatMessage cm;
                    const auto* rv = item.get("role");
                    cm.role = (rv && rv->is_string()) ? rv->as_string() : "user";
                    cm.content = extract_content_string(item.get("content"));
                    if (cm.content.empty()) {
                        cm.content = extract_content_string(item.get("text"));
                    }
                    messages.push_back(std::move(cm));
                }
            } else {
                send_openai_error(client_fd, 400, "Bad Request", "'input' must be a string or array");
                return;
            }
            if (messages.empty()) {
                send_openai_error(client_fd, 400, "Bad Request", "Empty 'input' messages");
                return;
            }
        } else {
            const auto* msgs_val = root.get("messages");
            if (!msgs_val || !msgs_val->is_array() || msgs_val->as_array().empty()) {
                send_openai_error(client_fd, 400, "Bad Request", "Missing or empty 'messages' array");
                return;
            }
            messages.reserve(msgs_val->as_array().size());
            for (const auto& m : msgs_val->as_array()) {
                if (!m.is_object()) continue;
                ChatMessage cm;
                const auto* rv = m.get("role");
                cm.role = (rv && rv->is_string()) ? rv->as_string() : "user";
                cm.content = extract_content_string(m.get("content"));
                if (const auto* rcv = m.get("reasoning_content"); rcv && rcv->is_string()) {
                    cm.reasoning_content = rcv->as_string();
                }
                if (const auto* nv = m.get("name"); nv && nv->is_string()) {
                    cm.tool_name = nv->as_string();
                }
                if (const auto* cid = m.get("tool_call_id"); cid && cid->is_string()) {
                    cm.tool_call_id = cid->as_string();
                }
                if (const auto* tcs = m.get("tool_calls"); tcs && tcs->is_array()) {
                    for (const auto& tci : tcs->as_array()) {
                        if (!tci.is_object()) continue;
                        ChatToolCall tc;
                        if (const auto* idv = tci.get("id"); idv && idv->is_string()) tc.id = idv->as_string();
                        const auto* fn = tci.get("function");
                        const auto& src = (fn && fn->is_object()) ? *fn : tci;
                        if (const auto* fn_name = src.get("name"); fn_name && fn_name->is_string()) {
                            tc.name = fn_name->as_string();
                        }
                        if (const auto* fn_args = src.get("arguments")) {
                            tc.arguments_json = fn_args->is_string() ? fn_args->as_string()
                                                                     : json_lite::dump_compact(*fn_args);
                        }
                        if (!tc.name.empty()) cm.tool_calls.push_back(std::move(tc));
                    }
                }
                messages.push_back(std::move(cm));
            }
        }

        std::string tools_declare_json = build_tools_declare_json(root.get("tools"));

        std::string req_model = cfg.model_name;
        if (const auto* mv = root.get("model"); mv && mv->is_string() && !mv->as_string().empty()) {
            req_model = mv->as_string();
        }

        int max_tokens = is_responses ? 1024 : 512;
        if (const auto* mt0 = root.get("max_output_tokens"); mt0 && mt0->is_number()) {
            max_tokens = std::max(1, mt0->as_int());
        } else if (const auto* mt = root.get("max_completion_tokens"); mt && mt->is_number()) {
            max_tokens = std::max(1, mt->as_int());
        } else if (const auto* mt2 = root.get("max_tokens"); mt2 && mt2->is_number()) {
            max_tokens = std::max(1, mt2->as_int());
        }

        uint64_t req_id = req_counter.fetch_add(1) + 1;

        SamplingParams sampling;
        sampling.temperature = 0.0f;
        sampling.top_p = 1.0f;
        sampling.seed = static_cast<uint64_t>(std::time(nullptr)) ^ (req_id * 2654435761ULL);
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
        if (const auto* robj = root.get("reasoning"); robj && robj->is_object()) {
            if (const auto* eff = robj->get("effort")) {
                think_effort = map_think_effort(eff, think_effort);
            }
        } else if (const auto* rv = root.get("reasoning_effort")) {
            think_effort = map_think_effort(rv, think_effort);
        } else if (const auto* thv = root.get("think")) {
            think_effort = map_think_effort(thv, think_effort);
        }

        std::string rendered = InklingTokenizer::render_messages(messages, think_effort, tools_declare_json);
        std::vector<int32_t> prompt_ids = engine.tokenizer.encode(rendered);

        // Compute leading system + tools prefix token count so the prefix cache can snapshot right after it!
        int sys_prefix_len = 0;
        {
            std::vector<ChatMessage> sys_msgs;
            for (const auto& m : messages) {
                if (m.role == "system" || m.role == "developer") sys_msgs.push_back(m);
                else break;
            }
            if (!sys_msgs.empty() || !tools_declare_json.empty()) {
                std::string sys_rendered = InklingTokenizer::render_messages(sys_msgs, think_effort, tools_declare_json);
                auto sys_ids = engine.tokenizer.encode(sys_rendered);
                if (sys_ids.size() < prompt_ids.size() &&
                    std::equal(sys_ids.begin(), sys_ids.end(), prompt_ids.begin())) {
                    sys_prefix_len = static_cast<int>(sys_ids.size());
                }
            }
        }
        if (static_cast<int>(prompt_ids.size()) + 1 >= cfg.max_seq) {
            send_openai_error(client_fd, 400, "Bad Request",
                              "Prompt exceeds max context length (" + std::to_string(cfg.max_seq) + ")");
            return;
        }
        int clamped_max_tokens = std::min(max_tokens, cfg.max_seq - static_cast<int>(prompt_ids.size()) - 1);

        int64_t created_ts = static_cast<int64_t>(std::time(nullptr));
        char id_buf[64];
        snprintf(id_buf, sizeof(id_buf), "%s-%lld-%llu",
                 is_responses ? "resp" : "chatcmpl",
                 static_cast<long long>(created_ts), static_cast<unsigned long long>(req_id));
        std::string comp_id(id_buf);
        std::string rs_item_id = "rs_" + std::to_string(created_ts) + "_" + std::to_string(req_id);
        std::string msg_item_id = "msg_" + std::to_string(created_ts) + "_" + std::to_string(req_id);
        std::string call_id_prefix = "call_" + std::to_string(created_ts) + "_" + std::to_string(req_id);

        fprintf(stderr, "[req #%llu] %s stream=%d prompt=%zu tok (running...)\n",
                static_cast<unsigned long long>(req_id), req.path.c_str(), stream ? 1 : 0, prompt_ids.size());

        std::lock_guard<std::mutex> gpu_lock(gpu_mutex);

        // Speculative decoding is used when enabled and greedy (temperature == 0);
        // non-zero temperature automatically falls back to exact plain sampling.
        bool run_spec = cfg.use_spec && engine.weights.draft.loaded && sampling.temperature <= 0.0f;

        OutputChannelParser parser;
        parser.init(engine.tokenizer, stops, call_id_prefix);
        StreamChunkBuilder scb{comp_id, created_ts, req_model};

        // Responses API streaming state
        bool client_ok = true;
        int resp_seq_num = 0;
        int resp_next_out_idx = 0;
        bool resp_rs_open = false;
        bool resp_rs_closed = false;
        int resp_rs_idx = 0;
        bool resp_msg_open = false;
        bool resp_msg_closed = false;
        int resp_msg_idx = 0;

        auto send_resp_sse = [&](const char* event_name, const std::string& extra_fields) -> bool {
            if (!client_ok) return false;
            std::ostringstream oss;
            oss << "event: " << event_name << "\n"
                << "data: {\"type\":\"" << event_name
                << "\",\"sequence_number\":" << (resp_seq_num++);
            if (!extra_fields.empty()) oss << "," << extra_fields;
            oss << "}\n\n";
            client_ok = send_str(client_fd, oss.str());
            return client_ok;
        };

        auto build_resp_object = [&](const char* status_str, const std::string& output_json,
                                     const std::string& output_text_str, const std::string& usage_json) {
            std::ostringstream oss;
            oss << "{\"id\":\"" << comp_id
                << "\",\"object\":\"response\",\"created_at\":" << created_ts
                << ",\"status\":\"" << status_str
                << "\",\"model\":\"" << json_escape(req_model)
                << "\",\"output\":" << output_json;
            if (!output_text_str.empty() || std::strcmp(status_str, "in_progress") != 0) {
                oss << ",\"output_text\":\"" << json_escape(output_text_str) << "\"";
            }
            if (!usage_json.empty()) oss << ",\"usage\":" << usage_json;
            oss << "}";
            return oss.str();
        };

        auto close_resp_reasoning = [&]() {
            if (!resp_rs_open || resp_rs_closed) return;
            resp_rs_closed = true;
            std::string esc_r = json_escape(parser.reasoning_acc);
            send_resp_sse("response.reasoning_summary_text.done",
                          "\"item_id\":\"" + rs_item_id + "\",\"output_index\":" + std::to_string(resp_rs_idx) +
                              ",\"summary_index\":0,\"text\":\"" + esc_r + "\"");
            send_resp_sse("response.reasoning_summary_part.done",
                          "\"item_id\":\"" + rs_item_id + "\",\"output_index\":" + std::to_string(resp_rs_idx) +
                              ",\"summary_index\":0,\"part\":{\"type\":\"summary_text\",\"text\":\"" + esc_r + "\"}");
            send_resp_sse("response.output_item.done",
                          "\"output_index\":" + std::to_string(resp_rs_idx) +
                              ",\"item\":{\"id\":\"" + rs_item_id +
                              "\",\"type\":\"reasoning\",\"status\":\"completed\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"" +
                              esc_r + "\"}]}");
        };

        auto open_resp_message = [&]() {
            if (resp_msg_open) return;
            close_resp_reasoning();
            resp_msg_open = true;
            resp_msg_idx = resp_next_out_idx++;
            send_resp_sse("response.output_item.added",
                          "\"output_index\":" + std::to_string(resp_msg_idx) +
                              ",\"item\":{\"id\":\"" + msg_item_id +
                              "\",\"type\":\"message\",\"status\":\"in_progress\",\"role\":\"assistant\",\"content\":[]}");
            send_resp_sse("response.content_part.added",
                          "\"item_id\":\"" + msg_item_id + "\",\"output_index\":" + std::to_string(resp_msg_idx) +
                              ",\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}");
        };

        auto close_resp_message = [&]() {
            if (!resp_msg_open || resp_msg_closed) return;
            resp_msg_closed = true;
            std::string esc_c = json_escape(parser.content_acc);
            send_resp_sse("response.output_text.done",
                          "\"item_id\":\"" + msg_item_id + "\",\"output_index\":" + std::to_string(resp_msg_idx) +
                              ",\"content_index\":0,\"text\":\"" + esc_c + "\"");
            send_resp_sse("response.content_part.done",
                          "\"item_id\":\"" + msg_item_id + "\",\"output_index\":" + std::to_string(resp_msg_idx) +
                              ",\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"" + esc_c +
                              "\",\"annotations\":[]}");
            send_resp_sse("response.output_item.done",
                          "\"output_index\":" + std::to_string(resp_msg_idx) +
                              ",\"item\":{\"id\":\"" + msg_item_id +
                              "\",\"type\":\"message\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"" +
                              esc_c + "\",\"annotations\":[]}]}");
        };

        if (stream) {
            std::string sse_hdr =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Cache-Control: no-cache\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n\r\n";
            client_ok = send_str(client_fd, sse_hdr);
            if (client_ok) {
                if (is_responses) {
                    std::string init_resp = build_resp_object("in_progress", "[]", "", "");
                    send_resp_sse("response.created", "\"response\":" + init_resp);
                    send_resp_sse("response.in_progress", "\"response\":" + init_resp);
                } else {
                    client_ok = send_str(client_fd, scb.make_chunk("{\"role\":\"assistant\",\"content\":\"\"}", nullptr));
                }
            }
        }

        auto emit_piece = [&](bool is_reasoning, const std::string& text) -> bool {
            if (!stream || text.empty()) return true;
            if (!client_ok) return false;
            if (is_responses) {
                if (is_reasoning) {
                    if (!resp_rs_open) {
                        resp_rs_open = true;
                        resp_rs_idx = resp_next_out_idx++;
                        send_resp_sse("response.output_item.added",
                                      "\"output_index\":" + std::to_string(resp_rs_idx) +
                                          ",\"item\":{\"id\":\"" + rs_item_id +
                                          "\",\"type\":\"reasoning\",\"status\":\"in_progress\",\"summary\":[]}");
                        send_resp_sse("response.reasoning_summary_part.added",
                                      "\"item_id\":\"" + rs_item_id + "\",\"output_index\":" + std::to_string(resp_rs_idx) +
                                          ",\"summary_index\":0,\"part\":{\"type\":\"summary_text\",\"text\":\"\"}");
                    }
                    return send_resp_sse("response.reasoning_summary_text.delta",
                                         "\"item_id\":\"" + rs_item_id + "\",\"output_index\":" + std::to_string(resp_rs_idx) +
                                             ",\"summary_index\":0,\"delta\":\"" + json_escape(text) + "\"");
                }
                open_resp_message();
                return send_resp_sse("response.output_text.delta",
                                     "\"item_id\":\"" + msg_item_id + "\",\"output_index\":" + std::to_string(resp_msg_idx) +
                                         ",\"content_index\":0,\"delta\":\"" + json_escape(text) + "\"");
            }
            std::string delta = is_reasoning
                                    ? ("{\"reasoning_content\":\"" + json_escape(text) + "\"}")
                                    : ("{\"content\":\"" + json_escape(text) + "\"}");
            client_ok = send_str(client_fd, scb.make_chunk(delta, nullptr));
            return client_ok;
        };

        auto emit_tool_call = [&](const ChatToolCall& tc, size_t idx) -> bool {
            if (!stream) return true;
            if (!client_ok) return false;
            if (is_responses) {
                close_resp_reasoning();
                close_resp_message();
                int fc_idx = resp_next_out_idx++;
                std::string fc_id = "fc_" + std::to_string(created_ts) + "_" + std::to_string(req_id) + "_" +
                                    std::to_string(idx);
                std::string esc_name = json_escape(tc.name);
                std::string esc_args = json_escape(tc.arguments_json);
                send_resp_sse("response.output_item.added",
                              "\"output_index\":" + std::to_string(fc_idx) +
                                  ",\"item\":{\"id\":\"" + fc_id +
                                  "\",\"type\":\"function_call\",\"status\":\"in_progress\",\"call_id\":\"" +
                                  json_escape(tc.id) + "\",\"name\":\"" + esc_name + "\",\"arguments\":\"\"}");
                send_resp_sse("response.function_call_arguments.delta",
                              "\"item_id\":\"" + fc_id + "\",\"output_index\":" + std::to_string(fc_idx) +
                                  ",\"delta\":\"" + esc_args + "\"");
                send_resp_sse("response.function_call_arguments.done",
                              "\"item_id\":\"" + fc_id + "\",\"output_index\":" + std::to_string(fc_idx) +
                                  ",\"name\":\"" + esc_name + "\",\"arguments\":\"" + esc_args + "\"");
                return send_resp_sse("response.output_item.done",
                                     "\"output_index\":" + std::to_string(fc_idx) +
                                         ",\"item\":{\"id\":\"" + fc_id +
                                         "\",\"type\":\"function_call\",\"status\":\"completed\",\"call_id\":\"" +
                                         json_escape(tc.id) + "\",\"name\":\"" + esc_name + "\",\"arguments\":\"" +
                                         esc_args + "\"}");
            }
            std::ostringstream doss;
            doss << "{\"tool_calls\":[{\"index\":" << idx
                 << ",\"id\":\"" << json_escape(tc.id)
                 << "\",\"type\":\"function\",\"function\":{\"name\":\"" << json_escape(tc.name)
                 << "\",\"arguments\":\"" << json_escape(tc.arguments_json) << "\"}}]}";
            client_ok = send_str(client_fd, scb.make_chunk(doss.str(), nullptr));
            return client_ok;
        };

        auto on_token = [&](int32_t tok_id) -> bool {
            return parser.feed(engine.tokenizer, tok_id, emit_piece, emit_tool_call);
        };

        GenStats stats;
        std::vector<int32_t> out_ids;
        if (client_ok) {
            if (run_spec) {
                out_ids = engine.generate_speculative(seq, prompt_ids, clamped_max_tokens,
                                                      cfg.draft_depth, stats, false, on_token,
                                                      true, sys_prefix_len);
            } else {
                out_ids = engine.generate_plain(seq, prompt_ids, clamped_max_tokens,
                                                stats, false, sampling, on_token,
                                                true, sys_prefix_len);
            }
        }

        // Flush any remaining held-back bytes if generation hit max_tokens or EOS
        if (client_ok) {
            parser.flush_all(emit_piece, emit_tool_call);
        }

        bool hit_eos = (!out_ids.empty() && out_ids.back() == engine.tokenizer.eos_id());
        bool has_tools = !parser.tool_calls.empty();
        const char* finish_reason = has_tools ? "tool_calls" : ((hit_eos || parser.hit_stop) ? "stop" : "length");
        int prompt_toks = stats.prompt_tokens;
        int comp_toks = stats.gen_tokens;

        if (is_responses) {
            std::ostringstream out_arr;
            out_arr << "[";
            bool first_item = true;
            if (!parser.reasoning_acc.empty()) {
                out_arr << "{\"id\":\"" << rs_item_id
                        << "\",\"type\":\"reasoning\",\"status\":\"completed\",\"summary\":[{\"type\":\"summary_text\",\"text\":\""
                        << json_escape(parser.reasoning_acc) << "\"}]}";
                first_item = false;
            }
            if (!parser.content_acc.empty() || !has_tools) {
                if (!first_item) out_arr << ",";
                out_arr << "{\"id\":\"" << msg_item_id
                        << "\",\"type\":\"message\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\""
                        << json_escape(parser.content_acc) << "\",\"annotations\":[]}]}";
                first_item = false;
            }
            for (size_t i = 0; i < parser.tool_calls.size(); ++i) {
                const auto& tc = parser.tool_calls[i];
                if (!first_item) out_arr << ",";
                std::string fc_id = "fc_" + std::to_string(created_ts) + "_" + std::to_string(req_id) + "_" +
                                    std::to_string(i);
                out_arr << "{\"id\":\"" << fc_id
                        << "\",\"type\":\"function_call\",\"status\":\"completed\",\"call_id\":\""
                        << json_escape(tc.id) << "\",\"name\":\"" << json_escape(tc.name)
                        << "\",\"arguments\":\"" << json_escape(tc.arguments_json) << "\"}";
                first_item = false;
            }
            out_arr << "]";

            std::ostringstream usage_oss;
            usage_oss << "{\"input_tokens\":" << prompt_toks
                      << ",\"output_tokens\":" << comp_toks
                      << ",\"total_tokens\":" << (prompt_toks + comp_toks)
                      << ",\"input_tokens_details\":{\"cached_tokens\":" << stats.cached_tokens << "}"
                      << ",\"output_tokens_details\":{\"reasoning_tokens\":0}}";

            bool completed = (hit_eos || parser.hit_stop || has_tools);
            const char* resp_status = completed ? "completed" : "incomplete";
            std::string final_resp = build_resp_object(resp_status, out_arr.str(), parser.content_acc, usage_oss.str());

            if (stream) {
                if (client_ok) {
                    close_resp_reasoning();
                    if (!has_tools && !resp_msg_open) open_resp_message();
                    close_resp_message();
                    send_resp_sse(completed ? "response.completed" : "response.incomplete",
                                  "\"response\":" + final_resp);
                }
            } else {
                send_http_json(client_fd, 200, "OK", final_resp);
            }
        } else if (stream) {
            if (client_ok) {
                send_str(client_fd, scb.make_chunk("{}", finish_reason));
                if (include_usage) {
                    std::ostringstream uoss;
                    uoss << "{\"prompt_tokens\":" << prompt_toks
                         << ",\"completion_tokens\":" << comp_toks
                         << ",\"total_tokens\":" << (prompt_toks + comp_toks)
                         << ",\"prompt_tokens_details\":{\"cached_tokens\":" << stats.cached_tokens << "}}";
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
                 << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":";
            if (parser.content_acc.empty() && has_tools) {
                body << "null";
            } else {
                body << "\"" << json_escape(parser.content_acc) << "\"";
            }
            if (!parser.reasoning_acc.empty()) {
                body << ",\"reasoning_content\":\"" << json_escape(parser.reasoning_acc) << "\"";
            }
            if (has_tools) {
                body << ",\"tool_calls\":[";
                for (size_t i = 0; i < parser.tool_calls.size(); ++i) {
                    const auto& tc = parser.tool_calls[i];
                    if (i > 0) body << ",";
                    body << "{\"id\":\"" << json_escape(tc.id)
                         << "\",\"type\":\"function\",\"function\":{\"name\":\"" << json_escape(tc.name)
                         << "\",\"arguments\":\"" << json_escape(tc.arguments_json) << "\"}}";
                }
                body << "]";
            }
            body << "},\"logprobs\":null,\"finish_reason\":\"" << finish_reason << "\"}]"
                 << ",\"usage\":{\"prompt_tokens\":" << prompt_toks
                 << ",\"completion_tokens\":" << comp_toks
                 << ",\"total_tokens\":" << (prompt_toks + comp_toks)
                 << ",\"prompt_tokens_details\":{\"cached_tokens\":" << stats.cached_tokens << "}}}";
            send_http_json(client_fd, 200, "OK", body.str());
        }

        int dec_toks = std::max(0, stats.gen_tokens - 1);
        double dec_tps = stats.decode_ms > 0.0 ? dec_toks / (stats.decode_ms / 1000.0) : 0.0;
        fprintf(stderr,
                "[req #%llu] %s stream=%d spec=%d prompt=%d (cached=%d, prefill=%.0fms) gen=%d (%.1fms, %.2f tok/s) finish=%s\n",
                static_cast<unsigned long long>(req_id), req.path.c_str(), stream ? 1 : 0,
                run_spec ? 1 : 0, prompt_toks, stats.cached_tokens, stats.prefill_ms, comp_toks,
                stats.decode_ms, dec_tps, finish_reason);
        } // @autoreleasepool
    };

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

        active_workers.fetch_add(1);
        std::thread([&, client_fd]() { handle_client(client_fd); }).detach();
    }

    close(server_fd);
    while (active_workers.load() > 0) {
        usleep(10000);
    }
    fprintf(stderr, "[server] Shutdown cleanly.\n");
    return 0;
}
