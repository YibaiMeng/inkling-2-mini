#import <Foundation/Foundation.h>
#include "tokenizer.h"
#include "json_lite.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <fstream>
#include <sstream>

static void append_utf8_cp(std::string& s, uint32_t cp) {
    if (cp <= 0x7F) {
        s.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

void InklingTokenizer::init_byte_maps() {
    bool is_direct[256] = {};
    for (int b = 33; b <= 126; ++b) is_direct[b] = true;
    for (int b = 161; b <= 172; ++b) is_direct[b] = true;
    for (int b = 174; b <= 255; ++b) is_direct[b] = true;
    uint32_t extra = 256;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = is_direct[b] ? static_cast<uint32_t>(b) : extra++;
        byte_to_unicode_[b].clear();
        append_utf8_cp(byte_to_unicode_[b], cp);
        unicode_to_byte_[cp] = static_cast<uint8_t>(b);
    }
}

bool InklingTokenizer::load_from_json_file(const std::string& path) {
    init_byte_maps();
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    std::ostringstream oss;
    oss << ifs.rdbuf();
    std::string raw = oss.str();

    // Fast targeted parse of tokenizer.json (avoids building huge variant trees for 424k merges)
    json_lite::Value root = json_lite::parse(raw);

    const auto& model = root.at("model");
    const auto& vocab = model.at("vocab").as_object();
    const auto& added = root.at("added_tokens").as_array();

    size_t max_id = 0;
    for (const auto& kv : vocab) {
        max_id = std::max<size_t>(max_id, static_cast<size_t>(kv.second.as_i64()));
    }
    for (const auto& t : added) {
        max_id = std::max<size_t>(max_id, static_cast<size_t>(t.at("id").as_i64()));
    }

    id_to_token_.assign(max_id + 1, "");
    is_special_id_.assign(max_id + 1, false);
    token_to_id_.reserve(max_id + 16);

    for (const auto& kv : vocab) {
        int32_t id = kv.second.as_int();
        id_to_token_[id] = kv.first;
        token_to_id_[kv.first] = id;
    }

    special_tokens_.clear();
    for (const auto& t : added) {
        int32_t id = t.at("id").as_int();
        const std::string& content = t.at("content").as_string();
        id_to_token_[id] = content;
        token_to_id_[content] = id;
        is_special_id_[id] = true;
        special_tokens_.emplace_back(content, id);
        if (content == "<|end_message|>") end_message_id_ = id;
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    const auto& merges = model.at("merges").as_array();
    merge_rank_.reserve(merges.size() * 2);
    for (uint32_t rank = 0; rank < merges.size(); ++rank) {
        std::string a, b;
        if (merges[rank].is_string()) {
            const std::string& m = merges[rank].as_string();
            size_t sp = m.find(' ');
            if (sp == std::string::npos) continue;
            a = m.substr(0, sp);
            b = m.substr(sp + 1);
        } else if (merges[rank].is_array()) {
            const auto& arr = merges[rank].as_array();
            if (arr.size() != 2) continue;
            a = arr[0].as_string();
            b = arr[1].as_string();
        }
        auto ia = token_to_id_.find(a);
        auto ib = token_to_id_.find(b);
        if (ia != token_to_id_.end() && ib != token_to_id_.end()) {
            uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(ia->second)) << 32) |
                           static_cast<uint32_t>(ib->second);
            merge_rank_[key] = rank;
        }
    }
    return true;
}

static NSRegularExpression* get_pretok_regex() {
    static NSRegularExpression* rx = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSString* pat = @"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
        NSError* err = nil;
        rx = [NSRegularExpression regularExpressionWithPattern:pat options:0 error:&err];
        if (!rx) {
            fprintf(stderr, "Regex compile error: %s\n", err.localizedDescription.UTF8String);
        }
    });
    return rx;
}

void InklingTokenizer::bpe_piece(const std::string& raw_bytes, std::vector<int32_t>& out) const {
    if (raw_bytes.empty()) return;

    // Fast path: whole byte-level string is a single vocab token
    std::string whole;
    whole.reserve(raw_bytes.size() * 2);
    for (unsigned char c : raw_bytes) whole += byte_to_unicode_[c];
    auto it_whole = token_to_id_.find(whole);
    if (it_whole != token_to_id_.end()) {
        out.push_back(it_whole->second);
        return;
    }

    // Initialize symbols as individual byte-level tokens
    std::vector<int32_t> ids;
    ids.reserve(raw_bytes.size());
    for (unsigned char c : raw_bytes) {
        auto it = token_to_id_.find(byte_to_unicode_[c]);
        if (it != token_to_id_.end()) ids.push_back(it->second);
    }

    while (ids.size() >= 2) {
        uint32_t best_rank = UINT32_MAX;
        size_t best_pos = SIZE_MAX;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(ids[i])) << 32) |
                           static_cast<uint32_t>(ids[i + 1]);
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = i;
            }
        }
        if (best_pos == SIZE_MAX) break;
        std::string merged = id_to_token_[ids[best_pos]] + id_to_token_[ids[best_pos + 1]];
        auto it_m = token_to_id_.find(merged);
        if (it_m == token_to_id_.end()) break;
        int32_t merged_id = it_m->second;
        int32_t left = ids[best_pos], right = ids[best_pos + 1];
        size_t w = 0;
        for (size_t i = 0; i < ids.size();) {
            if (i + 1 < ids.size() && ids[i] == left && ids[i + 1] == right) {
                ids[w++] = merged_id;
                i += 2;
            } else {
                ids[w++] = ids[i++];
            }
        }
        ids.resize(w);
    }
    for (int32_t id : ids) out.push_back(id);
}

void InklingTokenizer::encode_ordinary(const std::string& text, std::vector<int32_t>& out) const {
    if (text.empty()) return;
    @autoreleasepool {
        NSString* ns = [[NSString alloc] initWithBytes:text.data() length:text.size() encoding:NSUTF8StringEncoding];
        if (!ns) {
            bpe_piece(text, out);
            return;
        }
        NSRegularExpression* rx = get_pretok_regex();
        NSArray<NSTextCheckingResult*>* matches = [rx matchesInString:ns options:0 range:NSMakeRange(0, ns.length)];
        for (NSTextCheckingResult* m in matches) {
            NSString* sub = [ns substringWithRange:m.range];
            const char* utf8 = sub.UTF8String;
            if (utf8) bpe_piece(std::string(utf8), out);
        }
    }
}

std::vector<int32_t> InklingTokenizer::encode(const std::string& text) const {
    std::vector<int32_t> out;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t next_special = std::string::npos;
        int32_t sp_id = -1;
        size_t sp_len = 0;
        for (const auto& sp : special_tokens_) {
            size_t f = text.find(sp.first, pos);
            if (f != std::string::npos && (f < next_special || (f == next_special && sp.first.size() > sp_len))) {
                next_special = f;
                sp_id = sp.second;
                sp_len = sp.first.size();
            }
        }
        if (next_special == std::string::npos) {
            encode_ordinary(text.substr(pos), out);
            break;
        }
        if (next_special > pos) {
            encode_ordinary(text.substr(pos, next_special - pos), out);
        }
        out.push_back(sp_id);
        pos = next_special + sp_len;
    }
    return out;
}

std::string InklingTokenizer::decode_token(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return "";
    if (is_special_id_[id]) return id_to_token_[id];
    const std::string& s = id_to_token_[id];
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        uint8_t c = static_cast<uint8_t>(s[i]);
        uint32_t cp = 0;
        if ((c & 0x80) == 0) {
            cp = c; i += 1;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
            cp = ((c & 0x1F) << 6) | (static_cast<uint8_t>(s[i + 1]) & 0x3F); i += 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
            cp = ((c & 0x0F) << 12) | ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 6) |
                 (static_cast<uint8_t>(s[i + 2]) & 0x3F); i += 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
            cp = ((c & 0x07) << 18) | ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 12) |
                 ((static_cast<uint8_t>(s[i + 2]) & 0x3F) << 6) | (static_cast<uint8_t>(s[i + 3]) & 0x3F); i += 4;
        } else {
            out.push_back(static_cast<char>(c)); i += 1; continue;
        }
        auto it = unicode_to_byte_.find(cp);
        if (it != unicode_to_byte_.end()) out.push_back(static_cast<char>(it->second));
    }
    return out;
}

std::string InklingTokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) out += decode_token(id);
    return out;
}

std::string InklingTokenizer::render_prompt(const std::string& user_prompt, const std::string& think_effort) {
    ChatMessage m;
    m.role = "user";
    m.content = user_prompt;
    return render_messages({m}, think_effort);
}

std::string InklingTokenizer::render_messages(const std::vector<ChatMessage>& messages,
                                              const std::string& think_effort,
                                              const std::string& tools_declare_json) {
    // Matches ollama/model/renderers/inkling2.go Render()
    std::string out;
    if (!tools_declare_json.empty()) {
        out += "<|message_system|>tool_declare<|content_xml|>";
        out += tools_declare_json;
        out += "<|end_message|>";
    }

    // Map tool_call_id -> tool name for tool response messages that omit tool_name
    std::unordered_map<std::string, std::string> call_id_to_name;
    for (const auto& msg : messages) {
        if (msg.role == "assistant") {
            for (const auto& tc : msg.tool_calls) {
                if (!tc.id.empty() && !tc.name.empty()) {
                    call_id_to_name[tc.id] = tc.name;
                }
            }
        }
    }

    bool effort_emitted = false;
    auto write_effort = [&]() {
        out += "<|message_system|><|content_text|>Thinking effort level: ";
        out += think_effort;
        out += "<|end_message|>";
        effort_emitted = true;
    };

    for (const auto& msg : messages) {
        std::string role_tag;
        bool is_system = (msg.role == "system" || msg.role == "developer");
        if (is_system) {
            role_tag = "<|message_system|>";
        } else if (msg.role == "user") {
            role_tag = "<|message_user|>";
        } else if (msg.role == "assistant") {
            role_tag = "<|message_model|>";
        } else if (msg.role == "tool") {
            std::string tname = msg.tool_name;
            if (tname.empty() && !msg.tool_call_id.empty()) {
                auto it = call_id_to_name.find(msg.tool_call_id);
                if (it != call_id_to_name.end()) tname = it->second;
            }
            role_tag = "<|message_tool|>" + tname;
        } else {
            role_tag = "<|message_user|>";
        }

        if (!effort_emitted && !is_system) {
            write_effort();
        }
        if (msg.role == "assistant" && !msg.reasoning_content.empty()) {
            out += role_tag;
            out += "<|content_thinking|>";
            out += msg.reasoning_content;
            out += "<|end_message|>";
        }
        bool has_extra = (msg.role == "assistant" && (!msg.tool_calls.empty() || !msg.reasoning_content.empty()));
        if (!msg.content.empty() || !has_extra) {
            out += role_tag;
            out += "<|content_text|>";
            out += msg.content;
            out += "<|end_message|>";
        }
        if (msg.role == "assistant") {
            for (const auto& tc : msg.tool_calls) {
                std::string args = tc.arguments_json.empty() ? "{}" : tc.arguments_json;
                out += "<|message_model|>";
                out += tc.name;
                out += "<|content_invoke_tool_json|>{\"name\":\"";
                out += json_lite::escape_string(tc.name);
                out += "\",\"args\":";
                out += args;
                out += "}<|end_message|>";
            }
            out += "<|content_model_end_sampling|>";
        }
    }
    if (!effort_emitted) {
        write_effort();
    }
    return out;
}
