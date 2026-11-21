#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ChatMessage {
    std::string role;
    std::string content;
    std::string reasoning_content;
};

class InklingTokenizer {
public:
    bool load_from_json_file(const std::string& path);

    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& ids) const;
    std::string decode_token(int32_t id) const;

    // Render standard Inkling 2 prompt for single user prompt or multi-turn messages
    static std::string render_prompt(const std::string& user_prompt, const std::string& think_effort = "0");
    static std::string render_messages(const std::vector<ChatMessage>& messages,
                                       const std::string& think_effort = "0");

    int32_t eos_id() const { return 200006; } // <|content_model_end_sampling|>
    int32_t end_message_id() const { return end_message_id_; }
    bool is_special(int32_t id) const {
        return id >= 0 && static_cast<size_t>(id) < is_special_id_.size() && is_special_id_[id];
    }
    int32_t token_id(const std::string& tok) const {
        auto it = token_to_id_.find(tok);
        return it == token_to_id_.end() ? -1 : it->second;
    }
    size_t vocab_size() const { return id_to_token_.size(); }

private:
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::vector<bool> is_special_id_;
    std::unordered_map<uint64_t, uint32_t> merge_rank_; // (left_id << 32 | right_id) -> rank
    std::vector<std::pair<std::string, int32_t>> special_tokens_; // sorted longest-first
    std::string byte_to_unicode_[256];
    std::unordered_map<uint32_t, uint8_t> unicode_to_byte_;
    int32_t end_message_id_ = -1;

    void init_byte_maps();
    void encode_ordinary(const std::string& text, std::vector<int32_t>& out) const;
    void bpe_piece(const std::string& byte_level_piece, std::vector<int32_t>& out) const;
};
