#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace json_lite {

struct Value;
using Object = std::unordered_map<std::string, Value>;
using Array = std::vector<Value>;

struct Value {
    using Storage = std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;
    Storage data;

    Value() : data(nullptr) {}
    Value(std::nullptr_t) : data(nullptr) {}
    Value(bool b) : data(b) {}
    Value(double d) : data(d) {}
    Value(std::string s) : data(std::move(s)) {}
    Value(Array a) : data(std::move(a)) {}
    Value(Object o) : data(std::move(o)) {}

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(data); }
    bool is_bool() const { return std::holds_alternative<bool>(data); }
    bool is_number() const { return std::holds_alternative<double>(data); }
    bool is_string() const { return std::holds_alternative<std::string>(data); }
    bool is_array() const { return std::holds_alternative<Array>(data); }
    bool is_object() const { return std::holds_alternative<Object>(data); }

    bool as_bool() const { return std::get<bool>(data); }
    double as_double() const { return std::get<double>(data); }
    int64_t as_i64() const { return static_cast<int64_t>(std::get<double>(data)); }
    int as_int() const { return static_cast<int>(std::get<double>(data)); }
    float as_float() const { return static_cast<float>(std::get<double>(data)); }
    const std::string& as_string() const { return std::get<std::string>(data); }
    const Array& as_array() const { return std::get<Array>(data); }
    const Object& as_object() const { return std::get<Object>(data); }

    const Value* get(const std::string& k) const {
        if (!is_object()) return nullptr;
        const auto& obj = as_object();
        auto it = obj.find(k);
        return it == obj.end() ? nullptr : &it->second;
    }

    const Value& at(const std::string& k) const {
        const Value* v = get(k);
        if (!v) throw std::runtime_error("JSON missing key: " + k);
        return *v;
    }
};

class Parser {
public:
    explicit Parser(std::string_view src) : s_(src), p_(0) {}

    Value parse() {
        skip_ws();
        Value v = parse_value();
        skip_ws();
        return v;
    }

private:
    std::string_view s_;
    size_t p_;

    void skip_ws() {
        while (p_ < s_.size()) {
            char c = s_[p_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') ++p_;
            else break;
        }
    }

    char peek() const {
        if (p_ >= s_.size()) throw std::runtime_error("Unexpected EOF in JSON");
        return s_[p_];
    }

    void expect(char c) {
        if (peek() != c) throw std::runtime_error(std::string("JSON expected '") + c + "'");
        ++p_;
    }

    Value parse_value() {
        skip_ws();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return Value(parse_string());
        if (c == 't') { p_ += 4; return Value(true); }
        if (c == 'f') { p_ += 5; return Value(false); }
        if (c == 'n') { p_ += 4; return Value(nullptr); }
        return parse_number();
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp <= 0x7F) {
            out.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    uint32_t parse_hex4() {
        if (p_ + 4 > s_.size()) throw std::runtime_error("JSON bad \\u escape");
        uint32_t val = 0;
        for (int i = 0; i < 4; ++i) {
            char c = s_[p_++];
            val <<= 4;
            if (c >= '0' && c <= '9') val |= (c - '0');
            else if (c >= 'a' && c <= 'f') val |= (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') val |= (c - 'A' + 10);
            else throw std::runtime_error("JSON bad hex char");
        }
        return val;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (p_ < s_.size()) {
            size_t start = p_;
            while (p_ < s_.size() && s_[p_] != '"' && s_[p_] != '\\') ++p_;
            if (p_ > start) out.append(s_.data() + start, p_ - start);
            if (p_ >= s_.size()) throw std::runtime_error("Unterminated JSON string");
            char c = s_[p_++];
            if (c == '"') return out;
            if (p_ >= s_.size()) throw std::runtime_error("Bad JSON escape");
            char e = s_[p_++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = parse_hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF && p_ + 2 <= s_.size() && s_[p_] == '\\' && s_[p_ + 1] == 'u') {
                        p_ += 2;
                        uint32_t lo = parse_hex4();
                        cp = 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: out.push_back(e); break;
            }
        }
        throw std::runtime_error("Unterminated JSON string");
    }

    Value parse_number() {
        size_t start = p_;
        if (s_[p_] == '-') ++p_;
        while (p_ < s_.size() && ((s_[p_] >= '0' && s_[p_] <= '9') || s_[p_] == '.' || s_[p_] == 'e' || s_[p_] == 'E' || s_[p_] == '+' || s_[p_] == '-')) {
            ++p_;
        }
        std::string tmp(s_.data() + start, p_ - start);
        return Value(std::strtod(tmp.c_str(), nullptr));
    }

    Value parse_array() {
        expect('[');
        Array arr;
        skip_ws();
        if (peek() == ']') { ++p_; return Value(std::move(arr)); }
        while (true) {
            arr.push_back(parse_value());
            skip_ws();
            char c = peek();
            if (c == ']') { ++p_; break; }
            expect(',');
        }
        return Value(std::move(arr));
    }

    Value parse_object() {
        expect('{');
        Object obj;
        skip_ws();
        if (peek() == '}') { ++p_; return Value(std::move(obj)); }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            obj.emplace(std::move(key), parse_value());
            skip_ws();
            char c = peek();
            if (c == '}') { ++p_; break; }
            expect(',');
        }
        return Value(std::move(obj));
    }
};

inline Value parse(std::string_view src) { return Parser(src).parse(); }

inline std::string escape_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
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

inline void dump_compact_into(const Value& v, std::string& out) {
    if (v.is_null()) {
        out += "null";
    } else if (v.is_bool()) {
        out += v.as_bool() ? "true" : "false";
    } else if (v.is_number()) {
        double d = v.as_double();
        if (!std::isfinite(d)) {
            out += "null";
        } else {
            double intpart = 0.0;
            if (std::modf(d, &intpart) == 0.0 && d >= -9007199254740992.0 && d <= 9007199254740992.0) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(d));
                out += buf;
            } else {
                char buf[64];
                snprintf(buf, sizeof(buf), "%.17g", d);
                out += buf;
            }
        }
    } else if (v.is_string()) {
        out.push_back('"');
        out += escape_string(v.as_string());
        out.push_back('"');
    } else if (v.is_array()) {
        out.push_back('[');
        const auto& arr = v.as_array();
        for (size_t i = 0; i < arr.size(); ++i) {
            if (i > 0) out.push_back(',');
            dump_compact_into(arr[i], out);
        }
        out.push_back(']');
    } else if (v.is_object()) {
        out.push_back('{');
        const auto& obj = v.as_object();
        std::vector<const std::string*> keys;
        keys.reserve(obj.size());
        for (const auto& kv : obj) keys.push_back(&kv.first);
        std::sort(keys.begin(), keys.end(),
                  [](const std::string* a, const std::string* b) { return *a < *b; });
        for (size_t i = 0; i < keys.size(); ++i) {
            if (i > 0) out.push_back(',');
            out.push_back('"');
            out += escape_string(*keys[i]);
            out += "\":";
            dump_compact_into(obj.at(*keys[i]), out);
        }
        out.push_back('}');
    }
}

inline std::string dump_compact(const Value& v) {
    std::string out;
    dump_compact_into(v, out);
    return out;
}

} // namespace json_lite
