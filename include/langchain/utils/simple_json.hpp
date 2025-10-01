#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <variant>
#include <sstream>
#include <regex>

namespace langchain::utils {

// Forward declaration for recursive types
class SimpleJson;

/**
 * @brief Simple JSON value type
 */
using JsonValue = std::variant<
    std::nullptr_t,
    bool,
    int,
    double,
    std::string,
    std::vector<SimpleJson>,
    std::unordered_map<std::string, SimpleJson>
>;

/**
 * @brief Simple JSON object
 */
class SimpleJson {
private:
    JsonValue value_;

    // Helper functions for parsing
    static std::string trim(const std::string& str) {
        static const std::regex whitespace("^\\s+|\\s+$");
        return std::regex_replace(str, whitespace, "");
    }

    static JsonValue parse_value(const std::string& json, size_t& pos);
    static std::unordered_map<std::string, JsonValue> parse_object(const std::string& json, size_t& pos);
    static std::vector<JsonValue> parse_array(const std::string& json, size_t& pos);
    static std::string parse_string(const std::string& json, size_t& pos);
    static JsonValue parse_number(const std::string& json, size_t& pos);
    static JsonValue parse_literal(const std::string& json, size_t& pos);

    static void serialize_value(const JsonValue& value, std::ostringstream& oss);
    static void serialize_object(const std::unordered_map<std::string, JsonValue>& obj, std::ostringstream& oss);
    static void serialize_array(const std::vector<JsonValue>& arr, std::ostringstream& oss);

public:
    SimpleJson() : value_(nullptr) {}
    SimpleJson(const JsonValue& value) : value_(value) {}
    SimpleJson(const std::string& json_str) { value_ = parse(json_str); }

    // Construction helpers
    static SimpleJson object() { return SimpleJson(std::unordered_map<std::string, JsonValue>{}); }
    static SimpleJson array() { return SimpleJson(std::vector<JsonValue>{}); }
    static SimpleJson string(const std::string& str) { return SimpleJson(str); }
    static SimpleJson number(double num) { return SimpleJson(num); }
    static SimpleJson boolean(bool val) { return SimpleJson(val); }
    static SimpleJson null() { return SimpleJson(nullptr); }

    // Parse from string
    static JsonValue parse(const std::string& json_str) {
        size_t pos = 0;
        auto result = parse_value(json_str, pos);
        // Skip trailing whitespace
        while (pos < json_str.size() && std::isspace(json_str[pos])) {
            pos++;
        }
        if (pos != json_str.size()) {
            throw std::runtime_error("Unexpected characters after JSON value");
        }
        return result;
    }

    // Serialize to string
    std::string dump() const {
        std::ostringstream oss;
        serialize_value(value_, oss);
        return oss.str();
    }

    // Accessors
    bool is_null() const { return std::holds_alternative<std::nullptr_t>(value_); }
    bool is_bool() const { return std::holds_alternative<bool>(value_); }
    bool is_number() const { return std::holds_alternative<int>(value_) || std::holds_alternative<double>(value_); }
    bool is_string() const { return std::holds_alternative<std::string>(value_); }
    bool is_array() const { return std::holds_alternative<std::vector<JsonValue>>(value_); }
    bool is_object() const { return std::holds_alternative<std::unordered_map<std::string, JsonValue>>(value_); }

    // Getters
    bool get_bool() const { return std::get<bool>(value_); }
    int get_int() const { return std::get<int>(value_); }
    double get_double() const {
        if (std::holds_alternative<int>(value_)) {
            return static_cast<double>(std::get<int>(value_));
        }
        return std::get<double>(value_);
    }
    std::string get_string() const { return std::get<std::string>(value_); }
    std::vector<JsonValue> get_array() const { return std::get<std::vector<JsonValue>>(value_); }
    std::unordered_map<std::string, JsonValue> get_object() const { return std::get<std::unordered_map<std::string, JsonValue>>(value_); }

    // Object access
    bool contains(const std::string& key) const {
        if (!is_object()) return false;
        const auto& obj = std::get<std::unordered_map<std::string, JsonValue>>(value_);
        return obj.find(key) != obj.end();
    }

    JsonValue& operator[](const std::string& key) {
        if (!is_object()) {
            value_ = std::unordered_map<std::string, JsonValue>{};
        }
        return std::get<std::unordered_map<std::string, JsonValue>>(value_)[key];
    }

    const JsonValue& operator[](const std::string& key) const {
        static JsonValue null_value = nullptr;
        if (!is_object()) return null_value;
        const auto& obj = std::get<std::unordered_map<std::string, JsonValue>>(value_);
        auto it = obj.find(key);
        return (it != obj.end()) ? it->second : null_value;
    }

    // Array access
    JsonValue& operator[](size_t index) {
        if (!is_array()) {
            throw std::runtime_error("Not an array");
        }
        return std::get<std::vector<JsonValue>>(value_)[index];
    }

    const JsonValue& operator[](size_t index) const {
        if (!is_array()) {
            throw std::runtime_error("Not an array");
        }
        return std::get<std::vector<JsonValue>>(value_)[index];
    }

    // Get underlying value
    const JsonValue& value() const { return value_; }
    JsonValue& value() { return value_; }
};

// Implementation of parsing functions
inline JsonValue SimpleJson::parse_value(const std::string& json, size_t& pos) {
    // Skip whitespace
    while (pos < json.size() && std::isspace(json[pos])) {
        pos++;
    }

    if (pos >= json.size()) {
        throw std::runtime_error("Unexpected end of JSON");
    }

    char c = json[pos];
    if (c == '{') {
        return parse_object(json, pos);
    } else if (c == '[') {
        return parse_array(json, pos);
    } else if (c == '"') {
        return parse_string(json, pos);
    } else if (c == '-' || (c >= '0' && c <= '9')) {
        return parse_number(json, pos);
    } else {
        return parse_literal(json, pos);
    }
}

inline std::unordered_map<std::string, JsonValue> SimpleJson::parse_object(const std::string& json, size_t& pos) {
    std::unordered_map<std::string, JsonValue> obj;
    pos++; // Skip '{'

    // Skip whitespace
    while (pos < json.size() && std::isspace(json[pos])) {
        pos++;
    }

    if (pos < json.size() && json[pos] == '}') {
        pos++; // Skip '}'
        return obj;
    }

    while (pos < json.size()) {
        // Parse key
        auto key = std::get<std::string>(parse_string(json, pos));

        // Skip whitespace and colon
        while (pos < json.size() && std::isspace(json[pos])) {
            pos++;
        }
        if (pos >= json.size() || json[pos] != ':') {
            throw std::runtime_error("Expected ':' in object");
        }
        pos++; // Skip ':'

        // Parse value
        auto value = parse_value(json, pos);

        obj[key] = value;

        // Skip whitespace
        while (pos < json.size() && std::isspace(json[pos])) {
            pos++;
        }

        if (pos >= json.size()) {
            throw std::runtime_error("Unexpected end in object");
        }

        if (json[pos] == '}') {
            pos++; // Skip '}'
            break;
        } else if (json[pos] == ',') {
            pos++; // Skip ','
        } else {
            throw std::runtime_error("Expected ',' or '}' in object");
        }
    }

    return obj;
}

inline std::vector<JsonValue> SimpleJson::parse_array(const std::string& json, size_t& pos) {
    std::vector<JsonValue> arr;
    pos++; // Skip '['

    // Skip whitespace
    while (pos < json.size() && std::isspace(json[pos])) {
        pos++;
    }

    if (pos < json.size() && json[pos] == ']') {
        pos++; // Skip ']'
        return arr;
    }

    while (pos < json.size()) {
        auto value = parse_value(json, pos);
        arr.push_back(value);

        // Skip whitespace
        while (pos < json.size() && std::isspace(json[pos])) {
            pos++;
        }

        if (pos >= json.size()) {
            throw std::runtime_error("Unexpected end in array");
        }

        if (json[pos] == ']') {
            pos++; // Skip ']'
            break;
        } else if (json[pos] == ',') {
            pos++; // Skip ','
        } else {
            throw std::runtime_error("Expected ',' or ']' in array");
        }
    }

    return arr;
}

inline std::string SimpleJson::parse_string(const std::string& json, size_t& pos) {
    if (pos >= json.size() || json[pos] != '"') {
        throw std::runtime_error("Expected '\"' for string");
    }
    pos++; // Skip '"'

    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\') {
            pos++; // Skip '\\'
            if (pos < json.size()) {
                char c = json[pos];
                switch (c) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    case 'u': {
                        // Unicode escape (simplified - just skip 4 hex digits)
                        if (pos + 4 < json.size()) {
                            result += json[pos + 1]; // Simplified: just take one char
                            pos += 4;
                        }
                        break;
                    }
                    default:
                        result += c;
                        break;
                }
                pos++;
            }
        } else {
            result += json[pos];
            pos++;
        }
    }

    if (pos >= json.size() || json[pos] != '"') {
        throw std::runtime_error("Unterminated string");
    }
    pos++; // Skip closing '"'

    return result;
}

inline JsonValue SimpleJson::parse_number(const std::string& json, size_t& pos) {
    size_t start = pos;
    if (pos < json.size() && json[pos] == '-') {
        pos++;
    }

    while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
        pos++;
    }

    if (pos < json.size() && json[pos] == '.') {
        pos++;
        while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
            pos++;
        }
        return std::stod(json.substr(start, pos - start));
    } else {
        return std::stoi(json.substr(start, pos - start));
    }
}

inline JsonValue SimpleJson::parse_literal(const std::string& json, size_t& pos) {
    if (pos + 4 <= json.size() && json.substr(pos, 4) == "true") {
        pos += 4;
        return true;
    } else if (pos + 5 <= json.size() && json.substr(pos, 5) == "false") {
        pos += 5;
        return false;
    } else if (pos + 4 <= json.size() && json.substr(pos, 4) == "null") {
        pos += 4;
        return nullptr;
    } else {
        throw std::runtime_error("Invalid JSON literal");
    }
}

inline void SimpleJson::serialize_value(const JsonValue& value, std::ostringstream& oss) {
    if (std::holds_alternative<std::nullptr_t>(value)) {
        oss << "null";
    } else if (std::holds_alternative<bool>(value)) {
        oss << (std::get<bool>(value) ? "true" : "false");
    } else if (std::holds_alternative<int>(value)) {
        oss << std::get<int>(value);
    } else if (std::holds_alternative<double>(value)) {
        oss << std::get<double>(value);
    } else if (std::holds_alternative<std::string>(value)) {
        oss << '"' << std::get<std::string>(value) << '"';
    } else if (std::holds_alternative<std::vector<JsonValue>>(value)) {
        serialize_array(std::get<std::vector<JsonValue>>(value), oss);
    } else if (std::holds_alternative<std::unordered_map<std::string, JsonValue>>(value)) {
        serialize_object(std::get<std::unordered_map<std::string, JsonValue>>(value), oss);
    }
}

inline void SimpleJson::serialize_object(const std::unordered_map<std::string, JsonValue>& obj, std::ostringstream& oss) {
    oss << '{';
    bool first = true;
    for (const auto& [key, value] : obj) {
        if (!first) oss << ',';
        oss << '"' << key << "\":";
        serialize_value(value, oss);
        first = false;
    }
    oss << '}';
}

inline void SimpleJson::serialize_array(const std::vector<JsonValue>& arr, std::ostringstream& oss) {
    oss << '[';
    for (size_t i = 0; i < arr.size(); ++i) {
        if (i > 0) oss << ',';
        serialize_value(arr[i], oss);
    }
    oss << ']';
}

} // namespace langchain::utils