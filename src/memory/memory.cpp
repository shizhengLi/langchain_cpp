#include "langchain/memory/memory.hpp"
#include <sstream>
#include <algorithm>
#include <fstream>

namespace langchain::memory {

// BufferMemory implementation
BufferMemory::BufferMemory(const MemoryConfig& config) : BaseMemory() {
    config_ = config;
    config_.validate();
}

void BufferMemory::add_message(const ChatMessage& message) {
    messages_.push_back(message);
    enforce_limits();
}

std::string BufferMemory::get_context() const {
    std::stringstream ss;
    for (const auto& message : messages_) {
        ss << format_message(message) << "\n";
    }
    return ss.str();
}

std::vector<ChatMessage> BufferMemory::get_messages() const {
    return messages_;
}

void BufferMemory::clear() {
    messages_.clear();
}

std::unordered_map<std::string, size_t> BufferMemory::get_stats() const {
    return {
        {"message_count", messages_.size()},
        {"max_messages", config_.max_messages}
    };
}

void BufferMemory::enforce_limits() {
    // Enforce message count limit
    while (messages_.size() > config_.max_messages) {
        messages_.erase(messages_.begin());
    }

    // Enforce age limit
    auto now = std::chrono::system_clock::now();
    auto it = messages_.begin();
    while (it != messages_.end()) {
        auto age = std::chrono::duration_cast<std::chrono::seconds>(now - it->timestamp);
        if (age > config_.max_age) {
            it = messages_.erase(it);
        } else {
            break;
        }
    }
}

std::string BufferMemory::format_message(const ChatMessage& message) const {
    std::string type_str = utils::message_type_to_string(message.type);
    std::string result = "[" + type_str + "]: " + message.content;

    if (config_.include_timestamps) {
        auto time_t = std::chrono::system_clock::to_time_t(message.timestamp);
        result += " (at " + std::string(std::ctime(&time_t));
        result.pop_back(); // Remove newline
        result += ")";
    }

    return result;
}

// TokenBufferMemory implementation
TokenBufferMemory::TokenBufferMemory(
    size_t max_tokens,
    std::function<size_t(const std::string&)> token_counter,
    const MemoryConfig& config
) : BaseMemory(), token_counter_(std::move(token_counter)), max_tokens_(max_tokens) {
    config_ = config;
    config_.validate();
}

void TokenBufferMemory::add_message(const ChatMessage& message) {
    size_t message_tokens = count_tokens(message.content);
    current_tokens_ += message_tokens;
    messages_.push_back(message);
    enforce_token_limit();
}

std::string TokenBufferMemory::get_context() const {
    std::stringstream ss;
    for (const auto& message : messages_) {
        ss << format_message(message) << "\n";
    }
    return ss.str();
}

std::vector<ChatMessage> TokenBufferMemory::get_messages() const {
    return messages_;
}

void TokenBufferMemory::clear() {
    messages_.clear();
    current_tokens_ = 0;
}

std::unordered_map<std::string, size_t> TokenBufferMemory::get_stats() const {
    return {
        {"message_count", messages_.size()},
        {"current_tokens", current_tokens_},
        {"max_tokens", max_tokens_}
    };
}

size_t TokenBufferMemory::count_tokens(const std::string& text) const {
    if (token_counter_) {
        return token_counter_(text);
    }
    // Simple approximation: 1 token ≈ 4 characters
    return (text.length() + 3) / 4;
}

void TokenBufferMemory::enforce_token_limit() {
    while (current_tokens_ > max_tokens_ && !messages_.empty()) {
        size_t removed_tokens = count_tokens(messages_.front().content);
        messages_.erase(messages_.begin());
        current_tokens_ -= removed_tokens;
    }
}

std::string TokenBufferMemory::format_message(const ChatMessage& message) const {
    std::string type_str = utils::message_type_to_string(message.type);
    return "[" + type_str + "]: " + message.content;
}

// SummaryMemory implementation
SummaryMemory::SummaryMemory(
    std::function<std::string(const std::vector<ChatMessage>&)> summarizer,
    const MemoryConfig& config
) : BaseMemory(), summarizer_(std::move(summarizer)) {
    config_ = config;
    config_.validate();
}

void SummaryMemory::add_message(const ChatMessage& message) {
    recent_messages_.push_back(message);

    if (should_summarize()) {
        summarize_messages();
    }
}

std::string SummaryMemory::get_context() const {
    std::stringstream ss;
    if (!summary_.empty()) {
        ss << "Previous conversation summary:\n" << summary_ << "\n\n";
    }

    if (!recent_messages_.empty()) {
        ss << "Recent messages:\n";
        for (const auto& message : recent_messages_) {
            std::string type_str = utils::message_type_to_string(message.type);
            ss << "[" + type_str + "]: " << message.content << "\n";
        }
    }

    return ss.str();
}

std::vector<ChatMessage> SummaryMemory::get_messages() const {
    std::vector<ChatMessage> all_messages;

    // Add summary as a system message if it exists
    if (!summary_.empty()) {
        all_messages.emplace_back(ChatMessage(ChatMessage::Type::SYSTEM, "Summary: " + summary_));
    }

    // Add recent messages
    all_messages.insert(all_messages.end(), recent_messages_.begin(), recent_messages_.end());

    return all_messages;
}

void SummaryMemory::clear() {
    recent_messages_.clear();
    summary_.clear();
}

std::unordered_map<std::string, size_t> SummaryMemory::get_stats() const {
    return {
        {"recent_message_count", recent_messages_.size()},
        {"summary_length", summary_.length()},
        {"total_messages", recent_messages_.size() + (summary_.empty() ? 0 : 1)}
    };
}

bool SummaryMemory::should_summarize() const {
    return recent_messages_.size() >= summary_frequency_;
}

void SummaryMemory::summarize_messages() {
    if (summarizer_) {
        summary_ = summarizer_(recent_messages_);
    } else {
        summary_ = basic_summarizer(recent_messages_);
    }
    recent_messages_.clear();
}

std::string SummaryMemory::basic_summarizer(const std::vector<ChatMessage>& messages) const {
    std::stringstream ss;
    ss << "Conversation summary with " << messages.size() << " messages.";
    return ss.str();
}

// MemoryFactory implementation
std::unique_ptr<BaseMemory> MemoryFactory::create_memory(MemoryType type, const MemoryConfig& config) {
    switch (type) {
        case MemoryType::BUFFER:
            return std::make_unique<BufferMemory>(config);
        case MemoryType::TOKEN_BUFFER:
            return std::make_unique<TokenBufferMemory>();
        case MemoryType::SUMMARY:
            return std::make_unique<SummaryMemory>(nullptr, config);
        default:
            throw std::invalid_argument("Unknown memory type");
    }
}

std::unique_ptr<BaseMemory> MemoryFactory::create_buffer_memory(const MemoryConfig& config) {
    return std::make_unique<BufferMemory>(config);
}

std::unique_ptr<BaseMemory> MemoryFactory::create_token_buffer_memory(size_t max_tokens) {
    return std::make_unique<TokenBufferMemory>(max_tokens);
}

std::unique_ptr<BaseMemory> MemoryFactory::create_summary_memory(const MemoryConfig& config) {
    return std::make_unique<SummaryMemory>(nullptr, config);
}

std::unique_ptr<BaseMemory> MemoryFactory::create_knowledge_graph_memory(const MemoryConfig& config) {
    // Simplified implementation
    return std::make_unique<BufferMemory>(config);
}

// Utility functions implementation
namespace utils {

std::vector<ChatMessage> load_conversation(const std::string& file_path) {
    std::vector<ChatMessage> messages;
    // Simplified implementation - would normally parse from file
    return messages;
}

void save_conversation(const std::vector<ChatMessage>& messages, const std::string& file_path) {
    // Simplified implementation - would normally serialize to file
}

std::string message_type_to_string(ChatMessage::Type type) {
    switch (type) {
        case ChatMessage::Type::HUMAN: return "Human";
        case ChatMessage::Type::AI: return "AI";
        case ChatMessage::Type::SYSTEM: return "System";
        case ChatMessage::Type::FUNCTION: return "Function";
        case ChatMessage::Type::TOOL: return "Tool";
        default: return "Unknown";
    }
}

std::function<size_t(const std::string&)> create_simple_token_counter() {
    return [](const std::string& text) -> size_t {
        return (text.length() + 3) / 4; // Rough approximation
    };
}

std::function<std::string(const std::vector<ChatMessage>&)> create_basic_summarizer() {
    return [](const std::vector<ChatMessage>& messages) -> std::string {
        std::stringstream ss;
        ss << "Summary of " << messages.size() << " messages.";
        return ss.str();
    };
}

} // namespace utils

} // namespace langchain::memory