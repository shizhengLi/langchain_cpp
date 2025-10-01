#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <optional>
#include <functional>

namespace langchain::memory {

/**
 * @brief Represents a single message in conversation history
 */
struct ChatMessage {
    enum class Type {
        HUMAN,
        AI,
        SYSTEM,
        FUNCTION,
        TOOL
    };

    Type type;
    std::string content;
    std::chrono::system_clock::time_point timestamp;
    std::optional<std::string> additional_data;  // For function calls, tool results, etc.

    ChatMessage(Type t, const std::string& c)
        : type(t), content(c), timestamp(std::chrono::system_clock::now()) {}

    ChatMessage(Type t, const std::string& c, const std::string& data)
        : type(t), content(c), timestamp(std::chrono::system_clock::now()), additional_data(data) {}
};

/**
 * @brief Configuration for memory modules
 */
struct MemoryConfig {
    size_t max_messages{100};           // Maximum messages to keep in memory
    std::chrono::seconds max_age{3600}; // Maximum age of messages
    bool include_timestamps{false};     // Whether to include timestamps in context
    bool summarize_long_conversations{false}; // Whether to summarize old messages
    size_t summary_threshold{50};       // When to trigger summarization
    std::string system_message{""};     // System prompt to include

    void validate() const {
        if (max_messages == 0) {
            throw std::invalid_argument("max_messages must be greater than 0");
        }
        if (max_age.count() <= 0) {
            throw std::invalid_argument("max_age must be positive");
        }
    }
};

/**
 * @brief Base class for all memory implementations
 */
class BaseMemory {
public:
    virtual ~BaseMemory() = default;

    /**
     * @brief Add a message to memory
     */
    virtual void add_message(const ChatMessage& message) = 0;

    /**
     * @brief Get conversation context for prompt generation
     */
    virtual std::string get_context() const = 0;

    /**
     * @brief Get all messages in memory
     */
    virtual std::vector<ChatMessage> get_messages() const = 0;

    /**
     * @brief Clear all messages from memory
     */
    virtual void clear() = 0;

    /**
     * @brief Get memory statistics
     */
    virtual std::unordered_map<std::string, size_t> get_stats() const = 0;

protected:
    MemoryConfig config_;
};

/**
 * @brief Simple buffer memory that stores recent messages
 */
class BufferMemory : public BaseMemory {
private:
    std::vector<ChatMessage> messages_;

public:
    explicit BufferMemory(const MemoryConfig& config = MemoryConfig{});

    void add_message(const ChatMessage& message) override;
    std::string get_context() const override;
    std::vector<ChatMessage> get_messages() const override;
    void clear() override;
    std::unordered_map<std::string, size_t> get_stats() const override;

private:
    void enforce_limits();
    std::string format_message(const ChatMessage& message) const;
};

/**
 * @brief Token buffer memory that limits based on token count
 */
class TokenBufferMemory : public BaseMemory {
private:
    std::vector<ChatMessage> messages_;
    std::function<size_t(const std::string&)> token_counter_;
    size_t max_tokens_{2000};
    size_t current_tokens_{0};

public:
    explicit TokenBufferMemory(
        size_t max_tokens = 2000,
        std::function<size_t(const std::string&)> token_counter = nullptr,
        const MemoryConfig& config = MemoryConfig{}
    );

    void add_message(const ChatMessage& message) override;
    std::string get_context() const override;
    std::vector<ChatMessage> get_messages() const override;
    void clear() override;
    std::unordered_map<std::string, size_t> get_stats() const override;

    // Set custom token counting function
    void set_token_counter(std::function<size_t(const std::string&)> counter) {
        token_counter_ = std::move(counter);
    }

private:
    size_t count_tokens(const std::string& text) const;
    void enforce_token_limit();
    std::string format_message(const ChatMessage& message) const;
};

/**
 * @brief Summary memory that summarizes old conversations
 */
class SummaryMemory : public BaseMemory {
private:
    std::vector<ChatMessage> recent_messages_;
    std::string summary_;
    size_t summary_frequency_{10};  // Summarize every N messages
    std::function<std::string(const std::vector<ChatMessage>&)> summarizer_;

public:
    explicit SummaryMemory(
        std::function<std::string(const std::vector<ChatMessage>&)> summarizer = nullptr,
        const MemoryConfig& config = MemoryConfig{}
    );

    void add_message(const ChatMessage& message) override;
    std::string get_context() const override;
    std::vector<ChatMessage> get_messages() const override;
    void clear() override;
    std::unordered_map<std::string, size_t> get_stats() const override;

    void set_summarizer(std::function<std::string(const std::vector<ChatMessage>&)> summarizer) {
        summarizer_ = std::move(summarizer);
    }

private:
    bool should_summarize() const;
    void summarize_messages();
    std::string basic_summarizer(const std::vector<ChatMessage>& messages) const;
};

/**
 * @brief Conversation summary memory with overlapping windows
 */
class ConversationSummaryMemory : public BaseMemory {
private:
    std::vector<ChatMessage> messages_;
    std::string summary_;
    size_t context_window_size_{10};
    std::function<std::string(const std::vector<ChatMessage>&, const std::string&)> predictive_summarizer_;

public:
    explicit ConversationSummaryMemory(
        size_t context_window_size = 10,
        std::function<std::string(const std::vector<ChatMessage>&, const std::string&)> predictive_summarizer = nullptr,
        const MemoryConfig& config = MemoryConfig{}
    );

    void add_message(const ChatMessage& message) override;
    std::string get_context() const override;
    std::vector<ChatMessage> get_messages() const override;
    void clear() override;
    std::unordered_map<std::string, size_t> get_stats() const override;

private:
    std::vector<ChatMessage> get_context_window() const;
    void update_summary();
};

/**
 * @brief Knowledge graph memory for entity relationships
 */
struct Entity {
    std::string name;
    std::string type;
    std::unordered_map<std::string, std::string> attributes;
};

struct Relation {
    std::string subject;
    std::string predicate;
    std::string object;
    double confidence{1.0};
};

class KnowledgeGraphMemory : public BaseMemory {
private:
    std::vector<ChatMessage> messages_;
    std::vector<Entity> entities_;
    std::vector<Relation> relations_;
    std::function<std::vector<Entity>(const std::string&)> entity_extractor_;
    std::function<std::vector<Relation>(const std::string&, const std::vector<Entity>&)> relation_extractor_;

public:
    explicit KnowledgeGraphMemory(
        std::function<std::vector<Entity>(const std::string&)> entity_extractor = nullptr,
        std::function<std::vector<Relation>(const std::string&, const std::vector<Entity>&)> relation_extractor = nullptr,
        const MemoryConfig& config = MemoryConfig{}
    );

    void add_message(const ChatMessage& message) override;
    std::string get_context() const override;
    std::vector<ChatMessage> get_messages() const override;
    void clear() override;
    std::unordered_map<std::string, size_t> get_stats() const override;

    // Knowledge graph specific methods
    const std::vector<Entity>& get_entities() const { return entities_; }
    const std::vector<Relation>& get_relations() const { return relations_; }
    void add_entity(const Entity& entity);
    void add_relation(const Relation& relation);

private:
    void extract_knowledge(const ChatMessage& message);
    std::string format_knowledge_graph() const;
};

/**
 * @brief Memory factory for creating different memory types
 */
class MemoryFactory {
public:
    enum class MemoryType {
        BUFFER,
        TOKEN_BUFFER,
        SUMMARY,
        CONVERSATION_SUMMARY,
        KNOWLEDGE_GRAPH
    };

    static std::unique_ptr<BaseMemory> create_memory(
        MemoryType type,
        const MemoryConfig& config = MemoryConfig{}
    );

    // Convenience factory methods
    static std::unique_ptr<BaseMemory> create_buffer_memory(const MemoryConfig& config = MemoryConfig{});
    static std::unique_ptr<BaseMemory> create_token_buffer_memory(size_t max_tokens = 2000);
    static std::unique_ptr<BaseMemory> create_summary_memory(const MemoryConfig& config = MemoryConfig{});
    static std::unique_ptr<BaseMemory> create_knowledge_graph_memory(const MemoryConfig& config = MemoryConfig{});
};

/**
 * @brief Utility functions for memory management
 */
namespace utils {
    // Load conversation from file
    std::vector<ChatMessage> load_conversation(const std::string& file_path);

    // Save conversation to file
    void save_conversation(const std::vector<ChatMessage>& messages, const std::string& file_path);

    // Convert message type to string
    std::string message_type_to_string(ChatMessage::Type type);

    // Create simple token counter
    std::function<size_t(const std::string&)> create_simple_token_counter();

    // Create basic summarizer
    std::function<std::string(const std::vector<ChatMessage>&)> create_basic_summarizer();
}

} // namespace langchain::memory