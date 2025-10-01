#pragma once

#include "../core/base.hpp"
#include "../core/config.hpp"
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <variant>
#include <unordered_map>
#include <chrono>

namespace langchain::llm {

/**
 * @brief Message role types for chat conversations
 */
enum class MessageRole {
    SYSTEM,
    USER,
    ASSISTANT,
    FUNCTION,
    TOOL
};

/**
 * @brief Convert message role to string
 */
inline std::string message_role_to_string(MessageRole role) {
    switch (role) {
        case MessageRole::SYSTEM: return "system";
        case MessageRole::USER: return "user";
        case MessageRole::ASSISTANT: return "assistant";
        case MessageRole::FUNCTION: return "function";
        case MessageRole::TOOL: return "tool";
        default: return "unknown";
    }
}

/**
 * @brief Convert string to message role
 */
inline MessageRole string_to_message_role(const std::string& role_str) {
    if (role_str == "system") return MessageRole::SYSTEM;
    if (role_str == "user") return MessageRole::USER;
    if (role_str == "assistant") return MessageRole::ASSISTANT;
    if (role_str == "function") return MessageRole::FUNCTION;
    if (role_str == "tool") return MessageRole::TOOL;
    return MessageRole::USER; // Default to user
}

/**
 * @brief Chat message structure
 */
struct ChatMessage {
    MessageRole role;
    std::string content;
    std::string name; // Optional name for function/tool messages
    std::unordered_map<std::string, std::string> metadata; // Additional metadata

    ChatMessage(MessageRole r, std::string c, std::string n = "")
        : role(r), content(std::move(c)), name(std::move(n)) {}

    ChatMessage(MessageRole r, std::string c,
                std::unordered_map<std::string, std::string> meta)
        : role(r), content(std::move(c)), metadata(std::move(meta)) {}
};

/**
 * @brief Function/tool call definition
 */
struct FunctionCall {
    std::string name;
    std::string arguments; // JSON string of arguments

    FunctionCall(std::string n, std::string args)
        : name(std::move(n)), arguments(std::move(args)) {}
};

/**
 * @brief Tool definition for function calling
 */
struct Tool {
    enum class Type {
        FUNCTION
    };

    Type type;
    std::string name;
    std::string description;
    std::string parameters_schema; // JSON schema for parameters

    Tool(Type t, std::string n, std::string desc, std::string params)
        : type(t), name(std::move(n)), description(std::move(desc)),
          parameters_schema(std::move(params)) {}
};

/**
 * @brief Token usage statistics
 */
struct TokenUsage {
    size_t prompt_tokens = 0;
    size_t completion_tokens = 0;
    size_t total_tokens = 0;

    TokenUsage() = default;
    TokenUsage(size_t prompt, size_t completion, size_t total)
        : prompt_tokens(prompt), completion_tokens(completion), total_tokens(total) {}
};

/**
 * @brief LLM response structure
 */
struct LLMResponse {
    std::string content;
    std::vector<ChatMessage> messages; // Full conversation history
    std::optional<FunctionCall> function_call;
    std::vector<FunctionCall> tool_calls;
    TokenUsage token_usage;
    std::string model;
    double duration_ms = 0.0;
    std::unordered_map<std::string, std::string> metadata;
    bool success = true;
    std::string error_message;

    LLMResponse() = default;
    LLMResponse(std::string cont, TokenUsage tokens, std::string mdl, double duration)
        : content(std::move(cont)), token_usage(std::move(tokens)),
          model(std::move(mdl)), duration_ms(duration) {}
};

/**
 * @brief LLM configuration parameters
 */
struct LLMConfig {
    // Model configuration
    std::string model;
    std::string api_key;
    std::string base_url;

    // Generation parameters
    double temperature = 0.7;
    double top_p = 1.0;
    size_t max_tokens = 2048;
    size_t n = 1; // Number of completions to generate

    // Sampling parameters
    std::optional<double> frequency_penalty;
    std::optional<double> presence_penalty;
    std::vector<std::string> stop_sequences;

    // Streaming configuration
    bool stream = false;
    std::optional<std::chrono::milliseconds> timeout;

    // Tool/function calling
    bool tool_calling_enabled = false;
    std::vector<Tool> tools;

    // Validation
    void validate() const;

    // Create default config
    static LLMConfig default_config() {
        return LLMConfig{};
    }
};

/**
 * @brief Base interface for all LLM implementations
 */
class BaseLLM {
public:
    virtual ~BaseLLM() = default;

    /**
     * @brief Generate completion for a single prompt
     * @param prompt Input prompt
     * @param config Optional configuration override
     * @return LLM response
     */
    virtual LLMResponse complete(
        const std::string& prompt,
        const std::optional<LLMConfig>& config = std::nullopt
    ) = 0;

    /**
     * @brief Generate chat completion
     * @param messages Conversation history
     * @param config Optional configuration override
     * @return LLM response
     */
    virtual LLMResponse chat(
        const std::vector<ChatMessage>& messages,
        const std::optional<LLMConfig>& config = std::nullopt
    ) = 0;

    /**
     * @brief Generate completion with streaming
     * @param prompt Input prompt
     * @param callback Stream callback function
     * @param config Optional configuration override
     * @return Final LLM response
     */
    virtual LLMResponse stream_complete(
        const std::string& prompt,
        std::function<void(const std::string&)> callback,
        const std::optional<LLMConfig>& config = std::nullopt
    ) = 0;

    /**
     * @brief Generate chat completion with streaming
     * @param messages Conversation history
     * @param callback Stream callback function
     * @param config Optional configuration override
     * @return Final LLM response
     */
    virtual LLMResponse stream_chat(
        const std::vector<ChatMessage>& messages,
        std::function<void(const std::string&)> callback,
        const std::optional<LLMConfig>& config = std::nullopt
    ) = 0;

    /**
     * @brief Get token count for text
     * @param text Input text
     * @return Number of tokens
     */
    virtual size_t count_tokens(const std::string& text) const = 0;

    /**
     * @brief Get supported models
     * @return Vector of supported model names
     */
    virtual std::vector<std::string> get_supported_models() const = 0;

    /**
     * @brief Check if model is supported
     * @param model Model name
     * @return True if supported
     */
    virtual bool is_model_supported(const std::string& model) const = 0;

    /**
     * @brief Get provider name
     * @return Provider name (e.g., "openai", "anthropic")
     */
    virtual std::string get_provider() const = 0;

    /**
     * @brief Get current configuration
     * @return Current LLM configuration
     */
    virtual const LLMConfig& get_config() const = 0;

    /**
     * @brief Update configuration
     * @param new_config New configuration
     */
    virtual void update_config(const LLMConfig& new_config) = 0;

    /**
     * @brief Validate configuration
     * @param config Configuration to validate
     * @throws std::invalid_argument if configuration is invalid
     */
    virtual void validate_config(const LLMConfig& config) const = 0;

    /**
     * @brief Get capabilities information
     * @return Map of capability names to boolean values
     */
    virtual std::unordered_map<std::string, bool> get_capabilities() const = 0;
};

/**
 * @brief Factory interface for creating LLM instances
 */
class LLMFactory {
public:
    virtual ~LLMFactory() = default;

    /**
     * @brief Create LLM instance with default configuration
     * @return Unique pointer to LLM instance
     */
    virtual std::unique_ptr<BaseLLM> create() const = 0;

    /**
     * @brief Create LLM instance with custom configuration
     * @param config LLM configuration
     * @return Unique pointer to LLM instance
     */
    virtual std::unique_ptr<BaseLLM> create(const LLMConfig& config) const = 0;

    /**
     * @brief Get provider name
     * @return Provider name
     */
    virtual std::string get_provider() const = 0;

    /**
     * @brief Check if provider supports given model
     * @param model Model name
     * @return True if supported
     */
    virtual bool supports_model(const std::string& model) const = 0;
};

/**
 * @brief Registry for LLM factories
 */
class LLMRegistry {
public:
    /**
     * @brief Register a factory
     * @param provider Provider name
     * @param factory Factory instance
     */
    static void register_factory(
        const std::string& provider,
        std::unique_ptr<LLMFactory> factory
    );

    /**
     * @brief Get factory for provider
     * @param provider Provider name
     * @return Factory instance or nullptr if not found
     */
    static LLMFactory* get_factory(const std::string& provider);

    /**
     * @brief Create LLM instance by provider name
     * @param provider Provider name
     * @param config Configuration
     * @return Unique pointer to LLM instance
     */
    static std::unique_ptr<BaseLLM> create_llm(
        const std::string& provider,
        const LLMConfig& config = LLMConfig::default_config()
    );

    /**
     * @brief List registered providers
     * @return Vector of provider names
     */
    static std::vector<std::string> list_providers();

    /**
     * @brief Check if provider is registered
     * @param provider Provider name
     * @return True if registered
     */
    static bool is_provider_registered(const std::string& provider);

private:
    static std::unordered_map<std::string, std::unique_ptr<LLMFactory>>& get_factories();
};

} // namespace langchain::llm