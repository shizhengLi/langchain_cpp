#pragma once

#include "base_llm.hpp"
#include <string>
#include <memory>
#include <vector>
#include <chrono>

namespace langchain::llm {

/**
 * @brief OpenAI-specific LLM configuration
 */
struct OpenAIApiConfig {
    std::string api_key;
    std::string base_url = "https://api.openai.com/v1";
    std::string organization; // Optional
    std::string model = "gpt-3.5-turbo";
    double temperature = 0.7;
    double top_p = 1.0;
    size_t max_tokens = 2048;
    std::optional<double> frequency_penalty;
    std::optional<double> presence_penalty;
    std::vector<std::string> stop_sequences;
    bool stream = false;
    std::optional<std::chrono::milliseconds> timeout = std::chrono::milliseconds(30000);

    void validate() const;
};

/**
 * @brief OpenAI chat model implementation
 */
class OpenAILLM : public BaseLLM {
private:
    OpenAIApiConfig config_;

    // Token counting for OpenAI models
    size_t count_tokens_approximate(const std::string& text) const;
    size_t count_tokens_chat_messages(const std::vector<ChatMessage>& messages) const;

public:
    /**
     * @brief Constructor with OpenAI-specific configuration
     */
    explicit OpenAILLM(const OpenAIApiConfig& config);

    /**
     * @brief Constructor with generic LLM config
     */
    explicit OpenAILLM(const LLMConfig& config);

    /**
     * @brief Destructor
     */
    ~OpenAILLM() override;

    // BaseLLM interface implementation
    LLMResponse complete(
        const std::string& prompt,
        const std::optional<LLMConfig>& config = std::nullopt
    ) override;

    LLMResponse chat(
        const std::vector<ChatMessage>& messages,
        const std::optional<LLMConfig>& config = std::nullopt
    ) override;

    LLMResponse stream_complete(
        const std::string& prompt,
        std::function<void(const std::string&)> callback,
        const std::optional<LLMConfig>& config = std::nullopt
    ) override;

    LLMResponse stream_chat(
        const std::vector<ChatMessage>& messages,
        std::function<void(const std::string&)> callback,
        const std::optional<LLMConfig>& config = std::nullopt
    ) override;

    size_t count_tokens(const std::string& text) const override;

    std::vector<std::string> get_supported_models() const override;

    bool is_model_supported(const std::string& model) const override;

    std::string get_provider() const override;

    const LLMConfig& get_config() const override;

    void update_config(const LLMConfig& new_config) override;

    void validate_config(const LLMConfig& config) const override;

    std::unordered_map<std::string, bool> get_capabilities() const override;

    // OpenAI-specific methods

    /**
     * @brief Get current OpenAI configuration
     */
    const OpenAIApiConfig& get_openai_config() const;

    /**
     * @brief Update OpenAI configuration
     */
    void update_openai_config(const OpenAIApiConfig& new_config);

    /**
     * @brief Test API connection
     */
    bool test_connection() const;
};

/**
 * @brief Factory for creating OpenAI LLM instances
 */
class OpenAIFactory : public LLMFactory {
public:
    std::unique_ptr<BaseLLM> create() const override;

    std::unique_ptr<BaseLLM> create(const LLMConfig& config) const override;

    std::string get_provider() const override;

    bool supports_model(const std::string& model) const override;
};

} // namespace langchain::llm