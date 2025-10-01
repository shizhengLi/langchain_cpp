#include "langchain/llm/openai_llm.hpp"
#include "langchain/utils/logging.hpp"
#include <sstream>
#include <algorithm>

namespace langchain::llm {

// OpenAIApiConfig implementation
void OpenAIApiConfig::validate() const {
    LOG_DEBUG("Validating OpenAI API configuration");

    if (api_key.empty()) {
        throw std::invalid_argument("OpenAI API key cannot be empty");
    }

    if (base_url.empty()) {
        throw std::invalid_argument("Base URL cannot be empty");
    }

    if (model.empty()) {
        throw std::invalid_argument("Model name cannot be empty");
    }

    if (temperature < 0.0 || temperature > 2.0) {
        throw std::invalid_argument("Temperature must be between 0.0 and 2.0");
    }

    if (top_p <= 0.0 || top_p > 1.0) {
        throw std::invalid_argument("Top_p must be between 0.0 and 1.0");
    }

    if (max_tokens == 0 || max_tokens > 128000) {
        throw std::invalid_argument("Max tokens must be between 1 and 128000");
    }

    if (frequency_penalty.has_value() &&
        (*frequency_penalty < -2.0 || *frequency_penalty > 2.0)) {
        throw std::invalid_argument("Frequency penalty must be between -2.0 and 2.0");
    }

    if (presence_penalty.has_value() &&
        (*presence_penalty < -2.0 || *presence_penalty > 2.0)) {
        throw std::invalid_argument("Presence penalty must be between -2.0 and 2.0");
    }

    if (timeout.has_value() &&
        (timeout->count() < 100 || timeout->count() > 300000)) {
        throw std::invalid_argument("Timeout must be between 100ms and 300000ms");
    }

    LOG_DEBUG("OpenAI API configuration validation passed");
}

// OpenAILLM implementation
OpenAILLM::OpenAILLM(const OpenAIApiConfig& config) : config_(config) {
    if (!config.api_key.empty()) {
        config_.validate();
    }
    LOG_INFO("OpenAI LLM initialized with model: " + config_.model);
}

OpenAILLM::OpenAILLM(const LLMConfig& config) {
    // Convert generic LLM config to OpenAI-specific config
    config_.api_key = config.api_key;
    config_.base_url = config.base_url.empty() ? "https://api.openai.com/v1" : config.base_url;
    config_.model = config.model.empty() ? "gpt-3.5-turbo" : config.model;
    config_.temperature = config.temperature;
    config_.top_p = config.top_p;
    config_.max_tokens = config.max_tokens;
    config_.frequency_penalty = config.frequency_penalty;
    config_.presence_penalty = config.presence_penalty;
    config_.stop_sequences = config.stop_sequences;
    config_.stream = config.stream;
    config_.timeout = config.timeout;

    if (!config_.api_key.empty()) {
        config_.validate();
    }
    LOG_INFO("OpenAI LLM initialized from generic config with model: " + config_.model);
}

OpenAILLM::~OpenAILLM() {
    LOG_DEBUG("OpenAI LLM destroyed");
}

LLMResponse OpenAILLM::complete(const std::string& prompt, const std::optional<LLMConfig>& config) {
    LOG_DEBUG("OpenAI completion request for prompt length: " + std::to_string(prompt.length()));

    // Mock implementation for development/testing
    LLMResponse result;
    result.content = "Mock OpenAI response to: " + prompt;
    result.model = config_.model;
    result.token_usage = TokenUsage{10, 20, 30};
    result.duration_ms = 200.0;
    result.success = true;

    LOG_DEBUG("OpenAI completion completed (mock)");
    return result;
}

LLMResponse OpenAILLM::chat(const std::vector<ChatMessage>& messages, const std::optional<LLMConfig>& config) {
    LOG_DEBUG("OpenAI chat request with " + std::to_string(messages.size()) + " messages");

    // Mock implementation for development/testing
    std::string context;
    for (const auto& msg : messages) {
        context += "[" + message_role_to_string(msg.role) + "] " + msg.content + "\\n";
    }

    LLMResponse result;
    result.content = "Mock OpenAI chat response to: " + context;
    result.model = config_.model;
    result.token_usage = TokenUsage{15, 25, 40};
    result.duration_ms = 250.0;
    result.success = true;

    LOG_DEBUG("OpenAI chat completed (mock)");
    return result;
}

LLMResponse OpenAILLM::stream_complete(
    const std::string& prompt,
    std::function<void(const std::string&)> callback,
    const std::optional<LLMConfig>& config
) {
    LOG_DEBUG("OpenAI streaming completion request");

    // Mock streaming implementation
    std::string full_response = "Mock streaming OpenAI response to: " + prompt;
    for (size_t i = 0; i < full_response.size(); i += 5) {
        size_t chunk_size = std::min(size_t(5), full_response.size() - i);
        callback(full_response.substr(i, chunk_size));
    }

    return complete(prompt, config);
}

LLMResponse OpenAILLM::stream_chat(
    const std::vector<ChatMessage>& messages,
    std::function<void(const std::string&)> callback,
    const std::optional<LLMConfig>& config
) {
    LOG_DEBUG("OpenAI streaming chat request");

    // Mock streaming implementation
    std::string full_response = "Mock streaming OpenAI chat response";
    for (size_t i = 0; i < full_response.size(); i += 3) {
        size_t chunk_size = std::min(size_t(3), full_response.size() - i);
        callback(full_response.substr(i, chunk_size));
    }

    return chat(messages, config);
}

size_t OpenAILLM::count_tokens(const std::string& text) const {
    return count_tokens_approximate(text);
}

size_t OpenAILLM::count_tokens_approximate(const std::string& text) const {
    // Simple approximation: ~4 characters per token for English text
    return (text.length() + 3) / 4;
}

size_t OpenAILLM::count_tokens_chat_messages(const std::vector<ChatMessage>& messages) const {
    size_t total_tokens = 0;
    for (const auto& msg : messages) {
        total_tokens += count_tokens_approximate(msg.content);
        total_tokens += 4; // Approximate overhead per message
    }
    return total_tokens;
}

std::vector<std::string> OpenAILLM::get_supported_models() const {
    return {
        "gpt-4",
        "gpt-4-turbo-preview",
        "gpt-4-32k",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "text-davinci-003",
        "text-davinci-002",
        "text-curie-001",
        "text-babbage-001",
        "text-ada-001"
    };
}

bool OpenAILLM::is_model_supported(const std::string& model) const {
    auto supported = get_supported_models();
    return std::find(supported.begin(), supported.end(), model) != supported.end();
}

std::string OpenAILLM::get_provider() const {
    return "openai";
}

const LLMConfig& OpenAILLM::get_config() const {
    static LLMConfig generic_config;
    generic_config.model = config_.model;
    generic_config.api_key = config_.api_key;
    generic_config.base_url = config_.base_url;
    generic_config.temperature = config_.temperature;
    generic_config.top_p = config_.top_p;
    generic_config.max_tokens = config_.max_tokens;
    generic_config.frequency_penalty = config_.frequency_penalty;
    generic_config.presence_penalty = config_.presence_penalty;
    generic_config.stop_sequences = config_.stop_sequences;
    generic_config.stream = config_.stream;
    generic_config.timeout = config_.timeout;

    return generic_config;
}

void OpenAILLM::update_config(const LLMConfig& new_config) {
    if (!new_config.model.empty()) config_.model = new_config.model;
    if (!new_config.api_key.empty()) config_.api_key = new_config.api_key;
    if (!new_config.base_url.empty()) config_.base_url = new_config.base_url;

    config_.temperature = new_config.temperature;
    config_.top_p = new_config.top_p;
    config_.max_tokens = new_config.max_tokens;
    config_.frequency_penalty = new_config.frequency_penalty;
    config_.presence_penalty = new_config.presence_penalty;
    config_.stop_sequences = new_config.stop_sequences;
    config_.stream = new_config.stream;
    config_.timeout = new_config.timeout;

    config_.validate();
    LOG_INFO("OpenAI LLM configuration updated");
}

void OpenAILLM::validate_config(const LLMConfig& config) const {
    OpenAIApiConfig openai_config;
    if (!config.api_key.empty()) openai_config.api_key = config.api_key;
    if (!config.base_url.empty()) openai_config.base_url = config.base_url;
    if (!config.model.empty()) openai_config.model = config.model;

    openai_config.temperature = config.temperature;
    openai_config.top_p = config.top_p;
    openai_config.max_tokens = config.max_tokens;
    openai_config.frequency_penalty = config.frequency_penalty;
    openai_config.presence_penalty = config.presence_penalty;
    openai_config.stop_sequences = config.stop_sequences;
    openai_config.stream = config.stream;
    openai_config.timeout = config.timeout;

    openai_config.validate();
}

std::unordered_map<std::string, bool> OpenAILLM::get_capabilities() const {
    return {
        {"completion", true},
        {"chat", true},
        {"streaming", true},
        {"function_calling", true},
        {"multimodal", false}
    };
}

const OpenAIApiConfig& OpenAILLM::get_openai_config() const {
    return config_;
}

void OpenAILLM::update_openai_config(const OpenAIApiConfig& new_config) {
    new_config.validate();
    config_ = new_config;
    LOG_INFO("OpenAI LLM configuration updated with new OpenAI config");
}

bool OpenAILLM::test_connection() const {
    // Mock connection test - always return true for now
    return true;
}

// OpenAIFactory implementation
std::unique_ptr<BaseLLM> OpenAIFactory::create() const {
    return std::make_unique<OpenAILLM>(OpenAIApiConfig{});
}

std::unique_ptr<BaseLLM> OpenAIFactory::create(const LLMConfig& config) const {
    return std::make_unique<OpenAILLM>(config);
}

std::string OpenAIFactory::get_provider() const {
    return "openai";
}

bool OpenAIFactory::supports_model(const std::string& model) const {
    OpenAIApiConfig config;
    config.api_key = "sk-temp-key-for-model-check"; // Temporary key for model support validation
    OpenAILLM temp_llm(config);
    return temp_llm.is_model_supported(model);
}

} // namespace langchain::llm