#include "langchain/llm/base_llm.hpp"
#include "langchain/utils/logging.hpp"
#include <stdexcept>
#include <algorithm>
#include <iomanip>

namespace langchain::llm {

void LLMConfig::validate() const {
    LOG_DEBUG("Validating LLM configuration");

    // Validate model name
    if (model.empty()) {
        throw std::invalid_argument("Model name cannot be empty");
    }

    // Validate temperature
    if (temperature < 0.0 || temperature > 2.0) {
        throw std::invalid_argument("Temperature must be between 0.0 and 2.0");
    }

    // Validate top_p
    if (top_p <= 0.0 || top_p > 1.0) {
        throw std::invalid_argument("Top_p must be between 0.0 and 1.0");
    }

    // Validate max_tokens
    if (max_tokens == 0 || max_tokens > 128000) {
        throw std::invalid_argument("Max tokens must be between 1 and 128000");
    }

    // Validate n (number of completions)
    if (n == 0 || n > 10) {
        throw std::invalid_argument("Number of completions must be between 1 and 10");
    }

    // Validate frequency penalty if provided
    if (frequency_penalty.has_value()) {
        if (*frequency_penalty < -2.0 || *frequency_penalty > 2.0) {
            throw std::invalid_argument("Frequency penalty must be between -2.0 and 2.0");
        }
    }

    // Validate presence penalty if provided
    if (presence_penalty.has_value()) {
        if (*presence_penalty < -2.0 || *presence_penalty > 2.0) {
            throw std::invalid_argument("Presence penalty must be between -2.0 and 2.0");
        }
    }

    // Validate timeout if provided
    if (timeout.has_value()) {
        if (timeout->count() < 100 || timeout->count() > 300000) { // 100ms to 5 minutes
            throw std::invalid_argument("Timeout must be between 100ms and 300000ms");
        }
    }

    // Validate tools if provided
    if (tool_calling_enabled && tools.empty()) {
        LOG_WARN("Tool calling enabled but no tools provided");
    }

    for (const auto& tool : tools) {
        if (tool.name.empty()) {
            throw std::invalid_argument("Tool name cannot be empty");
        }
        if (tool.type == Tool::Type::FUNCTION && tool.parameters_schema.empty()) {
            throw std::invalid_argument("Function tool must have parameters schema");
        }
    }

    LOG_DEBUG("LLM configuration validation passed");
}

// LLMRegistry implementation
std::unordered_map<std::string, std::unique_ptr<LLMFactory>>& LLMRegistry::get_factories() {
    static std::unordered_map<std::string, std::unique_ptr<LLMFactory>> factories;
    return factories;
}

void LLMRegistry::register_factory(
    const std::string& provider,
    std::unique_ptr<LLMFactory> factory
) {
    LOG_INFO("Registering LLM factory for provider: " + provider);

    if (provider.empty()) {
        throw std::invalid_argument("Provider name cannot be empty");
    }

    if (!factory) {
        throw std::invalid_argument("Factory cannot be null");
    }

    get_factories()[provider] = std::move(factory);
    LOG_INFO("Successfully registered factory for provider: " + provider);
}

LLMFactory* LLMRegistry::get_factory(const std::string& provider) {
    LOG_DEBUG("Getting factory for provider: " + provider);

    auto it = get_factories().find(provider);
    if (it != get_factories().end()) {
        LOG_DEBUG("Found factory for provider: " + provider);
        return it->second.get();
    }

    LOG_WARN("No factory found for provider: " + provider);
    return nullptr;
}

std::unique_ptr<BaseLLM> LLMRegistry::create_llm(
    const std::string& provider,
    const LLMConfig& config
) {
    LOG_INFO("Creating LLM instance for provider: " + provider);

    auto factory = get_factory(provider);
    if (!factory) {
        throw std::runtime_error("No factory registered for provider: " + provider);
    }

    try {
        auto llm = factory->create(config);
        LOG_INFO("Successfully created LLM instance for provider: " + provider);
        return llm;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create LLM instance for provider " +
                                   provider + ": " + e.what());
        throw;
    }
}

std::vector<std::string> LLMRegistry::list_providers() {
    std::vector<std::string> providers;
    const auto& factories = get_factories();

    providers.reserve(factories.size());
    for (const auto& [provider, _] : factories) {
        providers.push_back(provider);
    }

    std::sort(providers.begin(), providers.end());
    LOG_DEBUG("Listed " + std::to_string(providers.size()) + " providers");

    return providers;
}

bool LLMRegistry::is_provider_registered(const std::string& provider) {
    const auto& factories = get_factories();
    bool registered = factories.find(provider) != factories.end();

    LOG_DEBUG("Provider " + provider + " is " +
                               (registered ? "registered" : "not registered"));

    return registered;
}

} // namespace langchain::llm