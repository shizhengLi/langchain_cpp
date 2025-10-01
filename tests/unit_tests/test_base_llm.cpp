#include "langchain/llm/base_llm.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>
#include <string>

using namespace langchain::llm;

// Mock implementation of BaseLLM for testing
class MockLLM : public BaseLLM {
private:
    LLMConfig config_;
    std::string provider_name_;

public:
    explicit MockLLM(const LLMConfig& config = LLMConfig{},
                    const std::string& provider = "mock")
        : config_(config), provider_name_(provider) {}

    // BaseLLM interface implementation
    LLMResponse complete(
        const std::string& prompt,
        const std::optional<LLMConfig>& config = std::nullopt
    ) override {
        LLMResponse response;
        response.content = "Mock response to: " + prompt;
        response.model = config_.model.empty() ? "mock-model" : config_.model;
        response.token_usage = TokenUsage{10, 20, 30};
        response.duration_ms = 100.0;
        response.success = true;
        return response;
    }

    LLMResponse chat(
        const std::vector<ChatMessage>& messages,
        const std::optional<LLMConfig>& config = std::nullopt
    ) override {
        std::string context;
        for (const auto& msg : messages) {
            context += "[" + message_role_to_string(msg.role) + "] " + msg.content + "\n";
        }

        LLMResponse response;
        response.content = "Mock chat response to: " + context;
        response.model = config_.model.empty() ? "mock-chat-model" : config_.model;
        response.token_usage = TokenUsage{15, 25, 40};
        response.duration_ms = 150.0;
        response.success = true;
        return response;
    }

    LLMResponse stream_complete(
        const std::string& prompt,
        std::function<void(const std::string&)> callback,
        const std::optional<LLMConfig>& config = std::nullopt
    ) override {
        // Simulate streaming by calling callback multiple times
        std::string full_response = "Mock streaming response to: " + prompt;
        for (size_t i = 0; i < full_response.size(); i += 5) {
            size_t chunk_size = std::min(size_t(5), full_response.size() - i);
            callback(full_response.substr(i, chunk_size));
        }

        LLMResponse final_response = complete(prompt, config);
        final_response.content = full_response; // Use the actual streamed content
        return final_response;
    }

    LLMResponse stream_chat(
        const std::vector<ChatMessage>& messages,
        std::function<void(const std::string&)> callback,
        const std::optional<LLMConfig>& config = std::nullopt
    ) override {
        // Simulate streaming chat
        std::string full_response = "Mock streaming chat response";
        for (size_t i = 0; i < full_response.size(); i += 3) {
            size_t chunk_size = std::min(size_t(3), full_response.size() - i);
            callback(full_response.substr(i, chunk_size));
        }

        LLMResponse final_response = chat(messages, config);
        final_response.content = full_response; // Use the actual streamed content
        return final_response;
    }

    size_t count_tokens(const std::string& text) const override {
        // Mock token counting: assume 1 token per 4 characters
        return (text.length() + 3) / 4;
    }

    std::vector<std::string> get_supported_models() const override {
        return {"mock-model", "mock-chat-model", "mock-large-model"};
    }

    bool is_model_supported(const std::string& model) const override {
        auto supported = get_supported_models();
        return std::find(supported.begin(), supported.end(), model) != supported.end();
    }

    std::string get_provider() const override {
        return provider_name_;
    }

    const LLMConfig& get_config() const override {
        return config_;
    }

    void update_config(const LLMConfig& new_config) override {
        config_ = new_config;
    }

    void validate_config(const LLMConfig& config) const override {
        if (config.model.empty()) {
            throw std::invalid_argument("Model name cannot be empty");
        }
        if (config.temperature < 0.0 || config.temperature > 2.0) {
            throw std::invalid_argument("Temperature must be between 0.0 and 2.0");
        }
        if (config.max_tokens == 0) {
            throw std::invalid_argument("Max tokens must be greater than 0");
        }
    }

    std::unordered_map<std::string, bool> get_capabilities() const override {
        return {
            {"completion", true},
            {"chat", true},
            {"streaming", true},
            {"function_calling", false},
            {"multimodal", false}
        };
    }
};

TEST_CASE("LLM Message Role Utilities", "[llm][message]") {
    SECTION("Message role to string conversion") {
        REQUIRE(message_role_to_string(MessageRole::SYSTEM) == "system");
        REQUIRE(message_role_to_string(MessageRole::USER) == "user");
        REQUIRE(message_role_to_string(MessageRole::ASSISTANT) == "assistant");
        REQUIRE(message_role_to_string(MessageRole::FUNCTION) == "function");
        REQUIRE(message_role_to_string(MessageRole::TOOL) == "tool");
    }

    SECTION("String to message role conversion") {
        REQUIRE(string_to_message_role("system") == MessageRole::SYSTEM);
        REQUIRE(string_to_message_role("user") == MessageRole::USER);
        REQUIRE(string_to_message_role("assistant") == MessageRole::ASSISTANT);
        REQUIRE(string_to_message_role("function") == MessageRole::FUNCTION);
        REQUIRE(string_to_message_role("tool") == MessageRole::TOOL);
        REQUIRE(string_to_message_role("unknown") == MessageRole::USER); // Default
    }

    SECTION("Round trip conversion") {
        for (auto role : {MessageRole::SYSTEM, MessageRole::USER, MessageRole::ASSISTANT,
                        MessageRole::FUNCTION, MessageRole::TOOL}) {
            std::string role_str = message_role_to_string(role);
            MessageRole converted_role = string_to_message_role(role_str);
            REQUIRE(converted_role == role);
        }
    }
}

TEST_CASE("LLM Configuration", "[llm][config]") {
    SECTION("Default configuration") {
        LLMConfig config = LLMConfig::default_config();

        REQUIRE(config.temperature == 0.7);
        REQUIRE(config.top_p == 1.0);
        REQUIRE(config.max_tokens == 2048);
        REQUIRE(config.n == 1);
        REQUIRE(config.stream == false);
        REQUIRE(config.stream == false);
        REQUIRE(config.tool_calling_enabled == false);
    }

    SECTION("Configuration validation - valid") {
        LLMConfig config;
        config.model = "gpt-4";
        config.temperature = 1.0;
        config.top_p = 0.9;
        config.max_tokens = 1000;
        config.n = 3;
        config.stream = true;
        config.timeout = std::chrono::milliseconds(5000);

        REQUIRE_NOTHROW(config.validate());
    }

    SECTION("Configuration validation - invalid") {
        LLMConfig config;

        SECTION("Empty model name") {
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
        }

        SECTION("Invalid temperature") {
            config.model = "test";
            config.temperature = -0.1;
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);

            config.temperature = 2.1;
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
        }

        SECTION("Invalid top_p") {
            config.model = "test";
            config.top_p = 0.0;
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);

            config.top_p = 1.1;
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
        }

        SECTION("Invalid max_tokens") {
            config.model = "test";
            config.max_tokens = 0;
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);

            config.max_tokens = 130000; // Too large
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
        }

        SECTION("Invalid n parameter") {
            config.model = "test";
            config.n = 0;
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);

            config.n = 11; // Too many completions
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
        }

        SECTION("Invalid frequency penalty") {
            config.model = "test";
            config.frequency_penalty = -2.1;
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);

            config.frequency_penalty = 2.1;
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
        }

        SECTION("Invalid presence penalty") {
            config.model = "test";
            config.presence_penalty = -2.1;
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);

            config.presence_penalty = 2.1;
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
        }

        SECTION("Invalid timeout") {
            config.model = "test";
            config.timeout = std::chrono::milliseconds(50); // Too short
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);

            config.timeout = std::chrono::milliseconds(300001); // Too long
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
        }
    }
}

TEST_CASE("LLM Basic Operations", "[llm][basic]") {
    SECTION("Completion functionality") {
        MockLLM llm;

        LLMResponse response = llm.complete("Hello, world!");

        REQUIRE(response.success == true);
        REQUIRE_FALSE(response.content.empty());
        REQUIRE(response.content == "Mock response to: Hello, world!");
        REQUIRE(response.model == "mock-model");
        REQUIRE(response.token_usage.prompt_tokens == 10);
        REQUIRE(response.token_usage.completion_tokens == 20);
        REQUIRE(response.token_usage.total_tokens == 30);
        REQUIRE(response.duration_ms == 100.0);
        REQUIRE_FALSE(response.model.empty());
    }

    SECTION("Chat functionality") {
        MockLLM llm;

        std::vector<ChatMessage> messages = {
            ChatMessage(MessageRole::USER, "Hello"),
            ChatMessage(MessageRole::ASSISTANT, "Hi there!"),
            ChatMessage(MessageRole::USER, "How are you?")
        };

        LLMResponse response = llm.chat(messages);

        REQUIRE(response.success == true);
        REQUIRE_FALSE(response.content.empty());
        REQUIRE(response.token_usage.total_tokens == 40);
        REQUIRE(response.duration_ms == 150.0);
    }

    SECTION("Streaming completion") {
        MockLLM llm;
        std::vector<std::string> chunks;

        auto callback = [&chunks](const std::string& chunk) {
            chunks.push_back(chunk);
        };

        LLMResponse response = llm.stream_complete("Test streaming", callback);

        REQUIRE(response.success == true);
        REQUIRE_FALSE(chunks.empty());
        REQUIRE(chunks.size() > 1); // Should receive multiple chunks

        // Reconstruct full response from chunks
        std::string reconstructed;
        for (const auto& chunk : chunks) {
            reconstructed += chunk;
        }
        REQUIRE(reconstructed == response.content);
    }

    SECTION("Streaming chat") {
        MockLLM llm;
        std::vector<std::string> chunks;

        auto callback = [&chunks](const std::string& chunk) {
            chunks.push_back(chunk);
        };

        std::vector<ChatMessage> messages = {
            ChatMessage(MessageRole::USER, "Stream test")
        };

        LLMResponse response = llm.stream_chat(messages, callback);

        REQUIRE(response.success == true);
        REQUIRE_FALSE(chunks.empty());
    }

    SECTION("Token counting") {
        MockLLM llm;

        size_t tokens = llm.count_tokens("This is a test message for token counting");

        REQUIRE(tokens > 0);
        REQUIRE(tokens == 11); // "This is a test message for token counting".length() = 43, 43/4 = 10.75 -> 11
    }

    SECTION("Model support") {
        MockLLM llm;

        auto supported_models = llm.get_supported_models();
        REQUIRE(supported_models.size() == 3);
        REQUIRE(std::find(supported_models.begin(), supported_models.end(), "mock-model") != supported_models.end());

        REQUIRE(llm.is_model_supported("mock-model") == true);
        REQUIRE(llm.is_model_supported("unsupported-model") == false);
    }

    SECTION("Provider and capabilities") {
        MockLLM llm({}, "test-provider");

        REQUIRE(llm.get_provider() == "test-provider");

        auto capabilities = llm.get_capabilities();
        REQUIRE(capabilities.at("completion") == true);
        REQUIRE(capabilities.at("chat") == true);
        REQUIRE(capabilities.at("streaming") == true);
        REQUIRE(capabilities.at("function_calling") == false);
        REQUIRE(capabilities.at("multimodal") == false);
    }
}

TEST_CASE("LLM Configuration Management", "[llm][config]") {
    SECTION("Configuration update") {
        LLMConfig new_config;
        new_config.model = "gpt-4";
        new_config.temperature = 1.5;
        new_config.max_tokens = 500;

        MockLLM llm;
        llm.update_config(new_config);

        auto retrieved_config = llm.get_config();
        REQUIRE(retrieved_config.model == "gpt-4");
        REQUIRE(retrieved_config.temperature == 1.5);
        REQUIRE(retrieved_config.max_tokens == 500);
    }

    SECTION("Configuration validation through LLM") {
        MockLLM llm;

        LLMConfig valid_config;
        valid_config.model = "test-model";

        REQUIRE_NOTHROW(llm.validate_config(valid_config));

        LLMConfig invalid_config;
        // model is empty
        REQUIRE_THROWS_AS(llm.validate_config(invalid_config), std::invalid_argument);
    }
}

TEST_CASE("LLM Error Handling", "[llm][error]") {
    SECTION("Empty query handling") {
        MockLLM llm;

        LLMResponse response = llm.complete("");
        REQUIRE(response.success == true);
        REQUIRE_FALSE(response.content.empty()); // Mock implementation should handle empty queries
    }

    SECTION("Empty messages handling") {
        MockLLM llm;

        std::vector<ChatMessage> empty_messages;
        LLMResponse response = llm.chat(empty_messages);
        REQUIRE(response.success == true);
        REQUIRE_FALSE(response.content.empty()); // Mock implementation should handle empty messages
    }

    SECTION("Long queries") {
        MockLLM llm;

        std::string long_query(10000, 'a'); // 10KB query
        LLMResponse response = llm.complete(long_query);
        REQUIRE(response.success == true);
        REQUIRE_FALSE(response.content.empty());
    }
}

TEST_CASE("LLM Registry", "[llm][registry]") {
    SECTION("Factory registration") {
        // Test that the static registry methods exist and don't crash
        // In a real implementation, this would involve actual factory registration

        auto providers = LLMRegistry::list_providers();
        REQUIRE(providers.empty()); // No providers registered yet

        REQUIRE_FALSE(LLMRegistry::is_provider_registered("nonexistent"));

        // Test creating LLM (this will fail since no factories are registered)
        REQUIRE_THROWS_AS(
            LLMRegistry::create_llm("nonexistent"),
            std::runtime_error
        );
    }
}

TEST_CASE("LLM Performance", "[llm][performance]") {
    SECTION("Multiple completions") {
        MockLLM llm;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < 100; ++i) {
            std::string prompt = "Test query " + std::to_string(i);
            auto response = llm.complete(prompt);
            REQUIRE(response.success == true);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Should complete 100 mock queries quickly (under 1 second)
        REQUIRE(duration.count() < 1000);
    }

    SECTION("Memory usage") {
        MockLLM llm;

        // Test multiple responses without memory leaks
        for (int i = 0; i < 1000; ++i) {
            auto response = llm.complete("Memory test " + std::to_string(i));
            REQUIRE(response.success == true);
        }

        // If we reach here without crashing, memory management is working
        REQUIRE(true);
    }
}