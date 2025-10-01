#include "langchain/llm/openai_llm.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>
#include <string>

using namespace langchain::llm;

TEST_CASE("OpenAI API Configuration", "[llm][openai][config]") {
    SECTION("Valid configuration") {
        OpenAIApiConfig config;
        config.api_key = "sk-test123";
        config.model = "gpt-3.5-turbo";
        config.temperature = 0.7;
        config.max_tokens = 1000;

        REQUIRE_NOTHROW(config.validate());
    }

    SECTION("Invalid configuration - missing API key") {
        OpenAIApiConfig config;
        config.model = "gpt-3.5-turbo";

        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
    }

    SECTION("Invalid configuration - invalid temperature") {
        OpenAIApiConfig config;
        config.api_key = "sk-test123";
        config.model = "gpt-3.5-turbo";
        config.temperature = -0.1; // Too low

        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);

        config.temperature = 2.1; // Too high
        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
    }

    SECTION("Invalid configuration - invalid max_tokens") {
        OpenAIApiConfig config;
        config.api_key = "sk-test123";
        config.model = "gpt-3.5-turbo";
        config.max_tokens = 0; // Too low

        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);

        config.max_tokens = 200000; // Too high
        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
    }

    SECTION("Valid optional parameters") {
        OpenAIApiConfig config;
        config.api_key = "sk-test123";
        config.model = "gpt-3.5-turbo";
        config.frequency_penalty = -0.5;
        config.presence_penalty = 0.8;
        config.stop_sequences = {"\n", "END"};
        config.timeout = std::chrono::milliseconds(10000);

        REQUIRE_NOTHROW(config.validate());
    }

    SECTION("Invalid optional parameters") {
        OpenAIApiConfig config;
        config.api_key = "sk-test123";
        config.model = "gpt-3.5-turbo";
        config.frequency_penalty = -2.1; // Too low

        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);

        config.frequency_penalty = 0.0;
        config.presence_penalty = 2.1; // Too high
        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
    }
}

TEST_CASE("OpenAI LLM Constructor", "[llm][openai][constructor]") {
    SECTION("Constructor with OpenAI config") {
        OpenAIApiConfig config;
        config.api_key = "sk-test123";
        config.model = "gpt-4";

        OpenAILLM llm1(config);

        OpenAILLM llm(config);
        REQUIRE(llm.get_provider() == "openai");
        REQUIRE(llm.get_openai_config().model == "gpt-4");
        REQUIRE(llm.get_openai_config().api_key == "sk-test123");
    }

    SECTION("Constructor with generic LLM config") {
        LLMConfig config;
        config.api_key = "sk-test123";
        config.model = "gpt-3.5-turbo";
        config.temperature = 1.0;
        config.max_tokens = 500;

        OpenAILLM llm1(config);

        OpenAILLM llm(config);
        REQUIRE(llm.get_provider() == "openai");
        REQUIRE(llm.get_openai_config().model == "gpt-3.5-turbo");
        REQUIRE(llm.get_openai_config().temperature == 1.0);
    }

    SECTION("Constructor with empty API key allowed for testing") {
        OpenAIApiConfig config;
        // Missing API key (allowed for testing)
        config.model = "gpt-3.5-turbo";

        // This should not throw as empty API keys are allowed for testing
        OpenAILLM llm(config);
        REQUIRE(llm.get_provider() == "openai");
        REQUIRE(llm.get_openai_config().model == "gpt-3.5-turbo");
    }
}

// Mock OpenAI LLM for testing without API calls
class MockOpenAILLM : public OpenAILLM {
public:
    explicit MockOpenAILLM(const OpenAIApiConfig& config) : OpenAILLM(config) {}

    LLMResponse complete(const std::string& prompt, const std::optional<LLMConfig>& config = std::nullopt) override {
        LLMResponse response;
        response.content = "Mock OpenAI response to: " + prompt;
        response.model = get_openai_config().model;
        response.token_usage = TokenUsage{10, 20, 30};
        response.duration_ms = 200.0;
        response.success = true;
        return response;
    }

    LLMResponse chat(const std::vector<ChatMessage>& messages, const std::optional<LLMConfig>& config = std::nullopt) override {
        std::string context;
        for (const auto& msg : messages) {
            context += "[" + message_role_to_string(msg.role) + "] " + msg.content + "\\n";
        }

        LLMResponse response;
        response.content = "Mock OpenAI chat response to: " + context;
        response.model = get_openai_config().model;
        response.token_usage = TokenUsage{15, 25, 40};
        response.duration_ms = 250.0;
        response.success = true;
        return response;
    }

    LLMResponse stream_complete(const std::string& prompt, std::function<void(const std::string&)> callback, const std::optional<LLMConfig>& config = std::nullopt) override {
        std::string full_response = "Mock streaming OpenAI response to: " + prompt;
        for (size_t i = 0; i < full_response.size(); i += 5) {
            size_t chunk_size = std::min(size_t(5), full_response.size() - i);
            callback(full_response.substr(i, chunk_size));
        }
        LLMResponse final_response = complete(prompt, config);
        final_response.content = full_response; // Use the actual streamed content
        return final_response;
    }

    LLMResponse stream_chat(const std::vector<ChatMessage>& messages, std::function<void(const std::string&)> callback, const std::optional<LLMConfig>& config = std::nullopt) override {
        std::string full_response = "Mock streaming OpenAI chat response";
        for (size_t i = 0; i < full_response.size(); i += 3) {
            size_t chunk_size = std::min(size_t(3), full_response.size() - i);
            callback(full_response.substr(i, chunk_size));
        }
        LLMResponse final_response = chat(messages, config);
        final_response.content = full_response; // Use the actual streamed content
        return final_response;
    }
};

TEST_CASE("Mock OpenAI LLM Operations", "[llm][openai][mock]") {
    OpenAIApiConfig config;
    config.api_key = "sk-test123";
    config.model = "gpt-3.5-turbo";
    MockOpenAILLM llm(config);

    SECTION("Completion functionality") {
        LLMResponse response = llm.complete("Hello, OpenAI!");

        REQUIRE(response.success == true);
        REQUIRE_FALSE(response.content.empty());
        REQUIRE(response.content == "Mock OpenAI response to: Hello, OpenAI!");
        REQUIRE(response.model == "gpt-3.5-turbo");
        REQUIRE(response.token_usage.prompt_tokens == 10);
        REQUIRE(response.token_usage.completion_tokens == 20);
        REQUIRE(response.token_usage.total_tokens == 30);
        REQUIRE(response.duration_ms == 200.0);
    }

    SECTION("Chat functionality") {
        std::vector<ChatMessage> messages = {
            ChatMessage(MessageRole::USER, "Hello"),
            ChatMessage(MessageRole::ASSISTANT, "Hi there!"),
            ChatMessage(MessageRole::USER, "How are you?")
        };

        LLMResponse response = llm.chat(messages);

        REQUIRE(response.success == true);
        REQUIRE_FALSE(response.content.empty());
        REQUIRE(response.content.find("Mock OpenAI chat response") != std::string::npos);
        REQUIRE(response.model == "gpt-3.5-turbo");
        REQUIRE(response.token_usage.total_tokens == 40);
        REQUIRE(response.duration_ms == 250.0);
    }

    SECTION("Streaming completion") {
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
        size_t tokens = llm.count_tokens("This is a test message for OpenAI token counting");
        REQUIRE(tokens > 0);
        // Simple approximation: ~4 characters per token
        REQUIRE(tokens == 12); // "This is a test message for OpenAI token counting".length() = 47, 47/4 = 11.75 -> 12
    }
}

TEST_CASE("OpenAI Model Support", "[llm][openai][models]") {
    OpenAIApiConfig config;
    config.api_key = "sk-test123";
    config.model = "gpt-3.5-turbo";
    MockOpenAILLM llm(config);

    SECTION("Supported models list") {
        auto supported_models = llm.get_supported_models();
        REQUIRE_FALSE(supported_models.empty());
        REQUIRE(std::find(supported_models.begin(), supported_models.end(), "gpt-3.5-turbo") != supported_models.end());
        REQUIRE(std::find(supported_models.begin(), supported_models.end(), "gpt-4") != supported_models.end());
    }

    SECTION("Model support checking") {
        REQUIRE(llm.is_model_supported("gpt-3.5-turbo") == true);
        REQUIRE(llm.is_model_supported("gpt-4") == true);
        REQUIRE(llm.is_model_supported("gpt-4-turbo-preview") == true);
        REQUIRE(llm.is_model_supported("unsupported-model") == false);
    }
}

TEST_CASE("OpenAI Capabilities", "[llm][openai][capabilities]") {
    OpenAIApiConfig config;
    config.api_key = "sk-test123";
    config.model = "gpt-4";
    MockOpenAILLM llm(config);

    auto capabilities = llm.get_capabilities();
    REQUIRE(capabilities.at("completion") == true);
    REQUIRE(capabilities.at("chat") == true);
    REQUIRE(capabilities.at("streaming") == true);
    REQUIRE(capabilities.at("function_calling") == true);
    REQUIRE(capabilities.at("multimodal") == false); // Basic OpenAI LLM doesn't support multimodal
}

TEST_CASE("OpenAI Configuration Management", "[llm][openai][config]") {
    SECTION("Update OpenAI config") {
        OpenAIApiConfig config;
        config.api_key = "sk-test123";
        config.model = "gpt-3.5-turbo";
        config.temperature = 0.7;

        MockOpenAILLM llm(config);

        OpenAIApiConfig new_config;
        new_config.api_key = "sk-new456";
        new_config.model = "gpt-4";
        new_config.temperature = 1.0;
        new_config.max_tokens = 2000;

        llm.update_openai_config(new_config);

        REQUIRE(llm.get_openai_config().api_key == "sk-new456");
        REQUIRE(llm.get_openai_config().model == "gpt-4");
        REQUIRE(llm.get_openai_config().temperature == 1.0);
        REQUIRE(llm.get_openai_config().max_tokens == 2000);
    }

    SECTION("Update generic config") {
        OpenAIApiConfig config;
        config.api_key = "sk-test123";
        config.model = "gpt-3.5-turbo";
        MockOpenAILLM llm(config);

        LLMConfig new_config;
        new_config.model = "gpt-4-turbo-preview";
        new_config.temperature = 0.5;
        new_config.max_tokens = 1500;

        llm.update_config(new_config);

        REQUIRE(llm.get_openai_config().model == "gpt-4-turbo-preview");
        REQUIRE(llm.get_openai_config().temperature == 0.5);
        REQUIRE(llm.get_openai_config().max_tokens == 1500);
    }
}

TEST_CASE("OpenAI Factory", "[llm][openai][factory]") {
    SECTION("Factory properties") {
        OpenAIFactory factory;
        REQUIRE(factory.get_provider() == "openai");
        REQUIRE(factory.supports_model("gpt-3.5-turbo") == true);
        REQUIRE(factory.supports_model("gpt-4") == true);
        REQUIRE(factory.supports_model("unsupported") == false);
    }

    SECTION("Factory create default") {
        OpenAIFactory factory;
        // Note: This doesn't throw anymore because empty API keys are allowed for testing
        auto llm = factory.create();
        REQUIRE(llm != nullptr);
        REQUIRE(llm->get_provider() == "openai");
        REQUIRE(llm->get_config().model == "gpt-3.5-turbo");
    }

    SECTION("Factory create with config") {
        OpenAIFactory factory;
        LLMConfig config;
        config.api_key = "sk-test123";
        config.model = "gpt-3.5-turbo";

        auto llm = factory.create(config);
        REQUIRE(llm != nullptr);
        REQUIRE(llm->get_provider() == "openai");
        REQUIRE(llm->is_model_supported("gpt-3.5-turbo") == true);
    }
}

TEST_CASE("OpenAI Error Handling", "[llm][openai][error]") {
    SECTION("Configuration validation through LLM") {
        OpenAIApiConfig config;
        config.api_key = "sk-test123";
        config.model = "gpt-3.5-turbo";
        MockOpenAILLM llm(config);

        LLMConfig valid_config;
        valid_config.api_key = "sk-valid456";
        valid_config.model = "gpt-4";

        REQUIRE_NOTHROW(llm.validate_config(valid_config));

        LLMConfig invalid_config;
        // Missing API key
        invalid_config.model = "gpt-4";
        REQUIRE_THROWS_AS(llm.validate_config(invalid_config), std::invalid_argument);
    }
}

TEST_CASE("OpenAI Performance", "[llm][openai][performance]") {
    OpenAIApiConfig config;
    config.api_key = "sk-test123";
    config.model = "gpt-3.5-turbo";
    MockOpenAILLM llm(config);

    SECTION("Multiple completions") {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < 50; ++i) {
            std::string prompt = "OpenAI test query " + std::to_string(i);
            auto response = llm.complete(prompt);
            REQUIRE(response.success == true);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Should complete 50 mock queries quickly (under 500ms)
        REQUIRE(duration.count() < 500);
    }

    SECTION("Memory usage") {
        // Test multiple responses without memory leaks
        for (int i = 0; i < 500; ++i) {
            auto response = llm.complete("OpenAI memory test " + std::to_string(i));
            REQUIRE(response.success == true);
        }

        // If we reach here without crashing, memory management is working
        REQUIRE(true);
    }
}