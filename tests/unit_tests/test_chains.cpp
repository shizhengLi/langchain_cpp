#include "langchain/chains/base_chain.hpp"
#include "langchain/chains/llm_chain.hpp"
#include "langchain/chains/sequential_chain.hpp"
#include "langchain/llm/openai_llm.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <memory>
#include <string>

using namespace langchain::chains;
using namespace langchain::llm;

// Mock LLM for testing
class MockChainLLM : public BaseLLM {
public:
    explicit MockChainLLM(const std::string& response_prefix = "Mock response: ")
        : response_prefix_(response_prefix) {}

    LLMResponse complete(const std::string& prompt, const std::optional<LLMConfig>& config = std::nullopt) override {
        LLMResponse response;
        response.content = response_prefix_ + prompt;
        response.model = "mock-model";
        response.token_usage = TokenUsage{10, 20, 30};
        response.duration_ms = 100.0;
        response.success = true;
        return response;
    }

    LLMResponse chat(const std::vector<ChatMessage>& messages, const std::optional<LLMConfig>& config = std::nullopt) override {
        return complete("chat mock response", config);
    }

    LLMResponse stream_complete(const std::string& prompt, std::function<void(const std::string&)> callback,
                              const std::optional<LLMConfig>& config = std::nullopt) override {
        return complete(prompt, config);
    }

    LLMResponse stream_chat(const std::vector<ChatMessage>& messages, std::function<void(const std::string&)> callback,
                           const std::optional<LLMConfig>& config = std::nullopt) override {
        return chat(messages, config);
    }

    size_t count_tokens(const std::string& text) const override {
        return text.length() / 4;
    }

    std::vector<std::string> get_supported_models() const override {
        return {"mock-model"};
    }

    bool is_model_supported(const std::string& model) const override {
        return model == "mock-model";
    }

    std::string get_provider() const override {
        return "mock";
    }

    const LLMConfig& get_config() const override {
        static LLMConfig config;
        config.model = "mock-model";
        return config;
    }

    void update_config(const LLMConfig& new_config) override {
        // Mock implementation
    }

    void validate_config(const LLMConfig& config) const override {
        // Mock implementation - always valid
    }

    std::unordered_map<std::string, bool> get_capabilities() const override {
        return {{"completion", true}, {"chat", true}, {"streaming", true}};
    }

private:
    std::string response_prefix_;
};

TEST_CASE("ChainInput", "[chains][input]") {
    SECTION("Basic operations") {
        ChainInput input;
        input.set("key1", "value1");
        input.set("key2", "value2");

        REQUIRE(input.has("key1"));
        REQUIRE(input.has("key2"));
        REQUIRE_FALSE(input.has("key3"));

        REQUIRE(input.get("key1") == "value1");
        REQUIRE(input.get("key2") == "value2");
        REQUIRE(input.get("key3", "default") == "default");
    }

    SECTION("Empty input") {
        ChainInput input;
        REQUIRE_FALSE(input.has("any_key"));
        REQUIRE(input.get("any_key", "default") == "default");
    }
}

TEST_CASE("ChainOutput", "[chains][output]") {
    SECTION("Success output") {
        ChainOutput output;
        output.set("result", "success");
        output.success = true;

        REQUIRE(output.success);
        REQUIRE(output.get("result") == "success");
        REQUIRE_FALSE(output.error_message.has_value());
    }

    SECTION("Error output") {
        ChainOutput output;
        output.success = false;
        output.error_message = "Test error";

        REQUIRE_FALSE(output.success);
        REQUIRE(output.error_message.value() == "Test error");
    }
}

TEST_CASE("ChainConfig", "[chains][config]") {
    SECTION("Valid configuration") {
        ChainConfig config;
        config.verbose = true;
        config.timeout = std::chrono::milliseconds(5000);
        config.max_retries = 3;

        REQUIRE_NOTHROW(config.validate());
    }

    SECTION("Invalid timeout") {
        ChainConfig config;
        config.timeout = std::chrono::milliseconds(-100);

        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
    }

    SECTION("Too many retries") {
        ChainConfig config;
        config.max_retries = 15;

        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
    }
}

TEST_CASE("LLMChainConfig", "[chains][llm][config]") {
    SECTION("Valid configuration") {
        LLMChainConfig config;
        config.prompt_template = "Hello {name}!";
        config.input_key = "name";
        config.output_key = "greeting";

        REQUIRE_NOTHROW(config.validate());
    }

    SECTION("Empty prompt template") {
        LLMChainConfig config;
        config.prompt_template = "";
        config.input_key = "name";
        config.output_key = "greeting";

        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
    }

    SECTION("Empty input key") {
        LLMChainConfig config;
        config.prompt_template = "Hello {name}!";
        config.input_key = std::string("");
        config.output_key = "greeting";

        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
    }

    SECTION("Empty output key") {
        LLMChainConfig config;
        config.prompt_template = "Hello {name}!";
        config.input_key = "name";
        config.output_key = std::string("");

        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
    }
}

TEST_CASE("LLMChain", "[chains][llm]") {
    auto llm = std::make_shared<MockChainLLM>("LLMChain response: ");

    SECTION("Basic execution") {
        LLMChainConfig config;
        config.prompt_template = "Process: {input}";
        config.input_key = "input";
        config.output_key = "output";

        LLMChain chain(llm, config);

        ChainInput input;
        input.set("input", "test data");

        auto output = chain.run(input);

        REQUIRE(output.success);
        REQUIRE(output.has("output"));
        REQUIRE(output.get("output") == "LLMChain response: Process: test data");
        REQUIRE(output.execution_time.has_value());
    }

    SECTION("Missing input key") {
        LLMChainConfig config;
        config.prompt_template = "Process: {input}";
        config.input_key = "input";
        config.output_key = "output";

        LLMChain chain(llm, config);

        ChainInput input;
        input.set("wrong_key", "test data");

        auto output = chain.run(input);

        REQUIRE_FALSE(output.success);
        REQUIRE(output.error_message.has_value());
    }

    SECTION("Variable substitution") {
        LLMChainConfig config;
        config.prompt_template = "Hello {name}, you are {age} years old!";
        config.input_key = "name";
        config.output_key = "greeting";

        LLMChain chain(llm, config);

        ChainInput input;
        input.set("name", "Alice");
        input.set("age", "25");

        auto output = chain.run(input);

        REQUIRE(output.success);
        REQUIRE(output.get("greeting") == "LLMChain response: Hello Alice, you are 25 years old!");
    }

    SECTION("Strip whitespace") {
        LLMChainConfig config;
        config.prompt_template = "  {input}  ";
        config.input_key = "input";
        config.output_key = "output";
        config.strip_whitespace = true;

        LLMChain chain(llm, config);

        ChainInput input;
        input.set("input", "test");

        auto output = chain.run(input);

        REQUIRE(output.success);
        // Mock response should have whitespace stripped
        REQUIRE_FALSE(output.get("output").empty());
    }

    SECTION("Return intermediate steps") {
        LLMChainConfig config;
        config.prompt_template = "Process: {input}";
        config.input_key = "input";
        config.output_key = "output";
        config.return_intermediate_steps = true;

        LLMChain chain(llm, config);

        ChainInput input;
        input.set("input", "test");

        auto output = chain.run(input);

        REQUIRE(output.success);
        REQUIRE(output.has("output"));
        REQUIRE(output.has("prompt"));
        REQUIRE(output.has("llm_response"));
        REQUIRE(output.get("prompt") == "Process: test");
    }

    SECTION("Input and output keys") {
        LLMChainConfig config;
        config.prompt_template = "Process: {data} and {info}";
        config.input_key = "data";
        config.output_key = "result";

        LLMChain chain(llm, config);

        auto input_keys = chain.get_input_keys();
        auto output_keys = chain.get_output_keys();

        REQUIRE(input_keys.size() >= 2);
        REQUIRE(std::find(input_keys.begin(), input_keys.end(), "data") != input_keys.end());
        REQUIRE(std::find(input_keys.begin(), input_keys.end(), "info") != input_keys.end());
        REQUIRE(std::find(input_keys.begin(), input_keys.end(), "data") != input_keys.end());

        REQUIRE(output_keys.size() >= 1);
        REQUIRE(std::find(output_keys.begin(), output_keys.end(), "result") != output_keys.end());
    }

    SECTION("Null LLM") {
        LLMChainConfig config;
        config.prompt_template = "Hello {name}!";
        REQUIRE_THROWS_AS(LLMChain(nullptr, config), std::invalid_argument);
    }
}

TEST_CASE("LLMChain prompt formatting", "[chains][llm][prompt]") {
    auto llm = std::make_shared<MockChainLLM>();
    LLMChainConfig config;
    config.prompt_template = "Hello {name}!";
    LLMChain chain(llm, config);

    SECTION("Simple template") {
        ChainInput input;
        input.set("name", "World");
        std::string result = chain.format_prompt("Hello {name}!", input);
        REQUIRE(result == "Hello World!");
    }

    SECTION("Multiple variables") {
        ChainInput input;
        input.set("a", "1");
        input.set("b", "2");
        input.set("c", "3");
        std::string result = chain.format_prompt("{a} {b} {c}", input);
        REQUIRE(result == "1 2 3");
    }

    SECTION("Missing variable") {
        ChainInput input;
        input.set("other", "value");
        std::string result = chain.format_prompt("Hello {name}!", input);
        REQUIRE(result == "Hello !");
    }

    SECTION("Repeated variable") {
        ChainInput input;
        input.set("x", "test");
        std::string result = chain.format_prompt("{x} {x} {x}", input);
        REQUIRE(result == "test test test");
    }

    SECTION("No variables") {
        ChainInput input;
        std::string result = chain.format_prompt("Static text", input);
        REQUIRE(result == "Static text");
    }

    SECTION("Complex template") {
        ChainInput input;
        input.set("system", "You are helpful");
        input.set("user", "Hello");
        std::string result = chain.format_prompt(
            "System: {system}\nUser: {user}\nAssistant:",
            input
        );
        REQUIRE(result == "System: You are helpful\nUser: Hello\nAssistant:");
    }
}

TEST_CASE("SequentialChainConfig", "[chains][sequential][config]") {
    SECTION("Valid configuration") {
        SequentialChainConfig config;
        config.output_key = "final_result";

        REQUIRE_NOTHROW(config.validate());
    }

    SECTION("Empty output key") {
        SequentialChainConfig config;
        config.output_key = "";

        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
    }
}

TEST_CASE("SequentialChain", "[chains][sequential]") {
    SECTION("Empty chain") {
        SequentialChain chain;

        ChainInput input;
        input.set("test", "value");

        auto output = chain.run(input);

        REQUIRE_FALSE(output.success);
        REQUIRE(output.error_message.value() == "SequentialChain has no chains to execute");
    }

    SECTION("Single chain") {
        auto llm = std::make_shared<MockChainLLM>("Chain1: ");
        LLMChainConfig config;
        config.prompt_template = "Process: {input}";
        config.input_key = "input";
        config.output_key = "result";
        auto llm_chain = std::make_shared<LLMChain>(llm, config);

        SequentialChain seq_chain;
        seq_chain.add_chain(llm_chain);

        ChainInput input;
        input.set("input", "test");

        auto output = seq_chain.run(input);

        REQUIRE(output.success);
        REQUIRE(output.execution_time.has_value());
    }

    SECTION("Multiple chains") {
        auto llm1 = std::make_shared<MockChainLLM>("Chain1: ");
        auto llm2 = std::make_shared<MockChainLLM>("Chain2: ");

        LLMChainConfig config1;
        config1.prompt_template = "Step 1: {input}";
        config1.input_key = "input";
        config1.output_key = "step1";

        LLMChainConfig config2;
        config2.prompt_template = "Step 2: {step1}";
        config2.input_key = "step1";
        config2.output_key = "step2";

        auto chain1 = std::make_shared<LLMChain>(llm1, config1);
        auto chain2 = std::make_shared<LLMChain>(llm2, config2);

        SequentialChain seq_chain;
        seq_chain.add_chain(chain1);
        seq_chain.add_chain(chain2);

        ChainInput input;
        input.set("input", "initial");

        auto output = seq_chain.run(input);

        REQUIRE(output.success);
        REQUIRE(output.has("step2"));
        REQUIRE(output.get("step2").find("Chain2: Step 2: Chain1: Step 1: initial") != std::string::npos);
    }

    SECTION("Return all outputs") {
        auto llm1 = std::make_shared<MockChainLLM>("Chain1: ");
        auto llm2 = std::make_shared<MockChainLLM>("Chain2: ");

        LLMChainConfig config1;
        config1.prompt_template = "Process {input}";
        config1.input_key = "input";
        config1.output_key = "result1";

        LLMChainConfig config2;
        config2.prompt_template = "Process {result1}";
        config2.input_key = "result1";
        config2.output_key = "result2";

        auto chain1 = std::make_shared<LLMChain>(llm1, config1);
        auto chain2 = std::make_shared<LLMChain>(llm2, config2);

        SequentialChainConfig seq_config;
        seq_config.return_all_outputs = true;

        SequentialChain seq_chain(seq_config);
        seq_chain.add_chain(chain1);
        seq_chain.add_chain(chain2);

        ChainInput input;
        input.set("input", "test");

        auto output = seq_chain.run(input);

        REQUIRE(output.success);
        REQUIRE(output.has("result1"));
        REQUIRE(output.has("result2"));
    }

    SECTION("Chain management") {
        SequentialChain chain;
        auto llm = std::make_shared<MockChainLLM>();
        LLMChainConfig config;
        config.prompt_template = "Test {input}";
        auto llm_chain = std::make_shared<LLMChain>(llm, config);

        REQUIRE(chain.get_chain_count() == 0);

        chain.add_chain(llm_chain);
        REQUIRE(chain.get_chain_count() == 1);

        chain.remove_chain(0);
        REQUIRE(chain.get_chain_count() == 0);

        REQUIRE_THROWS_AS(chain.remove_chain(0), std::out_of_range);
        REQUIRE_THROWS_AS(chain.get_chain(0), std::out_of_range);
    }

    SECTION("Null chain") {
        SequentialChain chain;
        REQUIRE_THROWS_AS(chain.add_chain(nullptr), std::invalid_argument);
    }

    SECTION("Input and output keys") {
        auto llm1 = std::make_shared<MockChainLLM>();
        auto llm2 = std::make_shared<MockChainLLM>();

        LLMChainConfig config1;
        config1.prompt_template = "{input}";
        config1.input_key = "input";
        config1.output_key = "output1";

        LLMChainConfig config2;
        config2.prompt_template = "{output1}";
        config2.input_key = "output1";
        config2.output_key = "output2";

        auto chain1 = std::make_shared<LLMChain>(llm1, config1);
        auto chain2 = std::make_shared<LLMChain>(llm2, config2);

        SequentialChain seq_chain;
        seq_chain.add_chain(chain1);
        seq_chain.add_chain(chain2);

        auto input_keys = seq_chain.get_input_keys();
        auto output_keys = seq_chain.get_output_keys();

        REQUIRE(input_keys.size() >= 1);
        REQUIRE(std::find(input_keys.begin(), input_keys.end(), "input") != input_keys.end());

        REQUIRE(output_keys.size() >= 1);
        REQUIRE(std::find(output_keys.begin(), output_keys.end(), "output") != output_keys.end());
    }
}

TEST_CASE("ChainRegistry", "[chains][registry]") {
    SECTION("Singleton pattern") {
        auto& registry1 = ChainRegistry::instance();
        auto& registry2 = ChainRegistry::instance();

        REQUIRE(&registry1 == &registry2);
    }

    SECTION("Factory registration") {
        auto& registry = ChainRegistry::instance();
        auto llm = std::make_shared<MockChainLLM>();
        auto factory = std::make_unique<LLMChainFactory>(llm);

        registry.register_factory(std::move(factory));

        REQUIRE(registry.supports_chain_type("llm"));
        REQUIRE_FALSE(registry.supports_chain_type("unknown"));

        auto types = registry.get_available_types();
        REQUIRE(std::find(types.begin(), types.end(), "llm") != types.end());
    }

    SECTION("Chain creation") {
        auto& registry = ChainRegistry::instance();
        auto llm = std::make_shared<MockChainLLM>();
        auto factory = std::make_unique<LLMChainFactory>(llm);

        registry.register_factory(std::move(factory));

        auto chain = registry.create("llm");
        REQUIRE(chain != nullptr);
        REQUIRE(chain->get_output_keys().size() >= 1);

        LLMChainConfig config;
        config.prompt_template = "Test {input}";
        auto configured_chain = registry.create("llm", config);
        REQUIRE(configured_chain != nullptr);
    }

    SECTION("Unknown chain type") {
        auto& registry = ChainRegistry::instance();

        REQUIRE_THROWS_AS(registry.create("unknown"), std::invalid_argument);
        REQUIRE_THROWS_AS(registry.create("unknown", ChainConfig{}), std::invalid_argument);
    }
}

TEST_CASE("ChainFactory", "[chains][factory]") {
    SECTION("LLMChainFactory") {
        auto llm = std::make_shared<MockChainLLM>();
        LLMChainFactory factory(llm);

        REQUIRE(factory.get_chain_type() == "llm");
        REQUIRE(factory.supports_chain_type("llm"));
        REQUIRE_FALSE(factory.supports_chain_type("sequential"));

        auto chain = factory.create();
        REQUIRE(chain != nullptr);

        LLMChainConfig config;
        config.prompt_template = "Test {input}";
        auto configured_chain = factory.create(config);
        REQUIRE(configured_chain != nullptr);
    }

    SECTION("SequentialChainFactory") {
        SequentialChainFactory factory;

        REQUIRE(factory.get_chain_type() == "sequential");
        REQUIRE(factory.supports_chain_type("sequential"));
        REQUIRE_FALSE(factory.supports_chain_type("llm"));

        auto chain = factory.create();
        REQUIRE(chain != nullptr);

        SequentialChainConfig config;
        config.return_all_outputs = true;
        auto configured_chain = factory.create(config);
        REQUIRE(configured_chain != nullptr);
    }
}

TEST_CASE("Chain Performance", "[chains][performance]") {
    SECTION("LLMChain performance") {
        auto llm = std::make_shared<MockChainLLM>();
        LLMChainConfig config;
        config.prompt_template = "Process: {input}";
        config.input_key = "input";
        config.output_key = "output";

        LLMChain chain(llm, config);

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < 100; ++i) {
            ChainInput input;
            input.set("input", "test " + std::to_string(i));
            auto output = chain.run(input);
            REQUIRE(output.success);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Should complete 100 executions quickly
        REQUIRE(duration.count() < 1000);
    }

    SECTION("SequentialChain performance") {
        auto llm1 = std::make_shared<MockChainLLM>("Chain1: ");
        auto llm2 = std::make_shared<MockChainLLM>("Chain2: ");

        LLMChainConfig config1;
        config1.prompt_template = "Step 1: {input}";
        config1.input_key = "input";
        config1.output_key = "step1";

        LLMChainConfig config2;
        config2.prompt_template = "Step 2: {step1}";
        config2.input_key = "step1";
        config2.output_key = "step2";

        auto chain1 = std::make_shared<LLMChain>(llm1, config1);
        auto chain2 = std::make_shared<LLMChain>(llm2, config2);

        SequentialChain seq_chain;
        seq_chain.add_chain(chain1);
        seq_chain.add_chain(chain2);

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < 50; ++i) {
            ChainInput input;
            input.set("input", "test " + std::to_string(i));
            auto output = seq_chain.run(input);
            REQUIRE(output.success);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Should complete 50 sequential executions quickly
        REQUIRE(duration.count() < 1000);
    }
}