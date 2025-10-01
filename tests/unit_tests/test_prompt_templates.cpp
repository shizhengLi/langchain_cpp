#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include "langchain/prompts/prompt_template.hpp"
#include <stdexcept>

using namespace langchain::prompts;

TEST_CASE("PromptTemplate basic functionality", "[prompts][prompt_template]") {
    SECTION("Simple template with one variable") {
        PromptTemplate template_obj("Hello {name}!", {"name"});

        std::unordered_map<std::string, std::string> vars = {{"name", "World"}};
        std::string result = template_obj.format(vars);

        REQUIRE(result == "Hello World!");
        REQUIRE(template_obj.get_input_variables() == std::vector<std::string>{"name"});
        REQUIRE(template_obj.to_string() == "Hello {name}!");
    }

    SECTION("Template with multiple variables") {
        PromptTemplate template_obj("{greeting}, {name}! How are you {feeling}?",
                                   {"greeting", "name", "feeling"});

        std::unordered_map<std::string, std::string> vars = {
            {"greeting", "Hello"},
            {"name", "Alice"},
            {"feeling", "today"}
        };
        std::string result = template_obj.format(vars);

        REQUIRE(result == "Hello, Alice! How are you today?");
    }

    SECTION("Template with repeated variables") {
        PromptTemplate template_obj("{name} said hello to {name}", {"name"});

        std::unordered_map<std::string, std::string> vars = {{"name", "Bob"}};
        std::string result = template_obj.format(vars);

        REQUIRE(result == "Bob said hello to Bob");
    }

    SECTION("Template with no variables") {
        PromptTemplate template_obj("Static text without variables");

        std::unordered_map<std::string, std::string> vars;
        std::string result = template_obj.format(vars);

        REQUIRE(result == "Static text without variables");
        REQUIRE(template_obj.get_input_variables().empty());
    }

    SECTION("Auto-extract variables") {
        PromptTemplate template_obj("Hello {name}, welcome to {place}!");

        auto vars = template_obj.get_input_variables();
        std::sort(vars.begin(), vars.end());

        std::vector<std::string> expected = {"name", "place"};
        std::sort(expected.begin(), expected.end());

        REQUIRE(vars == expected);
    }
}

TEST_CASE("PromptTemplate error handling", "[prompts][prompt_template][errors]") {
    SECTION("Missing variable throws exception") {
        PromptTemplate template_obj("Hello {name}!", {"name"});

        std::unordered_map<std::string, std::string> vars; // Missing name
        REQUIRE_THROWS_AS(template_obj.format(vars), std::invalid_argument);
        REQUIRE_THROWS_WITH(template_obj.format(vars), "Missing value for variable: name");
    }

    SECTION("Template validation fails on mismatch") {
        REQUIRE_THROWS_AS(
            PromptTemplate("Hello {name}!", {"wrong_var"}, "f-string", true),
            std::invalid_argument
        );
    }

    SECTION("Empty template string") {
        PromptTemplate template_obj("");
        std::unordered_map<std::string, std::string> vars;
        REQUIRE(template_obj.format(vars) == "");
    }
}

TEST_CASE("PromptTemplate variable extraction", "[prompts][prompt_template]") {
    SECTION("Extract single variable") {
        auto vars = PromptTemplate::extract_variables("Hello {name}");
        REQUIRE(vars == std::vector<std::string>{"name"});
    }

    SECTION("Extract multiple variables") {
        auto vars = PromptTemplate::extract_variables("{a} and {b} and {c}");
        REQUIRE(vars == std::vector<std::string>{"a", "b", "c"});
    }

    SECTION("Extract repeated variable once") {
        auto vars = PromptTemplate::extract_variables("{x} {x} {x}");
        REQUIRE(vars == std::vector<std::string>{"x"});
    }

    SECTION("No variables in template") {
        auto vars = PromptTemplate::extract_variables("No variables here");
        REQUIRE(vars.empty());
    }

    SECTION("Complex variable names") {
        auto vars = PromptTemplate::extract_variables("Hello {user_name} and {user_id}");
        std::sort(vars.begin(), vars.end());
        std::vector<std::string> expected = {"user_id", "user_name"};
        std::sort(expected.begin(), expected.end());
        REQUIRE(vars == expected);
    }
}

TEST_CASE("FewShotPromptTemplate", "[prompts][few_shot]") {
    SECTION("Basic few-shot template") {
        std::vector<std::unordered_map<std::string, std::string>> examples = {
            {{"question", "What is 2+2?"}, {"answer", "4"}},
            {{"question", "What is 3+3?"}, {"answer", "6"}}
        };

        auto example_prompt = std::make_shared<PromptTemplate>(
            "Q: {question}\nA: {answer}",
            std::vector<std::string>{"question", "answer"}
        );

        FewShotPromptTemplate few_shot_template(
            examples,
            example_prompt,
            "\n---\n"
        );

        std::unordered_map<std::string, std::string> input = {
            {"question", "What is 5+5?"},
            {"answer", "10"}
        };

        std::string result = few_shot_template.format(input);

        REQUIRE(result.find("Q: What is 2+2?\nA: 4") != std::string::npos);
        REQUIRE(result.find("Q: What is 3+3?\nA: 6") != std::string::npos);
        REQUIRE(result.find("Q: What is 5+5?\nA: 10") != std::string::npos);
    }

    SECTION("Few-shot with prefix and suffix") {
        std::vector<std::unordered_map<std::string, std::string>> examples = {
            {{"input", "cat"}, {"output", "feline"}}
        };

        auto example_prompt = std::make_shared<PromptTemplate>(
            "Input: {input}\nOutput: {output}",
            std::vector<std::string>{"input", "output"}
        );

        FewShotPromptTemplate few_shot_template(
            examples,
            example_prompt,
            "\n",
            "Translate the following words:\n",
            "\nNow translate this:"
        );

        std::unordered_map<std::string, std::string> input = {
            {"input", "dog"},
            {"output", "canine"}
        };

        std::string result = few_shot_template.format(input);

        REQUIRE(result.find("Translate the following words:") != std::string::npos);
        REQUIRE(result.find("Input: cat\nOutput: feline") != std::string::npos);
        REQUIRE(result.find("Input: dog\nOutput: canine") != std::string::npos);
        REQUIRE(result.find("Now translate this:") != std::string::npos);
    }

    SECTION("Add examples dynamically") {
        std::vector<std::unordered_map<std::string, std::string>> examples = {
            {{"a", "1"}, {"b", "2"}}
        };

        auto example_prompt = std::make_shared<PromptTemplate>("{a} -> {b}");
        FewShotPromptTemplate few_shot_template(examples, example_prompt);

        REQUIRE(few_shot_template.example_count() == 1);

        few_shot_template.add_example({{"a", "3"}, {"b", "4"}});
        REQUIRE(few_shot_template.example_count() == 2);

        std::unordered_map<std::string, std::string> input = {{"a", "5"}, {"b", "6"}};
        std::string result = few_shot_template.format(input);

        REQUIRE(result.find("1 -> 2") != std::string::npos);
        REQUIRE(result.find("3 -> 4") != std::string::npos);
        REQUIRE(result.find("5 -> 6") != std::string::npos);
    }

    SECTION("Clear examples") {
        std::vector<std::unordered_map<std::string, std::string>> examples = {
            {{"x", "1"}}, {{"x", "2"}}
        };

        auto example_prompt = std::make_shared<PromptTemplate>("{x}");
        FewShotPromptTemplate few_shot_template(examples, example_prompt);

        REQUIRE(few_shot_template.example_count() == 2);

        few_shot_template.clear_examples();
        REQUIRE(few_shot_template.example_count() == 0);
    }
}

TEST_CASE("ChatPromptTemplate", "[prompts][chat]") {
    SECTION("Basic chat template") {
        std::vector<ChatMessage> messages = {
            ChatMessage(ChatMessageType::SYSTEM, "You are a helpful assistant."),
            ChatMessage(ChatMessageType::USER, "Hello, {name}!"),
            ChatMessage(ChatMessageType::ASSISTANT, "Hello {name}! How can I help you today?")
        };

        ChatPromptTemplate chat_template(messages, {"name"});

        std::unordered_map<std::string, std::string> vars = {{"name", "Alice"}};
        std::string result = chat_template.format(vars);

        REQUIRE(result.find("[system]: You are a helpful assistant.") != std::string::npos);
        REQUIRE(result.find("[user]: Hello, Alice!") != std::string::npos);
        REQUIRE(result.find("[assistant]: Hello, Alice! How can I help you today?") != std::string::npos);
    }

    SECTION("Add messages dynamically") {
        std::vector<ChatMessage> messages = {
            ChatMessage(ChatMessageType::USER, "Hello")
        };

        ChatPromptTemplate chat_template(messages);

        chat_template.add_message(ChatMessage(ChatMessageType::ASSISTANT, "Hi there!"));
        chat_template.add_message(ChatMessage(ChatMessageType::USER, "How are you?"));

        auto all_messages = chat_template.messages();
        REQUIRE(all_messages.size() == 3);
        REQUIRE(all_messages[0].type == ChatMessageType::USER);
        REQUIRE(all_messages[1].type == ChatMessageType::ASSISTANT);
        REQUIRE(all_messages[2].type == ChatMessageType::USER);
    }

    SECTION("Extract input variables from messages") {
        std::vector<ChatMessage> messages = {
            ChatMessage(ChatMessageType::USER, "Hello {name}, from {city}"),
            ChatMessage(ChatMessageType::ASSISTANT, "Welcome {name}!")
        };

        ChatPromptTemplate chat_template(messages);
        auto vars = chat_template.get_input_variables();
        std::sort(vars.begin(), vars.end());

        std::vector<std::string> expected = {"city", "name"};
        std::sort(expected.begin(), expected.end());

        REQUIRE(vars == expected);
    }

    SECTION("OpenAI format") {
        std::vector<ChatMessage> messages = {
            ChatMessage(ChatMessageType::SYSTEM, "System message"),
            ChatMessage(ChatMessageType::USER, "User message")
        };

        ChatPromptTemplate chat_template(messages);
        std::string openai_format = chat_template.format_openai();

        REQUIRE(openai_format.find("\"messages\"") != std::string::npos);
        REQUIRE(openai_format.find("\"role\": \"system\"") != std::string::npos);
        REQUIRE(openai_format.find("\"role\": \"user\"") != std::string::npos);
        REQUIRE(openai_format.find("\"content\": \"System message\"") != std::string::npos);
        REQUIRE(openai_format.find("\"content\": \"User message\"") != std::string::npos);
    }

    SECTION("Message type conversion") {
        std::vector<ChatMessage> messages = {
            ChatMessage(ChatMessageType::SYSTEM, "System"),
            ChatMessage(ChatMessageType::USER, "User"),
            ChatMessage(ChatMessageType::ASSISTANT, "Assistant"),
            ChatMessage(ChatMessageType::FUNCTION, "Function"),
            ChatMessage(ChatMessageType::TOOL, "Tool")
        };

        ChatPromptTemplate chat_template(messages);
        std::string result = chat_template.format_generic();

        REQUIRE(result.find("[system]: System") != std::string::npos);
        REQUIRE(result.find("[user]: User") != std::string::npos);
        REQUIRE(result.find("[assistant]: Assistant") != std::string::npos);
        REQUIRE(result.find("[function]: Function") != std::string::npos);
        REQUIRE(result.find("[tool]: Tool") != std::string::npos);
    }
}

TEST_CASE("PipelinePromptTemplate", "[prompts][pipeline]") {
    SECTION("Basic pipeline") {
        auto step1 = std::make_shared<PromptTemplate>("Step 1: {input}", std::vector<std::string>{"input"});
        auto step2 = std::make_shared<PromptTemplate>("Step 2: {step_1_result}", std::vector<std::string>{"step_1_result"});
        auto step3 = std::make_shared<PromptTemplate>("Final: {step_2_result}", std::vector<std::string>{"step_2_result"});

        std::vector<std::shared_ptr<BasePromptTemplate>> pipeline = {step1, step2, step3};
        PipelinePromptTemplate pipeline_template(pipeline, {"input"});

        std::unordered_map<std::string, std::string> vars = {{"input", "Start"}};
        std::string result = pipeline_template.format(vars);

        REQUIRE(result.find("Step 1: Start") != std::string::npos);
        REQUIRE(result.find("Step 2: Step 1: Start") != std::string::npos);
        REQUIRE(result.find("Final: Step 2: Step 1: Start") != std::string::npos);
    }

    SECTION("Add template to pipeline") {
        auto step1 = std::make_shared<PromptTemplate>("First: {input}");
        PipelinePromptTemplate pipeline_template({step1}, {"input"});

        auto step2 = std::make_shared<PromptTemplate>("Second: {step_1_result}");
        pipeline_template.add_template(step2);

        std::unordered_map<std::string, std::string> vars = {{"input", "test"}};
        std::string result = pipeline_template.format(vars);

        REQUIRE(result.find("First: test") != std::string::npos);
        REQUIRE(result.find("Second: First: test") != std::string::npos);
    }

    SECTION("Clear pipeline") {
        auto step1 = std::make_shared<PromptTemplate>("Step 1: {input}");
        auto step2 = std::make_shared<PromptTemplate>("Step 2: {step_1_result}");

        PipelinePromptTemplate pipeline_template({step1, step2}, {"input"});
        pipeline_template.clear_pipeline();

        REQUIRE_THROWS_AS(pipeline_template.format({{"input", "test"}}), std::logic_error);
    }

    SECTION("Empty pipeline throws exception") {
        REQUIRE_THROWS_AS(
            PipelinePromptTemplate({}, {}),
            std::invalid_argument
        );
    }
}

TEST_CASE("Prompt template utilities", "[prompts][utils]") {
    SECTION("Simple template creation") {
        auto template_obj = utils::simple_template("Hello {name}!");

        std::unordered_map<std::string, std::string> vars = {{"name", "World"}};
        std::string result = template_obj->format(vars);

        REQUIRE(result == "Hello World!");
    }

    SECTION("Few-shot template creation") {
        std::vector<std::unordered_map<std::string, std::string>> examples = {
            {{"input", "1+1"}, {"output", "2"}}
        };

        auto few_shot_template = utils::few_shot_template(examples, "Input: {input}\nOutput: {output}");

        std::unordered_map<std::string, std::string> vars = {{"input", "2+2"}, {"output", "4"}};
        std::string result = few_shot_template->format(vars);

        REQUIRE(result.find("Input: 1+1\nOutput: 2") != std::string::npos);
        REQUIRE(result.find("Input: 2+2\nOutput: 4") != std::string::npos);
    }

    SECTION("Chat template creation") {
        std::vector<ChatMessage> messages = {
            ChatMessage(ChatMessageType::USER, "Hello {name}")
        };

        auto chat_template = utils::chat_template(messages);

        std::unordered_map<std::string, std::string> vars = {{"name", "Alice"}};
        std::string result = chat_template->format(vars);

        REQUIRE(result.find("[user]: Hello Alice") != std::string::npos);
    }

    SECTION("Input validation") {
        std::vector<std::string> template_vars = {"name", "age"};
        std::unordered_map<std::string, std::string> input = {{"name", "Alice"}}; // Missing age

        REQUIRE_THROWS_AS(
            utils::validate_input_variables(template_vars, input),
            std::invalid_argument
        );
        REQUIRE_THROWS_WITH(
            utils::validate_input_variables(template_vars, input),
            "Missing required input variable: age"
        );
    }

    SECTION("Template sanitization") {
        std::string dirty_template = "Hello\n\n\n\n{name}!    \t\t\n\n";
        std::string clean_template = utils::sanitize_template(dirty_template);

        REQUIRE(clean_template == "Hello\n\n{name}!");
    }

    SECTION("Valid input passes validation") {
        std::vector<std::string> template_vars = {"name", "age"};
        std::unordered_map<std::string, std::string> input = {
            {"name", "Alice"},
            {"age", "25"},
            {"extra", "data"} // Extra data should be fine
        };

        REQUIRE_NOTHROW(utils::validate_input_variables(template_vars, input));
    }
}

TEST_CASE("Template format validation", "[prompts][validation]") {
    SECTION("Valid f-string format") {
        REQUIRE_NOTHROW(PromptTemplate("Hello {name}!", {"name"}, "f-string"));
    }

    SECTION("Different template formats") {
        REQUIRE_NOTHROW(PromptTemplate("Hello {{name}}!", {"name"}, "jinja2"));
    }

    SECTION("Template with validation disabled") {
        REQUIRE_NOTHROW(PromptTemplate("Hello {name}!", {"wrong_var"}, "f-string", false));
    }

    SECTION("Complex nested template") {
        std::string complex_template = R"(
System: You are a helpful assistant.
User: {user_question}
Context: {context}
Instructions: {instructions}
Assistant: I'll help you with that.
)";

        PromptTemplate template_obj(
            complex_template,
            {"user_question", "context", "instructions"},
            "f-string",
            true
        );

        std::unordered_map<std::string, std::string> vars = {
            {"user_question", "What is AI?"},
            {"context", "Artificial Intelligence context"},
            {"instructions", "Provide a detailed answer"}
        };

        std::string result = template_obj.format(vars);

        REQUIRE(result.find("What is AI?") != std::string::npos);
        REQUIRE(result.find("Artificial Intelligence context") != std::string::npos);
        REQUIRE(result.find("Provide a detailed answer") != std::string::npos);
    }
}