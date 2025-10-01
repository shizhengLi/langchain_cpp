#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include "langchain/memory/memory.hpp"
#include <thread>
#include <chrono>

using namespace langchain::memory;

TEST_CASE("BufferMemory basic functionality", "[memory][buffer]") {
    SECTION("Add and retrieve messages") {
        BufferMemory memory;

        ChatMessage msg1(ChatMessage::Type::HUMAN, "Hello");
        ChatMessage msg2(ChatMessage::Type::AI, "Hi there!");

        memory.add_message(msg1);
        memory.add_message(msg2);

        auto messages = memory.get_messages();
        REQUIRE(messages.size() == 2);
        REQUIRE(messages[0].content == "Hello");
        REQUIRE(messages[1].content == "Hi there!");
        REQUIRE(messages[0].type == ChatMessage::Type::HUMAN);
        REQUIRE(messages[1].type == ChatMessage::Type::AI);
    }

    SECTION("Get context string") {
        BufferMemory memory;

        memory.add_message(ChatMessage(ChatMessage::Type::HUMAN, "Hello"));
        memory.add_message(ChatMessage(ChatMessage::Type::AI, "Hi there!"));

        std::string context = memory.get_context();
        REQUIRE(context.find("[Human]: Hello") != std::string::npos);
        REQUIRE(context.find("[AI]: Hi there!") != std::string::npos);
    }

    SECTION("Message limits") {
        MemoryConfig config;
        config.max_messages = 2;

        BufferMemory memory(config);

        memory.add_message(ChatMessage(ChatMessage::Type::HUMAN, "Message 1"));
        memory.add_message(ChatMessage(ChatMessage::Type::AI, "Response 1"));
        memory.add_message(ChatMessage(ChatMessage::Type::HUMAN, "Message 2"));
        memory.add_message(ChatMessage(ChatMessage::Type::AI, "Response 2"));
        memory.add_message(ChatMessage(ChatMessage::Type::HUMAN, "Message 3"));

        auto messages = memory.get_messages();
        REQUIRE(messages.size() == 2);
        REQUIRE(messages[0].content == "Response 2");
        REQUIRE(messages[1].content == "Message 3");
    }

    SECTION("Clear memory") {
        BufferMemory memory;

        memory.add_message(ChatMessage(ChatMessage::Type::HUMAN, "Test"));
        REQUIRE(memory.get_messages().size() == 1);

        memory.clear();
        REQUIRE(memory.get_messages().empty());
    }

    SECTION("Memory statistics") {
        BufferMemory memory;

        memory.add_message(ChatMessage(ChatMessage::Type::HUMAN, "Test 1"));
        memory.add_message(ChatMessage(ChatMessage::Type::AI, "Test 2"));

        auto stats = memory.get_stats();
        REQUIRE(stats["message_count"] == 2);
        REQUIRE(stats["max_messages"] == 100); // Default value
    }
}

TEST_CASE("TokenBufferMemory", "[memory][token_buffer]") {
    SECTION("Token counting") {
        TokenBufferMemory memory(20); // 20 token limit

        // Add short messages
        memory.add_message(ChatMessage(ChatMessage::Type::HUMAN, "Hello"));
        memory.add_message(ChatMessage(ChatMessage::Type::AI, "Hi!"));

        auto stats = memory.get_stats();
        REQUIRE(stats["current_tokens"] > 0);
        REQUIRE(stats["current_tokens"] <= 20);
    }

    SECTION("Token limit enforcement") {
        TokenBufferMemory memory(10); // Very low limit

        // Add messages that exceed token limit
        memory.add_message(ChatMessage(ChatMessage::Type::HUMAN, "This is a very long message that exceeds the token limit"));
        memory.add_message(ChatMessage(ChatMessage::Type::AI, "This is another long message"));

        // Should have removed older messages to stay within limit
        auto stats = memory.get_stats();
        REQUIRE(stats["current_tokens"] <= 10);
    }

    SECTION("Custom token counter") {
        auto counter = [](const std::string& text) -> size_t {
            return text.length(); // 1 token per character
        };

        TokenBufferMemory memory(15, counter);

        memory.add_message(ChatMessage(ChatMessage::Type::HUMAN, "12345")); // 5 tokens
        memory.add_message(ChatMessage(ChatMessage::Type::AI, "67890"));   // 5 tokens

        auto stats = memory.get_stats();
        REQUIRE(stats["current_tokens"] == 10);
    }
}

TEST_CASE("SummaryMemory", "[memory][summary]") {
    SECTION("Basic summarization") {
        SummaryMemory memory;

        // Add messages below summary threshold
        memory.add_message(ChatMessage(ChatMessage::Type::HUMAN, "Hello"));
        memory.add_message(ChatMessage(ChatMessage::Type::AI, "Hi!"));

        auto messages = memory.get_messages();
        REQUIRE(messages.size() == 2); // No summarization yet
    }

    SECTION("Memory statistics") {
        SummaryMemory memory;

        memory.add_message(ChatMessage(ChatMessage::Type::HUMAN, "Test"));

        auto stats = memory.get_stats();
        REQUIRE(stats["recent_message_count"] == 1);
        REQUIRE(stats["summary_length"] == 0); // No summary yet
    }

    SECTION("Clear memory") {
        SummaryMemory memory;

        memory.add_message(ChatMessage(ChatMessage::Type::HUMAN, "Test"));
        memory.clear();

        auto messages = memory.get_messages();
        REQUIRE(messages.empty());
    }
}

TEST_CASE("MemoryFactory", "[memory][factory]") {
    SECTION("Create buffer memory") {
        MemoryConfig config;
        config.max_messages = 5;

        auto memory = MemoryFactory::create_buffer_memory(config);
        REQUIRE(memory != nullptr);

        // Test it's actually a BufferMemory
        memory->add_message(ChatMessage(ChatMessage::Type::HUMAN, "Test"));
        auto messages = memory->get_messages();
        REQUIRE(messages.size() == 1);
    }

    SECTION("Create token buffer memory") {
        auto memory = MemoryFactory::create_token_buffer_memory(100);
        REQUIRE(memory != nullptr);
    }

    SECTION("Create summary memory") {
        MemoryConfig config;
        auto memory = MemoryFactory::create_summary_memory(config);
        REQUIRE(memory != nullptr);
    }

    SECTION("Create memory by type") {
        MemoryConfig config;
        auto memory = MemoryFactory::create_memory(MemoryFactory::MemoryType::BUFFER, config);
        REQUIRE(memory != nullptr);
    }
}

TEST_CASE("ChatMessage functionality", "[memory][chat_message]") {
    SECTION("Message types") {
        ChatMessage human_msg(ChatMessage::Type::HUMAN, "Human message");
        ChatMessage ai_msg(ChatMessage::Type::AI, "AI message");
        ChatMessage system_msg(ChatMessage::Type::SYSTEM, "System message");
        ChatMessage function_msg(ChatMessage::Type::FUNCTION, "Function call");
        ChatMessage tool_msg(ChatMessage::Type::TOOL, "Tool result");

        REQUIRE(human_msg.type == ChatMessage::Type::HUMAN);
        REQUIRE(ai_msg.type == ChatMessage::Type::AI);
        REQUIRE(system_msg.type == ChatMessage::Type::SYSTEM);
        REQUIRE(function_msg.type == ChatMessage::Type::FUNCTION);
        REQUIRE(tool_msg.type == ChatMessage::Type::TOOL);
    }

    SECTION("Message with additional data") {
        ChatMessage msg(ChatMessage::Type::FUNCTION, "Call function", "function_data");

        REQUIRE(msg.type == ChatMessage::Type::FUNCTION);
        REQUIRE(msg.content == "Call function");
        REQUIRE(msg.additional_data.has_value());
        REQUIRE(msg.additional_data.value() == "function_data");
    }

    SECTION("Timestamp functionality") {
        auto before = std::chrono::system_clock::now();
        ChatMessage msg(ChatMessage::Type::HUMAN, "Test");
        auto after = std::chrono::system_clock::now();

        REQUIRE(msg.timestamp >= before);
        REQUIRE(msg.timestamp <= after);
    }
}

TEST_CASE("Memory utility functions", "[memory][utils]") {
    SECTION("Message type to string") {
        REQUIRE(utils::message_type_to_string(ChatMessage::Type::HUMAN) == "Human");
        REQUIRE(utils::message_type_to_string(ChatMessage::Type::AI) == "AI");
        REQUIRE(utils::message_type_to_string(ChatMessage::Type::SYSTEM) == "System");
        REQUIRE(utils::message_type_to_string(ChatMessage::Type::FUNCTION) == "Function");
        REQUIRE(utils::message_type_to_string(ChatMessage::Type::TOOL) == "Tool");
    }

    SECTION("Simple token counter") {
        auto counter = utils::create_simple_token_counter();

        REQUIRE(counter("") == 0);
        REQUIRE(counter("1234") == 1); // 4 chars / 4 = 1
        REQUIRE(counter("12345") == 2); // 5 chars / 4 = 1.25 -> 2
        REQUIRE(counter("12345678") == 2); // 8 chars / 4 = 2
    }

    SECTION("Basic summarizer") {
        auto summarizer = utils::create_basic_summarizer();

        std::vector<ChatMessage> messages = {
            ChatMessage(ChatMessage::Type::HUMAN, "Hello"),
            ChatMessage(ChatMessage::Type::AI, "Hi!")
        };

        std::string summary = summarizer(messages);
        REQUIRE(summary.find("2 messages") != std::string::npos);
    }
}

TEST_CASE("Memory configuration validation", "[memory][config]") {
    SECTION("Valid configuration") {
        MemoryConfig config;
        config.max_messages = 10;
        config.max_age = std::chrono::seconds(60);

        REQUIRE_NOTHROW(config.validate());
    }

    SECTION("Invalid max_messages") {
        MemoryConfig config;
        config.max_messages = 0;

        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
        REQUIRE_THROWS_WITH(config.validate(), "max_messages must be greater than 0");
    }

    SECTION("Invalid max_age") {
        MemoryConfig config;
        config.max_age = std::chrono::seconds(0);

        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
        REQUIRE_THROWS_WITH(config.validate(), "max_age must be positive");
    }
}

TEST_CASE("Memory age-based cleanup", "[memory][age]") {
    SECTION("Age limit enforcement") {
        MemoryConfig config;
        config.max_age = std::chrono::seconds(1);

        BufferMemory memory(config);

        memory.add_message(ChatMessage(ChatMessage::Type::HUMAN, "Old message"));

        // Wait for message to expire
        std::this_thread::sleep_for(std::chrono::milliseconds(1100));

        memory.add_message(ChatMessage(ChatMessage::Type::AI, "New message"));

        auto messages = memory.get_messages();
        REQUIRE(messages.size() == 1);
        REQUIRE(messages[0].content == "New message");
    }
}