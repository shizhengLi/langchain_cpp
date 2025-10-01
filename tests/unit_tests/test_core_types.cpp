#include <catch2/catch_all.hpp>
#include "langchain/core/types.hpp"
#include <chrono>
#include <thread>

using namespace langchain;
using Catch::Approx;

TEST_CASE("Document - Basic Operations", "[core][types][document]") {
    SECTION("Default construction") {
        Document doc;
        REQUIRE(doc.content.empty());
        REQUIRE(doc.metadata.empty());
        REQUIRE_FALSE(doc.id.empty());
    }

    SECTION("Construction with content") {
        Document doc("Test content", {{"key", "value"}});
        REQUIRE(doc.content == "Test content");
        REQUIRE(doc.metadata.size() == 1);
        REQUIRE(doc.metadata["key"] == "value");
        REQUIRE_FALSE(doc.id.empty());
    }

    SECTION("Construction with custom ID") {
        Document doc("Test content", {{"key", "value"}}, "custom_id");
        REQUIRE(doc.id == "custom_id");
    }

    SECTION("get_text_snippet") {
        Document doc("This is a long document content that should be truncated");

        auto snippet = doc.get_text_snippet(20);
        REQUIRE(snippet.length() <= 23);  // 20 + "..."
        REQUIRE(snippet.substr(snippet.length() - 3) == "...");

        auto short_snippet = doc.get_text_snippet(100);
        REQUIRE(short_snippet == doc.content);
    }

    SECTION("matches_filter") {
        Document doc("Content", {
            {"source", "test"},
            {"category", "ai"},
            {"author", "test_user"}
        });

        SECTION("Exact match") {
            std::unordered_map<std::string, std::string> filter = {
                {"source", "test"}
            };
            REQUIRE(doc.matches_filter(filter));
        }

        SECTION("Multiple filters") {
            std::unordered_map<std::string, std::string> filter = {
                {"source", "test"},
                {"category", "ai"}
            };
            REQUIRE(doc.matches_filter(filter));
        }

        SECTION("Non-matching filter") {
            std::unordered_map<std::string, std::string> filter = {
                {"source", "different"}
            };
            REQUIRE_FALSE(doc.matches_filter(filter));
        }

        SECTION("Missing metadata key") {
            std::unordered_map<std::string, std::string> filter = {
                {"missing_key", "value"}
            };
            REQUIRE_FALSE(doc.matches_filter(filter));
        }

        SECTION("Empty filter") {
            std::unordered_map<std::string, std::string> filter;
            REQUIRE(doc.matches_filter(filter));
        }
    }
}

TEST_CASE("RetrievedDocument - Document with Score", "[core][types][retrieved_document]") {
    Document base_doc("Base content", {{"source", "test"}});
    RetrievedDocument retrieved_doc(base_doc, 0.85);

    REQUIRE(retrieved_doc.content == "Base content");
    REQUIRE(retrieved_doc.relevance_score == 0.85);
    REQUIRE(retrieved_doc.metadata["source"] == "test");
}

TEST_CASE("RetrievalResult - Result Management", "[core][types][retrieval_result]") {
    RetrievalResult result;
    result.query = "test query";
    result.search_time = std::chrono::milliseconds(50);
    result.retrieval_method = "bm25";

    SECTION("Empty result") {
        REQUIRE(result.documents.empty());
        REQUIRE(result.total_results == 0);
        REQUIRE(result.get_average_score() == 0.0);
    }

    SECTION("Add documents and test operations") {
        // Add some test documents
        result.documents.emplace_back(
            Document("Doc 1", {{"id", "1"}}), 0.9);
        result.documents.emplace_back(
            Document("Doc 2", {{"id", "2"}}), 0.7);
        result.documents.emplace_back(
            Document("Doc 3", {{"id", "3"}}), 0.8);

        result.total_results = 3;

        SECTION("get_top_k") {
            auto top_2 = result.get_top_k(2);
            REQUIRE(top_2.size() == 2);
            REQUIRE(top_2[0].relevance_score == 0.9);
            REQUIRE(top_2[1].relevance_score == 0.8);
        }

        SECTION("get_top_k beyond available") {
            auto top_5 = result.get_top_k(5);
            REQUIRE(top_5.size() == 3);
        }

        SECTION("get_average_score") {
            double avg = result.get_average_score();
            REQUIRE(avg == Approx(0.8));  // (0.9 + 0.7 + 0.8) / 3
        }

        SECTION("filter_by_metadata") {
            // Add documents with different metadata
            result.documents[0].metadata["category"] = "ai";
            result.documents[1].metadata["category"] = "ml";
            result.documents[2].metadata["category"] = "ai";

            auto filtered = result.filter_by_metadata("category", "ai");
            REQUIRE(filtered.documents.size() == 2);
            REQUIRE(filtered.documents[0].metadata["category"] == "ai");
            REQUIRE(filtered.documents[1].metadata["category"] == "ai");
        }
    }
}

TEST_CASE("ConversationMessage - Message Management", "[core][types][conversation]") {
    auto timestamp = std::chrono::system_clock::now();

    SECTION("Human message") {
        ConversationMessage msg(ConversationMessage::Type::HUMAN, "Hello");
        REQUIRE(msg.type == ConversationMessage::Type::HUMAN);
        REQUIRE(msg.content == "Hello");
        REQUIRE(msg.additional_data.empty());
        // Allow small time difference for test timing
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(
            msg.timestamp - timestamp);
        REQUIRE(std::abs(diff.count()) < 100);
    }

    SECTION("AI message") {
        ConversationMessage msg(ConversationMessage::Type::AI, "Response");
        REQUIRE(msg.type == ConversationMessage::Type::AI);
        REQUIRE(msg.content == "Response");
    }

    SECTION("System message") {
        ConversationMessage msg(ConversationMessage::Type::SYSTEM, "System");
        REQUIRE(msg.type == ConversationMessage::Type::SYSTEM);
        REQUIRE(msg.content == "System");
    }
}

TEST_CASE("LLMResult - Generation Result", "[core][types][llm_result]") {
    LLMResult result;
    result.text = "Generated response";
    result.prompt_tokens = 10;
    result.completion_tokens = 20;
    result.total_tokens = 30;
    result.generation_time = std::chrono::milliseconds(100);
    result.finished = true;
    result.finish_reason = "stop";

    SECTION("Basic properties") {
        REQUIRE(result.text == "Generated response");
        REQUIRE(result.prompt_tokens == 10);
        REQUIRE(result.completion_tokens == 20);
        REQUIRE(result.total_tokens == 30);
        REQUIRE(result.generation_time == std::chrono::milliseconds(100));
        REQUIRE(result.finished);
        REQUIRE(result.finish_reason == "stop");
    }

    SECTION("is_successful") {
        REQUIRE(result.is_successful());

        SECTION("Unsuccessful - empty text") {
            result.text = "";
            REQUIRE_FALSE(result.is_successful());
        }

        SECTION("Unsuccessful - not finished") {
            result.text = "Some text";
            result.finished = false;
            REQUIRE_FALSE(result.is_successful());
        }
    }
}

TEST_CASE("GenerationConfig - Configuration Validation", "[core][types][generation_config]") {
    GenerationConfig config;

    SECTION("Default configuration") {
        REQUIRE(config.temperature == 0.7);
        REQUIRE(config.max_tokens == 1000);
        REQUIRE(config.top_p == 0.9);
        REQUIRE(config.top_k == 40);
        REQUIRE_FALSE(config.stream);
        REQUIRE(config.stop_sequences.empty());
        REQUIRE(config.presence_penalty == 0.0);
        REQUIRE(config.frequency_penalty == 0.0);
        REQUIRE(config.seed == -1);
    }

    SECTION("Valid configuration") {
        config.temperature = 1.0;
        config.max_tokens = 500;
        config.top_p = 0.95;
        config.top_k = 50;
        config.presence_penalty = 0.5;
        config.frequency_penalty = -0.5;
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid configurations") {
        SECTION("Invalid temperature") {
            config.temperature = -0.1;
            REQUIRE_FALSE(config.is_valid());

            config.temperature = 2.1;
            REQUIRE_FALSE(config.is_valid());
        }

        SECTION("Invalid max_tokens") {
            config.max_tokens = 0;
            REQUIRE_FALSE(config.is_valid());
        }

        SECTION("Invalid top_p") {
            config.top_p = -0.1;
            REQUIRE_FALSE(config.is_valid());

            config.top_p = 1.1;
            REQUIRE_FALSE(config.is_valid());
        }

        SECTION("Invalid top_k") {
            config.top_k = 0;
            REQUIRE_FALSE(config.is_valid());
        }

        SECTION("Invalid presence_penalty") {
            config.presence_penalty = -2.1;
            REQUIRE_FALSE(config.is_valid());

            config.presence_penalty = 2.1;
            REQUIRE_FALSE(config.is_valid());
        }

        SECTION("Invalid frequency_penalty") {
            config.frequency_penalty = -2.1;
            REQUIRE_FALSE(config.is_valid());

            config.frequency_penalty = 2.1;
            REQUIRE_FALSE(config.is_valid());
        }
    }
}

TEST_CASE("Prompt - Template Formatting", "[core][types][prompt]") {
    SECTION("Simple template") {
        Prompt prompt;
        prompt.template_str = "Hello {name}!";
        prompt.set_variable("name", "World");

        std::string result = prompt.format();
        REQUIRE(result == "Hello World!");
    }

    SECTION("Multiple variables") {
        Prompt prompt;
        prompt.template_str = "{greeting} {name}, today is {day}";
        prompt.set_variable("greeting", "Hello");
        prompt.set_variable("name", "Alice");
        prompt.set_variable("day", "Monday");

        std::string result = prompt.format();
        REQUIRE(result == "Hello Alice, today is Monday");
    }

    SECTION("Missing variable") {
        Prompt prompt;
        prompt.template_str = "Hello {name}!";
        // Don't set name variable

        std::string result = prompt.format();
        REQUIRE(result == "Hello {name}!");  // Variable not replaced
    }

    SECTION("Repeated variable") {
        Prompt prompt;
        prompt.template_str = "{name} says hello to {name}";
        prompt.set_variable("name", "Bob");

        std::string result = prompt.format();
        REQUIRE(result == "Bob says hello to Bob");
    }

    SECTION("Empty template") {
        Prompt prompt;
        prompt.template_str = "";
        prompt.set_variable("name", "Test");

        std::string result = prompt.format();
        REQUIRE(result.empty());
    }

    SECTION("No variables") {
        Prompt prompt;
        prompt.template_str = "Static text";
        prompt.set_variable("name", "Test");

        std::string result = prompt.format();
        REQUIRE(result == "Static text");
    }
}

TEST_CASE("EmbeddingResult - Vector Operations", "[core][types][embedding]") {
    std::vector<float> embedding1 = {1.0f, 0.0f, 0.0f};
    std::vector<float> embedding2 = {0.0f, 1.0f, 0.0f};
    std::vector<float> embedding3 = {1.0f, 0.0f, 0.0f};

    EmbeddingResult result;
    result.embedding = embedding1;
    result.tokens_used = 10;
    result.embedding_time = std::chrono::milliseconds(50);

    SECTION("Basic properties") {
        REQUIRE(result.embedding == embedding1);
        REQUIRE(result.tokens_used == 10);
        REQUIRE(result.embedding_time == std::chrono::milliseconds(50));
    }

    SECTION("cosine_similarity - orthogonal vectors") {
        float similarity = result.cosine_similarity(embedding2);
        REQUIRE(similarity == Approx(0.0).margin(1e-6));
    }

    SECTION("cosine_similarity - identical vectors") {
        float similarity = result.cosine_similarity(embedding3);
        REQUIRE(similarity == Approx(1.0).margin(1e-6));
    }

    SECTION("cosine_similarity - different dimensions") {
        std::vector<float> wrong_dim = {1.0f, 0.0f};  // Different dimension
        float similarity = result.cosine_similarity(wrong_dim);
        REQUIRE(similarity == 0.0);
    }

    SECTION("cosine_similarity - empty vectors") {
        std::vector<float> empty;
        float similarity = result.cosine_similarity(empty);
        REQUIRE(similarity == 0.0);
    }

    SECTION("cosine_similarity - actual calculation") {
        std::vector<float> vec1 = {1.0f, 2.0f, 3.0f};
        std::vector<float> vec2 = {4.0f, 5.0f, 6.0f};

        EmbeddingResult result2;
        result2.embedding = vec1;

        float similarity = result2.cosine_similarity(vec2);

        // Manual calculation for verification
        float dot = 1.0f*4.0f + 2.0f*5.0f + 3.0f*6.0f;  // 32
        float norm1 = std::sqrt(1.0f*1.0f + 2.0f*2.0f + 3.0f*3.0f);  // sqrt(14)
        float norm2 = std::sqrt(4.0f*4.0f + 5.0f*5.0f + 6.0f*6.0f);  // sqrt(77)
        float expected = dot / (norm1 * norm2);

        REQUIRE(similarity == Approx(expected).margin(1e-6));
    }
}

TEST_CASE("PerformanceMetrics - Metrics Collection", "[core][types][performance]") {
    PerformanceMetrics metrics;

    SECTION("Initial state") {
        REQUIRE(metrics.total_requests.load() == 0);
        REQUIRE(metrics.successful_requests.load() == 0);
        REQUIRE(metrics.failed_requests.load() == 0);
        REQUIRE(metrics.get_average_latency() == 0.0);
        REQUIRE(metrics.get_success_rate() == 0.0);
    }

    SECTION("Record successful requests") {
        metrics.record_request(100.0, true);
        metrics.record_request(200.0, true);
        metrics.record_request(150.0, true);

        REQUIRE(metrics.total_requests.load() == 3);
        REQUIRE(metrics.successful_requests.load() == 3);
        REQUIRE(metrics.failed_requests.load() == 0);
        REQUIRE(metrics.get_average_latency() == Approx(150.0));
        REQUIRE(metrics.get_success_rate() == Approx(100.0));
    }

    SECTION("Record mixed requests") {
        metrics.record_request(100.0, true);
        metrics.record_request(200.0, false);
        metrics.record_request(150.0, true);

        REQUIRE(metrics.total_requests.load() == 3);
        REQUIRE(metrics.successful_requests.load() == 2);
        REQUIRE(metrics.failed_requests.load() == 1);
        REQUIRE(metrics.get_average_latency() == Approx(150.0));
        REQUIRE(metrics.get_success_rate() == Approx(66.6667).margin(0.01));
    }

    SECTION("Min/max latency tracking") {
        metrics.record_request(100.0, true);
        metrics.record_request(300.0, true);
        metrics.record_request(50.0, true);
        metrics.record_request(200.0, true);

        REQUIRE(metrics.min_latency_ms.load() == Approx(50.0));
        REQUIRE(metrics.max_latency_ms.load() == Approx(300.0));
    }

    SECTION("Empty metrics calculations") {
        REQUIRE(metrics.get_average_latency() == 0.0);
        REQUIRE(metrics.get_success_rate() == 0.0);
    }
}

TEST_CASE("Task - Task Wrapper", "[core][types][task]") {
    SECTION("Simple task") {
        Task<int> task([]() { return 42; }, "Calculate answer");

        REQUIRE(task.description == "Calculate answer");
        REQUIRE(task.function() == 42);
    }

    SECTION("Task with string result") {
        Task<std::string> task([]() { return std::string("Hello"); }, "Greeting");

        REQUIRE(task.description == "Greeting");
        REQUIRE(task.function() == "Hello");
    }

    SECTION("Task creation timestamp") {
        auto before = std::chrono::system_clock::now();
        Task<int> task([]() { return 0; });
        auto after = std::chrono::system_clock::now();

        REQUIRE(task.created_at >= before);
        REQUIRE(task.created_at <= after);
    }
}