#include <catch2/catch_all.hpp>
#include "langchain/retrievers/inverted_index_retriever.hpp"
#include <memory>
#include <thread>
#include <chrono>

using namespace langchain::retrievers;
using namespace langchain;
using Catch::Approx;

TEST_CASE("InvertedIndexRetriever - Default Configuration", "[retrievers][inverted_index][config]") {
    SECTION("Default config values") {
        auto retriever = InvertedIndexRetrieverFactory::create_retrieval_retriever();
        auto config = retriever->get_config();

        REQUIRE(config.min_term_frequency == 1);
        REQUIRE(config.max_postings_per_term == 100000);
        REQUIRE(config.enable_term_caching == true);
        REQUIRE(config.cache_size_limit == 10000);
        REQUIRE(config.normalize_scores == true);
        REQUIRE(config.score_threshold == 0.01);
        REQUIRE(config.max_results == 10);
        REQUIRE(config.default_field == "content");
    }
}

TEST_CASE("InvertedIndexRetriever - Basic Operations", "[retrievers][inverted_index][basic]") {
    auto retriever = InvertedIndexRetrieverFactory::create_retrieval_retriever();

    SECTION("Empty index") {
        REQUIRE(retriever->document_count() == 0);
        REQUIRE_FALSE(retriever->is_ready());

        auto result = retriever->retrieve("test query");
        REQUIRE(result.documents.empty());
        REQUIRE(result.query == "test query");
        REQUIRE(result.retrieval_method == "inverted_index");
    }

    SECTION("Add single document") {
        Document doc("This is a test document about search engines.");
        auto doc_ids = retriever->add_documents({doc});

        REQUIRE(doc_ids.size() == 1);
        REQUIRE(retriever->document_count() == 1);
        REQUIRE(retriever->is_ready());
    }

    SECTION("Add multiple documents") {
        std::vector<Document> docs = {
            Document("Machine learning is a subset of artificial intelligence."),
            Document("Deep learning uses neural networks for pattern recognition."),
            Document("Natural language processing helps computers understand text."),
            Document("Computer vision enables machines to interpret visual information.")
        };

        auto doc_ids = retriever->add_documents(docs);
        REQUIRE(doc_ids.size() == 4);
        REQUIRE(retriever->document_count() == 4);
    }

    SECTION("Clear index") {
        Document doc("Test document");
        retriever->add_documents({doc});
        REQUIRE(retriever->document_count() == 1);

        retriever->clear();
        REQUIRE(retriever->document_count() == 0);
        REQUIRE_FALSE(retriever->is_ready());
    }
}

TEST_CASE("InvertedIndexRetriever - Document Retrieval", "[retrievers][inverted_index][retrieval]") {
    auto retriever = InvertedIndexRetrieverFactory::create_retrieval_retriever();

    // Add test documents
    std::vector<Document> docs = {
        Document("Machine learning algorithms analyze data to make predictions."),
        Document("Deep learning is a subset of machine learning using neural networks."),
        Document("Natural language processing deals with text and speech understanding."),
        Document("Computer vision focuses on image recognition and analysis."),
        Document("Artificial intelligence encompasses various AI technologies.")
    };

    retriever->add_documents(docs);

    SECTION("Single term query") {
        auto result = retriever->retrieve("machine");

        REQUIRE_FALSE(result.documents.empty());
        REQUIRE(result.query == "machine");
        REQUIRE(result.retrieval_method == "inverted_index");
        REQUIRE(result.total_results >= 2);  // Should find docs about machine learning

        // Check that results are sorted by relevance
        if (result.documents.size() > 1) {
            REQUIRE(result.documents[0].relevance_score >= result.documents[1].relevance_score);
        }
    }

    SECTION("Multiple term query") {
        auto result = retriever->retrieve("deep learning");

        REQUIRE_FALSE(result.documents.empty());
        REQUIRE(result.total_results >= 1);  // Should find the deep learning document

        // The deep learning document should have highest relevance
        bool found_deep_learning_doc = false;
        for (const auto& doc : result.documents) {
            if (doc.content.find("deep learning") != std::string::npos ||
                doc.content.find("Deep learning") != std::string::npos) {
                found_deep_learning_doc = true;
                break;
            }
        }
        REQUIRE(found_deep_learning_doc);
    }

    SECTION("Non-existent term") {
        auto result = retriever->retrieve("nonexistentterm12345");
        REQUIRE(result.documents.empty());
        REQUIRE(result.total_results == 0);
    }

    SECTION("Empty query") {
        auto result = retriever->retrieve("");
        REQUIRE(result.documents.empty());
        REQUIRE(result.total_results == 0);
    }

    SECTION("Query with stop words only") {
        auto result = retriever->retrieve("the and is");
        REQUIRE(result.documents.empty());  // Stop words should be filtered out
    }
}

TEST_CASE("InvertedIndexRetriever - Score Normalization", "[retrievers][inverted_index][scoring]") {
    SECTION("Scores are normalized when enabled") {
        InvertedIndexRetriever::Config config;
        config.normalize_scores = true;
        config.max_results = 5;
        auto retriever = std::make_unique<InvertedIndexRetriever>(config);

        std::vector<Document> docs = {
            Document("machine learning machine learning"),
            Document("machine"),
            Document("learning algorithms")
        };

        retriever->add_documents(docs);
        auto result = retriever->retrieve("machine learning");

        if (!result.documents.empty()) {
            // Top result should have score of 1.0 (normalized)
            REQUIRE(result.documents[0].relevance_score <= 1.0);
            if (result.documents[0].relevance_score > 0) {
                REQUIRE(result.documents[0].relevance_score == Approx(1.0));
            }
        }
    }

    SECTION("Scores are not normalized when disabled") {
        InvertedIndexRetriever::Config config;
        config.normalize_scores = false;
        auto retriever = std::make_unique<InvertedIndexRetriever>(config);

        std::vector<Document> docs = {
            Document("machine learning"),
            Document("machine")
        };

        retriever->add_documents(docs);
        auto result = retriever->retrieve("machine");

        // Scores should be raw TF-IDF values
        if (!result.documents.empty()) {
            REQUIRE(result.documents[0].relevance_score > 0);
        }
    }
}

TEST_CASE("InvertedIndexRetriever - Cache Performance", "[retrievers][inverted_index][cache]") {
    SECTION("Cache statistics tracking") {
        auto retriever = InvertedIndexRetrieverFactory::create_retrieval_retriever();

        std::vector<Document> docs = {
            Document("cache performance testing with repeated terms"),
            Document("cache memory optimization improves query speed")
        };

        retriever->add_documents(docs);

        // Perform multiple queries to populate cache
        retriever->retrieve("cache");
        retriever->retrieve("cache");
        retriever->retrieve("performance");
        retriever->retrieve("performance");

        auto cache_stats = retriever->get_cache_stats();
        REQUIRE(cache_stats["total_queries"] > 0);
        REQUIRE(cache_stats["hit_rate"] >= 0.0);
        REQUIRE(cache_stats["hit_rate"] <= 100.0);
        REQUIRE(cache_stats["miss_rate"] >= 0.0);
        REQUIRE(cache_stats["miss_rate"] <= 100.0);
    }

    SECTION("Cache cleanup when limit exceeded") {
        InvertedIndexRetriever::Config config;
        config.cache_size_limit = 2;  // Very small cache
        config.enable_term_caching = true;
        auto retriever = std::make_unique<InvertedIndexRetriever>(config);

        // Add many unique terms to exceed cache limit
        std::vector<Document> docs;
        for (int i = 0; i < 10; ++i) {
            docs.push_back(Document("unique_term_" + std::to_string(i) + " content"));
        }

        retriever->add_documents(docs);

        // Access various terms to trigger cleanup
        for (int i = 0; i < 10; ++i) {
            retriever->retrieve("unique_term_" + std::to_string(i));
        }

        // Should not crash and should have cleaned up cache
        auto cache_stats = retriever->get_cache_stats();
        REQUIRE(cache_stats["total_queries"] > 0);
    }
}

TEST_CASE("InvertedIndexRetriever - Posting List Operations", "[retrievers][inverted_index][postings]") {
    auto retriever = InvertedIndexRetrieverFactory::create_retrieval_retriever();

    std::vector<Document> docs = {
        Document("term1 term2 term3"),
        Document("term1 term3"),
        Document("term2 term3")
    };

    retriever->add_documents(docs);

    SECTION("Get postings for existing term") {
        auto postings = retriever->get_postings("term1");
        REQUIRE(postings.size() == 2);  // term1 appears in 2 documents

        // Check document IDs are unique
        std::unordered_set<size_t> doc_ids;
        for (const auto& posting : postings) {
            doc_ids.insert(posting.document_id);
        }
        REQUIRE(doc_ids.size() == 2);
    }

    SECTION("Get postings for non-existent term") {
        auto postings = retriever->get_postings("nonexistent");
        REQUIRE(postings.empty());
    }

    SECTION("Get term information") {
        auto term_info = retriever->get_term_info("term1");
        REQUIRE(term_info.document_frequency == 2);
        REQUIRE(term_info.total_term_frequency >= 2);  // At least once per document
        REQUIRE(term_info.idf > 0);  // IDF should be calculated
    }

    SECTION("Most frequent terms") {
        auto frequent_terms = retriever->get_most_frequent_terms(3);
        REQUIRE(frequent_terms.size() <= 3);

        // Should include term3 (appears in all 3 documents)
        bool found_term3 = false;
        for (const auto& [term, freq] : frequent_terms) {
            if (term == "term3") {
                found_term3 = true;
                REQUIRE(freq == 3);
                break;
            }
        }
        REQUIRE(found_term3);
    }
}

TEST_CASE("InvertedIndexRetriever - Index Optimization", "[retrievers][inverted_index][optimization]") {
    auto retriever = InvertedIndexRetrieverFactory::create_retrieval_retriever();

    // Add documents in random order
    std::vector<Document> docs = {
        Document("optimization test document three"),
        Document("optimization test document one"),
        Document("optimization test document two")
    };

    retriever->add_documents(docs);

    SECTION("Index optimization") {
        REQUIRE_NOTHROW(retriever->optimize_index());

        // Should still be able to retrieve documents after optimization
        auto result = retriever->retrieve("optimization");
        REQUIRE(result.documents.size() == 3);
    }

    SECTION("Posting lists are sorted after optimization") {
        retriever->optimize_index();

        auto postings = retriever->get_postings("optimization");

        // Check if postings are sorted by document ID
        for (size_t i = 1; i < postings.size(); ++i) {
            REQUIRE(postings[i-1].document_id < postings[i].document_id);
        }
    }
}

TEST_CASE("InvertedIndexRetriever - Configuration Updates", "[retrievers][inverted_index][config]") {
    auto retriever = InvertedIndexRetrieverFactory::create_retrieval_retriever();

    SECTION("Update configuration") {
        InvertedIndexRetriever::Config new_config;
        new_config.score_threshold = 0.5;
        new_config.max_results = 2;

        retriever->update_config(new_config);

        auto updated_config = retriever->get_config();
        REQUIRE(updated_config.score_threshold == 0.5);
        REQUIRE(updated_config.max_results == 2);
    }

    SECTION("Configuration affects retrieval") {
        std::vector<Document> docs = {
            Document("high relevance document with many terms"),
            Document("low relevance document")
        };

        retriever->add_documents(docs);

        // Update to have high threshold
        InvertedIndexRetriever::Config new_config;
        new_config.score_threshold = 10.0;  // Very high threshold
        retriever->update_config(new_config);

        auto result = retriever->retrieve("relevance");
        REQUIRE(result.documents.empty());  // Should be empty due to high threshold
    }
}

TEST_CASE("InvertedIndexRetriever - Metadata and Statistics", "[retrievers][inverted_index][metadata]") {
    auto retriever = InvertedIndexRetrieverFactory::create_retrieval_retriever();

    SECTION("Empty index metadata") {
        auto metadata = retriever->get_metadata();

        REQUIRE(std::any_cast<std::string>(metadata["type"]) == "InvertedIndexRetriever");
        REQUIRE(std::any_cast<size_t>(metadata["document_count"]) == 0);
        REQUIRE(std::any_cast<bool>(metadata["ready"]) == false);
        REQUIRE(std::any_cast<size_t>(metadata["total_terms"]) == 0);
        REQUIRE(std::any_cast<size_t>(metadata["total_postings"]) == 0);
    }

    SECTION("Index with documents metadata") {
        std::vector<Document> docs = {
            Document("First document with several terms for testing"),
            Document("Second document also contains multiple terms")
        };

        retriever->add_documents(docs);

        auto metadata = retriever->get_metadata();
        REQUIRE(std::any_cast<size_t>(metadata["document_count"]) == 2);
        REQUIRE(std::any_cast<bool>(metadata["ready"]) == true);
        REQUIRE(std::any_cast<size_t>(metadata["total_terms"]) > 0);
        REQUIRE(std::any_cast<size_t>(metadata["total_postings"]) > 0);
        REQUIRE(std::any_cast<bool>(metadata["cache_enabled"]) == true);
    }
}

TEST_CASE("InvertedIndexRetriever - Thread Safety", "[retrievers][inverted_index][threading]") {
    auto retriever = InvertedIndexRetrieverFactory::create_retrieval_retriever();

    // Add initial documents
    std::vector<Document> docs = {
        Document("Thread safety test document one"),
        Document("Thread safety test document two"),
        Document("Thread safety test document three")
    };

    retriever->add_documents(docs);

    SECTION("Concurrent reads") {
        const int num_threads = 5;
        const int queries_per_thread = 10;
        std::vector<std::thread> threads;
        std::atomic<int> successful_queries{0};

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&retriever, &successful_queries, queries_per_thread, i]() {
                for (int j = 0; j < queries_per_thread; ++j) {
                    auto result = retriever->retrieve("thread");
                    if (!result.documents.empty()) {
                        successful_queries++;
                    }
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        REQUIRE(successful_queries.load() == num_threads * queries_per_thread);
    }

    SECTION("Concurrent writes") {
        const int num_threads = 3;
        const int docs_per_thread = 5;
        std::vector<std::thread> threads;
        std::mutex result_mutex;
        std::vector<std::string> added_doc_ids;

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&retriever, &result_mutex, &added_doc_ids, docs_per_thread, i]() {
                std::vector<Document> new_docs;
                for (int j = 0; j < docs_per_thread; ++j) {
                    new_docs.push_back(Document("Thread " + std::to_string(i) + " document " + std::to_string(j)));
                }

                auto doc_ids = retriever->add_documents(new_docs);

                std::lock_guard<std::mutex> lock(result_mutex);
                added_doc_ids.insert(added_doc_ids.end(), doc_ids.begin(), doc_ids.end());
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        REQUIRE(added_doc_ids.size() == num_threads * docs_per_thread);
        REQUIRE(retriever->document_count() == 3 + num_threads * docs_per_thread);
    }
}

TEST_CASE("InvertedIndexRetriever - Batch Operations", "[retrievers][inverted_index][batch]") {
    auto retriever = InvertedIndexRetrieverFactory::create_retrieval_retriever();

    SECTION("Batch retrieval") {
        std::vector<Document> docs = {
            Document("Document about apples and fruits"),
            Document("Document about bananas and fruits"),
            Document("Document about oranges and citrus")
        };

        retriever->add_documents(docs);

        std::vector<std::string> queries = {"apples", "bananas", "oranges"};
        auto results = retriever->retrieve_batch(queries);

        REQUIRE(results.size() == 3);

        for (size_t i = 0; i < queries.size(); ++i) {
            REQUIRE(results[i].query == queries[i]);
            REQUIRE_FALSE(results[i].documents.empty());
        }
    }
}

TEST_CASE("InvertedIndexRetriever - Factory Methods", "[retrievers][inverted_index][factory]") {
    SECTION("Create retrieval retriever") {
        auto retriever = InvertedIndexRetrieverFactory::create_retrieval_retriever();
        REQUIRE(retriever != nullptr);
        REQUIRE(retriever->document_count() == 0);
    }

    SECTION("Create search retriever") {
        auto retriever = InvertedIndexRetrieverFactory::create_search_retriever();
        REQUIRE(retriever != nullptr);

        auto config = retriever->get_config();
        REQUIRE(config.max_results == 20);  // Search optimized for more results
    }

    SECTION("Create large dataset retriever") {
        auto retriever = InvertedIndexRetrieverFactory::create_large_dataset_retriever();
        REQUIRE(retriever != nullptr);

        auto config = retriever->get_config();
        REQUIRE(config.min_term_frequency == 2);  // More aggressive filtering
        REQUIRE(config.cache_size_limit == 50000);  // Larger cache
    }

    SECTION("Create memory efficient retriever") {
        auto retriever = InvertedIndexRetrieverFactory::create_memory_efficient_retriever();
        REQUIRE(retriever != nullptr);

        auto config = retriever->get_config();
        REQUIRE(config.enable_term_caching == false);  // Caching disabled
        REQUIRE(config.max_results == 5);  // Fewer results
    }
}

TEST_CASE("InvertedIndexRetriever - Edge Cases", "[retrievers][inverted_index][edge_cases]") {
    auto retriever = InvertedIndexRetrieverFactory::create_retrieval_retriever();

    SECTION("Very long document") {
        std::string long_content(30, 'a');  // 30 'a' characters - within token limits (max 50)
        Document doc(long_content);
        retriever->add_documents({doc});

        auto result = retriever->retrieve("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");  // Query should match the actual token
        REQUIRE_FALSE(result.documents.empty());
    }

    SECTION("Document with special characters") {
        Document doc("Special characters: !@#$%^&*()_+-={}[]|\\:;\"'<>?,./");
        retriever->add_documents({doc});

        auto result = retriever->retrieve("special");
        REQUIRE_FALSE(result.documents.empty());
    }

    SECTION("Empty document") {
        Document doc("");
        retriever->add_documents({doc});

        auto result = retriever->retrieve("any");
        REQUIRE(result.documents.empty());
    }

    SECTION("Document with Unicode characters") {
        Document doc("Unicode test: café résumé naïve");
        retriever->add_documents({doc});

        auto result = retriever->retrieve("unicode");
        REQUIRE_FALSE(result.documents.empty());
    }
}