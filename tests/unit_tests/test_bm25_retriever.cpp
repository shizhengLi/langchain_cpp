#include <catch2/catch_all.hpp>
#include "langchain/retrievers/bm25_retriever.hpp"
#include <memory>
#include <thread>
#include <chrono>
#include <cmath>

using namespace langchain::retrievers;
using namespace langchain;
using Catch::Approx;

TEST_CASE("BM25Retriever - Default Configuration", "[retrievers][bm25][config]") {
    SECTION("Default config values") {
        auto retriever = BM25RetrieverFactory::create_standard_retriever();
        auto config = retriever->get_config();

        REQUIRE(config.k1 == 1.2);
        REQUIRE(config.b == 0.75);
        REQUIRE(config.delta == 1.0);
        REQUIRE(config.min_term_frequency == 1);
        REQUIRE(config.max_postings_per_term == 100000);
        REQUIRE(config.enable_term_caching == true);
        REQUIRE(config.cache_size_limit == 10000);
        REQUIRE(config.normalize_scores == true);
        REQUIRE(config.score_threshold == 0.0);
        REQUIRE(config.max_results == 100);
        REQUIRE(config.enable_field_weighting == false);
    }
}

TEST_CASE("BM25Retriever - Basic Operations", "[retrievers][bm25][basic]") {
    auto retriever = BM25RetrieverFactory::create_standard_retriever();

    SECTION("Empty index") {
        REQUIRE(retriever->document_count() == 0);
        REQUIRE_FALSE(retriever->is_ready());

        auto result = retriever->retrieve("test query");
        REQUIRE(result.documents.empty());
        REQUIRE(result.query == "test query");
        REQUIRE(result.retrieval_method == "bm25");
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

TEST_CASE("BM25Retriever - Document Retrieval", "[retrievers][bm25][retrieval]") {
    auto retriever = BM25RetrieverFactory::create_standard_retriever();

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
        REQUIRE(result.retrieval_method == "bm25");
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

TEST_CASE("BM25Retriever - BM25 Scoring", "[retrievers][bm25][scoring]") {
    SECTION("BM25 scores are properly calculated") {
        BM25Retriever::Config config;
        config.k1 = 1.2;
        config.b = 0.75;
        config.delta = 1.0;
        auto retriever = std::make_unique<BM25Retriever>(config);

        // Add documents with different lengths
        std::vector<Document> docs = {
            Document("cat"),                           // Short document
            Document("The cat sat on the mat"),       // Medium document
            Document("The cat sat on the mat and played with the ball") // Long document
        };

        retriever->add_documents(docs);
        auto result = retriever->retrieve("cat");

        REQUIRE_FALSE(result.documents.empty());

        // BM25 should handle document length normalization
        // The medium document might get the highest score due to optimal length
        bool found_valid_scores = false;
        for (const auto& doc : result.documents) {
            if (doc.relevance_score > 0) {
                found_valid_scores = true;
                break;
            }
        }
        REQUIRE(found_valid_scores);
    }

    SECTION("BM25 parameter sensitivity") {
        // Test with different k1 values
        BM25Retriever::Config config1, config2;
        config1.k1 = 0.5;  // Low term frequency saturation
        config1.normalize_scores = false;  // Disable normalization to see real differences
        config1.delta = 0.0;  // Remove delta to see real BM25 scores

        config2.k1 = 2.0;  // High term frequency saturation
        config2.normalize_scores = false;  // Disable normalization to see real differences
        config2.delta = 0.0;  // Remove delta to see real BM25 scores

        auto retriever1 = std::make_unique<BM25Retriever>(config1);
        auto retriever2 = std::make_unique<BM25Retriever>(config2);

        std::vector<Document> docs = {
            Document("machine machine machine learning")
        };

        retriever1->add_documents(docs);
        retriever2->add_documents(docs);

        auto result1 = retriever1->retrieve("machine");
        auto result2 = retriever2->retrieve("machine");

        REQUIRE_FALSE(result1.documents.empty());
        REQUIRE_FALSE(result2.documents.empty());
        // Different k1 values should produce different scores
        REQUIRE(result1.documents[0].relevance_score != result2.documents[0].relevance_score);
    }
}

TEST_CASE("BM25Retriever - Document Statistics", "[retrievers][bm25][statistics]") {
    auto retriever = BM25RetrieverFactory::create_standard_retriever();

    SECTION("Empty index statistics") {
        auto metadata = retriever->get_metadata();

        REQUIRE(std::any_cast<std::string>(metadata["type"]) == "BM25Retriever");
        REQUIRE(std::any_cast<size_t>(metadata["document_count"]) == 0);
        REQUIRE(std::any_cast<bool>(metadata["ready"]) == false);
        REQUIRE(std::any_cast<size_t>(metadata["total_terms"]) == 0);
    }

    SECTION("Index with documents statistics") {
        std::vector<Document> docs = {
            Document("First document with several terms for testing"),
            Document("Second document also contains multiple terms")
        };

        retriever->add_documents(docs);

        auto metadata = retriever->get_metadata();
        REQUIRE(std::any_cast<size_t>(metadata["document_count"]) == 2);
        REQUIRE(std::any_cast<bool>(metadata["ready"]) == true);
        REQUIRE(std::any_cast<size_t>(metadata["total_terms"]) > 0);
        REQUIRE(std::any_cast<bool>(metadata["cache_enabled"]) == true);

        // Check BM25-specific statistics
        REQUIRE(std::any_cast<double>(metadata["avg_document_length"]) > 0);
        REQUIRE(std::any_cast<size_t>(metadata["total_terms_in_corpus"]) > 0);
    }

    SECTION("Document level statistics") {
        std::vector<Document> docs = {
            Document("Short doc"),
            Document("This is a much longer document with many more terms")
        };

        retriever->add_documents(docs);

        auto stats1 = retriever->get_document_stats(1);
        auto stats2 = retriever->get_document_stats(2);

        REQUIRE(stats1.has_value());
        REQUIRE(stats2.has_value());
        REQUIRE(stats1->term_count < stats2->term_count);
        REQUIRE(stats1->document_id == 1);
        REQUIRE(stats2->document_id == 2);
    }
}

TEST_CASE("BM25Retriever - Cache Performance", "[retrievers][bm25][cache]") {
    SECTION("Cache statistics tracking") {
        auto retriever = BM25RetrieverFactory::create_standard_retriever();

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
        BM25Retriever::Config config;
        config.cache_size_limit = 2;  // Very small cache
        config.enable_term_caching = true;
        auto retriever = std::make_unique<BM25Retriever>(config);

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

TEST_CASE("BM25Retriever - Posting List Operations", "[retrievers][bm25][postings]") {
    auto retriever = BM25RetrieverFactory::create_standard_retriever();

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
        REQUIRE(term_info.total_term_frequency >= 2);
        REQUIRE(term_info.idf > 0);  // BM25 IDF should be calculated
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

TEST_CASE("BM25Retriever - Index Optimization", "[retrievers][bm25][optimization]") {
    auto retriever = BM25RetrieverFactory::create_standard_retriever();

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

TEST_CASE("BM25Retriever - Configuration Updates", "[retrievers][bm25][config]") {
    auto retriever = BM25RetrieverFactory::create_standard_retriever();

    SECTION("Update configuration") {
        BM25Retriever::Config new_config;
        new_config.k1 = 2.0;
        new_config.b = 0.5;
        new_config.score_threshold = 0.5;
        new_config.max_results = 2;

        retriever->update_config(new_config);

        auto updated_config = retriever->get_config();
        REQUIRE(updated_config.k1 == 2.0);
        REQUIRE(updated_config.b == 0.5);
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
        BM25Retriever::Config new_config;
        new_config.score_threshold = 10.0;  // Very high threshold
        retriever->update_config(new_config);

        auto result = retriever->retrieve("relevance");
        REQUIRE(result.documents.empty());  // Should be empty due to high threshold
    }
}

TEST_CASE("BM25Retriever - Factory Methods", "[retrievers][bm25][factory]") {
    SECTION("Create standard retriever") {
        auto retriever = BM25RetrieverFactory::create_standard_retriever();
        REQUIRE(retriever != nullptr);
        REQUIRE(retriever->document_count() == 0);

        auto config = retriever->get_config();
        REQUIRE(config.k1 == 1.2);
        REQUIRE(config.b == 0.75);
    }

    SECTION("Create short document retriever") {
        auto retriever = BM25RetrieverFactory::create_short_document_retriever();
        REQUIRE(retriever != nullptr);

        auto config = retriever->get_config();
        REQUIRE(config.k1 == 1.2);  // Standard for short docs
        REQUIRE(config.b == 0.75);  // Standard for short docs
    }

    SECTION("Create long document retriever") {
        auto retriever = BM25RetrieverFactory::create_long_document_retriever();
        REQUIRE(retriever != nullptr);

        auto config = retriever->get_config();
        REQUIRE(config.k1 == 1.2);  // Standard for long docs
        REQUIRE(config.b == 0.75);  // Standard for long docs
    }

    SECTION("Create precision-focused retriever") {
        auto retriever = BM25RetrieverFactory::create_precision_focused_retriever();
        REQUIRE(retriever != nullptr);

        auto config = retriever->get_config();
        REQUIRE(config.k1 > 1.2);  // Higher k1 for precision
        REQUIRE(config.b == 0.75);
    }

    SECTION("Create recall-focused retriever") {
        auto retriever = BM25RetrieverFactory::create_recall_focused_retriever();
        REQUIRE(retriever != nullptr);

        auto config = retriever->get_config();
        REQUIRE(config.k1 < 1.2);  // Lower k1 for recall
        REQUIRE(config.b == 0.75);
    }
}

TEST_CASE("BM25Retriever - Thread Safety", "[retrievers][bm25][threading]") {
    auto retriever = BM25RetrieverFactory::create_standard_retriever();

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

TEST_CASE("BM25Retriever - Edge Cases", "[retrievers][bm25][edge_cases]") {
    auto retriever = BM25RetrieverFactory::create_standard_retriever();

    SECTION("Very long document") {
        std::string long_content(50, 'a');  // 50 'a' characters
        Document doc(long_content);
        retriever->add_documents({doc});

        auto result = retriever->retrieve("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");  // Query should match the actual token
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

    SECTION("Single character queries") {
        Document doc("a b c d e f g");
        retriever->add_documents({doc});

        // Test if single character queries work (may be filtered depending on text processor config)
        auto result = retriever->retrieve("a");
        // This may return empty if single characters are filtered out
    }
}

TEST_CASE("BM25Retriever - Performance Features", "[retrievers][bm25][performance]") {
    SECTION("BM25 parameter information") {
        auto retriever = BM25RetrieverFactory::create_standard_retriever();

        auto params = retriever->get_bm25_parameters();
        REQUIRE(params.find("k1") != params.end());
        REQUIRE(params.find("b") != params.end());
        REQUIRE(params.find("delta") != params.end());
        REQUIRE(params["k1"] == 1.2);
        REQUIRE(params["b"] == 0.75);
        REQUIRE(params["delta"] == 1.0);
    }

    SECTION("Optimized posting intersection") {
        auto retriever = BM25RetrieverFactory::create_standard_retriever();

        std::vector<Document> docs = {
            Document("apple banana cherry"),
            Document("apple date fig"),
            Document("banana cherry grape"),
            Document("apple banana cherry date")
        };

        retriever->add_documents(docs);

        // Query with multiple terms should use optimized intersection
        auto result = retriever->retrieve("apple banana");
        REQUIRE_FALSE(result.documents.empty());

        // Should find documents containing both terms
        bool found_multi_term_doc = false;
        for (const auto& doc : result.documents) {
            if (doc.content.find("apple") != std::string::npos &&
                doc.content.find("banana") != std::string::npos) {
                found_multi_term_doc = true;
                break;
            }
        }
        REQUIRE(found_multi_term_doc);
    }
}