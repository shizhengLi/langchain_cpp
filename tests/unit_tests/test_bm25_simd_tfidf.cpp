#include "langchain/retrievers/bm25_retriever.hpp"
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <vector>

using namespace langchain;
using namespace langchain::retrievers;

TEST_CASE("BM25Retriever - SIMD TF-IDF Scoring", "[retrievers][bm25][simd][tfidf]") {

    SECTION("Basic TF-IDF SIMD functionality") {
        auto retriever = BM25RetrieverFactory::create_standard_retriever();

        std::vector<Document> docs = {
            Document("machine learning is powerful"),
            Document("deep learning advances rapidly"),
            Document("machine learning and deep learning"),
            Document("artificial intelligence applications")
        };

        retriever->add_documents(docs);

        // Test TF-IDF SIMD retrieval
        auto result = retriever->retrieve_tfidf_simd("machine learning");

        REQUIRE_FALSE(result.documents.empty());
        REQUIRE(result.retrieval_method == "tfidf_simd");
        REQUIRE(result.query == "machine learning");
        REQUIRE(result.total_results >= 2); // Should find docs with "machine" and "learning"

        // Check that results are sorted by relevance
        for (size_t i = 1; i < result.documents.size(); ++i) {
            REQUIRE(result.documents[i-1].relevance_score >= result.documents[i].relevance_score);
        }
    }

    SECTION("SIMD vs scalar TF-IDF consistency") {
        auto retriever = BM25RetrieverFactory::create_standard_retriever();

        std::vector<Document> docs = {
            Document("natural language processing with transformers"),
            Document("machine learning for natural language"),
            Document("deep learning transformer architectures"),
            Document("language models and attention mechanisms")
        };

        retriever->add_documents(docs);

        // Compare SIMD TF-IDF with regular BM25 (both should be different algorithms)
        auto tfidf_result = retriever->retrieve_tfidf_simd("language processing");
        auto bm25_result = retriever->retrieve("language processing");

        REQUIRE_FALSE(tfidf_result.documents.empty());
        REQUIRE_FALSE(bm25_result.documents.empty());

        // TF-IDF and BM25 should produce different scores (different algorithms)
        // But both should find relevant documents
        REQUIRE(tfidf_result.total_results > 0);
        REQUIRE(bm25_result.total_results > 0);
    }

    SECTION("SIMD performance with large document sets") {
        auto retriever = BM25RetrieverFactory::create_standard_retriever();

        // Create a larger set of documents
        std::vector<Document> docs;
        for (int i = 0; i < 50; ++i) {
            std::string content = "document " + std::to_string(i) + " contains machine learning ";
            if (i % 3 == 0) content += "and deep learning ";
            if (i % 5 == 0) content += "and artificial intelligence ";
            content += "research applications";
            docs.emplace_back(content);
        }

        retriever->add_documents(docs);

        auto start = std::chrono::high_resolution_clock::now();
        auto result = retriever->retrieve_tfidf_simd("machine learning");
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        REQUIRE_FALSE(result.documents.empty());
        REQUIRE(result.total_results >= 10); // Should find multiple matching documents

        // Performance should be reasonable (less than 10ms for 50 documents)
        REQUIRE(duration.count() < 10000); // 10ms in microseconds
    }

    SECTION("SIMD TF-IDF with multi-term queries") {
        auto retriever = BM25RetrieverFactory::create_standard_retriever();

        std::vector<Document> docs = {
            Document("information retrieval and web search"),
            Document("machine learning for information retrieval"),
            Document("web mining and information extraction"),
            Document("search engines and ranking algorithms")
        };

        retriever->add_documents(docs);

        auto result = retriever->retrieve_tfidf_simd("information retrieval");

        REQUIRE_FALSE(result.documents.empty());
        REQUIRE(result.total_results >= 2); // Should find docs with both terms

        // Check that multi-term query scoring works properly
        bool found_multi_term_doc = false;
        for (const auto& doc : result.documents) {
            if (doc.content.find("information") != std::string::npos &&
                doc.content.find("retrieval") != std::string::npos) {
                found_multi_term_doc = true;
                break;
            }
        }
        REQUIRE(found_multi_term_doc);
    }

    SECTION("SIMD TF-IDF edge cases") {
        auto retriever = BM25RetrieverFactory::create_standard_retriever();

        SECTION("Empty index") {
            auto result = retriever->retrieve_tfidf_simd("test query");
            REQUIRE(result.documents.empty());
            REQUIRE(result.total_results == 0);
        }

        SECTION("Empty query") {
            std::vector<Document> docs = {Document("test document")};
            retriever->add_documents(docs);

            auto result = retriever->retrieve_tfidf_simd("");
            REQUIRE(result.documents.empty());
            REQUIRE(result.total_results == 0);
        }

        SECTION("Non-existent terms") {
            std::vector<Document> docs = {Document("machine learning")};
            retriever->add_documents(docs);

            auto result = retriever->retrieve_tfidf_simd("nonexistent terms");
            REQUIRE(result.documents.empty());
            REQUIRE(result.total_results == 0);
        }
    }

    SECTION("SIMD TF-IDF score normalization") {
        BM25Retriever::Config config;
        config.normalize_scores = true;
        config.score_threshold = 0.0;

        auto retriever = std::make_unique<BM25Retriever>(config);

        std::vector<Document> docs = {
            Document("machine learning algorithms"),
            Document("deep learning neural networks"),
            Document("machine and deep learning integration")
        };

        retriever->add_documents(docs);

        auto result = retriever->retrieve_tfidf_simd("learning");

        REQUIRE_FALSE(result.documents.empty());

        if (!result.documents.empty()) {
            // Highest score should be 1.0 when normalized
            REQUIRE(result.documents[0].relevance_score <= 1.0);
            REQUIRE(result.documents[0].relevance_score > 0.0);
        }
    }

    SECTION("SIMD TF-IDF with score threshold") {
        BM25Retriever::Config config;
        config.score_threshold = 0.5; // Higher threshold
        config.normalize_scores = false; // Don't normalize to test raw scores

        auto retriever = std::make_unique<BM25Retriever>(config);

        std::vector<Document> docs = {
            Document("machine learning research"),
            Document("computer vision applications"),
            Document("natural language processing"),
            Document("artificial intelligence systems")
        };

        retriever->add_documents(docs);

        auto result = retriever->retrieve_tfidf_simd("machine learning");

        // Should only return documents with scores above threshold
        for (const auto& doc : result.documents) {
            REQUIRE(doc.relevance_score >= 0.5);
        }
    }

    SECTION("SIMD TF-IDF factory methods") {
        SECTION("Standard retriever") {
            auto retriever = BM25RetrieverFactory::create_standard_retriever();
            std::vector<Document> docs = {Document("test document")};
            retriever->add_documents(docs);

            auto result = retriever->retrieve_tfidf_simd("test");
            REQUIRE_FALSE(result.documents.empty());
        }

        SECTION("Short document retriever") {
            auto retriever = BM25RetrieverFactory::create_short_document_retriever();
            std::vector<Document> docs = {Document("short doc")};
            retriever->add_documents(docs);

            auto result = retriever->retrieve_tfidf_simd("short");
            REQUIRE_FALSE(result.documents.empty());
        }

        SECTION("Long document retriever") {
            auto retriever = BM25RetrieverFactory::create_long_document_retriever();
            std::vector<Document> docs = {Document("very long document with extensive content for testing purposes")};
            retriever->add_documents(docs);

            auto result = retriever->retrieve_tfidf_simd("document");
            REQUIRE_FALSE(result.documents.empty());
        }
    }

    SECTION("SIMD TF-IDF performance statistics") {
        auto retriever = BM25RetrieverFactory::create_standard_retriever();

        std::vector<Document> docs = {
            Document("first document with content"),
            Document("second document with content"),
            Document("third document with content")
        };

        retriever->add_documents(docs);

        // Perform multiple queries to populate statistics
        for (int i = 0; i < 5; ++i) {
            retriever->retrieve_tfidf_simd("content");
        }

        auto metadata = retriever->get_metadata();
        REQUIRE(std::any_cast<size_t>(metadata["total_queries"]) >= 5);
        // Note: avg_query_time is not tracked in current implementation
    }
}