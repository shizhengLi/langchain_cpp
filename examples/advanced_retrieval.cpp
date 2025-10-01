#include "langchain/langchain.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace langchain;

int main() {
    std::cout << "=== LangChain++ Advanced Retrieval Example ===" << std::endl;

    try {
        // 1. Create text processor with advanced configuration
        auto text_processor = std::make_shared<text::TextProcessor>();
        text_processor->set_language("english");
        text_processor->enable_stemming(true);
        text_processor->enable_stop_words(true);
        text_processor->set_min_token_length(2);
        text_processor->set_max_token_length(20);

        // 2. Prepare sample documents
        std::vector<Document> documents = {
            {
                "Machine learning algorithms can automatically learn patterns from data without explicit programming. Deep learning uses neural networks with multiple layers.",
                {{"source", "ml_handbook"}, {"category", "ML"}, {"author", "Dr. Smith"}, {"year", "2023"}, {"difficulty", "intermediate"}}
            },
            {
                "Natural language processing enables computers to understand human language. Transformers and attention mechanisms have revolutionized NLP tasks.",
                {{"source", "nlp_papers"}, {"category", "NLP"}, {"author", "Prof. Johnson"}, {"year", "2023"}, {"difficulty", "advanced"}}
            },
            {
                "Computer vision algorithms process images and videos. Convolutional neural networks excel at image recognition and object detection tasks.",
                {{"source", "cv_tutorial"}, {"category", "CV"}, {"author", "Dr. Wang"}, {"year", "2022"}, {"difficulty", "intermediate"}}
            },
            {
                "Reinforcement learning learns optimal actions through trial and error. Q-learning and policy gradients are popular algorithms.",
                {{"source", "rl_guide"}, {"category", "RL"}, {"author", "Dr. Chen"}, {"year", "2023"}, {"difficulty", "beginner"}}
            },
            {
                "Large language models like GPT can generate human-like text. Few-shot learning enables them to perform tasks with minimal examples.",
                {{"source", "llm_survey"}, {"category", "LLM"}, {"author", "Dr. Kumar"}, {"year", "2023"}, {"difficulty", "advanced"}}
            },
            {
                "Information retrieval systems help users find relevant documents. Vector embeddings enable semantic search capabilities.",
                {{"source", "ir_textbook"}, {"category", "IR"}, {"author", "Dr. Lee"}, {"year", "2022"}, {"difficulty", "beginner"}}
            },
            {
                "Graph neural networks operate on graph-structured data. They are useful for social network analysis and molecular modeling.",
                {{"source", "gnn_review"}, {"category", "GNN"}, {"author", "Dr. Zhang"}, {"year", "2023"}, {"difficulty", "advanced"}}
            },
            {
                "Time series analysis deals with sequential data. LSTM networks can capture temporal dependencies in data.",
                {{"source", "ts_guide"}, {"category", "TS"}, {"author", "Dr. Brown"}, {"year", "2023"}, {"difficulty", "intermediate"}}
            }
        };

        // Add IDs and relevance scores to documents
        for (size_t i = 0; i < documents.size(); ++i) {
            documents[i].id = "doc_" + std::to_string(i + 1);
            documents[i].score = 1.0 - (i * 0.1); // Simulated relevance scores
        }

        std::cout << "Created " << documents.size() << " sample documents" << std::endl;

        // 3. Test BM25 Retrieval
        std::cout << "\n=== BM25 Retrieval ===" << std::endl;

        retrievers::BM25Retriever bm25_retriever;
        bm25_retriever.set_text_processor(text_processor);
        bm25_retriever.set_k1(1.2);
        bm25_retriever.set_b(0.75);

        auto bm25_start = std::chrono::high_resolution_clock::now();
        bm25_retriever.add_documents(documents);
        auto bm25_end = std::chrono::high_resolution_clock::now();

        auto bm25_indexing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            bm25_end - bm25_start).count();

        std::cout << "BM25 indexing time: " << bm25_indexing_time << " ms" << std::endl;

        // Test BM25 queries
        std::vector<std::string> queries = {
            "neural networks deep learning",
            "language understanding text processing",
            "image recognition computer vision",
            "reinforcement learning optimal policies"
        };

        RetrievalConfig config;
        config.top_k = 3;

        for (const auto& query : queries) {
            std::cout << "\nBM25 Query: \"" << query << "\"" << std::endl;
            std::cout << "----------------------------------------" << std::endl;

            auto start_time = std::chrono::high_resolution_clock::now();
            auto results = bm25_retriever.retrieve(query, config);
            auto end_time = std::chrono::high_resolution_clock::now();

            auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time).count();

            std::cout << "Found " << results.size() << " results in " << search_time << " μs" << std::endl;

            for (size_t i = 0; i < results.size(); ++i) {
                const auto& doc = results[i];
                std::cout << (i + 1) << ". " << doc.id
                         << " (score: " << std::fixed << std::setprecision(3) << doc.score << ")" << std::endl;
                std::cout << "   Category: " << doc.metadata.at("category") << std::endl;
                std::cout << "   Author: " << doc.metadata.at("author") << std::endl;
                std::cout << "   Difficulty: " << doc.metadata.at("difficulty") << std::endl;
            }
        }

        // 4. Test Vector Retrieval
        std::cout << "\n=== Vector Retrieval ===" << std::endl;

        retrievers::VectorRetriever vector_retriever;
        vector_retriever.set_text_processor(text_processor);
        vector_retriever.set_embedding_dimension(384);
        vector_retriever.set_similarity_metric(retrievers::VectorRetriever::SimilarityMetric::COSINE);

        auto vector_start = std::chrono::high_resolution_clock::now();
        vector_retriever.add_documents(documents);
        auto vector_end = std::chrono::high_resolution_clock::now();

        auto vector_indexing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            vector_end - vector_start).count();

        std::cout << "Vector indexing time: " << vector_indexing_time << " ms" << std::endl;
        std::cout << "Embedding dimension: " << vector_retriever.embedding_dimension() << std::endl;

        // Test Vector queries (same queries as BM25 for comparison)
        for (const auto& query : queries) {
            std::cout << "\nVector Query: \"" << query << "\"" << std::endl;
            std::cout << "----------------------------------------" << std::endl;

            auto start_time = std::chrono::high_resolution_clock::now();
            auto results = vector_retriever.retrieve(query, config);
            auto end_time = std::chrono::high_resolution_clock::now();

            auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time).count();

            std::cout << "Found " << results.size() << " results in " << search_time << " μs" << std::endl;

            for (size_t i = 0; i < results.size(); ++i) {
                const auto& doc = results[i];
                std::cout << (i + 1) << ". " << doc.id
                         << " (similarity: " << std::fixed << std::setprecision(3) << doc.score << ")" << std::endl;
                std::cout << "   Category: " << doc.metadata.at("category") << std::endl;
                std::cout << "   Year: " << doc.metadata.at("year") << std::endl;
            }
        }

        // 5. Test Hybrid Retrieval
        std::cout << "\n=== Hybrid Retrieval ===" << std::endl;

        retrievers::HybridRetriever hybrid_retriever;
        hybrid_retriever.add_retriever(std::make_shared<retrievers::BM25Retriever>(bm25_retriever));
        hybrid_retriever.add_retriever(std::make_shared<retrievers::VectorRetriever>(vector_retriever));
        hybrid_retriever.set_text_processor(text_processor);
        hybrid_retriever.set_weight_strategy(retrievers::HybridRetriever::WeightStrategy::RECIPROCAL_RANK);

        auto hybrid_start = std::chrono::high_resolution_clock::now();
        hybrid_retriever.add_documents(documents);
        auto hybrid_end = std::chrono::high_resolution_clock::now();

        auto hybrid_indexing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            hybrid_end - hybrid_start).count();

        std::cout << "Hybrid indexing time: " << hybrid_indexing_time << " ms" << std::endl;
        std::cout << "Number of retrievers: " << hybrid_retriever.num_retrievers() << std::endl;

        // Test Hybrid queries
        for (const auto& query : queries) {
            std::cout << "\nHybrid Query: \"" << query << "\"" << std::endl;
            std::cout << "----------------------------------------" << std::endl;

            auto start_time = std::chrono::high_resolution_clock::now();
            auto results = hybrid_retriever.retrieve(query, config);
            auto end_time = std::chrono::high_resolution_clock::now();

            auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time).count();

            std::cout << "Found " << results.size() << " results in " << search_time << " μs" << std::endl;

            for (size_t i = 0; i < results.size(); ++i) {
                const auto& doc = results[i];
                std::cout << (i + 1) << ". " << doc.id
                         << " (hybrid score: " << std::fixed << std::setprecision(3) << doc.score << ")" << std::endl;
                std::cout << "   Category: " << doc.metadata.at("category") << std::endl;
                std::cout << "   Difficulty: " << doc.metadata.at("difficulty") << std::endl;
                std::cout << "   Source: " << doc.metadata.at("source") << std::endl;
            }
        }

        // 6. Performance Comparison
        std::cout << "\n=== Performance Comparison ===" << std::endl;

        std::vector<std::string> test_queries = {
            "machine learning algorithms",
            "neural network architectures",
            "natural language understanding",
            "computer vision applications"
        };

        const int num_iterations = 100;

        std::cout << "Running " << num_iterations << " queries per retriever..." << std::endl;

        // BM25 Performance
        auto bm25_perf_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            for (const auto& query : test_queries) {
                bm25_retriever.retrieve(query, config);
            }
        }
        auto bm25_perf_end = std::chrono::high_resolution_clock::now();
        auto bm25_total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            bm25_perf_end - bm25_perf_start).count();

        // Vector Performance
        auto vector_perf_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            for (const auto& query : test_queries) {
                vector_retriever.retrieve(query, config);
            }
        }
        auto vector_perf_end = std::chrono::high_resolution_clock::now();
        auto vector_total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            vector_perf_end - vector_perf_start).count();

        // Hybrid Performance
        auto hybrid_perf_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            for (const auto& query : test_queries) {
                hybrid_retriever.retrieve(query, config);
            }
        }
        auto hybrid_perf_end = std::chrono::high_resolution_clock::now();
        auto hybrid_total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            hybrid_perf_end - hybrid_perf_start).count();

        std::cout << "\nPerformance Results:" << std::endl;
        std::cout << "BM25 Retriever:     " << std::setw(8) << bm25_total_time << " ms total, "
                  << std::setw(6) << std::fixed << std::setprecision(2)
                  << (double)bm25_total_time / (num_iterations * test_queries.size()) << " ms/query" << std::endl;
        std::cout << "Vector Retriever:   " << std::setw(8) << vector_total_time << " ms total, "
                  << std::setw(6) << std::fixed << std::setprecision(2)
                  << (double)vector_total_time / (num_iterations * test_queries.size()) << " ms/query" << std::endl;
        std::cout << "Hybrid Retriever:   " << std::setw(8) << hybrid_total_time << " ms total, "
                  << std::setw(6) << std::fixed << std::setprecision(2)
                  << (double)hybrid_total_time / (num_iterations * test_queries.size()) << " ms/query" << std::endl;

        // 7. Advanced Configuration
        std::cout << "\n=== Advanced Configuration ===" << std::endl;

        // Test different BM25 parameters
        std::cout << "\nTesting BM25 with different k1 values:" << std::endl;

        retrievers::BM25Retriever bm25_custom;
        bm25_custom.set_text_processor(text_processor);
        bm25_custom.add_documents(documents);

        std::vector<double> k1_values = {0.5, 1.0, 1.5, 2.0};
        std::string test_query = "neural networks machine learning";

        for (double k1 : k1_values) {
            bm25_custom.set_k1(k1);
            auto results = bm25_custom.retrieve(test_query, config);
            std::cout << "k1=" << std::fixed << std::setprecision(1) << k1
                     << ": " << results.size() << " results, top score: "
                     << (results.empty() ? 0.0 : results[0].score) << std::endl;
        }

        // Test different vector similarity metrics
        std::cout << "\nTesting Vector similarity metrics:" << std::endl;

        retrievers::VectorRetriever vector_custom;
        vector_custom.set_text_processor(text_processor);
        vector_custom.add_documents(documents);

        std::vector<std::pair<std::string, retrievers::VectorRetriever::SimilarityMetric>> metrics = {
            {"Cosine", retrievers::VectorRetriever::SimilarityMetric::COSINE},
            {"Dot Product", retrievers::VectorRetriever::SimilarityMetric::DOT_PRODUCT},
            {"Euclidean", retrievers::VectorRetriever::SimilarityMetric::EUCLIDEAN}
        };

        for (const auto& [name, metric] : metrics) {
            vector_custom.set_similarity_metric(metric);
            auto results = vector_custom.retrieve(test_query, config);
            std::cout << name << ": " << results.size() << " results, top similarity: "
                     << (results.empty() ? 0.0 : results[0].score) << std::endl;
        }

        // 8. Memory and Statistics
        std::cout << "\n=== Memory Usage and Statistics ===" << std::endl;

        std::cout << "BM25 Retriever:" << std::endl;
        std::cout << "  Documents: " << bm25_retriever.size() << std::endl;
        std::cout << "  Vocabulary: " << bm25_retriever.vocabulary_size() << " terms" << std::endl;
        std::cout << "  Total tokens: " << bm25_retriever.total_tokens() << std::endl;
        std::cout << "  Average doc length: " << std::fixed << std::setprecision(2)
                  << bm25_retriever.average_document_length() << " tokens" << std::endl;

        std::cout << "\nVector Retriever:" << std::endl;
        std::cout << "  Documents: " << vector_retriever.size() << std::endl;
        std::cout << "  Embedding dimension: " << vector_retriever.embedding_dimension() << std::endl;

        std::cout << "\nHybrid Retriever:" << std::endl;
        std::cout << "  Documents: " << hybrid_retriever.size() << std::endl;
        std::cout << "  Component retrievers: " << hybrid_retriever.num_retrievers() << std::endl;

        std::cout << "\n=== Advanced Retrieval Example completed successfully! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}