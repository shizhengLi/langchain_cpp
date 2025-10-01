#include "langchain/langchain.hpp"
#include "langchain/text/text_processor.hpp"
#include "langchain/retrievers/inverted_index_retriever.hpp"
#include <iostream>
#include <vector>
#include <chrono>

using namespace langchain;

int main() {
    std::cout << "=== LangChain++ Basic Retrieval Example ===" << std::endl;

    try {
        // 1. Create text processor with configuration
        langchain::text::TextProcessor::Config text_config;
        text_config.language = "english";
        text_config.enable_stemming = true;
        text_config.remove_stopwords = true;
        text_config.lowercase = true;

        auto text_processor = std::make_unique<langchain::text::TextProcessor>(text_config);

        // 2. Create retriever configuration
        langchain::retrievers::InvertedIndexRetriever::Config retriever_config;
        retriever_config.min_term_frequency = 1;
        retriever_config.max_results = 10;

        // 3. Create inverted index retriever
        langchain::retrievers::InvertedIndexRetriever retriever(retriever_config, std::move(text_processor));

        // 4. Prepare sample documents
        std::vector<Document> documents = {
            {
                "Artificial intelligence is transforming how we interact with technology. Machine learning algorithms enable computers to learn from data.",
                {{"source", "tech_article"}, {"category", "AI"}, {"page", "1"}}
            },
            {
                "Natural language processing allows computers to understand and generate human language. This includes tasks like translation, summarization, and question answering.",
                {{"source", "nlp_paper"}, {"category", "NLP"}, {"page", "5"}}
            },
            {
                "Deep learning uses neural networks with multiple layers to learn hierarchical representations of data. This has revolutionized computer vision and speech recognition.",
                {{"source", "dl_book"}, {"category", "Deep Learning"}, {"page", "12"}}
            },
            {
                "Information retrieval systems help users find relevant documents from large collections. Search engines use sophisticated algorithms to rank results by relevance.",
                {{"source", "ir_textbook"}, {"category", "IR"}, {"page", "8"}}
            },
            {
                "Vector databases store high-dimensional vectors and enable fast similarity search. They are essential for applications like recommendation systems and semantic search.",
                {{"source", "vector_db_guide"}, {"category", "Vector Search"}, {"page", "3"}}
            }
        };

        // Add IDs to documents
        for (size_t i = 0; i < documents.size(); ++i) {
            documents[i].id = "doc_" + std::to_string(i + 1);
        }

        std::cout << "Adding " << documents.size() << " documents to retriever..." << std::endl;

        // 5. Add documents to retriever
        auto start_time = std::chrono::high_resolution_clock::now();
        auto added_ids = retriever.add_documents(documents);
        auto end_time = std::chrono::high_resolution_clock::now();

        auto indexing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();

        std::cout << "Documents indexed in " << indexing_time << " ms" << std::endl;
        std::cout << "Retriever now contains " << retriever.document_count() << " documents" << std::endl;

        // 6. Perform searches
        std::vector<std::string> queries = {
            "machine learning algorithms",
            "neural networks and deep learning",
            "natural language processing",
            "vector similarity search",
            "information retrieval systems"
        };

        std::cout << "\n=== Search Results ===" << std::endl;

        for (const auto& query : queries) {
            std::cout << "\nQuery: \"" << query << "\"" << std::endl;
            std::cout << "----------------------------------------" << std::endl;

            // Measure search time
            start_time = std::chrono::high_resolution_clock::now();
            auto results = retriever.retrieve(query);
            end_time = std::chrono::high_resolution_clock::now();

            auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time).count();

            std::cout << "Found " << results.documents.size() << " results in " << search_time << " μs" << std::endl;

            // Display results
            for (size_t i = 0; i < results.documents.size(); ++i) {
                const auto& doc = results.documents[i];

                std::cout << "\n" << (i + 1) << ". " << doc.id << std::endl;
                std::cout << "   Score: " << std::fixed << std::setprecision(3) << doc.relevance_score << std::endl;
                std::cout << "   Source: " << doc.metadata.at("source") << std::endl;
                std::cout << "   Category: " << doc.metadata.at("category") << std::endl;

                // Show snippet (first 100 characters)
                std::string snippet = doc.content.substr(0, 100);
                if (doc.content.length() > 100) {
                    snippet += "...";
                }
                std::cout << "   Content: " << snippet << std::endl;
            }
        }

        // 7. Demonstrate advanced features
        std::cout << "\n=== Advanced Features ===" << std::endl;

        // Show retriever document count
        std::cout << "\nRetriever Statistics:" << std::endl;
        std::cout << "  Total documents: " << retriever.document_count() << std::endl;

        // Test partial matching
        std::cout << "\n=== Partial Matching Test ===" << std::endl;
        std::string partial_query = "machine";
        auto partial_results = retriever.retrieve(partial_query);
        std::cout << "Partial query '" << partial_query << "' found " << partial_results.documents.size() << " results:" << std::endl;

        for (size_t i = 0; i < std::min(size_t(3), partial_results.documents.size()); ++i) {
            const auto& doc = partial_results.documents[i];
            std::cout << "  - " << doc.id << ": " << doc.content.substr(0, 80) << "..." << std::endl;
        }

        // Performance test
        std::cout << "\n=== Performance Test ===" << std::endl;
        const int num_queries = 100;
        std::vector<std::string> test_queries = {
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "neural networks",
            "natural language"
        };

        start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_queries; ++i) {
            std::string query = test_queries[i % test_queries.size()];
            retriever.retrieve(query);
        }
        end_time = std::chrono::high_resolution_clock::now();

        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count();

        std::cout << "Performed " << num_queries << " queries in " << total_time << " μs" << std::endl;
        std::cout << "Average query time: " << (double)total_time / num_queries << " μs" << std::endl;

        // Test empty query handling
        std::cout << "\n=== Edge Cases Test ===" << std::endl;
        auto empty_results = retriever.retrieve("");
        std::cout << "Empty query returned " << empty_results.documents.size() << " results" << std::endl;

        // Test non-existent terms
        auto no_results = retriever.retrieve("nonexistent_term_xyz");
        std::cout << "Non-existent term query returned " << no_results.documents.size() << " results" << std::endl;

        std::cout << "\n=== Basic Retrieval Example completed successfully! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}