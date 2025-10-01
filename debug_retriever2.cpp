#include "langchain/retrievers/inverted_index_retriever.hpp"
#include <iostream>
#include <cassert>

using namespace langchain::retrievers;
using namespace langchain;

int main() {
    std::cout << "=== Debug Batch Operations Issue ===" << std::endl;

    // Create retriever
    auto retriever = InvertedIndexRetrieverFactory::create_retrieval_retriever();

    // Add test documents like the failing test
    std::vector<Document> docs = {
        Document("Document about apples and fruits"),
        Document("Document about bananas and fruits"),
        Document("Document about oranges and citrus")
    };

    std::cout << "Adding documents..." << std::endl;
    auto doc_ids = retriever->add_documents(docs);
    std::cout << "Document count: " << retriever->document_count() << std::endl;

    // Check frequent terms
    auto frequent_terms = retriever->get_most_frequent_terms(20);
    std::cout << "\nFrequent terms:" << std::endl;
    for (const auto& [term, freq] : frequent_terms) {
        std::cout << term << ": " << freq << std::endl;
    }

    // Test queries like the failing test
    std::vector<std::string> queries = {"apples", "bananas", "oranges"};

    // Check postings for each query term
    for (const auto& query : queries) {
        auto postings = retriever->get_postings(query);
        std::cout << "\nPostings for '" << query << "':" << std::endl;
        for (const auto& posting : postings) {
            std::cout << "Doc ID: " << posting.document_id << ", TF: " << posting.term_frequency << std::endl;
        }
    }

    // Test text processor directly
    std::cout << "\nTesting text processor:" << std::endl;
    auto text_processor = langchain::text::TextProcessorFactory::create_retrieval_processor();

    for (const auto& query : queries) {
        auto tokens = text_processor->tokenize(query);
        std::cout << "Tokens for query '" << query << "':" << std::endl;
        for (const auto& token : tokens) {
            std::cout << "'" << token << "'" << std::endl;
        }
    }

    // Test individual queries
    std::cout << "\nTesting individual queries:" << std::endl;
    for (const auto& query : queries) {
        auto result = retriever->retrieve(query);
        std::cout << "Query '" << query << "': " << result.documents.size() << " documents" << std::endl;
        for (const auto& doc : result.documents) {
            std::cout << "  Score: " << doc.relevance_score << ", Content: " << doc.content << std::endl;
        }
    }

    // Test batch retrieval
    std::cout << "\nTesting batch retrieval:" << std::endl;
    auto results = retriever->retrieve_batch(queries);

    std::cout << "Batch results: " << results.size() << std::endl;
    for (size_t i = 0; i < queries.size(); ++i) {
        std::cout << "Query '" << queries[i] << "': " << results[i].documents.size() << " documents" << std::endl;
        std::cout << "  Query field: '" << results[i].query << "'" << std::endl;
        if (!results[i].documents.empty()) {
            std::cout << "  First result: " << results[i].documents[0].content << std::endl;
        }
    }

    std::cout << "\n=== End Debug ===" << std::endl;
    return 0;
}