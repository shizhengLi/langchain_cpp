#include "langchain/retrievers/inverted_index_retriever.hpp"
#include <iostream>
#include <cassert>

using namespace langchain::retrievers;
using namespace langchain;

int main() {
    std::cout << "=== Debug InvertedIndexRetriever ===" << std::endl;

    // Create retriever
    auto retriever = InvertedIndexRetrieverFactory::create_retrieval_retriever();

    // Add test documents like the failing test
    std::vector<Document> docs = {
        Document("Document about apples and fruits"),
        Document("Document about bananas and fruits"),
        Document("Document about oranges and citrus")
    };

    std::cout << "Adding document..." << std::endl;
    auto doc_ids = retriever->add_documents(docs);
    std::cout << "Document count: " << retriever->document_count() << std::endl;
    std::cout << "Document IDs: ";
    for (const auto& id : doc_ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    // Check index state
    std::cout << "\nIndex state:" << std::endl;
    auto metadata = retriever->get_metadata();
    for (const auto& [key, value] : metadata) {
        if (key == "type") {
            std::cout << key << ": " << std::any_cast<std::string>(value) << std::endl;
        } else if (key == "ready" || key == "cache_enabled") {
            std::cout << key << ": " << std::any_cast<bool>(value) << std::endl;
        } else if (key == "total_terms" || key == "total_postings" || key == "document_count" || key == "total_queries") {
            std::cout << key << ": " << std::any_cast<size_t>(value) << std::endl;
        }
    }

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

    // Check postings for each query term
    for (const auto& query : queries) {
        auto postings = retriever->get_postings(query);
        std::cout << "\nPostings for '" << query << "':" << std::endl;
        for (const auto& posting : postings) {
            std::cout << "Doc ID: " << posting.document_id << ", TF: " << posting.term_frequency << std::endl;
        }
    }

    // Try retrieval like the failing test
    std::cout << "\nTesting batch retrieval:" << std::endl;
    auto results = retriever->retrieve_batch(queries);

    std::cout << "Batch results: " << results.size() << std::endl;
    for (size_t i = 0; i < queries.size(); ++i) {
        std::cout << "Query '" << queries[i] << "': " << results[i].documents.size() << " documents" << std::endl;
        if (!results[i].documents.empty()) {
            std::cout << "  First result: " << results[i].documents[0].content << std::endl;
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

    // Test text processor directly
    std::cout << "\nTesting text processor:" << std::endl;
    auto text_processor = langchain::text::TextProcessorFactory::create_retrieval_processor();
    auto tokens = text_processor->tokenize("machine learning");
    std::cout << "Tokens for 'machine learning':" << std::endl;
    for (const auto& token : tokens) {
        std::cout << "'" << token << "'" << std::endl;
    }

    // Test query tokenization
    auto query_tokens = text_processor->tokenize("machine");
    std::cout << "\nTokens for query 'machine':" << std::endl;
    for (const auto& token : query_tokens) {
        std::cout << "'" << token << "'" << std::endl;
    }

    // Test processed version
    auto processed = text_processor->process("machine");
    std::cout << "\nProcessed query 'machine':" << std::endl;
    for (const auto& token : processed) {
        std::cout << "'" << token << "'" << std::endl;
    }

    std::cout << "\n=== End Debug ===" << std::endl;
    return 0;
}