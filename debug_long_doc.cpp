#include "langchain/retrievers/inverted_index_retriever.hpp"
#include <iostream>

using namespace langchain::retrievers;
using namespace langchain;

int main() {
    std::cout << "=== Debug Long Document Issue ===" << std::endl;

    // Create retriever
    auto retriever = InvertedIndexRetrieverFactory::create_retrieval_retriever();

    // Test case from failing test
    std::string long_content(30, 'a');  // 30 'a' characters
    std::cout << "Document content: '" << long_content << "'" << std::endl;
    std::cout << "Content length: " << long_content.length() << std::endl;

    // Test with shorter content within max_token_length
    std::string short_content(15, 'a');  // 15 'a' characters
    std::cout << "\nShort content: '" << short_content << "'" << std::endl;
    std::cout << "Short content length: " << short_content.length() << std::endl;

    Document doc(long_content);
    Document doc2(short_content);
    retriever->add_documents({doc, doc2});

    std::cout << "Document count: " << retriever->document_count() << std::endl;

    // Check frequent terms
    auto frequent_terms = retriever->get_most_frequent_terms(10);
    std::cout << "\nFrequent terms:" << std::endl;
    for (const auto& [term, freq] : frequent_terms) {
        std::cout << "'" << term << "': " << freq << std::endl;
    }

    // Check postings for "a"
    auto postings = retriever->get_postings("a");
    std::cout << "\nPostings for 'a':" << std::endl;
    for (const auto& posting : postings) {
        std::cout << "Doc ID: " << posting.document_id << ", TF: " << posting.term_frequency << std::endl;
    }

    // Check term info for "a"
    auto term_info = retriever->get_term_info("a");
    std::cout << "\nTerm info for 'a':" << std::endl;
    std::cout << "Document frequency: " << term_info.document_frequency << std::endl;
    std::cout << "Total term frequency: " << term_info.total_term_frequency << std::endl;
    std::cout << "IDF: " << term_info.idf << std::endl;

    // Test text processor
    std::cout << "\nTesting text processor:" << std::endl;
    auto text_processor = langchain::text::TextProcessorFactory::create_retrieval_processor();

    auto doc_tokens = text_processor->process(long_content);
    std::cout << "Long document tokens:" << std::endl;
    for (const auto& token : doc_tokens) {
        std::cout << "'" << token << "'" << std::endl;
    }

    auto short_tokens = text_processor->process(short_content);
    std::cout << "\nShort document tokens:" << std::endl;
    for (const auto& token : short_tokens) {
        std::cout << "'" << token << "'" << std::endl;
    }

    auto query_tokens = text_processor->process("a");
    std::cout << "\nQuery tokens for 'a':" << std::endl;
    for (const auto& token : query_tokens) {
        std::cout << "'" << token << "'" << std::endl;
    }

    // Test retrieval
    std::cout << "\nTesting retrieval:" << std::endl;
    auto result = retriever->retrieve("a");
    std::cout << "Query: " << result.query << std::endl;
    std::cout << "Total results: " << result.total_results << std::endl;
    std::cout << "Documents found: " << result.documents.size() << std::endl;

    for (const auto& doc : result.documents) {
        std::cout << "Score: " << doc.relevance_score << ", Content: " << doc.content << std::endl;
    }

    std::cout << "\n=== End Debug ===" << std::endl;
    return 0;
}