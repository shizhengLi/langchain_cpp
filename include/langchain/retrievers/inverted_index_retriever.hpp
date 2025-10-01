#pragma once

#include "base_retriever.hpp"
#include "../text/text_processor.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <algorithm>
#include <numeric>

namespace langchain::retrievers {

/**
 * @brief Cache-friendly inverted index for efficient document retrieval
 *
 * This implementation focuses on memory locality and cache efficiency:
 * - Uses contiguous memory storage for posting lists
 * - Implements term-frequency caching
 * - Provides thread-safe operations with read-write locks
 * - Optimizes for both insertion and query performance
 */
class InvertedIndexRetriever : public BaseRetriever {
public:
    /**
     * @brief Configuration for inverted index
     */
    struct Config {
        size_t min_term_frequency = 1;        // Minimum term frequency to index
        size_t max_postings_per_term = 100000; // Maximum postings per term
        bool enable_term_caching = true;       // Enable term frequency caching
        size_t cache_size_limit = 10000;       // Maximum cached terms
        bool normalize_scores = true;          // Normalize relevance scores
        double score_threshold = 0.0;          // Minimum score threshold
        size_t max_results = 100;              // Maximum results to return
        std::string default_field = "content"; // Default document field
    };

    /**
     * @brief Posting list entry with cache-friendly layout
     */
    struct PostingEntry {
        size_t document_id;        // Document identifier
        size_t term_frequency;     // Term frequency in document
        std::vector<size_t> positions; // Term positions (optional)

        PostingEntry(size_t doc_id, size_t tf, std::vector<size_t> pos = {})
            : document_id(doc_id), term_frequency(tf), positions(std::move(pos)) {}
    };

    /**
     * @brief Term information with cached statistics
     */
    struct TermInfo {
        std::vector<PostingEntry> postings;      // Posting list
        size_t document_frequency = 0;           // Number of documents containing term
        size_t total_term_frequency = 0;         // Total term frequency across corpus
        double idf = 0.0;                        // Inverse document frequency (cached)
        size_t last_accessed = 0;                // Cache access timestamp

        /**
         * @brief Update cached IDF value
         * @param total_docs Total number of documents in corpus
         */
        void update_idf(size_t total_docs) {
            if (document_frequency > 0) {
                // IDF smoothing to avoid zero values in small corpora
                idf = std::log((static_cast<double>(total_docs) + 1.0) / document_frequency) + 1.0;
            } else {
                idf = 0.0;
            }
        }
    };

private:
    Config config_;
    std::unique_ptr<text::TextProcessor> text_processor_;

    // Core inverted index data structures
    std::unordered_map<std::string, TermInfo> inverted_index_;

    // Document storage for cache-friendly access
    std::vector<Document> documents_;           // Contiguous document storage
    std::unordered_map<std::string, size_t> doc_id_map_; // ID to index mapping

    // Performance optimization structures
    mutable std::shared_mutex index_mutex_;     // Read-write lock for thread safety
    std::atomic<size_t> next_doc_id_{1};       // Atomic document ID generator
    mutable std::atomic<size_t> cache_timestamp_{0}; // LRU cache timestamp

    // Statistics tracking
    mutable std::atomic<size_t> total_queries_{0};
    mutable std::atomic<size_t> cache_hits_{0};
    mutable std::atomic<size_t> cache_misses_{0};

public:
    /**
     * @brief Constructor with text processor
     * @param config Index configuration
     * @param text_processor Text processor for tokenization
     */
    explicit InvertedIndexRetriever(
        const Config& config,
        std::unique_ptr<text::TextProcessor> text_processor = nullptr);

    /**
     * @brief Destructor
     */
    ~InvertedIndexRetriever() override = default;

    // BaseRetriever interface implementation
    RetrievalResult retrieve(const std::string& query) override;
    std::vector<std::string> add_documents(const std::vector<Document>& documents) override;
    size_t document_count() const override;
    void clear() override;

    /**
     * @brief Get detailed index statistics
     * @return Map with statistical information
     */
    std::unordered_map<std::string, std::any> get_metadata() const override;

    /**
     * @brief Get posting list for a term (for debugging/analysis)
     * @param term Search term
     * @return Vector of posting entries (empty if term not found)
     */
    std::vector<PostingEntry> get_postings(const std::string& term);

    /**
     * @brief Get term information (for debugging/analysis)
     * @param term Search term
     * @return Term info (empty if term not found)
     */
    TermInfo get_term_info(const std::string& term);

    /**
     * @brief Get most frequent terms in the corpus
     * @param limit Number of terms to return
     * @return Vector of (term, frequency) pairs
     */
    std::vector<std::pair<std::string, size_t>> get_most_frequent_terms(size_t limit = 10) const;

    /**
     * @brief Optimize index for better query performance
     * - Sorts posting lists by document frequency
     * - Updates cached statistics
     * - Implements cache cleanup if needed
     */
    void optimize_index();

    /**
     * @brief Get cache performance statistics
     * @return Map with cache statistics
     */
    std::unordered_map<std::string, double> get_cache_stats() const;

    /**
     * @brief Update configuration
     * @param new_config New configuration settings
     */
    void update_config(const Config& new_config);

    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const Config& get_config() const { return config_; }

private:
    /**
     * @brief Process query and return term frequencies
     * @param query Query string
     * @return Map of term -> frequency
     */
    std::unordered_map<std::string, size_t> process_query(const std::string& query) const;

    /**
     * @brief Calculate relevance score for document
     * @param query_terms Query term frequencies
     * @param doc_id Document identifier
     * @return Relevance score
     */
    double calculate_score(
        const std::unordered_map<std::string, size_t>& query_terms,
        size_t doc_id) const;

    /**
     * @brief Perform intersection of posting lists
     * @param terms Vector of terms to intersect
     * @return Vector of document IDs containing all terms
     */
    std::vector<size_t> intersect_postings(const std::vector<std::string>& terms) const;

    /**
     * @brief Perform union of posting lists
     * @param terms Vector of terms to union
     * @return Vector of document IDs containing any term
     */
    std::vector<size_t> union_postings(const std::vector<std::string>& terms) const;

    /**
     * @brief Add document to index (internal method)
     * @param document Document to add
     * @return Document ID
     */
    size_t add_document_internal(const Document& document);

    /**
     * @brief Update term cache statistics
     * @param term Term that was accessed
     */
    void update_cache_stats(const std::string& term);

    /**
     * @brief Cleanup least recently used cache entries
     */
    void cleanup_cache();

    /**
     * @brief Generate document ID
     * @return Unique document ID
     */
    size_t generate_document_id();

    /**
     * @brief Get document by ID
     * @param doc_id Document identifier
     * @return Document (empty if not found)
     */
    std::optional<Document> get_document(size_t doc_id) const;
};

/**
 * @brief Factory for creating pre-configured inverted index retrievers
 */
class InvertedIndexRetrieverFactory {
public:
    /**
     * @brief Create a retriever optimized for document retrieval
     */
    static std::unique_ptr<InvertedIndexRetriever> create_retrieval_retriever();

    /**
     * @brief Create a retriever optimized for search queries
     */
    static std::unique_ptr<InvertedIndexRetriever> create_search_retriever();

    /**
     * @brief Create a retriever optimized for large datasets
     */
    static std::unique_ptr<InvertedIndexRetriever> create_large_dataset_retriever();

    /**
     * @brief Create a retriever optimized for memory efficiency
     */
    static std::unique_ptr<InvertedIndexRetriever> create_memory_efficient_retriever();
};

} // namespace langchain::retrievers