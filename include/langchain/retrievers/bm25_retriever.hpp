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
 * @brief BM25 (Best Match 25) retriever implementation
 *
 * This implementation uses the BM25 algorithm for document scoring, which is
 * considered one of the most effective and widely used ranking functions in
 * information retrieval. BM25 improves upon TF-IDF by incorporating document
 * length normalization and parameter tuning.
 *
 * Key features:
 * - Configurable BM25 parameters (k1, b, delta)
 * - Statistical optimization for query performance
 * - Thread-safe operations with read-write locks
 * - Cache-friendly posting list organization
 * - Support for field-level weighting
 */
class BM25Retriever : public BaseRetriever {
public:
    /**
     * @brief Configuration for BM25 retrieval
     */
    struct Config {
        // BM25 parameters
        double k1 = 1.2;                    // Term frequency saturation parameter
        double b = 0.75;                    // Document length normalization parameter
        double delta = 1.0;                 // Query term normalization parameter

        // Indexing parameters
        size_t min_term_frequency = 1;      // Minimum term frequency to index
        size_t max_postings_per_term = 100000; // Maximum postings per term
        bool enable_term_caching = true;     // Enable term frequency caching
        size_t cache_size_limit = 10000;     // Maximum cached terms

        // Scoring parameters
        bool normalize_scores = true;        // Normalize relevance scores
        double score_threshold = 0.0;        // Minimum score threshold
        size_t max_results = 100;            // Maximum results to return

        // Document parameters
        bool enable_field_weighting = false; // Enable field-level weighting
        std::unordered_map<std::string, double> field_weights; // Field importance

        // Performance parameters
        size_t doc_length_cache_size = 1000; // Document length cache size
        bool use_optimized_scoring = true;   // Use optimized scoring functions
    };

    /**
     * @brief Enhanced posting entry with statistical information
     */
    struct PostingEntry {
        size_t document_id;                 // Document identifier
        size_t term_frequency;              // Term frequency in document
        std::vector<size_t> positions;      // Term positions (optional)
        double normalized_tf;               // Normalized term frequency

        PostingEntry(size_t doc_id, size_t tf, std::vector<size_t> pos = {})
            : document_id(doc_id), term_frequency(tf), positions(std::move(pos)) {
            normalized_tf = 0.0; // Will be calculated during indexing
        }
    };

    /**
     * @brief Enhanced term information with BM25 statistics
     */
    struct TermInfo {
        std::vector<PostingEntry> postings; // Posting list
        size_t document_frequency = 0;      // Number of documents containing term
        size_t total_term_frequency = 0;    // Total term frequency across corpus
        double idf = 0.0;                  // Inverse document frequency
        double avg_document_length = 0.0;   // Average document length for this term
        size_t last_accessed = 0;           // Cache access timestamp

        /**
         * @brief Update cached IDF value using BM25 formula
         * @param total_docs Total number of documents in corpus
         */
        void update_idf(size_t total_docs) {
            if (document_frequency > 0 && document_frequency < total_docs) {
                // BM25 IDF formula: log((N - df + 0.5) / (df + 0.5))
                // Add 1.0 to ensure positive IDF for common cases
                idf = std::log(1.0 + (static_cast<double>(total_docs) - document_frequency + 0.5) /
                              (document_frequency + 0.5));
            } else if (document_frequency == total_docs) {
                // Terms appearing in all documents get minimal IDF
                idf = 0.01;
            } else {
                idf = 0.0;
            }
        }
    };

    /**
     * @brief Document statistics for length normalization
     */
    struct DocumentStats {
        size_t document_id = 0;
        size_t term_count = 0;              // Total terms in document
        size_t unique_terms = 0;            // Number of unique terms
        double bm25_normalization_factor = 1.0; // BM25 normalization factor

        DocumentStats() = default;
        DocumentStats(size_t doc_id) : document_id(doc_id) {}
    };

private:
    Config config_;
    std::unique_ptr<text::TextProcessor> text_processor_;

    // Core index structures
    std::unordered_map<std::string, TermInfo> inverted_index_;
    std::vector<Document> documents_;       // Document storage
    std::unordered_map<std::string, size_t> doc_id_map_;
    std::vector<DocumentStats> document_stats_; // Document statistics

    // Performance optimization structures
    mutable std::shared_mutex index_mutex_;
    std::atomic<size_t> next_doc_id_{1};
    mutable std::atomic<size_t> cache_timestamp_{0};

    // Statistics tracking
    mutable std::atomic<size_t> total_queries_{0};
    mutable std::atomic<size_t> cache_hits_{0};
    mutable std::atomic<size_t> cache_misses_{0};

    // BM25 specific statistics
    double avg_document_length_;           // Average document length in corpus
    size_t total_terms_in_corpus_;         // Total terms across all documents

public:
    /**
     * @brief Constructor with text processor
     * @param config BM25 retriever configuration
     * @param text_processor Text processor for tokenization
     */
    explicit BM25Retriever(
        const Config& config,
        std::unique_ptr<text::TextProcessor> text_processor = nullptr);

    /**
     * @brief Destructor
     */
    ~BM25Retriever() override = default;

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
     * @brief Get document statistics
     * @param doc_id Document identifier
     * @return Document statistics (empty if not found)
     */
    std::optional<DocumentStats> get_document_stats(size_t doc_id) const;

    /**
     * @brief Get most frequent terms in the corpus
     * @param limit Number of terms to return
     * @return Vector of (term, frequency) pairs
     */
    std::vector<std::pair<std::string, size_t>> get_most_frequent_terms(size_t limit = 10) const;

    /**
     * @brief Retrieve documents using SIMD-optimized TF-IDF scoring
     * @param query Search query
     * @return Retrieval result with TF-IDF scores
     */
    RetrievalResult retrieve_tfidf_simd(const std::string& query);

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

    /**
     * @brief Calculate BM25 parameters for query optimization
     * @return Map with parameter information
     */
    std::unordered_map<std::string, double> get_bm25_parameters() const;

private:
    /**
     * @brief Process query and return term frequencies
     * @param query Query string
     * @return Map of term -> frequency
     */
    std::unordered_map<std::string, size_t> process_query(const std::string& query) const;

    /**
     * @brief Calculate BM25 score for document
     * @param query_terms Query term frequencies
     * @param doc_id Document identifier
     * @return BM25 relevance score
     */
    double calculate_bm25_score(
        const std::unordered_map<std::string, size_t>& query_terms,
        size_t doc_id) const;

    /**
     * @brief Calculate normalized term frequency for BM25
     * @param raw_tf Raw term frequency
     * @param doc_id Document identifier
     * @return Normalized term frequency
     */
    double calculate_normalized_tf(size_t raw_tf, size_t doc_id) const;

    /**
     * @brief Calculate document length normalization factor
     * @param doc_id Document identifier
     * @return Normalization factor
     */
    double calculate_doc_normalization_factor(size_t doc_id) const;

    /**
     * @brief Calculate TF-IDF scores using SIMD optimization
     * @param query_terms Processed query terms with frequencies
     * @param doc_ids Vector of document IDs to score
     * @param scores Output vector for TF-IDF scores
     */
    void calculate_tfidf_scores_simd(
        const std::unordered_map<std::string, size_t>& query_terms,
        const std::vector<size_t>& doc_ids,
        std::vector<double>& scores) const;

    /**
     * @brief Perform intersection of posting lists (optimized)
     * @param terms Vector of terms to intersect
     * @return Vector of document IDs containing all terms
     */
    std::vector<size_t> intersect_postings_optimized(
        const std::vector<std::string>& terms) const;

    /**
     * @brief Perform union of posting lists (optimized)
     * @param terms Vector of terms to union
     * @return Vector of document IDs containing any term
     */
    std::vector<size_t> union_postings_optimized(
        const std::vector<std::string>& terms) const;

    /**
     * @brief Add document to index (internal method)
     * @param document Document to add
     * @return Document ID
     */
    size_t add_document_internal(const Document& document);

    /**
     * @brief Update document statistics
     * @param doc_id Document identifier
     * @param term_count Number of terms in document
     * @param unique_terms Number of unique terms
     */
    void update_document_statistics(size_t doc_id, size_t term_count, size_t unique_terms);

    /**
     * @brief Update corpus statistics
     * Called after adding documents to recalculate averages
     */
    void update_corpus_statistics();

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
 * @brief Factory for creating pre-configured BM25 retrievers
 */
class BM25RetrieverFactory {
public:
    /**
     * @brief Create a standard BM25 retriever
     */
    static std::unique_ptr<BM25Retriever> create_standard_retriever();

    /**
     * @brief Create a BM25 retriever optimized for short documents (tweets, titles)
     */
    static std::unique_ptr<BM25Retriever> create_short_document_retriever();

    /**
     * @brief Create a BM25 retriever optimized for long documents (articles, books)
     */
    static std::unique_ptr<BM25Retriever> create_long_document_retriever();

    /**
     * @brief Create a BM25 retriever optimized for high precision
     */
    static std::unique_ptr<BM25Retriever> create_precision_focused_retriever();

    /**
     * @brief Create a BM25 retriever optimized for high recall
     */
    static std::unique_ptr<BM25Retriever> create_recall_focused_retriever();
};

} // namespace langchain::retrievers