#pragma once

#include "base_retriever.hpp"
#include "bm25_retriever.hpp"
#include "../vectorstores/simple_vector_store.hpp"
#include "../text/text_processor.hpp"
#include <vector>
#include <memory>
#include <optional>
#include <unordered_map>
#include <algorithm>

namespace langchain::retrievers {

/**
 * @brief Hybrid retrieval result combining sparse and dense scores
 */
struct HybridRetrievalResult {
    size_t document_id;
    std::string content;
    std::unordered_map<std::string, std::string> metadata;

    // Sparse retrieval score (BM25)
    double sparse_score = 0.0;

    // Dense retrieval score (vector similarity)
    double dense_score = 0.0;

    // Combined weighted score
    double hybrid_score = 0.0;

    // Ranking information
    size_t sparse_rank = 0;  // Rank in sparse results
    size_t dense_rank = 0;   // Rank in dense results

    HybridRetrievalResult() = default;
    HybridRetrievalResult(size_t doc_id, std::string cont,
                         std::unordered_map<std::string, std::string> meta = {})
        : document_id(doc_id), content(std::move(cont)), metadata(std::move(meta)) {}
};

/**
 * @brief Hybrid retrieval configuration
 */
struct HybridRetrieverConfig {
    // Weighting parameters
    double sparse_weight = 0.5;    // Weight for BM25 score
    double dense_weight = 0.5;     // Weight for vector similarity

    // Result combination
    size_t top_k_sparse = 20;      // Number of top results from sparse retrieval
    size_t top_k_dense = 20;       // Number of top results from dense retrieval
    size_t top_k_hybrid = 10;      // Final number of results to return

    // Fusion methods
    enum class FusionMethod {
        WEIGHTED_AVERAGE,  // Weighted average of scores
        RRF,              // Reciprocal Rank Fusion
        CONDENSATION,     // Score condensation
        MAX               // Maximum score
    } fusion_method = FusionMethod::WEIGHTED_AVERAGE;

    // RRF parameters
    double rrf_k = 60.0;  // RRF constant (typically 60)

    // Score normalization
    bool normalize_scores = true;
    enum class NormalizationMethod {
        MIN_MAX,    // Min-max normalization
        Z_SCORE,    // Z-score normalization
        SUM         // Sum normalization
    } normalization_method = NormalizationMethod::MIN_MAX;

    // Document deduplication
    bool deduplicate_results = true;
    double deduplication_threshold = 0.95;  // Similarity threshold for deduplication

    // Validation
    void validate() const;
};

/**
 * @brief Hybrid retriever combining sparse and dense retrieval methods
 *
 * This retriever combines BM25 (sparse) and vector similarity (dense) retrieval
 * to provide better search results by leveraging both lexical and semantic information.
 */
class HybridRetriever : public BaseRetriever {
private:
    HybridRetrieverConfig config_;
    std::unique_ptr<BM25Retriever> sparse_retriever_;
    std::unique_ptr<vectorstores::SimpleVectorStore> dense_retriever_;
    std::unique_ptr<text::TextProcessor> text_processor_;

    // Document to vector mapping
    std::unordered_map<std::string, size_t> doc_to_vector_id_;
    std::unordered_map<size_t, std::string> vector_id_to_doc_;
    size_t next_vector_id_{1};

    // Thread safety
    mutable std::shared_mutex mutex_;

public:
    /**
     * @brief Constructor
     * @param config Hybrid retriever configuration
     * @param text_processor Optional text processor
     */
    explicit HybridRetriever(
        const HybridRetrieverConfig& config,
        std::unique_ptr<text::TextProcessor> text_processor = nullptr
    );

    /**
     * @brief Destructor
     */
    ~HybridRetriever() override = default;

    // BaseRetriever interface implementation
    RetrievalResult retrieve(const std::string& query) override;
    std::vector<std::string> add_documents(const std::vector<Document>& documents) override;
    size_t document_count() const override;
    void clear() override;
    std::unordered_map<std::string, std::any> get_metadata() const override;

    /**
     * @brief Add document with embedding
     * @param document Document to add
     * @param embedding Document embedding vector
     * @return Document ID
     */
    std::string add_document_with_embedding(
        const Document& document,
        const std::vector<double>& embedding
    );

    /**
     * @brief Add documents with embeddings
     * @param documents Documents to add
     * @param embeddings Corresponding embeddings
     * @return Document IDs
     */
    std::vector<std::string> add_documents_with_embeddings(
        const std::vector<Document>& documents,
        const std::vector<std::vector<double>>& embeddings
    );

    /**
     * @brief Retrieve with detailed hybrid results
     * @param query Search query
     * @return Detailed hybrid retrieval results
     */
    std::vector<HybridRetrievalResult> retrieve_detailed(const std::string& query);

    /**
     * @brief Get configuration
     * @return Current configuration
     */
    const HybridRetrieverConfig& get_config() const { return config_; }

    /**
     * @brief Update configuration
     * @param new_config New configuration
     */
    void update_config(const HybridRetrieverConfig& new_config);

    /**
     * @brief Get performance statistics
     * @return Performance statistics
     */
    std::unordered_map<std::string, double> get_performance_stats() const;

private:
    /**
     * @brief Perform sparse retrieval (BM25)
     * @param query Query string
     * @return Sparse retrieval results
     */
    std::vector<RetrievedDocument> perform_sparse_retrieval(const std::string& query) const;

    /**
     * @brief Perform dense retrieval (vector similarity)
     * @param query Query string
     * @return Dense retrieval results
     */
    std::vector<RetrievedDocument> perform_dense_retrieval(const std::string& query) const;

    /**
     * @brief Fuse sparse and dense results using specified method
     * @param sparse_results Sparse retrieval results
     * @param dense_results Dense retrieval results
     * @return Fused hybrid results
     */
    std::vector<HybridRetrievalResult> fuse_results(
        const std::vector<RetrievedDocument>& sparse_results,
        const std::vector<RetrievedDocument>& dense_results
    ) const;

    /**
     * @brief Normalize scores
     * @param scores Scores to normalize
     * @param method Normalization method
     * @return Normalized scores
     */
    std::vector<double> normalize_scores(
        const std::vector<double>& scores,
        HybridRetrieverConfig::NormalizationMethod method
    ) const;

    /**
     * @brief Apply Reciprocal Rank Fusion
     * @param sparse_results Sparse results with rankings
     * @param dense_results Dense results with rankings
     * @return RRF scores
     */
    std::vector<double> apply_rrf_fusion(
        const std::vector<std::pair<size_t, double>>& sparse_results,
        const std::vector<std::pair<size_t, double>>& dense_results
    ) const;

    /**
     * @brief Deduplicate results
     * @param results Results to deduplicate
     * @return Deduplicated results
     */
    std::vector<HybridRetrievalResult> deduplicate_results(
        std::vector<HybridRetrievalResult> results
    ) const;

    /**
     * @brief Calculate document similarity for deduplication
     * @param doc1 First document
     * @param doc2 Second document
     * @return Similarity score
     */
    double calculate_document_similarity(
        const std::string& doc1,
        const std::string& doc2
    ) const;

    /**
     * @brief Generate document embedding using text processing
     * @param text Document text
     * @return Embedding vector (mock implementation)
     */
    std::vector<double> generate_embedding(const std::string& text) const;

    /**
     * @brief Convert RetrievedDocument to HybridRetrievalResult
     * @param doc RetrievedDocument
     * @param sparse_score Sparse score
     * @param dense_score Dense score
     * @param sparse_rank Sparse rank
     * @param dense_rank Dense rank
     * @return HybridRetrievalResult
     */
    HybridRetrievalResult convert_to_hybrid_result(
        const RetrievedDocument& doc,
        double sparse_score,
        double dense_score,
        size_t sparse_rank,
        size_t dense_rank
    ) const;
};

/**
 * @brief Factory for creating hybrid retrievers
 */
class HybridRetrieverFactory {
public:
    /**
     * @brief Create standard hybrid retriever
     * @param config Optional configuration
     * @return Unique pointer to hybrid retriever
     */
    static std::unique_ptr<HybridRetriever> create_standard(
        const HybridRetrieverConfig& config = HybridRetrieverConfig{}
    );

    /**
     * @brief Create hybrid retriever optimized for specific use case
     * @param use_case Use case type ("search", "recommendation", "qa")
     * @param vector_dim Vector dimension for embeddings
     * @return Unique pointer to hybrid retriever
     */
    static std::unique_ptr<HybridRetriever> create_optimized(
        const std::string& use_case,
        size_t vector_dim = 384
    );

    /**
     * @brief Create hybrid retriever with custom components
     * @param sparse_retriever Sparse retriever instance
     * @param dense_retriever Dense retriever instance
     * @param config Hybrid configuration
     * @return Unique pointer to hybrid retriever
     */
    static std::unique_ptr<HybridRetriever> create_with_components(
        std::unique_ptr<BM25Retriever> sparse_retriever,
        std::unique_ptr<vectorstores::SimpleVectorStore> dense_retriever,
        const HybridRetrieverConfig& config = HybridRetrieverConfig{}
    );
};

} // namespace langchain::retrievers