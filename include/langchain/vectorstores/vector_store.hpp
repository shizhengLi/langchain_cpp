#pragma once

#include "../core/base.hpp"
#include "../utils/simd_ops.hpp"
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <algorithm>

namespace langchain::vectorstores {

/**
 * @brief Vector representation for embeddings
 */
struct Vector {
    std::vector<double> data;

    Vector() = default;
    explicit Vector(size_t size) : data(size, 0.0) {}
    Vector(std::vector<double> d) : data(std::move(d)) {}

    size_t size() const { return data.size(); }
    double& operator[](size_t idx) { return data[idx]; }
    const double& operator[](size_t idx) const { return data[idx]; }

    // Vector operations
    double dot(const Vector& other) const;
    double magnitude() const;
    double cosine_similarity(const Vector& other) const;
    double euclidean_distance(const Vector& other) const;

    // SIMD operations
    Vector add_simd(const Vector& other) const;
    Vector subtract_simd(const Vector& other) const;
    Vector multiply_simd(double scalar) const;
    double dot_simd(const Vector& other) const;
};

/**
 * @brief Vector entry with metadata
 */
struct VectorEntry {
    size_t id;
    Vector vector;
    std::string content;
    std::unordered_map<std::string, std::string> metadata;

    VectorEntry(size_t i, Vector v, std::string c = "")
        : id(i), vector(std::move(v)), content(std::move(c)) {}

    VectorEntry(size_t i, Vector v, std::unordered_map<std::string, std::string> meta)
        : id(i), vector(std::move(v)), metadata(std::move(meta)) {}
};

/**
 * @brief Similarity search result
 */
struct SimilarityResult {
    size_t entry_id;
    double similarity_score;
    std::string content;
    std::unordered_map<std::string, std::string> metadata;

    SimilarityResult(size_t id, double score, std::string content = "",
                    std::unordered_map<std::string, std::string> metadata = {})
        : entry_id(id), similarity_score(score), content(std::move(content)),
          metadata(std::move(metadata)) {}
};

/**
 * @brief Vector store configuration
 */
struct VectorStoreConfig {
    // Indexing parameters
    size_t max_vectors = 1000000;        // Maximum vectors in store
    size_t vector_dim = 384;             // Default embedding dimension
    bool normalize_vectors = true;       // Normalize vectors for cosine similarity

    // Search parameters
    size_t default_top_k = 10;           // Default number of results
    double similarity_threshold = 0.0;   // Minimum similarity threshold

    // Performance parameters
    bool use_simd = true;                // Use SIMD for vector operations
    bool cache_enabled = true;           // Enable search caching
    size_t cache_size = 1000;            // Search result cache size

    // Distance metric
    enum class DistanceMetric {
        COSINE,
        EUCLIDEAN,
        DOT_PRODUCT
    } distance_metric = DistanceMetric::COSINE;

    // Validation
    void validate() const;
};

/**
 * @brief Base interface for vector stores
 */
class BaseVectorStore {
public:
    virtual ~BaseVectorStore() = default;

    /**
     * @brief Add vector to store
     * @param vector Vector to add
     * @param content Associated content (optional)
     * @param metadata Additional metadata (optional)
     * @return Vector entry ID
     */
    virtual size_t add_vector(
        const Vector& vector,
        const std::string& content = "",
        const std::unordered_map<std::string, std::string>& metadata = {}
    ) = 0;

    /**
     * @brief Add multiple vectors
     * @param vectors Vectors to add
     * @return Vector of entry IDs
     */
    virtual std::vector<size_t> add_vectors(
        const std::vector<VectorEntry>& vectors
    ) = 0;

    /**
     * @brief Search for similar vectors
     * @param query_vector Query vector
     * @param top_k Number of results to return
     * @return Similarity search results
     */
    virtual std::vector<SimilarityResult> search(
        const Vector& query_vector,
        size_t top_k = 10
    ) = 0;

    /**
     * @brief Search with similarity threshold
     * @param query_vector Query vector
     * @param similarity_threshold Minimum similarity score
     * @param top_k Maximum number of results
     * @return Similarity search results
     */
    virtual std::vector<SimilarityResult> search_with_threshold(
        const Vector& query_vector,
        double similarity_threshold,
        size_t top_k = 10
    ) = 0;

    /**
     * @brief Get vector by ID
     * @param id Vector entry ID
     * @return Vector entry or empty if not found
     */
    virtual std::optional<VectorEntry> get_vector(size_t id) const = 0;

    /**
     * @brief Update vector
     * @param id Vector entry ID
     * @param vector New vector
     * @param content New content (optional)
     * @param metadata New metadata (optional)
     * @return True if updated successfully
     */
    virtual bool update_vector(
        size_t id,
        const Vector& vector,
        const std::string& content = "",
        const std::unordered_map<std::string, std::string>& metadata = {}
    ) = 0;

    /**
     * @brief Delete vector
     * @param id Vector entry ID
     * @return True if deleted successfully
     */
    virtual bool delete_vector(size_t id) = 0;

    /**
     * @brief Get number of vectors in store
     * @return Vector count
     */
    virtual size_t size() const = 0;

    /**
     * @brief Check if store is empty
     * @return True if empty
     */
    virtual bool empty() const = 0;

    /**
     * @brief Clear all vectors
     */
    virtual void clear() = 0;

    /**
     * @brief Get vector dimension
     * @return Vector dimension
     */
    virtual size_t get_dimension() const = 0;

    /**
     * @brief Get store metadata
     * @return Metadata map
     */
    virtual std::unordered_map<std::string, std::any> get_metadata() const = 0;

    /**
     * @brief Optimize index for better search performance
     */
    virtual void optimize_index() = 0;
};

/**
 * @brief In-memory vector store implementation with SIMD optimization
 */
class InMemoryVectorStore : public BaseVectorStore {
private:
    VectorStoreConfig config_;
    std::vector<VectorEntry> vectors_;
    std::atomic<size_t> next_id_{1};
    mutable std::shared_mutex mutex_;

    // Search cache
    mutable std::mutex cache_mutex_;
    mutable std::unordered_map<size_t, std::vector<SimilarityResult>> search_cache_;
    mutable std::vector<size_t> cache_access_order_;
    mutable size_t cache_timestamp_{0};

    // Statistics
    mutable std::atomic<size_t> total_searches_{0};
    mutable std::atomic<size_t> cache_hits_{0};

public:
    explicit InMemoryVectorStore(const VectorStoreConfig& config = VectorStoreConfig{});
    virtual ~InMemoryVectorStore() = default;

    // BaseVectorStore interface implementation
    size_t add_vector(
        const Vector& vector,
        const std::string& content = "",
        const std::unordered_map<std::string, std::string>& metadata = {}
    ) override;

    std::vector<size_t> add_vectors(
        const std::vector<VectorEntry>& vectors
    ) override;

    std::vector<SimilarityResult> search(
        const Vector& query_vector,
        size_t top_k = 10
    ) override;

    std::vector<SimilarityResult> search_with_threshold(
        const Vector& query_vector,
        double similarity_threshold,
        size_t top_k = 10
    ) override;

    std::optional<VectorEntry> get_vector(size_t id) const override;

    bool update_vector(
        size_t id,
        const Vector& vector,
        const std::string& content = "",
        const std::unordered_map<std::string, std::string>& metadata = {}
    ) override;

    bool delete_vector(size_t id) override;

    size_t size() const override;
    bool empty() const override;
    void clear() override;

    size_t get_dimension() const override;
    std::unordered_map<std::string, std::any> get_metadata() const override;
    void optimize_index() override;

    // Additional methods
    std::vector<SimilarityResult> search_batch(
        const std::vector<Vector>& query_vectors,
        size_t top_k = 10
    );

    std::unordered_map<std::string, double> get_performance_stats() const;
    void update_config(const VectorStoreConfig& new_config);
    const VectorStoreConfig& get_config() const { return config_; }

private:
    double calculate_similarity(const Vector& v1, const Vector& v2) const;
    void update_cache(size_t query_hash, const std::vector<SimilarityResult>& results) const;
    std::optional<std::vector<SimilarityResult>> get_from_cache(size_t query_hash) const;
    size_t hash_vector(const Vector& vector) const;
    void cleanup_cache() const;
};

/**
 * @brief Factory for creating vector stores
 */
class VectorStoreFactory {
public:
    /**
     * @brief Create in-memory vector store
     * @param config Configuration
     * @return Unique pointer to vector store
     */
    static std::unique_ptr<BaseVectorStore> create_in_memory_store(
        const VectorStoreConfig& config = VectorStoreConfig{}
    );

    /**
     * @brief Create vector store optimized for specific use case
     * @param use_case Use case type ("search", "clustering", "recommendation")
     * @param dimension Vector dimension
     * @return Unique pointer to vector store
     */
    static std::unique_ptr<BaseVectorStore> create_optimized_store(
        const std::string& use_case,
        size_t dimension = 384
    );
};

} // namespace langchain::vectorstores