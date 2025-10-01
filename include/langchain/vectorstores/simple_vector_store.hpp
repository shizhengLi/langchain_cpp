#pragma once

#include "../core/base.hpp"
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
 * @brief Simple vector representation for embeddings
 */
struct SimpleVector {
    std::vector<double> data;

    SimpleVector() = default;
    explicit SimpleVector(size_t size) : data(size, 0.0) {}
    SimpleVector(std::vector<double> d) : data(std::move(d)) {}

    size_t size() const { return data.size(); }
    double& operator[](size_t idx) { return data[idx]; }
    const double& operator[](size_t idx) const { return data[idx]; }

    // Basic vector operations
    double dot(const SimpleVector& other) const;
    double magnitude() const;
    double cosine_similarity(const SimpleVector& other) const;
};

/**
 * @brief Simple vector entry with metadata
 */
struct SimpleVectorEntry {
    size_t id;
    SimpleVector vector;
    std::string content;
    std::unordered_map<std::string, std::string> metadata;

    SimpleVectorEntry(size_t i, SimpleVector v, std::string c = "")
        : id(i), vector(std::move(v)), content(std::move(c)) {}
};

/**
 * @brief Simple similarity search result
 */
struct SimpleSimilarityResult {
    size_t entry_id;
    double similarity_score;
    std::string content;
    std::unordered_map<std::string, std::string> metadata;

    SimpleSimilarityResult() = default;
    SimpleSimilarityResult(size_t id, double score, std::string content = "",
                         std::unordered_map<std::string, std::string> metadata = {})
        : entry_id(id), similarity_score(score), content(std::move(content)),
          metadata(std::move(metadata)) {}
};

/**
 * @brief Simple vector store configuration
 */
struct SimpleVectorStoreConfig {
    size_t max_vectors = 100000;
    size_t vector_dim = 384;
    bool normalize_vectors = true;
    size_t default_top_k = 10;
    double similarity_threshold = 0.0;

    void validate() const;
};

/**
 * @brief Simple in-memory vector store implementation
 */
class SimpleVectorStore {
private:
    SimpleVectorStoreConfig config_;
    std::vector<SimpleVectorEntry> vectors_;
    std::atomic<size_t> next_id_{1};
    mutable std::shared_mutex mutex_;

public:
    explicit SimpleVectorStore(const SimpleVectorStoreConfig& config = SimpleVectorStoreConfig{});
    ~SimpleVectorStore() = default;

    // Core operations
    size_t add_vector(const SimpleVector& vector, const std::string& content = "");
    std::vector<size_t> add_vectors(const std::vector<SimpleVectorEntry>& vectors);
    std::vector<SimpleSimilarityResult> search(const SimpleVector& query_vector, size_t top_k = 10);
    std::optional<SimpleVectorEntry> get_vector(size_t id) const;
    bool update_vector(size_t id, const SimpleVector& vector, const std::string& content = "");
    bool delete_vector(size_t id);
    size_t size() const;
    bool empty() const;
    void clear();
    size_t get_dimension() const;

    // Configuration and metadata
    void update_config(const SimpleVectorStoreConfig& new_config);
    const SimpleVectorStoreConfig& get_config() const { return config_; }
    std::unordered_map<std::string, std::any> get_metadata() const;

private:
    double calculate_similarity(const SimpleVector& v1, const SimpleVector& v2) const;
};

} // namespace langchain::vectorstores