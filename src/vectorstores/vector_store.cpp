#include "langchain/vectorstores/vector_store.hpp"
#include "langchain/utils/logging.hpp"
#include "langchain/utils/simd_ops.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <functional>

namespace langchain::vectorstores {

// Vector operations implementation
double Vector::dot(const Vector& other) const {
    if (data.size() != other.data.size()) {
        throw std::invalid_argument("Vector dimensions must match");
    }

    double result = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        result += data[i] * other.data[i];
    }
    return result;
}

double Vector::magnitude() const {
    double sum_squares = 0.0;
    for (double val : data) {
        sum_squares += val * val;
    }
    return std::sqrt(sum_squares);
}

double Vector::cosine_similarity(const Vector& other) const {
    double dot_product = dot(other);
    double mag1 = magnitude();
    double mag2 = other.magnitude();

    if (mag1 == 0.0 || mag2 == 0.0) {
        return 0.0;
    }

    return dot_product / (mag1 * mag2);
}

double Vector::euclidean_distance(const Vector& other) const {
    if (data.size() != other.data.size()) {
        throw std::invalid_argument("Vector dimensions must match");
    }

    double sum_squares = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        double diff = data[i] - other.data[i];
        sum_squares += diff * diff;
    }
    return std::sqrt(sum_squares);
}

Vector Vector::add_simd(const Vector& other) const {
    if (data.size() != other.data.size()) {
        throw std::invalid_argument("Vector dimensions must match");
    }

    Vector result(data.size());
    // Use simple implementation for now - SIMD optimization can be added later
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Vector Vector::subtract_simd(const Vector& other) const {
    if (data.size() != other.data.size()) {
        throw std::invalid_argument("Vector dimensions must match");
    }

    Vector result(data.size());
    // Use simple implementation for now - SIMD optimization can be added later
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Vector Vector::multiply_simd(double scalar) const {
    Vector result(data.size());
    // Use simple implementation for now - SIMD optimization can be added later
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

double Vector::dot_simd(const Vector& other) const {
    if (data.size() != other.data.size()) {
        throw std::invalid_argument("Vector dimensions must match");
    }

    // Use VectorOps for SIMD optimization
    std::vector<float> float_data(data.begin(), data.end());
    std::vector<float> float_other(other.data.begin(), other.data.end());

    return static_cast<double>(utils::VectorOps::dot_product(
        float_data.data(), float_other.data(), data.size()));
}

void VectorStoreConfig::validate() const {
    if (max_vectors == 0) {
        throw std::invalid_argument("max_vectors must be greater than 0");
    }

    if (vector_dim == 0) {
        throw std::invalid_argument("vector_dim must be greater than 0");
    }

    if (vector_dim > 10000) {
        throw std::invalid_argument("vector_dim is too large (max 10000)");
    }

    if (default_top_k == 0) {
        throw std::invalid_argument("default_top_k must be greater than 0");
    }

    if (similarity_threshold < 0.0 || similarity_threshold > 1.0) {
        throw std::invalid_argument("similarity_threshold must be between 0.0 and 1.0");
    }

    if (cache_size == 0) {
        throw std::invalid_argument("cache_size must be greater than 0");
    }
}

// InMemoryVectorStore implementation
InMemoryVectorStore::InMemoryVectorStore(const VectorStoreConfig& config)
    : config_(config) {
    config_.validate();
    Logger::get_instance().info("Created InMemoryVectorStore with dimension " +
                               std::to_string(config_.vector_dim));
}

size_t InMemoryVectorStore::add_vector(
    const Vector& vector,
    const std::string& content,
    const std::unordered_map<std::string, std::string>& metadata
) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    if (vectors_.size() >= config_.max_vectors) {
        throw std::runtime_error("Vector store is at maximum capacity");
    }

    if (vector.size() != config_.vector_dim) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

    size_t id = next_id_++;

    Vector normalized_vector = vector;
    if (config_.normalize_vectors) {
        double mag = normalized_vector.magnitude();
        if (mag > 0.0) {
            normalized_vector = normalized_vector.multiply_simd(1.0 / mag);
        }
    }

    vectors_.emplace_back(id, normalized_vector, content, metadata);

    Logger::get_instance().debug("Added vector with ID " + std::to_string(id));
    return id;
}

std::vector<size_t> InMemoryVectorStore::add_vectors(
    const std::vector<VectorEntry>& vectors
) {
    std::vector<size_t> ids;
    ids.reserve(vectors.size());

    for (const auto& entry : vectors) {
        if (entry.vector.size() != config_.vector_dim) {
            throw std::invalid_argument("Vector dimension mismatch");
        }
        size_t id = add_vector(entry.vector, entry.content, entry.metadata);
        ids.push_back(id);
    }

    Logger::get_instance().info("Added " + std::to_string(vectors.size()) + " vectors");
    return ids;
}

std::vector<SimilarityResult> InMemoryVectorStore::search(
    const Vector& query_vector,
    size_t top_k
) {
    total_searches_++;

    // Check cache first
    size_t query_hash = hash_vector(query_vector);
    if (auto cached_results = get_from_cache(query_hash)) {
        cache_hits_++;
        Logger::get_instance().debug("Cache hit for vector search");
        return *cached_results;
    }

    if (query_vector.size() != config_.vector_dim) {
        throw std::invalid_argument("Query vector dimension mismatch");
    }

    Vector normalized_query = query_vector;
    if (config_.normalize_vectors) {
        double mag = normalized_query.magnitude();
        if (mag > 0.0) {
            normalized_query = normalized_query.multiply_simd(1.0 / mag);
        }
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<SimilarityResult> results;
    results.reserve(vectors_.size());

    // Calculate similarities
    for (const auto& entry : vectors_) {
        double similarity = calculate_similarity(normalized_query, entry.vector);
        if (similarity >= config_.similarity_threshold) {
            results.emplace_back(entry.id, similarity, entry.content, entry.metadata);
        }
    }

    // Sort by similarity (descending)
    std::sort(results.begin(), results.end(),
              [](const SimilarityResult& a, const SimilarityResult& b) {
                  return a.similarity_score > b.similarity_score;
              });

    // Take top_k results
    if (results.size() > top_k) {
        results.resize(top_k);
    }

    // Update cache
    update_cache(query_hash, results);

    Logger::get_instance().debug("Search returned " + std::to_string(results.size()) +
                                " results for top_k " + std::to_string(top_k));
    return results;
}

std::vector<SimilarityResult> InMemoryVectorStore::search_with_threshold(
    const Vector& query_vector,
    double similarity_threshold,
    size_t top_k
) {
    auto results = search(query_vector, top_k);

    // Filter by threshold
    results.erase(
        std::remove_if(results.begin(), results.end(),
                      [similarity_threshold](const SimilarityResult& r) {
                          return r.similarity_score < similarity_threshold;
                      }),
        results.end()
    );

    return results;
}

std::optional<VectorEntry> InMemoryVectorStore::get_vector(size_t id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    for (const auto& entry : vectors_) {
        if (entry.id == id) {
            return entry;
        }
    }

    return std::nullopt;
}

bool InMemoryVectorStore::update_vector(
    size_t id,
    const Vector& vector,
    const std::string& content,
    const std::unordered_map<std::string, std::string>& metadata
) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    if (vector.size() != config_.vector_dim) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

    for (auto& entry : vectors_) {
        if (entry.id == id) {
            Vector normalized_vector = vector;
            if (config_.normalize_vectors) {
                double mag = normalized_vector.magnitude();
                if (mag > 0.0) {
                    normalized_vector = normalized_vector.multiply_simd(1.0 / mag);
                }
            }

            entry.vector = normalized_vector;
            if (!content.empty()) {
                entry.content = content;
            }
            if (!metadata.empty()) {
                entry.metadata = metadata;
            }

            // Clear cache since vectors changed
            std::lock_guard<std::mutex> cache_lock(cache_mutex_);
            search_cache_.clear();
            cache_access_order_.clear();

            Logger::get_instance().debug("Updated vector with ID " + std::to_string(id));
            return true;
        }
    }

    return false;
}

bool InMemoryVectorStore::delete_vector(size_t id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto it = std::find_if(vectors_.begin(), vectors_.end(),
                          [id](const VectorEntry& entry) { return entry.id == id; });

    if (it != vectors_.end()) {
        vectors_.erase(it);

        // Clear cache
        std::lock_guard<std::mutex> cache_lock(cache_mutex_);
        search_cache_.clear();
        cache_access_order_.clear();

        Logger::get_instance().debug("Deleted vector with ID " + std::to_string(id));
        return true;
    }

    return false;
}

size_t InMemoryVectorStore::size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return vectors_.size();
}

bool InMemoryVectorStore::empty() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return vectors_.empty();
}

void InMemoryVectorStore::clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    vectors_.clear();

    std::lock_guard<std::mutex> cache_lock(cache_mutex_);
    search_cache_.clear();
    cache_access_order_.clear();

    Logger::get_instance().info("Cleared all vectors from store");
}

size_t InMemoryVectorStore::get_dimension() const {
    return config_.vector_dim;
}

std::unordered_map<std::string, std::any> InMemoryVectorStore::get_metadata() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::unordered_map<std::string, std::any> metadata;
    metadata["vector_count"] = vectors_.size();
    metadata["dimension"] = config_.vector_dim;
    metadata["max_vectors"] = config_.max_vectors;
    metadata["total_searches"] = total_searches_.load();
    metadata["cache_hits"] = cache_hits_.load();
    metadata["cache_hit_rate"] = total_searches_.load() > 0 ?
        static_cast<double>(cache_hits_.load()) / total_searches_.load() : 0.0;

    return metadata;
}

void InMemoryVectorStore::optimize_index() {
    Logger::get_instance().info("Optimizing vector store index");

    // For in-memory store, we can organize vectors for better cache locality
    std::unique_lock<std::shared_mutex> lock(mutex_);

    // Sort by ID for better access patterns
    std::sort(vectors_.begin(), vectors_.end(),
              [](const VectorEntry& a, const VectorEntry& b) {
                  return a.id < b.id;
              });

    // Clean up cache
    cleanup_cache();

    Logger::get_instance().info("Vector store optimization completed");
}

std::vector<SimilarityResult> InMemoryVectorStore::search_batch(
    const std::vector<Vector>& query_vectors,
    size_t top_k
) {
    std::vector<SimilarityResult> all_results;

    for (const auto& query_vector : query_vectors) {
        auto results = search(query_vector, top_k);
        all_results.insert(all_results.end(), results.begin(), results.end());
    }

    // Sort all results by similarity
    std::sort(all_results.begin(), all_results.end(),
              [](const SimilarityResult& a, const SimilarityResult& b) {
                  return a.similarity_score > b.similarity_score;
              });

    return all_results;
}

std::unordered_map<std::string, double> InMemoryVectorStore::get_performance_stats() const {
    std::unordered_map<std::string, double> stats;

    size_t total_searches = total_searches_.load();
    size_t cache_hits = cache_hits_.load();

    stats["total_searches"] = static_cast<double>(total_searches);
    stats["cache_hits"] = static_cast<double>(cache_hits);
    stats["cache_hit_rate"] = total_searches > 0 ?
        static_cast<double>(cache_hits) / total_searches : 0.0;
    stats["vector_count"] = static_cast<double>(size());
    stats["avg_search_time_ms"] = 0.0; // Could be measured with timing

    return stats;
}

void InMemoryVectorStore::update_config(const VectorStoreConfig& new_config) {
    new_config.validate();

    std::unique_lock<std::shared_mutex> lock(mutex_);

    // Validate that existing vectors match new dimension
    if (new_config.vector_dim != config_.vector_dim && !vectors_.empty()) {
        throw std::invalid_argument("Cannot change vector dimension when store is not empty");
    }

    config_ = new_config;
    Logger::get_instance().info("Updated vector store configuration");
}

double InMemoryVectorStore::calculate_similarity(const Vector& v1, const Vector& v2) const {
    switch (config_.distance_metric) {
        case VectorStoreConfig::DistanceMetric::COSINE:
            return config_.use_simd ? v1.cosine_similarity(v2) : v1.cosine_similarity(v2);
        case VectorStoreConfig::DistanceMetric::EUCLIDEAN: {
            double dist = v1.euclidean_distance(v2);
            return 1.0 / (1.0 + dist); // Convert distance to similarity
        }
        case VectorStoreConfig::DistanceMetric::DOT_PRODUCT:
            return config_.use_simd ? v1.dot_simd(v2) : v1.dot(v2);
        default:
            return v1.cosine_similarity(v2);
    }
}

void InMemoryVectorStore::update_cache(size_t query_hash, const std::vector<SimilarityResult>& results) const {
    if (!config_.cache_enabled) return;

    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Remove oldest entry if cache is full
    if (search_cache_.size() >= config_.cache_size) {
        cleanup_cache();
    }

    search_cache_[query_hash] = results;
    cache_access_order_.push_back(query_hash);
}

std::optional<std::vector<SimilarityResult>> InMemoryVectorStore::get_from_cache(size_t query_hash) const {
    if (!config_.cache_enabled) return std::nullopt;

    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = search_cache_.find(query_hash);
    if (it != search_cache_.end()) {
        // Update access order
        auto access_it = std::find(cache_access_order_.begin(),
                                  cache_access_order_.end(), query_hash);
        if (access_it != cache_access_order_.end()) {
            cache_access_order_.erase(access_it);
            cache_access_order_.push_back(query_hash);
        }
        return it->second;
    }

    return std::nullopt;
}

size_t InMemoryVectorStore::hash_vector(const Vector& vector) const {
    size_t hash = 0;
    for (double val : vector.data) {
        // Simple hash combining
        hash ^= std::hash<double>{}(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
}

void InMemoryVectorStore::cleanup_cache() const {
    if (search_cache_.size() <= config_.cache_size) return;

    // Remove oldest entries
    while (search_cache_.size() > config_.cache_size && !cache_access_order_.empty()) {
        size_t oldest_hash = cache_access_order_.front();
        search_cache_.erase(oldest_hash);
        cache_access_order_.erase(cache_access_order_.begin());
    }
}

// VectorStoreFactory implementation
std::unique_ptr<BaseVectorStore> VectorStoreFactory::create_in_memory_store(
    const VectorStoreConfig& config
) {
    return std::make_unique<InMemoryVectorStore>(config);
}

std::unique_ptr<BaseVectorStore> VectorStoreFactory::create_optimized_store(
    const std::string& use_case,
    size_t dimension
) {
    VectorStoreConfig config;
    config.vector_dim = dimension;

    if (use_case == "search") {
        config.default_top_k = 20;
        config.cache_size = 2000;
        config.distance_metric = VectorStoreConfig::DistanceMetric::COSINE;
    } else if (use_case == "clustering") {
        config.default_top_k = 50;
        config.cache_size = 500;
        config.distance_metric = VectorStoreConfig::DistanceMetric::EUCLIDEAN;
    } else if (use_case == "recommendation") {
        config.default_top_k = 10;
        config.cache_size = 1000;
        config.distance_metric = VectorStoreConfig::DistanceMetric::DOT_PRODUCT;
    } else {
        // Default configuration
        config.default_top_k = 10;
        config.cache_size = 1000;
        config.distance_metric = VectorStoreConfig::DistanceMetric::COSINE;
    }

    return std::make_unique<InMemoryVectorStore>(config);
}

} // namespace langchain::vectorstores