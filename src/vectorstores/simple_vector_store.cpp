#include "langchain/vectorstores/simple_vector_store.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace langchain::vectorstores {

// SimpleVector operations
double SimpleVector::dot(const SimpleVector& other) const {
    if (data.size() != other.data.size()) {
        throw std::invalid_argument("Vector dimensions must match");
    }

    double result = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        result += data[i] * other.data[i];
    }
    return result;
}

double SimpleVector::magnitude() const {
    double sum_squares = 0.0;
    for (double val : data) {
        sum_squares += val * val;
    }
    return std::sqrt(sum_squares);
}

double SimpleVector::cosine_similarity(const SimpleVector& other) const {
    double dot_product = dot(other);
    double mag1 = magnitude();
    double mag2 = other.magnitude();

    if (mag1 == 0.0 || mag2 == 0.0) {
        return 0.0;
    }

    return dot_product / (mag1 * mag2);
}

void SimpleVectorStoreConfig::validate() const {
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
}

// SimpleVectorStore implementation
SimpleVectorStore::SimpleVectorStore(const SimpleVectorStoreConfig& config)
    : config_(config) {
    config_.validate();
}

size_t SimpleVectorStore::add_vector(const SimpleVector& vector, const std::string& content) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    if (vectors_.size() >= config_.max_vectors) {
        throw std::runtime_error("Vector store is at maximum capacity");
    }

    if (vector.size() != config_.vector_dim) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

    size_t id = next_id_++;

    SimpleVector normalized_vector = vector;
    if (config_.normalize_vectors) {
        double mag = normalized_vector.magnitude();
        if (mag > 0.0) {
            for (double& val : normalized_vector.data) {
                val /= mag;
            }
        }
    }

    vectors_.emplace_back(id, normalized_vector, content);
    return id;
}

std::vector<size_t> SimpleVectorStore::add_vectors(const std::vector<SimpleVectorEntry>& vectors) {
    std::vector<size_t> ids;
    ids.reserve(vectors.size());

    for (const auto& entry : vectors) {
        if (entry.vector.size() != config_.vector_dim) {
            throw std::invalid_argument("Vector dimension mismatch");
        }
        size_t id = add_vector(entry.vector, entry.content);
        ids.push_back(id);
    }

    return ids;
}

std::vector<SimpleSimilarityResult> SimpleVectorStore::search(const SimpleVector& query_vector, size_t top_k) {
    if (query_vector.size() != config_.vector_dim) {
        throw std::invalid_argument("Query vector dimension mismatch");
    }

    SimpleVector normalized_query = query_vector;
    if (config_.normalize_vectors) {
        double mag = normalized_query.magnitude();
        if (mag > 0.0) {
            for (double& val : normalized_query.data) {
                val /= mag;
            }
        }
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<SimpleSimilarityResult> results;
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
              [](const SimpleSimilarityResult& a, const SimpleSimilarityResult& b) {
                  return a.similarity_score > b.similarity_score;
              });

    // Take top_k results
    if (results.size() > top_k) {
        results.resize(top_k);
    }

    return results;
}

std::optional<SimpleVectorEntry> SimpleVectorStore::get_vector(size_t id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    for (const auto& entry : vectors_) {
        if (entry.id == id) {
            return entry;
        }
    }

    return std::nullopt;
}

bool SimpleVectorStore::update_vector(size_t id, const SimpleVector& vector, const std::string& content) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    if (vector.size() != config_.vector_dim) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

    for (auto& entry : vectors_) {
        if (entry.id == id) {
            SimpleVector normalized_vector = vector;
            if (config_.normalize_vectors) {
                double mag = normalized_vector.magnitude();
                if (mag > 0.0) {
                    for (double& val : normalized_vector.data) {
                        val /= mag;
                    }
                }
            }

            entry.vector = normalized_vector;
            if (!content.empty()) {
                entry.content = content;
            }
            return true;
        }
    }

    return false;
}

bool SimpleVectorStore::delete_vector(size_t id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto it = std::find_if(vectors_.begin(), vectors_.end(),
                          [id](const SimpleVectorEntry& entry) { return entry.id == id; });

    if (it != vectors_.end()) {
        vectors_.erase(it);
        return true;
    }

    return false;
}

size_t SimpleVectorStore::size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return vectors_.size();
}

bool SimpleVectorStore::empty() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return vectors_.empty();
}

void SimpleVectorStore::clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    vectors_.clear();
}

size_t SimpleVectorStore::get_dimension() const {
    return config_.vector_dim;
}

void SimpleVectorStore::update_config(const SimpleVectorStoreConfig& new_config) {
    new_config.validate();

    std::unique_lock<std::shared_mutex> lock(mutex_);

    // Validate that existing vectors match new dimension
    if (new_config.vector_dim != config_.vector_dim && !vectors_.empty()) {
        throw std::invalid_argument("Cannot change vector dimension when store is not empty");
    }

    config_ = new_config;
}

std::unordered_map<std::string, std::any> SimpleVectorStore::get_metadata() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::unordered_map<std::string, std::any> metadata;
    metadata["vector_count"] = vectors_.size();
    metadata["dimension"] = config_.vector_dim;
    metadata["max_vectors"] = config_.max_vectors;

    return metadata;
}

double SimpleVectorStore::calculate_similarity(const SimpleVector& v1, const SimpleVector& v2) const {
    return v1.cosine_similarity(v2);
}

} // namespace langchain::vectorstores