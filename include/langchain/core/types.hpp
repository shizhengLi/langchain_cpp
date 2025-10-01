#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <chrono>
#include <functional>
#include <atomic>
#include <any>
#include <thread>
#include <limits>
#include <algorithm>

namespace langchain {

/**
 * @brief Core data structures and type definitions for LangChain++
 */

// Forward declarations
class EmbeddingModel;
class VectorStore;

/**
 * @brief Document model with content and metadata
 */
struct Document {
    std::string content;
    std::unordered_map<std::string, std::string> metadata;
    std::string id;

    Document() {
        // Generate unique ID for default constructor
        id = "doc_" + std::to_string(std::hash<std::string>{}("default_" + std::to_string(reinterpret_cast<uintptr_t>(this))));
    }
    Document(const std::string& content,
             const std::unordered_map<std::string, std::string>& metadata = {})
        : content(content), metadata(metadata) {
            // Generate unique ID
            id = "doc_" + std::to_string(std::hash<std::string>{}(content + std::to_string(reinterpret_cast<uintptr_t>(this))));
        }

    Document(const std::string& content,
             const std::unordered_map<std::string, std::string>& metadata,
             const std::string& id)
        : content(content), metadata(metadata), id(id) {}

    /**
     * @brief Get a text snippet of the document
     * @param max_length Maximum length of the snippet
     * @return Text snippet
     */
    std::string get_text_snippet(size_t max_length = 100) const {
        if (content.length() <= max_length) {
            return content;
        }
        return content.substr(0, max_length) + "...";
    }

    /**
     * @brief Check if document matches metadata filters
     * @param filter_dict Metadata filters
     * @return True if matches all filters
     */
    bool matches_filter(const std::unordered_map<std::string, std::string>& filter_dict) const {
        for (const auto& [key, value] : filter_dict) {
            auto it = metadata.find(key);
            if (it == metadata.end() || it->second != value) {
                return false;
            }
        }
        return true;
    }
};

/**
 * @brief Retrieved document with relevance score
 */
struct RetrievedDocument : public Document {
    double relevance_score = 0.0;
    std::unordered_map<std::string, std::any> additional_info;

    RetrievedDocument() = default;
    RetrievedDocument(const Document& doc, double score)
        : Document(doc), relevance_score(score) {}
};

/**
 * @brief Retrieval result with comprehensive information
 */
struct RetrievalResult {
    std::vector<RetrievedDocument> documents;
    std::string query;
    size_t total_results = 0;
    std::chrono::milliseconds search_time{0};
    std::string retrieval_method;
    std::unordered_map<std::string, std::any> metadata;

    /**
     * @brief Get top k results
     * @param k Number of results to return
     * @return Top k documents
     */
    std::vector<RetrievedDocument> get_top_k(size_t k) const {
        std::vector<RetrievedDocument> sorted_docs = documents;
        std::sort(sorted_docs.begin(), sorted_docs.end(),
                  [](const RetrievedDocument& a, const RetrievedDocument& b) {
                      return a.relevance_score > b.relevance_score;
                  });
        k = std::min(k, sorted_docs.size());
        return std::vector<RetrievedDocument>(sorted_docs.begin(), sorted_docs.begin() + k);
    }

    /**
     * @brief Calculate average relevance score
     * @return Average score
     */
    double get_average_score() const {
        if (documents.empty()) return 0.0;

        double sum = 0.0;
        for (const auto& doc : documents) {
            sum += doc.relevance_score;
        }
        return sum / documents.size();
    }

    /**
     * @brief Filter results by metadata
     * @param key Metadata key
     * @param value Metadata value
     * @return Filtered result
     */
    RetrievalResult filter_by_metadata(const std::string& key, const std::string& value) const {
        RetrievalResult filtered;
        filtered.query = query;
        filtered.total_results = total_results;
        filtered.search_time = search_time;
        filtered.retrieval_method = retrieval_method;
        filtered.metadata = metadata;

        for (const auto& doc : documents) {
            auto it = doc.metadata.find(key);
            if (it != doc.metadata.end() && it->second == value) {
                filtered.documents.push_back(doc);
            }
        }

        return filtered;
    }
};

/**
 * @brief Conversation message for memory management
 */
struct ConversationMessage {
    enum class Type {
        HUMAN,
        AI,
        SYSTEM
    };

    Type type;
    std::string content;
    std::chrono::system_clock::time_point timestamp;
    std::unordered_map<std::string, std::any> additional_data;

    ConversationMessage(Type type, const std::string& content)
        : type(type), content(content), timestamp(std::chrono::system_clock::now()) {}
};

/**
 * @brief LLM generation result
 */
struct LLMResult {
    std::string text;
    size_t prompt_tokens = 0;
    size_t completion_tokens = 0;
    size_t total_tokens = 0;
    std::chrono::milliseconds generation_time{0};
    std::unordered_map<std::string, std::any> metadata;
    bool finished = true;
    std::string finish_reason;

    /**
     * @brief Check if generation was successful
     * @return True if successful
     */
    bool is_successful() const {
        return finished && !text.empty();
    }
};

/**
 * @brief Generation configuration for LLM
 */
struct GenerationConfig {
    double temperature = 0.7;
    size_t max_tokens = 1000;
    double top_p = 0.9;
    size_t top_k = 40;
    bool stream = false;
    std::vector<std::string> stop_sequences;
    double presence_penalty = 0.0;
    double frequency_penalty = 0.0;
    int seed = -1;  // -1 means random seed

    /**
     * @brief Validate configuration
     * @return True if valid
     */
    bool is_valid() const {
        return temperature >= 0.0 && temperature <= 2.0 &&
               max_tokens > 0 &&
               top_p >= 0.0 && top_p <= 1.0 &&
               top_k > 0 &&
               presence_penalty >= -2.0 && presence_penalty <= 2.0 &&
               frequency_penalty >= -2.0 && frequency_penalty <= 2.0;
    }
};

/**
 * @brief Prompt template for LLM
 */
struct Prompt {
    std::string template_str;
    std::unordered_map<std::string, std::string> variables;
    std::unordered_map<std::string, std::any> metadata;

    /**
     * @brief Format prompt with variables
     * @return Formatted prompt string
     */
    std::string format() const {
        std::string result = template_str;
        for (const auto& [key, value] : variables) {
            std::string placeholder = "{" + key + "}";
            size_t pos = 0;
            while ((pos = result.find(placeholder, pos)) != std::string::npos) {
                result.replace(pos, placeholder.length(), value);
                pos += value.length();
            }
        }
        return result;
    }

    /**
     * @brief Set a variable value
     * @param key Variable name
     * @param value Variable value
     */
    void set_variable(const std::string& key, const std::string& value) {
        variables[key] = value;
    }
};

/**
 * @brief Vector search result
 */
struct VectorSearchResult {
    std::string id;
    double score = 0.0;
    std::unordered_map<std::string, std::any> metadata;

    VectorSearchResult() = default;
    VectorSearchResult(const std::string& id, double score)
        : id(id), score(score) {}
};

/**
 * @brief Embedding result
 */
struct EmbeddingResult {
    std::vector<float> embedding;
    size_t tokens_used = 0;
    std::chrono::milliseconds embedding_time{0};

    /**
     * @brief Calculate cosine similarity with another embedding
     * @param other Other embedding
     * @return Similarity score
     */
    double cosine_similarity(const std::vector<float>& other) const {
        if (embedding.size() != other.size() || embedding.empty()) {
            return 0.0;
        }

        double dot_product = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;

        for (size_t i = 0; i < embedding.size(); ++i) {
            dot_product += embedding[i] * other[i];
            norm_a += embedding[i] * embedding[i];
            norm_b += other[i] * other[i];
        }

        if (norm_a == 0.0 || norm_b == 0.0) {
            return 0.0;
        }

        return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
};

/**
 * @brief Task for asynchronous processing
 */
template<typename T>
struct Task {
    std::function<T()> function;
    std::string description;
    std::chrono::system_clock::time_point created_at;

    Task(std::function<T()> func, const std::string& desc = "")
        : function(std::move(func)), description(desc),
          created_at(std::chrono::system_clock::now()) {}
};

/**
 * @brief Performance metrics
 */
struct PerformanceMetrics {
    std::atomic<size_t> total_requests{0};
    std::atomic<size_t> successful_requests{0};
    std::atomic<size_t> failed_requests{0};
    std::atomic<double> total_latency_ms{0.0};
    std::atomic<double> min_latency_ms{std::numeric_limits<double>::max()};
    std::atomic<double> max_latency_ms{0.0};

    /**
     * @brief Record a request completion
     * @param latency_ms Request latency in milliseconds
     * @param success Whether the request was successful
     */
    void record_request(double latency_ms, bool success) {
        total_requests++;
        if (success) {
            successful_requests++;
        } else {
            failed_requests++;
        }

        total_latency_ms += latency_ms;

        // Update min/max latency (note: this is not perfectly thread-safe, but good enough for metrics)
        double current_min = min_latency_ms.load();
        while (latency_ms < current_min && !min_latency_ms.compare_exchange_weak(current_min, latency_ms)) {}

        double current_max = max_latency_ms.load();
        while (latency_ms > current_max && !max_latency_ms.compare_exchange_weak(current_max, latency_ms)) {}
    }

    /**
     * @brief Get average latency
     * @return Average latency in milliseconds
     */
    double get_average_latency() const {
        size_t total = total_requests.load();
        return total > 0 ? total_latency_ms.load() / total : 0.0;
    }

    /**
     * @brief Get success rate
     * @return Success rate as percentage
     */
    double get_success_rate() const {
        size_t total = total_requests.load();
        return total > 0 ? (static_cast<double>(successful_requests.load()) / total) * 100.0 : 0.0;
    }
};

} // namespace langchain