#pragma once

#include "types.hpp"
#include <string>
#include <vector>
#include <optional>
#include <unordered_map>
#include <cmath>

namespace langchain {

/**
 * @brief Configuration classes for different components
 */

/**
 * @brief Configuration for retrieval operations
 */
struct RetrievalConfig {
    // Core parameters
    size_t top_k = 5;
    std::optional<double> score_threshold;
    std::string search_type = "similarity";  // similarity, mmr, hybrid, tfidf, bm25

    // MMR parameters
    double mmr_lambda = 0.5;  // 0 = diversity, 1 = relevance
    size_t fetch_k = 20;      // Number of candidates for MMR

    // Performance parameters
    bool enable_caching = true;
    std::optional<double> cache_ttl_seconds = std::nullopt;
    size_t batch_size = 32;

    // Filtering parameters
    std::unordered_map<std::string, std::string> filter_dict;

    // Document retriever specific
    bool enable_stop_words = true;
    std::vector<std::string> custom_stop_words;

    // Vector retriever specific
    std::string similarity_metric = "cosine";  // cosine, euclidean, manhattan

    /**
     * @brief Validate configuration
     * @return True if configuration is valid
     */
    bool is_valid() const {
        if (top_k == 0 || top_k > 1000) return false;
        if (score_threshold && (*score_threshold < 0.0 || *score_threshold > 1.0)) return false;
        if (mmr_lambda < 0.0 || mmr_lambda > 1.0) return false;
        if (fetch_k < top_k || fetch_k > 10000) return false;
        if (cache_ttl_seconds && *cache_ttl_seconds <= 0.0) return false;
        if (batch_size == 0 || batch_size > 1000) return false;

        // Validate search type
        const std::vector<std::string> valid_search_types = {
            "similarity", "mmr", "hybrid", "tfidf", "bm25"
        };
        if (std::find(valid_search_types.begin(), valid_search_types.end(), search_type) == valid_search_types.end()) {
            return false;
        }

        // Validate similarity metric
        const std::vector<std::string> valid_metrics = {
            "cosine", "euclidean", "manhattan"
        };
        if (std::find(valid_metrics.begin(), valid_metrics.end(), similarity_metric) == valid_metrics.end()) {
            return false;
        }

        return true;
    }

    /**
     * @brief Get default configuration for different search types
     * @param type Search type
     * @return Default configuration
     */
    static RetrievalConfig get_default_for_search_type(const std::string& type) {
        RetrievalConfig config;
        config.search_type = type;

        if (type == "bm25") {
            config.top_k = 10;
            config.enable_stop_words = true;
        } else if (type == "mmr") {
            config.mmr_lambda = 0.7;
            config.fetch_k = 50;
        } else if (type == "similarity") {
            config.score_threshold = 0.0;
        }

        return config;
    }
};

/**
 * @brief Configuration for vector stores
 */
struct VectorStoreConfig {
    size_t dimension = 384;
    std::string metric = "cosine";
    size_t max_elements = 1000000;
    std::string storage_backend = "memory";  // memory, mmap, redis
    std::string index_type = "hnsw";         // hnsw, ivf, flat

    // HNSW specific parameters
    size_t m = 16;            // Number of connections
    size_t ef_construction = 200;  // Build time parameter
    size_t ef_search = 50;    // Search time parameter

    // IVF specific parameters
    size_t nlist = 100;       // Number of clusters
    size_t nprobe = 10;       // Number of clusters to search

    // Memory-mapped file specific
    std::optional<std::string> mmap_file_path;
    bool read_only = false;

    /**
     * @brief Validate configuration
     * @return True if configuration is valid
     */
    bool is_valid() const {
        if (dimension == 0 || dimension > 10000) return false;
        if (max_elements == 0) return false;

        // Validate metric
        const std::vector<std::string> valid_metrics = {
            "cosine", "euclidean", "manhattan", "dotproduct"
        };
        if (std::find(valid_metrics.begin(), valid_metrics.end(), metric) == valid_metrics.end()) {
            return false;
        }

        // Validate storage backend
        const std::vector<std::string> valid_backends = {
            "memory", "mmap", "redis"
        };
        if (std::find(valid_backends.begin(), valid_backends.end(), storage_backend) == valid_backends.end()) {
            return false;
        }

        // Validate index type
        const std::vector<std::string> valid_indices = {
            "hnsw", "ivf", "flat"
        };
        if (std::find(valid_indices.begin(), valid_indices.end(), index_type) == valid_indices.end()) {
            return false;
        }

        // Validate HNSW parameters
        if (m == 0 || m > 200) return false;
        if (ef_construction == 0 || ef_construction > 1000) return false;
        if (ef_search == 0 || ef_search > 1000) return false;

        // Validate IVF parameters
        if (nlist == 0 || nlist > 100000) return false;
        if (nprobe == 0 || nprobe > nlist) return false;

        return true;
    }

    /**
     * @brief Get HNSW configuration
     * @return HNSW-specific configuration
     */
    static VectorStoreConfig get_hnsw_config(size_t dimension, size_t max_elements = 1000000) {
        VectorStoreConfig config;
        config.dimension = dimension;
        config.max_elements = max_elements;
        config.index_type = "hnsw";
        config.metric = "cosine";
        config.m = 16;
        config.ef_construction = 200;
        config.ef_search = 50;
        return config;
    }

    /**
     * @brief Get IVF configuration
     * @return IVF-specific configuration
     */
    static VectorStoreConfig get_ivf_config(size_t dimension, size_t max_elements = 1000000) {
        VectorStoreConfig config;
        config.dimension = dimension;
        config.max_elements = max_elements;
        config.index_type = "ivf";
        config.metric = "cosine";
        config.nlist = std::min(static_cast<size_t>(std::round(std::sqrt(max_elements))), static_cast<size_t>(10000));
        config.nprobe = std::min(config.nlist / 10, static_cast<size_t>(100));
        return config;
    }
};

/**
 * @brief Configuration for memory management
 */
struct MemoryConfig {
    size_t max_messages = 100;
    size_t max_tokens = 4000;
    std::string strategy = "buffer";  // buffer, summary, sliding_window
    std::string storage_backend = "memory";  // memory, file, redis

    // Buffer memory specific
    std::optional<std::string> memory_key;  // No default key

    // Summary memory specific
    std::string summary_prompt = "Summarize the conversation history concisely:";
    size_t max_summary_tokens = 500;

    // Sliding window specific
    double window_ratio = 0.8;  // Keep 80% of recent content

    // Storage specific
    std::optional<std::string> storage_path;
    bool persistent = false;

    /**
     * @brief Validate configuration
     * @return True if configuration is valid
     */
    bool is_valid() const {
        if (max_messages == 0 || max_messages > 10000) return false;
        if (max_tokens == 0 || max_tokens > 100000) return false;

        // Validate strategy
        const std::vector<std::string> valid_strategies = {
            "buffer", "summary", "sliding_window"
        };
        if (std::find(valid_strategies.begin(), valid_strategies.end(), strategy) == valid_strategies.end()) {
            return false;
        }

        // Validate storage backend
        const std::vector<std::string> valid_backends = {
            "memory", "file", "redis"
        };
        if (std::find(valid_backends.begin(), valid_backends.end(), storage_backend) == valid_backends.end()) {
            return false;
        }

        if (window_ratio <= 0.0 || window_ratio > 1.0) return false;
        if (max_summary_tokens == 0 || max_summary_tokens > max_tokens) return false;

        return true;
    }

    /**
     * @brief Get buffer memory configuration
     * @return Buffer memory configuration
     */
    static MemoryConfig get_buffer_config(size_t max_messages = 100, size_t max_tokens = 4000) {
        MemoryConfig config;
        config.strategy = "buffer";
        config.max_messages = max_messages;
        config.max_tokens = max_tokens;
        return config;
    }

    /**
     * @brief Get summary memory configuration
     * @return Summary memory configuration
     */
    static MemoryConfig get_summary_config(size_t max_tokens = 4000) {
        MemoryConfig config;
        config.strategy = "summary";
        config.max_tokens = max_tokens;
        config.max_summary_tokens = max_tokens / 4;  // 25% for summary
        return config;
    }
};

/**
 * @brief Configuration for LLM operations
 */
struct LLMConfig {
    std::string model_name = "gpt-3.5-turbo";
    std::string api_base = "https://api.openai.com/v1";
    std::string api_key;
    std::string organization;

    // Connection parameters
    size_t max_connections = 10;
    std::chrono::milliseconds connection_timeout{30000};
    std::chrono::milliseconds read_timeout{60000};
    size_t max_retries = 3;
    std::chrono::milliseconds retry_delay{1000};

    // Rate limiting
    std::optional<size_t> requests_per_minute;
    std::optional<size_t> tokens_per_minute;

    // Caching
    bool enable_cache = true;
    size_t cache_size = 1000;
    std::chrono::hours cache_ttl{1};

    /**
     * @brief Validate configuration
     * @return True if configuration is valid
     */
    bool is_valid() const {
        if (model_name.empty()) return false;
        if (api_base.empty()) return false;
        if (max_connections == 0 || max_connections > 1000) return false;
        if (connection_timeout.count() == 0 || connection_timeout.count() > 300000) return false;
        if (read_timeout.count() == 0 || read_timeout.count() > 600000) return false;
        if (max_retries > 10) return false;
        if (retry_delay.count() > 60000) return false;
        if (cache_size == 0 || cache_size > 100000) return false;

        return true;
    }

    /**
     * @brief Get OpenAI configuration
     * @param api_key OpenAI API key
     * @param model Model name
     * @return OpenAI configuration
     */
    static LLMConfig get_openai_config(const std::string& api_key, const std::string& model = "gpt-3.5-turbo") {
        LLMConfig config;
        config.api_base = "https://api.openai.com/v1";
        config.model_name = model;
        config.api_key = api_key;
        config.max_connections = 10;
        config.requests_per_minute = 3000;  // OpenAI rate limit
        return config;
    }
};

/**
 * @brief Configuration for embedding operations
 */
struct EmbeddingConfig {
    std::string model_name = "text-embedding-ada-002";
    std::string api_base = "https://api.openai.com/v1";
    std::string api_key;
    size_t embedding_dimension = 1536;
    size_t batch_size = 100;
    std::chrono::milliseconds timeout{30000};

    // Similar to LLM config
    size_t max_connections = 5;
    size_t max_retries = 3;
    bool enable_cache = true;
    size_t cache_size = 10000;

    /**
     * @brief Validate configuration
     * @return True if configuration is valid
     */
    bool is_valid() const {
        if (model_name.empty()) return false;
        if (api_base.empty()) return false;
        if (embedding_dimension == 0 || embedding_dimension > 10000) return false;
        if (batch_size == 0 || batch_size > 1000) return false;
        if (timeout.count() == 0 || timeout.count() > 300000) return false;
        if (max_connections == 0 || max_connections > 100) return false;
        if (max_retries > 10) return false;
        if (cache_size == 0 || cache_size > 100000) return false;

        return true;
    }

    /**
     * @brief Get OpenAI embedding configuration
     * @param api_key OpenAI API key
     * @param model Model name
     * @return OpenAI embedding configuration
     */
    static EmbeddingConfig get_openai_config(const std::string& api_key, const std::string& model = "text-embedding-ada-002") {
        EmbeddingConfig config;
        config.api_base = "https://api.openai.com/v1";
        config.model_name = model;
        config.api_key = api_key;
        config.embedding_dimension = model == "text-embedding-ada-002" ? 1536 : 1536;
        return config;
    }
};

/**
 * @brief Global configuration for the entire library
 */
struct LangChainConfig {
    // Logging
    std::string log_level = "info";  // debug, info, warn, error
    std::optional<std::string> log_file;
    bool enable_console_logging = true;

    // Performance
    size_t default_thread_pool_size = 0;  // 0 = auto-detect
    bool enable_performance_metrics = true;
    std::chrono::seconds metrics_collection_interval{60};

    // Memory
    bool enable_memory_pooling = true;
    size_t default_memory_pool_size = 1024 * 1024;  // 1MB

    // SIMD
    bool enable_simd = true;
    bool prefer_avx512 = false;

    // Concurrency
    bool enable_async_operations = true;
    size_t max_concurrent_operations = 100;

    /**
     * @brief Validate configuration
     * @return True if configuration is valid
     */
    bool is_valid() const {
        // Validate log level
        const std::vector<std::string> valid_log_levels = {
            "debug", "info", "warn", "error"
        };
        if (std::find(valid_log_levels.begin(), valid_log_levels.end(), log_level) == valid_log_levels.end()) {
            return false;
        }

        if (default_thread_pool_size > 1000) return false;
        if (metrics_collection_interval.count() == 0 || metrics_collection_interval.count() > 3600) return false;
        if (default_memory_pool_size == 0 || default_memory_pool_size > 1024 * 1024 * 1024) return false;
        if (max_concurrent_operations == 0 || max_concurrent_operations > 10000) return false;

        return true;
    }

    /**
     * @brief Get default configuration
     * @return Default configuration
     */
    static LangChainConfig get_default() {
        LangChainConfig config;

        // Auto-detect thread pool size
        config.default_thread_pool_size = std::thread::hardware_concurrency();
        if (config.default_thread_pool_size == 0) {
            config.default_thread_pool_size = 4;  // Fallback
        }

        return config;
    }
};

} // namespace langchain