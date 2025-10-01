#include <catch2/catch_all.hpp>
#include "langchain/core/config.hpp"

using namespace langchain;

TEST_CASE("RetrievalConfig - Default Values", "[core][config][retrieval]") {
    RetrievalConfig config;

    REQUIRE(config.top_k == 5);
    REQUIRE_FALSE(config.score_threshold.has_value());
    REQUIRE(config.search_type == "similarity");
    REQUIRE(config.mmr_lambda == 0.5);
    REQUIRE(config.fetch_k == 20);
    REQUIRE(config.enable_caching == true);
    REQUIRE_FALSE(config.cache_ttl_seconds.has_value());
    REQUIRE(config.batch_size == 32);
    REQUIRE(config.filter_dict.empty());
    REQUIRE(config.enable_stop_words == true);
    REQUIRE(config.custom_stop_words.empty());
    REQUIRE(config.similarity_metric == "cosine");
}

TEST_CASE("RetrievalConfig - Validation", "[core][config][retrieval]") {
    RetrievalConfig config;

    SECTION("Valid configuration") {
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid top_k") {
        config.top_k = 0;
        REQUIRE_FALSE(config.is_valid());

        config.top_k = 1001;
        REQUIRE_FALSE(config.is_valid());
    }

    SECTION("Invalid score_threshold") {
        config.score_threshold = -0.1;
        REQUIRE_FALSE(config.is_valid());

        config.score_threshold = 1.1;
        REQUIRE_FALSE(config.is_valid());

        config.score_threshold = 0.5;
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid search_type") {
        config.search_type = "invalid";
        REQUIRE_FALSE(config.is_valid());
    }

    SECTION("Valid search_types") {
        std::vector<std::string> valid_types = {
            "similarity", "mmr", "hybrid", "tfidf", "bm25"
        };

        for (const auto& type : valid_types) {
            config.search_type = type;
            REQUIRE(config.is_valid());
        }
    }

    SECTION("Invalid mmr_lambda") {
        config.mmr_lambda = -0.1;
        REQUIRE_FALSE(config.is_valid());

        config.mmr_lambda = 1.1;
        REQUIRE_FALSE(config.is_valid());

        config.mmr_lambda = 0.7;
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid fetch_k") {
        config.fetch_k = 0;
        REQUIRE_FALSE(config.is_valid());

        config.fetch_k = 10001;
        REQUIRE_FALSE(config.is_valid());

        // fetch_k must be >= top_k
        config.top_k = 10;
        config.fetch_k = 5;
        REQUIRE_FALSE(config.is_valid());

        config.fetch_k = 15;
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid cache_ttl") {
        config.cache_ttl_seconds = -1.0;
        REQUIRE_FALSE(config.is_valid());

        config.cache_ttl_seconds = 0.0;
        REQUIRE_FALSE(config.is_valid());

        config.cache_ttl_seconds = 300.0;
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid batch_size") {
        config.batch_size = 0;
        REQUIRE_FALSE(config.is_valid());

        config.batch_size = 1001;
        REQUIRE_FALSE(config.is_valid());

        config.batch_size = 64;
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid similarity_metric") {
        config.similarity_metric = "invalid";
        REQUIRE_FALSE(config.is_valid());

        std::vector<std::string> valid_metrics = {
            "cosine", "euclidean", "manhattan"
        };

        for (const auto& metric : valid_metrics) {
            config.similarity_metric = metric;
            REQUIRE(config.is_valid());
        }
    }
}

TEST_CASE("RetrievalConfig - Default Configurations", "[core][config][retrieval]") {
    SECTION("BM25 default") {
        auto config = RetrievalConfig::get_default_for_search_type("bm25");
        REQUIRE(config.search_type == "bm25");
        REQUIRE(config.top_k == 10);
        REQUIRE(config.enable_stop_words == true);
    }

    SECTION("MMR default") {
        auto config = RetrievalConfig::get_default_for_search_type("mmr");
        REQUIRE(config.search_type == "mmr");
        REQUIRE(config.mmr_lambda == 0.7);
        REQUIRE(config.fetch_k == 50);
    }

    SECTION("Similarity default") {
        auto config = RetrievalConfig::get_default_for_search_type("similarity");
        REQUIRE(config.search_type == "similarity");
        REQUIRE(config.score_threshold.has_value());
        REQUIRE(config.score_threshold.value() == 0.0);
    }
}

TEST_CASE("VectorStoreConfig - Default Values", "[core][config][vector_store]") {
    VectorStoreConfig config;

    REQUIRE(config.dimension == 384);
    REQUIRE(config.metric == "cosine");
    REQUIRE(config.max_elements == 1000000);
    REQUIRE(config.storage_backend == "memory");
    REQUIRE(config.index_type == "hnsw");
    REQUIRE(config.m == 16);
    REQUIRE(config.ef_construction == 200);
    REQUIRE(config.ef_search == 50);
    REQUIRE(config.nlist == 100);
    REQUIRE(config.nprobe == 10);
    REQUIRE_FALSE(config.mmap_file_path.has_value());
    REQUIRE_FALSE(config.read_only);
}

TEST_CASE("VectorStoreConfig - Validation", "[core][config][vector_store]") {
    VectorStoreConfig config;

    SECTION("Valid configuration") {
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid dimension") {
        config.dimension = 0;
        REQUIRE_FALSE(config.is_valid());

        config.dimension = 10001;
        REQUIRE_FALSE(config.is_valid());

        config.dimension = 768;
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid metric") {
        config.metric = "invalid";
        REQUIRE_FALSE(config.is_valid());

        std::vector<std::string> valid_metrics = {
            "cosine", "euclidean", "manhattan", "dotproduct"
        };

        for (const auto& metric : valid_metrics) {
            config.metric = metric;
            REQUIRE(config.is_valid());
        }
    }

    SECTION("Invalid storage_backend") {
        config.storage_backend = "invalid";
        REQUIRE_FALSE(config.is_valid());

        std::vector<std::string> valid_backends = {
            "memory", "mmap", "redis"
        };

        for (const auto& backend : valid_backends) {
            config.storage_backend = backend;
            REQUIRE(config.is_valid());
        }
    }

    SECTION("Invalid index_type") {
        config.index_type = "invalid";
        REQUIRE_FALSE(config.is_valid());

        std::vector<std::string> valid_indices = {
            "hnsw", "ivf", "flat"
        };

        for (const auto& index : valid_indices) {
            config.index_type = index;
            REQUIRE(config.is_valid());
        }
    }

    SECTION("Invalid HNSW parameters") {
        config.m = 0;
        REQUIRE_FALSE(config.is_valid());

        config.m = 201;
        REQUIRE_FALSE(config.is_valid());

        config.m = 32;
        REQUIRE(config.is_valid());

        config.ef_construction = 0;
        REQUIRE_FALSE(config.is_valid());

        config.ef_construction = 1001;
        REQUIRE_FALSE(config.is_valid());

        config.ef_construction = 400;
        REQUIRE(config.is_valid());

        config.ef_search = 0;
        REQUIRE_FALSE(config.is_valid());

        config.ef_search = 1001;
        REQUIRE_FALSE(config.is_valid());

        config.ef_search = 100;
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid IVF parameters") {
        config.nlist = 0;
        REQUIRE_FALSE(config.is_valid());

        config.nlist = 100001;
        REQUIRE_FALSE(config.is_valid());

        config.nlist = 1000;
        REQUIRE(config.is_valid());

        config.nprobe = 0;
        REQUIRE_FALSE(config.is_valid());

        config.nprobe = config.nlist + 1;  // Cannot exceed nlist
        REQUIRE_FALSE(config.is_valid());

        config.nprobe = 50;
        REQUIRE(config.is_valid());
    }
}

TEST_CASE("VectorStoreConfig - Default Configurations", "[core][config][vector_store]") {
    SECTION("HNSW configuration") {
        auto config = VectorStoreConfig::get_hnsw_config(768, 100000);
        REQUIRE(config.dimension == 768);
        REQUIRE(config.max_elements == 100000);
        REQUIRE(config.index_type == "hnsw");
        REQUIRE(config.metric == "cosine");
        REQUIRE(config.m == 16);
        REQUIRE(config.ef_construction == 200);
        REQUIRE(config.ef_search == 50);
    }

    SECTION("IVF configuration") {
        auto config = VectorStoreConfig::get_ivf_config(512, 50000);
        REQUIRE(config.dimension == 512);
        REQUIRE(config.max_elements == 50000);
        REQUIRE(config.index_type == "ivf");
        REQUIRE(config.metric == "cosine");
        REQUIRE(config.nlist == 224);  // sqrt(50000) limited to 10000
        REQUIRE(config.nprobe == 22);  // nlist / 10 limited to 100
    }
}

TEST_CASE("MemoryConfig - Default Values", "[core][config][memory]") {
    MemoryConfig config;

    REQUIRE(config.max_messages == 100);
    REQUIRE(config.max_tokens == 4000);
    REQUIRE(config.strategy == "buffer");
    REQUIRE(config.storage_backend == "memory");
    REQUIRE_FALSE(config.memory_key.has_value());
    REQUIRE(config.summary_prompt == "Summarize the conversation history concisely:");
    REQUIRE(config.max_summary_tokens == 500);
    REQUIRE(config.window_ratio == 0.8);
    REQUIRE_FALSE(config.storage_path.has_value());
    REQUIRE_FALSE(config.persistent);
}

TEST_CASE("MemoryConfig - Validation", "[core][config][memory]") {
    MemoryConfig config;

    SECTION("Valid configuration") {
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid max_messages") {
        config.max_messages = 0;
        REQUIRE_FALSE(config.is_valid());

        config.max_messages = 10001;
        REQUIRE_FALSE(config.is_valid());

        config.max_messages = 50;
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid max_tokens") {
        config.max_tokens = 0;
        REQUIRE_FALSE(config.is_valid());

        config.max_tokens = 100001;
        REQUIRE_FALSE(config.is_valid());

        config.max_tokens = 8000;
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid strategy") {
        config.strategy = "invalid";
        REQUIRE_FALSE(config.is_valid());

        std::vector<std::string> valid_strategies = {
            "buffer", "summary", "sliding_window"
        };

        for (const auto& strategy : valid_strategies) {
            config.strategy = strategy;
            REQUIRE(config.is_valid());
        }
    }

    SECTION("Invalid storage_backend") {
        config.storage_backend = "invalid";
        REQUIRE_FALSE(config.is_valid());

        std::vector<std::string> valid_backends = {
            "memory", "file", "redis"
        };

        for (const auto& backend : valid_backends) {
            config.storage_backend = backend;
            REQUIRE(config.is_valid());
        }
    }

    SECTION("Invalid window_ratio") {
        config.window_ratio = -0.1;
        REQUIRE_FALSE(config.is_valid());

        config.window_ratio = 1.1;
        REQUIRE_FALSE(config.is_valid());

        config.window_ratio = 0.5;
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid max_summary_tokens") {
        config.max_summary_tokens = 0;
        REQUIRE_FALSE(config.is_valid());

        config.max_summary_tokens = config.max_tokens + 1;  // Cannot exceed max_tokens
        REQUIRE_FALSE(config.is_valid());

        config.max_summary_tokens = config.max_tokens / 2;
        REQUIRE(config.is_valid());
    }
}

TEST_CASE("MemoryConfig - Default Configurations", "[core][config][memory]") {
    SECTION("Buffer configuration") {
        auto config = MemoryConfig::get_buffer_config(200, 8000);
        REQUIRE(config.strategy == "buffer");
        REQUIRE(config.max_messages == 200);
        REQUIRE(config.max_tokens == 8000);
    }

    SECTION("Summary configuration") {
        auto config = MemoryConfig::get_summary_config(6000);
        REQUIRE(config.strategy == "summary");
        REQUIRE(config.max_tokens == 6000);
        REQUIRE(config.max_summary_tokens == 1500);  // 25% of max_tokens
    }
}

TEST_CASE("LLMConfig - Default Values", "[core][config][llm]") {
    LLMConfig config;

    REQUIRE(config.model_name == "gpt-3.5-turbo");
    REQUIRE(config.api_base == "https://api.openai.com/v1");
    REQUIRE(config.api_key.empty());
    REQUIRE(config.organization.empty());
    REQUIRE(config.max_connections == 10);
    REQUIRE(config.connection_timeout == std::chrono::milliseconds(30000));
    REQUIRE(config.read_timeout == std::chrono::milliseconds(60000));
    REQUIRE(config.max_retries == 3);
    REQUIRE(config.retry_delay == std::chrono::milliseconds(1000));
    REQUIRE_FALSE(config.requests_per_minute.has_value());
    REQUIRE_FALSE(config.tokens_per_minute.has_value());
    REQUIRE(config.enable_cache == true);
    REQUIRE(config.cache_size == 1000);
    REQUIRE(config.cache_ttl == std::chrono::hours(1));
}

TEST_CASE("LLMConfig - Validation", "[core][config][llm]") {
    LLMConfig config;

    SECTION("Valid configuration") {
        config.api_key = "test_key";
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid model_name") {
        config.model_name = "";
        REQUIRE_FALSE(config.is_valid());

        config.model_name = "gpt-4";
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid api_base") {
        config.api_base = "";
        REQUIRE_FALSE(config.is_valid());

        config.api_base = "https://api.example.com/v1";
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid max_connections") {
        config.max_connections = 0;
        REQUIRE_FALSE(config.is_valid());

        config.max_connections = 1001;
        REQUIRE_FALSE(config.is_valid());

        config.max_connections = 20;
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid timeouts") {
        config.connection_timeout = std::chrono::milliseconds(0);
        REQUIRE_FALSE(config.is_valid());

        config.connection_timeout = std::chrono::milliseconds(300001);
        REQUIRE_FALSE(config.is_valid());

        config.connection_timeout = std::chrono::milliseconds(15000);
        REQUIRE(config.is_valid());

        config.read_timeout = std::chrono::milliseconds(0);
        REQUIRE_FALSE(config.is_valid());

        config.read_timeout = std::chrono::milliseconds(600001);
        REQUIRE_FALSE(config.is_valid());

        config.read_timeout = std::chrono::milliseconds(30000);
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid max_retries") {
        config.max_retries = 11;
        REQUIRE_FALSE(config.is_valid());

        config.max_retries = 5;
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid retry_delay") {
        config.retry_delay = std::chrono::milliseconds(60001);
        REQUIRE_FALSE(config.is_valid());

        config.retry_delay = std::chrono::milliseconds(2000);
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid cache_size") {
        config.cache_size = 0;
        REQUIRE_FALSE(config.is_valid());

        config.cache_size = 100001;
        REQUIRE_FALSE(config.is_valid());

        config.cache_size = 5000;
        REQUIRE(config.is_valid());
    }
}

TEST_CASE("LLMConfig - OpenAI Configuration", "[core][config][llm]") {
    auto config = LLMConfig::get_openai_config("sk-test123", "gpt-4");

    REQUIRE(config.api_base == "https://api.openai.com/v1");
    REQUIRE(config.model_name == "gpt-4");
    REQUIRE(config.api_key == "sk-test123");
    REQUIRE(config.max_connections == 10);
    REQUIRE(config.requests_per_minute.has_value());
    REQUIRE(config.requests_per_minute.value() == 3000);
}

TEST_CASE("EmbeddingConfig - Default Values", "[core][config][embedding]") {
    EmbeddingConfig config;

    REQUIRE(config.model_name == "text-embedding-ada-002");
    REQUIRE(config.api_base == "https://api.openai.com/v1");
    REQUIRE(config.api_key.empty());
    REQUIRE(config.embedding_dimension == 1536);
    REQUIRE(config.batch_size == 100);
    REQUIRE(config.timeout == std::chrono::milliseconds(30000));
    REQUIRE(config.max_connections == 5);
    REQUIRE(config.max_retries == 3);
    REQUIRE(config.enable_cache == true);
    REQUIRE(config.cache_size == 10000);
}

TEST_CASE("EmbeddingConfig - Validation", "[core][config][embedding]") {
    EmbeddingConfig config;

    SECTION("Valid configuration") {
        config.api_key = "test_key";
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid embedding_dimension") {
        config.embedding_dimension = 0;
        REQUIRE_FALSE(config.is_valid());

        config.embedding_dimension = 10001;
        REQUIRE_FALSE(config.is_valid());

        config.embedding_dimension = 768;
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid batch_size") {
        config.batch_size = 0;
        REQUIRE_FALSE(config.is_valid());

        config.batch_size = 1001;
        REQUIRE_FALSE(config.is_valid());

        config.batch_size = 50;
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid timeout") {
        config.timeout = std::chrono::milliseconds(0);
        REQUIRE_FALSE(config.is_valid());

        config.timeout = std::chrono::milliseconds(300001);
        REQUIRE_FALSE(config.is_valid());

        config.timeout = std::chrono::milliseconds(15000);
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid max_connections") {
        config.max_connections = 0;
        REQUIRE_FALSE(config.is_valid());

        config.max_connections = 101;
        REQUIRE_FALSE(config.is_valid());

        config.max_connections = 10;
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid cache_size") {
        config.cache_size = 0;
        REQUIRE_FALSE(config.is_valid());

        config.cache_size = 100001;
        REQUIRE_FALSE(config.is_valid());

        config.cache_size = 50000;
        REQUIRE(config.is_valid());
    }
}

TEST_CASE("EmbeddingConfig - OpenAI Configuration", "[core][config][embedding]") {
    auto config = EmbeddingConfig::get_openai_config("sk-test123", "text-embedding-3-small");

    REQUIRE(config.api_base == "https://api.openai.com/v1");
    REQUIRE(config.model_name == "text-embedding-3-small");
    REQUIRE(config.api_key == "sk-test123");
    REQUIRE(config.embedding_dimension == 1536);  // Default dimension
}

TEST_CASE("LangChainConfig - Default Values", "[core][config][global]") {
    auto config = LangChainConfig::get_default();

    REQUIRE(config.log_level == "info");
    REQUIRE_FALSE(config.log_file.has_value());
    REQUIRE(config.enable_console_logging == true);
    REQUIRE(config.default_thread_pool_size > 0);
    REQUIRE(config.enable_performance_metrics == true);
    REQUIRE(config.metrics_collection_interval == std::chrono::seconds(60));
    REQUIRE(config.enable_memory_pooling == true);
    REQUIRE(config.default_memory_pool_size == 1024 * 1024);
    REQUIRE(config.enable_simd == true);
    REQUIRE_FALSE(config.prefer_avx512);
    REQUIRE(config.enable_async_operations == true);
    REQUIRE(config.max_concurrent_operations == 100);
}

TEST_CASE("LangChainConfig - Validation", "[core][config][global]") {
    LangChainConfig config = LangChainConfig::get_default();

    SECTION("Valid configuration") {
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid log_level") {
        config.log_level = "invalid";
        REQUIRE_FALSE(config.is_valid());

        std::vector<std::string> valid_levels = {
            "debug", "info", "warn", "error"
        };

        for (const auto& level : valid_levels) {
            config.log_level = level;
            REQUIRE(config.is_valid());
        }
    }

    SECTION("Invalid default_thread_pool_size") {
        config.default_thread_pool_size = 1001;
        REQUIRE_FALSE(config.is_valid());

        config.default_thread_pool_size = 8;
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid metrics_collection_interval") {
        config.metrics_collection_interval = std::chrono::seconds(0);
        REQUIRE_FALSE(config.is_valid());

        config.metrics_collection_interval = std::chrono::seconds(3601);
        REQUIRE_FALSE(config.is_valid());

        config.metrics_collection_interval = std::chrono::seconds(30);
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid default_memory_pool_size") {
        config.default_memory_pool_size = 0;
        REQUIRE_FALSE(config.is_valid());

        config.default_memory_pool_size = 1024 * 1024 * 1024 + 1;
        REQUIRE_FALSE(config.is_valid());

        config.default_memory_pool_size = 2 * 1024 * 1024;  // 2MB
        REQUIRE(config.is_valid());
    }

    SECTION("Invalid max_concurrent_operations") {
        config.max_concurrent_operations = 0;
        REQUIRE_FALSE(config.is_valid());

        config.max_concurrent_operations = 10001;
        REQUIRE_FALSE(config.is_valid());

        config.max_concurrent_operations = 200;
        REQUIRE(config.is_valid());
    }
}