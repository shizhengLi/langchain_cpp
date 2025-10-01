#include "langchain/vectorstores/vector_store.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include <random>

using namespace langchain::vectorstores;

// Helper function to generate random vectors
Vector generate_random_vector(size_t dim, double min_val = -1.0, double max_val = 1.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min_val, max_val);

    Vector vector(dim);
    for (size_t i = 0; i < dim; ++i) {
        vector[i] = dis(gen);
    }
    return vector;
}

// Helper function to generate unit vectors
Vector generate_unit_vector(size_t dim) {
    Vector vector = generate_random_vector(dim);
    double mag = vector.magnitude();
    if (mag > 0.0) {
        vector = vector.multiply_simd(1.0 / mag);
    }
    return vector;
}

TEST_CASE("Vector Operations", "[vector][operations]") {
    SECTION("Basic vector operations") {
        Vector v1({1.0, 2.0, 3.0});
        Vector v2({4.0, 5.0, 6.0});

        // Dot product
        double dot = v1.dot(v2);
        REQUIRE(dot == Approx(32.0));

        // Magnitude
        double mag1 = v1.magnitude();
        REQUIRE(mag1 == Approx(std::sqrt(14.0)));

        double mag2 = v2.magnitude();
        REQUIRE(mag2 == Approx(std::sqrt(77.0)));

        // Cosine similarity
        double cosine = v1.cosine_similarity(v2);
        double expected = 32.0 / (std::sqrt(14.0) * std::sqrt(77.0));
        REQUIRE(cosine == Approx(expected));

        // Euclidean distance
        double euclidean = v1.euclidean_distance(v2);
        REQUIRE(euclidean == Approx(std::sqrt(27.0)));
    }

    SECTION("SIMD operations") {
        size_t dim = 256;
        Vector v1 = generate_random_vector(dim);
        Vector v2 = generate_random_vector(dim);

        // Addition
        Vector add_result = v1.add_simd(v2);
        for (size_t i = 0; i < dim; ++i) {
            REQUIRE(add_result[i] == Approx(v1[i] + v2[i]));
        }

        // Subtraction
        Vector sub_result = v1.subtract_simd(v2);
        for (size_t i = 0; i < dim; ++i) {
            REQUIRE(sub_result[i] == Approx(v1[i] - v2[i]));
        }

        // Scalar multiplication
        double scalar = 2.5;
        Vector mul_result = v1.multiply_simd(scalar);
        for (size_t i = 0; i < dim; ++i) {
            REQUIRE(mul_result[i] == Approx(v1[i] * scalar));
        }

        // Dot product
        double dot_simd = v1.dot_simd(v2);
        double dot_normal = v1.dot(v2);
        REQUIRE(dot_simd == Approx(dot_normal));
    }

    SECTION("Vector operations edge cases") {
        Vector zero_vector({0.0, 0.0, 0.0});
        Vector unit_vector({1.0, 0.0, 0.0});

        // Zero vector magnitude
        REQUIRE(zero_vector.magnitude() == 0.0);

        // Zero vector similarity
        REQUIRE(zero_vector.cosine_similarity(unit_vector) == 0.0);
        REQUIRE(unit_vector.cosine_similarity(zero_vector) == 0.0);

        // Unit vector magnitude
        REQUIRE(unit_vector.magnitude() == 1.0);

        // Self similarity
        REQUIRE(unit_vector.cosine_similarity(unit_vector) == 1.0);
    }
}

TEST_CASE("Vector Store Configuration", "[vector][config]") {
    SECTION("Valid configuration") {
        VectorStoreConfig config;
        config.vector_dim = 384;
        config.max_vectors = 10000;
        config.default_top_k = 10;
        config.similarity_threshold = 0.5;
        config.cache_size = 1000;

        REQUIRE_NOTHROW(config.validate());
    }

    SECTION("Invalid configuration") {
        VectorStoreConfig config;

        SECTION("Invalid max_vectors") {
            config.max_vectors = 0;
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
        }

        SECTION("Invalid vector_dim") {
            config.vector_dim = 0;
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);

            config.vector_dim = 10001;
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
        }

        SECTION("Invalid default_top_k") {
            config.default_top_k = 0;
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
        }

        SECTION("Invalid similarity_threshold") {
            config.similarity_threshold = -0.1;
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);

            config.similarity_threshold = 1.1;
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
        }

        SECTION("Invalid cache_size") {
            config.cache_size = 0;
            REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
        }
    }
}

TEST_CASE("In-Memory Vector Store", "[vector][store]") {
    SECTION("Basic operations") {
        VectorStoreConfig config;
        config.vector_dim = 3;
        config.max_vectors = 100;
        config.normalize_vectors = true;

        auto store = std::make_unique<InMemoryVectorStore>(config);

        // Add vectors
        Vector v1({1.0, 0.0, 0.0});
        Vector v2({0.0, 1.0, 0.0});
        Vector v3({0.0, 0.0, 1.0});

        size_t id1 = store->add_vector(v1, "Content 1");
        size_t id2 = store->add_vector(v2, "Content 2");
        size_t id3 = store->add_vector(v3, "Content 3");

        REQUIRE(id1 == 1);
        REQUIRE(id2 == 2);
        REQUIRE(id3 == 3);
        REQUIRE(store->size() == 3);
        REQUIRE_FALSE(store->empty());

        // Search
        Vector query({1.0, 0.0, 0.0});
        auto results = store->search(query, 2);

        REQUIRE(results.size() == 2);
        REQUIRE(results[0].entry_id == id1);
        REQUIRE(results[0].content == "Content 1");
        REQUIRE(results[0].similarity_score == Approx(1.0));

        // Get vector
        auto entry = store->get_vector(id1);
        REQUIRE(entry.has_value());
        REQUIRE(entry->id == id1);
        REQUIRE(entry->content == "Content 1");

        // Update vector
        Vector updated({1.0, 1.0, 0.0});
        bool updated_success = store->update_vector(id1, updated, "Updated content");
        REQUIRE(updated_success);

        auto updated_entry = store->get_vector(id1);
        REQUIRE(updated_entry->content == "Updated content");

        // Delete vector
        bool deleted_success = store->delete_vector(id1);
        REQUIRE(deleted_success);
        REQUIRE(store->size() == 2);

        auto deleted_entry = store->get_vector(id1);
        REQUIRE_FALSE(deleted_entry.has_value());
    }

    SECTION("Batch operations") {
        VectorStoreConfig config;
        config.vector_dim = 2;
        config.max_vectors = 100;

        auto store = std::make_unique<InMemoryVectorStore>(config);

        std::vector<VectorEntry> vectors = {
            VectorEntry(0, Vector({1.0, 0.0}), "Vector 1"),
            VectorEntry(0, Vector({0.0, 1.0}), "Vector 2"),
            VectorEntry(0, Vector({1.0, 1.0}), "Vector 3")
        };

        auto ids = store->add_vectors(vectors);
        REQUIRE(ids.size() == 3);
        REQUIRE(store->size() == 3);

        // Search for all vectors
        Vector query({1.0, 0.0});
        auto results = store->search(query, 5);
        REQUIRE(results.size() == 3);
    }

    SECTION("Similarity threshold") {
        VectorStoreConfig config;
        config.vector_dim = 2;
        config.similarity_threshold = 0.5;

        auto store = std::make_unique<InMemoryVectorStore>(config);

        // Add orthogonal vectors
        store->add_vector(Vector({1.0, 0.0}), "Horizontal");
        store->add_vector(Vector({0.0, 1.0}), "Vertical");
        store->add_vector(Vector({1.0, 1.0}), "Diagonal");

        // Search with low threshold
        Vector query({1.0, 0.0});
        auto results = store->search(query, 10);
        REQUIRE(results.size() >= 2); // Should find similar vectors

        // Search with high threshold
        auto filtered_results = store->search_with_threshold(query, 0.8, 10);
        REQUIRE(filtered_results.size() <= results.size());
    }

    SECTION("Distance metrics") {
        size_t dim = 4;
        VectorStoreConfig config;
        config.vector_dim = dim;

        SECTION("Cosine similarity") {
            config.distance_metric = VectorStoreConfig::DistanceMetric::COSINE;
            auto store = std::make_unique<InMemoryVectorStore>(config);

            Vector v1 = generate_unit_vector(dim);
            Vector v2 = generate_unit_vector(dim);

            size_t id1 = store->add_vector(v1);
            size_t id2 = store->add_vector(v2);

            auto results = store->search(v1, 2);
            REQUIRE(results.size() == 2);
            REQUIRE(results[0].similarity_score >= 0.0);
            REQUIRE(results[0].similarity_score <= 1.0);
        }

        SECTION("Euclidean distance") {
            config.distance_metric = VectorStoreConfig::DistanceMetric::EUCLIDEAN;
            auto store = std::make_unique<InMemoryVectorStore>(config);

            Vector v1({1.0, 0.0, 0.0, 0.0});
            Vector v2({0.0, 1.0, 0.0, 0.0});

            size_t id1 = store->add_vector(v1);
            size_t id2 = store->add_vector(v2);

            auto results = store->search(v1, 2);
            REQUIRE(results.size() == 2);
            REQUIRE(results[0].similarity_score >= 0.0);
            REQUIRE(results[0].similarity_score <= 1.0);
        }

        SECTION("Dot product") {
            config.distance_metric = VectorStoreConfig::DistanceMetric::DOT_PRODUCT;
            auto store = std::make_unique<InMemoryVectorStore>(config);

            Vector v1({1.0, 1.0, 1.0, 1.0});
            Vector v2({1.0, 1.0, 1.0, 1.0});

            size_t id1 = store->add_vector(v1);
            size_t id2 = store->add_vector(v2);

            auto results = store->search(v1, 2);
            REQUIRE(results.size() == 2);
        }
    }

    SECTION("Caching") {
        VectorStoreConfig config;
        config.vector_dim = 3;
        config.cache_enabled = true;
        config.cache_size = 2;

        auto store = std::make_unique<InMemoryVectorStore>(config);

        // Add vectors
        store->add_vector(Vector({1.0, 0.0, 0.0}), "Vector 1");
        store->add_vector(Vector({0.0, 1.0, 0.0}), "Vector 2");

        Vector query({1.0, 0.0, 0.0});

        // First search (cache miss)
        auto results1 = store->search(query, 2);

        // Second search (cache hit)
        auto results2 = store->search(query, 2);

        REQUIRE(results1.size() == results2.size());

        // Check cache statistics
        auto stats = store->get_performance_stats();
        REQUIRE(stats["total_searches"] >= 2.0);
        REQUIRE(stats["cache_hits"] >= 1.0);
        REQUIRE(stats["cache_hit_rate"] > 0.0);
    }

    SECTION("Performance and optimization") {
        VectorStoreConfig config;
        config.vector_dim = 128;
        config.max_vectors = 1000;
        config.use_simd = true;

        auto store = std::make_unique<InMemoryVectorStore>(config);

        // Add many vectors
        std::vector<Vector> vectors;
        for (int i = 0; i < 100; ++i) {
            vectors.push_back(generate_unit_vector(128));
            store->add_vector(vectors.back(), "Vector " + std::to_string(i));
        }

        REQUIRE(store->size() == 100);

        // Batch search
        std::vector<Vector> queries;
        for (int i = 0; i < 5; ++i) {
            queries.push_back(generate_unit_vector(128));
        }

        auto batch_results = store->search_batch(queries, 10);
        REQUIRE(batch_results.size() == 50); // 5 queries * 10 results each

        // Optimize index
        REQUIRE_NOTHROW(store->optimize_index());

        // Get metadata
        auto metadata = store->get_metadata();
        REQUIRE(std::any_cast<size_t>(metadata["vector_count"]) == 100);
        REQUIRE(std::any_cast<size_t>(metadata["dimension"]) == 128);
    }

    SECTION("Error handling") {
        VectorStoreConfig config;
        config.vector_dim = 3;
        config.max_vectors = 2; // Small limit

        auto store = std::make_unique<InMemoryVectorStore>(config);

        // Add vectors up to limit
        store->add_vector(Vector({1.0, 0.0, 0.0}));
        store->add_vector(Vector({0.0, 1.0, 0.0}));

        // Exceed capacity
        REQUIRE_THROWS_AS(store->add_vector(Vector({0.0, 0.0, 1.0})),
                          std::runtime_error);

        // Wrong dimension
        REQUIRE_THROWS_AS(store->add_vector(Vector({1.0, 0.0})),
                          std::invalid_argument);

        // Search with wrong dimension
        REQUIRE_THROWS_AS(store->search(Vector({1.0, 0.0})),
                          std::invalid_argument);

        // Update non-existent vector
        bool updated = store->update_vector(999, Vector({1.0, 1.0, 1.0}));
        REQUIRE_FALSE(updated);

        // Delete non-existent vector
        bool deleted = store->delete_vector(999);
        REQUIRE_FALSE(deleted);
    }

    SECTION("Clear operations") {
        VectorStoreConfig config;
        config.vector_dim = 2;

        auto store = std::make_unique<InMemoryVectorStore>(config);

        // Add vectors
        store->add_vector(Vector({1.0, 0.0}));
        store->add_vector(Vector({0.0, 1.0}));
        REQUIRE(store->size() == 2);

        // Clear all
        store->clear();
        REQUIRE(store->size() == 0);
        REQUIRE(store->empty());

        // Search after clear
        Vector query({1.0, 0.0});
        auto results = store->search(query, 10);
        REQUIRE(results.empty());
    }
}

TEST_CASE("Vector Store Factory", "[vector][factory]") {
    SECTION("Create in-memory store") {
        VectorStoreConfig config;
        config.vector_dim = 10;
        config.max_vectors = 1000;

        auto store = VectorStoreFactory::create_in_memory_store(config);
        REQUIRE(store != nullptr);
        REQUIRE(store->get_dimension() == 10);

        // Test basic functionality
        Vector v(10, 1.0);
        size_t id = store->add_vector(v);
        REQUIRE(id == 1);
        REQUIRE(store->size() == 1);
    }

    SECTION("Create optimized stores") {
        SECTION("Search optimized") {
            auto store = VectorStoreFactory::create_optimized_store("search", 256);
            REQUIRE(store != nullptr);
            REQUIRE(store->get_dimension() == 256);
        }

        SECTION("Clustering optimized") {
            auto store = VectorStoreFactory::create_optimized_store("clustering", 128);
            REQUIRE(store != nullptr);
            REQUIRE(store->get_dimension() == 128);
        }

        SECTION("Recommendation optimized") {
            auto store = VectorStoreFactory::create_optimized_store("recommendation", 64);
            REQUIRE(store != nullptr);
            REQUIRE(store->get_dimension() == 64);
        }

        SECTION("Default optimized") {
            auto store = VectorStoreFactory::create_optimized_store("unknown", 32);
            REQUIRE(store != nullptr);
            REQUIRE(store->get_dimension() == 32);
        }
    }
}

TEST_CASE("Vector Store Integration", "[vector][integration]") {
    SECTION("Large scale operations") {
        VectorStoreConfig config;
        config.vector_dim = 64;
        config.max_vectors = 10000;
        config.use_simd = true;
        config.cache_enabled = true;
        config.cache_size = 100;

        auto store = std::make_unique<InMemoryVectorStore>(config);

        // Add many vectors
        size_t num_vectors = 1000;
        std::vector<size_t> ids;
        ids.reserve(num_vectors);

        for (size_t i = 0; i < num_vectors; ++i) {
            Vector v = generate_random_vector(64);
            size_t id = store->add_vector(v, "Document " + std::to_string(i));
            ids.push_back(id);
        }

        REQUIRE(store->size() == num_vectors);

        // Perform multiple searches
        for (int i = 0; i < 10; ++i) {
            Vector query = generate_random_vector(64);
            auto results = store->search(query, 10);
            REQUIRE(results.size() <= 10);
            REQUIRE(results.size() > 0);

            // Verify results are sorted by similarity
            for (size_t j = 1; j < results.size(); ++j) {
                REQUIRE(results[j-1].similarity_score >= results[j].similarity_score);
            }
        }

        // Check performance stats
        auto stats = store->get_performance_stats();
        REQUIRE(stats["total_searches"] >= 10.0);
        REQUIRE(stats["vector_count"] == static_cast<double>(num_vectors));

        // Test metadata access
        auto metadata = store->get_metadata();
        REQUIRE(std::any_cast<size_t>(metadata["vector_count"]) == num_vectors);
        REQUIRE(std::any_cast<size_t>(metadata["total_searches"]) >= 10);
    }

    SECTION("Concurrent operations safety") {
        VectorStoreConfig config;
        config.vector_dim = 8;
        config.max_vectors = 100;

        auto store = std::make_unique<InMemoryVectorStore>(config);

        // Add vectors from different threads (basic test)
        std::vector<std::thread> threads;
        for (int i = 0; i < 5; ++i) {
            threads.emplace_back([&store, i]() {
                Vector v(8, static_cast<double>(i));
                store->add_vector(v, "Thread " + std::to_string(i));
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        REQUIRE(store->size() == 5);
    }
}