#include "langchain/vectorstores/simple_vector_store.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>
#include <random>

using namespace langchain::vectorstores;

// Helper function to generate random vectors
SimpleVector generate_random_vector(size_t dim, double min_val = -1.0, double max_val = 1.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min_val, max_val);

    SimpleVector vector(dim);
    for (size_t i = 0; i < dim; ++i) {
        vector[i] = dis(gen);
    }
    return vector;
}

// Helper function to generate unit vectors
SimpleVector generate_unit_vector(size_t dim) {
    SimpleVector vector = generate_random_vector(dim);
    double mag = vector.magnitude();
    if (mag > 0.0) {
        for (double& val : vector.data) {
            val /= mag;
        }
    }
    return vector;
}

TEST_CASE("Simple Vector Operations", "[simple_vector][operations]") {
    SECTION("Basic vector operations") {
        SimpleVector v1({1.0, 2.0, 3.0});
        SimpleVector v2({4.0, 5.0, 6.0});

        // Dot product
        double dot = v1.dot(v2);
        REQUIRE(dot == Catch::Approx(32.0));

        // Magnitude
        double mag1 = v1.magnitude();
        REQUIRE(mag1 == Catch::Approx(std::sqrt(14.0)));

        double mag2 = v2.magnitude();
        REQUIRE(mag2 == Catch::Approx(std::sqrt(77.0)));

        // Cosine similarity
        double cosine = v1.cosine_similarity(v2);
        double expected = 32.0 / (std::sqrt(14.0) * std::sqrt(77.0));
        REQUIRE(cosine == Catch::Approx(expected));
    }

    SECTION("Vector operations edge cases") {
        SimpleVector zero_vector({0.0, 0.0, 0.0});
        SimpleVector unit_vector({1.0, 0.0, 0.0});

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

TEST_CASE("Simple Vector Store Configuration", "[simple_vector][config]") {
    SECTION("Valid configuration") {
        SimpleVectorStoreConfig config;
        config.vector_dim = 384;
        config.max_vectors = 10000;
        config.default_top_k = 10;
        config.similarity_threshold = 0.5;

        REQUIRE_NOTHROW(config.validate());
    }

    SECTION("Invalid configuration") {
        SimpleVectorStoreConfig config;

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
    }
}

TEST_CASE("Simple Vector Store", "[simple_vector][store]") {
    SECTION("Basic operations") {
        SimpleVectorStoreConfig config;
        config.vector_dim = 3;
        config.max_vectors = 100;
        config.normalize_vectors = true;

        SimpleVectorStore store(config);

        // Add vectors
        SimpleVector v1({1.0, 0.0, 0.0});
        SimpleVector v2({0.0, 1.0, 0.0});
        SimpleVector v3({0.0, 0.0, 1.0});

        size_t id1 = store.add_vector(v1, "Content 1");
        size_t id2 = store.add_vector(v2, "Content 2");
        size_t id3 = store.add_vector(v3, "Content 3");

        REQUIRE(id1 == 1);
        REQUIRE(id2 == 2);
        REQUIRE(id3 == 3);
        REQUIRE(store.size() == 3);
        REQUIRE_FALSE(store.empty());

        // Search
        SimpleVector query({1.0, 0.0, 0.0});
        auto results = store.search(query, 2);

        REQUIRE(results.size() == 2);
        REQUIRE(results[0].entry_id == id1);
        REQUIRE(results[0].content == "Content 1");
        REQUIRE(results[0].similarity_score == Catch::Approx(1.0));

        // Get vector
        auto entry = store.get_vector(id1);
        REQUIRE(entry.has_value());
        REQUIRE(entry->id == id1);
        REQUIRE(entry->content == "Content 1");

        // Update vector
        SimpleVector updated({1.0, 1.0, 0.0});
        bool updated_success = store.update_vector(id1, updated, "Updated content");
        REQUIRE(updated_success);

        auto updated_entry = store.get_vector(id1);
        REQUIRE(updated_entry->content == "Updated content");

        // Delete vector
        bool deleted_success = store.delete_vector(id1);
        REQUIRE(deleted_success);
        REQUIRE(store.size() == 2);

        auto deleted_entry = store.get_vector(id1);
        REQUIRE_FALSE(deleted_entry.has_value());
    }

    SECTION("Batch operations") {
        SimpleVectorStoreConfig config;
        config.vector_dim = 2;
        config.max_vectors = 100;

        SimpleVectorStore store(config);

        std::vector<SimpleVectorEntry> vectors = {
            SimpleVectorEntry(0, SimpleVector({1.0, 0.0}), "Vector 1"),
            SimpleVectorEntry(0, SimpleVector({0.0, 1.0}), "Vector 2"),
            SimpleVectorEntry(0, SimpleVector({1.0, 1.0}), "Vector 3")
        };

        auto ids = store.add_vectors(vectors);
        REQUIRE(ids.size() == 3);
        REQUIRE(store.size() == 3);

        // Search for all vectors
        SimpleVector query({1.0, 0.0});
        auto results = store.search(query, 5);
        REQUIRE(results.size() == 3);
    }

    SECTION("Similarity threshold") {
        SimpleVectorStoreConfig config;
        config.vector_dim = 2;
        config.similarity_threshold = 0.5;

        SimpleVectorStore store(config);

        // Add orthogonal vectors
        store.add_vector(SimpleVector({1.0, 0.0}), "Horizontal");
        store.add_vector(SimpleVector({0.0, 1.0}), "Vertical");
        store.add_vector(SimpleVector({1.0, 1.0}), "Diagonal");

        // Search with low threshold
        SimpleVector query({1.0, 0.0});
        auto results = store.search(query, 10);
        REQUIRE(results.size() >= 2); // Should find similar vectors

        // Search with high threshold
        config.similarity_threshold = 0.8;
        store.update_config(config);
        auto filtered_results = store.search(query, 10);
        REQUIRE(filtered_results.size() <= results.size());
    }

    SECTION("Error handling") {
        SimpleVectorStoreConfig config;
        config.vector_dim = 3;
        config.max_vectors = 2; // Small limit

        SimpleVectorStore store(config);

        // Wrong dimension (test this first)
        REQUIRE_THROWS_AS(store.add_vector(SimpleVector({1.0, 0.0})),
                          std::invalid_argument);

        // Add vectors up to limit
        store.add_vector(SimpleVector({1.0, 0.0, 0.0}));
        store.add_vector(SimpleVector({0.0, 1.0, 0.0}));

        // Exceed capacity
        REQUIRE_THROWS_AS(store.add_vector(SimpleVector({0.0, 0.0, 1.0})),
                          std::runtime_error);

        // Search with wrong dimension
        REQUIRE_THROWS_AS(store.search(SimpleVector({1.0, 0.0})),
                          std::invalid_argument);

        // Update non-existent vector
        bool updated = store.update_vector(999, SimpleVector({1.0, 1.0, 1.0}));
        REQUIRE_FALSE(updated);

        // Delete non-existent vector
        bool deleted = store.delete_vector(999);
        REQUIRE_FALSE(deleted);
    }

    SECTION("Clear operations") {
        SimpleVectorStoreConfig config;
        config.vector_dim = 2;

        SimpleVectorStore store(config);

        // Add vectors
        store.add_vector(SimpleVector({1.0, 0.0}));
        store.add_vector(SimpleVector({0.0, 1.0}));
        REQUIRE(store.size() == 2);

        // Clear all
        store.clear();
        REQUIRE(store.size() == 0);
        REQUIRE(store.empty());

        // Search after clear
        SimpleVector query({1.0, 0.0});
        auto results = store.search(query, 10);
        REQUIRE(results.empty());
    }

    SECTION("Performance and metadata") {
        SimpleVectorStoreConfig config;
        config.vector_dim = 128;
        config.max_vectors = 1000;

        SimpleVectorStore store(config);

        // Add many vectors
        for (int i = 0; i < 100; ++i) {
            SimpleVector v = generate_unit_vector(128);
            store.add_vector(v, "Vector " + std::to_string(i));
        }

        REQUIRE(store.size() == 100);

        // Get metadata
        auto metadata = store.get_metadata();
        REQUIRE(std::any_cast<size_t>(metadata["vector_count"]) == 100);
        REQUIRE(std::any_cast<size_t>(metadata["dimension"]) == 128);

        // Update config
        SimpleVectorStoreConfig new_config = config;
        new_config.default_top_k = 20;
        store.update_config(new_config);
    }
}

TEST_CASE("Simple Vector Store Integration", "[simple_vector][integration]") {
    SECTION("Large scale operations") {
        SimpleVectorStoreConfig config;
        config.vector_dim = 64;
        config.max_vectors = 10000;

        SimpleVectorStore store(config);

        // Add many vectors
        size_t num_vectors = 1000;
        std::vector<size_t> ids;
        ids.reserve(num_vectors);

        for (size_t i = 0; i < num_vectors; ++i) {
            SimpleVector v = generate_random_vector(64);
            size_t id = store.add_vector(v, "Document " + std::to_string(i));
            ids.push_back(id);
        }

        REQUIRE(store.size() == num_vectors);

        // Perform multiple searches
        for (int i = 0; i < 10; ++i) {
            SimpleVector query = generate_random_vector(64);
            auto results = store.search(query, 10);
            REQUIRE(results.size() <= 10);

            // Verify results are sorted by similarity
            for (size_t j = 1; j < results.size(); ++j) {
                REQUIRE(results[j-1].similarity_score >= results[j].similarity_score);
            }
        }

        // Test metadata access
        auto metadata = store.get_metadata();
        REQUIRE(std::any_cast<size_t>(metadata["vector_count"]) == num_vectors);
    }

    SECTION("Thread safety basic test") {
        SimpleVectorStoreConfig config;
        config.vector_dim = 8;
        config.max_vectors = 100;

        SimpleVectorStore store(config);

        // Add vectors
        for (int i = 0; i < 5; ++i) {
            SimpleVector v = SimpleVector(std::vector<double>(8, static_cast<double>(i)));
            store.add_vector(v, "Vector " + std::to_string(i));
        }

        REQUIRE(store.size() == 5);
    }
}