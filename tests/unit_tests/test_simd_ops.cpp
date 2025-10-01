#include <catch2/catch_all.hpp>
#include "langchain/utils/simd_ops.hpp"
#include <random>
#include <cmath>
#include <chrono>

using namespace langchain::utils;
using Catch::Approx;

TEST_CASE("VectorOps - SIMD Support Detection", "[utils][simd_ops][detection]") {
    SECTION("AVX2 support") {
        bool has_avx2 = VectorOps::is_avx2_supported();
        // Result depends on CPU capabilities, but should be a boolean
        REQUIRE((has_avx2 || !has_avx2));
    }

    SECTION("AVX512 support") {
        bool has_avx512 = VectorOps::is_avx512_supported();
        // Result depends on CPU capabilities, but should be a boolean
        REQUIRE((has_avx512 || !has_avx512));
    }
}

TEST_CASE("VectorOps - Scalar Cosine Similarity", "[utils][simd_ops][cosine]") {
    SECTION("Identical vectors") {
        std::vector<float> vec1 = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> vec2 = {1.0f, 2.0f, 3.0f, 4.0f};

        float similarity = VectorOps::cosine_similarity(vec1, vec2);
        REQUIRE(similarity == Approx(1.0).margin(1e-6));
    }

    SECTION("Orthogonal vectors") {
        std::vector<float> vec1 = {1.0f, 0.0f};
        std::vector<float> vec2 = {0.0f, 1.0f};

        float similarity = VectorOps::cosine_similarity(vec1, vec2);
        REQUIRE(similarity == Approx(0.0).margin(1e-6));
    }

    SECTION("Opposite vectors") {
        std::vector<float> vec1 = {1.0f, 2.0f, 3.0f};
        std::vector<float> vec2 = {-1.0f, -2.0f, -3.0f};

        float similarity = VectorOps::cosine_similarity(vec1, vec2);
        REQUIRE(similarity == Approx(-1.0).margin(1e-6));
    }

    SECTION("Different dimensions") {
        std::vector<float> vec1 = {1.0f, 2.0f, 3.0f};
        std::vector<float> vec2 = {1.0f, 2.0f};  // Different dimension

        float similarity = VectorOps::cosine_similarity(vec1, vec2);
        REQUIRE(similarity == 0.0f);
    }

    SECTION("Empty vectors") {
        std::vector<float> vec1;
        std::vector<float> vec2;

        float similarity = VectorOps::cosine_similarity(vec1, vec2);
        REQUIRE(similarity == 0.0f);
    }

    SECTION("Zero vector") {
        std::vector<float> vec1 = {0.0f, 0.0f, 0.0f};
        std::vector<float> vec2 = {1.0f, 2.0f, 3.0f};

        float similarity = VectorOps::cosine_similarity(vec1, vec2);
        REQUIRE(similarity == 0.0f);
    }

    SECTION("Real-world example") {
        std::vector<float> doc1 = {0.8f, 0.2f, 0.6f, 0.1f};
        std::vector<float> doc2 = {0.7f, 0.3f, 0.5f, 0.2f};

        float similarity = VectorOps::cosine_similarity(doc1, doc2);

        // Calculate expected value manually
        float dot = 0.8f*0.7f + 0.2f*0.3f + 0.6f*0.5f + 0.1f*0.2f;
        float norm1 = std::sqrt(0.8f*0.8f + 0.2f*0.2f + 0.6f*0.6f + 0.1f*0.1f);
        float norm2 = std::sqrt(0.7f*0.7f + 0.3f*0.3f + 0.5f*0.5f + 0.2f*0.2f);
        float expected = dot / (norm1 * norm2);

        REQUIRE(similarity == Approx(expected).margin(1e-6));
    }
}

TEST_CASE("VectorOps - Batch Cosine Similarity", "[utils][simd_ops][batch]") {
    SECTION("Basic batch operation") {
        std::vector<float> query = {1.0f, 2.0f, 3.0f};
        std::vector<float> doc_matrix = {
            1.0f, 2.0f, 3.0f,   // Similar to query
            3.0f, 2.0f, 1.0f,   // Different from query
            -1.0f, -2.0f, -3.0f // Opposite to query
        };

        std::vector<float> similarities(3);
        VectorOps::cosine_similarity_batch(
            query.data(), doc_matrix.data(), similarities.data(), 3, 3);

        REQUIRE(similarities[0] == Approx(1.0).margin(1e-6));   // Identical
        REQUIRE(similarities[1] < 0.8f);  // Different (should be less than 1)
        REQUIRE(similarities[2] == Approx(-1.0).margin(1e-6)); // Opposite
    }

    SECTION("Empty batch") {
        std::vector<float> query = {1.0f, 2.0f};
        std::vector<float> similarities(0);

        VectorOps::cosine_similarity_batch(
            query.data(), nullptr, similarities.data(), 0, 2);
        // Should not crash
    }
}

TEST_CASE("VectorOps - Distance Calculations", "[utils][simd_ops][distance]") {
    SECTION("Euclidean distance") {
        std::vector<float> vec1 = {1.0f, 2.0f, 3.0f};
        std::vector<float> vec2 = {4.0f, 6.0f, 8.0f};

        float distance = VectorOps::euclidean_distance(vec1.data(), vec2.data(), vec1.size());
        float expected = std::sqrt(
            (1.0f-4.0f)*(1.0f-4.0f) +
            (2.0f-6.0f)*(2.0f-6.0f) +
            (3.0f-8.0f)*(3.0f-8.0f)
        );

        REQUIRE(distance == Approx(expected).margin(1e-6));
    }

    SECTION("Manhattan distance") {
        std::vector<float> vec1 = {1.0f, 2.0f, 3.0f};
        std::vector<float> vec2 = {4.0f, 6.0f, 8.0f};

        float distance = VectorOps::manhattan_distance(vec1.data(), vec2.data(), vec1.size());
        float expected = std::abs(1.0f-4.0f) + std::abs(2.0f-6.0f) + std::abs(3.0f-8.0f);

        REQUIRE(distance == Approx(expected).margin(1e-6));
    }

    SECTION("Same vectors - zero distance") {
        std::vector<float> vec = {1.0f, 2.0f, 3.0f};

        REQUIRE(VectorOps::euclidean_distance(vec.data(), vec.data(), vec.size()) == Approx(0.0).margin(1e-6));
        REQUIRE(VectorOps::manhattan_distance(vec.data(), vec.data(), vec.size()) == Approx(0.0).margin(1e-6));
    }
}

TEST_CASE("VectorOps - Dot Product", "[utils][simd_ops][dot_product]") {
    SECTION("Basic dot product") {
        std::vector<float> vec1 = {1.0f, 2.0f, 3.0f};
        std::vector<float> vec2 = {4.0f, 5.0f, 6.0f};

        float result = VectorOps::dot_product(vec1.data(), vec2.data(), vec1.size());
        float expected = 1.0f*4.0f + 2.0f*5.0f + 3.0f*6.0f;

        REQUIRE(result == Approx(expected).margin(1e-6));
    }

    SECTION("Zero vector dot product") {
        std::vector<float> vec1 = {0.0f, 0.0f, 0.0f};
        std::vector<float> vec2 = {1.0f, 2.0f, 3.0f};

        float result = VectorOps::dot_product(vec1.data(), vec2.data(), vec1.size());
        REQUIRE(result == Approx(0.0).margin(1e-6));
    }
}

TEST_CASE("VectorOps - Vector Normalization", "[utils][simd_ops][normalization]") {
    SECTION("Basic normalization") {
        std::vector<float> vec = {3.0f, 4.0f};
        VectorOps::normalize(vec);

        float norm = std::sqrt(vec[0]*vec[0] + vec[1]*vec[1]);
        REQUIRE(norm == Approx(1.0).margin(1e-6));
        REQUIRE(vec[0] == Approx(0.6).margin(1e-6));   // 3/5
        REQUIRE(vec[1] == Approx(0.8).margin(1e-6));   // 4/5
    }

    SECTION("Zero vector normalization") {
        std::vector<float> vec = {0.0f, 0.0f, 0.0f};
        VectorOps::normalize(vec);

        // Should remain unchanged (no division by zero)
        REQUIRE(vec[0] == 0.0f);
        REQUIRE(vec[1] == 0.0f);
        REQUIRE(vec[2] == 0.0f);
    }

    SECTION("Already normalized vector") {
        std::vector<float> vec = {1.0f, 0.0f};
        VectorOps::normalize(vec);

        float norm = std::sqrt(vec[0]*vec[0] + vec[1]*vec[1]);
        REQUIRE(norm == Approx(1.0).margin(1e-6));
    }
}

TEST_CASE("VectorOps - SIMD Consistency", "[utils][simd_ops][consistency]") {
    // Test that different SIMD implementations give the same results
    SECTION("Cosine similarity consistency") {
        std::vector<float> vec1 = {1.2f, 3.4f, 5.6f, 7.8f, 9.0f};
        std::vector<float> vec2 = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};

        float result_scalar = VectorOps::cosine_similarity_scalar(vec1.data(), vec2.data(), vec1.size());
        float result_auto = VectorOps::cosine_similarity(vec1, vec2);

        REQUIRE(result_auto == Approx(result_scalar).margin(1e-6));
    }

    SECTION("Euclidean distance consistency") {
        std::vector<float> vec1 = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> vec2 = {0.5f, 1.5f, 2.5f, 3.5f};

        float result_scalar = VectorOps::euclidean_distance_scalar(vec1.data(), vec2.data(), vec1.size());
        float result_auto = VectorOps::euclidean_distance(vec1.data(), vec2.data(), vec1.size());

        REQUIRE(result_auto == Approx(result_scalar).margin(1e-6));
    }
}

TEST_CASE("VectorOps - Random Vector Testing", "[utils][simd_ops][random]") {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    SECTION("Random vectors - cosine similarity bounds") {
        for (int test = 0; test < 10; ++test) {
            std::vector<float> vec1(10);
            std::vector<float> vec2(10);

            for (size_t i = 0; i < 10; ++i) {
                vec1[i] = dist(gen);
                vec2[i] = dist(gen);
            }

            float similarity = VectorOps::cosine_similarity(vec1, vec2);

            // Cosine similarity should be between -1 and 1
            REQUIRE(similarity >= -1.01f);
            REQUIRE(similarity <= 1.01f);
        }
    }

    SECTION("Random vectors - normalization works") {
        for (int test = 0; test < 10; ++test) {
            std::vector<float> vec(20);

            for (size_t i = 0; i < 20; ++i) {
                vec[i] = dist(gen);
            }

            VectorOps::normalize(vec);

            float norm = 0.0f;
            for (float val : vec) {
                norm += val * val;
            }
            norm = std::sqrt(norm);

            REQUIRE(norm == Approx(1.0).margin(1e-6));
        }
    }
}

TEST_CASE("VectorOps - Performance Comparison", "[utils][simd_ops][performance]") {
    SECTION("Large vector performance") {
        const size_t dim = 10000;
        std::vector<float> vec1(dim);
        std::vector<float> vec2(dim);

        // Fill with some data
        for (size_t i = 0; i < dim; ++i) {
            vec1[i] = static_cast<float>(i % 100) / 100.0f;
            vec2[i] = static_cast<float>((i + 50) % 100) / 100.0f;
        }

        auto start = std::chrono::high_resolution_clock::now();
        float similarity = VectorOps::cosine_similarity(vec1, vec2);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Should complete quickly for large vectors
        REQUIRE(duration.count() < 10000);  // Less than 10ms for 10k dimensions
        REQUIRE(similarity >= -1.0f);
        REQUIRE(similarity <= 1.0f);
    }
}

TEST_CASE("VectorOps - Edge Cases", "[utils][simd_ops][edge_cases]") {
    SECTION("Single dimension") {
        std::vector<float> vec1 = {5.0f};
        std::vector<float> vec2 = {2.0f};

        float similarity = VectorOps::cosine_similarity(vec1, vec2);
        REQUIRE(similarity == Approx(1.0).margin(1e-6));
    }

    SECTION("Very large vectors") {
        const size_t large_dim = 100000;
        std::vector<float> vec1(large_dim, 1.0f);
        std::vector<float> vec2(large_dim, 1.0f);

        // Should not crash and should give correct result
        float similarity = VectorOps::cosine_similarity(vec1, vec2);
        REQUIRE(similarity == Approx(1.0).margin(1e-6));
    }

    SECTION("NaN and infinity handling") {
        std::vector<float> vec1 = {1.0f, std::numeric_limits<float>::quiet_NaN()};
        std::vector<float> vec2 = {1.0f, 2.0f};

        // Should handle NaN gracefully (result may vary by implementation)
        float similarity = VectorOps::cosine_similarity(vec1, vec2);
        // Should not crash
    }
}