#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

#ifdef __x86_64__
#include <immintrin.h>  // For SIMD intrinsics (x86_64 only)
#endif

namespace langchain::utils {

/**
 * @brief SIMD-optimized vector operations for high-performance similarity calculations
 */
class VectorOps {
public:
    /**
     * @brief Check AVX512 support at runtime
     * @return True if AVX512 is supported
     */
    static bool is_avx512_supported() {
#if defined(__AVX512F__) && defined(__x86_64__)
        uint32_t eax, ebx, ecx, edx;
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        return (ebx & (1 << 16)) != 0;  // AVX512F bit
#else
        return false;
#endif
    }

    /**
     * @brief Check AVX2 support at runtime
     * @return True if AVX2 is supported
     */
    static bool is_avx2_supported() {
#if defined(__AVX2__) && defined(__x86_64__)
        uint32_t eax, ebx, ecx, edx;
        __cpuid(7, eax, ebx, ecx, edx);
        return (ebx & (1 << 5)) != 0;  // AVX2 bit
#else
        return false;
#endif
    }

    /**
     * @brief Compute cosine similarity using the best available SIMD instruction set
     * @param a First vector
     * @param b Second vector
     * @param dim Vector dimension
     * @return Cosine similarity
     */
    static float cosine_similarity(const float* a, const float* b, size_t dim) {
        if (dim == 0) return 0.0f;

        if (is_avx512_supported() && dim >= 16) {
            return cosine_similarity_avx512(a, b, dim);
        } else if (is_avx2_supported() && dim >= 8) {
            return cosine_similarity_avx2(a, b, dim);
        } else {
            return cosine_similarity_scalar(a, b, dim);
        }
    }

    /**
     * @brief Compute cosine similarity for std::vector
     * @param a First vector
     * @param b Second vector
     * @return Cosine similarity
     */
    static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            return 0.0f;
        }
        return cosine_similarity(a.data(), b.data(), a.size());
    }

    /**
     * @brief Batch cosine similarity computation
     * @param query_vec Query vector
     * @param doc_matrix Document matrix (doc_count x dim)
     * @param similarities Output array for similarities
     * @param doc_count Number of documents
     * @param dim Vector dimension
     */
    static void cosine_similarity_batch(const float* query_vec,
                                       const float* doc_matrix,
                                       float* similarities,
                                       size_t doc_count,
                                       size_t dim) {
        for (size_t i = 0; i < doc_count; ++i) {
            const float* doc_vec = doc_matrix + i * dim;
            similarities[i] = cosine_similarity(query_vec, doc_vec, dim);
        }
    }

    /**
     * @brief Compute Euclidean distance
     * @param a First vector
     * @param b Second vector
     * @param dim Vector dimension
     * @return Euclidean distance
     */
    static float euclidean_distance(const float* a, const float* b, size_t dim) {
        if (is_avx512_supported() && dim >= 16) {
            return euclidean_distance_avx512(a, b, dim);
        } else if (is_avx2_supported() && dim >= 8) {
            return euclidean_distance_avx2(a, b, dim);
        } else {
            return euclidean_distance_scalar(a, b, dim);
        }
    }

    /**
     * @brief Compute Manhattan distance
     * @param a First vector
     * @param b Second vector
     * @param dim Vector dimension
     * @return Manhattan distance
     */
    static float manhattan_distance(const float* a, const float* b, size_t dim) {
        if (is_avx2_supported() && dim >= 8) {
            return manhattan_distance_avx2(a, b, dim);
        } else {
            return manhattan_distance_scalar(a, b, dim);
        }
    }

    /**
     * @brief Compute dot product
     * @param a First vector
     * @param b Second vector
     * @param dim Vector dimension
     * @return Dot product
     */
    static float dot_product(const float* a, const float* b, size_t dim) {
        if (is_avx512_supported() && dim >= 16) {
            return dot_product_avx512(a, b, dim);
        } else if (is_avx2_supported() && dim >= 8) {
            return dot_product_avx2(a, b, dim);
        } else {
            return dot_product_scalar(a, b, dim);
        }
    }

    /**
     * @brief Normalize vector to unit length
     * @param vec Vector to normalize
     * @param dim Vector dimension
     */
    static void normalize(float* vec, size_t dim) {
        float norm = std::sqrt(dot_product(vec, vec, dim));
        if (norm > 0.0f) {
            float inv_norm = 1.0f / norm;
            if (is_avx512_supported() && dim >= 16) {
                normalize_vector_avx512(vec, dim);
            } else if (is_avx2_supported() && dim >= 8) {
                normalize_vector_avx2(vec, dim);
            } else {
                for (size_t i = 0; i < dim; ++i) {
            vec[i] *= inv_norm;
        }
            }
        }
    }

    /**
     * @brief Normalize std::vector
     * @param vec Vector to normalize
     */
    static void normalize(std::vector<float>& vec) {
        normalize(vec.data(), vec.size());
    }

public:
    // Scalar implementations (fallback) - for testing
    static float cosine_similarity_scalar(const float* a, const float* b, size_t dim) {
        float dot_product = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;

        for (size_t i = 0; i < dim; ++i) {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if (norm_a == 0.0f || norm_b == 0.0f) {
            return 0.0f;
        }

        return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    static float euclidean_distance_scalar(const float* a, const float* b, size_t dim) {
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    static float manhattan_distance_scalar(const float* a, const float* b, size_t dim) {
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            sum += std::abs(a[i] - b[i]);
        }
        return sum;
    }

    static float dot_product_scalar(const float* a, const float* b, size_t dim) {
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    static void normalize_scalar(float* vec, size_t dim, float inv_norm) {
        for (size_t i = 0; i < dim; ++i) {
            vec[i] *= inv_norm;
        }
    }

#if defined(__AVX2__) && defined(__x86_64__)
    // AVX2 implementations
    static float cosine_similarity_avx2(const float* a, const float* b, size_t dim) {
        __m256 dot_sum = _mm256_setzero_ps();
        __m256 norm_a_sum = _mm256_setzero_ps();
        __m256 norm_b_sum = _mm256_setzero_ps();

        size_t simd_end = (dim / 8) * 8;
        for (size_t i = 0; i < simd_end; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);

            dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
            norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
            norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
        }

        // Horizontal sum
        float dot_array[8];
        float norm_a_array[8];
        float norm_b_array[8];

        _mm256_storeu_ps(dot_array, dot_sum);
        _mm256_storeu_ps(norm_a_array, norm_a_sum);
        _mm256_storeu_ps(norm_b_array, norm_b_sum);

        float dot_product = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;

        for (int i = 0; i < 8; ++i) {
            dot_product += dot_array[i];
            norm_a += norm_a_array[i];
            norm_b += norm_b_array[i];
        }

        // Handle remaining elements
        for (size_t i = simd_end; i < dim; ++i) {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if (norm_a == 0.0f || norm_b == 0.0f) {
            return 0.0f;
        }

        return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    static float euclidean_distance_avx2(const float* a, const float* b, size_t dim) {
        __m256 sum_vec = _mm256_setzero_ps();

        size_t simd_end = (dim / 8) * 8;
        for (size_t i = 0; i < simd_end; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 diff = _mm256_sub_ps(va, vb);
            sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
        }

        // Horizontal sum
        float sum_array[8];
        _mm256_storeu_ps(sum_array, sum_vec);

        float sum = 0.0f;
        for (int i = 0; i < 8; ++i) {
            sum += sum_array[i];
        }

        // Handle remaining elements
        for (size_t i = simd_end; i < dim; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }

        return std::sqrt(sum);
    }

    static float dot_product_avx2(const float* a, const float* b, size_t dim) {
        __m256 sum_vec = _mm256_setzero_ps();

        size_t simd_end = (dim / 8) * 8;
        for (size_t i = 0; i < simd_end; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
        }

        // Horizontal sum
        float sum_array[8];
        _mm256_storeu_ps(sum_array, sum_vec);

        float sum = 0.0f;
        for (int i = 0; i < 8; ++i) {
            sum += sum_array[i];
        }

        // Handle remaining elements
        for (size_t i = simd_end; i < dim; ++i) {
            sum += a[i] * b[i];
        }

        return sum;
    }

    static void normalize_avx2(float* vec, size_t dim, float inv_norm) {
        __m256 inv_norm_vec = _mm256_set1_ps(inv_norm);

        size_t simd_end = (dim / 8) * 8;
        for (size_t i = 0; i < simd_end; i += 8) {
            __m256 v = _mm256_loadu_ps(&vec[i]);
            v = _mm256_mul_ps(v, inv_norm_vec);
            _mm256_storeu_ps(&vec[i], v);
        }

        // Handle remaining elements
        for (size_t i = simd_end; i < dim; ++i) {
            vec[i] *= inv_norm;
        }
    }

    static float manhattan_distance_avx2(const float* a, const float* b, size_t dim) {
        __m256 sum_vec = _mm256_setzero_ps();

        size_t simd_end = (dim / 8) * 8;
        for (size_t i = 0; i < simd_end; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 diff = _mm256_sub_ps(va, vb);
            __m256 abs_diff = _mm256_abs_ps(diff);
            sum_vec = _mm256_add_ps(sum_vec, abs_diff);
        }

        // Horizontal sum
        float sum_array[8];
        _mm256_storeu_ps(sum_array, sum_vec);

        float sum = 0.0f;
        for (int i = 0; i < 8; ++i) {
            sum += sum_array[i];
        }

        // Handle remaining elements
        for (size_t i = simd_end; i < dim; ++i) {
            sum += std::abs(a[i] - b[i]);
        }

        return sum;
    }
#endif

#if defined(__AVX512F__) && defined(__x86_64__)
    // AVX512 implementations
    static float cosine_similarity_avx512(const float* a, const float* b, size_t dim) {
        __m512 dot_sum = _mm512_setzero_ps();
        __m512 norm_a_sum = _mm512_setzero_ps();
        __m512 norm_b_sum = _mm512_setzero_ps();

        size_t simd_end = (dim / 16) * 16;
        for (size_t i = 0; i < simd_end; i += 16) {
            __m512 va = _mm512_loadu_ps(&a[i]);
            __m512 vb = _mm512_loadu_ps(&b[i]);

            dot_sum = _mm512_fmadd_ps(va, vb, dot_sum);
            norm_a_sum = _mm512_fmadd_ps(va, va, norm_a_sum);
            norm_b_sum = _mm512_fmadd_ps(vb, vb, norm_b_sum);
        }

        // Horizontal sum
        float dot_product = _mm512_reduce_add_ps(dot_sum);
        float norm_a = _mm512_reduce_add_ps(norm_a_sum);
        float norm_b = _mm512_reduce_add_ps(norm_b_sum);

        // Handle remaining elements
        for (size_t i = simd_end; i < dim; ++i) {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if (norm_a == 0.0f || norm_b == 0.0f) {
            return 0.0f;
        }

        return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    static float euclidean_distance_avx512(const float* a, const float* b, size_t dim) {
        __m512 sum_vec = _mm512_setzero_ps();

        size_t simd_end = (dim / 16) * 16;
        for (size_t i = 0; i < simd_end; i += 16) {
            __m512 va = _mm512_loadu_ps(&a[i]);
            __m512 vb = _mm512_loadu_ps(&b[i]);
            __m512 diff = _mm512_sub_ps(va, vb);
            sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec);
        }

        float sum = _mm512_reduce_add_ps(sum_vec);

        // Handle remaining elements
        for (size_t i = simd_end; i < dim; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }

        return std::sqrt(sum);
    }

    static float dot_product_avx512(const float* a, const float* b, size_t dim) {
        __m512 sum_vec = _mm512_setzero_ps();

        size_t simd_end = (dim / 16) * 16;
        for (size_t i = 0; i < simd_end; i += 16) {
            __m512 va = _mm512_loadu_ps(&a[i]);
            __m512 vb = _mm512_loadu_ps(&b[i]);
            sum_vec = _mm512_fmadd_ps(va, vb, sum_vec);
        }

        float sum = _mm512_reduce_add_ps(sum_vec);

        // Handle remaining elements
        for (size_t i = simd_end; i < dim; ++i) {
            sum += a[i] * b[i];
        }

        return sum;
    }

    static void normalize_avx512(float* vec, size_t dim, float inv_norm) {
        __m512 inv_norm_vec = _mm512_set1_ps(inv_norm);

        size_t simd_end = (dim / 16) * 16;
        for (size_t i = 0; i < simd_end; i += 16) {
            __m512 v = _mm512_loadu_ps(&vec[i]);
            v = _mm512_mul_ps(v, inv_norm_vec);
            _mm512_storeu_ps(&vec[i], v);
        }

        // Handle remaining elements
        for (size_t i = simd_end; i < dim; ++i) {
            vec[i] *= inv_norm;
        }
    }
#endif

// Fallback implementations for non-x86 platforms
#ifndef __x86_64__
    static float cosine_similarity_avx2(const float* a, const float* b, size_t dim) {
        return cosine_similarity(a, b, dim);
    }

    static float euclidean_distance_avx2(const float* a, const float* b, size_t dim) {
        return euclidean_distance(a, b, dim);
    }

    static float dot_product_avx2(const float* a, const float* b, size_t dim) {
        return dot_product(a, b, dim);
    }

    static void normalize_vector_avx2(float* vec, size_t dim) {
        float norm = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            norm += vec[i] * vec[i];
        }
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            float inv_norm = 1.0f / norm;
            for (size_t i = 0; i < dim; ++i) {
                vec[i] *= inv_norm;
            }
        }
    }

    static float manhattan_distance_avx2(const float* a, const float* b, size_t dim) {
        return manhattan_distance(a, b, dim);
    }

    static float cosine_similarity_avx512(const float* a, const float* b, size_t dim) {
        return cosine_similarity(a, b, dim);
    }

    static float euclidean_distance_avx512(const float* a, const float* b, size_t dim) {
        return euclidean_distance(a, b, dim);
    }

    static float dot_product_avx512(const float* a, const float* b, size_t dim) {
        return dot_product(a, b, dim);
    }

    static void normalize_vector_avx512(float* vec, size_t dim) {
        float norm = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            norm += vec[i] * vec[i];
        }
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            float inv_norm = 1.0f / norm;
            for (size_t i = 0; i < dim; ++i) {
                vec[i] *= inv_norm;
            }
        }
    }

    static float manhattan_distance_avx512(const float* a, const float* b, size_t dim) {
        return manhattan_distance(a, b, dim);
    }
#endif
};

} // namespace langchain::utils