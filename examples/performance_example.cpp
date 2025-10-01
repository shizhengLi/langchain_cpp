#include "langchain/langchain.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <random>
#include <iomanip>

using namespace langchain;

int main() {
    std::cout << "=== LangChain++ Performance Optimization Example ===" << std::endl;

    try {
        // 1. Memory Pool Performance
        std::cout << "\n=== Memory Pool Performance ===" << std::endl;

        memory::MemoryPool memory_pool(1024 * 1024); // 1MB pool
        std::cout << "Created 1MB memory pool" << std::endl;

        // Test memory allocation performance
        const int num_allocations = 10000;
        const size_t allocation_size = 256;

        // Standard allocation
        auto standard_start = std::chrono::high_resolution_clock::now();
        std::vector<void*> standard_pointers;
        for (int i = 0; i < num_allocations; ++i) {
            standard_pointers.push_back(std::malloc(allocation_size));
        }
        auto standard_end = std::chrono::high_resolution_clock::now();

        // Memory pool allocation
        auto pool_start = std::chrono::high_resolution_clock::now();
        std::vector<void*> pool_pointers;
        for (int i = 0; i < num_allocations; ++i) {
            pool_pointers.push_back(memory_pool.allocate(allocation_size));
        }
        auto pool_end = std::chrono::high_resolution_clock::now();

        auto standard_time = std::chrono::duration_cast<std::chrono::microseconds>(
            standard_end - standard_start).count();
        auto pool_time = std::chrono::duration_cast<std::chrono::microseconds>(
            pool_end - pool_start).count();

        std::cout << "Allocation Performance (" << num_allocations << " allocations of " << allocation_size << " bytes):" << std::endl;
        std::cout << "  Standard malloc: " << standard_time << " μs" << std::endl;
        std::cout << "  Memory pool:     " << pool_time << " μs" << std::endl;
        std::cout << "  Speedup:         " << std::fixed << std::setprecision(2)
                 << (double)standard_time / pool_time << "x" << std::endl;

        // Memory usage statistics
        std::cout << "\nMemory Pool Statistics:" << std::endl;
        std::cout << "  Total allocated: " << memory_pool.total_allocated() << " bytes" << std::endl;
        std::cout << "  Available:      " << memory_pool.available() << " bytes" << std::endl;
        std::cout << "  Utilization:    " << std::fixed << std::setprecision(1)
                 << (double)memory_pool.total_allocated() / memory_pool.total_capacity() * 100 << "%" << std::endl;
        std::cout << "  Fragmentation:  " << memory_pool.fragmentation() * 100 << "%" << std::endl;

        // Cleanup
        for (auto ptr : standard_pointers) {
            std::free(ptr);
        }
        for (auto ptr : pool_pointers) {
            memory_pool.deallocate(ptr);
        }

        // 2. Concurrent Processing
        std::cout << "\n=== Concurrent Processing ===" << std::endl;

        // Create sample documents for concurrent processing
        std::vector<Document> documents;
        documents.reserve(1000);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(50, 500);

        for (int i = 0; i < 1000; ++i) {
            std::string content = "Document " + std::to_string(i) + " content with random length " +
                                 std::string(dis(gen), 'x') + " and some text for processing.";
            documents.push_back({
                content,
                {{"id", std::to_string(i)}, {"type", i % 3 == 0 ? "technical" : "general"}}
            });
        }

        std::cout << "Created " << documents.size() << " test documents" << std::endl;

        // Test sequential vs concurrent processing
        auto text_processor = std::make_shared<text::TextProcessor>();

        // Sequential processing
        auto sequential_start = std::chrono::high_resolution_clock::now();
        for (auto& doc : documents) {
            doc.content = text_processor->normalize_text(doc.content);
            auto tokens = text_processor->tokenize(doc.content);
            // Simulate some processing work
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        auto sequential_end = std::chrono::high_resolution_clock::now();

        // Concurrent processing
        auto concurrent_start = std::chrono::high_resolution_clock::now();

        const int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        int docs_per_thread = documents.size() / num_threads;

        for (int t = 0; t < num_threads; ++t) {
            int start_idx = t * docs_per_thread;
            int end_idx = (t == num_threads - 1) ? documents.size() : (t + 1) * docs_per_thread;

            threads.emplace_back([&, start_idx, end_idx]() {
                for (int i = start_idx; i < end_idx; ++i) {
                    documents[i].content = text_processor->normalize_text(documents[i].content);
                    auto tokens = text_processor->tokenize(documents[i].content);
                    // Simulate some processing work
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
        auto concurrent_end = std::chrono::high_resolution_clock::now();

        auto sequential_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            sequential_end - sequential_start).count();
        auto concurrent_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            concurrent_end - concurrent_start).count();

        std::cout << "Processing Performance:" << std::endl;
        std::cout << "  Sequential:   " << sequential_time << " ms" << std::endl;
        std::cout << "  Concurrent:   " << concurrent_time << " ms" << std::endl;
        std::cout << "  Speedup:      " << std::fixed << std::setprecision(2)
                 << (double)sequential_time / concurrent_time << "x" << std::endl;
        std::cout << "  Efficiency:   " << std::fixed << std::setprecision(1)
                 << (double)sequential_time / concurrent_time / num_threads * 100 << "%" << std::endl;

        // 3. Caching Performance
        std::cout << "\n=== Caching Performance ===" << std::endl;

        cache::LRUCache<std::string, std::string> lru_cache(100);
        cache::MemoryCache<std::string, std::vector<std::string>> memory_cache(50);

        std::cout << "Created LRU cache (100 entries) and Memory cache (50 entries)" << std::endl;

        // Generate test data
        std::vector<std::string> cache_keys;
        for (int i = 0; i < 200; ++i) {
            cache_keys.push_back("key_" + std::to_string(i));
        }

        // Test cache performance
        const int cache_operations = 10000;

        // Without cache (simulated expensive operation)
        auto no_cache_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < cache_operations; ++i) {
            std::string key = cache_keys[i % cache_keys.size()];
            // Simulate expensive computation
            std::string result = "expensive_result_for_" + key;
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        auto no_cache_end = std::chrono::high_resolution_clock::now();

        // With cache
        auto with_cache_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < cache_operations; ++i) {
            std::string key = cache_keys[i % cache_keys.size()];

            // Try to get from cache
            if (!lru_cache.get(key)) {
                // Simulate expensive computation
                std::string result = "expensive_result_for_" + key;
                lru_cache.put(key, result);
            }
        }
        auto with_cache_end = std::chrono::high_resolution_clock::now();

        auto no_cache_time = std::chrono::duration_cast<std::chrono::microseconds>(
            no_cache_end - no_cache_start).count();
        auto with_cache_time = std::chrono::duration_cast<std::chrono::microseconds>(
            with_cache_end - with_cache_start).count();

        std::cout << "Cache Performance (" << cache_operations << " operations):" << std::endl;
        std::cout << "  No cache:     " << no_cache_time << " μs" << std::endl;
        std::cout << "  With cache:   " << with_cache_time << " μs" << std::endl;
        std::cout << "  Speedup:      " << std::fixed << std::setprecision(2)
                 << (double)no_cache_time / with_cache_time << "x" << std::endl;

        // Cache statistics
        std::cout << "\nCache Statistics:" << std::endl;
        std::cout << "  LRU Cache:" << std::endl;
        std::cout << "    Size:       " << lru_cache.size() << " / " << lru_cache.capacity() << std::endl;
        std::cout << "    Hit rate:   " << std::fixed << std::setprecision(1) << lru_cache.hit_rate() * 100 << "%" << std::endl;
        std::cout << "    Hits:       " << lru_cache.hits() << std::endl;
        std::cout << "    Misses:     " << lru_cache.misses() << std::endl;

        // 4. SIMD-Optimized Text Processing
        std::cout << "\n=== SIMD-Optimized Text Processing ===" << std::endl;

        // Create large text for processing
        std::string large_text;
        large_text.reserve(100000);
        for (int i = 0; i < 10000; ++i) {
            large_text += "This is a sample sentence for processing. ";
            large_text += "It contains various words and punctuation. ";
        }

        std::cout << "Created test text (" << large_text.length() << " characters)" << std::endl;

        // Test standard vs SIMD processing
        auto simd_processor = std::make_shared<text::SIMDTextProcessor>();

        // Standard text processing
        auto standard_text_start = std::chrono::high_resolution_clock::now();
        auto standard_tokens = text_processor->tokenize(large_text);
        auto standard_text_end = std::chrono::high_resolution_clock::now();

        // SIMD text processing
        auto simd_text_start = std::chrono::high_resolution_clock::now();
        auto simd_tokens = simd_processor->tokenize_simd(large_text);
        auto simd_text_end = std::chrono::high_resolution_clock::now();

        auto standard_text_time = std::chrono::duration_cast<std::chrono::microseconds>(
            standard_text_end - standard_text_start).count();
        auto simd_text_time = std::chrono::duration_cast<std::chrono::microseconds>(
            simd_text_end - simd_text_start).count();

        std::cout << "Text Tokenization Performance:" << std::endl;
        std::cout << "  Standard: " << standard_tokens.size() << " tokens in " << standard_text_time << " μs" << std::endl;
        std::cout << "  SIMD:     " << simd_tokens.size() << " tokens in " << simd_text_time << " μs" << std::endl;
        std::cout << "  Speedup:  " << std::fixed << std::setprecision(2)
                 << (double)standard_text_time / simd_text_time << "x" << std::endl;

        // Verify results are similar
        bool results_similar = std::abs((int)standard_tokens.size() - (int)simd_tokens.size()) < 10;
        std::cout << "  Results similar: " << (results_similar ? "YES" : "NO") << std::endl;

        // 5. I/O Optimization
        std::cout << "\n=== I/O Optimization ===" << std::endl;

        // Test buffered vs unbuffered file operations
        std::vector<std::string> test_lines;
        for (int i = 0; i < 1000; ++i) {
            test_lines.push_back("Line " + std::to_string(i) + " with some content for file I/O testing.\n");
        }

        // Unbuffered writing
        auto unbuffered_start = std::chrono::high_resolution_clock::now();
        std::ofstream unbuffered_file("/tmp/unbuffered_test.txt");
        for (const auto& line : test_lines) {
            unbuffered_file << line;
            unbuffered_file.flush(); // Force immediate write
        }
        unbuffered_file.close();
        auto unbuffered_end = std::chrono::high_resolution_clock::now();

        // Buffered writing
        auto buffered_start = std::chrono::high_resolution_clock::now();
        std::ofstream buffered_file("/tmp/buffered_test.txt");
        for (const auto& line : test_lines) {
            buffered_file << line;
        }
        buffered_file.close();
        auto buffered_end = std::chrono::high_resolution_clock::now();

        auto unbuffered_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            unbuffered_end - unbuffered_start).count();
        auto buffered_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            buffered_end - buffered_start).count();

        std::cout << "File I/O Performance (" << test_lines.size() << " lines):" << std::endl;
        std::cout << "  Unbuffered: " << unbuffered_time << " ms" << std::endl;
        std::cout << "  Buffered:   " << buffered_time << " ms" << std::endl;
        std::cout << "  Speedup:    " << std::fixed << std::setprecision(2)
                 << (double)unbuffered_time / buffered_time << "x" << std::endl;

        // 6. Database Connection Pooling
        std::cout << "\n=== Database Connection Pooling ===" << std::endl;

        // Simulate database operations with and without connection pooling
        const int db_operations = 100;

        // Without connection pooling (create new connection each time)
        auto no_pool_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < db_operations; ++i) {
            // Simulate connection creation
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            // Simulate database operation
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            // Simulate connection cleanup
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        auto no_pool_end = std::chrono::high_resolution_clock::now();

        // With connection pooling
        std::vector<std::thread> db_threads;
        std::mutex pool_mutex;
        int pool_connections = 5;
        int active_connections = 0;

        auto with_pool_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < db_operations; ++i) {
            db_threads.emplace_back([&, i]() {
                // Simulate getting connection from pool
                {
                    std::lock_guard<std::mutex> lock(pool_mutex);
                    while (active_connections >= pool_connections) {
                        // Wait for available connection
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                    }
                    active_connections++;
                }

                // Simulate database operation
                std::this_thread::sleep_for(std::chrono::microseconds(100));

                // Return connection to pool
                {
                    std::lock_guard<std::mutex> lock(pool_mutex);
                    active_connections--;
                }
            });
        }

        for (auto& thread : db_threads) {
            thread.join();
        }
        auto with_pool_end = std::chrono::high_resolution_clock::now();

        auto no_pool_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            no_pool_end - no_pool_start).count();
        auto with_pool_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            with_pool_end - with_pool_start).count();

        std::cout << "Database Performance (" << db_operations << " operations):" << std::endl;
        std::cout << "  No pooling: " << no_pool_time << " ms" << std::endl;
        std::cout << "  With pool:  " << with_pool_time << " ms" << std::endl;
        std::cout << "  Speedup:    " << std::fixed << std::setprecision(2)
                 << (double)no_pool_time / with_pool_time << "x" << std::endl;

        // 7. Performance Profiling
        std::cout << "\n=== Performance Profiling ===" << std::endl;

        performance::Profiler profiler;
        profiler.start_profiling();

        // Profile different operations
        {
            auto scope = profiler.create_scope("text_processing");
            for (int i = 0; i < 100; ++i) {
                text_processor->tokenize("Sample text for profiling " + std::to_string(i));
            }
        }

        {
            auto scope = profiler.create_scope("document_indexing");
            for (int i = 0; i < 50; ++i) {
                Document doc{"Sample document " + std::to_string(i), {{"id", std::to_string(i)}}};
                // Simulate indexing operation
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

        {
            auto scope = profiler.create_scope("memory_operations");
            for (int i = 0; i < 200; ++i) {
                void* ptr = memory_pool.allocate(128);
                memory_pool.deallocate(ptr);
            }
        }

        profiler.stop_profiling();

        // Display profiling results
        auto report = profiler.generate_report();
        std::cout << "Performance Profile:" << std::endl;
        for (const auto& entry : report) {
            std::cout << "  " << entry.name << ": " << entry.total_time << " μs"
                     << " (" << entry.call_count << " calls, avg: "
                     << entry.total_time / entry.call_count << " μs)" << std::endl;
        }

        // 8. Resource Usage Monitoring
        std::cout << "\n=== Resource Usage Monitoring ===" << std::endl;

        performance::ResourceMonitor resource_monitor;
        resource_monitor.start_monitoring();

        // Simulate resource-intensive operations
        std::vector<std::thread> resource_threads;
        for (int i = 0; i < 4; ++i) {
            resource_threads.emplace_back([&]() {
                for (int j = 0; j < 1000; ++j) {
                    // CPU-intensive operation
                    std::string dummy;
                    dummy.reserve(1000);
                    for (int k = 0; k < 100; ++k) {
                        dummy += std::to_string(k * k);
                    }

                    // Memory operation
                    void* ptr = memory_pool.allocate(256);
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                    memory_pool.deallocate(ptr);
                }
            });
        }

        for (auto& thread : resource_threads) {
            thread.join();
        }

        resource_monitor.stop_monitoring();

        // Display resource usage
        auto cpu_usage = resource_monitor.get_cpu_usage();
        auto memory_usage = resource_monitor.get_memory_usage();

        std::cout << "Resource Usage:" << std::endl;
        std::cout << "  CPU Usage:    " << std::fixed << std::setprecision(1) << cpu_usage << "%" << std::endl;
        std::cout << "  Memory Usage: " << memory_usage / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Threads:      " << resource_monitor.get_thread_count() << std::endl;

        // 9. Performance Recommendations
        std::cout << "\n=== Performance Recommendations ===" << std::endl;

        performance::PerformanceAnalyzer analyzer;

        // Analyze the performance data collected
        analyzer.add_metric("memory_pool_speedup", (double)standard_time / pool_time);
        analyzer.add_metric("concurrent_speedup", (double)sequential_time / concurrent_time);
        analyzer.add_metric("cache_speedup", (double)no_cache_time / with_cache_time);
        analyzer.add_metric("simd_speedup", (double)standard_text_time / simd_text_time);
        analyzer.add_metric("io_speedup", (double)unbuffered_time / buffered_time);

        auto recommendations = analyzer.generate_recommendations();

        std::cout << "Performance Analysis Results:" << std::endl;
        for (const auto& rec : recommendations) {
            std::cout << "  - " << rec << std::endl;
        }

        // 10. Performance Benchmarks
        std::cout << "\n=== Performance Benchmarks ===" << std::endl;

        performance::BenchmarkSuite benchmarks;

        // Add benchmarks
        benchmarks.add_benchmark("text_tokenization", [&]() {
            auto tokens = text_processor->tokenize(large_text);
            return tokens.size();
        });

        benchmarks.add_benchmark("memory_allocation", [&]() {
            void* ptr = memory_pool.allocate(1024);
            memory_pool.deallocate(ptr);
            return 1;
        });

        benchmarks.add_benchmark("cache_operations", [&]() {
            std::string key = "benchmark_key";
            std::string value = "benchmark_value";
            lru_cache.put(key, value);
            auto result = lru_cache.get(key);
            return result.has_value();
        });

        // Run benchmarks
        auto benchmark_results = benchmarks.run_all(100); // 100 iterations each

        std::cout << "Benchmark Results (100 iterations each):" << std::endl;
        for (const auto& result : benchmark_results) {
            std::cout << "  " << result.name << ":" << std::endl;
            std::cout << "    Average: " << result.average_time << " μs" << std::endl;
            std::cout << "    Min:     " << result.min_time << " μs" << std::endl;
            std::cout << "    Max:     " << result.max_time << " μs" << std::endl;
            std::cout << "    StdDev:  " << std::fixed << std::setprecision(2) << result.std_deviation << " μs" << std::endl;
        }

        std::cout << "\n=== Performance Optimization Example completed successfully! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}