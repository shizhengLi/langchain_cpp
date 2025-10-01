# LangChain++ Performance Optimization Guide

## Overview

This guide provides comprehensive performance optimization techniques and best practices for LangChain++ applications, covering profiling, optimization strategies, and benchmarking approaches.

## Table of Contents

- [Performance Profiling](#performance-profiling)
- [Memory Optimization](#memory-optimization)
- [CPU Optimization](#cpu-optimization)
- [I/O Optimization](#io-optimization)
- [Concurrency Optimization](#concurrency-optimization)
- [Caching Strategies](#caching-strategies)
- [Database Optimization](#database-optimization)
- [Network Optimization](#network-optimization)
- [Benchmarking](#benchmarking)
- [Performance Monitoring](#performance-monitoring)

---

## Performance Profiling

### Built-in Profiling Tools

LangChain++ includes built-in profiling capabilities:

```cpp
#include "langchain/metrics/metrics.hpp"

// Initialize metrics collector
MetricsCollector metrics;

// Create performance timer
{
    auto timer = metrics.create_timer("document_indexing");

    // Code to profile
    retriever.add_documents(documents);

} // Timer automatically records when it goes out of scope

// Get performance statistics
auto stats = metrics.get_histogram_stats("document_indexing_time");
std::cout << "Average indexing time: " << stats.mean << "ms" << std::endl;
```

### Manual Profiling

For detailed profiling, use system profilers:

```bash
# With perf (Linux)
perf record ./your_application
perf report

# With Instruments (macOS)
instruments -t "Time Profiler" ./your_application

# With gprof
gprof ./your_application gmon.out > analysis.txt
```

### Memory Profiling

```cpp
// Enable memory tracking
MemoryTracker tracker;

// Profile memory usage
{
    MemoryProfile profile = tracker.start_profile("indexing_operation");

    // Your code here
    build_large_index();

} // Profile automatically records memory usage

auto memory_stats = tracker.get_memory_stats("indexing_operation");
std::cout << "Peak memory: " << memory_stats.peak_memory << " MB" << std::endl;
```

---

## Memory Optimization

### 1. Use Memory Pools

```cpp
// Configure memory pools for frequent allocations
MemoryPoolConfig config;
config.document_pool_size = 1000;
config.vector_pool_size = 500;
config.token_pool_size = 10000;

MemoryManager::instance().configure(config);

// Use pools instead of new/delete
auto doc = MemoryManager::instance().allocate<Document>();
// ... use document ...
MemoryManager::instance().deallocate(doc);
```

### 2. Optimize Data Structures

#### Cache-Friendly Layout

```cpp
// Bad: Poor cache locality
struct BadDocument {
    std::string content;           // Large string
    std::string id;                // Small identifier
    std::vector<double> embedding; // Large vector
    std::unordered_map<std::string, std::string> metadata;
};

// Good: Cache-friendly layout
struct OptimizedDocument {
    // Small, frequently accessed fields first
    std::string id;
    uint64_t content_hash;
    size_t content_length;

    // Large, less frequently accessed fields last
    std::string content;
    std::vector<double> embedding;
    std::unordered_map<std::string, std::string> metadata;
};
```

#### Use Reserve for Containers

```cpp
// Bad: Multiple reallocations
std::vector<Document> documents;
for (const auto& doc : source_documents) {
    documents.push_back(doc);  // May cause reallocation
}

// Good: Pre-allocate memory
std::vector<Document> documents;
documents.reserve(source_documents.size());  // Allocate once
for (const auto& doc : source_documents) {
    documents.push_back(doc);
}
```

### 3. Use Move Semantics

```cpp
// Bad: Unnecessary copies
std::vector<Document> process_documents(std::vector<Document> docs) {
    std::vector<Document> results;
    for (const auto& doc : docs) {
        Document processed = heavy_processing(doc);  // Copy
        results.push_back(processed);  // Another copy
    }
    return results;  // Another copy
}

// Good: Use move semantics
std::vector<Document> process_documents(std::vector<Document> docs) {
    std::vector<Document> results;
    results.reserve(docs.size());

    for (auto&& doc : docs) {  // Rvalue reference
        Document processed = heavy_processing(std::move(doc));
        results.push_back(std::move(processed));
    }
    return results;
}
```

### 4. Optimize String Usage

```cpp
// Bad: Excessive string copying
std::string process_text(const std::string& text) {
    std::string result = text;  // Copy
    result = clean_text(result);  // Another copy
    result = tokenize_text(result);  // Another copy
    return result;  // Another copy
}

// Good: Use string_view when possible
std::string process_text(std::string_view text) {
    // Process without copying
    std::string result = clean_and_tokenize(text);
    return result;
}

// For temporary strings, use std::string&&
void add_document(std::string&& content,
                  std::unordered_map<std::string, std::string>&& metadata = {}) {
    Document doc;
    doc.content = std::move(content);  // Move instead of copy
    doc.metadata = std::move(metadata);
    add_document_internal(std::move(doc));
}
```

---

## CPU Optimization

### 1. SIMD Vectorization

LangChain++ automatically uses SIMD for vector operations:

```cpp
// Automatic SIMD usage
std::vector<double> vec1 = {1.0, 2.0, 3.0, /* ... */};
std::vector<double> vec2 = {4.0, 5.0, 6.0, /* ... */};

// This will use SIMD when available
double similarity = cosine_similarity(vec1, vec2);
```

#### Manual SIMD for Custom Operations

```cpp
#include <immintrin.h>  // For AVX intrinsics

// Manual SIMD optimization for custom operations
double dot_product_avx(const double* a, const double* b, size_t size) {
    __m256d sum = _mm256_setzero_pd();

    for (size_t i = 0; i < size; i += 4) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        sum = _mm256_add_pd(sum, _mm256_mul_pd(va, vb));
    }

    // Horizontal sum
    double result[4];
    _mm256_store_pd(result, sum);
    return result[0] + result[1] + result[2] + result[3];
}
```

### 2. Compiler Optimizations

#### Use Compiler Flags

```cmake
# In CMakeLists.txt
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(langchain_cpp PRIVATE
        -O3                    # Maximum optimization
        -march=native          # Optimize for this CPU
        -flto                  # Link-time optimization
        -DNDEBUG              # Remove debug code
        -mvector              # Enable auto-vectorization
    )
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
```

#### Profile-Guided Optimization (PGO)

```bash
# Step 1: Build with profiling
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PGO=ON ..
make -j$(nproc)

# Step 2: Run representative workload
./your_application benchmark_workload

# Step 3: Rebuild with profile data
make -j$(nproc)

# Step 4: Final optimized build
make install
```

### 3. Algorithm Optimization

#### Use Efficient Data Structures

```cpp
// Bad: O(n) search in vector
bool find_document_bad(const std::vector<Document>& docs, const std::string& id) {
    for (const auto& doc : docs) {
        if (doc.id == id) return true;
    }
    return false;
}

// Good: O(1) lookup in unordered_map
bool find_document_good(const std::unordered_map<std::string, Document>& docs,
                        const std::string& id) {
    return docs.find(id) != docs.end();
}

// Even better: Pre-sorted vector with binary search for small datasets
bool find_document_binary(const std::vector<Document>& docs,
                          const std::string& id) {
    auto it = std::lower_bound(docs.begin(), docs.end(), id,
        [](const Document& doc, const std::string& id) {
            return doc.id < id;
        });
    return it != docs.end() && it->id == id;
}
```

---

## I/O Optimization

### 1. File I/O Optimization

```cpp
// Bad: Many small reads
std::vector<Document> load_documents_slow(const std::vector<std::string>& paths) {
    std::vector<Document> docs;
    for (const auto& path : paths) {
        std::ifstream file(path);
        Document doc;
        file >> doc;  // Small read
        docs.push_back(doc);
    }
    return docs;
}

// Good: Buffered I/O and bulk operations
std::vector<Document> load_documents_fast(const std::vector<std::string>& paths) {
    std::vector<Document> docs;
    docs.reserve(paths.size());

    // Use memory-mapped files for large files
    for (const auto& path : paths) {
        std::ifstream file(path, std::ios::binary);
        file.rdbuf()->pubsetbuf(nullptr, 0);  // Unbuffered for large files

        // Read entire file at once
        std::string content((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());

        Document doc = parse_document(content);
        docs.push_back(std::move(doc));
    }

    return docs;
}
```

### 2. Database I/O Optimization

```cpp
// Bad: N+1 query problem
std::vector<Document> get_documents_with_metadata_bad(Database& db,
                                                     const std::vector<std::string>& ids) {
    std::vector<Document> docs;
    for (const auto& id : ids) {
        auto doc = db.get_document(id);        // One query per document
        auto metadata = db.get_metadata(id);  // Another query per document
        doc.metadata = metadata;
        docs.push_back(doc);
    }
    return docs;
}

// Good: Bulk operations
std::vector<Document> get_documents_with_metadata_good(Database& db,
                                                      const std::vector<std::string>& ids) {
    // Single bulk query
    auto docs = db.get_documents_bulk(ids);
    auto metadata_map = db.get_metadata_bulk(ids);

    // Combine results in memory
    for (auto& doc : docs) {
        if (metadata_map.find(doc.id) != metadata_map.end()) {
            doc.metadata = metadata_map[doc.id];
        }
    }

    return docs;
}
```

### 3. Asynchronous I/O

```cpp
#include <future>
#include <async>

class AsyncIndexer {
public:
    std::future<void> index_documents_async(const std::vector<Document>& docs) {
        return std::async(std::launch::async, [this, docs]() {
            return index_documents(docs);
        });
    }

    std::future<std::vector<Document>> search_async(const std::string& query) {
        return std::async(std::launch::async, [this, query]() {
            return search(query);
        });
    }
};
```

---

## Concurrency Optimization

### 1. Thread Pool Optimization

```cpp
// Bad: Creating threads for each task
void process_documents_bad(const std::vector<Document>& docs) {
    std::vector<std::thread> threads;
    for (const auto& doc : docs) {
        threads.emplace_back([doc]() {
            process_document(doc);
        });
    }
    for (auto& t : threads) {
        t.join();
    }
}

// Good: Use thread pool
void process_documents_good(const std::vector<Document>& docs) {
    ThreadPool pool(std::thread::hardware_concurrency());

    std::vector<std::future<void>> futures;
    for (const auto& doc : docs) {
        futures.push_back(pool.submit([doc]() {
            process_document(doc);
        }));
    }

    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
}
```

### 2. Lock-Free Algorithms

```cpp
// Bad: Lock contention on shared counter
class CounterBad {
private:
    std::mutex mutex_;
    uint64_t count_ = 0;

public:
    void increment() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++count_;
    }
};

// Good: Atomic operations
class CounterGood {
private:
    std::atomic<uint64_t> count_{0};

public:
    void increment() {
        count_.fetch_add(1, std::memory_order_relaxed);
    }

    uint64_t get() const {
        return count_.load(std::memory_order_acquire);
    }
};
```

### 3. Work-Stealing Queue

```cpp
template<typename T>
class WorkStealingQueue {
private:
    std::deque<T> queue_;
    mutable std::mutex mutex_;

public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_front(std::move(item));
    }

    bool pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;

        item = std::move(queue_.back());
        queue_.pop_back();
        return true;
    }

    bool steal(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;

        item = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }
};
```

---

## Caching Strategies

### 1. Multi-Level Caching

```cpp
class MultiLevelCache {
private:
    // L1: In-memory cache (fastest)
    std::unordered_map<std::string, std::vector<Document>> l1_cache_;
    mutable std::shared_mutex l1_mutex_;

    // L2: Compressed cache (slower, more capacity)
    std::unordered_map<std::string, std::vector<uint8_t>> l2_cache_;
    mutable std::shared_mutex l2_mutex_;

    // L3: Disk cache (slowest, most capacity)
    std::unique_ptr<DiskCache> disk_cache_;

public:
    std::vector<Document> get(const std::string& key) {
        // Try L1 cache
        {
            std::shared_lock lock(l1_mutex_);
            auto it = l1_cache_.find(key);
            if (it != l1_cache_.end()) {
                return it->second;
            }
        }

        // Try L2 cache
        {
            std::shared_lock lock(l2_mutex_);
            auto it = l2_cache_.find(key);
            if (it != l2_cache_.end()) {
                auto decompressed = decompress(it->second);

                // Promote to L1
                {
                    std::unique_lock lock(l1_mutex_);
                    l1_cache_[key] = decompressed;
                }

                return decompressed;
            }
        }

        // Try L3 cache
        if (disk_cache_) {
            auto data = disk_cache_->get(key);
            if (!data.empty()) {
                auto decompressed = decompress(data);

                // Promote to higher levels
                {
                    std::unique_lock lock(l2_mutex_);
                    l2_cache_[key] = data;
                }
                {
                    std::unique_lock lock(l1_mutex_);
                    l1_cache_[key] = decompressed;
                }

                return decompressed;
            }
        }

        return {};  // Cache miss
    }
};
```

### 2. Cache Eviction Policies

```cpp
template<typename K, typename V>
class LRUCache {
private:
    struct Node {
        K key;
        V value;
        std::chrono::steady_clock::time_point access_time;
    };

    std::list<Node> cache_list_;
    std::unordered_map<K, typename std::list<Node>::iterator> cache_map_;
    size_t capacity_;

    void evict() {
        if (cache_list_.size() >= capacity_) {
            auto lru = cache_list_.back();
            cache_map_.erase(lru.key);
            cache_list_.pop_back();
        }
    }

public:
    LRUCache(size_t capacity) : capacity_(capacity) {}

    void put(const K& key, const V& value) {
        evict();

        Node node{key, value, std::chrono::steady_clock::now()};
        cache_list_.push_front(node);
        cache_map_[key] = cache_list_.begin();
    }

    std::optional<V> get(const K& key) {
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) {
            return std::nullopt;
        }

        // Move to front (most recently used)
        auto node = *it->second;
        cache_list_.erase(it->second);
        cache_list_.push_front(node);
        cache_map_[key] = cache_list_.begin();

        return node.value;
    }
};
```

### 3. Intelligent Cache Warming

```cpp
class CacheWarmer {
public:
    void warm_cache(Retriever& retriever, const std::vector<std::string>& common_queries) {
        ThreadPool pool(std::thread::hardware_concurrency());

        std::vector<std::future<void>> futures;
        for (const auto& query : common_queries) {
            futures.push_back(pool.submit([&retriever, query]() {
                // Pre-load cache with common query results
                retriever.retrieve(query);
            }));
        }

        // Wait for all warm-up queries to complete
        for (auto& future : futures) {
            future.wait();
        }
    }

    void warm_based_on_usage_stats(Retriever& retriever,
                                 const UsageStats& stats) {
        // Identify most frequently accessed content
        auto popular_queries = stats.get_top_queries(100);
        warm_cache(retriever, popular_queries);
    }
};
```

---

## Database Optimization

### 1. Connection Pooling

```cpp
class ConnectionPool {
private:
    std::queue<std::unique_ptr<DatabaseConnection>> available_;
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t max_size_;
    std::string connection_string_;

public:
    ConnectionPool(const std::string& connection_string, size_t max_size)
        : connection_string_(connection_string), max_size_(max_size) {
        // Pre-allocate connections
        for (size_t i = 0; i < max_size; ++i) {
            available_.push(std::make_unique<DatabaseConnection>(connection_string));
        }
    }

    std::unique_ptr<DatabaseConnection> acquire() {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait for available connection
        cv_.wait(lock, [this]() { return !available_.empty(); });

        auto conn = std::move(available_.front());
        available_.pop();
        return conn;
    }

    void release(std::unique_ptr<DatabaseConnection> conn) {
        std::lock_guard<std::mutex> lock(mutex_);
        available_.push(std::move(conn));
        cv_.notify_one();
    }
};
```

### 2. Batch Operations

```cpp
class OptimizedDatabase {
private:
    ConnectionPool pool_;

public:
    void insert_documents_batch(const std::vector<Document>& docs) {
        auto conn = pool_.acquire();

        try {
            // Begin transaction
            conn->begin_transaction();

            // Prepare statement once
            auto stmt = conn->prepare(
                "INSERT INTO documents (id, content, metadata) VALUES (?, ?, ?)"
            );

            // Execute in batch
            for (const auto& doc : docs) {
                stmt.bind(1, doc.id);
                stmt.bind(2, doc.content);
                stmt.bind(3, serialize_metadata(doc.metadata));
                stmt.execute();
            }

            // Commit transaction
            conn->commit_transaction();

        } catch (const std::exception& e) {
            conn->rollback_transaction();
            throw;
        }

        pool_.release(std::move(conn));
    }
};
```

---

## Network Optimization

### 1. HTTP Connection Reuse

```cpp
class OptimizedHTTPClient {
private:
    std::unique_ptr<CURL*> curl_handle_;
    std::string base_url_;

public:
    OptimizedHTTPClient(const std::string& base_url) : base_url_(base_url) {
        curl_handle_ = std::make_unique<CURL*>();
        *curl_handle_ = curl_easy_init();

        // Configure for reuse
        curl_easy_setopt(*curl_handle_, CURLOPT_TCP_KEEPALIVE, 1L);
        curl_easy_setopt(*curl_handle_, CURLOPT_TCP_KEEPIDLE, 60L);
        curl_easy_setopt(*curl_handle_, CURLOPT_FORBID_REUSE, 0L);
    }

    ~OptimizedHTTPClient() {
        if (*curl_handle_) {
            curl_easy_cleanup(*curl_handle_);
        }
    }

    std::string post(const std::string& endpoint, const std::string& data) {
        std::string url = base_url_ + endpoint;

        curl_easy_setopt(*curl_handle_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(*curl_handle_, CURLOPT_POST, 1L);
        curl_easy_setopt(*curl_handle_, CURLOPT_POSTFIELDS, data.c_str());

        std::string response;
        curl_easy_setopt(*curl_handle_, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(*curl_handle_, CURLOPT_WRITEDATA, &response);

        CURLcode res = curl_easy_perform(*curl_handle_);
        if (res != CURLE_OK) {
            throw std::runtime_error("HTTP request failed");
        }

        return response;
    }

private:
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        ((std::string*)userp)->append((char*)contents, size * nmemb);
        return size * nmemb;
    }
};
```

### 2. Request Batching

```cpp
class BatchProcessor {
public:
    struct Request {
        std::string endpoint;
        std::string data;
        std::promise<std::string> promise;
    };

    void add_request(Request&& request) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        request_queue_.push(std::move(request));

        if (request_queue_.size() >= batch_size_) {
            process_batch();
        }
    }

private:
    void process_batch() {
        std::vector<Request> batch;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);

            while (!request_queue_.empty() && batch.size() < batch_size_) {
                batch.push_back(std::move(request_queue_.front()));
                request_queue_.pop();
            }
        }

        if (batch.empty()) return;

        // Process batch in separate thread
        std::thread([this, batch = std::move(batch)]() mutable {
            process_batch_async(std::move(batch));
        }).detach();
    }

    void process_batch_async(std::vector<Request> batch) {
        // Combine requests into single HTTP call
        std::string combined_request = combine_requests(batch);

        auto response = http_client_->post("/batch", combined_request);
        auto responses = parse_batch_response(response);

        // Resolve promises
        for (size_t i = 0; i < batch.size() && i < responses.size(); ++i) {
            batch[i].promise.set_value(responses[i]);
        }
    }
};
```

---

## Benchmarking

### 1. Performance Benchmarks

```cpp
class PerformanceBenchmark {
private:
    MetricsCollector metrics_;

public:
    void benchmark_retrieval() {
        InvertedIndexRetriever retriever;

        // Setup test data
        auto documents = generate_test_documents(10000);
        retriever.add_documents(documents);

        auto queries = generate_test_queries(1000);

        // Benchmark retrieval performance
        {
            auto timer = metrics_.create_timer("retrieval_benchmark");

            for (const auto& query : queries) {
                auto results = retriever.retrieve(query);

                // Record metrics
                metrics_.increment_counter("queries_processed");
                metrics_.record_histogram("result_count", results.size());
            }
        }

        // Print results
        auto stats = metrics_.get_histogram_stats("retrieval_benchmark");
        std::cout << "Average query time: " << stats.mean << "ms" << std::endl;
        std::cout << "Queries per second: " << 1000.0 / stats.mean << std::endl;
    }

    void benchmark_memory_usage() {
        MemoryTracker tracker;

        {
            auto profile = tracker.start_profile("large_index");

            InvertedIndexRetriever retriever;
            auto documents = generate_large_dataset(100000);
            retriever.add_documents(documents);

        } // Profile automatically records

        auto stats = tracker.get_memory_stats("large_index");
        std::cout << "Peak memory usage: " << stats.peak_memory << " MB" << std::endl;
        std::cout << "Average memory usage: " << stats.average_memory << " MB" << std::endl;
    }
};
```

### 2. Stress Testing

```cpp
class StressTest {
public:
    void stress_concurrent_access() {
        InvertedIndexRetriever retriever;
        auto documents = generate_test_documents(1000);
        retriever.add_documents(documents);

        const int num_threads = 10;
        const int queries_per_thread = 1000;

        std::vector<std::thread> threads;
        std::atomic<int> successful_queries{0};

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&retriever, &successful_queries, queries_per_thread]() {
                for (int j = 0; j < queries_per_thread; ++j) {
                    try {
                        std::string query = "test query " + std::to_string(j);
                        auto results = retriever.retrieve(query);
                        successful_queries.fetch_add(1);
                    } catch (const std::exception& e) {
                        // Log error but continue
                        std::cerr << "Query failed: " << e.what() << std::endl;
                    }
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        int expected_queries = num_threads * queries_per_thread;
        std::cout << "Successful queries: " << successful_queries.load() << "/" << expected_queries << std::endl;
        std::cout << "Success rate: " << (100.0 * successful_queries.load() / expected_queries) << "%" << std::endl;
    }
};
```

---

## Performance Monitoring

### 1. Real-time Monitoring

```cpp
class PerformanceMonitor {
private:
    MetricsCollector metrics_;
    std::thread monitoring_thread_;
    std::atomic<bool> should_stop_{false};

public:
    void start_monitoring() {
        monitoring_thread_ = std::thread([this]() {
            while (!should_stop_.load()) {
                // Collect metrics every second
                collect_metrics();

                // Check for performance issues
                check_performance_issues();

                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });
    }

    void stop_monitoring() {
        should_stop_ = true;
        if (monitoring_thread_.joinable()) {
            monitoring_thread_.join();
        }
    }

private:
    void collect_metrics() {
        // Memory usage
        size_t memory_usage = get_current_memory_usage();
        metrics_.set_gauge("memory_usage_mb", memory_usage / 1024.0 / 1024.0);

        // CPU usage
        double cpu_usage = get_cpu_usage();
        metrics_.set_gauge("cpu_usage_percent", cpu_usage);

        // Disk I/O
        auto disk_stats = get_disk_io_stats();
        metrics_.record_histogram("disk_read_mb", disk_stats.read_mb);
        metrics_.record_histogram("disk_write_mb", disk_stats.write_mb);
    }

    void check_performance_issues() {
        auto memory_gauge = metrics_.get_gauge("memory_usage_mb");
        auto cpu_gauge = metrics_.get_gauge("cpu_usage_percent");

        if (memory_gauge > 1000.0) {  // > 1GB
            log_warning("High memory usage: " + std::to_string(memory_gauge) + " MB");
        }

        if (cpu_gauge > 80.0) {  // > 80%
            log_warning("High CPU usage: " + std::to_string(cpu_gauge) + "%");
        }
    }
};
```

### 2. Performance Alerts

```cpp
class PerformanceAlertManager {
private:
    struct AlertThreshold {
        std::string metric_name;
        double warning_threshold;
        double critical_threshold;
        std::chrono::seconds duration;
    };

    std::vector<AlertThreshold> thresholds_;
    MetricsCollector* metrics_;

public:
    void add_threshold(const std::string& metric_name,
                       double warning_threshold,
                       double critical_threshold,
                       std::chrono::seconds duration) {
        thresholds_.push_back({metric_name, warning_threshold, critical_threshold, duration});
    }

    void check_alerts() {
        for (const auto& threshold : thresholds_) {
            double current_value = metrics_->get_gauge(threshold.metric_name);

            if (current_value >= threshold.critical_threshold) {
                send_alert(AlertLevel::CRITICAL, threshold.metric_name, current_value);
            } else if (current_value >= threshold.warning_threshold) {
                send_alert(AlertLevel::WARNING, threshold.metric_name, current_value);
            }
        }
    }

private:
    void send_alert(AlertLevel level, const std::string& metric, double value) {
        std::string message = format_alert_message(level, metric, value);

        // Send to monitoring system
        MonitoringSystem::instance().send_alert(message);

        // Log the alert
        if (level == AlertLevel::CRITICAL) {
            LOG_ERROR(message);
        } else {
            LOG_WARN(message);
        }
    }
};
```

---

## Performance Best Practices Summary

### 1. Memory
- Use memory pools for frequent allocations
- Pre-allocate containers when size is known
- Use move semantics to avoid copies
- Optimize data structure layout for cache locality

### 2. CPU
- Enable compiler optimizations (-O3, -march=native)
- Use SIMD vectorization for numerical operations
- Choose appropriate algorithms and data structures
- Profile to identify bottlenecks

### 3. I/O
- Use buffered I/O for file operations
- Batch database operations
- Implement connection pooling
- Use asynchronous I/O when possible

### 4. Concurrency
- Use thread pools instead of creating threads per task
- Prefer atomic operations over locks
- Implement work-stealing for load balancing
- Minimize lock contention

### 5. Caching
- Implement multi-level caching
- Use appropriate eviction policies
- Warm up caches based on usage patterns
- Monitor cache hit rates

### 6. Monitoring
- Continuously monitor performance metrics
- Set up alerts for performance issues
- Regularly run performance benchmarks
- Profile under realistic workloads

---

*This performance optimization guide provides the foundation for building high-performance LangChain++ applications. Always profile your specific use case to identify the most effective optimizations.*