#include <catch2/catch_all.hpp>
#include "langchain/utils/memory_pool.hpp"
#include <vector>
#include <thread>
#include <chrono>

using namespace langchain::utils;

// Helper functions for PoolPtr comparison with nullptr
template<typename T>
bool operator==(std::nullptr_t, const PoolPtr<T>& ptr) {
    return ptr == nullptr;
}

template<typename T>
bool operator!=(std::nullptr_t, const PoolPtr<T>& ptr) {
    return ptr != nullptr;
}

TEST_CASE("MemoryPool - Basic Operations", "[utils][memory_pool]") {
    SECTION("Default construction") {
        MemoryPool<1024> pool;
        auto stats = pool.get_statistics();
        REQUIRE(stats.at("total_blocks") == 1);  // default construction creates 1 block
        REQUIRE(stats.at("active_allocations") == 0); // active allocations
        REQUIRE(stats.at("free_blocks") == 1); // 1 free block
    }

    SECTION("Construction with initial blocks") {
        MemoryPool<512> pool(5);
        auto stats = pool.get_statistics();
        REQUIRE(stats.at("total_blocks") == 5);   // allocated_blocks
        REQUIRE(stats.at("active_allocations") == 0); // none used yet
        REQUIRE(stats.at("free_blocks") == 5);  // free_blocks
    }

    SECTION("Allocate and deallocate") {
        MemoryPool<256> pool(2);

        void* ptr1 = pool.allocate(100);
        REQUIRE(ptr1 != nullptr);

        auto stats_after_alloc = pool.get_statistics();
        REQUIRE(stats_after_alloc.at("total_blocks") == 2);
        REQUIRE(stats_after_alloc.at("active_allocations") == 1);  // One block used
        REQUIRE(stats_after_alloc.at("free_blocks") == 1);  // One free block

        pool.deallocate(ptr1);
        auto stats_after_dealloc = pool.get_statistics();
        REQUIRE(stats_after_dealloc.at("total_blocks") == 2);
        REQUIRE(stats_after_dealloc.at("active_allocations") == 0);  // No active allocations
        REQUIRE(stats_after_dealloc.at("free_blocks") == 2);  // All blocks free
    }

    SECTION("Allocate larger than block size") {
        MemoryPool<256> pool;
        // Large allocations should fallback to malloc and not throw
        void* ptr = pool.allocate(300);
        REQUIRE(ptr != nullptr);
        pool.deallocate(ptr);  // Should handle deallocation properly
    }

    SECTION("Multiple allocations") {
        MemoryPool<128> pool(3);

        std::vector<void*> ptrs;
        for (int i = 0; i < 3; ++i) {
            void* ptr = pool.allocate(50);
            REQUIRE(ptr != nullptr);
            ptrs.push_back(ptr);
        }

        auto stats = pool.get_statistics();
        REQUIRE(stats.at("total_blocks") == 3);   // 3 blocks allocated
        REQUIRE(stats.at("active_allocations") == 3);  // All blocks in use
        REQUIRE(stats.at("free_blocks") == 0);  // No free blocks

        // Deallocate all
        for (void* ptr : ptrs) {
            pool.deallocate(ptr);
        }

        stats = pool.get_statistics();
        REQUIRE(stats.at("total_blocks") == 3);
        REQUIRE(stats.at("active_allocations") == 0);  // No active allocations
        REQUIRE(stats.at("free_blocks") == 3);  // All blocks returned to pool
    }

    SECTION("Allocate beyond initial capacity") {
        MemoryPool<64> pool(2);

        void* ptr1 = pool.allocate(32);
        void* ptr2 = pool.allocate(32);
        REQUIRE(ptr1 != nullptr);
        REQUIRE(ptr2 != nullptr);

        // This should trigger allocation of a new block
        void* ptr3 = pool.allocate(32);
        REQUIRE(ptr3 != nullptr);

        auto stats = pool.get_statistics();
        REQUIRE(stats.at("total_blocks") == 3);   // 3 blocks total
        REQUIRE(stats.at("active_allocations") == 3);  // All blocks in use
        REQUIRE(stats.at("free_blocks") == 0);  // No free blocks
    }
}

TEST_CASE("ThreadLocalPool - Thread Safety", "[utils][memory_pool][thread_local]") {
    SECTION("Basic usage") {
        void* ptr1 = ThreadLocalPool<256>::allocate(100);
        void* ptr2 = ThreadLocalPool<256>::allocate(100);

        REQUIRE(ptr1 != nullptr);
        REQUIRE(ptr2 != nullptr);
        REQUIRE(ptr1 != ptr2);

        ThreadLocalPool<256>::deallocate(ptr1);
        ThreadLocalPool<256>::deallocate(ptr2);
    }

    SECTION("Different thread pools") {
        void* ptr1 = nullptr;
        void* ptr2 = nullptr;

        std::thread t1([&ptr1]() {
            ptr1 = ThreadLocalPool<256>::allocate(100);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            ThreadLocalPool<256>::deallocate(ptr1);
        });

        std::thread t2([&ptr2]() {
            ptr2 = ThreadLocalPool<256>::allocate(100);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            ThreadLocalPool<256>::deallocate(ptr2);
        });

        t1.join();
        t2.join();

        // Pointers should be different since they're from different thread-local pools
        REQUIRE(ptr1 != nullptr);
        REQUIRE(ptr2 != nullptr);
        // Note: We can't guarantee ptr1 != ptr2 since they're deallocated,
        // but the test verifies thread safety
    }
}

TEST_CASE("PoolAllocator - STL Integration", "[utils][memory_pool][allocator]") {
    SECTION("Vector with pool allocator") {
        MemoryPool<sizeof(int)> pool;
        PoolAllocator<int> allocator(&pool);

        std::vector<int, PoolAllocator<int>> vec(allocator);

        // Push some elements
        for (int i = 0; i < 100; ++i) {
            vec.push_back(i);
        }

        REQUIRE(vec.size() == 100);
        for (int i = 0; i < 100; ++i) {
            REQUIRE(vec[i] == i);
        }
    }

    SECTION("Multiple vectors sharing pool") {
        MemoryPool<sizeof(int)> pool;
        PoolAllocator<int> allocator(&pool);

        std::vector<int, PoolAllocator<int>> vec1(allocator);
        std::vector<int, PoolAllocator<int>> vec2(allocator);

        vec1.push_back(1);
        vec1.push_back(2);
        vec2.push_back(3);
        vec2.push_back(4);

        REQUIRE(vec1.size() == 2);
        REQUIRE(vec2.size() == 2);
        REQUIRE(vec1[0] == 1);
        REQUIRE(vec2[0] == 3);
    }

    SECTION("Allocator comparison") {
        MemoryPool<sizeof(int)> pool;
        PoolAllocator<int> alloc1(&pool);
        PoolAllocator<int> alloc2(&pool);
        PoolAllocator<int> alloc3;  // Different pool

        REQUIRE(alloc1 == alloc2);
        REQUIRE(alloc1 != alloc3);
    }
}

TEST_CASE("PoolPtr - Smart Pointer with Pool", "[utils][memory_pool][smart_ptr]") {
    SECTION("Basic usage") {
        auto ptr = make_pool_unique<int>(42);

        REQUIRE(ptr != nullptr);
        REQUIRE(*ptr == 42);

        // Test arrow operator
        auto str_ptr = make_pool_unique<std::string>("Hello");
        REQUIRE(str_ptr->size() == 5);
    }

    SECTION("Move semantics") {
        auto ptr1 = make_pool_unique<int>(42);
        PoolPtr<int> ptr2 = std::move(ptr1);

        REQUIRE(ptr1 == nullptr);
        REQUIRE(ptr2 != nullptr);
        REQUIRE(*ptr2 == 42);
    }

    SECTION("Custom pool") {
        MemoryPool<sizeof(std::string)> pool;
        auto ptr = make_pool_unique<std::string>(pool, "Test");

        REQUIRE(ptr != nullptr);
        REQUIRE(*ptr == "Test");
    }

    SECTION("Reset functionality") {
        auto ptr = make_pool_unique<std::string>("Hello");
        REQUIRE(ptr != nullptr);
        REQUIRE(*ptr == "Hello");

        ptr.reset();
        REQUIRE(ptr == nullptr);
    }

    SECTION("Release ownership") {
        auto ptr = make_pool_unique<int>(42);
        int* raw = ptr.get();

        int* released = ptr.release();
        REQUIRE(released == raw);
        REQUIRE(ptr == nullptr);

        // Note: We need to manually deallocate since we released ownership
        // In practice, this should be avoided
        ThreadLocalPool<sizeof(int)>::deallocate(released);
    }

    SECTION("Boolean conversion") {
        auto ptr1 = make_pool_unique<int>(42);
        PoolPtr<int> ptr2;

        REQUIRE(ptr1);
        REQUIRE_FALSE(ptr2);
    }
}

/*
TEST_CASE("LRUCache - Cache with Memory Pool", "[utils][memory_pool][cache]") {
    SECTION("Basic operations") {
        LRUCache<std::string, int> cache(3);

        REQUIRE(cache.size() == 0);
        REQUIRE_FALSE(cache.contains("key1"));

        cache.put("key1", 1);
        REQUIRE(cache.size() == 1);
        REQUIRE(cache.contains("key1"));

        auto value = cache.get("key1");
        REQUIRE(value.has_value());
        REQUIRE(value.value() == 1);

        // Non-existent key
        REQUIRE_FALSE(cache.get("nonexistent").has_value());
    }

    SECTION("LRU eviction") {
        LRUCache<int, std::string> cache(2);

        cache.put(1, "one");
        cache.put(2, "two");
        REQUIRE(cache.size() == 2);

        // Add third item, should evict first
        cache.put(3, "three");
        REQUIRE(cache.size() == 2);
        REQUIRE_FALSE(cache.contains(1));
        REQUIRE(cache.contains(2));
        REQUIRE(cache.contains(3));
    }

    SECTION("Update existing key") {
        LRUCache<std::string, int> cache(2);

        cache.put("key", 1);
        cache.put("key", 2);  // Update same key

        REQUIRE(cache.size() == 1);
        auto value = cache.get("key");
        REQUIRE(value.has_value());
        REQUIRE(value.value() == 2);
    }

    SECTION("Access moves to front") {
        LRUCache<int, std::string> cache(2);

        cache.put(1, "one");
        cache.put(2, "two");

        // Access key 1, moves it to front
        auto value = cache.get(1);
        REQUIRE(value.has_value());

        // Add key 3, should evict key 2 (not key 1)
        cache.put(3, "three");
        REQUIRE(cache.size() == 2);
        REQUIRE(cache.contains(1));   // Should still be there
        REQUIRE_FALSE(cache.contains(2));  // Should be evicted
        REQUIRE(cache.contains(3));
    }

    SECTION("Clear cache") {
        LRUCache<std::string, int> cache(5);

        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);
        REQUIRE(cache.size() == 3);

        cache.clear();
        REQUIRE(cache.size() == 0);
        REQUIRE_FALSE(cache.contains("a"));
        REQUIRE_FALSE(cache.contains("b"));
        REQUIRE_FALSE(cache.contains("c"));
    }

    SECTION("Complex key and value types") {
        LRUCache<std::vector<int>, std::string> cache(2);

        std::vector<int> key1 = {1, 2, 3};
        std::vector<int> key2 = {4, 5, 6};

        cache.put(key1, "first");
        cache.put(key2, "second");

        auto value = cache.get(key1);
        REQUIRE(value.has_value());
        REQUIRE(value.value() == "first");
    }
}
*/

TEST_CASE("MemoryPool - Thread Safety", "[utils][memory_pool][thread_safety]") {
    SECTION("Concurrent allocations and deallocations") {
        MemoryPool<256> pool(10);
        std::vector<std::thread> threads;
        std::atomic<int> successful_operations{0};

        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([&pool, &successful_operations]() {
                for (int i = 0; i < 100; ++i) {
                    try {
                        void* ptr = pool.allocate(100);
                        if (ptr != nullptr) {
                            // Simulate some work
                            std::this_thread::sleep_for(std::chrono::microseconds(1));
                            pool.deallocate(ptr);
                            successful_operations++;
                        }
                    } catch (...) {
                        // Allocation might fail if pool is exhausted
                    }
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        // Should have completed many operations successfully
        REQUIRE(successful_operations.load() > 0);

        // Pool should still be consistent
        auto stats = pool.get_statistics();
        REQUIRE(stats.at("total_blocks") >= stats.at("free_blocks"));  // total >= free
    }

    SECTION("ThreadLocalPool concurrent usage") {
        std::vector<std::thread> threads;
        std::atomic<int> successful_operations{0};

        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([&successful_operations]() {
                for (int i = 0; i < 50; ++i) {
                    void* ptr = ThreadLocalPool<128>::allocate(50);
                    if (ptr != nullptr) {
                        ThreadLocalPool<128>::deallocate(ptr);
                        successful_operations++;
                    }
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        // All operations should succeed since each thread has its own pool
        REQUIRE(successful_operations.load() == 8 * 50);
    }
}