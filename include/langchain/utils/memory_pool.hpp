#pragma once

#include <memory>
#include <vector>
#include <mutex>
#include <atomic>
#include <cstddef>
#include <new>
#include <type_traits>
#include <unordered_map>
#include <thread>

namespace langchain::utils {

/**
 * @brief High-performance memory pool for frequent allocations
 *
 * This memory pool provides fast allocation and deallocation for objects
 * of the same size, reducing the overhead of repeated malloc/free calls.
 */
template<size_t BlockSize = 4096>
class MemoryPool {
private:
    struct Block {
        alignas(std::max(alignof(std::max_align_t), size_t(16))) std::byte data[BlockSize];
        Block* next;
    };

    struct LargeAllocation {
        void* ptr;
        size_t size;
        LargeAllocation* next;
    };

    std::vector<std::unique_ptr<Block>> blocks_;
    std::vector<std::unique_ptr<LargeAllocation>> large_allocations_;
    Block* free_list_ = nullptr;
    LargeAllocation* large_free_list_ = nullptr;
    mutable std::mutex mutex_;
    std::atomic<size_t> allocated_blocks_{0};
    std::atomic<size_t> active_allocations_{0};

public:
    /**
     * @brief Constructor
     * @param initial_blocks Initial number of blocks to pre-allocate
     */
    explicit MemoryPool(size_t initial_blocks = 1) {
        for (size_t i = 0; i < initial_blocks; ++i) {
            allocate_new_block();
        }
    }

    /**
     * @brief Destructor
     */
    ~MemoryPool() {
        std::lock_guard<std::mutex> lock(mutex_);
        blocks_.clear();
    }

    /**
     * @brief Allocate memory from pool
     * @param size Size of memory to allocate
     * @return Pointer to allocated memory
     */
    void* allocate(size_t size) {
        if (size > BlockSize) {
            // Handle large allocations with malloc
            std::lock_guard<std::mutex> lock(mutex_);

            // Try to reuse from free list first
            LargeAllocation* alloc = large_free_list_;
            LargeAllocation* prev = nullptr;
            while (alloc) {
                if (alloc->size >= size) {
                    // Found a suitable allocation
                    if (prev) {
                        prev->next = alloc->next;
                    } else {
                        large_free_list_ = alloc->next;
                    }
                    active_allocations_++;
                    return alloc->ptr;
                }
                prev = alloc;
                alloc = alloc->next;
            }

            // Create new large allocation
            void* ptr = ::operator new(size);
            auto large_alloc = std::make_unique<LargeAllocation>();
            large_alloc->ptr = ptr;
            large_alloc->size = size;
            large_alloc->next = nullptr;

            LargeAllocation* raw_ptr = large_alloc.get();
            large_allocations_.push_back(std::move(large_alloc));
            active_allocations_++;

            return ptr;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (!free_list_) {
            allocate_new_block();
        }

        Block* block = free_list_;
        free_list_ = free_list_->next;
        active_allocations_++;

        return static_cast<void*>(block);
    }

    /**
     * @brief Deallocate memory back to pool
     * @param ptr Pointer to deallocate
     */
    void deallocate(void* ptr) {
        if (!ptr) return;

        std::lock_guard<std::mutex> lock(mutex_);

        // Check if it's a large allocation
        for (auto& large_alloc : large_allocations_) {
            if (large_alloc->ptr == ptr) {
                large_alloc->next = large_free_list_;
                large_free_list_ = large_alloc.get();
                active_allocations_--;
                return;
            }
        }

        // Assume it's a regular block allocation
        Block* block = static_cast<Block*>(ptr);
        block->next = free_list_;
        free_list_ = block;
        active_allocations_--;
    }

    /**
     * @brief Get pool statistics
     * @return Map of statistics
     */
    std::unordered_map<std::string, size_t> get_statistics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {
            {"total_blocks", allocated_blocks_.load()},
            {"active_allocations", active_allocations_.load()},
            {"free_blocks", allocated_blocks_.load() - active_allocations_.load()},
            {"block_size", BlockSize}
        };
    }

    /**
     * @brief Get pool statistics (alias for get_statistics)
     * @return Map of statistics
     */
    std::unordered_map<std::string, size_t> get_stats() const {
        return get_statistics();
    }

    /**
     * @brief Get total memory usage
     * @return Total memory usage in bytes
     */
    size_t get_memory_usage() const {
        return allocated_blocks_.load() * BlockSize;
    }

    /**
     * @brief Clear all free blocks (keeps allocated blocks)
     */
    void clear_free_blocks() {
        std::lock_guard<std::mutex> lock(mutex_);
        // Rebuild free list from scratch
        free_list_ = nullptr;
        // Note: We don't actually deallocate the blocks, just clear the free list
        // This is a simplified implementation
    }

private:
    void allocate_new_block() {
        auto block = std::make_unique<Block>();
        Block* raw_block = block.get();
        raw_block->next = free_list_;
        free_list_ = raw_block;
        blocks_.push_back(std::move(block));
        allocated_blocks_++;
    }
};

/**
 * @brief Thread-local memory pool for even better performance
 */
template<size_t BlockSize>
class ThreadLocalPool {
public:
    static MemoryPool<BlockSize>& get_pool() {
        thread_local static MemoryPool<BlockSize> pool;
        return pool;
    }

    static void* allocate(size_t size) {
        return get_pool().allocate(size);
    }

    static void deallocate(void* ptr) {
        get_pool().deallocate(ptr);
    }
};

/**
 * @brief STL-compatible allocator for memory pool
 */
template<typename T>
class PoolAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    template<typename U>
    struct rebind {
        using other = PoolAllocator<U>;
    };

private:
    MemoryPool<sizeof(T)>* pool_;

public:
    explicit PoolAllocator(MemoryPool<sizeof(T)>* pool = nullptr)
        : pool_(pool ? pool : &ThreadLocalPool<sizeof(T)>::get_pool()) {}

    template<typename U>
    PoolAllocator(const PoolAllocator<U>& other) : pool_(other.get_pool()) {}

    pointer allocate(size_type n) {
        if (n == 0) return nullptr;
        if (n > 1) {
            return static_cast<pointer>(::operator new(n * sizeof(T)));
        }
        return static_cast<pointer>(pool_->allocate(sizeof(T)));
    }

    void deallocate(pointer p, size_type n) {
        if (!p) return;
        if (n > 1) {
            ::operator delete(p);
        } else {
            pool_->deallocate(p);
        }
    }

    template<typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template<typename U>
    void destroy(U* p) {
        p->~U();
    }

    template<typename U>
    bool operator==(const PoolAllocator<U>& other) const {
        return pool_ == other.get_pool();
    }

    template<typename U>
    bool operator!=(const PoolAllocator<U>& other) const {
        return !(*this == other);
    }

    MemoryPool<sizeof(T)>* get_pool() const { return pool_; }
};

/**
 * @brief Smart pointer for pool-allocated objects
 */
template<typename T>
class PoolPtr {
private:
    T* ptr_;
    MemoryPool<sizeof(T)>* pool_;

public:
    /**
     * @brief Constructor
     * @param ptr Raw pointer
     * @param pool Memory pool
     */
    explicit PoolPtr(T* ptr = nullptr, MemoryPool<sizeof(T)>* pool = nullptr)
        : ptr_(ptr), pool_(pool ? pool : &ThreadLocalPool<sizeof(T)>::get_pool()) {}

    /**
     * @brief Destructor
     */
    ~PoolPtr() {
        if (ptr_) {
            ptr_->~T();
            pool_->deallocate(ptr_);
        }
    }

    /**
     * @brief Copy constructor (deleted - unique ownership)
     */
    PoolPtr(const PoolPtr&) = delete;

    /**
     * @brief Copy assignment (deleted)
     */
    PoolPtr& operator=(const PoolPtr&) = delete;

    /**
     * @brief Move constructor
     */
    PoolPtr(PoolPtr&& other) noexcept
        : ptr_(other.ptr_), pool_(other.pool_) {
        other.ptr_ = nullptr;
        other.pool_ = nullptr;
    }

    /**
     * @brief Move assignment
     */
    PoolPtr& operator=(PoolPtr&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            pool_ = other.pool_;
            other.ptr_ = nullptr;
            other.pool_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Dereference operator
     */
    T& operator*() const { return *ptr_; }

    /**
     * @brief Arrow operator
     */
    T* operator->() const { return ptr_; }

    /**
     * @brief Get raw pointer
     */
    T* get() const { return ptr_; }

    /**
     * @brief Check if pointer is valid
     */
    explicit operator bool() const { return ptr_ != nullptr; }

    /**
     * @brief Reset pointer
     */
    void reset() {
        if (ptr_) {
            ptr_->~T();
            pool_->deallocate(ptr_);
            ptr_ = nullptr;
        }
    }

    /**
     * @brief Release ownership
     */
    T* release() {
        T* temp = ptr_;
        ptr_ = nullptr;
        return temp;
    }

    /**
     * @brief Equality comparison with nullptr
     */
    bool operator==(std::nullptr_t) const { return ptr_ == nullptr; }

    /**
     * @brief Inequality comparison with nullptr
     */
    bool operator!=(std::nullptr_t) const { return ptr_ != nullptr; }

    /**
     * @brief Equality comparison with another PoolPtr
     */
    bool operator==(const PoolPtr& other) const { return ptr_ == other.ptr_; }

    /**
     * @brief Inequality comparison with another PoolPtr
     */
    bool operator!=(const PoolPtr& other) const { return ptr_ != other.ptr_; }
};

/**
 * @brief Factory function to create pool-allocated objects
 */
template<typename T, typename... Args>
PoolPtr<T> make_pool_unique(Args&&... args) {
    auto& pool = ThreadLocalPool<sizeof(T)>::get_pool();
    void* memory = pool.allocate(sizeof(T));
    T* ptr = new (memory) T(std::forward<Args>(args)...);
    return PoolPtr<T>(ptr, &pool);
}

/**
 * @brief Factory function with custom pool
 */
template<typename T, size_t N, typename... Args>
PoolPtr<T> make_pool_unique(MemoryPool<N>& pool, Args&&... args) {
    static_assert(N >= sizeof(T), "Pool block size too small for type T");
    void* memory = pool.allocate(sizeof(T));
    T* ptr = new (memory) T(std::forward<Args>(args)...);
    return PoolPtr<T>(ptr, &pool);
}

// LRUCache implementation will be added in a future version
// For now, we focus on the core MemoryPool functionality

} // namespace langchain::utils