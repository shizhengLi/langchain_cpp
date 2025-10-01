#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace langchain::utils {

/**
 * @brief Work-stealing thread pool for high-performance task execution
 *
 * This thread pool implements work stealing to balance load across threads
 * and maximize CPU utilization for concurrent operations.
 */
class ThreadPool {
private:
    std::vector<std::thread> workers_;
    std::vector<std::queue<std::function<void()>>> task_queues_;
    std::vector<std::unique_ptr<std::mutex>> queue_mutexes_;
    std::vector<std::unique_ptr<std::condition_variable>> queue_conditions_;
    std::atomic<bool> stop_flag_{false};
    std::atomic<size_t> next_worker_{0};
    std::atomic<size_t> active_tasks_{0};
    std::atomic<size_t> completed_tasks_{0};

    // Performance metrics
    std::atomic<uint64_t> total_wait_time_us_{0};
    std::atomic<uint64_t> total_execution_time_us_{0};

public:
    /**
     * @brief Constructor
     * @param num_threads Number of worker threads (0 = auto-detect)
     */
    explicit ThreadPool(size_t num_threads = 0) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) {
                num_threads = 4;  // Fallback
            }
        }

        workers_.reserve(num_threads);
        task_queues_.resize(num_threads);

        // Initialize mutex and condition variable pointers
        for (size_t i = 0; i < num_threads; ++i) {
            queue_mutexes_.push_back(std::make_unique<std::mutex>());
            queue_conditions_.push_back(std::make_unique<std::condition_variable>());
        }

        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back(&ThreadPool::worker_loop, this, i);
        }
    }

    /**
     * @brief Destructor - waits for all tasks to complete
     */
    ~ThreadPool() {
        shutdown();
    }

    /**
     * @brief Submit a task to the thread pool
     * @tparam F Function type
     * @tparam Args Argument types
     * @param f Function to execute
     * @param args Arguments to pass to the function
     * @return Future containing the result
     */
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using return_type = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> result = task->get_future();

        // Queue to next worker in round-robin fashion
        size_t worker_id = next_worker_.fetch_add(1) % workers_.size();

        {
            std::unique_lock<std::mutex> lock(*queue_mutexes_[worker_id]);
            task_queues_[worker_id].emplace([task]() { (*task)(); });
        }
        queue_conditions_[worker_id]->notify_one();

        active_tasks_++;

        return result;
    }

    /**
     * @brief Submit a task to a specific worker
     * @tparam F Function type
     * @tparam Args Argument types
     * @param worker_id Target worker ID
     * @param f Function to execute
     * @param args Arguments to pass to the function
     * @return Future containing the result
     */
    template<typename F, typename... Args>
    auto submit_to_worker(size_t worker_id, F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        if (worker_id >= workers_.size()) {
            throw std::out_of_range("Worker ID out of range");
        }

        using return_type = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> result = task->get_future();

        {
            std::unique_lock<std::mutex> lock(*queue_mutexes_[worker_id]);
            task_queues_[worker_id].emplace([task]() { (*task)(); });
        }
        queue_conditions_[worker_id]->notify_one();

        active_tasks_++;

        return result;
    }

    /**
     * @brief Wait for all submitted tasks to complete
     */
    void wait_for_all() {
        while (active_tasks_.load() > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    /**
     * @brief Get number of worker threads
     * @return Number of threads
     */
    size_t size() const {
        return workers_.size();
    }

    /**
     * @brief Get number of active tasks
     * @return Number of active tasks
     */
    size_t get_active_tasks() const {
        return active_tasks_.load();
    }

    /**
     * @brief Get number of completed tasks
     * @return Number of completed tasks
     */
    size_t get_completed_tasks() const {
        return completed_tasks_.load();
    }

    /**
     * @brief Get performance metrics
     * @return Map of performance metrics
     */
    std::unordered_map<std::string, double> get_performance_metrics() const {
        uint64_t total_wait = total_wait_time_us_.load();
        uint64_t total_exec = total_execution_time_us_.load();
        uint64_t completed = completed_tasks_.load();

        std::unordered_map<std::string, double> metrics;
        metrics["average_wait_time_us"] = completed > 0 ? static_cast<double>(total_wait) / completed : 0.0;
        metrics["average_execution_time_us"] = completed > 0 ? static_cast<double>(total_exec) / completed : 0.0;
        metrics["total_tasks_completed"] = static_cast<double>(completed);
        metrics["active_tasks"] = static_cast<double>(active_tasks_.load());
        metrics["worker_threads"] = static_cast<double>(workers_.size());

        return metrics;
    }

    /**
     * @brief Shutdown the thread pool
     */
    void shutdown() {
        stop_flag_.store(true);

        // Wake up all workers
        for (auto& condition : queue_conditions_) {
            condition->notify_all();
        }

        // Wait for all workers to finish
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        workers_.clear();
    }

private:
    void worker_loop(size_t worker_id) {
        while (!stop_flag_.load()) {
            std::function<void()> task;

            // Try to get task from own queue
            {
                std::unique_lock<std::mutex> lock(*queue_mutexes_[worker_id]);
                auto start_time = std::chrono::high_resolution_clock::now();

                queue_conditions_[worker_id]->wait(lock, [this, worker_id]() {
                    return !task_queues_[worker_id].empty() || stop_flag_.load();
                });

                if (stop_flag_.load() && task_queues_[worker_id].empty()) {
                    break;
                }

                if (!task_queues_[worker_id].empty()) {
                    task = std::move(task_queues_[worker_id].front());
                    task_queues_[worker_id].pop();

                    auto wait_end = std::chrono::high_resolution_clock::now();
                    auto wait_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        wait_end - start_time);
                    total_wait_time_us_ += wait_duration.count();
                }
            }

            if (task) {
                auto exec_start = std::chrono::high_resolution_clock::now();
                task();
                auto exec_end = std::chrono::high_resolution_clock::now();

                auto exec_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    exec_end - exec_start);
                total_execution_time_us_ += exec_duration.count();

                active_tasks_--;
                completed_tasks_++;
            } else {
                // Try to steal work from other queues
                if (!try_steal_work(worker_id)) {
                    // No work available, sleep briefly
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
            }
        }
    }

    bool try_steal_work(size_t thief_id) {
        size_t num_workers = workers_.size();

        for (size_t i = 1; i < num_workers; ++i) {
            size_t victim_id = (thief_id + i) % num_workers;

            std::unique_lock<std::mutex> lock(*queue_mutexes_[victim_id]);
            if (!task_queues_[victim_id].empty()) {
                auto task = std::move(task_queues_[victim_id].front());
                task_queues_[victim_id].pop();
                lock.unlock();

                // Execute stolen task
                auto exec_start = std::chrono::high_resolution_clock::now();
                task();
                auto exec_end = std::chrono::high_resolution_clock::now();

                auto exec_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    exec_end - exec_start);
                total_execution_time_us_ += exec_duration.count();

                active_tasks_--;
                completed_tasks_++;

                return true;
            }
        }

        return false;
    }
};

/**
 * @brief Lock-free queue for task submission
 */
template<typename T>
class LockFreeQueue {
private:
    struct Node {
        std::atomic<T*> data{nullptr};
        std::atomic<Node*> next{nullptr};
    };

    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;

public:
    LockFreeQueue() {
        Node* dummy = new Node;
        head_.store(dummy);
        tail_.store(dummy);
    }

    ~LockFreeQueue() {
        while (Node* head = head_.load()) {
            head_.store(head->next);
            delete head;
        }
    }

    /**
     * @brief Enqueue an item
     * @param item Item to enqueue
     */
    void enqueue(T item) {
        Node* new_node = new Node;
        T* item_ptr = new T(std::move(item));
        new_node->data.store(item_ptr);

        Node* prev_tail = tail_.exchange(new_node);
        prev_tail->next.store(new_node);
    }

    /**
     * @brief Dequeue an item
     * @return Optional dequeued item
     */
    std::optional<T> dequeue() {
        Node* head = head_.load();
        Node* next = head->next.load();

        if (next == nullptr) {
            return std::nullopt;  // Queue is empty
        }

        T* data = next->data.exchange(nullptr);
        if (data) {
            head_.store(next);
            delete head;
            T result = std::move(*data);
            delete data;
            return result;
        }

        return std::nullopt;
    }

    /**
     * @brief Check if queue is empty
     * @return True if empty
     */
    bool empty() const {
        Node* head = head_.load();
        return head->next.load() == nullptr;
    }
};

/**
 * @brief RAII timer for performance measurement
 */
class Timer {
private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::function<void(const std::string&, double)> callback_;

public:
    /**
     * @brief Constructor
     * @param name Timer name
     * @param callback Callback function called on destruction
     */
    explicit Timer(const std::string& name,
                  std::function<void(const std::string&, double)> callback = nullptr)
        : name_(name), start_time_(std::chrono::high_resolution_clock::now()),
          callback_(callback) {}

    /**
     * @brief Destructor - reports elapsed time
     */
    ~Timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_);
        double elapsed_ms = duration.count() / 1000.0;

        if (callback_) {
            callback_(name_, elapsed_ms);
        }
    }

    /**
     * @brief Get elapsed time so far
     * @return Elapsed time in milliseconds
     */
    double elapsed_ms() const {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            current_time - start_time_);
        return duration.count() / 1000.0;
    }
};

/**
 * @brief Global thread pool instance
 */
class GlobalThreadPool {
private:
    static std::unique_ptr<ThreadPool> instance_;
    static std::mutex mutex_;

public:
    /**
     * @brief Get or create global thread pool
     * @return Reference to global thread pool
     */
    static ThreadPool& get_instance() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!instance_) {
            instance_ = std::make_unique<ThreadPool>();
        }
        return *instance_;
    }

    /**
     * @brief Submit task to global thread pool
     */
    template<typename F, typename... Args>
    static auto submit(F&& f, Args&&... args) {
        return get_instance().submit(std::forward<F>(f), std::forward<Args>(args)...);
    }

    /**
     * @brief Wait for all tasks in global pool
     */
    static void wait_for_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (instance_) {
            instance_->wait_for_all();
        }
    }

    /**
     * @brief Shutdown global thread pool
     */
    static void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        instance_.reset();
    }
};

// Static member definitions
std::unique_ptr<ThreadPool> GlobalThreadPool::instance_;
std::mutex GlobalThreadPool::mutex_;

} // namespace langchain::utils