#include "langchain/distributed/distributed_processing.hpp"
#include <random>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <iostream>

namespace langchain {
namespace distributed {

// Task implementation
Task::Task(TaskId id, TaskFunction func, TaskPriority priority)
    : id_(std::move(id)), function_(std::move(func)), priority_(priority) {}

// TaskScheduler implementation
TaskScheduler::TaskScheduler(size_t max_concurrent_tasks)
    : max_concurrent_tasks_(max_concurrent_tasks) {}

TaskScheduler::~TaskScheduler() {
    stop();
}

std::future<std::string> TaskScheduler::submit_task(TaskPtr task) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        all_tasks_[task->id()] = task;
        task_promises_[task->id()] = std::promise<std::string>();
        task_queue_.push(task);
    }
    queue_condition_.notify_one();
    return task_promises_[task->id()].get_future();
}

std::future<std::string> TaskScheduler::submit_task(Task::TaskId id, Task::TaskFunction func,
                                                   TaskPriority priority) {
    auto task = std::make_shared<Task>(std::move(id), std::move(func), priority);
    return submit_task(task);
}

void TaskScheduler::start() {
    if (running_.load()) return;

    running_.store(true);
    worker_threads_.reserve(max_concurrent_tasks_.load());

    for (size_t i = 0; i < max_concurrent_tasks_.load(); ++i) {
        worker_threads_.emplace_back(&TaskScheduler::worker_loop, this);
    }
}

void TaskScheduler::stop() {
    if (!running_.load()) return;

    running_.store(false);
    queue_condition_.notify_all();

    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
}

void TaskScheduler::pause() {
    paused_.store(true);
}

void TaskScheduler::resume() {
    paused_.store(false);
    queue_condition_.notify_all();
}

size_t TaskScheduler::pending_tasks() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return task_queue_.size();
}

size_t TaskScheduler::running_tasks() const {
    return running_tasks_.load();
}

size_t TaskScheduler::completed_tasks() const {
    return completed_tasks_.load();
}

std::vector<Task::TaskId> TaskScheduler::get_pending_task_ids() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    std::vector<Task::TaskId> ids;

    // Copy the queue to access tasks
    auto temp_queue = task_queue_;
    while (!temp_queue.empty()) {
        ids.push_back(temp_queue.top()->id());
        temp_queue.pop();
    }

    return ids;
}

bool TaskScheduler::cancel_task(const Task::TaskId& task_id) {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    auto task_it = all_tasks_.find(task_id);
    if (task_it != all_tasks_.end()) {
        auto task = task_it->second;
        if (task->status() == TaskStatus::PENDING) {
            task->set_status(TaskStatus::CANCELLED);

            // Try to find and remove from queue
            std::vector<TaskPtr> temp_tasks;
            while (!task_queue_.empty()) {
                auto top_task = task_queue_.top();
                task_queue_.pop();
                if (top_task->id() != task_id) {
                    temp_tasks.push_back(top_task);
                }
            }

            // Re-add remaining tasks
            for (const auto& temp_task : temp_tasks) {
                task_queue_.push(temp_task);
            }

            // Set promise exception
            auto promise_it = task_promises_.find(task_id);
            if (promise_it != task_promises_.end()) {
                try {
                    throw std::runtime_error("Task cancelled");
                } catch (...) {
                    promise_it->second.set_exception(std::current_exception());
                }
                task_promises_.erase(promise_it);
            }

            return true;
        }
    }
    return false;
}

TaskScheduler::TaskPtr TaskScheduler::get_task(const Task::TaskId& task_id) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    auto it = all_tasks_.find(task_id);
    return (it != all_tasks_.end()) ? it->second : nullptr;
}

void TaskScheduler::worker_loop() {
    while (running_.load()) {
        if (paused_.load()) {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_condition_.wait(lock, [this] { return !paused_.load() || !running_.load(); });
            continue;
        }

        auto task = get_next_task();
        if (task) {
            running_tasks_.fetch_add(1);

            try {
                auto result = task->execute();
                completed_tasks_.fetch_add(1);

                // Set promise result
                std::lock_guard<std::mutex> lock(queue_mutex_);
                auto promise_it = task_promises_.find(task->id());
                if (promise_it != task_promises_.end()) {
                    promise_it->second.set_value(result);
                    task_promises_.erase(promise_it);
                }
            } catch (...) {
                // Set promise exception
                std::lock_guard<std::mutex> lock(queue_mutex_);
                auto promise_it = task_promises_.find(task->id());
                if (promise_it != task_promises_.end()) {
                    promise_it->second.set_exception(std::current_exception());
                    task_promises_.erase(promise_it);
                }
            }

            running_tasks_.fetch_sub(1);
            finished_condition_.notify_all();
        } else {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_condition_.wait_for(lock, std::chrono::milliseconds(100));
        }
    }
}

TaskScheduler::TaskPtr TaskScheduler::get_next_task() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (task_queue_.empty()) {
        return nullptr;
    }

    auto task = task_queue_.top();
    task_queue_.pop();
    return task;
}

// NodeManager implementation
NodeManager::NodeManager() {
    running_.store(true);
    health_check_thread_ = std::thread(&NodeManager::health_check_loop, this);
}

NodeManager::~NodeManager() {
    running_.store(false);
    if (health_check_thread_.joinable()) {
        health_check_thread_.join();
    }
}

bool NodeManager::register_node(const std::string& node_id, const std::string& address,
                               int port, int max_concurrent_tasks) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);

    if (nodes_.find(node_id) != nodes_.end()) {
        return false; // Node already exists
    }

    auto node = std::make_shared<NodeInfo>(node_id, address, port, max_concurrent_tasks);
    nodes_[node_id] = node;
    return true;
}

bool NodeManager::unregister_node(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    return nodes_.erase(node_id) > 0;
}

NodeManager::NodePtr NodeManager::get_best_node() {
    std::lock_guard<std::mutex> lock(nodes_mutex_);

    NodePtr best_node = nullptr;
    size_t min_load = SIZE_MAX;

    for (const auto& [id, node] : nodes_) {
        if (node->is_available && static_cast<int>(node->current_load.load()) < node->max_concurrent_tasks) {
            if (node->current_load.load() < min_load) {
                min_load = node->current_load.load();
                best_node = node;
            }
        }
    }

    return best_node;
}

std::vector<NodeManager::NodePtr> NodeManager::get_available_nodes() {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    std::vector<NodeManager::NodePtr> available_nodes;

    for (const auto& [id, node] : nodes_) {
        if (node->is_available) {
            available_nodes.push_back(node);
        }
    }

    return available_nodes;
}

NodeManager::NodePtr NodeManager::get_node(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    auto it = nodes_.find(node_id);
    return (it != nodes_.end()) ? it->second : nullptr;
}

std::vector<std::string> NodeManager::get_all_node_ids() const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    std::vector<std::string> ids;

    for (const auto& [id, node] : nodes_) {
        ids.push_back(id);
    }

    return ids;
}

size_t NodeManager::available_nodes_count() const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    size_t count = 0;

    for (const auto& [id, node] : nodes_) {
        if (node->is_available) {
            count++;
        }
    }

    return count;
}

void NodeManager::update_node_load(const std::string& node_id, size_t load) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    auto it = nodes_.find(node_id);
    if (it != nodes_.end()) {
        it->second->current_load.store(load);
    }
}

void NodeManager::health_check_loop() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));

        std::lock_guard<std::mutex> lock(nodes_mutex_);
        // Simple health check simulation
        for (auto& [id, node] : nodes_) {
            // Simulate health check - in real implementation would ping the node
            node->is_available = true; // Assume all nodes are healthy for simulation
        }
    }
}

// DistributedExecutor implementation
DistributedExecutor::DistributedExecutor(NodeManager& node_manager)
    : node_manager_(node_manager) {}

// MockDistributedExecutor implementation
MockDistributedExecutor::MockDistributedExecutor(NodeManager& node_manager)
    : DistributedExecutor(node_manager) {}

std::future<std::string> MockDistributedExecutor::execute_remote_task(
    const Task::TaskId& task_id,
    Task::TaskFunction function,
    const std::string& target_node_id) {

    return std::async(std::launch::async, [this, task_id, function, target_node_id]() {
        // Simulate remote execution delay
        std::this_thread::sleep_for(std::chrono::milliseconds(50 + (rand() % 100)));

        // Simulate network latency and processing
        simulated_remote_tasks_.fetch_add(1);

        try {
            auto result = function();
            // Simulate result transmission back
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            return result;
        } catch (...) {
            throw;
        }
    });
}

bool MockDistributedExecutor::is_node_reachable(const std::string& node_id) {
    // Simulate node reachability check
    auto node = node_manager_.get_node(node_id);
    return node && node->is_available;
}

std::vector<std::string> MockDistributedExecutor::get_available_nodes() {
    auto nodes = node_manager_.get_available_nodes();
    std::vector<std::string> node_ids;

    for (const auto& node : nodes) {
        node_ids.push_back(node->id);
    }

    return node_ids;
}

// DistributedProcessor implementation
DistributedProcessor::DistributedProcessor(size_t local_threads, bool enable_remote)
    : scheduler_(std::make_unique<TaskScheduler>(local_threads)),
      node_manager_(std::make_unique<NodeManager>()),
      remote_enabled_(enable_remote) {}

DistributedProcessor::~DistributedProcessor() {
    stop();
}

void DistributedProcessor::set_executor(std::unique_ptr<DistributedExecutor> executor) {
    executor_ = std::move(executor);
}

std::future<std::string> DistributedProcessor::submit_task(Task::TaskFunction function) {
    return submit_task(std::move(function), TaskPriority::NORMAL);
}

std::future<std::string> DistributedProcessor::submit_task(Task::TaskFunction function, TaskPriority priority) {
    auto task_id = generate_task_id();
    return scheduler_->submit_task(task_id, std::move(function), priority);
}

std::future<std::string> DistributedProcessor::submit_remote_task(Task::TaskFunction function,
                                                                 const std::string& preferred_node) {
    if (!remote_enabled_.load() || !executor_) {
        // Fallback to local execution
        return submit_task(std::move(function), TaskPriority::NORMAL);
    }

    auto task_id = generate_task_id();

    // Select target node
    NodeManager::NodePtr target_node = nullptr;
    if (!preferred_node.empty()) {
        target_node = node_manager_->get_node(preferred_node);
    }

    if (!target_node) {
        target_node = node_manager_->get_best_node();
    }

    if (!target_node || !executor_->is_node_reachable(target_node->id)) {
        // Fallback to local execution if no suitable remote node
        return submit_task(std::move(function), TaskPriority::NORMAL);
    }

    // Update node load
    node_manager_->update_node_load(target_node->id, target_node->current_load.load() + 1);

    auto future = executor_->execute_remote_task(task_id, std::move(function), target_node->id);

    // For now, execute synchronously to avoid threading issues
    // In a real implementation, this would be truly asynchronous
    try {
        auto result = future.get();
        remote_tasks_completed_.fetch_add(1);

        // Safely decrement node load
        if (auto node = node_manager_->get_node(target_node->id)) {
            auto current_load = node->current_load.load();
            if (current_load > 0) {
                node_manager_->update_node_load(target_node->id,
                    static_cast<int>(current_load) - 1);
            }
        }

        // Return a ready future with the result
        std::promise<std::string> promise;
        promise.set_value(result);
        return promise.get_future();
    } catch (...) {
        // Safely decrement node load on error
        if (auto node = node_manager_->get_node(target_node->id)) {
            auto current_load = node->current_load.load();
            if (current_load > 0) {
                node_manager_->update_node_load(target_node->id,
                    static_cast<int>(current_load) - 1);
            }
        }
        throw;
    }
}

bool DistributedProcessor::add_node(const std::string& node_id, const std::string& address,
                                   int port, int max_tasks) {
    return node_manager_->register_node(node_id, address, port, max_tasks);
}

bool DistributedProcessor::remove_node(const std::string& node_id) {
    return node_manager_->unregister_node(node_id);
}

void DistributedProcessor::start() {
    if (running_.load()) return;

    running_.store(true);
    scheduler_->start();

    if (!executor_ && remote_enabled_.load()) {
        executor_ = std::make_unique<MockDistributedExecutor>(*node_manager_);
    }
}

void DistributedProcessor::stop() {
    if (!running_.load()) return;

    running_.store(false);
    scheduler_->stop();
}

size_t DistributedProcessor::pending_tasks() const {
    return scheduler_->pending_tasks();
}

size_t DistributedProcessor::running_tasks() const {
    return scheduler_->running_tasks();
}

size_t DistributedProcessor::completed_tasks() const {
    return scheduler_->completed_tasks();
}

std::vector<std::string> DistributedProcessor::get_available_nodes() const {
    return node_manager_->get_all_node_ids();
}

Task::TaskId DistributedProcessor::generate_task_id() const {
    static std::atomic<size_t> counter{0};
    std::ostringstream oss;
    oss << "task_" << std::chrono::steady_clock::now().time_since_epoch().count()
        << "_" << counter.fetch_add(1);
    return oss.str();
}

// Utility functions implementation
TaskPriority string_to_priority(const std::string& priority_str) {
    if (priority_str == "LOW") return TaskPriority::LOW;
    if (priority_str == "NORMAL") return TaskPriority::NORMAL;
    if (priority_str == "HIGH") return TaskPriority::HIGH;
    if (priority_str == "CRITICAL") return TaskPriority::CRITICAL;
    return TaskPriority::NORMAL;
}

std::string priority_to_string(TaskPriority priority) {
    switch (priority) {
        case TaskPriority::LOW: return "LOW";
        case TaskPriority::NORMAL: return "NORMAL";
        case TaskPriority::HIGH: return "HIGH";
        case TaskPriority::CRITICAL: return "CRITICAL";
        default: return "NORMAL";
    }
}

TaskStatus string_to_status(const std::string& status_str) {
    if (status_str == "PENDING") return TaskStatus::PENDING;
    if (status_str == "RUNNING") return TaskStatus::RUNNING;
    if (status_str == "COMPLETED") return TaskStatus::COMPLETED;
    if (status_str == "FAILED") return TaskStatus::FAILED;
    if (status_str == "CANCELLED") return TaskStatus::CANCELLED;
    return TaskStatus::PENDING;
}

std::string status_to_string(TaskStatus status) {
    switch (status) {
        case TaskStatus::PENDING: return "PENDING";
        case TaskStatus::RUNNING: return "RUNNING";
        case TaskStatus::COMPLETED: return "COMPLETED";
        case TaskStatus::FAILED: return "FAILED";
        case TaskStatus::CANCELLED: return "CANCELLED";
        default: return "PENDING";
    }
}

} // namespace distributed
} // namespace langchain