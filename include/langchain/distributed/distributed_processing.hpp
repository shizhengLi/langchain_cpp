#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unordered_map>
#include <thread>

namespace langchain {
namespace distributed {

// Task status enumeration
enum class TaskStatus {
    PENDING,
    RUNNING,
    COMPLETED,
    FAILED,
    CANCELLED
};

// Task priority levels
enum class TaskPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

// Forward declarations
class Task;
class TaskScheduler;
class DistributedExecutor;
class NodeManager;

// Task base class
class Task {
public:
    using TaskFunction = std::function<std::string()>;
    using TaskId = std::string;

    Task(TaskId id, TaskFunction func, TaskPriority priority = TaskPriority::NORMAL);
    virtual ~Task() = default;

    TaskId id() const { return id_; }
    TaskStatus status() const { return status_.load(); }
    TaskPriority priority() const { return priority_; }

    void set_status(TaskStatus status) { status_.store(status); }
    void set_result(const std::string& result) { result_ = result; }
    void set_error(const std::string& error) { error_ = error; }

    std::string result() const { return result_; }
    std::string error() const { return error_; }

    TaskFunction function() const { return function_; }

    virtual std::string execute() {
        try {
            set_status(TaskStatus::RUNNING);
            auto result = function_();
            set_result(result);
            set_status(TaskStatus::COMPLETED);
            return result;
        } catch (const std::exception& e) {
            set_error(e.what());
            set_status(TaskStatus::FAILED);
            throw;
        }
    }

private:
    TaskId id_;
    TaskFunction function_;
    TaskPriority priority_;
    std::atomic<TaskStatus> status_{TaskStatus::PENDING};
    std::string result_;
    std::string error_;
};

// Task scheduler for managing task queues
class TaskScheduler {
public:
    using TaskPtr = std::shared_ptr<Task>;

    TaskScheduler(size_t max_concurrent_tasks = 4);
    ~TaskScheduler();

    // Task submission and management
    std::future<std::string> submit_task(TaskPtr task);
    std::future<std::string> submit_task(Task::TaskId id, Task::TaskFunction func,
                                        TaskPriority priority = TaskPriority::NORMAL);

    // Scheduler control
    void start();
    void stop();
    void pause();
    void resume();

    // Status and monitoring
    size_t pending_tasks() const;
    size_t running_tasks() const;
    size_t completed_tasks() const;
    std::vector<Task::TaskId> get_pending_task_ids() const;

    // Task management
    bool cancel_task(const Task::TaskId& task_id);
    TaskPtr get_task(const Task::TaskId& task_id);

private:
    void worker_loop();
    TaskPtr get_next_task();

    mutable std::mutex queue_mutex_;
    std::condition_variable queue_condition_;
    std::condition_variable finished_condition_;

    // Priority queue with custom comparator
    struct TaskComparator {
        bool operator()(const TaskPtr& a, const TaskPtr& b) const {
            return a->priority() < b->priority();
        }
    };

    std::priority_queue<TaskPtr, std::vector<TaskPtr>, TaskComparator> task_queue_;
    std::unordered_map<Task::TaskId, TaskPtr> all_tasks_;
    std::unordered_map<Task::TaskId, std::promise<std::string>> task_promises_;

    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};
    std::atomic<size_t> max_concurrent_tasks_;
    std::atomic<size_t> running_tasks_{0};
    std::atomic<size_t> completed_tasks_{0};
};

// Node information for distributed execution
struct NodeInfo {
    std::string id;
    std::string address;
    int port;
    int max_concurrent_tasks;
    bool is_available;
    std::atomic<size_t> current_load{0};

    NodeInfo(const std::string& node_id, const std::string& addr, int p, int max_tasks)
        : id(node_id), address(addr), port(p), max_concurrent_tasks(max_tasks), is_available(true) {}
};

// Node manager for handling distributed nodes
class NodeManager {
public:
    using NodePtr = std::shared_ptr<NodeInfo>;

    NodeManager();
    ~NodeManager();

    // Node registration and management
    bool register_node(const std::string& node_id, const std::string& address,
                      int port, int max_concurrent_tasks);
    bool unregister_node(const std::string& node_id);

    // Node selection
    NodePtr get_best_node();
    std::vector<NodePtr> get_available_nodes();
    NodePtr get_node(const std::string& node_id);

    // Status monitoring
    std::vector<std::string> get_all_node_ids() const;
    size_t available_nodes_count() const;
    void update_node_load(const std::string& node_id, size_t load);

private:
    mutable std::mutex nodes_mutex_;
    std::unordered_map<std::string, NodePtr> nodes_;

    // Health check thread
    std::thread health_check_thread_;
    std::atomic<bool> running_{false};
    void health_check_loop();
};

// Distributed executor for remote task execution
class DistributedExecutor {
public:
    DistributedExecutor(NodeManager& node_manager);
    virtual ~DistributedExecutor() = default;

    virtual std::future<std::string> execute_remote_task(
        const Task::TaskId& task_id,
        Task::TaskFunction function,
        const std::string& target_node_id = "") = 0;

    virtual bool is_node_reachable(const std::string& node_id) = 0;
    virtual std::vector<std::string> get_available_nodes() = 0;

protected:
    NodeManager& node_manager_;
};

// Mock distributed executor for testing
class MockDistributedExecutor : public DistributedExecutor {
public:
    MockDistributedExecutor(NodeManager& node_manager);

    std::future<std::string> execute_remote_task(
        const Task::TaskId& task_id,
        Task::TaskFunction function,
        const std::string& target_node_id = "") override;

    bool is_node_reachable(const std::string& node_id) override;
    std::vector<std::string> get_available_nodes() override;

private:
    std::atomic<size_t> simulated_remote_tasks_{0};
    std::mutex mock_mutex_;
};

// Main distributed processing coordinator
class DistributedProcessor {
public:
    DistributedProcessor(size_t local_threads = 4, bool enable_remote = false);
    ~DistributedProcessor();

    // Configuration
    void set_executor(std::unique_ptr<DistributedExecutor> executor);
    void enable_remote_processing(bool enable) { remote_enabled_ = enable; }

    // Task submission
    std::future<std::string> submit_task(Task::TaskFunction function);
    std::future<std::string> submit_task(Task::TaskFunction function, TaskPriority priority);
    std::future<std::string> submit_remote_task(Task::TaskFunction function,
                                               const std::string& preferred_node = "");

    // Node management
    bool add_node(const std::string& node_id, const std::string& address,
                  int port, int max_tasks = 4);
    bool remove_node(const std::string& node_id);

    // Control
    void start();
    void stop();

    // Monitoring
    size_t pending_tasks() const;
    size_t running_tasks() const;
    size_t completed_tasks() const;
    std::vector<std::string> get_available_nodes() const;

private:
    std::unique_ptr<TaskScheduler> scheduler_;
    std::unique_ptr<NodeManager> node_manager_;
    std::unique_ptr<DistributedExecutor> executor_;
    std::atomic<bool> remote_enabled_{false};
    std::atomic<bool> running_{false};

    // Statistics
    mutable std::mutex stats_mutex_;
    std::atomic<size_t> local_tasks_completed_{0};
    std::atomic<size_t> remote_tasks_completed_{0};

    Task::TaskId generate_task_id() const;
};

// Utility functions
TaskPriority string_to_priority(const std::string& priority_str);
std::string priority_to_string(TaskPriority priority);
TaskStatus string_to_status(const std::string& status_str);
std::string status_to_string(TaskStatus status);

} // namespace distributed
} // namespace langchain