#include <catch2/catch_all.hpp>
#include "langchain/utils/thread_pool.hpp"
#include <atomic>
#include <chrono>
#include <vector>
#include <iostream>

using namespace langchain::utils;

TEST_CASE("ThreadPool - Basic Operations", "[utils][thread_pool]") {
    SECTION("Default construction") {
        ThreadPool pool;
        REQUIRE(pool.size() >= 1);
        REQUIRE(pool.get_active_tasks() == 0);
        REQUIRE(pool.get_completed_tasks() == 0);
    }

    SECTION("Custom number of threads") {
        ThreadPool pool(4);
        REQUIRE(pool.size() == 4);
    }

    SECTION("Submit simple task") {
        ThreadPool pool(2);

        auto future = pool.submit([]() { return 42; });

        REQUIRE(future.get() == 42);
        REQUIRE(pool.get_completed_tasks() == 1);
    }

    SECTION("Submit task with arguments") {
        ThreadPool pool(2);

        auto future = pool.submit([](int a, int b) { return a + b; }, 20, 22);

        REQUIRE(future.get() == 42);
    }

    SECTION("Submit void task") {
        ThreadPool pool(2);

        bool executed = false;
        auto future = pool.submit([&executed]() { executed = true; });

        future.get();
        REQUIRE(executed);
    }

    SECTION("Multiple tasks") {
        ThreadPool pool(4);

        std::vector<std::future<int>> futures;
        for (int i = 0; i < 10; ++i) {
            futures.push_back(pool.submit([i]() { return i * 2; }));
        }

        for (int i = 0; i < 10; ++i) {
            REQUIRE(futures[i].get() == i * 2);
        }

        REQUIRE(pool.get_completed_tasks() == 10);
    }
}

TEST_CASE("ThreadPool - Task Distribution", "[utils][thread_pool]") {
    SECTION("Round-robin distribution") {
        ThreadPool pool(3);
        std::atomic<int> worker_0_tasks{0};
        std::atomic<int> worker_1_tasks{0};
        std::atomic<int> worker_2_tasks{0};

        std::vector<std::future<void>> futures;
        for (int i = 0; i < 9; ++i) {
            futures.push_back(pool.submit_to_worker(i % 3, [&worker_0_tasks, &worker_1_tasks, &worker_2_tasks, i]() {
                if (i % 3 == 0) worker_0_tasks++;
                else if (i % 3 == 1) worker_1_tasks++;
                else worker_2_tasks++;
            }));
        }

        for (auto& future : futures) {
            future.get();
        }

        REQUIRE(worker_0_tasks.load() == 3);
        REQUIRE(worker_1_tasks.load() == 3);
        REQUIRE(worker_2_tasks.load() == 3);
    }

    SECTION("Invalid worker ID") {
        ThreadPool pool(2);

        REQUIRE_THROWS_AS(pool.submit_to_worker(5, []() {}), std::out_of_range);
    }
}

TEST_CASE("ThreadPool - Wait for All", "[utils][thread_pool]") {
    SECTION("Wait for completion") {
        ThreadPool pool(4);
        std::atomic<int> completed{0};

        for (int i = 0; i < 10; ++i) {
            pool.submit([&completed]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                completed++;
            });
        }

        pool.wait_for_all();
        REQUIRE(completed.load() == 10);
        REQUIRE(pool.get_active_tasks() == 0);
    }

    SECTION("Wait with no tasks") {
        ThreadPool pool(2);
        pool.wait_for_all();
        REQUIRE(pool.get_active_tasks() == 0);
    }
}

TEST_CASE("ThreadPool - Performance Metrics", "[utils][thread_pool]") {
    SECTION("Metrics collection") {
        ThreadPool pool(2);

        // Submit some tasks that take time
        for (int i = 0; i < 5; ++i) {
            pool.submit([]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            });
        }

        pool.wait_for_all();

        auto metrics = pool.get_performance_metrics();
        REQUIRE(metrics.at("total_tasks_completed") == 5.0);
        REQUIRE(metrics.at("active_tasks") == 0.0);
        REQUIRE(metrics.at("worker_threads") == 2.0);
        REQUIRE(metrics.at("average_execution_time_us") > 0.0);
    }

    SECTION("Metrics accuracy") {
        ThreadPool pool(2);
        const int num_tasks = 20;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_tasks; ++i) {
            pool.submit([]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            });
        }

        pool.wait_for_all();
        auto end_time = std::chrono::high_resolution_clock::now();

        auto metrics = pool.get_performance_metrics();
        REQUIRE(metrics.at("total_tasks_completed") == num_tasks);

        // The average execution time should be reasonable
        double avg_exec_time = metrics.at("average_execution_time_us");
        REQUIRE(avg_exec_time > 500.0);  // At least 500 microseconds
        REQUIRE(avg_exec_time < 5000.0); // Less than 5 milliseconds
    }
}

TEST_CASE("ThreadPool - Concurrent Execution", "[utils][thread_pool][concurrent]") {
    SECTION("Concurrent task submission") {
        ThreadPool pool(4);
        std::atomic<int> counter{0};
        std::vector<std::future<void>> futures;

        for (int i = 0; i < 100; ++i) {
            futures.push_back(pool.submit([&counter]() {
                counter.fetch_add(1);
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }));
        }

        for (auto& future : futures) {
            future.get();
        }

        REQUIRE(counter.load() == 100);
    }

    SECTION("Stress test") {
        ThreadPool pool(std::thread::hardware_concurrency());
        std::atomic<int> completed{0};
        const int num_tasks = 1000;

        std::vector<std::future<void>> futures;
        futures.reserve(num_tasks);

        for (int i = 0; i < num_tasks; ++i) {
            futures.push_back(pool.submit([&completed, i]() {
                // Do some work
                volatile int sum = 0;
                for (int j = 0; j < 100; ++j) {
                    sum += j;
                }
                completed++;
                (void)sum;  // Prevent optimization
            }));
        }

        for (auto& future : futures) {
            future.get();
        }

        REQUIRE(completed.load() == num_tasks);
    }
}

TEST_CASE("ThreadPool - Exception Handling", "[utils][thread_pool]") {
    SECTION("Task throws exception") {
        ThreadPool pool(2);

        auto future = pool.submit([]() -> int {
            throw std::runtime_error("Test exception");
            return 42;
        });

        REQUIRE_THROWS_AS(future.get(), std::runtime_error);
    }

    SECTION("Mixed successful and failed tasks") {
        ThreadPool pool(2);
        std::atomic<int> successful{0};
        std::atomic<int> failed{0};

        std::vector<std::future<void>> futures;
        for (int i = 0; i < 10; ++i) {
            futures.push_back(pool.submit([&successful, &failed, i]() {
                if (i % 3 == 0) {
                    throw std::runtime_error("Error in task " + std::to_string(i));
                } else {
                    successful++;
                }
            }));
        }

        // Some should succeed, some should fail
        int success_count = 0;
        int exception_count = 0;

        for (auto& future : futures) {
            try {
                future.get();
                success_count++;
            } catch (...) {
                exception_count++;
            }
        }

        REQUIRE(success_count + exception_count == 10);  // All 10 tasks should complete (either success or failure)
        REQUIRE(exception_count >= 3); // At least the 3 expected failures
        REQUIRE(successful.load() >= 6); // At least 6 tasks should succeed
    }
}

TEST_CASE("ThreadPool - Shutdown", "[utils][thread_pool]") {
    SECTION("Graceful shutdown") {
        auto pool = std::make_unique<ThreadPool>(2);

        // Submit some tasks
        std::atomic<int> completed{0};
        for (int i = 0; i < 5; ++i) {
            pool->submit([&completed]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                completed++;
            });
        }

        pool->wait_for_all();
        REQUIRE(completed.load() == 5);

        pool->shutdown();
        REQUIRE(pool->size() == 0);
    }

    SECTION("Shutdown with active tasks") {
        auto pool = std::make_unique<ThreadPool>(2);

        std::atomic<int> started{0};
        std::atomic<int> completed{0};

        // Submit long-running tasks
        std::vector<std::future<void>> futures;
        for (int i = 0; i < 3; ++i) {
            futures.push_back(pool->submit([&started, &completed]() {
                started++;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                completed++;
            }));
        }

        // Wait for tasks to start
        while (started.load() < 3) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        pool->shutdown();

        // Most tasks should complete due to graceful shutdown
        pool->wait_for_all();
    }
}

TEST_CASE("LockFreeQueue - Basic Operations", "[utils][lock_free_queue]") {
    SECTION("Enqueue and dequeue") {
        LockFreeQueue<int> queue;

        REQUIRE(queue.empty());

        queue.enqueue(42);
        REQUIRE_FALSE(queue.empty());

        auto result = queue.dequeue();
        REQUIRE(result.has_value());
        REQUIRE(result.value() == 42);
        REQUIRE(queue.empty());
    }

    SECTION("Multiple items") {
        LockFreeQueue<std::string> queue;

        queue.enqueue("first");
        queue.enqueue("second");
        queue.enqueue("third");

        auto item1 = queue.dequeue();
        auto item2 = queue.dequeue();
        auto item3 = queue.dequeue();

        REQUIRE(item1.value() == "first");
        REQUIRE(item2.value() == "second");
        REQUIRE(item3.value() == "third");

        REQUIRE(queue.empty());
    }

    SECTION("Dequeue from empty") {
        LockFreeQueue<int> queue;

        auto result = queue.dequeue();
        REQUIRE_FALSE(result.has_value());
    }

    SECTION("Complex types") {
        LockFreeQueue<std::vector<int>> queue;

        std::vector<int> vec = {1, 2, 3, 4, 5};
        queue.enqueue(vec);

        auto result = queue.dequeue();
        REQUIRE(result.has_value());
        REQUIRE(result.value() == vec);
    }
}

TEST_CASE("Timer - Performance Measurement", "[utils][timer]") {
    SECTION("Basic timing") {
        bool callback_called = false;
        std::string callback_name;
        double callback_time = 0.0;

        {
            Timer timer("test_timer", [&](const std::string& name, double time) {
                callback_called = true;
                callback_name = name;
                callback_time = time;
            });

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        REQUIRE(callback_called);
        REQUIRE(callback_name == "test_timer");
        REQUIRE(callback_time >= 5.0);  // At least 5ms
        REQUIRE(callback_time < 50.0);  // Less than 50ms
    }

    SECTION("Timer without callback") {
        Timer timer("no_callback");

        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        double elapsed = timer.elapsed_ms();
        REQUIRE(elapsed >= 3.0);
        REQUIRE(elapsed < 20.0);
    }

    SECTION("Multiple timers") {
        std::vector<std::pair<std::string, double>> timings;

        {
            Timer timer1("timer1", [&](const std::string& name, double time) {
                timings.emplace_back(name, time);
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(5));

            Timer timer2("timer2", [&](const std::string& name, double time) {
                timings.emplace_back(name, time);
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        REQUIRE(timings.size() == 2);
        REQUIRE(timings[0].first == "timer2");  // timer2 finishes first
        REQUIRE(timings[1].first == "timer1");
        REQUIRE(timings[0].second < timings[1].second);
    }
}

TEST_CASE("GlobalThreadPool - Singleton Access", "[utils][global_thread_pool]") {
    SECTION("Get instance") {
        ThreadPool& pool1 = GlobalThreadPool::get_instance();
        ThreadPool& pool2 = GlobalThreadPool::get_instance();

        REQUIRE(&pool1 == &pool2);
        REQUIRE(pool1.size() >= 1);
    }

    SECTION("Submit tasks through global interface") {
        auto future = GlobalThreadPool::submit([]() { return 123; });

        REQUIRE(future.get() == 123);
    }

    SECTION("Wait for all in global pool") {
        std::atomic<int> completed{0};

        for (int i = 0; i < 5; ++i) {
            GlobalThreadPool::submit([&completed]() {
                completed++;
            });
        }

        GlobalThreadPool::wait_for_all();
        REQUIRE(completed.load() == 5);
    }
}