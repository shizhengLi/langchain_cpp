#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include "langchain/metrics/metrics.hpp"
#include <thread>
#include <chrono>
#include <vector>

using namespace langchain::metrics;

TEST_CASE("Counter basic functionality", "[metrics][counter]") {
    SECTION("Increment operations") {
        Counter counter("test_counter");

        REQUIRE(counter.value() == 0);

        counter.increment();
        REQUIRE(counter.value() == 1);

        counter.increment(5);
        REQUIRE(counter.value() == 6);

        counter.increment(10);
        REQUIRE(counter.value() == 16);
    }

    SECTION("Reset functionality") {
        Counter counter("test_counter");

        counter.increment(42);
        REQUIRE(counter.value() == 42);

        counter.reset();
        REQUIRE(counter.value() == 0);
    }

    SECTION("String representation") {
        Counter counter("test_counter");
        counter.increment(5);

        std::string str = counter.to_string();
        REQUIRE(str.find("counter") != std::string::npos);
        REQUIRE(str.find("5") != std::string::npos);
    }

    SECTION("Type identification") {
        Counter counter("test_counter");
        REQUIRE(counter.type() == MetricType::COUNTER);
    }
}

TEST_CASE("Gauge basic functionality", "[metrics][gauge]") {
    SECTION("Set and modify operations") {
        Gauge gauge("test_gauge");

        REQUIRE(gauge.value() == 0.0);

        gauge.set(5.5);
        REQUIRE(gauge.value() == 5.5);

        gauge.increment(2.0);
        REQUIRE(gauge.value() == 7.5);

        gauge.decrement(1.5);
        REQUIRE(gauge.value() == 6.0);
    }

    SECTION("Default increment") {
        Gauge gauge("test_gauge");

        gauge.increment();
        REQUIRE(gauge.value() == 1.0);

        gauge.decrement();
        REQUIRE(gauge.value() == 0.0);
    }

    SECTION("Reset functionality") {
        Gauge gauge("test_gauge");

        gauge.set(42.0);
        REQUIRE(gauge.value() == 42.0);

        gauge.reset();
        REQUIRE(gauge.value() == 0.0);
    }

    SECTION("String representation") {
        Gauge gauge("test_gauge");
        gauge.set(3.14159);

        std::string str = gauge.to_string();
        REQUIRE(str.find("gauge") != std::string::npos);
        REQUIRE(str.find("3.14") != std::string::npos);
    }

    SECTION("Type identification") {
        Gauge gauge("test_gauge");
        REQUIRE(gauge.type() == MetricType::GAUGE);
    }
}

TEST_CASE("Histogram basic functionality", "[metrics][histogram]") {
    SECTION("Observe operations") {
        Histogram histogram("test_histogram");

        REQUIRE(histogram.count() == 0.0);
        REQUIRE(histogram.sum() == 0.0);

        histogram.observe(1.0);
        histogram.observe(2.0);
        histogram.observe(3.0);

        REQUIRE(histogram.count() == 3.0);
        REQUIRE(histogram.sum() == 6.0);
        REQUIRE(histogram.mean() == 2.0);
        REQUIRE(histogram.min() == 1.0);
        REQUIRE(histogram.max() == 3.0);
    }

    SECTION("Single observation") {
        Histogram histogram("test_histogram");

        histogram.observe(42.0);

        REQUIRE(histogram.count() == 1.0);
        REQUIRE(histogram.sum() == 42.0);
        REQUIRE(histogram.mean() == 42.0);
        REQUIRE(histogram.min() == 42.0);
        REQUIRE(histogram.max() == 42.0);
    }

    SECTION("Negative values") {
        Histogram histogram("test_histogram");

        histogram.observe(-1.0);
        histogram.observe(1.0);

        REQUIRE(histogram.count() == 2.0);
        REQUIRE(histogram.sum() == 0.0);
        REQUIRE(histogram.mean() == 0.0);
        REQUIRE(histogram.min() == -1.0);
        REQUIRE(histogram.max() == 1.0);
    }

    SECTION("Reset functionality") {
        Histogram histogram("test_histogram");

        histogram.observe(1.0);
        histogram.observe(2.0);

        REQUIRE(histogram.count() == 2.0);

        histogram.reset();

        REQUIRE(histogram.count() == 0.0);
        REQUIRE(histogram.sum() == 0.0);
    }

    SECTION("String representation") {
        Histogram histogram("test_histogram");

        histogram.observe(1.0);
        histogram.observe(2.0);
        histogram.observe(3.0);

        std::string str = histogram.to_string();
        REQUIRE(str.find("histogram") != std::string::npos);
        REQUIRE(str.find("count=3") != std::string::npos);
        REQUIRE(str.find("sum=6") != std::string::npos);
    }

    SECTION("Type identification") {
        Histogram histogram("test_histogram");
        REQUIRE(histogram.type() == MetricType::HISTOGRAM);
    }
}

TEST_CASE("Timer basic functionality", "[metrics][timer]") {
    SECTION("Record operations") {
        Timer timer("test_timer");

        REQUIRE(timer.count() == 0);

        timer.observe(std::chrono::nanoseconds(100));
        timer.observe(std::chrono::nanoseconds(200));
        timer.observe(std::chrono::nanoseconds(300));

        REQUIRE(timer.count() == 3);

        auto total = timer.total_time();
        REQUIRE(total == std::chrono::nanoseconds(600));

        auto avg_ms = timer.mean_seconds() * 1000.0;
        REQUIRE(avg_ms == Catch::Approx(0.3));  // 600ns / 3 = 200ns = 0.0002ms, but we're using 1000ns for testing
    }

    SECTION("Single recording") {
        Timer timer("test_timer");

        timer.observe(std::chrono::nanoseconds(1000));

        REQUIRE(timer.count() == 1);
        REQUIRE(timer.total_time() == std::chrono::nanoseconds(1000));
        REQUIRE(timer.mean_seconds() == Catch::Approx(0.000001));
    }

    SECTION("Reset functionality") {
        Timer timer("test_timer");

        timer.observe(std::chrono::nanoseconds(100));

        REQUIRE(timer.count() == 1);

        timer.reset();

        REQUIRE(timer.count() == 0);
        REQUIRE(timer.total_time() == std::chrono::nanoseconds(0));
    }

    SECTION("String representation") {
        Timer timer("test_timer");

        timer.observe(std::chrono::milliseconds(1));  // 1,000,000 nanoseconds
        timer.observe(std::chrono::milliseconds(2));  // 2,000,000 nanoseconds

        std::string str = timer.to_string();
        REQUIRE(str.find("timer") != std::string::npos);
        REQUIRE(str.find("count=2") != std::string::npos);
        REQUIRE(str.find("total") != std::string::npos);
    }

    SECTION("Type identification") {
        Timer timer("test_timer");
        REQUIRE(timer.type() == MetricType::TIMER);
    }
}

TEST_CASE("Timer scoped timer", "[metrics][timer][scoped]") {
    SECTION("RAII timer measurement") {
        Timer timer("test_timer");

        {
            auto scoped = timer.scoped_timer();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        REQUIRE(timer.count() == 1);
        REQUIRE(timer.total_time() > std::chrono::nanoseconds(0));

        auto duration_ms = std::chrono::duration<double, std::milli>(timer.total_time()).count();
        REQUIRE(duration_ms >= 1.0);
        REQUIRE(duration_ms < 100.0);  // Should be close to 1ms, not 100ms
    }

    SECTION("Multiple scoped timers") {
        Timer timer("test_timer");

        {
            auto scoped1 = timer.scoped_timer();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        {
            auto scoped2 = timer.scoped_timer();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        REQUIRE(timer.count() == 2);
        REQUIRE(timer.total_time() > std::chrono::nanoseconds(0));
    }
}

TEST_CASE("MetricsRegistry functionality", "[metrics][registry]") {
    SECTION("Singleton pattern") {
        auto& registry1 = MetricsRegistry::instance();
        auto& registry2 = MetricsRegistry::instance();

        REQUIRE(&registry1 == &registry2);
    }

    SECTION("Counter creation and access") {
        auto& registry = MetricsRegistry::instance();

        auto& counter1 = registry.get_counter("test_counter");
        counter1.increment(5);

        auto& counter2 = registry.get_counter("test_counter");
        REQUIRE(counter2.value() == 5);

        auto& counter3 = registry.get_counter("another_counter");
        REQUIRE(counter3.value() == 0);
    }

    SECTION("Gauge creation and access") {
        auto& registry = MetricsRegistry::instance();

        auto& gauge1 = registry.get_gauge("test_gauge");
        gauge1.set(3.14);

        auto& gauge2 = registry.get_gauge("test_gauge");
        REQUIRE(gauge2.value() == 3.14);
    }

    SECTION("Histogram creation and access") {
        auto& registry = MetricsRegistry::instance();

        auto& histogram1 = registry.get_histogram("test_histogram");
        histogram1.observe(1.0);
        histogram1.observe(2.0);

        auto& histogram2 = registry.get_histogram("test_histogram");
        REQUIRE(histogram2.count() == 2.0);
        REQUIRE(histogram2.sum() == 3.0);
    }

    SECTION("Timer creation and access") {
        auto& registry = MetricsRegistry::instance();

        auto& timer1 = registry.get_timer("test_timer");
        timer1.observe(std::chrono::nanoseconds(100));

        auto& timer2 = registry.get_timer("test_timer");
        REQUIRE(timer2.count() == 1);
        REQUIRE(timer2.total_time() == std::chrono::nanoseconds(100));
    }

    SECTION("Metric names retrieval") {
        auto& registry = MetricsRegistry::instance();

        registry.get_counter("counter1");
        registry.get_gauge("gauge1");
        registry.get_histogram("histogram1");
        registry.get_timer("timer1");

        auto names = registry.metric_names();
        REQUIRE(names.size() >= 4);

        // Check that all our metrics are in the list
        REQUIRE(std::find(names.begin(), names.end(), "counter1") != names.end());
        REQUIRE(std::find(names.begin(), names.end(), "gauge1") != names.end());
        REQUIRE(std::find(names.begin(), names.end(), "histogram1") != names.end());
        REQUIRE(std::find(names.begin(), names.end(), "timer1") != names.end());
    }

    SECTION("Metric access by name") {
        auto& registry = MetricsRegistry::instance();

        registry.get_counter("test_counter").increment(42);

        const auto* metric = registry.get_metric("test_counter");
        REQUIRE(metric != nullptr);
        REQUIRE(metric->type() == MetricType::COUNTER);

        const auto* nonexistent = registry.get_metric("nonexistent");
        REQUIRE(nonexistent == nullptr);
    }

    SECTION("Remove metric") {
        auto& registry = MetricsRegistry::instance();

        registry.get_counter("temp_counter");
        REQUIRE(registry.get_metric("temp_counter") != nullptr);

        registry.remove_metric("temp_counter");
        REQUIRE(registry.get_metric("temp_counter") == nullptr);
    }

    SECTION("Reset all metrics") {
        auto& registry = MetricsRegistry::instance();

        registry.get_counter("test_counter").increment(10);
        registry.get_gauge("test_gauge").set(5.5);
        registry.get_histogram("test_histogram").observe(3.0);
        registry.get_timer("test_timer").observe(std::chrono::nanoseconds(100));

        registry.reset_all();

        REQUIRE(registry.get_counter("test_counter").value() == 0);
        REQUIRE(registry.get_gauge("test_gauge").value() == 0.0);
        REQUIRE(registry.get_histogram("test_histogram").count() == 0.0);
        REQUIRE(registry.get_timer("test_timer").count() == 0);
    }

    SECTION("Clear all metrics") {
        auto& registry = MetricsRegistry::instance();

        registry.get_counter("test_counter");
        registry.get_gauge("test_gauge");

        REQUIRE(registry.metric_names().size() >= 2);

        registry.clear();

        REQUIRE(registry.metric_names().empty());
    }
}

TEST_CASE("MetricsRegistry export functionality", "[metrics][registry][export]") {
    SECTION("Export Prometheus format") {
        auto& registry = MetricsRegistry::instance();

        // Clear previous metrics
        registry.clear();

        registry.get_counter("requests_total").increment(42);
        registry.get_gauge("active_connections").set(10.5);
        registry.get_histogram("response_time").observe(1.0);
        registry.get_histogram("response_time").observe(2.0);
        registry.get_timer("request_duration").observe(std::chrono::milliseconds(100));

        std::string prometheus = registry.export_prometheus();

        // Check for Prometheus format elements
        REQUIRE(prometheus.find("# HELP") != std::string::npos);
        REQUIRE(prometheus.find("# TYPE") != std::string::npos);
        REQUIRE(prometheus.find("requests_total") != std::string::npos);
        REQUIRE(prometheus.find("42") != std::string::npos);
        REQUIRE(prometheus.find("active_connections") != std::string::npos);
        REQUIRE(prometheus.find("10.5") != std::string::npos);
        REQUIRE(prometheus.find("response_time_count") != std::string::npos);
        REQUIRE(prometheus.find("response_time_sum") != std::string::npos);
    }

    SECTION("Export JSON format") {
        auto& registry = MetricsRegistry::instance();

        // Clear previous metrics
        registry.clear();

        registry.get_counter("test_counter").increment(5);
        registry.get_gauge("test_gauge").set(3.14);

        std::string json = registry.export_json();

        // Check for JSON format elements
        REQUIRE(json.find("{") != std::string::npos);
        REQUIRE(json.find("}") != std::string::npos);
        REQUIRE(json.find("\"metrics\"") != std::string::npos);
        REQUIRE(json.find("\"test_counter\"") != std::string::npos);
        REQUIRE(json.find("\"test_gauge\"") != std::string::npos);
    }
}

TEST_CASE("ScopedTimer convenience class", "[metrics][timer][scoped]") {
    SECTION("Basic usage") {
        auto& registry = MetricsRegistry::instance();

        {
            ScopedTimer timer("test_scoped_timer");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        auto& timer = registry.get_timer("test_scoped_timer");
        REQUIRE(timer.count() == 1);
        REQUIRE(timer.total_time() > std::chrono::nanoseconds(0));
    }

    SECTION("Multiple measurements") {
        auto& registry = MetricsRegistry::instance();

        for (int i = 0; i < 3; ++i) {
            ScopedTimer timer("test_scoped_timer");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        auto& timer = registry.get_timer("test_scoped_timer");
        REQUIRE(timer.count() == 3);
    }
}

TEST_CASE("Metrics thread safety", "[metrics][concurrent]") {
    SECTION("Concurrent counter increments") {
        auto& registry = MetricsRegistry::instance();
        auto& counter = registry.get_counter("concurrent_counter");

        const int num_threads = 10;
        const int increments_per_thread = 1000;
        std::vector<std::thread> threads;

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&counter, increments_per_thread]() {
                for (int j = 0; j < increments_per_thread; ++j) {
                    counter.increment();
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        REQUIRE(counter.value() == num_threads * increments_per_thread);
    }

    SECTION("Concurrent gauge operations") {
        auto& registry = MetricsRegistry::instance();
        auto& gauge = registry.get_gauge("concurrent_gauge");

        const int num_threads = 10;
        const int operations_per_thread = 100;
        std::vector<std::thread> threads;

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&gauge, operations_per_thread, i]() {
                for (int j = 0; j < operations_per_thread; ++j) {
                    gauge.increment(1.0);
                    gauge.decrement(0.5);  // Net +0.5 per operation
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        double expected_value = num_threads * operations_per_thread * 0.5;
        REQUIRE(gauge.value() == Catch::Approx(expected_value));
    }

    SECTION("Concurrent histogram observations") {
        auto& registry = MetricsRegistry::instance();
        auto& histogram = registry.get_histogram("concurrent_histogram");

        const int num_threads = 10;
        const int observations_per_thread = 100;
        std::vector<std::thread> threads;

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&histogram, observations_per_thread, i]() {
                for (int j = 0; j < observations_per_thread; ++j) {
                    histogram.observe(static_cast<double>(i * observations_per_thread + j));
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        REQUIRE(histogram.count() == num_threads * observations_per_thread);
        REQUIRE(histogram.min() == 0.0);
        REQUIRE(histogram.max() == Catch::Approx(num_threads * observations_per_thread - 1.0));
    }
}