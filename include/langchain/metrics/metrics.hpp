#pragma once

#include <chrono>
#include <atomic>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>
#include <mutex>
#include <functional>

namespace langchain {
namespace metrics {

enum class MetricType {
    COUNTER,
    GAUGE,
    HISTOGRAM,
    TIMER
};

class MetricValue {
public:
    virtual ~MetricValue() = default;
    virtual std::string to_string() const = 0;
    virtual void reset() = 0;
    virtual MetricType type() const = 0;
};

class Counter : public MetricValue {
private:
    std::atomic<uint64_t> value_{0};
    std::string name_;

public:
    explicit Counter(const std::string& name) : name_(name) {}

    void increment() { value_.fetch_add(1, std::memory_order_relaxed); }
    void increment(uint64_t delta) { value_.fetch_add(delta, std::memory_order_relaxed); }
    uint64_t value() const { return value_.load(std::memory_order_relaxed); }

    std::string to_string() const override;
    void reset() override { value_.store(0, std::memory_order_relaxed); }
    MetricType type() const override { return MetricType::COUNTER; }
};

class Gauge : public MetricValue {
private:
    std::atomic<double> value_{0.0};
    std::string name_;

public:
    explicit Gauge(const std::string& name) : name_(name) {}

    void set(double value) { value_.store(value, std::memory_order_relaxed); }
    void increment(double delta = 1.0) {
        auto current = value_.load(std::memory_order_relaxed);
        value_.store(current + delta, std::memory_order_relaxed);
    }
    void decrement(double delta = 1.0) {
        auto current = value_.load(std::memory_order_relaxed);
        value_.store(current - delta, std::memory_order_relaxed);
    }
    double value() const { return value_.load(std::memory_order_relaxed); }

    std::string to_string() const override;
    void reset() override { value_.store(0.0, std::memory_order_relaxed); }
    MetricType type() const override { return MetricType::GAUGE; }
};

class Histogram : public MetricValue {
private:
    mutable std::mutex mutex_;
    std::vector<double> observations_;
    double sum_{0.0};
    uint64_t count_{0};
    std::string name_;

public:
    explicit Histogram(const std::string& name) : name_(name) {}

    void observe(double value);
    double sum() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return sum_;
    }
    uint64_t count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }
    double mean() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_ > 0 ? sum_ / count_ : 0.0;
    }
    double min() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return observations_.empty() ? 0.0 : *std::min_element(observations_.begin(), observations_.end());
    }
    double max() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return observations_.empty() ? 0.0 : *std::max_element(observations_.begin(), observations_.end());
    }
    std::vector<double> observations() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return observations_;
    }

    std::string to_string() const override;
    void reset() override;
    MetricType type() const override { return MetricType::HISTOGRAM; }
};

class Timer : public MetricValue {
private:
    mutable std::mutex mutex_;
    std::vector<std::chrono::nanoseconds> observations_;
    std::chrono::nanoseconds total_time_{0};
    uint64_t count_{0};
    std::string name_;

public:
    using TimePoint = std::chrono::high_resolution_clock::time_point;

    explicit Timer(const std::string& name) : name_(name) {}

    class ScopedTimer {
    private:
        Timer* timer_;
        TimePoint start_time_;

    public:
        explicit ScopedTimer(Timer* timer)
            : timer_(timer), start_time_(std::chrono::high_resolution_clock::now()) {}

        ~ScopedTimer() {
            auto end_time = std::chrono::high_resolution_clock::now();
            timer_->observe(std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time_));
        }
    };

    void observe(std::chrono::nanoseconds duration);
    ScopedTimer start_timer() { return ScopedTimer(this); }
    ScopedTimer scoped_timer() { return ScopedTimer(this); }

    std::chrono::nanoseconds total_time() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_time_;
    }
    uint64_t count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }
    double mean_seconds() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ == 0) return 0.0;
        auto total_seconds = std::chrono::duration<double>(total_time_).count();
        return total_seconds / count_;
    }
    std::chrono::nanoseconds min() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return observations_.empty() ? std::chrono::nanoseconds{0} : *std::min_element(observations_.begin(), observations_.end());
    }
    std::chrono::nanoseconds max() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return observations_.empty() ? std::chrono::nanoseconds{0} : *std::max_element(observations_.begin(), observations_.end());
    }

    std::string to_string() const override;
    void reset() override;
    MetricType type() const override { return MetricType::TIMER; }
};

class MetricsRegistry {
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<MetricValue>> metrics_;

public:
    static MetricsRegistry& instance();

    Counter& get_counter(const std::string& name);
    Gauge& get_gauge(const std::string& name);
    Histogram& get_histogram(const std::string& name);
    Timer& get_timer(const std::string& name);

    void remove_metric(const std::string& name);
    void reset_all();
    void clear();
    std::vector<std::string> metric_names() const;
    const MetricValue* get_metric(const std::string& name) const;

    // Export metrics in various formats
    std::string export_prometheus() const;
    std::string export_json() const;
};

class ScopedTimer {
private:
    Timer* timer_;
    Timer::ScopedTimer scoped_timer_;

public:
    explicit ScopedTimer(const std::string& metric_name)
        : timer_(&MetricsRegistry::instance().get_timer(metric_name))
        , scoped_timer_(timer_->start_timer()) {}
};

// Convenience macros for metrics
#define METRICS_COUNTER(name) langchain::metrics::MetricsRegistry::instance().get_counter(name)
#define METRICS_GAUGE(name) langchain::metrics::MetricsRegistry::instance().get_gauge(name)
#define METRICS_HISTOGRAM(name) langchain::metrics::MetricsRegistry::instance().get_histogram(name)
#define METRICS_TIMER(name) langchain::metrics::MetricsRegistry::instance().get_timer(name)
#define METRICS_SCOPE_TIMER(name) langchain::metrics::ScopedTimer(name)

} // namespace metrics
} // namespace langchain