#include "langchain/metrics/metrics.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

namespace langchain {
namespace metrics {

// Counter implementation
std::string Counter::to_string() const {
    return "counter{value=" + std::to_string(value()) + "}";
}

// Gauge implementation
std::string Gauge::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << "gauge{value=" << value() << "}";
    return oss.str();
}

// Histogram implementation
void Histogram::observe(double value) {
    std::lock_guard<std::mutex> lock(mutex_);
    observations_.push_back(value);
    sum_ += value;
    ++count_;
}

std::string Histogram::to_string() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "histogram{count=" << count_ << ",sum=" << sum_;
    if (!observations_.empty()) {
        auto min_val = *std::min_element(observations_.begin(), observations_.end());
        auto max_val = *std::max_element(observations_.begin(), observations_.end());
        double mean = sum_ / count_;
        oss << ",min=" << min_val << ",max=" << max_val << ",mean=" << mean;
    }
    oss << "}";
    return oss.str();
}

void Histogram::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    observations_.clear();
    sum_ = 0.0;
    count_ = 0;
}

// Timer implementation
void Timer::observe(std::chrono::nanoseconds duration) {
    std::lock_guard<std::mutex> lock(mutex_);
    observations_.push_back(duration);
    total_time_ += duration;
    ++count_;
}

std::string Timer::to_string() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);

    double total_ms = std::chrono::duration<double, std::milli>(total_time_).count();
    double avg_ms = count_ > 0 ? total_ms / count_ : 0.0;

    oss << "timer{count=" << count_ << ",total_ms=" << total_ms << ",avg_ms=" << avg_ms;

    if (!observations_.empty()) {
        auto min_duration = *std::min_element(observations_.begin(), observations_.end());
        auto max_duration = *std::max_element(observations_.begin(), observations_.end());
        double min_ms = std::chrono::duration<double, std::milli>(min_duration).count();
        double max_ms = std::chrono::duration<double, std::milli>(max_duration).count();
        oss << ",min_ms=" << min_ms << ",max_ms=" << max_ms;
    }
    oss << "}";
    return oss.str();
}

void Timer::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    observations_.clear();
    total_time_ = std::chrono::nanoseconds{0};
    count_ = 0;
}

// MetricsRegistry implementation
MetricsRegistry& MetricsRegistry::instance() {
    static MetricsRegistry instance;
    return instance;
}

Counter& MetricsRegistry::get_counter(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = metrics_.find(name);
    if (it == metrics_.end()) {
        auto counter = std::make_unique<Counter>(name);
        it = metrics_.emplace(name, std::move(counter)).first;
    }
    return static_cast<Counter&>(*it->second);
}

Gauge& MetricsRegistry::get_gauge(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = metrics_.find(name);
    if (it == metrics_.end()) {
        auto gauge = std::make_unique<Gauge>(name);
        it = metrics_.emplace(name, std::move(gauge)).first;
    }
    return static_cast<Gauge&>(*it->second);
}

Histogram& MetricsRegistry::get_histogram(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = metrics_.find(name);
    if (it == metrics_.end()) {
        auto histogram = std::make_unique<Histogram>(name);
        it = metrics_.emplace(name, std::move(histogram)).first;
    }
    return static_cast<Histogram&>(*it->second);
}

Timer& MetricsRegistry::get_timer(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = metrics_.find(name);
    if (it == metrics_.end()) {
        auto timer = std::make_unique<Timer>(name);
        it = metrics_.emplace(name, std::move(timer)).first;
    }
    return static_cast<Timer&>(*it->second);
}

void MetricsRegistry::remove_metric(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    metrics_.erase(name);
}

void MetricsRegistry::reset_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [name, metric] : metrics_) {
        metric->reset();
    }
}

void MetricsRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    metrics_.clear();
}

std::vector<std::string> MetricsRegistry::metric_names() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(metrics_.size());
    for (const auto& [name, metric] : metrics_) {
        names.push_back(name);
    }
    return names;
}

const MetricValue* MetricsRegistry::get_metric(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = metrics_.find(name);
    return it != metrics_.end() ? it->second.get() : nullptr;
}

std::string MetricsRegistry::export_prometheus() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;

    for (const auto& [name, metric] : metrics_) {
        switch (metric->type()) {
            case MetricType::COUNTER: {
                auto* counter = static_cast<const Counter*>(metric.get());
                oss << "# HELP " << name << " counter metric\n";
                oss << "# TYPE " << name << " counter\n";
                oss << name << " " << counter->value() << "\n\n";
                break;
            }
            case MetricType::GAUGE: {
                auto* gauge = static_cast<const Gauge*>(metric.get());
                oss << "# HELP " << name << " gauge metric\n";
                oss << "# TYPE " << name << " gauge\n";
                oss << name << " " << gauge->value() << "\n\n";
                break;
            }
            case MetricType::HISTOGRAM: {
                auto* histogram = static_cast<const Histogram*>(metric.get());
                oss << "# HELP " << name << " histogram metric\n";
                oss << "# TYPE " << name << " histogram\n";
                oss << name << "_count " << histogram->count() << "\n";
                oss << name << "_sum " << histogram->sum() << "\n\n";
                break;
            }
            case MetricType::TIMER: {
                auto* timer = static_cast<const Timer*>(metric.get());
                oss << "# HELP " << name << " timer metric\n";
                oss << "# TYPE " << name << " histogram\n";
                oss << name << "_count " << timer->count() << "\n";
                auto total_seconds = std::chrono::duration<double>(timer->total_time()).count();
                oss << name << "_sum " << total_seconds << "\n\n";
                break;
            }
        }
    }

    return oss.str();
}

std::string MetricsRegistry::export_json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;

    oss << "{\n";
    oss << "  \"metrics\": {\n";

    bool first = true;
    for (const auto& [name, metric] : metrics_) {
        if (!first) {
            oss << ",\n";
        }
        first = false;

        oss << "    \"" << name << "\": " << metric->to_string();
    }

    oss << "\n  }\n";
    oss << "}\n";

    return oss.str();
}

} // namespace metrics
} // namespace langchain