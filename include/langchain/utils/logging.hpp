#pragma once

#include <string>
#include <fstream>
#include <mutex>
#include <atomic>
#include <chrono>
#include <sstream>
#include <iostream>
#include <memory>

namespace langchain::utils {

/**
 * @brief High-performance logging system with thread safety
 */
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3
};

/**
 * @brief Thread-safe logger implementation
 */
class Logger {
private:
    static std::unique_ptr<Logger> instance_;
    static std::mutex instance_mutex_;

    std::atomic<LogLevel> min_level_{LogLevel::INFO};
    std::unique_ptr<std::ofstream> file_stream_;
    std::mutex log_mutex_;
    bool console_output_ = true;
    std::atomic<uint64_t> total_messages_{0};

    Logger() = default;

public:
    /**
     * @brief Get singleton logger instance
     * @return Reference to logger instance
     */
    static Logger& get_instance() {
        std::lock_guard<std::mutex> lock(instance_mutex_);
        if (!instance_) {
            instance_ = std::unique_ptr<Logger>(new Logger());
        }
        return *instance_;
    }

    /**
     * @brief Set minimum log level
     * @param level Minimum level to log
     */
    void set_level(LogLevel level) {
        min_level_.store(level);
    }

    /**
     * @brief Enable/disable console output
     * @param enabled Whether to enable console output
     */
    void set_console_output(bool enabled) {
        console_output_ = enabled;
    }

    /**
     * @brief Set log file
     * @param filename Log file path
     */
    void set_log_file(const std::string& filename) {
        std::lock_guard<std::mutex> lock(log_mutex_);
        if (file_stream_) {
            file_stream_->close();
        }
        file_stream_ = std::make_unique<std::ofstream>(filename, std::ios::app);
    }

    /**
     * @brief Log a message
     * @param level Log level
     * @param message Message to log
     */
    void log(LogLevel level, const std::string& message) {
        if (level < min_level_.load()) {
            return;
        }

        total_messages_++;

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::ostringstream oss;
        oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        oss << " [" << level_to_string(level) << "] " << message;

        std::string log_line = oss.str();

        std::lock_guard<std::mutex> lock(log_mutex_);

        if (console_output_) {
            std::cout << log_line << std::endl;
        }

        if (file_stream_ && file_stream_->is_open()) {
            *file_stream_ << log_line << std::endl;
            file_stream_->flush();
        }
    }

    /**
     * @brief Get total number of logged messages
     * @return Total message count
     */
    uint64_t get_total_messages() const {
        return total_messages_.load();
    }

    /**
     * @brief Flush all log streams
     */
    void flush() {
        std::lock_guard<std::mutex> lock(log_mutex_);
        std::cout.flush();
        if (file_stream_) {
            file_stream_->flush();
        }
    }

private:
    std::string level_to_string(LogLevel level) const {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return "INFO ";
            case LogLevel::WARN:  return "WARN ";
            case LogLevel::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }
};

// Static member definitions
inline std::unique_ptr<Logger> Logger::instance_;
inline std::mutex Logger::instance_mutex_;

/**
 * @brief Convenience macros for logging
 */
#define LOG_DEBUG(message) \
    langchain::utils::Logger::get_instance().log(langchain::utils::LogLevel::DEBUG, message)

#define LOG_INFO(message) \
    langchain::utils::Logger::get_instance().log(langchain::utils::LogLevel::INFO, message)

#define LOG_WARN(message) \
    langchain::utils::Logger::get_instance().log(langchain::utils::LogLevel::WARN, message)

#define LOG_ERROR(message) \
    langchain::utils::Logger::get_instance().log(langchain::utils::LogLevel::ERROR, message)

/**
 * @brief RAII logger for function timing
 */
class FunctionLogger {
private:
    std::string function_name_;
    std::chrono::high_resolution_clock::time_point start_time_;

public:
    explicit FunctionLogger(const std::string& function_name)
        : function_name_(function_name),
          start_time_(std::chrono::high_resolution_clock::now()) {
        LOG_DEBUG("Entering function: " + function_name_);
    }

    ~FunctionLogger() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_);
        double elapsed_ms = duration.count() / 1000.0;

        std::ostringstream oss;
        oss << "Exiting function: " << function_name_ << " (took " << std::fixed
            << std::setprecision(2) << elapsed_ms << " ms)";
        LOG_DEBUG(oss.str());
    }
};

/**
 * @brief Macro for automatic function logging
 */
#define LOG_FUNCTION() langchain::utils::FunctionLogger _func_logger(__FUNCTION__)

} // namespace langchain::utils