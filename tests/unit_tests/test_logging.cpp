#include <catch2/catch_all.hpp>
#include "langchain/utils/logging.hpp"
#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>

using namespace langchain::utils;

TEST_CASE("Logger - Basic Operations", "[utils][logging]") {
    SECTION("Singleton access") {
        Logger& logger1 = Logger::get_instance();
        Logger& logger2 = Logger::get_instance();
        REQUIRE(&logger1 == &logger2);
    }

    SECTION("Log level setting") {
        Logger& logger = Logger::get_instance();

        logger.set_level(LogLevel::WARN);
        logger.log(LogLevel::DEBUG, "This should not appear");
        logger.log(LogLevel::ERROR, "This should appear");

        // Reset to INFO for other tests
        logger.set_level(LogLevel::INFO);
    }

    SECTION("Console output toggle") {
        Logger& logger = Logger::get_instance();

        logger.set_console_output(false);
        logger.log(LogLevel::INFO, "Silent message");

        logger.set_console_output(true);
        logger.log(LogLevel::INFO, "Visible message");
    }

    SECTION("Message counting") {
        Logger& logger = Logger::get_instance();
        uint64_t initial_count = logger.get_total_messages();

        logger.log(LogLevel::INFO, "Test message 1");
        logger.log(LogLevel::ERROR, "Test message 2");

        uint64_t final_count = logger.get_total_messages();
        REQUIRE(final_count == initial_count + 2);
    }
}

TEST_CASE("Logger - Log Levels", "[utils][logging]") {
    Logger& logger = Logger::get_instance();

    // Set to DEBUG to capture all levels
    logger.set_level(LogLevel::DEBUG);

    SECTION("All log levels") {
        uint64_t initial_count = logger.get_total_messages();

        logger.log(LogLevel::DEBUG, "Debug message");
        logger.log(LogLevel::INFO, "Info message");
        logger.log(LogLevel::WARN, "Warning message");
        logger.log(LogLevel::ERROR, "Error message");

        REQUIRE(logger.get_total_messages() == initial_count + 4);
    }

    // Reset to INFO
    logger.set_level(LogLevel::INFO);
}

TEST_CASE("Logger - File Logging", "[utils][logging]") {
    Logger& logger = Logger::get_instance();
    std::string test_file = "test_log.txt";

    // Clean up any existing test file
    std::remove(test_file.c_str());

    SECTION("File logging setup") {
        logger.set_log_file(test_file);
        logger.log(LogLevel::INFO, "Test file message");
        logger.flush();  // Ensure message is written

        // Check if file exists and contains our message
        std::ifstream file(test_file);
        REQUIRE(file.good());

        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        REQUIRE_FALSE(content.empty());
        REQUIRE(content.find("Test file message") != std::string::npos);

        file.close();
    }

    // Clean up
    std::remove(test_file.c_str());
}

TEST_CASE("Logger - Thread Safety", "[utils][logging][thread_safety]") {
    Logger& logger = Logger::get_instance();
    logger.set_level(LogLevel::INFO);

    SECTION("Concurrent logging") {
        uint64_t initial_count = logger.get_total_messages();
        const int num_threads = 4;
        const int messages_per_thread = 10;

        std::vector<std::thread> threads;

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&logger, t, messages_per_thread]() {
                for (int i = 0; i < messages_per_thread; ++i) {
                    logger.log(LogLevel::INFO, "Thread " + std::to_string(t) + " message " + std::to_string(i));
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        uint64_t final_count = logger.get_total_messages();
        REQUIRE(final_count == initial_count + (num_threads * messages_per_thread));
    }
}

TEST_CASE("Logger - Level Filtering", "[utils][logging]") {
    Logger& logger = Logger::get_instance();
    uint64_t initial_count = logger.get_total_messages();

    SECTION("INFO level filtering") {
        logger.set_level(LogLevel::INFO);

        logger.log(LogLevel::DEBUG, "Debug message - should be filtered");
        logger.log(LogLevel::INFO, "Info message - should pass");
        logger.log(LogLevel::WARN, "Warning message - should pass");
        logger.log(LogLevel::ERROR, "Error message - should pass");

        // Only 3 messages should pass (INFO, WARN, ERROR)
        REQUIRE(logger.get_total_messages() == initial_count + 3);
    }

    SECTION("ERROR level filtering") {
        logger.set_level(LogLevel::ERROR);
        initial_count = logger.get_total_messages();

        logger.log(LogLevel::DEBUG, "Debug message - should be filtered");
        logger.log(LogLevel::INFO, "Info message - should be filtered");
        logger.log(LogLevel::WARN, "Warning message - should be filtered");
        logger.log(LogLevel::ERROR, "Error message - should pass");

        // Only 1 message should pass (ERROR)
        REQUIRE(logger.get_total_messages() == initial_count + 1);
    }

    // Reset to INFO
    logger.set_level(LogLevel::INFO);
}

TEST_CASE("Logging Macros", "[utils][logging][macros]") {
    Logger& logger = Logger::get_instance();
    logger.set_level(LogLevel::DEBUG);

    SECTION("LOG_DEBUG macro") {
        uint64_t initial_count = logger.get_total_messages();
        LOG_DEBUG("Debug macro message");
        REQUIRE(logger.get_total_messages() == initial_count + 1);
    }

    SECTION("LOG_INFO macro") {
        uint64_t initial_count = logger.get_total_messages();
        LOG_INFO("Info macro message");
        REQUIRE(logger.get_total_messages() == initial_count + 1);
    }

    SECTION("LOG_WARN macro") {
        uint64_t initial_count = logger.get_total_messages();
        LOG_WARN("Warning macro message");
        REQUIRE(logger.get_total_messages() == initial_count + 1);
    }

    SECTION("LOG_ERROR macro") {
        uint64_t initial_count = logger.get_total_messages();
        LOG_ERROR("Error macro message");
        REQUIRE(logger.get_total_messages() == initial_count + 1);
    }

    logger.set_level(LogLevel::INFO);
}

TEST_CASE("FunctionLogger - RAII Logging", "[utils][logging][function_logger]") {
    Logger& logger = Logger::get_instance();
    logger.set_level(LogLevel::DEBUG);

    uint64_t initial_count = logger.get_total_messages();

    SECTION("Basic function logging") {
        {
            FunctionLogger func_logger("test_function");
            // Do some work
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Should have logged both entry and exit
        REQUIRE(logger.get_total_messages() == initial_count + 2);
    }

    SECTION("Function logger with timing") {
        auto start = std::chrono::high_resolution_clock::now();

        {
            FunctionLogger func_logger("timed_function");
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // The function should have taken at least 5ms
        REQUIRE(duration.count() >= 5);
    }
}

TEST_CASE("LOG_FUNCTION macro", "[utils][logging][function_logger]") {
    Logger& logger = Logger::get_instance();
    logger.set_level(LogLevel::DEBUG);

    uint64_t initial_count = logger.get_total_messages();

    SECTION("Function macro") {
        auto test_function = [&logger, initial_count]() -> int {
            LOG_FUNCTION();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            return 42;
        };

        int result = test_function();

        REQUIRE(result == 42);
        // Should have logged entry and exit
        REQUIRE(logger.get_total_messages() == initial_count + 2);
    }

    logger.set_level(LogLevel::INFO);
}

TEST_CASE("Logger Performance", "[utils][logging][performance]") {
    Logger& logger = Logger::get_instance();
    logger.set_level(LogLevel::INFO);

    SECTION("High-frequency logging") {
        uint64_t initial_count = logger.get_total_messages();
        const int num_messages = 1000;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_messages; ++i) {
            LOG_INFO("Performance test message " + std::to_string(i));
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        uint64_t final_count = logger.get_total_messages();
        REQUIRE(final_count == initial_count + num_messages);

        // Should complete within reasonable time (less than 1 second for 1000 messages)
        REQUIRE(duration.count() < 1000);

        // Calculate messages per second
        double messages_per_second = num_messages * 1000.0 / duration.count();
        REQUIRE(messages_per_second > 100);  // Should handle at least 100 msg/sec
    }
}

TEST_CASE("Logger Error Handling", "[utils][logging][error_handling]") {
    Logger& logger = Logger::get_instance();

    SECTION("Invalid log file path") {
        // Try to set an invalid file path (directory that doesn't exist)
        logger.set_log_file("/nonexistent/directory/test.log");

        // Should not crash, but may fail silently or handle gracefully
        LOG_INFO("Message with invalid file path");

        // Reset to console logging
        logger.set_console_output(true);
    }

    SECTION("File permission issues") {
        // This test may not work on all systems, so we'll just ensure it doesn't crash
        std::string restricted_path = "/root/test_log.txt";  // Likely inaccessible

        try {
            logger.set_log_file(restricted_path);
            LOG_INFO("Test message");
        } catch (...) {
            // Any exceptions should be handled gracefully
        }

        // Reset
        logger.set_console_output(true);
    }
}