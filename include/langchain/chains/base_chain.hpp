#pragma once

#include "../core/base.hpp"
#include "../llm/base_llm.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <optional>
#include <chrono>
#include <future>
#include <set>

namespace langchain::chains {

/**
 * @brief Input data for chain execution
 */
struct ChainInput {
    std::unordered_map<std::string, std::string> values;

    // Convenience accessors
    std::string get(const std::string& key, const std::string& default_value = "") const {
        auto it = values.find(key);
        return (it != values.end()) ? it->second : default_value;
    }

    void set(const std::string& key, const std::string& value) {
        values[key] = value;
    }

    bool has(const std::string& key) const {
        return values.find(key) != values.end();
    }
};

/**
 * @brief Output data from chain execution
 */
struct ChainOutput {
    std::unordered_map<std::string, std::string> values;
    bool success = true;
    std::optional<std::string> error_message;
    std::optional<std::chrono::milliseconds> execution_time;

    // Convenience accessors
    std::string get(const std::string& key, const std::string& default_value = "") const {
        auto it = values.find(key);
        return (it != values.end()) ? it->second : default_value;
    }

    void set(const std::string& key, const std::string& value) {
        values[key] = value;
    }

    bool has(const std::string& key) const {
        return values.find(key) != values.end();
    }
};

/**
 * @brief Chain configuration base class
 */
struct ChainConfig {
    bool verbose = false;
    std::optional<std::chrono::milliseconds> timeout;
    std::optional<size_t> max_retries;
    bool return_intermediate_steps = false;

    virtual void validate() const {
        if (timeout && timeout->count() <= 0) {
            throw std::invalid_argument("Timeout must be positive");
        }
        if (max_retries && *max_retries > 10) {
            throw std::invalid_argument("Max retries should not exceed 10");
        }
    }

    virtual ~ChainConfig() = default;
};

/**
 * @brief Abstract base class for all chains
 */
class BaseChain {
public:
    explicit BaseChain(const ChainConfig& config = ChainConfig{}) : config_(config) {
        config_.validate();
    }

    virtual ~BaseChain() = default;

    /**
     * @brief Execute the chain with given input
     */
    virtual ChainOutput run(const ChainInput& input) = 0;

    /**
     * @brief Execute the chain asynchronously (future enhancement)
     */
    virtual std::future<ChainOutput> run_async(const ChainInput& input) {
        return std::async(std::launch::async, [this, input]() {
            return this->run(input);
        });
    }

    /**
     * @brief Get input keys required by this chain
     */
    virtual std::vector<std::string> get_input_keys() const = 0;

    /**
     * @brief Get output keys produced by this chain
     */
    virtual std::vector<std::string> get_output_keys() const = 0;

    /**
     * @brief Validate that input contains all required keys
     */
    virtual bool validate_input(const ChainInput& input) const {
        auto required_keys = get_input_keys();
        for (const auto& key : required_keys) {
            if (!input.has(key)) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Get chain configuration
     */
    const ChainConfig& get_config() const { return config_; }

    /**
     * @brief Update chain configuration
     */
    virtual void update_config(const ChainConfig& new_config) {
        new_config.validate();
        config_ = new_config;
    }

protected:
    ChainConfig config_;

    /**
     * @brief Create error output
     */
    ChainOutput create_error_output(const std::string& error_message) const {
        ChainOutput output;
        output.success = false;
        output.error_message = error_message;
        return output;
    }

    /**
     * @brief Create success output
     */
    ChainOutput create_success_output(const std::unordered_map<std::string, std::string>& values) const {
        ChainOutput output;
        output.success = true;
        output.values = values;
        return output;
    }

    /**
     * @brief Measure execution time
     */
    template<typename Func>
    auto measure_execution_time(Func&& func) const -> decltype(func()) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = func();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // If result is ChainOutput, set execution time
        if constexpr (std::is_same_v<decltype(result), ChainOutput>) {
            result.execution_time = duration;
        }

        return result;
    }
};

/**
 * @brief Chain factory interface
 */
class ChainFactory {
public:
    virtual ~ChainFactory() = default;

    /**
     * @brief Create a chain instance with default configuration
     */
    virtual std::unique_ptr<BaseChain> create() const = 0;

    /**
     * @brief Create a chain instance with custom configuration
     */
    virtual std::unique_ptr<BaseChain> create(const ChainConfig& config) const = 0;

    /**
     * @brief Get chain type identifier
     */
    virtual std::string get_chain_type() const = 0;

    /**
     * @brief Check if this factory supports the given chain type
     */
    virtual bool supports_chain_type(const std::string& chain_type) const {
        return chain_type == get_chain_type();
    }
};

/**
 * @brief Chain registry for managing multiple chain types
 */
class ChainRegistry {
public:
    static ChainRegistry& instance() {
        static ChainRegistry instance;
        return instance;
    }

    /**
     * @brief Register a chain factory
     */
    void register_factory(std::unique_ptr<ChainFactory> factory) {
        if (factory) {
            factories_[factory->get_chain_type()] = std::move(factory);
        }
    }

    /**
     * @brief Create a chain by type
     */
    std::unique_ptr<BaseChain> create(const std::string& chain_type) const {
        auto it = factories_.find(chain_type);
        if (it != factories_.end()) {
            return it->second->create();
        }
        throw std::invalid_argument("Unknown chain type: " + chain_type);
    }

    /**
     * @brief Create a chain by type with configuration
     */
    std::unique_ptr<BaseChain> create(const std::string& chain_type, const ChainConfig& config) const {
        auto it = factories_.find(chain_type);
        if (it != factories_.end()) {
            return it->second->create(config);
        }
        throw std::invalid_argument("Unknown chain type: " + chain_type);
    }

    /**
     * @brief Get available chain types
     */
    std::vector<std::string> get_available_types() const {
        std::vector<std::string> types;
        types.reserve(factories_.size());

        for (const auto& [type, factory] : factories_) {
            types.push_back(type);
        }

        return types;
    }

    /**
     * @brief Check if a chain type is supported
     */
    bool supports_chain_type(const std::string& chain_type) const {
        return factories_.find(chain_type) != factories_.end();
    }

private:
    std::unordered_map<std::string, std::unique_ptr<ChainFactory>> factories_;
};

} // namespace langchain::chains