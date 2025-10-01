#pragma once

#include "base_chain.hpp"
#include <vector>
#include <memory>
#include <string>

namespace langchain::chains {

/**
 * @brief Configuration for Sequential Chain
 */
struct SequentialChainConfig : public ChainConfig {
    bool return_all_outputs = false;
    std::string output_key = "output";
    bool stop_on_error = true;

    void validate() const override {
        ChainConfig::validate();

        if (output_key.empty()) {
            throw std::invalid_argument("Output key cannot be empty");
        }
    }
};

/**
 * @brief Chain that executes multiple chains in sequence
 */
class SequentialChain : public BaseChain {
public:
    explicit SequentialChain(const SequentialChainConfig& config = SequentialChainConfig{});

    // Chain management
    void add_chain(std::shared_ptr<BaseChain> chain);
    void remove_chain(size_t index);
    void clear_chains();
    size_t get_chain_count() const { return chains_.size(); }

    // BaseChain interface
    ChainOutput run(const ChainInput& input) override;
    std::vector<std::string> get_input_keys() const override;
    std::vector<std::string> get_output_keys() const override;

    /**
     * @brief Get chain at specific index
     */
    std::shared_ptr<BaseChain> get_chain(size_t index) const;

    /**
     * @brief Get sequential chain configuration
     */
    const SequentialChainConfig& get_sequential_config() const { return sequential_config_; }

    /**
     * @brief Update sequential chain configuration
     */
    void update_sequential_config(const SequentialChainConfig& new_config);

private:
    std::vector<std::shared_ptr<BaseChain>> chains_;
    SequentialChainConfig sequential_config_;

    /**
     * @brief Merge outputs from one chain as inputs to the next
     */
    ChainInput merge_outputs(const ChainOutput& output, const ChainInput& original_input) const;
};

/**
 * @brief Factory for creating Sequential chains
 */
class SequentialChainFactory : public ChainFactory {
public:
    std::unique_ptr<BaseChain> create() const override;
    std::unique_ptr<BaseChain> create(const ChainConfig& config) const override;
    std::string get_chain_type() const override;
};

} // namespace langchain::chains