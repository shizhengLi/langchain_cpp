#pragma once

#include "base_chain.hpp"
#include "../llm/base_llm.hpp"
#include <string>
#include <vector>
#include <memory>

namespace langchain::chains {

/**
 * @brief Configuration for LLM Chain
 */
struct LLMChainConfig : public ChainConfig {
    std::string prompt_template;
    std::string input_key = "input";
    std::string output_key = "output";
    bool strip_whitespace = true;
    std::vector<std::string> stop_sequences;

    void validate() const override {
        ChainConfig::validate();

        if (prompt_template.empty()) {
            throw std::invalid_argument("Prompt template cannot be empty");
        }

        if (input_key.empty()) {
            throw std::invalid_argument("Input key cannot be empty");
        }

        if (output_key.empty()) {
            throw std::invalid_argument("Output key cannot be empty");
        }
    }
};

/**
 * @brief Chain that uses an LLM to process input
 */
class LLMChain : public BaseChain {
public:
    LLMChain(std::shared_ptr<llm::BaseLLM> llm, const LLMChainConfig& config = LLMChainConfig{});

    // BaseChain interface
    ChainOutput run(const ChainInput& input) override;
    std::vector<std::string> get_input_keys() const override;
    std::vector<std::string> get_output_keys() const override;

    /**
     * @brief Apply the prompt template with input values
     */
    std::string apply_prompt_template(const ChainInput& input) const;

    /**
     * @brief Format prompt with variables
     */
    std::string format_prompt(const std::string& template_str, const ChainInput& input) const;

    /**
     * @brief Get the LLM instance
     */
    std::shared_ptr<llm::BaseLLM> get_llm() const { return llm_; }

    /**
     * @brief Get LLM chain configuration
     */
    const LLMChainConfig& get_llm_config() const { return llm_config_; }

    /**
     * @brief Update LLM chain configuration
     */
    void update_llm_config(const LLMChainConfig& new_config);

private:
    std::shared_ptr<llm::BaseLLM> llm_;
    LLMChainConfig llm_config_;

    /**
     * @brief Extract variables from template
     */
    std::vector<std::string> extract_variables(const std::string& template_str) const;

    /**
     * @brief Replace variables in template with values
     */
    std::string replace_variables(const std::string& template_str, const ChainInput& input) const;
};

/**
 * @brief Factory for creating LLM chains
 */
class LLMChainFactory : public ChainFactory {
public:
    explicit LLMChainFactory(std::shared_ptr<llm::BaseLLM> llm);

    std::unique_ptr<BaseChain> create() const override;
    std::unique_ptr<BaseChain> create(const ChainConfig& config) const override;
    std::string get_chain_type() const override;

private:
    std::shared_ptr<llm::BaseLLM> llm_;
};

} // namespace langchain::chains