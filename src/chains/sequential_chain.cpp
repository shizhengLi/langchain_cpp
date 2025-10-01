#include "langchain/chains/sequential_chain.hpp"
#include "langchain/utils/logging.hpp"
#include <algorithm>

namespace langchain::chains {

SequentialChain::SequentialChain(const SequentialChainConfig& config)
    : BaseChain(config), sequential_config_(config) {

    sequential_config_.validate();
    LOG_INFO("SequentialChain initialized with return_all_outputs: " +
             std::to_string(sequential_config_.return_all_outputs));
}

void SequentialChain::add_chain(std::shared_ptr<BaseChain> chain) {
    if (!chain) {
        throw std::invalid_argument("Cannot add null chain to SequentialChain");
    }

    chains_.push_back(chain);
    LOG_DEBUG("Added chain to SequentialChain. Total chains: " + std::to_string(chains_.size()));
}

void SequentialChain::remove_chain(size_t index) {
    if (index >= chains_.size()) {
        throw std::out_of_range("Chain index out of range: " + std::to_string(index));
    }

    chains_.erase(chains_.begin() + index);
    LOG_DEBUG("Removed chain at index " + std::to_string(index) +
              ". Remaining chains: " + std::to_string(chains_.size()));
}

void SequentialChain::clear_chains() {
    chains_.clear();
    LOG_DEBUG("Cleared all chains from SequentialChain");
}

std::shared_ptr<BaseChain> SequentialChain::get_chain(size_t index) const {
    if (index >= chains_.size()) {
        throw std::out_of_range("Chain index out of range: " + std::to_string(index));
    }

    return chains_[index];
}

ChainOutput SequentialChain::run(const ChainInput& input) {
    LOG_DEBUG("SequentialChain executing with " + std::to_string(chains_.size()) + " chains");

    if (chains_.empty()) {
        return create_error_output("SequentialChain has no chains to execute");
    }

    if (!validate_input(input)) {
        return create_error_output("Input validation failed");
    }

    return measure_execution_time([&]() -> ChainOutput {
        try {
            ChainInput current_input = input;
            std::unordered_map<std::string, std::string> all_outputs;

            // Execute chains in sequence
            for (size_t i = 0; i < chains_.size(); ++i) {
                auto& chain = chains_[i];

                LOG_DEBUG("Executing chain " + std::to_string(i) +
                          " of type: " + chain->get_output_keys()[0]);

                auto output = chain->run(current_input);

                if (!output.success) {
                    std::string error_msg = "Chain " + std::to_string(i) + " failed: " +
                                          output.error_message.value_or("Unknown error");

                    if (sequential_config_.stop_on_error) {
                        LOG_ERROR("SequentialChain stopped due to error: " + error_msg);
                        return create_error_output(error_msg);
                    } else {
                        LOG_WARN("SequentialChain continuing despite error: " + error_msg);
                        continue;
                    }
                }

                // Merge outputs for next chain
                current_input = merge_outputs(output, input);

                // Store all outputs
                for (const auto& [key, value] : output.values) {
                    all_outputs[key] = value;
                }

                if (sequential_config_.verbose) {
                    LOG_DEBUG("Chain " + std::to_string(i) + " completed successfully");
                }
            }

            // Create final output
            std::unordered_map<std::string, std::string> final_values;

            if (sequential_config_.return_all_outputs) {
                final_values = all_outputs;
            } else {
                // Only return the output from the last chain
                auto last_chain_keys = chains_.back()->get_output_keys();
                for (const auto& key : last_chain_keys) {
                    auto it = all_outputs.find(key);
                    if (it != all_outputs.end()) {
                        final_values[key] = it->second;
                    }
                }
            }

            // Ensure we have the specified output key
            if (!final_values.empty() && final_values.find(sequential_config_.output_key) == final_values.end()) {
                // Use the first output value as the main output
                final_values[sequential_config_.output_key] = final_values.begin()->second;
            }

            auto result = create_success_output(final_values);

            if (sequential_config_.verbose) {
                LOG_INFO("SequentialChain execution completed. Output keys: " +
                         std::to_string(final_values.size()));
            }

            return result;

        } catch (const std::exception& e) {
            LOG_ERROR("SequentialChain execution error: " + std::string(e.what()));
            return create_error_output("SequentialChain execution failed: " + std::string(e.what()));
        }
    });
}

std::vector<std::string> SequentialChain::get_input_keys() const {
    if (chains_.empty()) {
        return {};
    }

    // Return input keys from the first chain
    return chains_[0]->get_input_keys();
}

std::vector<std::string> SequentialChain::get_output_keys() const {
    if (chains_.empty()) {
        return {sequential_config_.output_key};
    }

    std::vector<std::string> all_keys;

    if (sequential_config_.return_all_outputs) {
        // Collect output keys from all chains
        for (const auto& chain : chains_) {
            auto chain_keys = chain->get_output_keys();
            all_keys.insert(all_keys.end(), chain_keys.begin(), chain_keys.end());
        }

        // Remove duplicates
        std::sort(all_keys.begin(), all_keys.end());
        all_keys.erase(std::unique(all_keys.begin(), all_keys.end()), all_keys.end());
    } else {
        // Return output keys from the last chain
        all_keys = chains_.back()->get_output_keys();
    }

    // Ensure the output key is included
    if (std::find(all_keys.begin(), all_keys.end(), sequential_config_.output_key) == all_keys.end()) {
        all_keys.push_back(sequential_config_.output_key);
    }

    return all_keys;
}

ChainInput SequentialChain::merge_outputs(const ChainOutput& output, const ChainInput& original_input) const {
    ChainInput merged;

    // Start with original input
    merged.values = original_input.values;

    // Override/add output values
    for (const auto& [key, value] : output.values) {
        merged.values[key] = value;
    }

    return merged;
}

void SequentialChain::update_sequential_config(const SequentialChainConfig& new_config) {
    new_config.validate();
    sequential_config_ = new_config;
    LOG_INFO("SequentialChain configuration updated");
}

// SequentialChainFactory implementation

std::unique_ptr<BaseChain> SequentialChainFactory::create() const {
    SequentialChainConfig config;
    return std::make_unique<SequentialChain>(config);
}

std::unique_ptr<BaseChain> SequentialChainFactory::create(const ChainConfig& config) const {
    try {
        const auto& seq_config = dynamic_cast<const SequentialChainConfig&>(config);
        return std::make_unique<SequentialChain>(seq_config);
    } catch (const std::bad_cast&) {
        // If it's not a SequentialChainConfig, use default config
        SequentialChainConfig seq_config;
        seq_config.verbose = config.verbose;
        seq_config.timeout = config.timeout;
        seq_config.max_retries = config.max_retries;
        seq_config.return_intermediate_steps = config.return_intermediate_steps;
        return std::make_unique<SequentialChain>(seq_config);
    }
}

std::string SequentialChainFactory::get_chain_type() const {
    return "sequential";
}

} // namespace langchain::chains