#include "langchain/chains/llm_chain.hpp"
#include "langchain/utils/logging.hpp"
#include <regex>
#include <sstream>

namespace langchain::chains {

LLMChain::LLMChain(std::shared_ptr<llm::BaseLLM> llm, const LLMChainConfig& config)
    : BaseChain(config), llm_(std::move(llm)), llm_config_(config) {

    if (!llm_) {
        throw std::invalid_argument("LLM instance cannot be null");
    }

    llm_config_.validate();
    LOG_INFO("LLMChain initialized with prompt template: " + llm_config_.prompt_template.substr(0, 50) + "...");
}

ChainOutput LLMChain::run(const ChainInput& input) {
    LOG_DEBUG("LLMChain executing with input keys: " + std::to_string(input.values.size()));

    if (!validate_input(input)) {
        return create_error_output("Missing required input key: " + llm_config_.input_key);
    }

    return measure_execution_time([&]() -> ChainOutput {
        try {
            // Apply prompt template
            std::string prompt = apply_prompt_template(input);
            LOG_DEBUG("Formatted prompt: " + prompt.substr(0, 100) + "...");

            // Configure LLM
            llm::LLMConfig llm_config;
            llm_config.model = "gpt-3.5-turbo"; // Default model
            llm_config.stop_sequences = llm_config_.stop_sequences;

            // Call LLM
            auto llm_response = llm_->complete(prompt, llm_config);

            if (!llm_response.success) {
                std::string error_msg = "LLM call failed: ";
                if (!llm_response.error_message.empty()) {
                    error_msg += llm_response.error_message;
                } else {
                    error_msg += "Unknown error";
                }
                return create_error_output(error_msg);
            }

            // Process response
            std::string output_text = llm_response.content;
            if (llm_config_.strip_whitespace) {
                output_text = std::regex_replace(output_text, std::regex("^\\s+|\\s+$"), "");
            }

            // Create output
            std::unordered_map<std::string, std::string> output_values;
            output_values[llm_config_.output_key] = output_text;

            if (llm_config_.return_intermediate_steps) {
                output_values["prompt"] = prompt;
                output_values["llm_response"] = llm_response.content;
            }

            auto result = create_success_output(output_values);

            if (llm_config_.verbose) {
                LOG_INFO("LLMChain execution successful. Output length: " + std::to_string(output_text.length()));
            }

            return result;

        } catch (const std::exception& e) {
            LOG_ERROR("LLMChain execution error: " + std::string(e.what()));
            return create_error_output("LLMChain execution failed: " + std::string(e.what()));
        }
    });
}

std::vector<std::string> LLMChain::get_input_keys() const {
    auto variables = extract_variables(llm_config_.prompt_template);

    // Add explicit input key if it's not already in variables
    if (std::find(variables.begin(), variables.end(), llm_config_.input_key) == variables.end()) {
        variables.push_back(llm_config_.input_key);
    }

    return variables;
}

std::vector<std::string> LLMChain::get_output_keys() const {
    std::vector<std::string> keys = {llm_config_.output_key};

    if (llm_config_.return_intermediate_steps) {
        keys.push_back("prompt");
        keys.push_back("llm_response");
    }

    return keys;
}

std::string LLMChain::apply_prompt_template(const ChainInput& input) const {
    return format_prompt(llm_config_.prompt_template, input);
}

std::string LLMChain::format_prompt(const std::string& template_str, const ChainInput& input) const {
    auto variables = extract_variables(template_str);

    std::string result = template_str;

    // Replace all variables
    for (const auto& var : variables) {
        std::string placeholder = "{" + var + "}";
        std::string value = input.get(var, "");

        // Simple string replacement
        size_t pos = 0;
        while ((pos = result.find(placeholder, pos)) != std::string::npos) {
            result.replace(pos, placeholder.length(), value);
            pos += value.length();
        }
    }

    return result;
}

std::vector<std::string> LLMChain::extract_variables(const std::string& template_str) const {
    std::vector<std::string> variables;
    std::regex pattern(R"(\{([^}]+)\})");
    std::smatch matches;

    std::string::const_iterator search_start(template_str.begin());
    std::set<std::string> unique_vars; // Use set to avoid duplicates

    while (std::regex_search(search_start, template_str.end(), matches, pattern)) {
        if (matches.size() > 1) {
            unique_vars.insert(matches[1].str());
        }
        search_start = matches.suffix().first;
    }

    for (const auto& var : unique_vars) {
        variables.push_back(var);
    }

    return variables;
}

std::string LLMChain::replace_variables(const std::string& template_str, const ChainInput& input) const {
    return format_prompt(template_str, input);
}

void LLMChain::update_llm_config(const LLMChainConfig& new_config) {
    new_config.validate();
    llm_config_ = new_config;
    LOG_INFO("LLMChain configuration updated");
}

// LLMChainFactory implementation

LLMChainFactory::LLMChainFactory(std::shared_ptr<llm::BaseLLM> llm)
    : llm_(std::move(llm)) {

    if (!llm_) {
        throw std::invalid_argument("LLM instance cannot be null");
    }
}

std::unique_ptr<BaseChain> LLMChainFactory::create() const {
    LLMChainConfig config;
    config.prompt_template = "Input: {input}\nOutput:";
    return std::make_unique<LLMChain>(llm_, config);
}

std::unique_ptr<BaseChain> LLMChainFactory::create(const ChainConfig& config) const {
    try {
        const auto& llm_config = dynamic_cast<const LLMChainConfig&>(config);
        return std::make_unique<LLMChain>(llm_, llm_config);
    } catch (const std::bad_cast&) {
        // If it's not an LLMChainConfig, use default LLM config
        LLMChainConfig llm_config;
        llm_config.verbose = config.verbose;
        llm_config.timeout = config.timeout;
        llm_config.max_retries = config.max_retries;
        llm_config.return_intermediate_steps = config.return_intermediate_steps;
        return std::make_unique<LLMChain>(llm_, llm_config);
    }
}

std::string LLMChainFactory::get_chain_type() const {
    return "llm";
}

} // namespace langchain::chains