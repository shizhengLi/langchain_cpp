#include "langchain/prompts/prompt_template.hpp"
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace langchain::prompts {

// PromptTemplate implementation
PromptTemplate::PromptTemplate(
    const std::string& template_str,
    const std::vector<std::string>& input_variables,
    const std::string& template_format,
    bool validate_template
) : template_str_(template_str), input_variables_(input_variables),
    template_format_(template_format), validate_template_(validate_template),
    variable_regex_(R"(\{([^}]+)\})") {

    if (input_variables_.empty()) {
        input_variables_ = extract_variables(template_str_);
    }

    if (validate_template_) {
        PromptTemplate::validate_template(template_str_, input_variables_);
    }
}

std::string PromptTemplate::format(const std::unordered_map<std::string, std::string>& variables) const {
    std::string result = template_str_;

    for (const auto& var : input_variables_) {
        std::string placeholder = "{" + var + "}";

        auto it = variables.find(var);
        if (it == variables.end()) {
            throw std::invalid_argument("Missing value for variable: " + var);
        }

        size_t pos = 0;
        while ((pos = result.find(placeholder, pos)) != std::string::npos) {
            result.replace(pos, placeholder.length(), it->second);
            pos += it->second.length();
        }
    }

    return result;
}

std::vector<std::string> PromptTemplate::extract_variables(const std::string& template_str) {
    std::vector<std::string> variables;
    std::regex variable_regex(R"(\{([^}]+)\})");
    std::smatch match;

    std::string str = template_str;
    while (std::regex_search(str, match, variable_regex)) {
        std::string var = match[1].str();
        if (std::find(variables.begin(), variables.end(), var) == variables.end()) {
            variables.push_back(var);
        }
        str = match.suffix();
    }

    return variables;
}

void PromptTemplate::validate_template(const std::string& template_str, const std::vector<std::string>& variables) {
    std::vector<std::string> found_vars = extract_variables(template_str);

    for (const auto& var : found_vars) {
        if (std::find(variables.begin(), variables.end(), var) == variables.end()) {
            throw std::invalid_argument("Variable found in template but not in input_variables: " + var);
        }
    }
}


// FewShotPromptTemplate implementation
FewShotPromptTemplate::FewShotPromptTemplate(
    const std::vector<std::unordered_map<std::string, std::string>>& examples,
    std::shared_ptr<BasePromptTemplate> example_prompt,
    const std::string& example_separator,
    const std::string& prefix,
    const std::string& suffix,
    const std::vector<std::string>& input_variables
) : examples_(examples), example_prompt_(example_prompt),
    example_separator_(example_separator), prefix_(prefix), suffix_(suffix),
    input_variables_(input_variables) {

    if (!example_prompt_) {
        throw std::invalid_argument("Example prompt template cannot be null");
    }
}

std::string FewShotPromptTemplate::format(const std::unordered_map<std::string, std::string>& variables) const {
    std::stringstream result;

    if (!prefix_.empty()) {
        result << prefix_ << "\n\n";
    }

    result << format_examples();

    if (!example_separator_.empty()) {
        result << example_separator_ << "\n\n";
    }

    // Format the actual input
    std::string formatted_input = example_prompt_->format(variables);
    result << formatted_input;

    if (!suffix_.empty()) {
        result << "\n\n" << suffix_;
    }

    return result.str();
}

std::string FewShotPromptTemplate::to_string() const {
    std::stringstream result;
    if (!prefix_.empty()) result << prefix_ << "\n\n";
    result << format_examples();
    if (!suffix_.empty()) result << "\n\n" << suffix_;
    return result.str();
}

void FewShotPromptTemplate::add_example(const std::unordered_map<std::string, std::string>& example) {
    examples_.push_back(example);
}

std::string FewShotPromptTemplate::format_examples() const {
    std::stringstream result;

    for (size_t i = 0; i < examples_.size(); ++i) {
        if (i > 0 && !example_separator_.empty()) {
            result << example_separator_ << "\n\n";
        }
        result << example_prompt_->format(examples_[i]);
    }

    return result.str();
}

// ChatPromptTemplate implementation
ChatPromptTemplate::ChatPromptTemplate(
    const std::vector<ChatMessage>& messages,
    const std::vector<std::string>& input_variables
) : messages_(messages), input_variables_(input_variables) {

    // Extract input variables from messages if not provided
    if (input_variables_.empty()) {
        for (const auto& message : messages_) {
            auto vars = PromptTemplate::extract_variables(message.content);
            for (const auto& var : vars) {
                if (std::find(input_variables_.begin(), input_variables_.end(), var) == input_variables_.end()) {
                    input_variables_.push_back(var);
                }
            }
        }
    }
}

std::string ChatPromptTemplate::format(const std::unordered_map<std::string, std::string>& variables) const {
    return format_generic();
}

std::string ChatPromptTemplate::to_string() const {
    return format_generic();
}

void ChatPromptTemplate::add_message(const ChatMessage& message) {
    messages_.push_back(message);

    // Extract new input variables
    auto vars = PromptTemplate::extract_variables(message.content);
    for (const auto& var : vars) {
        if (std::find(input_variables_.begin(), input_variables_.end(), var) == input_variables_.end()) {
            input_variables_.push_back(var);
        }
    }
}

std::string ChatPromptTemplate::format_openai() const {
    std::stringstream result;
    result << "{\n";
    result << "  \"messages\": [\n";

    for (size_t i = 0; i < messages_.size(); ++i) {
        const auto& msg = messages_[i];
        result << "    {\"role\": \"" << message_type_to_string(msg.type)
               << "\", \"content\": \"" << msg.content << "\"}";
        if (i < messages_.size() - 1) result << ",";
        result << "\n";
    }

    result << "  ]\n";
    result << "}";

    return result.str();
}

std::string ChatPromptTemplate::format_generic() const {
    std::stringstream result;

    for (const auto& message : messages_) {
        result << "[" << message_type_to_string(message.type) << "]: " << message.content << "\n\n";
    }

    return result.str();
}

std::string ChatPromptTemplate::format_message(
    const ChatMessage& message,
    const std::unordered_map<std::string, std::string>& variables
) const {
    std::string content = message.content;

    // Replace variables
    for (const auto& var : input_variables_) {
        std::string placeholder = "{" + var + "}";
        auto it = variables.find(var);
        if (it != variables.end()) {
            size_t pos = 0;
            while ((pos = content.find(placeholder, pos)) != std::string::npos) {
                content.replace(pos, placeholder.length(), it->second);
                pos += it->second.length();
            }
        }
    }

    return content;
}

std::string ChatPromptTemplate::message_type_to_string(ChatMessageType type) const {
    switch (type) {
        case ChatMessageType::SYSTEM: return "system";
        case ChatMessageType::USER: return "user";
        case ChatMessageType::ASSISTANT: return "assistant";
        case ChatMessageType::FUNCTION: return "function";
        case ChatMessageType::TOOL: return "tool";
        default: return "unknown";
    }
}

// PipelinePromptTemplate implementation
PipelinePromptTemplate::PipelinePromptTemplate(
    const std::vector<std::shared_ptr<BasePromptTemplate>>& pipeline,
    const std::vector<std::string>& input_variables
) : pipeline_(pipeline), input_variables_(input_variables) {

    if (pipeline_.empty()) {
        throw std::invalid_argument("Pipeline cannot be empty");
    }
}

std::string PipelinePromptTemplate::format(const std::unordered_map<std::string, std::string>& variables) const {
    std::unordered_map<std::string, std::string> current_vars = variables;

    for (size_t i = 0; i < pipeline_.size(); ++i) {
        current_vars = apply_template(i, current_vars);
    }

    // Return the final result
    return current_vars["final_result"];
}

std::string PipelinePromptTemplate::to_string() const {
    std::stringstream result;
    for (size_t i = 0; i < pipeline_.size(); ++i) {
        result << "Step " << (i + 1) << ":\n";
        result << pipeline_[i]->to_string();
        if (i < pipeline_.size() - 1) {
            result << "\n\n---\n\n";
        }
    }
    return result.str();
}

void PipelinePromptTemplate::add_template(std::shared_ptr<BasePromptTemplate> template_ptr) {
    pipeline_.push_back(template_ptr);
}

std::unordered_map<std::string, std::string> PipelinePromptTemplate::apply_template(
    size_t index,
    const std::unordered_map<std::string, std::string>& variables
) const {
    std::string result = pipeline_[index]->format(variables);

    std::unordered_map<std::string, std::string> new_vars = variables;
    if (index == pipeline_.size() - 1) {
        new_vars["final_result"] = result;
    } else {
        // Create intermediate variable name
        std::string intermediate_name = "step_" + std::to_string(index + 1) + "_result";
        new_vars[intermediate_name] = result;
    }

    return new_vars;
}

// Utility functions implementation
namespace utils {

std::shared_ptr<PromptTemplate> load_from_file(const std::string& file_path) {
    // This would typically read from file
    // For now, return a simple template
    return simple_template("Template from file: " + file_path);
}

std::shared_ptr<PromptTemplate> simple_template(const std::string& template_str) {
    return std::make_shared<PromptTemplate>(template_str);
}

std::shared_ptr<FewShotPromptTemplate> few_shot_template(
    const std::vector<std::unordered_map<std::string, std::string>>& examples,
    const std::string& example_template
) {
    auto example_prompt = std::make_shared<PromptTemplate>(example_template);
    return std::make_shared<FewShotPromptTemplate>(examples, example_prompt);
}

std::shared_ptr<ChatPromptTemplate> chat_template(const std::vector<ChatMessage>& messages) {
    return std::make_shared<ChatPromptTemplate>(messages);
}

void validate_input_variables(
    const std::vector<std::string>& template_vars,
    const std::unordered_map<std::string, std::string>& input
) {
    for (const auto& var : template_vars) {
        if (input.find(var) == input.end()) {
            throw std::invalid_argument("Missing required input variable: " + var);
        }
    }
}

std::string sanitize_template(const std::string& template_str) {
    // Basic sanitization - remove excessive whitespace
    std::string result = template_str;

    // Replace multiple consecutive newlines with single newline
    static const std::regex multiple_newlines(R"(\n{3,})");
    result = std::regex_replace(result, multiple_newlines, "\n\n");

    // Trim leading/trailing whitespace
    result.erase(0, result.find_first_not_of(" \t\n\r"));
    result.erase(result.find_last_not_of(" \t\n\r") + 1);

    return result;
}

} // namespace utils

} // namespace langchain::prompts