#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <regex>
#include <stdexcept>

namespace langchain::prompts {

/**
 * @brief Base class for all prompt templates
 */
class BasePromptTemplate {
public:
    virtual ~BasePromptTemplate() = default;
    virtual std::string format(const std::unordered_map<std::string, std::string>& variables) const = 0;
    virtual std::vector<std::string> get_input_variables() const = 0;
    virtual std::string to_string() const = 0;
};

/**
 * @brief Simple string prompt template with variable substitution
 */
class PromptTemplate : public BasePromptTemplate {
private:
    std::string template_str_;
    std::vector<std::string> input_variables_;
    std::string template_format_{"f-string"};  // f-string, jinja2, etc.
    bool validate_template_{true};

public:
    PromptTemplate(
        const std::string& template_str,
        const std::vector<std::string>& input_variables = {},
        const std::string& template_format = "f-string",
        bool validate_template = true
    );

    std::string format(const std::unordered_map<std::string, std::string>& variables) const override;
    std::vector<std::string> get_input_variables() const override { return input_variables_; }
    std::string to_string() const override { return template_str_; }

    const std::string& template_string() const { return template_str_; }
    const std::string& template_format() const { return template_format_; }

    // Extract variables from template string
    static std::vector<std::string> extract_variables(const std::string& template_str);
    // Validate template has valid variable syntax
    static void validate_template(const std::string& template_str, const std::vector<std::string>& variables);

private:
    mutable std::regex variable_regex_;
};

/**
 * @brief Few-shot prompt template with examples
 */
class FewShotPromptTemplate : public BasePromptTemplate {
private:
    std::vector<std::unordered_map<std::string, std::string>> examples_;
    std::shared_ptr<BasePromptTemplate> example_prompt_;
    std::string example_separator_{"\n\n"};
    std::string prefix_{};
    std::string suffix_{};
    std::vector<std::string> input_variables_;

public:
    FewShotPromptTemplate(
        const std::vector<std::unordered_map<std::string, std::string>>& examples,
        std::shared_ptr<BasePromptTemplate> example_prompt,
        const std::string& example_separator = "\n\n",
        const std::string& prefix = "",
        const std::string& suffix = "",
        const std::vector<std::string>& input_variables = {}
    );

    std::string format(const std::unordered_map<std::string, std::string>& variables) const override;
    std::vector<std::string> get_input_variables() const override { return input_variables_; }
    std::string to_string() const override;

    void add_example(const std::unordered_map<std::string, std::string>& example);
    void clear_examples() { examples_.clear(); }
    size_t example_count() const { return examples_.size(); }

private:
    std::string format_examples() const;
};

/**
 * @brief Chat message prompt template for conversational prompts
 */
enum class ChatMessageType {
    SYSTEM,
    USER,
    ASSISTANT,
    FUNCTION,
    TOOL
};

struct ChatMessage {
    ChatMessageType type;
    std::string content;
    std::string name;  // Optional, for function/tool messages

    ChatMessage(ChatMessageType t, const std::string& c, const std::string& n = "")
        : type(t), content(c), name(n) {}
};

class ChatPromptTemplate : public BasePromptTemplate {
private:
    std::vector<ChatMessage> messages_;
    std::vector<std::string> input_variables_;

public:
    ChatPromptTemplate(
        const std::vector<ChatMessage>& messages,
        const std::vector<std::string>& input_variables = {}
    );

    std::string format(const std::unordered_map<std::string, std::string>& variables) const override;
    std::vector<std::string> get_input_variables() const override { return input_variables_; }
    std::string to_string() const override;

    void add_message(const ChatMessage& message);
    void clear_messages() { messages_.clear(); }
    const std::vector<ChatMessage>& messages() const { return messages_; }

    // Format for different chat models
    std::string format_openai() const;
    std::string format_generic() const;

private:
    std::string format_message(const ChatMessage& message, const std::unordered_map<std::string, std::string>& variables) const;
    std::string message_type_to_string(ChatMessageType type) const;
};

/**
 * @brief Pipeline prompt template that chains multiple templates together
 */
class PipelinePromptTemplate : public BasePromptTemplate {
private:
    std::vector<std::shared_ptr<BasePromptTemplate>> pipeline_;
    std::vector<std::string> input_variables_;

public:
    PipelinePromptTemplate(
        const std::vector<std::shared_ptr<BasePromptTemplate>>& pipeline,
        const std::vector<std::string>& input_variables = {}
    );

    std::string format(const std::unordered_map<std::string, std::string>& variables) const override;
    std::vector<std::string> get_input_variables() const override { return input_variables_; }
    std::string to_string() const override;

    void add_template(std::shared_ptr<BasePromptTemplate> template_ptr);
    void clear_pipeline() { pipeline_.clear(); }

private:
    std::unordered_map<std::string, std::string> apply_template(
        size_t index,
        const std::unordered_map<std::string, std::string>& variables
    ) const;
};

/**
 * @brief Utility functions for prompt template creation and manipulation
 */
namespace utils {
    // Load prompt template from file
    std::shared_ptr<PromptTemplate> load_from_file(const std::string& file_path);

    // Create simple prompt template
    std::shared_ptr<PromptTemplate> simple_template(const std::string& template_str);

    // Create few-shot template from examples
    std::shared_ptr<FewShotPromptTemplate> few_shot_template(
        const std::vector<std::unordered_map<std::string, std::string>>& examples,
        const std::string& example_template
    );

    // Create chat template from messages
    std::shared_ptr<ChatPromptTemplate> chat_template(const std::vector<ChatMessage>& messages);

    // Validate template variables are present in input
    void validate_input_variables(
        const std::vector<std::string>& template_vars,
        const std::unordered_map<std::string, std::string>& input
    );

    // Sanitize template string
    std::string sanitize_template(const std::string& template_str);
}

} // namespace langchain::prompts