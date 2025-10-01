#pragma once

#include "../llm/base_llm.hpp"
#include "../memory/memory.hpp"
#include "../chains/base_chain.hpp"
#include "../prompts/prompt_template.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <optional>

namespace langchain::agents {

/**
 * @brief Base class for all tools that agents can use
 */
class BaseTool {
public:
    virtual ~BaseTool() = default;

    /**
     * @brief Name of the tool
     */
    virtual std::string name() const = 0;

    /**
     * @brief Description of what the tool does
     */
    virtual std::string description() const = 0;

    /**
     * @brief Input schema for the tool (JSON schema format)
     */
    virtual std::string input_schema() const = 0;

    /**
     * @brief Execute the tool with given input
     */
    virtual std::string run(const std::string& input) = 0;

    /**
     * @brief Whether the tool requires async execution
     */
    virtual bool is_async() const { return false; }

    /**
     * @brief Tool category
     */
    virtual std::string category() const { return "general"; }
};

/**
 * @brief Simple function-based tool
 */
class FunctionTool : public BaseTool {
private:
    std::string name_;
    std::string description_;
    std::string input_schema_;
    std::function<std::string(const std::string&)> function_;

public:
    FunctionTool(
        const std::string& name,
        const std::string& description,
        const std::string& input_schema,
        std::function<std::string(const std::string&)> func
    );

    std::string name() const override { return name_; }
    std::string description() const override { return description_; }
    std::string input_schema() const override { return input_schema_; }
    std::string run(const std::string& input) override;
};

/**
 * @brief Agent action result
 */
struct ActionResult {
    std::string tool_name;
    std::string tool_input;
    std::string output;
    bool success{true};
    std::chrono::milliseconds execution_time{0};

    ActionResult(const std::string& tool, const std::string& input, const std::string& result)
        : tool_name(tool), tool_input(input), output(result) {}
};

/**
 * @brief Agent step in reasoning chain
 */
struct AgentStep {
    enum class StepType {
        THOUGHT,
        ACTION,
        OBSERVATION,
        FINAL_ANSWER
    };

    StepType type;
    std::string content;
    std::optional<ActionResult> action_result;
    std::chrono::system_clock::time_point timestamp;

    AgentStep(StepType t, const std::string& c)
        : type(t), content(c), timestamp(std::chrono::system_clock::now()) {}

    AgentStep(StepType t, const std::string& c, const ActionResult& result)
        : type(t), content(c), action_result(result), timestamp(std::chrono::system_clock::now()) {}
};

/**
 * @brief Agent configuration
 */
struct AgentConfig {
    size_t max_iterations{10};
    std::chrono::seconds max_execution_time{300};
    bool verbose{false};
    bool return_intermediate_steps{false};
    std::string stopping_condition{"Final Answer:"};
    bool handle_parsing_errors{true};
    bool require_confirmation{false};  // For dangerous operations

    void validate() const {
        if (max_iterations == 0) {
            throw std::invalid_argument("max_iterations must be greater than 0");
        }
        if (max_execution_time.count() <= 0) {
            throw std::invalid_argument("max_execution_time must be positive");
        }
    }
};

/**
 * @brief Base class for all agents
 */
class BaseAgent {
protected:
    std::shared_ptr<llm::BaseLLM> llm_;
    std::vector<std::shared_ptr<BaseTool>> tools_;
    AgentConfig config_;
    std::unique_ptr<memory::BaseMemory> memory_;

public:
    BaseAgent(
        std::shared_ptr<llm::BaseLLM> llm,
        std::vector<std::shared_ptr<BaseTool>> tools = {},
        const AgentConfig& config = AgentConfig{}
    );

    virtual ~BaseAgent() = default;

    /**
     * @brief Execute agent with given input
     */
    virtual std::string run(const std::string& input) = 0;

    /**
     * @brief Execute agent with intermediate steps
     */
    virtual std::pair<std::string, std::vector<AgentStep>> run_with_steps(const std::string& input) = 0;

    /**
     * @brief Add a tool to the agent
     */
    void add_tool(std::shared_ptr<BaseTool> tool);

    /**
     * @brief Remove a tool from the agent
     */
    void remove_tool(const std::string& tool_name);

    /**
     * @brief Get available tools
     */
    const std::vector<std::shared_ptr<BaseTool>>& get_tools() const { return tools_; }

    /**
     * @brief Set memory for the agent
     */
    void set_memory(std::unique_ptr<memory::BaseMemory> memory);

    /**
     * @brief Get agent configuration
     */
    const AgentConfig& get_config() const { return config_; }

    /**
     * @brief Get tool information as formatted string
     */
    std::string get_tools_info() const;

protected:
    /**
     * @brief Parse LLM response to extract action
     */
    virtual std::pair<std::string, std::string> parse_action(const std::string& response) = 0;

    /**
     * @brief Check if agent should stop
     */
    bool should_stop(const std::string& response) const;

    /**
     * @brief Execute tool with given name and input
     */
    ActionResult execute_tool(const std::string& tool_name, const std::string& tool_input);

    /**
     * @brief Handle parsing errors
     */
    std::pair<std::string, std::string> handle_parsing_error(const std::string& response) const;
};

/**
 * @brief ReAct (Reasoning and Acting) Agent
 */
class ReActAgent : public BaseAgent {
private:
    std::shared_ptr<prompts::BasePromptTemplate> prompt_template_;

public:
    ReActAgent(
        std::shared_ptr<llm::BaseLLM> llm,
        std::vector<std::shared_ptr<BaseTool>> tools,
        std::shared_ptr<prompts::BasePromptTemplate> prompt_template = nullptr,
        const AgentConfig& config = AgentConfig{}
    );

    std::string run(const std::string& input) override;
    std::pair<std::string, std::vector<AgentStep>> run_with_steps(const std::string& input) override;

    // Create default ReAct prompt template
    static std::shared_ptr<prompts::BasePromptTemplate> create_default_prompt();

protected:
    std::pair<std::string, std::string> parse_action(const std::string& response) override;
    std::string format_prompt(const std::string& input, const std::vector<AgentStep>& steps) const;
};

/**
 * @brief Zero-shot React Agent
 */
class ZeroShotReactAgent : public BaseAgent {
private:
    std::shared_ptr<prompts::BasePromptTemplate> prompt_template_;

public:
    ZeroShotReactAgent(
        std::shared_ptr<llm::BaseLLM> llm,
        std::vector<std::shared_ptr<BaseTool>> tools,
        const AgentConfig& config = AgentConfig{}
    );

    std::string run(const std::string& input) override;
    std::pair<std::string, std::vector<AgentStep>> run_with_steps(const std::string& input) override;

    static std::shared_ptr<prompts::BasePromptTemplate> create_default_prompt();

protected:
    std::pair<std::string, std::string> parse_action(const std::string& response) override;
};

/**
 * @brief Conversational Agent
 */
class ConversationalAgent : public BaseAgent {
private:
    std::shared_ptr<prompts::BasePromptTemplate> prompt_template_;
    bool use_memory_{true};

public:
    ConversationalAgent(
        std::shared_ptr<llm::BaseLLM> llm,
        std::vector<std::shared_ptr<BaseTool>> tools,
        std::unique_ptr<memory::BaseMemory> memory = nullptr,
        const AgentConfig& config = AgentConfig{}
    );

    std::string run(const std::string& input) override;
    std::pair<std::string, std::vector<AgentStep>> run_with_steps(const std::string& input) override;

    void set_use_memory(bool use_memory) { use_memory_ = use_memory; }

    static std::shared_ptr<prompts::BasePromptTemplate> create_default_prompt();

protected:
    std::pair<std::string, std::string> parse_action(const std::string& response) override;
    std::string format_conversation_prompt(const std::string& input, const std::vector<AgentStep>& steps) const;
};

/**
 * @brief Multi-agent coordinator for managing multiple agents
 */
class MultiAgentCoordinator {
private:
    struct AgentInfo {
        std::shared_ptr<BaseAgent> agent;
        std::string name;
        std::string description;
        std::vector<std::string> capabilities;
    };

    std::unordered_map<std::string, AgentInfo> agents_;
    std::shared_ptr<llm::BaseLLM> coordinator_llm_;
    std::shared_ptr<prompts::BasePromptTemplate> routing_prompt_;
    std::unique_ptr<memory::BaseMemory> shared_memory_;

public:
    MultiAgentCoordinator(
        std::shared_ptr<llm::BaseLLM> coordinator_llm = nullptr,
        std::unique_ptr<memory::BaseMemory> shared_memory = nullptr
    );

    /**
     * @brief Register an agent with the coordinator
     */
    void register_agent(
        const std::string& name,
        std::shared_ptr<BaseAgent> agent,
        const std::string& description,
        const std::vector<std::string>& capabilities = {}
    );

    /**
     * @brief Route task to appropriate agent
     */
    std::string route_and_execute(const std::string& task);

    /**
     * @brief Get available agents
     */
    std::vector<std::string> get_available_agents() const;

    /**
     * @brief Get agent information
     */
    std::string get_agent_info(const std::string& agent_name) const;

private:
    std::string select_agent_for_task(const std::string& task);
    std::shared_ptr<BaseAgent> get_agent(const std::string& name);
    void log_agent_execution(const std::string& agent_name, const std::string& task, const std::string& result);
};

/**
 * @brief Built-in tools for common operations
 */
namespace tools {
    // Search tool
    class SearchTool : public BaseTool {
    private:
        std::function<std::vector<std::string>(const std::string&)> search_function_;
    public:
        explicit SearchTool(std::function<std::vector<std::string>(const std::string&)> search_func);
        std::string name() const override { return "search"; }
        std::string description() const override;
        std::string input_schema() const override;
        std::string run(const std::string& input) override;
    };

    // Calculator tool
    class CalculatorTool : public BaseTool {
    public:
        std::string name() const override { return "calculator"; }
        std::string description() const override;
        std::string input_schema() const override;
        std::string run(const std::string& input) override;
    };

    // File operations tool
    class FileTool : public BaseTool {
    public:
        std::string name() const override { return "file_operations"; }
        std::string description() const override;
        std::string input_schema() const override;
        std::string run(const std::string& input) override;
    };

    // Create built-in tools
    std::shared_ptr<SearchTool> create_search_tool();
    std::shared_ptr<CalculatorTool> create_calculator_tool();
    std::shared_ptr<FileTool> create_file_tool();
}

/**
 * @brief Agent factory for creating different agent types
 */
class AgentFactory {
public:
    enum class AgentType {
        REACT,
        ZERO_SHOT_REACT,
        CONVERSATIONAL
    };

    static std::unique_ptr<BaseAgent> create_agent(
        AgentType type,
        std::shared_ptr<llm::BaseLLM> llm,
        std::vector<std::shared_ptr<BaseTool>> tools = {},
        const AgentConfig& config = AgentConfig{}
    );

    // Convenience factory methods
    static std::unique_ptr<ReActAgent> create_react_agent(
        std::shared_ptr<llm::BaseLLM> llm,
        std::vector<std::shared_ptr<BaseTool>> tools = {}
    );

    static std::unique_ptr<ZeroShotReactAgent> create_zero_shot_react_agent(
        std::shared_ptr<llm::BaseLLM> llm,
        std::vector<std::shared_ptr<BaseTool>> tools = {}
    );

    static std::unique_ptr<ConversationalAgent> create_conversational_agent(
        std::shared_ptr<llm::BaseLLM> llm,
        std::vector<std::shared_ptr<BaseTool>> tools = {}
    );
};

} // namespace langchain::agents