#include "langchain/agents/agent.hpp"
#include <sstream>
#include <algorithm>
#include <regex>

namespace langchain::agents {

// FunctionTool implementation
FunctionTool::FunctionTool(
    const std::string& name,
    const std::string& description,
    const std::string& input_schema,
    std::function<std::string(const std::string&)> func
) : name_(name), description_(description), input_schema_(input_schema),
    function_(std::move(func)) {}

std::string FunctionTool::run(const std::string& input) {
    if (!function_) {
        return "Error: No function defined for tool " + name_;
    }
    return function_(input);
}

// BaseAgent implementation
BaseAgent::BaseAgent(
    std::shared_ptr<llm::BaseLLM> llm,
    std::vector<std::shared_ptr<BaseTool>> tools,
    const AgentConfig& config
) : llm_(std::move(llm)), tools_(std::move(tools)), config_(config) {
    config_.validate();
}

void BaseAgent::add_tool(std::shared_ptr<BaseTool> tool) {
    if (tool) {
        tools_.push_back(std::move(tool));
    }
}

void BaseAgent::remove_tool(const std::string& tool_name) {
    tools_.erase(
        std::remove_if(tools_.begin(), tools_.end(),
            [&tool_name](const std::shared_ptr<BaseTool>& tool) {
                return tool && tool->name() == tool_name;
            }),
        tools_.end()
    );
}

std::string BaseAgent::get_tools_info() const {
    std::stringstream ss;
    ss << "Available tools:\n";
    for (const auto& tool : tools_) {
        if (tool) {
            ss << "- " << tool->name() << ": " << tool->description() << "\n";
        }
    }
    return ss.str();
}

bool BaseAgent::should_stop(const std::string& response) const {
    return response.find(config_.stopping_condition) != std::string::npos;
}

ActionResult BaseAgent::execute_tool(const std::string& tool_name, const std::string& tool_input) {
    auto start_time = std::chrono::high_resolution_clock::now();

    auto it = std::find_if(tools_.begin(), tools_.end(),
        [&tool_name](const std::shared_ptr<BaseTool>& tool) {
            return tool && tool->name() == tool_name;
        });

    ActionResult result(tool_name, tool_input, "Tool not found: " + tool_name);
    result.success = false;

    if (it != tools_.end()) {
        try {
            std::string tool_output = (*it)->run(tool_input);
            result = ActionResult(tool_name, tool_input, tool_output);
            result.success = true;
        } catch (const std::exception& e) {
            result = ActionResult(tool_name, tool_input, "Error: " + std::string(e.what()));
            result.success = false;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    return result;
}

std::pair<std::string, std::string> BaseAgent::handle_parsing_error(const std::string& response) const {
    // Default behavior: try to extract tool name and input using regex
    std::regex action_regex(R"(Action:\s*(\w+)[\s]*\[?([^\]]*)\]?)");
    std::smatch match;

    if (std::regex_search(response, match, action_regex)) {
        return {match[1].str(), match[2].str()};
    }

    return {"", ""};
}

// ReActAgent implementation
ReActAgent::ReActAgent(
    std::shared_ptr<llm::BaseLLM> llm,
    std::vector<std::shared_ptr<BaseTool>> tools,
    std::shared_ptr<prompts::BasePromptTemplate> prompt_template,
    const AgentConfig& config
) : BaseAgent(std::move(llm), std::move(tools), config), prompt_template_(std::move(prompt_template)) {

    if (!prompt_template_) {
        prompt_template_ = create_default_prompt();
    }
}

std::string ReActAgent::run(const std::string& input) {
    auto [result, steps] = run_with_steps(input);
    return result;
}

std::pair<std::string, std::vector<AgentStep>> ReActAgent::run_with_steps(const std::string& input) {
    std::vector<AgentStep> steps;
    std::string current_input = input;

    for (size_t iteration = 0; iteration < config_.max_iterations; ++iteration) {
        // Format prompt with current context
        std::string prompt = format_prompt(current_input, steps);

        // Get LLM response
        auto response = llm_->complete(prompt);

        if (response.success && response.error_message.empty()) {
            std::string llm_output = response.content;

            // Check if we should stop
            if (should_stop(llm_output)) {
                steps.emplace_back(AgentStep::StepType::FINAL_ANSWER, llm_output);
                break;
            }

            // Parse action
            auto [tool_name, tool_input] = parse_action(llm_output);

            if (!tool_name.empty()) {
                // Execute tool
                ActionResult action_result = execute_tool(tool_name, tool_input);
                steps.emplace_back(AgentStep::StepType::ACTION, llm_output, action_result);
                steps.emplace_back(AgentStep::StepType::OBSERVATION, action_result.output);

                current_input = "Observation: " + action_result.output;
            } else {
                steps.emplace_back(AgentStep::StepType::THOUGHT, llm_output);
            }
        } else {
            steps.emplace_back(AgentStep::StepType::THOUGHT, "Error: " + response.error_message);
            break;
        }
    }

    // Extract final answer from the last step
    std::string final_answer = steps.empty() ? "No answer generated" : steps.back().content;
    return {final_answer, steps};
}

std::shared_ptr<prompts::BasePromptTemplate> ReActAgent::create_default_prompt() {
    std::string template_str = R"(
You are a helpful assistant. Use the following tools to answer the question:

{tools_info}

Question: {input}

Think step by step and use the available tools when necessary.
When you have the final answer, start your response with "Final Answer:".

Thought: )";

    return prompts::utils::simple_template(template_str);
}

std::pair<std::string, std::string> ReActAgent::parse_action(const std::string& response) {
    // Look for "Action: tool_name[tool_input]" pattern
    std::regex action_regex(R"(Action:\s*(\w+)\s*\[?([^\]]*)\]?)");
    std::smatch match;

    if (std::regex_search(response, match, action_regex)) {
        std::string tool_name = match[1].str();
        std::string tool_input = match[2].str();
        return {tool_name, tool_input};
    }

    return {"", ""};
}

std::string ReActAgent::format_prompt(const std::string& input, const std::vector<AgentStep>& steps) const {
    std::stringstream ss;
    ss << get_tools_info() << "\n\n";
    ss << "Question: " << input << "\n\n";

    for (const auto& step : steps) {
        switch (step.type) {
            case AgentStep::StepType::THOUGHT:
                ss << "Thought: " << step.content << "\n";
                break;
            case AgentStep::StepType::ACTION:
                ss << "Action: " << step.action_result->tool_name;
                if (!step.action_result->tool_input.empty()) {
                    ss << "[" << step.action_result->tool_input << "]";
                }
                ss << "\n";
                break;
            case AgentStep::StepType::OBSERVATION:
                ss << "Observation: " << step.content << "\n";
                break;
            case AgentStep::StepType::FINAL_ANSWER:
                ss << "Final Answer: " << step.content << "\n";
                break;
        }
    }

    ss << "Thought: ";
    return ss.str();
}

// ZeroShotReactAgent implementation
ZeroShotReactAgent::ZeroShotReactAgent(
    std::shared_ptr<llm::BaseLLM> llm,
    std::vector<std::shared_ptr<BaseTool>> tools,
    const AgentConfig& config
) : BaseAgent(std::move(llm), std::move(tools), config) {
    prompt_template_ = create_default_prompt();
}

std::string ZeroShotReactAgent::run(const std::string& input) {
    auto [result, steps] = run_with_steps(input);
    return result;
}

std::pair<std::string, std::vector<AgentStep>> ZeroShotReactAgent::run_with_steps(const std::string& input) {
    // Similar to ReActAgent but with zero-shot prompting
    std::vector<AgentStep> steps;

    // Simplified implementation
    std::string prompt = "Answer this question: " + input;
    auto response = llm_->complete(prompt);

    if (response.success && response.error_message.empty()) {
        steps.emplace_back(AgentStep::StepType::FINAL_ANSWER, response.content);
        return {response.content, steps};
    } else {
        steps.emplace_back(AgentStep::StepType::THOUGHT, "Error: " + response.error_message);
        return {"Error occurred", steps};
    }
}

std::shared_ptr<prompts::BasePromptTemplate> ZeroShotReactAgent::create_default_prompt() {
    return prompts::utils::simple_template("Question: {input}\n\nAnswer:");
}

std::pair<std::string, std::string> ZeroShotReactAgent::parse_action(const std::string& response) {
    // Simplified parsing for zero-shot
    return {"", ""};
}

// ConversationalAgent implementation
ConversationalAgent::ConversationalAgent(
    std::shared_ptr<llm::BaseLLM> llm,
    std::vector<std::shared_ptr<BaseTool>> tools,
    std::unique_ptr<memory::BaseMemory> memory,
    const AgentConfig& config
) : BaseAgent(std::move(llm), std::move(tools), config) {
    memory_ = std::move(memory);

    prompt_template_ = create_default_prompt();
}

std::string ConversationalAgent::run(const std::string& input) {
    auto [result, steps] = run_with_steps(input);
    return result;
}

std::pair<std::string, std::vector<AgentStep>> ConversationalAgent::run_with_steps(const std::string& input) {
    std::vector<AgentStep> steps;

    // Add user input to memory if available
    if (memory_ && use_memory_) {
        memory_->add_message(memory::ChatMessage(memory::ChatMessage::Type::HUMAN, input));
    }

    // Format conversation prompt
    std::string prompt = format_conversation_prompt(input, steps);

    auto response = llm_->complete(prompt);

    if (response.success && response.error_message.empty()) {
        steps.emplace_back(AgentStep::StepType::FINAL_ANSWER, response.content);

        // Add assistant response to memory if available
        if (memory_ && use_memory_) {
            memory_->add_message(memory::ChatMessage(memory::ChatMessage::Type::AI, response.content));
        }

        return {response.content, steps};
    } else {
        steps.emplace_back(AgentStep::StepType::THOUGHT, "Error: " + response.error_message);
        return {"Error occurred", steps};
    }
}

std::shared_ptr<prompts::BasePromptTemplate> ConversationalAgent::create_default_prompt() {
    return prompts::utils::simple_template("Conversation:\n{history}\nHuman: {input}\nAssistant:");
}

std::pair<std::string, std::string> ConversationalAgent::parse_action(const std::string& response) {
    // Conversational agent typically doesn't use tools in the same way
    return {"", ""};
}

std::string ConversationalAgent::format_conversation_prompt(const std::string& input, const std::vector<AgentStep>& steps) const {
    std::stringstream ss;

    if (memory_ && use_memory_) {
        ss << memory_->get_context() << "\n";
    }

    ss << "Human: " << input << "\nAssistant: ";
    return ss.str();
}

// MultiAgentCoordinator implementation
MultiAgentCoordinator::MultiAgentCoordinator(
    std::shared_ptr<llm::BaseLLM> coordinator_llm,
    std::unique_ptr<memory::BaseMemory> shared_memory
) : coordinator_llm_(std::move(coordinator_llm)), shared_memory_(std::move(shared_memory)) {}

void MultiAgentCoordinator::register_agent(
    const std::string& name,
    std::shared_ptr<BaseAgent> agent,
    const std::string& description,
    const std::vector<std::string>& capabilities
) {
    AgentInfo info;
    info.agent = std::move(agent);
    info.name = name;
    info.description = description;
    info.capabilities = capabilities;

    agents_[name] = std::move(info);
}

std::string MultiAgentCoordinator::route_and_execute(const std::string& task) {
    std::string selected_agent = select_agent_for_task(task);

    auto agent_it = agents_.find(selected_agent);
    if (agent_it != agents_.end()) {
        auto result = agent_it->second.agent->run(task);
        log_agent_execution(selected_agent, task, result);
        return result;
    }

    return "No suitable agent found for task: " + task;
}

std::vector<std::string> MultiAgentCoordinator::get_available_agents() const {
    std::vector<std::string> names;
    for (const auto& [name, info] : agents_) {
        names.push_back(name);
    }
    return names;
}

std::string MultiAgentCoordinator::get_agent_info(const std::string& agent_name) const {
    auto it = agents_.find(agent_name);
    if (it != agents_.end()) {
        const auto& info = it->second;
        std::stringstream ss;
        ss << "Agent: " << info.name << "\n";
        ss << "Description: " << info.description << "\n";
        ss << "Capabilities: ";
        for (const auto& cap : info.capabilities) {
            ss << cap << ", ";
        }
        return ss.str();
    }
    return "Agent not found: " + agent_name;
}

std::string MultiAgentCoordinator::select_agent_for_task(const std::string& task) {
    // Simple routing logic - select first available agent
    if (agents_.empty()) {
        return "";
    }
    return agents_.begin()->first;
}

std::shared_ptr<BaseAgent> MultiAgentCoordinator::get_agent(const std::string& name) {
    auto it = agents_.find(name);
    return it != agents_.end() ? it->second.agent : nullptr;
}

void MultiAgentCoordinator::log_agent_execution(const std::string& agent_name, const std::string& task, const std::string& result) {
    // Simple logging - could be enhanced with proper logging system
    if (shared_memory_) {
        shared_memory_->add_message(memory::ChatMessage(
            memory::ChatMessage::Type::SYSTEM,
            "Agent " + agent_name + " executed task: " + task + " -> " + result
        ));
    }
}

// Tool implementations
namespace tools {

std::string SearchTool::description() const {
    return "Search for information using a query string";
}

std::string SearchTool::input_schema() const {
    return R"({"type": "string", "description": "Search query"})";
}

std::string SearchTool::run(const std::string& input) {
    if (!search_function_) {
        return "Search function not configured";
    }

    try {
        auto results = search_function_(input);
        std::stringstream ss;
        ss << "Search results for '" << input << "':\n";
        for (const auto& result : results) {
            ss << "- " << result << "\n";
        }
        return ss.str();
    } catch (const std::exception& e) {
        return "Search error: " + std::string(e.what());
    }
}

SearchTool::SearchTool(std::function<std::vector<std::string>(const std::string&)> search_func)
    : search_function_(std::move(search_func)) {}

std::string CalculatorTool::description() const {
    return "Perform mathematical calculations";
}

std::string CalculatorTool::input_schema() const {
    return R"({"type": "string", "description": "Mathematical expression to evaluate"})";
}

std::string CalculatorTool::run(const std::string& input) {
    // Simple calculator implementation - in practice, use a proper expression parser
    return "Calculator result for: " + input;
}

std::string FileTool::description() const {
    return "Perform file operations like read, write, list";
}

std::string FileTool::input_schema() const {
    return R"({"type": "string", "description": "File operation command"})";
}

std::string FileTool::run(const std::string& input) {
    return "File operation: " + input;
}

std::shared_ptr<SearchTool> create_search_tool() {
    return std::make_shared<SearchTool>(nullptr);
}

std::shared_ptr<CalculatorTool> create_calculator_tool() {
    return std::make_shared<CalculatorTool>();
}

std::shared_ptr<FileTool> create_file_tool() {
    return std::make_shared<FileTool>();
}

} // namespace tools

// AgentFactory implementation
std::unique_ptr<BaseAgent> AgentFactory::create_agent(
    AgentType type,
    std::shared_ptr<llm::BaseLLM> llm,
    std::vector<std::shared_ptr<BaseTool>> tools,
    const AgentConfig& config
) {
    switch (type) {
        case AgentType::REACT:
            return std::make_unique<ReActAgent>(std::move(llm), std::move(tools), nullptr, config);
        case AgentType::ZERO_SHOT_REACT:
            return std::make_unique<ZeroShotReactAgent>(std::move(llm), std::move(tools), config);
        case AgentType::CONVERSATIONAL:
            return std::make_unique<ConversationalAgent>(std::move(llm), std::move(tools), nullptr, config);
        default:
            throw std::invalid_argument("Unknown agent type");
    }
}

std::unique_ptr<ReActAgent> AgentFactory::create_react_agent(
    std::shared_ptr<llm::BaseLLM> llm,
    std::vector<std::shared_ptr<BaseTool>> tools
) {
    return std::make_unique<ReActAgent>(std::move(llm), std::move(tools));
}

std::unique_ptr<ZeroShotReactAgent> AgentFactory::create_zero_shot_react_agent(
    std::shared_ptr<llm::BaseLLM> llm,
    std::vector<std::shared_ptr<BaseTool>> tools
) {
    return std::make_unique<ZeroShotReactAgent>(std::move(llm), std::move(tools));
}

std::unique_ptr<ConversationalAgent> AgentFactory::create_conversational_agent(
    std::shared_ptr<llm::BaseLLM> llm,
    std::vector<std::shared_ptr<BaseTool>> tools
) {
    return std::make_unique<ConversationalAgent>(std::move(llm), std::move(tools));
}

} // namespace langchain::agents