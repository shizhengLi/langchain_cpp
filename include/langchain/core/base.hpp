#pragma once

#include "types.hpp"
#include "config.hpp"
#include <memory>
#include <vector>
#include <string>
#include <future>
#include <iterator>
#include <exception>

namespace langchain {

/**
 * @brief Base classes and interfaces for all LangChain++ components
 */

/**
 * @brief Base exception class for LangChain++
 */
class LangChainException : public std::exception {
private:
    std::string message_;
    std::string error_code_;

public:
    LangChainException(const std::string& message, const std::string& error_code = "")
        : message_(message), error_code_(error_code) {}

    const char* what() const noexcept override {
        return message_.c_str();
    }

    const std::string& error_code() const noexcept {
        return error_code_;
    }
};

/**
 * @brief Configuration exception
 */
class ConfigurationException : public LangChainException {
public:
    explicit ConfigurationException(const std::string& message)
        : LangChainException(message, "CONFIG_ERROR") {}
};

/**
 * @brief Retrieval exception
 */
class RetrievalException : public LangChainException {
public:
    explicit RetrievalException(const std::string& message)
        : LangChainException(message, "RETRIEVAL_ERROR") {}
};

/**
 * @brief LLM exception
 */
class LLMException : public LangChainException {
public:
    explicit LLMException(const std::string& message)
        : LangChainException(message, "LLM_ERROR") {}
};

/**
 * @brief Memory exception
 */
class MemoryException : public LangChainException {
public:
    explicit MemoryException(const std::string& message)
        : LangChainException(message, "MEMORY_ERROR") {}
};


/**
 * @brief Base interface for LLM implementations
 */
class BaseLLM {
public:
    virtual ~BaseLLM() = default;

    /**
     * @brief Generate response for a prompt
     * @param prompt Input prompt
     * @param config Generation configuration
     * @return LLM result
     */
    virtual LLMResult generate(const Prompt& prompt,
                              const GenerationConfig& config = {}) = 0;

    /**
     * @brief Asynchronous generation
     * @param prompt Input prompt
     * @param config Generation configuration
     * @return Future containing LLM result
     */
    virtual std::future<LLMResult> generate_async(const Prompt& prompt,
                                                 const GenerationConfig& config = {}) {
        return std::async(std::launch::async, [this, prompt, config]() {
            return this->generate(prompt, config);
        });
    }

    /**
     * @brief Batch generation for multiple prompts
     * @param prompts Vector of prompts
     * @param config Generation configuration
     * @return Vector of LLM results
     */
    virtual std::vector<LLMResult> generate_batch(const std::vector<Prompt>& prompts,
                                                 const GenerationConfig& config = {}) {
        std::vector<LLMResult> results;
        results.reserve(prompts.size());

        for (const auto& prompt : prompts) {
            results.push_back(generate(prompt, config));
        }

        return results;
    }

    /**
     * @brief Generate streaming response
     * @param prompt Input prompt
     * @param config Generation configuration
     * @return Iterator for streaming response
     */
    virtual std::unique_ptr<std::iterator<std::input_iterator_tag, std::string>>
    generate_stream(const Prompt& prompt, const GenerationConfig& config = {}) = 0;

    /**
     * @brief Get model information
     * @return Model information map
     */
    virtual std::unordered_map<std::string, std::any> get_model_info() const = 0;

    /**
     * @brief Check if the LLM is available
     * @return True if available
     */
    virtual bool is_available() const = 0;

    /**
     * @brief Get performance metrics
     * @return Performance metrics
     */
    virtual PerformanceMetrics get_performance_metrics() const = 0;
};

/**
 * @brief Base interface for embedding models
 */
class EmbeddingModel {
public:
    virtual ~EmbeddingModel() = default;

    /**
     * @brief Embed a single text
     * @param text Input text
     * @return Embedding result
     */
    virtual EmbeddingResult embed(const std::string& text) = 0;

    /**
     * @brief Asynchronous embedding
     * @param text Input text
     * @return Future containing embedding result
     */
    virtual std::future<EmbeddingResult> embed_async(const std::string& text) {
        return std::async(std::launch::async, [this, text]() {
            return this->embed(text);
        });
    }

    /**
     * @brief Embed multiple texts
     * @param texts Vector of texts
     * @return Vector of embedding results
     */
    virtual std::vector<EmbeddingResult> embed_batch(const std::vector<std::string>& texts) = 0;

    /**
     * @brief Get embedding dimension
     * @return Embedding dimension
     */
    virtual size_t get_dimension() const = 0;

    /**
     * @brief Get model information
     * @return Model information map
     */
    virtual std::unordered_map<std::string, std::any> get_model_info() const = 0;

    /**
     * @brief Check if the model is available
     * @return True if available
     */
    virtual bool is_available() const = 0;
};

/**
 * @brief Base interface for vector stores
 */
class VectorStore {
public:
    virtual ~VectorStore() = default;

    /**
     * @brief Add a vector to the store
     * @param id Vector ID
     * @param vector Vector data
     * @param metadata Optional metadata
     */
    virtual void add_vector(const std::string& id,
                           const std::vector<float>& vector,
                           const std::unordered_map<std::string, std::any>& metadata = {}) = 0;

    /**
     * @brief Add multiple vectors
     * @param vectors Vector of (id, vector) pairs
     */
    virtual void add_vectors(const std::vector<std::pair<std::string, std::vector<float>>>& vectors) {
        for (const auto& [id, vector] : vectors) {
            add_vector(id, vector);
        }
    }

    /**
     * @brief Search for similar vectors
     * @param query_vector Query vector
     * @param top_k Number of results to return
     * @return Vector of search results
     */
    virtual std::vector<VectorSearchResult> search_similar(const std::vector<float>& query_vector,
                                                          size_t top_k = 10) = 0;

    /**
     * @brief Get a vector by ID
     * @param id Vector ID
     * @return Vector data, or empty if not found
     */
    virtual std::optional<std::vector<float>> get_vector(const std::string& id) = 0;

    /**
     * @brief Delete a vector
     * @param id Vector ID
     * @return True if deleted successfully
     */
    virtual bool delete_vector(const std::string& id) = 0;

    /**
     * @brief Get the number of vectors in the store
     * @return Vector count
     */
    virtual size_t size() const = 0;

    /**
     * @brief Check if the store is empty
     * @return True if empty
     */
    virtual bool empty() const {
        return size() == 0;
    }

    /**
     * @brief Clear all vectors
     */
    virtual void clear() = 0;

    /**
     * @brief Save the vector store to disk
     * @param path File path
     */
    virtual void save(const std::string& path) = 0;

    /**
     * @brief Load the vector store from disk
     * @param path File path
     */
    virtual void load(const std::string& path) = 0;
};

/**
 * @brief Base interface for memory management
 */
class BaseMemory {
public:
    virtual ~BaseMemory() = default;

    /**
     * @brief Add a conversation message
     * @param message Conversation message
     */
    virtual void add_message(const ConversationMessage& message) = 0;

    /**
     * @brief Get recent conversation messages
     * @param limit Maximum number of messages to return
     * @return Vector of messages
     */
    virtual std::vector<ConversationMessage> get_messages(size_t limit = 0) const = 0;

    /**
     * @brief Clear all messages
     */
    virtual void clear() = 0;

    /**
     * @brief Get memory statistics
     * @return Statistics map
     */
    virtual std::unordered_map<std::string, std::any> get_statistics() const = 0;

    /**
     * @brief Save memory to persistent storage
     * @param path Storage path
     */
    virtual void save(const std::string& path) = 0;

    /**
     * @brief Load memory from persistent storage
     * @param path Storage path
     */
    virtual void load(const std::string& path) = 0;
};

/**
 * @brief Base interface for chains
 */
class BaseChain {
public:
    virtual ~BaseChain() = default;

    /**
     * @brief Execute the chain
     * @param inputs Input values
     * @return Output values
     */
    virtual std::unordered_map<std::string, std::any> run(
        const std::unordered_map<std::string, std::any>& inputs) = 0;

    /**
     * @brief Asynchronous chain execution
     * @param inputs Input values
     * @return Future containing outputs
     */
    virtual std::future<std::unordered_map<std::string, std::any>> run_async(
        const std::unordered_map<std::string, std::any>& inputs) {
        return std::async(std::launch::async, [this, inputs]() {
            return this->run(inputs);
        });
    }

    /**
     * @brief Get chain information
     * @return Chain information map
     */
    virtual std::unordered_map<std::string, std::any> get_chain_info() const = 0;

    /**
     * @brief Get input keys expected by this chain
     * @return Vector of input keys
     */
    virtual std::vector<std::string> get_input_keys() const = 0;

    /**
     * @brief Get output keys produced by this chain
     * @return Vector of output keys
     */
    virtual std::vector<std::string> get_output_keys() const = 0;
};

/**
 * @brief Base interface for tools
 */
class BaseTool {
public:
    virtual ~BaseTool() = default;

    /**
     * @brief Execute the tool
     * @param input Tool input
     * @return Tool output
     */
    virtual std::string run(const std::string& input) = 0;

    /**
     * @brief Asynchronous tool execution
     * @param input Tool input
     * @return Future containing tool output
     */
    virtual std::future<std::string> run_async(const std::string& input) {
        return std::async(std::launch::async, [this, input]() {
            return this->run(input);
        });
    }

    /**
     * @brief Get tool name
     * @return Tool name
     */
    virtual std::string get_name() const = 0;

    /**
     * @brief Get tool description
     * @return Tool description
     */
    virtual std::string get_description() const = 0;

    /**
     * @brief Get tool input schema
     * @return Input schema description
     */
    virtual std::string get_input_schema() const = 0;

    /**
     * @brief Validate tool input
     * @param input Input to validate
     * @return True if valid
     */
    virtual bool validate_input(const std::string& input) const = 0;
};

/**
 * @brief Base interface for agents
 */
class BaseAgent {
public:
    virtual ~BaseAgent() = default;

    /**
     * @brief Run the agent
     * @param input Agent input
     * @return Agent response
     */
    virtual std::string run(const std::string& input) = 0;

    /**
     * @brief Asynchronous agent execution
     * @param input Agent input
     * @return Future containing agent response
     */
    virtual std::future<std::string> run_async(const std::string& input) {
        return std::async(std::launch::async, [this, input]() {
            return this->run(input);
        });
    }

    /**
     * @brief Get agent information
     * @return Agent information map
     */
    virtual std::unordered_map<std::string, std::any> get_agent_info() const = 0;

    /**
     * @brief Get available tools
     * @return Vector of available tools
     */
    virtual std::vector<std::shared_ptr<BaseTool>> get_tools() const = 0;

    /**
     * @brief Add a tool to the agent
     * @param tool Tool to add
     */
    virtual void add_tool(std::shared_ptr<BaseTool> tool) = 0;
};

} // namespace langchain