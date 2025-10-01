#include "langchain/langchain.hpp"
#include <iostream>
#include <vector>
#include <chrono>

using namespace langchain;

int main() {
    std::cout << "=== LangChain++ LLM Integration Example ===" << std::endl;

    try {
        // 1. Check for OpenAI API key
        const char* api_key = std::getenv("OPENAI_API_KEY");
        if (!api_key) {
            std::cout << "Warning: OPENAI_API_KEY environment variable not set." << std::endl;
            std::cout << "Using mock LLM for demonstration." << std::endl;
            api_key = "mock-api-key";
        }

        // 2. Create OpenAI LLM instance
        auto llm = std::make_shared<llm::OpenAILLM>(api_key);

        // Configure LLM
        llm::OpenAIConfig config;
        config.model = "gpt-3.5-turbo";
        config.max_tokens = 500;
        config.temperature = 0.7;
        config.timeout = std::chrono::seconds(30);
        config.max_retries = 3;

        llm->configure(config);

        std::cout << "LLM Configuration:" << std::endl;
        std::cout << "  Model: " << config.model << std::endl;
        std::cout << "  Max tokens: " << config.max_tokens << std::endl;
        std::cout << "  Temperature: " << config.temperature << std::endl;

        // 3. Basic text generation
        std::cout << "\n=== Basic Text Generation ===" << std::endl;

        std::vector<std::string> prompts = {
            "Explain quantum computing in simple terms.",
            "What are the main benefits of using C++ for AI applications?",
            "How does natural language processing work?",
            "Write a haiku about artificial intelligence."
        };

        for (const auto& prompt : prompts) {
            std::cout << "\nPrompt: \"" << prompt << "\"" << std::endl;
            std::cout << "----------------------------------------" << std::endl;

            auto start_time = std::chrono::high_resolution_clock::now();
            std::string response = llm->generate(prompt);
            auto end_time = std::chrono::high_resolution_clock::now();

            auto generation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();

            std::cout << "Response (took " << generation_time << " ms):" << std::endl;
            std::cout << response << std::endl;

            // Get token usage
            auto usage = llm->get_last_usage();
            std::cout << "Token usage: " << usage.prompt_tokens << " prompt, "
                      << usage.completion_tokens << " completion tokens" << std::endl;

            // Estimate cost
            double cost = llm->estimate_cost(usage);
            std::cout << "Estimated cost: $" << std::fixed << std::setprecision(6) << cost << std::endl;
        }

        // 4. Streaming responses
        std::cout << "\n=== Streaming Response ===" << std::endl;

        std::string streaming_prompt = "Write a short story about a robot learning to paint.";
        std::cout << "Prompt: \"" << streaming_prompt << "\"" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Streaming response:" << std::endl;

        start_time = std::chrono::high_resolution_clock::now();

        llm->generate_stream(streaming_prompt,
            [](const std::string& chunk) {
                std::cout << chunk << std::flush;
            },
            config
        );

        end_time = std::chrono::high_resolution_clock::now();
        auto streaming_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();

        std::cout << "\n\nStreaming completed in " << streaming_time << " ms" << std::endl;

        // 5. Conversation with messages
        std::cout << "\n=== Conversation Example ===" << std::endl;

        std::vector<llm::Message> conversation = {
            {"system", "You are a helpful assistant that explains technical concepts clearly and concisely."},
            {"user", "What is the difference between supervised and unsupervised learning?"}
        };

        std::cout << "Human: What is the difference between supervised and unsupervised learning?" << std::endl;
        std::cout << "Assistant: ";

        std::string response1 = llm->generate(conversation, config);
        std::cout << response1 << std::endl;

        // Add assistant response to conversation
        conversation.push_back({"assistant", response1});
        conversation.push_back({"user", "Can you give me an example of each?"});

        std::cout << "\nHuman: Can you give me an example of each?" << std::endl;
        std::cout << "Assistant: ";

        std::string response2 = llm->generate(conversation, config);
        std::cout << response2 << std::endl;

        // 6. Chain integration
        std::cout << "\n=== Chain Integration ===" << std::endl;

        // Create a simple prompt template
        auto prompt_template = std::make_shared<prompts::PromptTemplate>(
            "You are an expert {domain} assistant. Please answer the following question: {question}",
            {"domain", "question"}
        );

        // Create LLM chain
        chains::LLMChain chain(llm, prompt_template);

        // Run the chain
        chains::ChainInput input = {
            {"domain", "computer science"},
            {"question", "What is the time complexity of quicksort?"}
        };

        std::cout << "Running LLM chain..." << std::endl;
        auto chain_output = chain.run(input);
        std::cout << "Chain output: " << std::any_cast<std::string>(chain_output["text"]) << std::endl;

        // 7. Error handling and retry logic
        std::cout << "\n=== Error Handling Example ===" << std::endl;

        // Create a chain with retries
        auto retry_chain = std::make_shared<chains::LLMChain>(llm, prompt_template);

        // Simulate a failure scenario with retries
        chains::ChainInput retry_input = {
            {"domain", "physics"},
            {"question", "Explain the concept of quantum entanglement."}
        };

        try {
            auto retry_output = retry_chain->run(retry_input);
            std::cout << "Chain completed successfully with retries" << std::endl;
        } catch (const llm::LLMException& e) {
            std::cout << "LLM error after retries: " << e.what() << std::endl;
        }

        // 8. Async operations
        std::cout << "\n=== Async Operations ===" << std::endl;

        std::vector<std::future<std::string>> async_results;

        for (int i = 0; i < 3; ++i) {
            std::string prompt = "Generate a creative name for AI project #" + std::to_string(i + 1);
            async_results.push_back(llm->generate_async(prompt, config));
        }

        std::cout << "Processing 3 async requests..." << std::endl;

        for (size_t i = 0; i < async_results.size(); ++i) {
            try {
                std::string result = async_results[i].get();
                std::cout << "AI Project #" << (i + 1) << ": " << result << std::endl;
            } catch (const std::exception& e) {
                std::cout << "Async request " << (i + 1) << " failed: " << e.what() << std::endl;
            }
        }

        // 9. Show LLM capabilities
        std::cout << "\n=== LLM Capabilities ===" << std::endl;

        std::cout << "Streaming support: " << (llm->supports_streaming() ? "Yes" : "No") << std::endl;
        std::cout << "Function calling: " << (llm->supports_function_calling() ? "Yes" : "No") << std::endl;
        std::cout << "Vision support: " << (llm->supports_vision() ? "Yes" : "No") << std::endl;

        std::cout << "\nSupported models:" << std::endl;
        auto models = llm->supported_models();
        for (const auto& model : models) {
            std::cout << "  - " << model << std::endl;
        }

        // 10. Rate limiting and usage monitoring
        std::cout << "\n=== Usage Monitoring ===" << std::endl;

        llm::UsageInfo total_usage;
        size_t total_requests = 0;

        // Simulate multiple requests
        for (int i = 0; i < 5; ++i) {
            std::string prompt = "Count to " + std::to_string(i + 1);
            try {
                llm->generate(prompt, config);
                auto usage = llm->get_last_usage();
                total_usage += usage;
                total_requests++;
            } catch (const std::exception& e) {
                std::cout << "Request failed: " << e.what() << std::endl;
            }
        }

        std::cout << "Total requests: " << total_requests << std::endl;
        std::cout << "Total prompt tokens: " << total_usage.prompt_tokens << std::endl;
        std::cout << "Total completion tokens: " << total_usage.completion_tokens << std::endl;
        std::cout << "Total cost: $" << std::fixed << std::setprecision(6) << llm->estimate_cost(total_usage) << std::endl;

        std::cout << "\n=== LLM Integration Example completed successfully! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}