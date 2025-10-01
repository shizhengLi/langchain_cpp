#include "langchain/langchain.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace langchain;

int main() {
    std::cout << "=== LangChain++ Chain Composition Example ===" << std::endl;

    try {
        // 1. Create OpenAI LLM instance
        const char* api_key = std::getenv("OPENAI_API_KEY");
        if (!api_key) {
            std::cout << "Warning: OPENAI_API_KEY environment variable not set." << std::endl;
            std::cout << "Using mock LLM for demonstration." << std::endl;
            api_key = "mock-api-key";
        }

        auto llm = std::make_shared<llm::OpenAILLM>(api_key);

        // Configure LLM
        llm::OpenAIConfig config;
        config.model = "gpt-3.5-turbo";
        config.max_tokens = 300;
        config.temperature = 0.7;
        llm->configure(config);

        // 2. Create simple prompt templates
        std::cout << "\n=== Basic Prompt Templates ===" << std::endl;

        auto summarizer_template = std::make_shared<prompts::PromptTemplate>(
            "Summarize the following text in 2-3 sentences:\n\n{text}\n\nSummary:",
            {{"text"}}
        );

        auto translator_template = std::make_shared<prompts::PromptTemplate>(
            "Translate the following text to {target_language}:\n\n{text}\n\nTranslation:",
            {{"text", "target_language"}}
        );

        auto analyzer_template = std::make_shared<prompts::PromptTemplate>(
            "Analyze the following text for tone, sentiment, and main topics:\n\n{text}\n\nAnalysis:",
            {{"text"}}
        );

        std::cout << "Created prompt templates for summarization, translation, and analysis" << std::endl;

        // 3. Create individual LLM chains
        std::cout << "\n=== Individual Chains ===" << std::endl;

        chains::LLMChain summarizer_chain(llm, summarizer_template);
        chains::LLMChain translator_chain(llm, translator_template);
        chains::LLMChain analyzer_chain(llm, analyzer_template);

        // Test individual chains
        std::string sample_text = "Artificial intelligence has revolutionized how we interact with technology. "
                                 "From natural language processing to computer vision, AI systems are becoming "
                                 "increasingly sophisticated and capable of performing tasks that once required "
                                 "human intelligence. This transformation is reshaping industries and creating "
                                 "new opportunities for innovation.";

        std::cout << "Original text: " << sample_text.substr(0, 100) << "..." << std::endl;

        // Test summarizer chain
        chains::ChainInput summarizer_input = {{"text", sample_text}};
        auto summarizer_start = std::chrono::high_resolution_clock::now();
        auto summarizer_output = summarizer_chain.run(summarizer_input);
        auto summarizer_end = std::chrono::high_resolution_clock::now();

        auto summarizer_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            summarizer_end - summarizer_start).count();

        std::cout << "\nSummarizer Chain (" << summarizer_time << " ms):" << std::endl;
        std::cout << std::any_cast<std::string>(summarizer_output["text"]) << std::endl;

        // Test translator chain
        chains::ChainInput translator_input = {
            {"text", std::any_cast<std::string>(summarizer_output["text"])},
            {"target_language", "Spanish"}
        };
        auto translator_start = std::chrono::high_resolution_clock::now();
        auto translator_output = translator_chain.run(translator_input);
        auto translator_end = std::chrono::high_resolution_clock::now();

        auto translator_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            translator_end - translator_start).count();

        std::cout << "\nTranslator Chain (" << translator_time << " ms):" << std::endl;
        std::cout << std::any_cast<std::string>(translator_output["text"]) << std::endl;

        // Test analyzer chain
        chains::ChainInput analyzer_input = {{"text", sample_text}};
        auto analyzer_start = std::chrono::high_resolution_clock::now();
        auto analyzer_output = analyzer_chain.run(analyzer_input);
        auto analyzer_end = std::chrono::high_resolution_clock::now();

        auto analyzer_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            analyzer_end - analyzer_start).count();

        std::cout << "\nAnalyzer Chain (" << analyzer_time << " ms):" << std::endl;
        std::cout << std::any_cast<std::string>(analyzer_output["text"]) << std::endl;

        // 4. Create Sequential Chain
        std::cout << "\n=== Sequential Chain ===" << std::endl;

        chains::SequentialChain sequential_chain;
        sequential_chain.add_chain(std::make_shared<chains::LLMChain>(summarizer_chain));
        sequential_chain.add_chain(std::make_shared<chains::LLMChain>(analyzer_chain));

        // Define input/output mapping for sequential chain
        std::vector<std::string> input_vars = {"text"};
        std::vector<std::string> output_vars = {"summary", "analysis"};

        std::cout << "Created sequential chain: Summarizer -> Analyzer" << std::endl;

        // Test sequential chain
        chains::ChainInput sequential_input = {{"text", sample_text}};
        auto sequential_start = std::chrono::high_resolution_clock::now();
        auto sequential_output = sequential_chain.run(sequential_input);
        auto sequential_end = std::chrono::high_resolution_clock::now();

        auto sequential_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            sequential_end - sequential_start).count();

        std::cout << "\nSequential Chain Results (" << sequential_time << " ms):" << std::endl;
        std::cout << "Summary: " << std::any_cast<std::string>(sequential_output["summary"]) << std::endl;
        std::cout << "Analysis: " << std::any_cast<std::string>(sequential_output["analysis"]) << std::endl;

        // 5. Create Parallel Chain (simulated with multiple chains)
        std::cout << "\n=== Parallel Processing ===" << std::endl;

        // Create multiple templates for parallel processing
        auto tone_template = std::make_shared<prompts::PromptTemplate>(
            "Analyze the tone of this text (formal, informal, technical, casual):\n\n{text}\n\nTone:",
            {{"text"}}
        );

        auto sentiment_template = std::make_shared<prompts::PromptTemplate>(
            "Analyze the sentiment of this text (positive, negative, neutral):\n\n{text}\n\nSentiment:",
            {{"text"}}
        );

        auto keywords_template = std::make_shared<prompts::PromptTemplate>(
            "Extract 5 main keywords from this text:\n\n{text}\n\nKeywords:",
            {{"text"}}
        );

        chains::LLMChain tone_chain(llm, tone_template);
        chains::LLMChain sentiment_chain(llm, sentiment_template);
        chains::LLMChain keywords_chain(llm, keywords_template);

        // Run chains in parallel using async
        chains::ChainInput parallel_input = {{"text", sample_text}};

        auto parallel_start = std::chrono::high_resolution_clock::now();

        std::future<chains::ChainOutput> tone_future = tone_chain.run_async(parallel_input);
        std::future<chains::ChainOutput> sentiment_future = sentiment_chain.run_async(parallel_input);
        std::future<chains::ChainOutput> keywords_future = keywords_chain.run_async(parallel_input);

        // Wait for all to complete
        auto tone_output = tone_future.get();
        auto sentiment_output = sentiment_future.get();
        auto keywords_output = keywords_future.get();

        auto parallel_end = std::chrono::high_resolution_clock::now();
        auto parallel_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            parallel_end - parallel_start).count();

        std::cout << "Parallel Processing Results (" << parallel_time << " ms):" << std::endl;
        std::cout << "Tone: " << std::any_cast<std::string>(tone_output["text"]) << std::endl;
        std::cout << "Sentiment: " << std::any_cast<std::string>(sentiment_output["text"]) << std::endl;
        std::cout << "Keywords: " << std::any_cast<std::string>(keywords_output["text"]) << std::endl;

        // 6. Complex Chain with Conditional Logic
        std::cout << "\n=== Conditional Chain Logic ===" << std::endl;

        // Create a content evaluator template
        auto evaluator_template = std::make_shared<prompts::PromptTemplate>(
            "Evaluate if this text is primarily technical or non-technical. Answer with only 'technical' or 'non-technical':\n\n{text}",
            {{"text"}}
        );

        chains::LLMChain evaluator_chain(llm, evaluator_template);

        // Create specialized templates
        auto technical_processor_template = std::make_shared<prompts::PromptTemplate>(
            "Extract technical concepts and explain them for a non-technical audience:\n\n{text}\n\nExplanation:",
            {{"text"}}
        );

        auto general_processor_template = std::make_shared<prompts::PromptTemplate>(
            "Provide a general overview of this text suitable for a broad audience:\n\n{text}\n\nOverview:",
            {{"text"}}
        );

        chains::LLMChain technical_processor(llm, technical_processor_template);
        chains::LLMChain general_processor(llm, general_processor_template);

        // Test conditional chain logic
        std::string technical_text = "The implementation uses a convolutional neural network with ReLU activation "
                                   "functions and Adam optimizer. The model architecture includes batch normalization "
                                   "layers and dropout regularization to prevent overfitting.";

        std::cout << "Evaluating text: " << technical_text.substr(0, 100) << "..." << std::endl;

        // Step 1: Evaluate content type
        chains::ChainInput eval_input = {{"text", technical_text}};
        auto eval_output = evaluator_chain.run(eval_input);
        std::string evaluation = std::any_cast<std::string>(eval_output["text"]);
        std::transform(evaluation.begin(), evaluation.end(), evaluation.begin(), ::tolower);

        std::cout << "Evaluation result: " << evaluation << std::endl;

        // Step 2: Process based on evaluation
        chains::ChainOutput conditional_output;
        auto conditional_start = std::chrono::high_resolution_clock::now();

        if (evaluation.find("technical") != std::string::npos) {
            std::cout << "Using technical processor..." << std::endl;
            conditional_output = technical_processor.run(eval_input);
        } else {
            std::cout << "Using general processor..." << std::endl;
            conditional_output = general_processor.run(eval_input);
        }

        auto conditional_end = std::chrono::high_resolution_clock::now();
        auto conditional_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            conditional_end - conditional_start).count();

        std::cout << "Conditional Processing (" << conditional_time << " ms):" << std::endl;
        std::cout << std::any_cast<std::string>(conditional_output["text"]) << std::endl;

        // 7. Chain Composition with Memory
        std::cout << "\n=== Chain with Memory ===" << std::endl;

        // Create a conversation template
        auto conversation_template = std::make_shared<prompts::PromptTemplate>(
            "You are a helpful assistant. Previous context:\n{conversation_history}\n\nHuman: {user_input}\nAssistant:",
            {{"conversation_history", "user_input"}}
        );

        chains::LLMChain conversation_chain(llm, conversation_template);

        // Simulate a conversation
        std::vector<std::string> user_inputs = {
            "What is artificial intelligence?",
            "Can you give me an example?",
            "How is AI used in everyday life?"
        };

        std::string conversation_history = "";

        for (size_t i = 0; i < user_inputs.size(); ++i) {
            std::cout << "\nHuman: " << user_inputs[i] << std::endl;
            std::cout << "----------------------------------------" << std::endl;

            chains::ChainInput conv_input = {
                {"conversation_history", conversation_history},
                {"user_input", user_inputs[i]}
            };

            auto conv_output = conversation_chain.run(conv_input);
            std::string assistant_response = std::any_cast<std::string>(conv_output["text"]);

            std::cout << "Assistant: " << assistant_response << std::endl;

            // Update conversation history
            conversation_history += "\nHuman: " + user_inputs[i] + "\nAssistant: " + assistant_response + "\n";
        }

        // 8. Performance Metrics
        std::cout << "\n=== Performance Metrics ===" << std::endl;

        // Calculate average processing times
        double individual_avg = (summarizer_time + translator_time + analyzer_time) / 3.0;
        double total_individual = summarizer_time + translator_time + analyzer_time;

        std::cout << "Chain Performance Analysis:" << std::endl;
        std::cout << "Average individual chain time: " << std::fixed << std::setprecision(2) << individual_avg << " ms" << std::endl;
        std::cout << "Total individual chain time: " << total_individual << " ms" << std::endl;
        std::cout << "Sequential chain time: " << sequential_time << " ms" << std::endl;
        std::cout << "Parallel chain time: " << parallel_time << " ms" << std::endl;
        std::cout << "Conditional chain time: " << conditional_time << " ms" << std::endl;

        std::cout << "\nEfficiency Metrics:" << std::endl;
        if (parallel_time > 0) {
            std::cout << "Parallel vs Sequential speedup: " << std::fixed << std::setprecision(2)
                     << (double)sequential_time / parallel_time << "x" << std::endl;
        }
        if (individual_avg > 0) {
            std::cout << "Sequential overhead: " << std::fixed << std::setprecision(2)
                     << ((sequential_time - individual_avg * 2) / individual_avg * 100) << "%" << std::endl;
        }

        // 9. Chain Factory and Registry
        std::cout << "\n=== Chain Factory and Registry ===" << std::endl;

        // Register chain types
        chains::ChainRegistry::register_chain_type("LLMChain", chains::ChainRegistry::create_llm_chain);
        chains::ChainRegistry::register_chain_type("SequentialChain", chains::ChainRegistry::create_sequential_chain);

        std::cout << "Registered chain types: LLMChain, SequentialChain" << std::endl;
        std::cout << "Available chain types: " << chains::ChainRegistry::registered_types().size() << std::endl;

        // Create chains using factory
        auto factory_chain = chains::ChainFactory::create_chain("LLMChain");
        if (factory_chain) {
            std::cout << "Successfully created LLM chain using factory" << std::endl;
        }

        // 10. Error Handling and Validation
        std::cout << "\n=== Error Handling and Validation ===" << std::endl;

        try {
            // Test with invalid template variables
            auto invalid_template = std::make_shared<prompts::PromptTemplate>(
                "This template has {nonexistent_variable} placeholder.",
                {{"real_variable"}}  // Mismatch between template and variables
            );

            chains::LLMChain invalid_chain(llm, invalid_template);
            chains::ChainInput invalid_input = {{"real_variable", "test"}};

            auto invalid_output = invalid_chain.run(invalid_input);
            std::cout << "Chain handled template mismatch gracefully" << std::endl;

        } catch (const std::exception& e) {
            std::cout << "Expected error with invalid template: " << e.what() << std::endl;
        }

        std::cout << "\n=== Chain Composition Example completed successfully! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}