#include "langchain/langchain.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace langchain;

int main() {
    std::cout << "=== LangChain++ Memory Systems Example ===" << std::endl;

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
        config.max_tokens = 200;
        config.temperature = 0.7;
        llm->configure(config);

        // 2. Simple Memory Buffer
        std::cout << "\n=== Simple Memory Buffer ===" << std::endl;

        memory::ConversationBufferMemory buffer_memory;
        buffer_memory.set_max_tokens(1000);

        std::cout << "Created conversation buffer memory with 1000 token limit" << std::endl;

        // Simulate a conversation
        std::vector<std::pair<std::string, std::string>> conversation_pairs = {
            {"What is machine learning?", "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming."},
            {"How does it work?", "ML algorithms identify patterns in training data and use these patterns to make predictions or decisions on new data."},
            {"What are the main types?", "The main types are supervised learning, unsupervised learning, and reinforcement learning."},
            {"Can you explain supervised learning?", "Supervised learning uses labeled training data to learn input-output mappings and make predictions on new inputs."},
            {"What about deep learning?", "Deep learning is a subset of ML using neural networks with multiple layers to learn hierarchical representations."}
        };

        for (const auto& [human_msg, ai_msg] : conversation_pairs) {
            std::cout << "\nHuman: " << human_msg << std::endl;
            std::cout << "AI: " << ai_msg << std::endl;

            // Add to memory
            memory::Message user_message{memory::MessageRole::USER, human_msg};
            memory::Message ai_message{memory::MessageRole::ASSISTANT, ai_msg};

            buffer_memory.add_message(user_message);
            buffer_memory.add_message(ai_message);

            // Show memory statistics
            std::cout << "Memory: " << buffer_memory.num_messages() << " messages, "
                     << buffer_memory.current_token_count() << " tokens" << std::endl;
        }

        // Show memory contents
        std::cout << "\nCurrent memory contents:" << std::endl;
        std::cout << buffer_memory.get_memory_string() << std::endl;

        // Test memory with context window
        std::cout << "\n=== Testing Memory Context Window ===" << std::endl;

        buffer_memory.set_max_tokens(200);  // Reduce limit to test windowing
        std::cout << "Reduced token limit to 200 tokens" << std::endl;

        // Add a new message to trigger window management
        memory::Message new_user_msg{memory::MessageRole::USER, "What's the difference between ML and AI?"};
        buffer_memory.add_message(new_user_msg);

        std::cout << "After adding new message:" << std::endl;
        std::cout << "Memory: " << buffer_memory.num_messages() << " messages, "
                 << buffer_memory.current_token_count() << " tokens" << std::endl;
        std::cout << buffer_memory.get_memory_string() << std::endl;

        // 3. Summary Memory
        std::cout << "\n=== Summary Memory ===" << std::endl;

        memory::ConversationSummaryMemory summary_memory(llm);
        summary_memory.set_prompt_template(
            "Summarize this conversation in 2-3 sentences, focusing on the key topics discussed:\n\n{chat_history}"
        );

        std::cout << "Created conversation summary memory" << std::endl;

        // Add messages to summary memory
        summary_memory.add_message({memory::MessageRole::USER, "I want to learn about AI and ML."});
        summary_memory.add_message({memory::MessageRole::ASSISTANT, "AI is the broader field of creating intelligent machines, while ML is a subset focusing on learning from data."});
        summary_memory.add_message({memory::MessageRole::USER, "What are the applications?"});
        summary_memory.add_message({memory::MessageRole::ASSISTANT, "Applications include image recognition, natural language processing, recommendation systems, and autonomous vehicles."});

        std::cout << "Summary: " << summary_memory.get_summary() << std::endl;

        // Add more messages and see how summary updates
        summary_memory.add_message({memory::MessageRole::USER, "How do I get started with ML?"});
        summary_memory.add_message({memory::MessageRole::ASSISTANT, "Start with Python, learn statistics and linear algebra, then work through online courses and practice with datasets."});

        std::cout << "Updated Summary: " << summary_memory.get_summary() << std::endl;

        // 4. Token Buffer Memory with Window
        std::cout << "\n=== Token Buffer Memory ===" << std::endl;

        memory::ConversationTokenBufferMemory token_buffer_memory(llm);
        token_buffer_memory.set_max_token_limit(150);

        std::cout << "Created token buffer memory with 150 token limit" << std::endl;

        // Add a conversation that exceeds the limit
        std::vector<memory::Message> long_conversation = {
            {memory::MessageRole::USER, "Explain quantum computing"},
            {memory::MessageRole::ASSISTANT, "Quantum computing uses quantum mechanics principles like superposition and entanglement to process information in fundamentally different ways than classical computers."},
            {memory::MessageRole::USER, "What is superposition?"},
            {memory::MessageRole::ASSISTANT, "Superposition allows quantum bits (qubits) to exist in multiple states simultaneously, unlike classical bits which are either 0 or 1."},
            {memory::MessageRole::USER, "And entanglement?"},
            {memory::MessageRole::ASSISTANT, "Entanglement is a quantum phenomenon where qubits become correlated in such a way that the state of one instantly affects the state of another, regardless of distance."},
            {memory::MessageRole::USER, "What are the applications?"},
            {memory::MessageRole::ASSISTANT, "Potential applications include drug discovery, cryptography, optimization problems, and simulating quantum systems for scientific research."}
        };

        for (const auto& message : long_conversation) {
            token_buffer_memory.add_message(message);
            std::cout << "Added " << (message.role == memory::MessageRole::USER ? "User" : "AI")
                     << " message - Tokens: " << token_buffer_memory.current_token_count() << std::endl;
        }

        std::cout << "\nFinal token buffer contents:" << std::endl;
        std::cout << token_buffer_memory.get_memory_string() << std::endl;

        // 5. Knowledge Graph Memory
        std::cout << "\n=== Knowledge Graph Memory ===" << std::endl;

        memory::KnowledgeGraphMemory graph_memory;
        graph_memory.set_entity_extractor(llm);

        std::cout << "Created knowledge graph memory" << std::endl;

        // Add text with entities and relationships
        std::vector<std::string> knowledge_texts = {
            "Alan Turing was a British mathematician and computer scientist who is considered the father of theoretical computer science.",
            "He worked at Bletchley Park during World War II and helped break the Enigma code.",
            "The Turing Test was proposed by Alan Turing to determine if a machine can exhibit intelligent behavior.",
            "John von Neumann contributed to the development of computer architecture and game theory.",
            "Von Neumann worked with Turing at Princeton University in the 1930s."
        };

        for (const auto& text : knowledge_texts) {
            std::cout << "\nProcessing: " << text.substr(0, 80) << "..." << std::endl;
            graph_memory.add_knowledge(text);
        }

        // Show extracted entities and relationships
        std::cout << "\nExtracted Entities:" << std::endl;
        auto entities = graph_memory.get_entities();
        for (const auto& entity : entities) {
            std::cout << "  - " << entity << std::endl;
        }

        std::cout << "\nExtracted Relationships:" << std::endl;
        auto relationships = graph_memory.get_relationships();
        for (const auto& rel : relationships) {
            std::cout << "  - " << rel.subject << " --[" << rel.type << "]--> " << rel.object << std::endl;
        }

        // Query the knowledge graph
        std::cout << "\nKnowledge Graph Queries:" << std::endl;
        auto alan_turing_rels = graph_memory.get_relationships("Alan Turing");
        std::cout << "Relationships for 'Alan Turing':" << std::endl;
        for (const auto& rel : alan_turing_rels) {
            std::cout << "  - " << rel.subject << " --[" << rel.type << "]--> " << rel.object << std::endl;
        }

        auto bletchley_park_entities = graph_memory.get_related_entities("Bletchley Park");
        std::cout << "Entities related to 'Bletchley Park':" << std::endl;
        for (const auto& entity : bletchley_park_entities) {
            std::cout << "  - " << entity << std::endl;
        }

        // 6. Long-term Memory with Persistence
        std::cout << "\n=== Long-term Memory ===" << std::endl;

        memory::LongTermMemory long_term_memory;
        long_term_memory.set_persistence_path("/tmp/langchain_memory.json");
        long_term_memory.enable_auto_save(true);

        std::cout << "Created long-term memory with persistence" << std::endl;

        // Add episodic memories
        memory::MemoryEntry episode1{
            memory::MemoryType::EPISODIC,
            "First interaction with AI system",
            "User asked about basic ML concepts and got explanations of supervised and unsupervised learning.",
            std::chrono::system_clock::now(),
            0.8,  // importance score
            {{"topic", "machine_learning"}, {"difficulty", "beginner"}}
        };

        memory::MemoryEntry episode2{
            memory::MemoryType::EPISODIC,
            "Advanced quantum computing discussion",
            "User explored quantum computing concepts including superposition and entanglement with detailed explanations.",
            std::chrono::system_clock::now(),
            0.9,  // importance score
            {{"topic", "quantum_computing"}, {"difficulty", "advanced"}}
        };

        long_term_memory.add_memory(episode1);
        long_term_memory.add_memory(episode2);

        // Add semantic memories
        memory::MemoryEntry semantic1{
            memory::MemoryType::SEMANTIC,
            "Machine Learning Definition",
            "Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.",
            std::chrono::system_clock::now(),
            1.0,  // very important
            {{"category", "definition"}, {"domain", "AI"}}
        };

        long_term_memory.add_memory(semantic1);

        std::cout << "Added " << long_term_memory.total_memories() << " memories to long-term storage" << std::endl;

        // Search memories
        auto ml_memories = long_term_memory.search_memories("machine learning");
        std::cout << "\nFound " << ml_memories.size() << " memories about 'machine learning':" << std::endl;
        for (const auto& memory : ml_memories) {
            std::cout << "  - [" << memory.type << "] " << memory.title << std::endl;
        }

        auto recent_memories = long_term_memory.get_recent_memories(2);
        std::cout << "\nRecent memories:" << std::endl;
        for (const auto& memory : recent_memories) {
            std::cout << "  - " << memory.title << " (importance: " << memory.importance << ")" << std::endl;
        }

        // 7. Memory Performance Testing
        std::cout << "\n=== Memory Performance Testing ===" << std::endl;

        const int num_messages = 100;
        std::vector<memory::Message> test_messages;

        // Generate test messages
        for (int i = 0; i < num_messages; ++i) {
            memory::MessageRole role = (i % 2 == 0) ? memory::MessageRole::USER : memory::MessageRole::ASSISTANT;
            std::string content = "This is test message number " + std::to_string(i) +
                                 " with some additional content to simulate real conversation.";
            test_messages.push_back({role, content});
        }

        // Test buffer memory performance
        memory::ConversationBufferMemory perf_memory;

        auto buffer_start = std::chrono::high_resolution_clock::now();
        for (const auto& msg : test_messages) {
            perf_memory.add_message(msg);
        }
        auto buffer_end = std::chrono::high_resolution_clock::now();

        auto buffer_time = std::chrono::duration_cast<std::chrono::microseconds>(
            buffer_end - buffer_start).count();

        std::cout << "Buffer Memory Performance:" << std::endl;
        std::cout << "  Added " << num_messages << " messages in " << buffer_time << " μs" << std::endl;
        std::cout << "  Average: " << (double)buffer_time / num_messages << " μs per message" << std::endl;
        std::cout << "  Total messages: " << perf_memory.num_messages() << std::endl;
        std::cout << "  Total tokens: " << perf_memory.current_token_count() << std::endl;

        // Test memory retrieval
        auto retrieval_start = std::chrono::high_resolution_clock::now();
        auto memory_string = perf_memory.get_memory_string();
        auto retrieval_end = std::chrono::high_resolution_clock::now();

        auto retrieval_time = std::chrono::duration_cast<std::chrono::microseconds>(
            retrieval_end - retrieval_start).count();

        std::cout << "  Retrieval time: " << retrieval_time << " μs" << std::endl;
        std::cout << "  Memory string length: " << memory_string.length() << " characters" << std::endl;

        // 8. Memory Integration with Chains
        std::cout << "\n=== Memory-Backed Chains ===" << std::endl;

        // Create a memory-backed LLM chain
        auto memory_template = std::make_shared<prompts::PromptTemplate>(
            "You are having a conversation with a user. Use the conversation history to provide contextual responses.\n\n"
            "Conversation History:\n{memory}\n\n"
            "Current User Message: {user_input}\n\n"
            "Response:",
            {{"memory", "user_input"}}
        );

        chains::LLMChain memory_chain(llm, memory_template);

        // Simulate a contextual conversation
        std::vector<std::string> contextual_inputs = {
            "My name is Sarah and I'm a software developer.",
            "What programming languages should I learn?",
            "Actually, I'm most interested in AI and machine learning.",
            "Can you recommend some Python libraries for ML?",
            "Thanks! By the way, what did I say my name was?"
        };

        for (const auto& input : contextual_inputs) {
            std::cout << "\nUser: " << input << std::endl;
            std::cout << "----------------------------------------" << std::endl;

            // Get current memory context
            std::string memory_context = buffer_memory.get_memory_string();

            chains::ChainInput chain_input = {
                {"memory", memory_context},
                {"user_input", input}
            };

            auto chain_output = memory_chain.run(chain_input);
            std::string response = std::any_cast<std::string>(chain_output["text"]);

            std::cout << "Assistant: " << response << std::endl;

            // Add to memory
            memory::Message user_msg{memory::MessageRole::USER, input};
            memory::Message ai_msg{memory::MessageRole::ASSISTANT, response};
            buffer_memory.add_message(user_msg);
            buffer_memory.add_message(ai_msg);
        }

        // 9. Memory Statistics and Management
        std::cout << "\n=== Memory Statistics ===" << std::endl;

        std::cout << "Buffer Memory:" << std::endl;
        std::cout << "  Messages: " << buffer_memory.num_messages() << std::endl;
        std::cout << "  Tokens: " << buffer_memory.current_token_count() << std::endl;
        std::cout << "  Max tokens: " << buffer_memory.max_tokens() << std::endl;
        std::cout << "  Utilization: " << std::fixed << std::setprecision(1)
                 << (double)buffer_memory.current_token_count() / buffer_memory.max_tokens() * 100 << "%" << std::endl;

        std::cout << "\nSummary Memory:" << std::endl;
        std::cout << "  Summary length: " << summary_memory.get_summary().length() << " characters" << std::endl;
        std::cout << "  Context messages: " << summary_memory.num_context_messages() << std::endl;

        std::cout << "\nKnowledge Graph:" << std::endl;
        std::cout << "  Entities: " << graph_memory.entity_count() << std::endl;
        std::cout << "  Relationships: " << graph_memory.relationship_count() << std::endl;

        std::cout << "\nLong-term Memory:" << std::endl;
        std::cout << "  Total memories: " << long_term_memory.total_memories() << std::endl;
        std::cout << "  Episodic: " << long_term_memory.count_by_type(memory::MemoryType::EPISODIC) << std::endl;
        std::cout << "  Semantic: " << long_term_memory.count_by_type(memory::MemoryType::SEMANTIC) << std::endl;

        // Clear some memory
        std::cout << "\n=== Memory Management ===" << std::endl;
        std::cout << "Clearing buffer memory..." << std::endl;
        buffer_memory.clear();
        std::cout << "Buffer memory cleared. Messages: " << buffer_memory.num_messages() << std::endl;

        // Save memory states
        std::cout << "\nSaving memory states..." << std::endl;
        bool saved = long_term_memory.save_to_disk();
        std::cout << "Memory save " << (saved ? "successful" : "failed") << std::endl;

        std::cout << "\n=== Memory Systems Example completed successfully! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}