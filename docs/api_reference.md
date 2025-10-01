# LangChain++ API Reference

## Overview

This document provides a comprehensive API reference for the LangChain++ C++ library, including all major components, classes, and functions.

## Table of Contents

- [Core Types](#core-types)
- [Text Processing](#text-processing)
- [Retrieval System](#retrieval-system)
- [Vector Storage](#vector-storage)
- [LLM Integration](#llm-integration)
- [Chain System](#chain-system)
- [Memory System](#memory-system)
- [Security](#security)
- [Monitoring](#monitoring)
- [Persistence](#persistence)

---

## Core Types

### Document

```cpp
struct Document {
    std::string content;
    std::unordered_map<std::string, std::string> metadata;
    std::string id;

    Document();  // Default constructor
    Document(const std::string& content,
             const std::unordered_map<std::string, std::string>& metadata = {});
    Document(const std::string& content,
             const std::unordered_map<std::string, std::string>& metadata,
             const std::string& id);
};
```

**Description**: Represents a document with content and metadata.

**Members**:
- `content`: The main text content of the document
- `metadata`: Key-value pairs for additional document information
- `id`: Unique identifier for the document

**Example**:
```cpp
Document doc("Hello world", {{"source", "test"}, {"page", "1"}});
```

### RetrievalConfig

```cpp
struct RetrievalConfig {
    size_t top_k = 5;
    std::optional<double> score_threshold;
    std::string search_type = "similarity";
    double mmr_lambda = 0.5;
    size_t fetch_k = 20;
    bool enable_caching = true;
    size_t cache_size = 1000;
    std::unordered_map<std::string, std::string> params;
};
```

**Description**: Configuration for retrieval operations.

**Members**:
- `top_k`: Number of top results to return
- `score_threshold`: Minimum similarity score threshold
- `search_type`: Type of search ("similarity", "mmr", "hybrid", "tfidf", "bm25")
- `mmr_lambda`: Lambda parameter for MMR (Maximal Marginal Relevance)
- `fetch_k`: Number of candidates to fetch for MMR
- `enable_caching`: Whether to enable result caching
- `cache_size`: Maximum number of cached results
- `params`: Additional parameters as key-value pairs

---

## Text Processing

### TextProcessor

```cpp
class TextProcessor {
public:
    TextProcessor();
    ~TextProcessor();

    // Tokenization
    std::vector<std::string> tokenize(const std::string& text) const;
    std::vector<std::string> tokenize_with_positions(const std::string& text,
                                                     std::vector<size_t>& positions) const;

    // Text analysis
    size_t count_tokens(const std::string& text) const;
    std::unordered_map<std::string, size_t> get_token_frequencies(const std::string& text) const;

    // Language detection
    std::string detect_language(const std::string& text) const;
    bool is_english(const std::string& text) const;

    // Text cleaning
    std::string clean_text(const std::string& text) const;
    std::string normalize_text(const std::string& text) const;

    // Advanced processing
    std::vector<std::string> extract_ngrams(const std::string& text, size_t n) const;
    std::vector<std::string> extract_sentences(const std::string& text) const;

    // Configuration
    void set_language(const std::string& language);
    void set_tokenizer_type(TokenizerType type);
    void set_stop_words(const std::unordered_set<std::string>& stop_words);
};
```

**Description**: Main class for text processing operations including tokenization, analysis, and cleaning.

**Key Methods**:
- `tokenize()`: Splits text into tokens
- `count_tokens()`: Counts number of tokens in text
- `get_token_frequencies()`: Returns frequency map of tokens
- `detect_language()`: Detects the language of the text
- `clean_text()`: Removes unwanted characters and normalizes text

**Example**:
```cpp
TextProcessor processor;
processor.set_language("english");
auto tokens = processor.tokenize("Hello, world! This is a test.");
size_t count = processor.count_tokens("Hello world");
```

---

## Retrieval System

### BaseRetriever

```cpp
class BaseRetriever {
public:
    virtual ~BaseRetriever() = default;

    // Core retrieval methods
    virtual std::vector<Document> retrieve(const std::string& query,
                                        const RetrievalConfig& config = {}) = 0;
    virtual std::vector<Document> retrieve_with_scores(const std::string& query,
                                                     std::vector<double>& scores,
                                                     const RetrievalConfig& config = {}) = 0;

    // Document management
    virtual bool add_document(const Document& document) = 0;
    virtual bool add_documents(const std::vector<Document>& documents) = 0;
    virtual bool remove_document(const std::string& doc_id) = 0;
    virtual std::optional<Document> get_document(const std::string& doc_id) = 0;
    virtual std::vector<Document> get_all_documents() const = 0;

    // Index management
    virtual bool build_index() = 0;
    virtual bool save_index(const std::string& path) = 0;
    virtual bool load_index(const std::string& path) = 0;
    virtual size_t size() const = 0;
    virtual bool empty() const = 0;

    // Configuration
    virtual void configure(const RetrievalConfig& config) = 0;
    virtual RetrievalConfig get_config() const = 0;
};
```

**Description**: Abstract base class for all retrieval implementations.

### InvertedIndexRetriever

```cpp
class InvertedIndexRetriever : public BaseRetriever {
public:
    InvertedIndexRetriever();
    explicit InvertedIndexRetriever(const RetrievalConfig& config);
    ~InvertedIndexRetriever() override;

    // BaseRetriever implementation
    std::vector<Document> retrieve(const std::string& query,
                                const RetrievalConfig& config = {}) override;
    std::vector<Document> retrieve_with_scores(const std::string& query,
                                             std::vector<double>& scores,
                                             const RetrievalConfig& config = {}) override;

    bool add_document(const Document& document) override;
    bool add_documents(const std::vector<Document>& documents) override;
    bool remove_document(const std::string& doc_id) override;

    // Index operations
    bool build_index() override;
    bool build_index_parallel(size_t num_threads = 0);
    bool save_index(const std::string& path) override;
    bool load_index(const std::string& path) override;

    // Statistics
    size_t size() const override;
    size_t vocabulary_size() const;
    size_t total_tokens() const;
    double average_document_length() const;

    // Advanced features
    void set_text_processor(std::shared_ptr<TextProcessor> processor);
    void enable_stemming(bool enable);
    void enable_stop_words(bool enable);
    void set_min_term_frequency(size_t min_freq);
    void set_max_term_frequency(double max_ratio);
};
```

**Description**: Inverted index-based retriever for keyword search.

**Key Features**:
- Full-text search with TF-IDF scoring
- Stemming and stop word removal
- Parallel index building
- Configurable text processing pipeline

**Example**:
```cpp
InvertedIndexRetriever retriever;
retriever.enable_stemming(true);
retriever.enable_stop_words(true);

Document doc("Information retrieval is the process of obtaining relevant information.");
retriever.add_document(doc);

auto results = retriever.retrieve("information process");
```

### BM25Retriever

```cpp
class BM25Retriever : public BaseRetriever {
public:
    BM25Retriever();
    explicit BM25Retriever(const RetrievalConfig& config);
    ~BM25Retriever() override;

    // BM25-specific configuration
    void set_k1(double k1);
    void set_b(double b);
    void set_delta(double delta);

    // BaseRetriever implementation
    std::vector<Document> retrieve(const std::string& query,
                                const RetrievalConfig& config = {}) override;
    std::vector<Document> retrieve_with_scores(const std::string& query,
                                             std::vector<double>& scores,
                                             const RetrievalConfig& config = {}) override;

    // Advanced BM25 features
    void enable_query_expansion(bool enable);
    void enable_query_likelihood(bool enable);
    void set_field_weights(const std::unordered_map<std::string, double>& weights);

    // Statistics
    double calculate_bm25_score(const std::string& query, const Document& doc) const;
    std::unordered_map<std::string, double> get_term_scores(const std::string& query) const;
};
```

**Description**: BM25 (Best Matching 25) retriever for probabilistic relevance ranking.

**BM25 Parameters**:
- `k1`: Term frequency saturation parameter (default: 1.2)
- `b`: Document length normalization parameter (default: 0.75)
- `delta`: Query term weight parameter (default: 1.0)

**Example**:
```cpp
BM25Retriever retriever;
retriever.set_k1(1.5);
retriever.set_b(0.8);

// Add documents and build index
retriever.add_documents(documents);
retriever.build_index();

// Search with BM25 scoring
auto results = retriever.retrieve("machine learning algorithms");
```

### HybridRetriever

```cpp
class HybridRetriever : public BaseRetriever {
public:
    enum class FusionMethod {
        RECIPROCAL_RANK,
        CONDENSATION,
        WEIGHTED_SUM,
        RRf (Reciprocal Rank Fusion)
    };

    HybridRetriever();
    HybridRetriever(std::vector<std::shared_ptr<BaseRetriever>> retrievers);
    ~HybridRetriever() override;

    // Retriever management
    void add_retriever(std::shared_ptr<BaseRetriever> retriever, double weight = 1.0);
    void remove_retriever(size_t index);
    void set_retriever_weight(size_t index, double weight);

    // Fusion methods
    void set_fusion_method(FusionMethod method);
    void set_fusion_parameters(const std::unordered_map<std::string, double>& params);

    // BaseRetriever implementation
    std::vector<Document> retrieve(const std::string& query,
                                const RetrievalConfig& config = {}) override;
    std::vector<Document> retrieve_with_scores(const std::string& query,
                                             std::vector<double>& scores,
                                             const RetrievalConfig& config = {}) override;

    // Advanced features
    void enable_diversification(bool enable);
    void set_diversification_lambda(double lambda);
    void enable_query_classification(bool enable);
};
```

**Description**: Hybrid retriever that combines multiple retrieval strategies.

**Fusion Methods**:
- `RECIPROCAL_RANK`: Reciprocal rank fusion
- `CONDENSATION`: Condensation fusion
- `WEIGHTED_SUM`: Weighted sum of scores
- `RRf`: Reciprocal Rank Fusion with improved scoring

**Example**:
```cpp
HybridRetriever hybrid;
auto vector_retriever = std::make_shared<VectorStoreRetriever>();
auto bm25_retriever = std::make_shared<BM25Retriever>();

hybrid.add_retriever(vector_retriever, 0.6);  // 60% weight
hybrid.add_retriever(bm25_retriever, 0.4);    // 40% weight
hybrid.set_fusion_method(HybridRetriever::FusionMethod::RECIPROCAL_RANK);

auto results = hybrid.retrieve("deep learning transformers");
```

---

## Vector Storage

### SimpleVectorStore

```cpp
class SimpleVectorStore {
public:
    SimpleVectorStore();
    explicit SimpleVectorStore(size_t dimensions);
    ~SimpleVectorStore();

    // Vector operations
    bool add_vector(const std::string& id, const std::vector<double>& vector);
    bool add_vectors(const std::vector<std::pair<std::string, std::vector<double>>>& vectors);
    bool remove_vector(const std::string& id);
    std::optional<std::vector<double>> get_vector(const std::string& id) const;

    // Similarity search
    std::vector<std::pair<std::string, double>> search(
        const std::vector<double>& query_vector,
        size_t k = 10,
        const std::string& metric = "cosine"
    ) const;

    // Batch operations
    std::vector<std::pair<std::string, double>> search_multiple(
        const std::vector<std::vector<double>>& query_vectors,
        size_t k = 10,
        const std::string& metric = "cosine"
    ) const;

    // Index management
    bool build_index(const std::string& index_type = "hnsw");
    bool save_index(const std::string& path) const;
    bool load_index(const std::string& path);

    // Configuration
    void set_dimensions(size_t dimensions);
    void set_index_parameters(const std::unordered_map<std::string, double>& params);

    // Statistics
    size_t size() const;
    size_t dimensions() const;
    double memory_usage() const;
};
```

**Description**: Simple vector store for similarity search with multiple distance metrics.

**Distance Metrics**:
- `cosine`: Cosine similarity
- `euclidean`: Euclidean distance
- `manhattan`: Manhattan distance
- `dot_product`: Dot product similarity

**Example**:
```cpp
SimpleVectorStore store(384);  // 384 dimensions

std::vector<double> vector = {0.1, 0.2, 0.3, /* ... */};
store.add_vector("doc1", vector);

auto results = store.search(vector, 5, "cosine");
```

---

## LLM Integration

### BaseLLM

```cpp
class BaseLLM {
public:
    virtual ~BaseLLM() = default;

    // Core LLM methods
    virtual std::string generate(const std::string& prompt,
                               const GenerationConfig& config = {}) = 0;
    virtual std::string generate(const std::vector<Message>& messages,
                               const GenerationConfig& config = {}) = 0;

    // Streaming generation
    virtual void generate_stream(const std::string& prompt,
                                std::function<void(const std::string&)> callback,
                                const GenerationConfig& config = {}) = 0;
    virtual void generate_stream(const std::vector<Message>& messages,
                                std::function<void(const std::string&)> callback,
                                const GenerationConfig& config = {}) = 0;

    // Token operations
    virtual size_t count_tokens(const std::string& text) const = 0;
    virtual size_t count_tokens(const std::vector<Message>& messages) const = 0;
    virtual std::vector<int> tokenize(const std::string& text) const = 0;

    // Configuration
    virtual void configure(const LLMConfig& config) = 0;
    virtual LLMConfig get_config() const = 0;

    // Capabilities
    virtual bool supports_streaming() const = 0;
    virtual bool supports_function_calling() const = 0;
    virtual bool supports_vision() const = 0;
    virtual std::vector<std::string> supported_models() const = 0;
};
```

**Description**: Abstract base class for LLM implementations.

### OpenAILLM

```cpp
class OpenAILLM : public BaseLLM {
public:
    OpenAILLM();
    explicit OpenAILLM(const std::string& api_key);
    explicit OpenAILLM(const OpenAIConfig& config);
    ~OpenAILLM() override;

    // Configuration
    void set_api_key(const std::string& api_key);
    void set_base_url(const std::string& base_url);
    void set_model(const std::string& model);
    void set_timeout(std::chrono::seconds timeout);
    void set_max_retries(size_t max_retries);

    // BaseLLM implementation
    std::string generate(const std::string& prompt,
                       const GenerationConfig& config = {}) override;
    std::string generate(const std::vector<Message>& messages,
                       const GenerationConfig& config = {}) override;

    void generate_stream(const std::string& prompt,
                        std::function<void(const std::string&)> callback,
                        const GenerationConfig& config = {}) override;

    // OpenAI-specific features
    void enable_function_calling(bool enable);
    void register_function(const FunctionDefinition& function);
    std::string call_function(const std::string& function_name,
                             const std::string& arguments);

    // Async operations
    std::future<std::string> generate_async(const std::string& prompt,
                                          const GenerationConfig& config = {});

    // Usage tracking
    UsageInfo get_last_usage() const;
    double estimate_cost(const UsageInfo& usage) const;
};
```

**Description**: OpenAI API implementation for LLM integration.

**Example**:
```cpp
OpenAILLM llm("your-api-key");
llm.set_model("gpt-4");

GenerationConfig config;
config.max_tokens = 1000;
config.temperature = 0.7;

std::string response = llm.generate("Explain quantum computing in simple terms.", config);

// Streaming
llm.generate_stream("Write a story about...",
    [](const std::string& chunk) {
        std::cout << chunk << std::flush;
    }, config);
```

---

## Chain System

### BaseChain

```cpp
class BaseChain {
public:
    virtual ~BaseChain() = default;

    // Core chain methods
    virtual ChainOutput run(const ChainInput& input) = 0;
    virtual ChainOutput run(const std::unordered_map<std::string, std::any>& input) = 0;

    // Batch processing
    virtual std::vector<ChainOutput> run_batch(
        const std::vector<ChainInput>& inputs) = 0;

    // Configuration
    virtual void configure(const ChainConfig& config) = 0;
    virtual ChainConfig get_config() const = 0;

    // Metadata
    virtual std::string get_chain_type() const = 0;
    virtual std::vector<std::string> get_input_keys() const = 0;
    virtual std::vector<std::string> get_output_keys() const = 0;

    // Validation
    virtual bool validate_input(const ChainInput& input) const = 0;
    virtual bool validate_output(const ChainOutput& output) const = 0;
};
```

### LLMChain

```cpp
class LLMChain : public BaseChain {
public:
    LLMChain();
    LLMChain(std::shared_ptr<BaseLLM> llm,
            std::shared_ptr<PromptTemplate> prompt_template);
    ~LLMChain() override;

    // Configuration
    void set_llm(std::shared_ptr<BaseLLM> llm);
    void set_prompt_template(std::shared_ptr<PromptTemplate> prompt_template);
    void set_output_parser(std::shared_ptr<OutputParser> parser);

    // BaseChain implementation
    ChainOutput run(const ChainInput& input) override;
    ChainOutput run(const std::unordered_map<std::string, std::any>& input) override;

    // Advanced features
    void enable_memory(bool enable);
    void set_memory(std::shared_ptr<BaseMemory> memory);
    void enable_callback(bool enable);
    void set_callback(std::shared_ptr<BaseCallbackHandler> callback);

    // Streaming
    void run_stream(const ChainInput& input,
                   std::function<void(const std::string&)> callback);
};
```

**Example**:
```cpp
auto llm = std::make_shared<OpenAILLM>("api-key");
auto prompt = std::make_shared<PromptTemplate>(
    "Explain {topic} in simple terms:",
    {"topic"}
);

LLMChain chain(llm, prompt);

ChainInput input = {{"topic", "artificial intelligence"}};
auto output = chain.run(input);
std::cout << std::any_cast<std::string>(output["text"]) << std::endl;
```

---

## Security

### SecurityManager

```cpp
class SecurityManager {
public:
    explicit SecurityManager(const SecurityConfig& config = {});
    ~SecurityManager() = default;

    // Lifecycle
    bool initialize();
    void shutdown();

    // Authentication
    std::optional<Session> login(const std::string& username,
                               const std::string& password,
                               const std::string& ip_address = "",
                               const std::string& user_agent = "");
    bool logout(const std::string& session_token);
    bool validate_session(const std::string& session_token);
    std::optional<User> get_current_user(const std::string& session_token);

    // Authorization
    bool check_permission(const std::string& session_token,
                         const std::string& resource,
                         PermissionType permission);

    // User management
    bool create_user(const User& user);
    std::optional<User> get_user(const std::string& user_id);
    bool update_user(const User& user);
    bool delete_user(const std::string& user_id);

    // Encryption
    std::string encrypt_data(const std::string& data);
    std::string decrypt_data(const std::string& encrypted_data);

    // Configuration
    void update_config(const SecurityConfig& config);
    SecurityConfig get_config() const;

    // Health check
    bool is_healthy() const;
    std::unordered_map<std::string, std::string> get_stats() const;
};
```

**Example**:
```cpp
SecurityConfig config;
config.level = SecurityLevel::HIGH;
config.enable_encryption = true;

SecurityManager security(config);
security.initialize();

// Create user
User user;
user.username = "alice";
user.credentials.password_hash = "SecurePassword123";
security.create_user(user);

// Login
auto session = security.login("alice", "SecurePassword123", "127.0.0.1");
if (session) {
    std::cout << "Login successful!" << std::endl;
}
```

---

## Monitoring

### MetricsCollector

```cpp
class MetricsCollector {
public:
    MetricsCollector();
    ~MetricsCollector();

    // Counter operations
    void increment_counter(const std::string& name, uint64_t value = 1);
    uint64_t get_counter(const std::string& name) const;

    // Gauge operations
    void set_gauge(const std::string& name, double value);
    double get_gauge(const std::string& name) const;

    // Histogram operations
    void record_histogram(const std::string& name, double value);
    HistogramStats get_histogram_stats(const std::string& name) const;

    // Timer operations
    std::unique_ptr<Timer> create_timer(const std::string& name);
    void record_timing(const std::string& name, std::chrono::milliseconds duration);

    // Batch operations
    std::unordered_map<std::string, uint64_t> get_all_counters() const;
    std::unordered_map<std::string, double> get_all_gauges() const;

    // Export
    std::string export_prometheus_format() const;
    std::string export_json_format() const;

    // Reset
    void reset_all();
    void reset_metric(const std::string& name);
};
```

**Example**:
```cpp
MetricsCollector metrics;

// Record metrics
metrics.increment_counter("requests_total");
metrics.set_gauge("active_connections", 42.0);
metrics.record_histogram("response_time_ms", 150.5);

// Timer usage
{
    auto timer = metrics.create_timer("operation_time");
    // ... do work ...
} // Timer automatically records when it goes out of scope
```

---

## Error Handling

Most LangChain++ functions use standard C++ exception handling:

```cpp
try {
    InvertedIndexRetriever retriever;
    retriever.add_documents(documents);
    auto results = retriever.retrieve("search query");
} catch (const RetrievalException& e) {
    std::cerr << "Retrieval error: " << e.what() << std::endl;
} catch (const std::exception& e) {
    std::cerr << "General error: " << e.what() << std::endl;
}
```

### Common Exception Types

- `RetrievalException`: Retrieval-related errors
- `LLMException`: LLM operation errors
- `SecurityException`: Security-related errors
- `ConfigurationException`: Configuration errors
- `ValidationException`: Input validation errors

---

## Thread Safety

Most LangChain++ classes are designed to be thread-safe:

```cpp
// Safe to share across threads
std::shared_ptr<InvertedIndexRetriever> retriever = std::make_shared<InvertedIndexRetriever>();

std::vector<std::thread> threads;
for (int i = 0; i < 4; ++i) {
    threads.emplace_back([retriever, i]() {
        auto results = retriever.retrieve("query " + std::to_string(i));
        // Process results...
    });
}

for (auto& thread : threads) {
    thread.join();
}
```

---

## Performance Tips

1. **Reuse Objects**: Create retrievers and LLM instances once and reuse them
2. **Batch Operations**: Use batch methods when processing multiple items
3. **Parallel Processing**: Use built-in parallel methods for large operations
4. **Memory Management**: Monitor memory usage with built-in metrics
5. **Caching**: Enable caching for repeated queries

---

*This API reference covers the main components of LangChain++. For more detailed examples and advanced usage, see the other documentation files.*