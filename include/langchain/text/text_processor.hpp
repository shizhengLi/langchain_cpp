#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <locale>
#include <codecvt>
#include <regex>

namespace langchain::text {

/**
 * @brief Text processing utilities for document retrieval
 *
 * This class provides comprehensive text processing capabilities including
 * tokenization, stop word filtering, stemming, and normalization.
 */
class TextProcessor {
public:
    /**
     * @brief Configuration for text processing
     */
    struct Config {
        bool lowercase = true;
        bool remove_punctuation = true;
        bool remove_numbers = false;
        bool remove_stopwords = true;
        bool enable_stemming = false;
        std::string language = "english";
        size_t min_token_length = 2;
        size_t max_token_length = 50;
    };

private:
    Config config_;
    std::unordered_set<std::string> stopwords_;
    std::locale locale_;
    std::wregex word_regex_;
    std::wregex punctuation_regex_;
    std::wregex number_regex_;

public:
    /**
     * @brief Constructor
     * @param config Processing configuration
     */
    explicit TextProcessor(const Config& config = Config{});

    /**
     * @brief Tokenize text into individual tokens
     * @param text Input text
     * @return Vector of tokens
     */
    std::vector<std::string> tokenize(const std::string& text);

    /**
     * @brief Process text with full pipeline
     * @param text Input text
     * @return Vector of processed tokens
     */
    std::vector<std::string> process(const std::string& text);

    /**
     * @brief Normalize text (lowercase, remove accents, etc.)
     * @param text Input text
     * @return Normalized text
     */
    std::string normalize(const std::string& text);

    /**
     * @brief Filter stop words from tokens
     * @param tokens Input tokens
     * @return Filtered tokens
     */
    std::vector<std::string> filter_stopwords(const std::vector<std::string>& tokens);

    /**
     * @brief Apply stemming to tokens
     * @param tokens Input tokens
     * @return Stemmed tokens
     */
    std::vector<std::string> stem(const std::vector<std::string>& tokens);

    /**
     * @brief Extract n-grams from tokens
     * @param tokens Input tokens
     * @param n N-gram size
     * @return Vector of n-grams
     */
    std::vector<std::string> extract_ngrams(const std::vector<std::string>& tokens, size_t n);

    /**
     * @brief Check if a token is a stop word
     * @param token Token to check
     * @return True if token is a stop word
     */
    bool is_stopword(const std::string& token) const;

    /**
     * @brief Get stop words for a language
     * @param language Language code
     * @return Set of stop words
     */
    static std::unordered_set<std::string> get_stopwords(const std::string& language);

    /**
     * @brief Simple Porter Stemmer implementation
     * @param word Input word
     * @return Stemmed word
     */
    static std::string porter_stem(const std::string& word);

    /**
     * @brief Update configuration
     * @param config New configuration
     */
    void set_config(const Config& config) { config_ = config; }

    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const Config& get_config() const { return config_; }

private:
    void initialize_regexes();
    void initialize_stopwords();
    std::wstring utf8_to_wstring(const std::string& str);
    std::string wstring_to_utf8(const std::wstring& wstr);
    bool is_punctuation(wchar_t ch);
    bool is_valid_token(const std::string& token);
};

/**
 * @brief Factory for creating pre-configured text processors
 */
class TextProcessorFactory {
public:
    /**
     * @brief Create a processor optimized for document retrieval
     */
    static std::unique_ptr<TextProcessor> create_retrieval_processor();

    /**
     * @brief Create a processor optimized for search queries
     */
    static std::unique_ptr<TextProcessor> create_search_processor();

    /**
     * @brief Create a minimal processor (tokenization only)
     */
    static std::unique_ptr<TextProcessor> create_minimal_processor();
};

} // namespace langchain::text