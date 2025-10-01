#include <catch2/catch_all.hpp>
#include "langchain/text/text_processor.hpp"
#include <memory>

using namespace langchain::text;

TEST_CASE("TextProcessor - Default Configuration", "[text][processor]") {
    TextProcessor processor({});

    SECTION("Default config values") {
        auto config = processor.get_config();
        REQUIRE(config.lowercase == true);
        REQUIRE(config.remove_punctuation == true);
        REQUIRE(config.remove_numbers == false);
        REQUIRE(config.remove_stopwords == true);
        REQUIRE(config.enable_stemming == false);
        REQUIRE(config.language == "english");
        REQUIRE(config.min_token_length == 2);
        REQUIRE(config.max_token_length == 50);
    }
}

TEST_CASE("TextProcessor - Tokenization", "[text][processor][tokenize]") {
    TextProcessor processor({});

    SECTION("Basic tokenization") {
        auto tokens = processor.tokenize("Hello world! This is a test.");
        REQUIRE(tokens.size() >= 5);  // At least: hello, world, test

        // Check that common words are present
        std::vector<std::string> expected = {"hello", "world", "test"};
        for (const auto& word : expected) {
            REQUIRE(std::find(tokens.begin(), tokens.end(), word) != tokens.end());
        }
    }

    SECTION("Empty string") {
        auto tokens = processor.tokenize("");
        REQUIRE(tokens.empty());
    }

    SECTION("Whitespace only") {
        auto tokens = processor.tokenize("   \t\n   ");
        REQUIRE(tokens.empty());
    }

    SECTION("Multiple spaces") {
        auto tokens = processor.tokenize("Hello    world   test");
        REQUIRE(tokens.size() == 3);
        REQUIRE(tokens[0] == "hello");
        REQUIRE(tokens[1] == "world");
        REQUIRE(tokens[2] == "test");
    }

    SECTION("Punctuation handling") {
        auto tokens = processor.tokenize("Hello, world! How are you? I'm fine.");
        REQUIRE(tokens.size() >= 4);  // hello, world, fine, etc.
    }

    SECTION("Numbers") {
        auto tokens = processor.tokenize("There are 123 cats and 45 dogs");
        REQUIRE(std::find(tokens.begin(), tokens.end(), "123") != tokens.end());
        REQUIRE(std::find(tokens.begin(), tokens.end(), "45") != tokens.end());
    }

    SECTION("Mixed case") {
        auto tokens = processor.tokenize("HeLLo WoRLd");
        REQUIRE(std::find(tokens.begin(), tokens.end(), "hello") != tokens.end());
        REQUIRE(std::find(tokens.begin(), tokens.end(), "world") != tokens.end());
    }
}

TEST_CASE("TextProcessor - Stop Words", "[text][processor][stopwords]") {
    TextProcessor processor({});

    SECTION("English stop words") {
        REQUIRE(processor.is_stopword("the"));
        REQUIRE(processor.is_stopword("is"));
        REQUIRE(processor.is_stopword("and"));
        REQUIRE(processor.is_stopword("a"));
        REQUIRE(processor.is_stopword("an"));
    }

    SECTION("Non-stop words") {
        REQUIRE_FALSE(processor.is_stopword("computer"));
        REQUIRE_FALSE(processor.is_stopword("algorithm"));
        REQUIRE_FALSE(processor.is_stopword("programming"));
    }

    SECTION("Case insensitive stop words") {
        REQUIRE(processor.is_stopword("The"));
        REQUIRE(processor.is_stopword("IS"));
        REQUIRE(processor.is_stopword("AND"));
    }

    SECTION("Filter stop words") {
        std::vector<std::string> tokens = {"this", "is", "a", "test", "document"};
        auto filtered = processor.filter_stopwords(tokens);

        // Should remove: this, is, a
        // Should keep: test, document
        REQUIRE(std::find(filtered.begin(), filtered.end(), "test") != filtered.end());
        REQUIRE(std::find(filtered.begin(), filtered.end(), "document") != filtered.end());
        REQUIRE(std::find(filtered.begin(), filtered.end(), "this") == filtered.end());
        REQUIRE(std::find(filtered.begin(), filtered.end(), "is") == filtered.end());
        REQUIRE(std::find(filtered.begin(), filtered.end(), "a") == filtered.end());
    }

    SECTION("Stop word factory") {
        auto stopwords = TextProcessor::get_stopwords("english");
        REQUIRE(stopwords.size() > 100);  // English has many stop words
        REQUIRE(stopwords.find("the") != stopwords.end());
        REQUIRE(stopwords.find("is") != stopwords.end());
    }
}

TEST_CASE("TextProcessor - Porter Stemmer", "[text][processor][stemming]") {
    SECTION("Basic stemming") {
        REQUIRE(TextProcessor::porter_stem("running") == "run");
        REQUIRE(TextProcessor::porter_stem("runs") == "run");
        REQUIRE(TextProcessor::porter_stem("ran") == "ran");
        REQUIRE(TextProcessor::porter_stem("easily") == "easili");
        REQUIRE(TextProcessor::porter_stem("organization") == "organ");
    }

    SECTION("Stemming with stemming disabled") {
        TextProcessor::Config config;
        config.enable_stemming = false;
        TextProcessor processor(config);

        std::vector<std::string> tokens = {"running", "jumps", "quickly"};
        auto stemmed = processor.stem(tokens);

        // Should remain unchanged when stemming is disabled
        REQUIRE(stemmed[0] == "running");
        REQUIRE(stemmed[1] == "jumps");
        REQUIRE(stemmed[2] == "quickly");
    }

    SECTION("Stemming with stemming enabled") {
        TextProcessor::Config config;
        config.enable_stemming = true;
        TextProcessor processor(config);

        std::vector<std::string> tokens = {"running", "jumps", "quickly"};
        auto stemmed = processor.stem(tokens);

        // Should be stemmed
        REQUIRE(stemmed[0] == "run");
        REQUIRE(stemmed[1] == "jump");
        REQUIRE(stemmed[2] == "quickli");
    }
}

TEST_CASE("TextProcessor - N-grams", "[text][processor][ngrams]") {
    TextProcessor processor({});

    SECTION("Bigrams") {
        std::vector<std::string> tokens = {"the", "quick", "brown", "fox"};
        auto bigrams = processor.extract_ngrams(tokens, 2);

        REQUIRE(bigrams.size() == 3);
        REQUIRE(bigrams[0] == "the_quick");
        REQUIRE(bigrams[1] == "quick_brown");
        REQUIRE(bigrams[2] == "brown_fox");
    }

    SECTION("Trigrams") {
        std::vector<std::string> tokens = {"the", "quick", "brown", "fox"};
        auto trigrams = processor.extract_ngrams(tokens, 3);

        REQUIRE(trigrams.size() == 2);
        REQUIRE(trigrams[0] == "the_quick_brown");
        REQUIRE(trigrams[1] == "quick_brown_fox");
    }

    SECTION("N-gram size larger than tokens") {
        std::vector<std::string> tokens = {"hello", "world"};
        auto ngrams = processor.extract_ngrams(tokens, 5);
        REQUIRE(ngrams.empty());
    }

    SECTION("Single token") {
        std::vector<std::string> tokens = {"hello"};
        auto bigrams = processor.extract_ngrams(tokens, 2);
        REQUIRE(bigrams.empty());
    }
}

TEST_CASE("TextProcessor - Full Processing Pipeline", "[text][processor][pipeline]") {
    SECTION("Complete processing") {
        TextProcessor::Config config;
        config.enable_stemming = true;  // Enable stemming for this test
        TextProcessor processor(config);
        std::string text = "The quick brown foxes are running and jumping!";

        auto processed = processor.process(text);

        // Should contain meaningful tokens, not stop words
        REQUIRE(processed.size() > 0);
        REQUIRE(std::find(processed.begin(), processed.end(), "quick") != processed.end());
        REQUIRE(std::find(processed.begin(), processed.end(), "brown") != processed.end());
        REQUIRE(std::find(processed.begin(), processed.end(), "fox") != processed.end());  // foxes -> fox
        REQUIRE(std::find(processed.begin(), processed.end(), "run") != processed.end());   // running -> run
        REQUIRE(std::find(processed.begin(), processed.end(), "jump") != processed.end());  // jumping -> jump

        // Should not contain stop words
        REQUIRE(std::find(processed.begin(), processed.end(), "the") == processed.end());
        REQUIRE(std::find(processed.begin(), processed.end(), "are") == processed.end());
        REQUIRE(std::find(processed.begin(), processed.end(), "and") == processed.end());
    }

    SECTION("Processing with different configurations") {
        TextProcessor::Config config;
        config.lowercase = false;
        config.remove_punctuation = false;
        config.remove_stopwords = false;
        config.enable_stemming = false;

        TextProcessor processor(config);
        std::string text = "Hello World!";
        auto processed = processor.process(text);

        // Should preserve original case and punctuation
        bool found_hello = false, found_world = false;
        for (const auto& token : processed) {
            if (token.find("Hello") != std::string::npos) found_hello = true;
            if (token.find("World") != std::string::npos) found_world = true;
        }
        REQUIRE(found_hello);
        REQUIRE(found_world);
    }
}

TEST_CASE("TextProcessor - Edge Cases", "[text][processor][edge_cases]") {
    TextProcessor processor({});

    SECTION("Very long text") {
        std::string long_text(30, 'a');  // 30 'a' characters - long but within limits
        auto tokens = processor.tokenize(long_text);
        REQUIRE(tokens.size() >= 1);  // Should handle long strings
        // Check that token is within acceptable length limits
        if (!tokens.empty()) {
            REQUIRE(tokens[0].length() == 30);  // Should preserve the full token
        }
    }

    SECTION("Special characters") {
        auto tokens = processor.tokenize("C++ & Python @ $% #");
        REQUIRE(tokens.size() >= 2);  // Should find 'c' and 'python'
    }

    SECTION("Unicode characters") {
        auto tokens = processor.tokenize("café résumé naïve");
        REQUIRE(tokens.size() >= 3);  // Should handle Unicode
    }

    SECTION("Mixed content") {
        std::string text = "Email: user@example.com, URL: https://example.com";
        auto tokens = processor.tokenize(text);
        REQUIRE(tokens.size() >= 3);  // Should extract meaningful tokens
    }
}

TEST_CASE("TextProcessor - Configuration", "[text][processor][config]") {
    SECTION("Custom configuration") {
        TextProcessor::Config config;
        config.min_token_length = 3;
        config.max_token_length = 10;
        config.remove_numbers = true;

        TextProcessor processor(config);
        auto tokens = processor.tokenize("It 12 34567890 12345678901");

        // Should filter out short tokens and very long tokens and numbers
        REQUIRE(std::find(tokens.begin(), tokens.end(), "12") == tokens.end());
        REQUIRE(std::find(tokens.begin(), tokens.end(), "34567890") == tokens.end());
    }

    SECTION("Configuration update") {
        TextProcessor processor({});

        TextProcessor::Config new_config;
        new_config.remove_stopwords = false;
        processor.set_config(new_config);

        auto tokens = processor.tokenize("the and is");
        REQUIRE(tokens.size() >= 3);  // Should include stop words now
    }
}

TEST_CASE("TextProcessorFactory - Pre-configured Processors", "[text][processor][factory]") {
    SECTION("Retrieval processor") {
        auto processor = TextProcessorFactory::create_retrieval_processor();
        REQUIRE(processor != nullptr);

        auto config = processor->get_config();
        REQUIRE(config.lowercase == true);
        REQUIRE(config.remove_punctuation == true);
        REQUIRE(config.remove_stopwords == true);
    }

    SECTION("Search processor") {
        auto processor = TextProcessorFactory::create_search_processor();
        REQUIRE(processor != nullptr);

        auto tokens = processor->tokenize("search query test");
        REQUIRE(tokens.size() >= 2);
    }

    SECTION("Minimal processor") {
        auto processor = TextProcessorFactory::create_minimal_processor();
        REQUIRE(processor != nullptr);

        auto config = processor->get_config();
        REQUIRE(config.remove_stopwords == false);  // Minimal shouldn't remove stop words
        REQUIRE(config.enable_stemming == false);
    }
}