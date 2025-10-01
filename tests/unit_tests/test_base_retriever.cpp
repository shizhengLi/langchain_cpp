#include <catch2/catch_all.hpp>
#include "langchain/retrievers/base_retriever.hpp"
#include <memory>

using namespace langchain;
using namespace langchain::retrievers;

// Mock implementation for testing
class MockRetriever : public BaseRetriever {
private:
    std::vector<langchain::Document> documents_;
    size_t call_count_{0};

public:
    langchain::RetrievalResult retrieve(const std::string& query) override {
        call_count_++;
        langchain::RetrievalResult result;
        result.query = query;
        result.total_results = documents_.size();

        // Simple mock retrieval: return all documents with fixed scores
        for (const auto& doc : documents_) {
            result.documents.emplace_back(doc, 0.5);
        }

        return result;
    }

    std::vector<std::string> add_documents(const std::vector<langchain::Document>& documents) override {
        for (const auto& doc : documents) {
            documents_.push_back(doc);
        }

        std::vector<std::string> ids;
        for (const auto& doc : documents) {
            ids.push_back(doc.id);
        }

        return ids;
    }

    size_t document_count() const override {
        return documents_.size();
    }

    void clear() override {
        documents_.clear();
    }

    // Test helper methods
    size_t get_call_count() const { return call_count_; }
    void reset_call_count() { call_count_ = 0; }
};

TEST_CASE("BaseRetriever - Interface Contract", "[retrievers][base]") {
    SECTION("Default virtual methods") {
        MockRetriever retriever;

        // Test default implementations
        REQUIRE_FALSE(retriever.is_ready());  // No documents yet
        auto metadata = retriever.get_metadata();

        REQUIRE(std::any_cast<std::string>(metadata.at("type")) == std::string("BaseRetriever"));
        REQUIRE(std::any_cast<size_t>(metadata.at("document_count")) == 0);
        REQUIRE(std::any_cast<bool>(metadata.at("ready")) == false);
    }
}

TEST_CASE("BaseRetriever - Basic Operations", "[retrievers][base]") {
    MockRetriever retriever;

    SECTION("Add documents and count") {
        std::vector<langchain::Document> docs = {
            langchain::Document("Hello world", {{"source", "test1"}}),
            langchain::Document("Test document", {{"source", "test2"}})
        };

        auto ids = retriever.add_documents(docs);
        REQUIRE(ids.size() == 2);
        REQUIRE(retriever.document_count() == 2);
        REQUIRE(retriever.is_ready());
    }

    SECTION("Retrieve single query") {
        retriever.add_documents({langchain::Document("Test content")});

        auto result = retriever.retrieve("test query");
        REQUIRE(result.query == "test query");
        REQUIRE(result.total_results == 1);
        REQUIRE(result.documents.size() == 1);
        REQUIRE(result.documents[0].content == "Test content");
        REQUIRE(result.documents[0].relevance_score == 0.5);
    }

    SECTION("Retrieve batch queries") {
        retriever.add_documents({langchain::Document("Content")});

        std::vector<std::string> queries = {"query1", "query2", "query3"};
        auto results = retriever.retrieve_batch(queries);

        REQUIRE(results.size() == 3);
        for (size_t i = 0; i < queries.size(); ++i) {
            REQUIRE(results[i].query == queries[i]);
            REQUIRE(results[i].documents.size() == 1);
        }
    }

    SECTION("Clear documents") {
        retriever.add_documents({langchain::Document("Test")});
        REQUIRE(retriever.document_count() == 1);

        retriever.clear();
        REQUIRE(retriever.document_count() == 0);
        REQUIRE_FALSE(retriever.is_ready());
    }
}

TEST_CASE("BaseRetriever - Metadata Updates", "[retrievers][base]") {
    MockRetriever retriever;

    SECTION("Metadata changes with document count") {
        auto initial_metadata = retriever.get_metadata();
        REQUIRE(std::any_cast<size_t>(initial_metadata.at("document_count")) == 0);
        REQUIRE(std::any_cast<bool>(initial_metadata.at("ready")) == false);

        retriever.add_documents({Document("Test")});

        auto updated_metadata = retriever.get_metadata();
        REQUIRE(std::any_cast<size_t>(updated_metadata.at("document_count")) == 1);
        REQUIRE(std::any_cast<bool>(updated_metadata.at("ready")) == true);
    }
}

TEST_CASE("BaseRetriever - Edge Cases", "[retrievers][base]") {
    MockRetriever retriever;

    SECTION("Empty batch retrieval") {
        auto results = retriever.retrieve_batch({});
        REQUIRE(results.empty());
    }

    SECTION("Multiple document additions") {
        std::vector<langchain::Document> docs1 = {langchain::Document("Doc1"), langchain::Document("Doc2")};
        std::vector<langchain::Document> docs2 = {langchain::Document("Doc3")};

        auto ids1 = retriever.add_documents(docs1);
        auto ids2 = retriever.add_documents(docs2);

        REQUIRE(ids1.size() == 2);
        REQUIRE(ids2.size() == 1);
        REQUIRE(retriever.document_count() == 3);
    }

    SECTION("Clear empty retriever") {
        REQUIRE_NOTHROW(retriever.clear());
        REQUIRE(retriever.document_count() == 0);
    }
}

TEST_CASE("BaseRetriever - Exception Safety", "[retrievers][base]") {
    SECTION("RetrieverException") {
        RetrieverException ex("Test retriever error");
        REQUIRE(std::string(ex.what()) == "Test retriever error");
        REQUIRE(std::string(ex.error_code()) == "RETRIEVAL_ERROR");
    }

    SECTION("RetrieverConfigurationException") {
        RetrieverConfigurationException ex("Test config error");
        REQUIRE(std::string(ex.what()) == "Test config error");
        REQUIRE(std::string(ex.error_code()) == "CONFIG_ERROR");
    }
}