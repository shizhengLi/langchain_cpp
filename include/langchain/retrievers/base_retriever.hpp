#pragma once

#include "../core/types.hpp"
#include "../core/base.hpp"
#include <string>
#include <vector>
#include <memory>
#include <optional>

namespace langchain::retrievers {

/**
 * @brief Base class for all retrievers in LangChain++
 *
 * This abstract interface defines the common contract that all retrievers
 * must implement. Retrievers are responsible for finding and returning
 * relevant documents based on a query.
 */
class BaseRetriever {
public:
    virtual ~BaseRetriever() = default;

    /**
     * @brief Retrieve relevant documents for a given query
     * @param query The search query string
     * @return RetrievalResult containing relevant documents with scores
     * @throws RetrievalException if retrieval fails
     */
    virtual RetrievalResult retrieve(const std::string& query) = 0;

    /**
     * @brief Retrieve relevant documents for multiple queries (batch processing)
     * @param queries Vector of query strings
     * @return Vector of RetrievalResults, one for each query
     * @throws RetrievalException if any retrieval fails
     */
    virtual std::vector<RetrievalResult> retrieve_batch(
        const std::vector<std::string>& queries) {
        std::vector<RetrievalResult> results;
        results.reserve(queries.size());

        for (const auto& query : queries) {
            results.push_back(retrieve(query));
        }

        return results;
    }

    /**
     * @brief Add documents to the retriever's index
     * @param documents Vector of documents to index
     * @return Vector of document IDs
     * @throws RetrievalException if indexing fails
     */
    virtual std::vector<std::string> add_documents(
        const std::vector<Document>& documents) = 0;

    /**
     * @brief Get the number of documents in the retriever's index
     * @return Number of indexed documents
     */
    virtual size_t document_count() const = 0;

    /**
     * @brief Check if the retriever is ready for querying
     * @return true if the retriever has been initialized and has documents
     */
    virtual bool is_ready() const {
        return document_count() > 0;
    }

    /**
     * @brief Get retriever metadata/information
     * @return Metadata map with retriever-specific information
     */
    virtual std::unordered_map<std::string, std::any> get_metadata() const {
        return {
            {"type", std::string("BaseRetriever")},
            {"document_count", document_count()},
            {"ready", is_ready()}
        };
    }

    /**
     * @brief Reset/clear the retriever's index
     */
    virtual void clear() = 0;
};

/**
 * @brief Exception class for retrieval operations
 */
class RetrieverException : public langchain::RetrievalException {
public:
    explicit RetrieverException(const std::string& message)
        : langchain::RetrievalException(message) {}
};

/**
 * @brief Configuration exception for retriever setup
 */
class RetrieverConfigurationException : public langchain::ConfigurationException {
public:
    explicit RetrieverConfigurationException(const std::string& message)
        : langchain::ConfigurationException(message) {}
};

} // namespace langchain::retrievers