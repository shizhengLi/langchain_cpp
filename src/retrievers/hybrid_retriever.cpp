#include "langchain/retrievers/hybrid_retriever.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <sstream>
#include <unordered_set>
#include <random>

namespace langchain::retrievers {

void HybridRetrieverConfig::validate() const {
    if (sparse_weight < 0.0 || sparse_weight > 1.0) {
        throw std::invalid_argument("sparse_weight must be between 0.0 and 1.0");
    }

    if (dense_weight < 0.0 || dense_weight > 1.0) {
        throw std::invalid_argument("dense_weight must be between 0.0 and 1.0");
    }

    if (std::abs(sparse_weight + dense_weight - 1.0) > 1e-6) {
        throw std::invalid_argument("sparse_weight + dense_weight must equal 1.0");
    }

    if (top_k_sparse == 0 || top_k_dense == 0 || top_k_hybrid == 0) {
        throw std::invalid_argument("top_k values must be greater than 0");
    }

    if (rrf_k <= 0.0) {
        throw std::invalid_argument("rrf_k must be greater than 0");
    }

    if (deduplication_threshold < 0.0 || deduplication_threshold > 1.0) {
        throw std::invalid_argument("deduplication_threshold must be between 0.0 and 1.0");
    }
}

HybridRetriever::HybridRetriever(
    const HybridRetrieverConfig& config,
    std::unique_ptr<text::TextProcessor> text_processor
) : config_(config), text_processor_(std::move(text_processor)) {
    config_.validate();

    // Create sparse retriever
    BM25Retriever::Config bm25_config;
    bm25_config.normalize_scores = true;
    sparse_retriever_ = std::make_unique<BM25Retriever>(bm25_config);

    // Create dense retriever
    vectorstores::SimpleVectorStoreConfig vector_config;
    vector_config.vector_dim = 384; // Default embedding dimension
    vector_config.normalize_vectors = true;
    dense_retriever_ = std::make_unique<vectorstores::SimpleVectorStore>(vector_config);

    if (!text_processor_) {
        text::TextProcessor::Config text_config;
        text_processor_ = std::make_unique<text::TextProcessor>(text_config);
    }
}

RetrievalResult HybridRetriever::retrieve(const std::string& query) {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    auto hybrid_results = retrieve_detailed(query);

    // Convert to RetrievalResult
    RetrievalResult result;
    result.query = query;
    result.total_results = hybrid_results.size();
    result.retrieval_method = "hybrid";

    for (const auto& hybrid : hybrid_results) {
        RetrievedDocument doc;
        doc.id = std::to_string(hybrid.document_id);
        doc.content = hybrid.content;
        doc.metadata = hybrid.metadata;
        doc.relevance_score = hybrid.hybrid_score;

        result.documents.push_back(doc);
    }

    return result;
}

std::vector<std::string> HybridRetriever::add_documents(const std::vector<Document>& documents) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    std::vector<std::string> doc_ids;
    std::vector<std::vector<double>> embeddings;

    // Generate embeddings for all documents
    for (const auto& doc : documents) {
        std::vector<double> embedding = generate_embedding(doc.content);
        embeddings.push_back(embedding);
    }

    return add_documents_with_embeddings(documents, embeddings);
}

std::string HybridRetriever::add_document_with_embedding(
    const Document& document,
    const std::vector<double>& embedding
) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    // Add to sparse retriever
    std::vector<std::string> sparse_ids = sparse_retriever_->add_documents({document});

    // Add to dense retriever
    vectorstores::SimpleVector vector(embedding);
    size_t vector_id = dense_retriever_->add_vector(vector, document.content);

    // Update mappings
    if (!sparse_ids.empty()) {
        doc_to_vector_id_[sparse_ids[0]] = vector_id;
        vector_id_to_doc_[vector_id] = sparse_ids[0];
    }

    return sparse_ids.empty() ? "" : sparse_ids[0];
}

std::vector<std::string> HybridRetriever::add_documents_with_embeddings(
    const std::vector<Document>& documents,
    const std::vector<std::vector<double>>& embeddings
) {
    if (documents.size() != embeddings.size()) {
        throw std::invalid_argument("Number of documents must match number of embeddings");
    }

    std::unique_lock<std::shared_mutex> lock(mutex_);

    // Add to sparse retriever
    std::vector<std::string> sparse_ids = sparse_retriever_->add_documents(documents);

    // Add to dense retriever
    std::vector<vectorstores::SimpleVectorEntry> vector_entries;
    for (size_t i = 0; i < documents.size(); ++i) {
        vectorstores::SimpleVector vector(embeddings[i]);
        vector_entries.emplace_back(next_vector_id_++, vector, documents[i].content);
    }

    std::vector<size_t> vector_ids = dense_retriever_->add_vectors(vector_entries);

    // Update mappings
    for (size_t i = 0; i < sparse_ids.size() && i < vector_ids.size(); ++i) {
        doc_to_vector_id_[sparse_ids[i]] = vector_ids[i];
        vector_id_to_doc_[vector_ids[i]] = sparse_ids[i];
    }

    return sparse_ids;
}

size_t HybridRetriever::document_count() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return sparse_retriever_->document_count();
}

void HybridRetriever::clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    sparse_retriever_->clear();
    dense_retriever_->clear();
    doc_to_vector_id_.clear();
    vector_id_to_doc_.clear();
    next_vector_id_ = 1;
}

std::unordered_map<std::string, std::any> HybridRetriever::get_metadata() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::unordered_map<std::string, std::any> metadata;
    metadata["sparse_count"] = sparse_retriever_->document_count();
    metadata["dense_count"] = dense_retriever_->size();
    metadata["config_sparse_weight"] = config_.sparse_weight;
    metadata["config_dense_weight"] = config_.dense_weight;
    metadata["config_fusion_method"] = static_cast<int>(config_.fusion_method);

    return metadata;
}

std::vector<HybridRetrievalResult> HybridRetriever::retrieve_detailed(const std::string& query) {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    // Perform sparse and dense retrieval
    auto sparse_results = perform_sparse_retrieval(query);
    auto dense_results = perform_dense_retrieval(query);

    // Fuse results
    auto hybrid_results = fuse_results(sparse_results, dense_results);

    // Deduplicate if enabled
    if (config_.deduplicate_results) {
        hybrid_results = deduplicate_results(hybrid_results);
    }

    // Take top_k results
    if (hybrid_results.size() > config_.top_k_hybrid) {
        hybrid_results.resize(config_.top_k_hybrid);
    }

    return hybrid_results;
}

void HybridRetriever::update_config(const HybridRetrieverConfig& new_config) {
    new_config.validate();
    std::unique_lock<std::shared_mutex> lock(mutex_);
    config_ = new_config;
}

std::unordered_map<std::string, double> HybridRetriever::get_performance_stats() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::unordered_map<std::string, double> stats;
    stats["sparse_documents"] = static_cast<double>(sparse_retriever_->document_count());
    stats["dense_vectors"] = static_cast<double>(dense_retriever_->size());
    stats["mapping_count"] = static_cast<double>(doc_to_vector_id_.size());

    return stats;
}

std::vector<RetrievedDocument> HybridRetriever::perform_sparse_retrieval(const std::string& query) const {
    auto sparse_result = sparse_retriever_->retrieve(query);

    // Limit to top_k_sparse
    std::vector<RetrievedDocument> results = sparse_result.documents;
    if (results.size() > config_.top_k_sparse) {
        results.resize(config_.top_k_sparse);
    }

    return results;
}

std::vector<RetrievedDocument> HybridRetriever::perform_dense_retrieval(const std::string& query) const {
    // Generate query embedding
    std::vector<double> query_embedding = generate_embedding(query);
    vectorstores::SimpleVector query_vector(query_embedding);

    // Perform dense search
    auto dense_results = dense_retriever_->search(query_vector, config_.top_k_dense);

    // Convert to RetrievedDocument format
    std::vector<RetrievedDocument> results;
    for (const auto& dense_result : dense_results) {
        RetrievedDocument doc;
        doc.id = std::to_string(dense_result.entry_id);
        doc.content = dense_result.content;
        doc.metadata = dense_result.metadata;
        doc.relevance_score = dense_result.similarity_score;

        // Map vector ID back to document ID
        auto doc_it = vector_id_to_doc_.find(dense_result.entry_id);
        if (doc_it != vector_id_to_doc_.end()) {
            doc.id = doc_it->second;
        }

        results.push_back(doc);
    }

    return results;
}

std::vector<HybridRetrievalResult> HybridRetriever::fuse_results(
    const std::vector<RetrievedDocument>& sparse_results,
    const std::vector<RetrievedDocument>& dense_results
) const {
    // Collect all unique documents
    std::unordered_map<std::string, HybridRetrievalResult> hybrid_results_map;

    // Add sparse results
    for (size_t i = 0; i < sparse_results.size(); ++i) {
        const auto& doc = sparse_results[i];
        HybridRetrievalResult hybrid;
        // Use hash of doc.id as document_id to ensure it's numeric
        hybrid.document_id = std::hash<std::string>{}(doc.id);
        hybrid.content = doc.content;
        hybrid.metadata = doc.metadata;
        hybrid.sparse_score = doc.relevance_score;
        hybrid.sparse_rank = i + 1;
        hybrid_results_map[doc.id] = hybrid;
    }

    // Add/update with dense results
    for (size_t i = 0; i < dense_results.size(); ++i) {
        const auto& doc = dense_results[i];
        auto it = hybrid_results_map.find(doc.id);

        if (it != hybrid_results_map.end()) {
            // Update existing result
            it->second.dense_score = doc.relevance_score;
            it->second.dense_rank = i + 1;
        } else {
            // Add new result
            HybridRetrievalResult hybrid;
            // Use hash of doc.id as document_id to ensure it's numeric
            hybrid.document_id = std::hash<std::string>{}(doc.id);
            hybrid.content = doc.content;
            hybrid.metadata = doc.metadata;
            hybrid.sparse_score = 0.0; // Not found in sparse results
            hybrid.sparse_rank = 0;
            hybrid.dense_score = doc.relevance_score;
            hybrid.dense_rank = i + 1;
            hybrid_results_map[doc.id] = hybrid;
        }
    }

    // Convert to vector
    std::vector<HybridRetrievalResult> hybrid_results;
    for (const auto& pair : hybrid_results_map) {
        hybrid_results.push_back(pair.second);
    }

    // Normalize scores if enabled
    if (config_.normalize_scores) {
        std::vector<double> sparse_scores, dense_scores;
        for (const auto& result : hybrid_results) {
            sparse_scores.push_back(result.sparse_score);
            dense_scores.push_back(result.dense_score);
        }

        auto normalized_sparse = normalize_scores(sparse_scores, config_.normalization_method);
        auto normalized_dense = normalize_scores(dense_scores, config_.normalization_method);

        for (size_t i = 0; i < hybrid_results.size(); ++i) {
            hybrid_results[i].sparse_score = normalized_sparse[i];
            hybrid_results[i].dense_score = normalized_dense[i];
        }
    }

    // Apply fusion method
    for (auto& result : hybrid_results) {
        switch (config_.fusion_method) {
            case HybridRetrieverConfig::FusionMethod::WEIGHTED_AVERAGE:
                result.hybrid_score = config_.sparse_weight * result.sparse_score +
                                    config_.dense_weight * result.dense_score;
                break;
            case HybridRetrieverConfig::FusionMethod::RRF:
                {
                    double rrf_sparse = result.sparse_rank > 0 ? 1.0 / (config_.rrf_k + result.sparse_rank) : 0.0;
                    double rrf_dense = result.dense_rank > 0 ? 1.0 / (config_.rrf_k + result.dense_rank) : 0.0;
                    result.hybrid_score = config_.sparse_weight * rrf_sparse + config_.dense_weight * rrf_dense;
                }
                break;
            case HybridRetrieverConfig::FusionMethod::MAX:
                result.hybrid_score = std::max(result.sparse_score, result.dense_score);
                break;
            case HybridRetrieverConfig::FusionMethod::CONDENSATION:
                result.hybrid_score = (result.sparse_score + result.dense_score) / 2.0;
                break;
        }
    }

    // Sort by hybrid score (descending)
    std::sort(hybrid_results.begin(), hybrid_results.end(),
              [](const HybridRetrievalResult& a, const HybridRetrievalResult& b) {
                  return a.hybrid_score > b.hybrid_score;
              });

    return hybrid_results;
}

std::vector<double> HybridRetriever::normalize_scores(
    const std::vector<double>& scores,
    HybridRetrieverConfig::NormalizationMethod method
) const {
    if (scores.empty()) {
        return {};
    }

    std::vector<double> normalized(scores.size());

    switch (method) {
        case HybridRetrieverConfig::NormalizationMethod::MIN_MAX: {
            auto [min_it, max_it] = std::minmax_element(scores.begin(), scores.end());
            double min_val = *min_it;
            double max_val = *max_it;
            double range = max_val - min_val;

            if (range > 1e-10) {
                for (size_t i = 0; i < scores.size(); ++i) {
                    normalized[i] = (scores[i] - min_val) / range;
                }
            } else {
                std::fill(normalized.begin(), normalized.end(), 0.5);
            }
            break;
        }
        case HybridRetrieverConfig::NormalizationMethod::Z_SCORE: {
            double mean = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
            double sq_sum = std::inner_product(scores.begin(), scores.end(), scores.begin(), 0.0);
            double std_dev = std::sqrt(sq_sum / scores.size() - mean * mean);

            if (std_dev > 1e-10) {
                for (size_t i = 0; i < scores.size(); ++i) {
                    normalized[i] = (scores[i] - mean) / std_dev;
                }
            } else {
                std::fill(normalized.begin(), normalized.end(), 0.0);
            }
            break;
        }
        case HybridRetrieverConfig::NormalizationMethod::SUM: {
            double sum = std::accumulate(scores.begin(), scores.end(), 0.0);
            if (sum > 1e-10) {
                for (size_t i = 0; i < scores.size(); ++i) {
                    normalized[i] = scores[i] / sum;
                }
            } else {
                double uniform_val = 1.0 / scores.size();
                std::fill(normalized.begin(), normalized.end(), uniform_val);
            }
            break;
        }
    }

    return normalized;
}

std::vector<HybridRetrievalResult> HybridRetriever::deduplicate_results(
    std::vector<HybridRetrievalResult> results
) const {
    if (results.size() <= 1) {
        return results;
    }

    std::vector<HybridRetrievalResult> deduplicated;
    deduplicated.reserve(results.size());

    for (const auto& result : results) {
        bool is_duplicate = false;
        for (const auto& existing : deduplicated) {
            double similarity = calculate_document_similarity(result.content, existing.content);
            if (similarity >= config_.deduplication_threshold) {
                is_duplicate = true;
                break;
            }
        }

        if (!is_duplicate) {
            deduplicated.push_back(result);
        }
    }

    return deduplicated;
}

double HybridRetriever::calculate_document_similarity(
    const std::string& doc1,
    const std::string& doc2
) const {
    // Simple similarity calculation based on word overlap
    // In a real implementation, this would use more sophisticated methods
    std::unordered_set<std::string> words1, words2;

    // Tokenize (simple implementation)
    std::istringstream iss1(doc1);
    std::string word;
    while (iss1 >> word) {
        words1.insert(word);
    }

    std::istringstream iss2(doc2);
    while (iss2 >> word) {
        words2.insert(word);
    }

    // Calculate Jaccard similarity
    std::unordered_set<std::string> intersection;
    for (const auto& w : words1) {
        if (words2.find(w) != words2.end()) {
            intersection.insert(w);
        }
    }

    std::unordered_set<std::string> union_set = words1;
    for (const auto& w : words2) {
        union_set.insert(w);
    }

    if (union_set.empty()) {
        return 0.0;
    }

    return static_cast<double>(intersection.size()) / union_set.size();
}

std::vector<double> HybridRetriever::generate_embedding(const std::string& text) const {
    // Mock implementation - in a real system, this would use actual embedding models
    std::vector<double> embedding(384, 0.0);

    // Generate pseudo-random but deterministic embedding based on text hash
    size_t hash = std::hash<std::string>{}(text);
    std::mt19937 gen(hash);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (double& val : embedding) {
        val = dist(gen);
    }

    // Normalize embedding
    double mag = std::sqrt(std::inner_product(embedding.begin(), embedding.end(),
                                             embedding.begin(), 0.0));
    if (mag > 0.0) {
        for (double& val : embedding) {
            val /= mag;
        }
    }

    return embedding;
}

HybridRetrievalResult HybridRetriever::convert_to_hybrid_result(
    const RetrievedDocument& doc,
    double sparse_score,
    double dense_score,
    size_t sparse_rank,
    size_t dense_rank
) const {
    HybridRetrievalResult hybrid;
    hybrid.document_id = std::stoull(doc.id);
    hybrid.content = doc.content;
    hybrid.metadata = doc.metadata;
    hybrid.sparse_score = sparse_score;
    hybrid.dense_score = dense_score;
    hybrid.sparse_rank = sparse_rank;
    hybrid.dense_rank = dense_rank;
    hybrid.hybrid_score = config_.sparse_weight * sparse_score + config_.dense_weight * dense_score;

    return hybrid;
}

// HybridRetrieverFactory implementation
std::unique_ptr<HybridRetriever> HybridRetrieverFactory::create_standard(
    const HybridRetrieverConfig& config
) {
    return std::make_unique<HybridRetriever>(config);
}

std::unique_ptr<HybridRetriever> HybridRetrieverFactory::create_optimized(
    const std::string& use_case,
    size_t vector_dim
) {
    HybridRetrieverConfig config;

    if (use_case == "search") {
        config.sparse_weight = 0.6;
        config.dense_weight = 0.4;
        config.fusion_method = HybridRetrieverConfig::FusionMethod::WEIGHTED_AVERAGE;
        config.top_k_sparse = 30;
        config.top_k_dense = 20;
        config.top_k_hybrid = 10;
    } else if (use_case == "recommendation") {
        config.sparse_weight = 0.3;
        config.dense_weight = 0.7;
        config.fusion_method = HybridRetrieverConfig::FusionMethod::RRF;
        config.top_k_sparse = 15;
        config.top_k_dense = 30;
        config.top_k_hybrid = 10;
    } else if (use_case == "qa") {
        config.sparse_weight = 0.7;
        config.dense_weight = 0.3;
        config.fusion_method = HybridRetrieverConfig::FusionMethod::MAX;
        config.top_k_sparse = 25;
        config.top_k_dense = 15;
        config.top_k_hybrid = 5;
    } else {
        // Default configuration
        config.sparse_weight = 0.5;
        config.dense_weight = 0.5;
    }

    return std::make_unique<HybridRetriever>(config);
}

std::unique_ptr<HybridRetriever> HybridRetrieverFactory::create_with_components(
    std::unique_ptr<BM25Retriever> sparse_retriever,
    std::unique_ptr<vectorstores::SimpleVectorStore> dense_retriever,
    const HybridRetrieverConfig& config
) {
    // This would require modifying the HybridRetriever constructor to accept components
    // For now, use the standard constructor
    return std::make_unique<HybridRetriever>(config);
}

} // namespace langchain::retrievers