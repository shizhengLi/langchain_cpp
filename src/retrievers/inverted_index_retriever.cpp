#include "langchain/retrievers/inverted_index_retriever.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace langchain::retrievers {

InvertedIndexRetriever::InvertedIndexRetriever(
    const Config& config,
    std::unique_ptr<text::TextProcessor> text_processor)
    : config_(config), text_processor_(std::move(text_processor)) {

    if (!text_processor_) {
        // Create default text processor
        text_processor_ = text::TextProcessorFactory::create_retrieval_processor();
    }
}

RetrievalResult InvertedIndexRetriever::retrieve(const std::string& query) {
    auto start_time = std::chrono::high_resolution_clock::now();

    RetrievalResult result;
    result.query = query;
    result.retrieval_method = "inverted_index";

    total_queries_++;

    if (documents_.empty()) {
        result.search_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);
        return result;
    }

    // Process query into terms
    auto query_terms = process_query(query);
    if (query_terms.empty()) {
        result.search_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);
        return result;
    }

    // Get candidate documents using posting list intersection
    std::vector<std::string> term_strings;
    term_strings.reserve(query_terms.size());
    for (const auto& [term, _] : query_terms) {
        term_strings.push_back(term);
    }

    std::vector<size_t> candidate_docs;
    if (term_strings.size() == 1) {
        // Single term query
        auto it = inverted_index_.find(term_strings[0]);
        if (it != inverted_index_.end()) {
            candidate_docs.reserve(it->second.postings.size());
            for (const auto& posting : it->second.postings) {
                candidate_docs.push_back(posting.document_id);
            }
        }
    } else {
        // Multi-term query - use intersection for better precision
        candidate_docs = intersect_postings(term_strings);
        if (candidate_docs.empty()) {
            // Fallback to union if intersection yields no results
            candidate_docs = union_postings(term_strings);
        }
    }

    // Calculate scores and create results
    std::vector<RetrievedDocument> retrieved_docs;
    retrieved_docs.reserve(candidate_docs.size());

    for (size_t doc_id : candidate_docs) {
        auto doc_opt = get_document(doc_id);
        if (!doc_opt) continue;

        double score = calculate_score(query_terms, doc_id);
        if (score >= config_.score_threshold) {
            RetrievedDocument retrieved_doc(*doc_opt, score);
            retrieved_docs.push_back(std::move(retrieved_doc));
        }
    }

    // Sort by relevance score (descending)
    std::sort(retrieved_docs.begin(), retrieved_docs.end(),
              [](const RetrievedDocument& a, const RetrievedDocument& b) {
                  return a.relevance_score > b.relevance_score;
              });

    // Limit results
    size_t max_results = std::min(config_.max_results, retrieved_docs.size());
    result.documents.assign(retrieved_docs.begin(), retrieved_docs.begin() + max_results);
    result.total_results = retrieved_docs.size();

    // Normalize scores if requested
    if (config_.normalize_scores && !result.documents.empty()) {
        double max_score = result.documents[0].relevance_score;
        if (max_score > 0) {
            for (auto& doc : result.documents) {
                doc.relevance_score /= max_score;
            }
        }
    }

    result.search_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time);

    // Update metadata
    result.metadata["query_terms_count"] = query_terms.size();
    result.metadata["candidate_docs"] = candidate_docs.size();
    result.metadata["cache_hit_rate"] = get_cache_stats()["hit_rate"];

    return result;
}

std::vector<std::string> InvertedIndexRetriever::add_documents(
    const std::vector<Document>& documents) {

    std::vector<std::string> doc_ids;
    doc_ids.reserve(documents.size());

    std::unique_lock<std::shared_mutex> lock(index_mutex_);

    for (const auto& doc : documents) {
        size_t doc_id = add_document_internal(doc);
        doc_ids.push_back("doc_" + std::to_string(doc_id));
    }

    // Update IDF values for all terms
    size_t total_docs = documents_.size();
    for (auto& [term, term_info] : inverted_index_) {
        term_info.update_idf(total_docs);
    }

    return doc_ids;
}

size_t InvertedIndexRetriever::document_count() const {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);
    return documents_.size();
}

void InvertedIndexRetriever::clear() {
    std::unique_lock<std::shared_mutex> lock(index_mutex_);

    inverted_index_.clear();
    documents_.clear();
    doc_id_map_.clear();
    next_doc_id_ = 1;
    cache_timestamp_ = 0;
}

std::unordered_map<std::string, std::any> InvertedIndexRetriever::get_metadata() const {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);

    auto metadata = BaseRetriever::get_metadata();
    metadata["type"] = std::string("InvertedIndexRetriever");
    metadata["total_terms"] = inverted_index_.size();
    metadata["total_postings"] = std::accumulate(
        inverted_index_.begin(), inverted_index_.end(), size_t(0),
        [](size_t sum, const auto& pair) {
            return sum + pair.second.postings.size();
        });
    metadata["cache_enabled"] = config_.enable_term_caching;
    metadata["total_queries"] = total_queries_.load();

    return metadata;
}

std::vector<InvertedIndexRetriever::PostingEntry>
InvertedIndexRetriever::get_postings(const std::string& term) {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);

    update_cache_stats(term);

    auto it = inverted_index_.find(term);
    if (it != inverted_index_.end()) {
        return it->second.postings;
    }
    return {};
}

InvertedIndexRetriever::TermInfo
InvertedIndexRetriever::get_term_info(const std::string& term) {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);

    update_cache_stats(term);

    auto it = inverted_index_.find(term);
    if (it != inverted_index_.end()) {
        return it->second;
    }
    return {};
}

std::vector<std::pair<std::string, size_t>>
InvertedIndexRetriever::get_most_frequent_terms(size_t limit) const {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);

    std::vector<std::pair<std::string, size_t>> terms;
    terms.reserve(inverted_index_.size());

    for (const auto& [term, term_info] : inverted_index_) {
        terms.emplace_back(term, term_info.total_term_frequency);
    }

    std::sort(terms.begin(), terms.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    if (terms.size() > limit) {
        terms.resize(limit);
    }

    return terms;
}

void InvertedIndexRetriever::optimize_index() {
    std::unique_lock<std::shared_mutex> lock(index_mutex_);

    // Sort posting lists by document ID for better cache locality
    for (auto& [term, term_info] : inverted_index_) {
        std::sort(term_info.postings.begin(), term_info.postings.end(),
                  [](const PostingEntry& a, const PostingEntry& b) {
                      return a.document_id < b.document_id;
                  });
    }

    // Update all IDF values
    size_t total_docs = documents_.size();
    for (auto& [term, term_info] : inverted_index_) {
        term_info.update_idf(total_docs);
    }

    // Cleanup cache if needed
    if (config_.enable_term_caching) {
        cleanup_cache();
    }
}

std::unordered_map<std::string, double>
InvertedIndexRetriever::get_cache_stats() const {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);

    size_t total = total_queries_.load();
    size_t hits = cache_hits_.load();
    size_t misses = cache_misses_.load();

    double hit_rate = total > 0 ? (static_cast<double>(hits) / total) * 100.0 : 0.0;
    double miss_rate = total > 0 ? (static_cast<double>(misses) / total) * 100.0 : 0.0;

    return {
        {"hit_rate", hit_rate},
        {"miss_rate", miss_rate},
        {"total_queries", static_cast<double>(total)},
        {"cache_hits", static_cast<double>(hits)},
        {"cache_misses", static_cast<double>(misses)}
    };
}

void InvertedIndexRetriever::update_config(const Config& new_config) {
    std::unique_lock<std::shared_mutex> lock(index_mutex_);
    config_ = new_config;
}

std::unordered_map<std::string, size_t>
InvertedIndexRetriever::process_query(const std::string& query) const {
    auto tokens = text_processor_->process(query);
    std::unordered_map<std::string, size_t> term_frequencies;

    for (const auto& token : tokens) {
        term_frequencies[token]++;
    }

    return term_frequencies;
}

double InvertedIndexRetriever::calculate_score(
    const std::unordered_map<std::string, size_t>& query_terms,
    size_t doc_id) const {

    double score = 0.0;

    for (const auto& [term, query_tf] : query_terms) {
        auto term_it = inverted_index_.find(term);
        if (term_it == inverted_index_.end()) continue;

        const TermInfo& term_info = term_it->second;

        // Find document posting
        auto posting_it = std::find_if(term_info.postings.begin(), term_info.postings.end(),
                                      [doc_id](const PostingEntry& posting) {
                                          return posting.document_id == doc_id;
                                      });

        if (posting_it == term_info.postings.end()) continue;

        // TF-IDF scoring: tf * idf * query_tf
        double tf = static_cast<double>(posting_it->term_frequency);
        double idf = term_info.idf;
        double query_weight = std::log(1.0 + query_tf);  // Log normalization

        score += tf * idf * query_weight;
    }

    return score;
}

std::vector<size_t> InvertedIndexRetriever::intersect_postings(
    const std::vector<std::string>& terms) const {

    if (terms.empty()) return {};

    // Find smallest posting list to start intersection
    std::string smallest_term;
    const std::vector<PostingEntry>* smallest_postings = nullptr;
    size_t smallest_size = SIZE_MAX;

    for (const auto& term : terms) {
        auto it = inverted_index_.find(term);
        if (it != inverted_index_.end() && it->second.postings.size() < smallest_size) {
            smallest_size = it->second.postings.size();
            smallest_postings = &it->second.postings;
            smallest_term = term;
        }
    }

    if (!smallest_postings) return {};

    std::vector<size_t> result;

    // For each document in smallest list, check if it exists in all other lists
    for (const auto& posting : *smallest_postings) {
        bool found_in_all = true;

        for (const auto& term : terms) {
            if (term == smallest_term) continue; // Skip the smallest term

            auto it = inverted_index_.find(term);
            if (it == inverted_index_.end()) {
                found_in_all = false;
                break;
            }

            bool found = std::binary_search(it->second.postings.begin(),
                                          it->second.postings.end(),
                                          posting,
                                          [](const PostingEntry& a, const PostingEntry& b) {
                                              return a.document_id < b.document_id;
                                          });

            if (!found) {
                found_in_all = false;
                break;
            }
        }

        if (found_in_all) {
            result.push_back(posting.document_id);
        }
    }

    return result;
}

std::vector<size_t> InvertedIndexRetriever::union_postings(
    const std::vector<std::string>& terms) const {

    std::unordered_set<size_t> doc_ids;

    for (const auto& term : terms) {
        auto it = inverted_index_.find(term);
        if (it != inverted_index_.end()) {
            for (const auto& posting : it->second.postings) {
                doc_ids.insert(posting.document_id);
            }
        }
    }

    return std::vector<size_t>(doc_ids.begin(), doc_ids.end());
}

size_t InvertedIndexRetriever::add_document_internal(const Document& document) {
    size_t doc_id = generate_document_id();

    // Add to document storage
    documents_.push_back(document);
    doc_id_map_[document.id] = doc_id;

    // Process document content
    auto tokens = text_processor_->process(document.content);
    std::unordered_map<std::string, size_t> term_frequencies;
    std::unordered_map<std::string, std::vector<size_t>> term_positions;

    // Track term frequencies and positions
    for (size_t pos = 0; pos < tokens.size(); ++pos) {
        const std::string& token = tokens[pos];
        term_frequencies[token]++;
        term_positions[token].push_back(pos);
    }

    // Update inverted index
    for (const auto& [term, tf] : term_frequencies) {
        if (tf < config_.min_term_frequency) continue;

        TermInfo& term_info = inverted_index_[term];

        // Check if we've exceeded max postings per term
        if (term_info.postings.size() >= config_.max_postings_per_term) {
            continue;  // Skip this term for this document
        }

        // Add posting entry
        PostingEntry posting(doc_id, tf, term_positions[term]);
        term_info.postings.push_back(std::move(posting));

        // Update term statistics
        term_info.document_frequency += 1;  // Increment document frequency
        term_info.total_term_frequency += tf;
    }

    return doc_id;
}

void InvertedIndexRetriever::update_cache_stats(const std::string& term) {
    if (!config_.enable_term_caching) return;

    auto it = inverted_index_.find(term);
    if (it != inverted_index_.end()) {
        if (it->second.last_accessed > 0) {
            cache_hits_++;
        } else {
            cache_misses_++;
        }
        it->second.last_accessed = ++cache_timestamp_;
    }
}

void InvertedIndexRetriever::cleanup_cache() {
    if (inverted_index_.size() <= config_.cache_size_limit) return;

    // Sort terms by last accessed time
    std::vector<std::pair<std::string, size_t>> terms_by_access;
    terms_by_access.reserve(inverted_index_.size());

    for (const auto& [term, term_info] : inverted_index_) {
        terms_by_access.emplace_back(term, term_info.last_accessed);
    }

    std::sort(terms_by_access.begin(), terms_by_access.end(),
              [](const auto& a, const auto& b) {
                  return a.second < b.second;  // Oldest first
              });

    // Remove least recently used terms to maintain cache size limit
    size_t terms_to_remove = inverted_index_.size() - config_.cache_size_limit;
    for (size_t i = 0; i < terms_to_remove; ++i) {
        inverted_index_.erase(terms_by_access[i].first);
    }
}

size_t InvertedIndexRetriever::generate_document_id() {
    return next_doc_id_++;
}

std::optional<Document> InvertedIndexRetriever::get_document(size_t doc_id) const {
    if (doc_id == 0 || doc_id > documents_.size()) {
        return std::nullopt;
    }
    return documents_[doc_id - 1];  // doc_id is 1-based, vector is 0-based
}

// Factory methods
std::unique_ptr<InvertedIndexRetriever>
InvertedIndexRetrieverFactory::create_retrieval_retriever() {
    InvertedIndexRetriever::Config config;
    config.min_term_frequency = 1;
    config.max_postings_per_term = 100000;
    config.enable_term_caching = true;
    config.cache_size_limit = 10000;
    config.normalize_scores = true;
    config.score_threshold = 0.01;
    config.max_results = 10;

    return std::make_unique<InvertedIndexRetriever>(
        config,
        text::TextProcessorFactory::create_retrieval_processor());
}

std::unique_ptr<InvertedIndexRetriever>
InvertedIndexRetrieverFactory::create_search_retriever() {
    InvertedIndexRetriever::Config config;
    config.min_term_frequency = 1;
    config.max_postings_per_term = 50000;
    config.enable_term_caching = true;
    config.cache_size_limit = 5000;
    config.normalize_scores = true;
    config.score_threshold = 0.05;
    config.max_results = 20;

    return std::make_unique<InvertedIndexRetriever>(
        config,
        text::TextProcessorFactory::create_search_processor());
}

std::unique_ptr<InvertedIndexRetriever>
InvertedIndexRetrieverFactory::create_large_dataset_retriever() {
    InvertedIndexRetriever::Config config;
    config.min_term_frequency = 2;
    config.max_postings_per_term = 1000000;
    config.enable_term_caching = true;
    config.cache_size_limit = 50000;
    config.normalize_scores = true;
    config.score_threshold = 0.001;
    config.max_results = 100;

    return std::make_unique<InvertedIndexRetriever>(
        config,
        text::TextProcessorFactory::create_retrieval_processor());
}

std::unique_ptr<InvertedIndexRetriever>
InvertedIndexRetrieverFactory::create_memory_efficient_retriever() {
    InvertedIndexRetriever::Config config;
    config.min_term_frequency = 3;
    config.max_postings_per_term = 10000;
    config.enable_term_caching = false;
    config.cache_size_limit = 1000;
    config.normalize_scores = false;
    config.score_threshold = 0.1;
    config.max_results = 5;

    return std::make_unique<InvertedIndexRetriever>(
        config,
        text::TextProcessorFactory::create_minimal_processor());
}

} // namespace langchain::retrievers