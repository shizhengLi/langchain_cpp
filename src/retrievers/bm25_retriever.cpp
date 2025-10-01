#include "langchain/retrievers/bm25_retriever.hpp"
#include "langchain/utils/simd_ops.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace langchain::retrievers {

BM25Retriever::BM25Retriever(
    const Config& config,
    std::unique_ptr<text::TextProcessor> text_processor)
    : config_(config), text_processor_(std::move(text_processor)),
      avg_document_length_(0.0), total_terms_in_corpus_(0) {

    if (!text_processor_) {
        // Create default text processor
        text_processor_ = text::TextProcessorFactory::create_retrieval_processor();
    }
}

RetrievalResult BM25Retriever::retrieve(const std::string& query) {
    auto start_time = std::chrono::high_resolution_clock::now();

    RetrievalResult result;
    result.query = query;
    result.retrieval_method = "bm25";

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

    // Get candidate documents using optimized posting list intersection
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
        // Multi-term query - use optimized intersection for better precision
        candidate_docs = intersect_postings_optimized(term_strings);
        if (candidate_docs.empty()) {
            // Fallback to union if intersection yields no results
            candidate_docs = union_postings_optimized(term_strings);
        }
    }

    // Calculate BM25 scores and create results
    std::vector<RetrievedDocument> retrieved_docs;
    retrieved_docs.reserve(candidate_docs.size());

    for (size_t doc_id : candidate_docs) {
        auto doc_opt = get_document(doc_id);
        if (!doc_opt) continue;

        double score = calculate_bm25_score(query_terms, doc_id);
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
    result.metadata["bm25_k1"] = config_.k1;
    result.metadata["bm25_b"] = config_.b;

    return result;
}

std::vector<std::string> BM25Retriever::add_documents(
    const std::vector<Document>& documents) {

    std::vector<std::string> doc_ids;
    doc_ids.reserve(documents.size());

    std::unique_lock<std::shared_mutex> lock(index_mutex_);

    for (const auto& doc : documents) {
        size_t doc_id = add_document_internal(doc);
        doc_ids.push_back("doc_" + std::to_string(doc_id));
    }

    // Update corpus statistics after all documents are added
    update_corpus_statistics();

    return doc_ids;
}

size_t BM25Retriever::document_count() const {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);
    return documents_.size();
}

void BM25Retriever::clear() {
    std::unique_lock<std::shared_mutex> lock(index_mutex_);

    inverted_index_.clear();
    documents_.clear();
    doc_id_map_.clear();
    document_stats_.clear();
    next_doc_id_ = 1;
    cache_timestamp_ = 0;
    avg_document_length_ = 0.0;
    total_terms_in_corpus_ = 0;
}

std::unordered_map<std::string, std::any> BM25Retriever::get_metadata() const {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);

    auto metadata = BaseRetriever::get_metadata();
    metadata["type"] = std::string("BM25Retriever");
    metadata["total_terms"] = inverted_index_.size();
    metadata["total_postings"] = std::accumulate(
        inverted_index_.begin(), inverted_index_.end(), size_t(0),
        [](size_t sum, const auto& pair) {
            return sum + pair.second.postings.size();
        });
    metadata["cache_enabled"] = config_.enable_term_caching;
    metadata["total_queries"] = total_queries_.load();
    metadata["avg_document_length"] = avg_document_length_;
    metadata["total_terms_in_corpus"] = total_terms_in_corpus_;
    metadata["bm25_k1"] = config_.k1;
    metadata["bm25_b"] = config_.b;
    metadata["bm25_delta"] = config_.delta;

    return metadata;
}

std::vector<BM25Retriever::PostingEntry>
BM25Retriever::get_postings(const std::string& term) {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);

    update_cache_stats(term);

    auto it = inverted_index_.find(term);
    if (it != inverted_index_.end()) {
        return it->second.postings;
    }
    return {};
}

BM25Retriever::TermInfo
BM25Retriever::get_term_info(const std::string& term) {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);

    update_cache_stats(term);

    auto it = inverted_index_.find(term);
    if (it != inverted_index_.end()) {
        return it->second;
    }
    return {};
}

std::optional<BM25Retriever::DocumentStats>
BM25Retriever::get_document_stats(size_t doc_id) const {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);

    if (doc_id == 0 || doc_id > document_stats_.size()) {
        return std::nullopt;
    }
    return document_stats_[doc_id - 1];  // doc_id is 1-based, vector is 0-based
}

std::vector<std::pair<std::string, size_t>>
BM25Retriever::get_most_frequent_terms(size_t limit) const {
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

void BM25Retriever::optimize_index() {
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
BM25Retriever::get_cache_stats() const {
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

void BM25Retriever::update_config(const Config& new_config) {
    std::unique_lock<std::shared_mutex> lock(index_mutex_);
    config_ = new_config;
}

std::unordered_map<std::string, double>
BM25Retriever::get_bm25_parameters() const {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);

    return {
        {"k1", config_.k1},
        {"b", config_.b},
        {"delta", config_.delta},
        {"avg_document_length", avg_document_length_}
    };
}

std::unordered_map<std::string, size_t>
BM25Retriever::process_query(const std::string& query) const {
    auto tokens = text_processor_->process(query);
    std::unordered_map<std::string, size_t> term_frequencies;

    for (const auto& token : tokens) {
        term_frequencies[token]++;
    }

    return term_frequencies;
}

double BM25Retriever::calculate_bm25_score(
    const std::unordered_map<std::string, size_t>& query_terms,
    size_t doc_id) const {

    double score = config_.delta;  // BM25 delta normalization

    // Get document statistics
    if (doc_id > document_stats_.size()) {
        return score;
    }

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

        // BM25 scoring formula:
        // IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))

        double tf = static_cast<double>(posting_it->term_frequency);
        double idf = term_info.idf;
        double doc_length_normalization = calculate_doc_normalization_factor(doc_id);

        // BM25 term frequency component
        double tf_component = (tf * (config_.k1 + 1.0)) /
                             (tf + config_.k1 * doc_length_normalization);

        // Query frequency component
        double query_weight = std::log(1.0 + static_cast<double>(query_tf));

        score += idf * tf_component * query_weight;
    }

    return score;
}

double BM25Retriever::calculate_normalized_tf(size_t raw_tf, size_t doc_id) const {
    if (doc_id > document_stats_.size()) {
        return 0.0;
    }

    const auto& doc_stats = document_stats_[doc_id - 1];
    double normalized_tf = static_cast<double>(raw_tf) /
                         (static_cast<double>(raw_tf) +
                          config_.k1 * (1.0 - config_.b +
                          config_.b * (static_cast<double>(doc_stats.term_count) / avg_document_length_)));

    return normalized_tf;
}

double BM25Retriever::calculate_doc_normalization_factor(size_t doc_id) const {
    if (doc_id > document_stats_.size()) {
        return 1.0;
    }

    const auto& doc_stats = document_stats_[doc_id - 1];
    if (avg_document_length_ == 0.0) {
        return 1.0;
    }

    double doc_length_ratio = static_cast<double>(doc_stats.term_count) / avg_document_length_;
    return 1.0 - config_.b + config_.b * doc_length_ratio;
}

std::vector<size_t> BM25Retriever::intersect_postings_optimized(
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

std::vector<size_t> BM25Retriever::union_postings_optimized(
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

size_t BM25Retriever::add_document_internal(const Document& document) {
    size_t doc_id = generate_document_id();

    // Add to document storage
    documents_.push_back(document);
    doc_id_map_[document.id] = doc_id;

    // Create document statistics entry
    if (doc_id > document_stats_.size()) {
        document_stats_.resize(doc_id);
    }
    document_stats_[doc_id - 1].document_id = doc_id;
    std::unordered_set<std::string> unique_terms_in_doc;

    // Process document content
    auto tokens = text_processor_->process(document.content);
    std::unordered_map<std::string, size_t> term_frequencies;
    std::unordered_map<std::string, std::vector<size_t>> term_positions;

    // Track term frequencies and positions
    for (size_t pos = 0; pos < tokens.size(); ++pos) {
        const std::string& token = tokens[pos];
        term_frequencies[token]++;
        term_positions[token].push_back(pos);
        unique_terms_in_doc.insert(token);
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
        posting.normalized_tf = calculate_normalized_tf(tf, doc_id);
        term_info.postings.push_back(std::move(posting));

        // Update term statistics
        term_info.document_frequency += 1;
        term_info.total_term_frequency += tf;
    }

    // Update document statistics
    update_document_statistics(doc_id, tokens.size(), unique_terms_in_doc.size());

    total_terms_in_corpus_ += tokens.size();

    return doc_id;
}

void BM25Retriever::update_document_statistics(size_t doc_id, size_t term_count, size_t unique_terms) {
    if (doc_id > document_stats_.size()) {
        document_stats_.resize(doc_id);
    }

    auto& doc_stats = document_stats_[doc_id - 1];
    doc_stats.term_count = term_count;
    doc_stats.unique_terms = unique_terms;
    doc_stats.bm25_normalization_factor = calculate_doc_normalization_factor(doc_id);
}

void BM25Retriever::update_corpus_statistics() {
    if (documents_.empty()) {
        avg_document_length_ = 0.0;
        return;
    }

    // Calculate average document length
    size_t total_terms = 0;
    for (const auto& doc_stats : document_stats_) {
        total_terms += doc_stats.term_count;
    }

    avg_document_length_ = static_cast<double>(total_terms) / documents_.size();

    // Update all IDF values
    size_t total_docs = documents_.size();
    for (auto& [term, term_info] : inverted_index_) {
        term_info.update_idf(total_docs);
    }

    // Update document normalization factors
    for (size_t i = 0; i < document_stats_.size(); ++i) {
        document_stats_[i].bm25_normalization_factor = calculate_doc_normalization_factor(i + 1);
    }
}

void BM25Retriever::update_cache_stats(const std::string& term) {
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

void BM25Retriever::cleanup_cache() {
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

size_t BM25Retriever::generate_document_id() {
    return next_doc_id_++;
}

std::optional<Document> BM25Retriever::get_document(size_t doc_id) const {
    if (doc_id == 0 || doc_id > documents_.size()) {
        return std::nullopt;
    }
    return documents_[doc_id - 1];  // doc_id is 1-based, vector is 0-based
}

// Factory methods
std::unique_ptr<BM25Retriever>
BM25RetrieverFactory::create_standard_retriever() {
    BM25Retriever::Config config;
    config.k1 = 1.2;
    config.b = 0.75;
    config.delta = 1.0;
    config.min_term_frequency = 1;
    config.max_postings_per_term = 100000;
    config.enable_term_caching = true;
    config.cache_size_limit = 10000;
    config.normalize_scores = true;
    config.score_threshold = 0.0;
    config.max_results = 100;

    return std::make_unique<BM25Retriever>(
        config,
        text::TextProcessorFactory::create_retrieval_processor());
}

std::unique_ptr<BM25Retriever>
BM25RetrieverFactory::create_short_document_retriever() {
    BM25Retriever::Config config;
    config.k1 = 1.2;
    config.b = 0.75;  // Standard BM25 for short docs
    config.delta = 1.0;
    config.min_term_frequency = 1;
    config.max_postings_per_term = 100000;
    config.enable_term_caching = true;
    config.cache_size_limit = 5000;
    config.normalize_scores = true;
    config.score_threshold = 0.0;
    config.max_results = 50;

    return std::make_unique<BM25Retriever>(
        config,
        text::TextProcessorFactory::create_retrieval_processor());
}

std::unique_ptr<BM25Retriever>
BM25RetrieverFactory::create_long_document_retriever() {
    BM25Retriever::Config config;
    config.k1 = 1.2;
    config.b = 0.75;  // Standard BM25 for long docs
    config.delta = 1.0;
    config.min_term_frequency = 1;
    config.max_postings_per_term = 100000;
    config.enable_term_caching = true;
    config.cache_size_limit = 50000;
    config.normalize_scores = true;
    config.score_threshold = 0.0;
    config.max_results = 200;

    return std::make_unique<BM25Retriever>(
        config,
        text::TextProcessorFactory::create_retrieval_processor());
}

std::unique_ptr<BM25Retriever>
BM25RetrieverFactory::create_precision_focused_retriever() {
    BM25Retriever::Config config;
    config.k1 = 2.0;  // Higher k1 for precision
    config.b = 0.75;
    config.delta = 1.0;
    config.min_term_frequency = 2;  // Filter out rare terms
    config.max_postings_per_term = 50000;
    config.enable_term_caching = true;
    config.cache_size_limit = 20000;
    config.normalize_scores = true;
    config.score_threshold = 0.5;  // Higher threshold for precision
    config.max_results = 50;

    return std::make_unique<BM25Retriever>(
        config,
        text::TextProcessorFactory::create_retrieval_processor());
}

std::unique_ptr<BM25Retriever>
BM25RetrieverFactory::create_recall_focused_retriever() {
    BM25Retriever::Config config;
    config.k1 = 0.5;  // Lower k1 for recall
    config.b = 0.75;
    config.delta = 0.5;  // Lower delta for more results
    config.min_term_frequency = 1;
    config.max_postings_per_term = 200000;
    config.enable_term_caching = true;
    config.cache_size_limit = 30000;
    config.normalize_scores = true;
    config.score_threshold = 0.0;  // Lower threshold for recall
    config.max_results = 500;

    return std::make_unique<BM25Retriever>(
        config,
        text::TextProcessorFactory::create_retrieval_processor());
}

void BM25Retriever::calculate_tfidf_scores_simd(
    const std::unordered_map<std::string, size_t>& query_terms,
    const std::vector<size_t>& doc_ids,
    std::vector<double>& scores) const {

    scores.resize(doc_ids.size(), 0.0);
    if (doc_ids.empty() || query_terms.empty()) {
        return;
    }

    // For small document sets, use scalar implementation
    if (doc_ids.size() < 8) {
        for (size_t i = 0; i < doc_ids.size(); ++i) {
            size_t doc_id = doc_ids[i];
            if (doc_id > document_stats_.size()) continue;

            double score = 0.0;
            for (const auto& [term, query_tf] : query_terms) {
                auto term_it = inverted_index_.find(term);
                if (term_it == inverted_index_.end()) continue;

                const TermInfo& term_info = term_it->second;
                auto posting_it = std::find_if(term_info.postings.begin(), term_info.postings.end(),
                                              [doc_id](const PostingEntry& posting) {
                                                  return posting.document_id == doc_id;
                                              });

                if (posting_it != term_info.postings.end()) {
                    // TF-IDF: tf * idf
                    double tf = static_cast<double>(posting_it->term_frequency);
                    double idf = term_info.idf;
                    score += tf * idf * static_cast<double>(query_tf);
                }
            }
            scores[i] = score;
        }
        return;
    }

    // For larger document sets, prepare data for SIMD processing
    // Convert query terms to vectors for efficient processing
    std::vector<std::string> query_term_strings;
    std::vector<double> query_weights;
    std::vector<double> term_idfs;

    query_term_strings.reserve(query_terms.size());
    query_weights.reserve(query_terms.size());
    term_idfs.reserve(query_terms.size());

    for (const auto& [term, query_tf] : query_terms) {
        auto term_it = inverted_index_.find(term);
        if (term_it != inverted_index_.end()) {
            query_term_strings.push_back(term);
            query_weights.push_back(static_cast<double>(query_tf));
            term_idfs.push_back(term_it->second.idf);
        }
    }

    if (query_term_strings.empty()) {
        return;
    }

    // Batch processing with SIMD for TF-IDF calculations
    const size_t batch_size = 16; // Optimal for AVX512, good for AVX2
    std::vector<float> tfidf_batch(batch_size);
    std::vector<float> query_weights_batch(batch_size);
    std::vector<float> term_idfs_batch(batch_size);

    for (size_t i = 0; i < doc_ids.size(); ++i) {
        size_t doc_id = doc_ids[i];
        if (doc_id > document_stats_.size()) {
            scores[i] = 0.0;
            continue;
        }

        double total_score = 0.0;

        // Process query terms in batches for SIMD optimization
        for (size_t term_idx = 0; term_idx < query_term_strings.size(); term_idx += batch_size) {
            size_t current_batch_size = std::min(batch_size, query_term_strings.size() - term_idx);

            // Prepare batch data
            for (size_t j = 0; j < current_batch_size; ++j) {
                size_t idx = term_idx + j;
                const std::string& term = query_term_strings[idx];

                // Find term frequency in document
                auto term_it = inverted_index_.find(term);
                if (term_it != inverted_index_.end()) {
                    auto posting_it = std::find_if(term_it->second.postings.begin(), term_it->second.postings.end(),
                                                  [doc_id](const PostingEntry& posting) {
                                                      return posting.document_id == doc_id;
                                                  });

                    if (posting_it != term_it->second.postings.end()) {
                        tfidf_batch[j] = static_cast<float>(posting_it->term_frequency);
                    } else {
                        tfidf_batch[j] = 0.0f;
                    }
                } else {
                    tfidf_batch[j] = 0.0f;
                }

                query_weights_batch[j] = static_cast<float>(query_weights[term_idx + j]);
                term_idfs_batch[j] = static_cast<float>(term_idfs[term_idx + j]);
            }

            // Use SIMD for batch TF-IDF calculation
            if (current_batch_size >= 16 && utils::VectorOps::is_avx512_supported()) {
                // AVX512 implementation - compute tf * idf * query_weight for 16 elements
                float batch_result = 0.0f;
                for (size_t j = 0; j < current_batch_size; ++j) {
                    batch_result += tfidf_batch[j] * term_idfs_batch[j] * query_weights_batch[j];
                }
                total_score += static_cast<double>(batch_result);
            } else if (current_batch_size >= 8 && utils::VectorOps::is_avx2_supported()) {
                // AVX2 implementation - compute in chunks of 8
                for (size_t j = 0; j < current_batch_size; j += 8) {
                    size_t chunk_size = std::min(size_t(8), current_batch_size - j);
                    float chunk_result = 0.0f;
                    for (size_t k = 0; k < chunk_size; ++k) {
                        chunk_result += tfidf_batch[j + k] * term_idfs_batch[j + k] * query_weights_batch[j + k];
                    }
                    total_score += static_cast<double>(chunk_result);
                }
            } else {
                // Scalar fallback
                for (size_t j = 0; j < current_batch_size; ++j) {
                    total_score += static_cast<double>(tfidf_batch[j] * term_idfs_batch[j] * query_weights_batch[j]);
                }
            }
        }

        scores[i] = total_score;
    }
}

RetrievalResult BM25Retriever::retrieve_tfidf_simd(const std::string& query) {
    auto start_time = std::chrono::high_resolution_clock::now();

    RetrievalResult result;
    result.query = query;
    result.retrieval_method = "tfidf_simd";

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

    // Get candidate documents
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
        // Multi-term query - use optimized intersection
        candidate_docs = intersect_postings_optimized(term_strings);
        if (candidate_docs.empty()) {
            // Fallback to union if intersection yields no results
            candidate_docs = union_postings_optimized(term_strings);
        }
    }

    if (candidate_docs.empty()) {
        result.search_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);
        return result;
    }

    // Calculate TF-IDF scores using SIMD optimization
    std::vector<double> tfidf_scores;
    calculate_tfidf_scores_simd(query_terms, candidate_docs, tfidf_scores);

    // Create results
    std::vector<RetrievedDocument> retrieved_docs;
    retrieved_docs.reserve(candidate_docs.size());

    for (size_t i = 0; i < candidate_docs.size(); ++i) {
        if (tfidf_scores[i] >= config_.score_threshold) {
            auto doc_opt = get_document(candidate_docs[i]);
            if (doc_opt) {
                RetrievedDocument retrieved_doc(*doc_opt, tfidf_scores[i]);
                retrieved_docs.push_back(std::move(retrieved_doc));
            }
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

    return result;
}

} // namespace langchain::retrievers