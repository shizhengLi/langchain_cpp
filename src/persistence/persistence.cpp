#include "langchain/persistence/persistence.hpp"
#include "langchain/core/types.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <random>
#include <stdexcept>

namespace langchain {
namespace persistence {

// Utility functions implementation
std::string field_value_to_string(const FieldValue& value) {
    std::ostringstream oss;
    std::visit([&oss](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
            oss << std::quoted(v);
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            oss << "[";
            for (size_t i = 0; i < v.size(); ++i) {
                if (i > 0) oss << ",";
                oss << std::quoted(v[i]);
            }
            oss << "]";
        } else if constexpr (std::is_same_v<T, std::vector<double>>) {
            oss << "[";
            for (size_t i = 0; i < v.size(); ++i) {
                if (i > 0) oss << ",";
                oss << std::fixed << std::setprecision(6) << v[i];
            }
            oss << "]";
        } else if constexpr (std::is_same_v<T, double>) {
            oss << std::fixed << std::setprecision(6) << v;
        } else if constexpr (std::is_same_v<T, std::chrono::system_clock::time_point>) {
            auto time_t = std::chrono::system_clock::to_time_t(v);
            oss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
        } else {
            oss << v;
        }
    }, value);
    return oss.str();
}

std::string generate_unique_id() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    std::ostringstream oss;
    oss << "id_" << timestamp << "_" << dis(gen);
    return oss.str();
}

std::chrono::system_clock::time_point current_time() {
    return std::chrono::system_clock::now();
}

// JsonFileBackend implementation
JsonFileBackend::JsonFileBackend(const std::string& base_path)
    : base_path_(base_path) {
    // Ensure base directory exists
    std::filesystem::create_directories(base_path_);
}

std::string JsonFileBackend::get_collection_path(const std::string& collection) const {
    return base_path_ + "/" + collection + ".json";
}

std::string JsonFileBackend::record_to_json(const Record& record) const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"id\":" << std::quoted(record.id) << ",";
    oss << "\"created_at\":\"" << field_value_to_string(record.created_at) << "\",";
    oss << "\"updated_at\":\"" << field_value_to_string(record.updated_at) << "\",";
    oss << "\"fields\":{";

    bool first = true;
    for (const auto& [key, value] : record.fields) {
        if (!first) oss << ",";
        oss << std::quoted(key) << ":" << field_value_to_string(value);
        first = false;
    }

    oss << "}}";
    return oss.str();
}

Record JsonFileBackend::record_from_json_simple(const std::string& json_str) const {
    Record record;
    record.id = "";
    record.created_at = current_time();
    record.updated_at = current_time();

    // Simple parsing for our record format
    // Parse id
    size_t id_pos = json_str.find("\"id\":");
    if (id_pos != std::string::npos) {
        size_t start = json_str.find("\"", id_pos + 5);
        size_t end = json_str.find("\"", start + 1);
        if (start != std::string::npos && end != std::string::npos) {
            record.id = json_str.substr(start + 1, end - start - 1);
        }
    }

    // Parse fields
    size_t fields_pos = json_str.find("\"fields\":{");
    if (fields_pos != std::string::npos) {
        size_t start = fields_pos + 10; // after "fields":{
        size_t end = start;
        int brace_count = 1;
        while (end < json_str.length() && brace_count > 0) {
            if (json_str[end] == '{') brace_count++;
            else if (json_str[end] == '}') brace_count--;
            end++;
        }

        std::string fields_str = json_str.substr(start, end - start - 1);

        // Parse key-value pairs in fields
        size_t pos = 0;
        while (pos < fields_str.length()) {
            // Find key
            size_t key_start = fields_str.find("\"", pos);
            if (key_start == std::string::npos) break;

            size_t key_end = fields_str.find("\"", key_start + 1);
            if (key_end == std::string::npos) break;

            std::string key = fields_str.substr(key_start + 1, key_end - key_start - 1);

            // Find value
            size_t value_start = fields_str.find(":", key_end);
            if (value_start == std::string::npos) break;
            value_start++;

            // Skip whitespace
            while (value_start < fields_str.length() && isspace(fields_str[value_start])) {
                value_start++;
            }

            size_t value_end = value_start;
            if (fields_str[value_start] == '"') {
                // String value
                value_start++;
                value_end = fields_str.find("\"", value_start);
                if (value_end != std::string::npos) {
                    std::string value = fields_str.substr(value_start, value_end - value_start);
                    record.fields[key] = value;
                    pos = value_end + 1;
                }
            } else if (isdigit(fields_str[value_start]) || fields_str[value_start] == '-') {
                // Numeric value
                while (value_end < fields_str.length() &&
                       (isdigit(fields_str[value_end]) || fields_str[value_end] == '.' ||
                        fields_str[value_end] == '-')) {
                    value_end++;
                }
                std::string value_str = fields_str.substr(value_start, value_end - value_start);
                if (value_str.find('.') != std::string::npos) {
                    record.fields[key] = std::stod(value_str);
                } else {
                    record.fields[key] = static_cast<int64_t>(std::stoll(value_str));
                }
                pos = value_end;
            } else if (fields_str.substr(value_start, 4) == "true") {
                record.fields[key] = true;
                pos = value_start + 4;
            } else if (fields_str.substr(value_start, 5) == "false") {
                record.fields[key] = false;
                pos = value_start + 5;
            } else {
                // Skip to next field
                size_t comma_pos = fields_str.find(",", value_start);
                if (comma_pos != std::string::npos) {
                    pos = comma_pos + 1;
                } else {
                    break;
                }
            }
        }
    }

    return record;
}

bool JsonFileBackend::save_record(const std::string& collection, const Record& record) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::map<std::string, Record> records;
    load_collection(collection, records);

    Record updated_record = record;
    updated_record.updated_at = current_time();
    records[record.id] = updated_record;

    return save_collection(collection, records);
}

std::optional<Record> JsonFileBackend::get_record(const std::string& collection, const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::map<std::string, Record> records;
    if (!load_collection(collection, records)) {
        return std::nullopt;
    }

    auto it = records.find(id);
    return (it != records.end()) ? std::make_optional(it->second) : std::nullopt;
}

bool JsonFileBackend::update_record(const std::string& collection, const std::string& id, const Record& record) {
    return save_record(collection, record); // save_record handles timestamps
}

bool JsonFileBackend::delete_record(const std::string& collection, const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::map<std::string, Record> records;
    if (!load_collection(collection, records)) {
        return false;
    }

    auto it = records.find(id);
    if (it == records.end()) {
        return false;
    }

    records.erase(it);
    return save_collection(collection, records);
}

std::vector<Record> JsonFileBackend::query_records(const std::string& collection, const Query& query) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::map<std::string, Record> records;
    if (!load_collection(collection, records)) {
        return {};
    }

    std::vector<Record> results;

    for (const auto& [id, record] : records) {
        bool matches = true;

        for (const auto& condition : query.conditions) {
            if (!matches_condition(record, condition)) {
                matches = false;
                break;
            }
        }

        if (matches) {
            results.push_back(record);
        }
    }

    // Apply ordering
    if (!query.order_by.empty()) {
        std::sort(results.begin(), results.end(), [&query](const Record& a, const Record& b) {
            auto it_a = a.fields.find(query.order_by);
            auto it_b = b.fields.find(query.order_by);

            if (it_a == a.fields.end() || it_b == b.fields.end()) {
                return query.ascending;
            }

            // Simple comparison for strings and numbers
            if (std::holds_alternative<std::string>(it_a->second) &&
                std::holds_alternative<std::string>(it_b->second)) {
                const auto& str_a = std::get<std::string>(it_a->second);
                const auto& str_b = std::get<std::string>(it_b->second);
                return query.ascending ? (str_a < str_b) : (str_a > str_b);
            }

            if (std::holds_alternative<int64_t>(it_a->second) &&
                std::holds_alternative<int64_t>(it_b->second)) {
                const auto& num_a = std::get<int64_t>(it_a->second);
                const auto& num_b = std::get<int64_t>(it_b->second);
                return query.ascending ? (num_a < num_b) : (num_a > num_b);
            }

            return query.ascending;
        });
    }

    // Apply pagination
    size_t start = std::min(query.offset, results.size());
    size_t end = std::min(start + query.limit, results.size());

    if (start >= results.size()) {
        return {};
    }

    return std::vector<Record>(results.begin() + start, results.begin() + end);
}

size_t JsonFileBackend::count_records(const std::string& collection, const Query& query) {
    Query count_query = query;
    count_query.limit = SIZE_MAX;
    count_query.offset = 0;
    return query_records(collection, count_query).size();
}

bool JsonFileBackend::create_collection(const std::string& collection) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string path = get_collection_path(collection);
    if (std::filesystem::exists(path)) {
        return false; // Already exists
    }

    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    file << "{}"; // Empty JSON object
    return true;
}

bool JsonFileBackend::drop_collection(const std::string& collection) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string path = get_collection_path(collection);
    return std::filesystem::remove(path);
}

std::vector<std::string> JsonFileBackend::list_collections() {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> collections;

    for (const auto& entry : std::filesystem::directory_iterator(base_path_)) {
        if (entry.path().extension() == ".json") {
            collections.push_back(entry.path().stem().string());
        }
    }

    return collections;
}

bool JsonFileBackend::begin_transaction() {
    // File backend doesn't support real transactions
    // For simplicity, just return true
    return true;
}

bool JsonFileBackend::commit_transaction() {
    return true;
}

bool JsonFileBackend::rollback_transaction() {
    return true;
}

bool JsonFileBackend::backup(const std::string& backup_path) {
    try {
        if (!std::filesystem::create_directories(backup_path)) {
            return false;
        }

        for (const auto& entry : std::filesystem::directory_iterator(base_path_)) {
            if (entry.path().extension() == ".json") {
                std::filesystem::copy_file(
                    entry.path(),
                    backup_path + "/" + entry.path().filename().string(),
                    std::filesystem::copy_options::overwrite_existing
                );
            }
        }

        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool JsonFileBackend::restore(const std::string& backup_path) {
    try {
        for (const auto& entry : std::filesystem::directory_iterator(backup_path)) {
            if (entry.path().extension() == ".json") {
                std::filesystem::copy_file(
                    entry.path(),
                    base_path_ + "/" + entry.path().filename().string(),
                    std::filesystem::copy_options::overwrite_existing
                );
            }
        }

        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool JsonFileBackend::is_healthy() const {
    return std::filesystem::exists(base_path_) &&
           std::filesystem::is_directory(base_path_);
}

std::map<std::string, std::string> JsonFileBackend::get_stats() const {
    std::map<std::string, std::string> stats;

    stats["backend_type"] = "json_file";
    stats["base_path"] = base_path_;
    stats["healthy"] = is_healthy() ? "true" : "false";

    try {
        size_t total_files = 0;
        size_t total_size = 0;

        for (const auto& entry : std::filesystem::directory_iterator(base_path_)) {
            if (entry.path().extension() == ".json") {
                total_files++;
                total_size += std::filesystem::file_size(entry.path());
            }
        }

        stats["collections"] = std::to_string(total_files);
        stats["total_size_bytes"] = std::to_string(total_size);
    } catch (const std::exception&) {
        stats["error"] = "Unable to calculate stats";
    }

    return stats;
}

bool JsonFileBackend::load_collection(const std::string& collection, std::map<std::string, Record>& records) const {
    std::string path = get_collection_path(collection);

    if (!std::filesystem::exists(path)) {
        return const_cast<JsonFileBackend*>(this)->create_collection(collection);
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // For now, implement simple parsing for our specific format
    records.clear();

    // If empty JSON object, return empty records
    if (content == "{}") {
        return true;
    }

    // Simple parsing for our record format: {"id1":{...},"id2":{...}}
    size_t pos = 0;
    while (pos < content.length()) {
        // Find next ID start
        size_t id_start = content.find("\"", pos);
        if (id_start == std::string::npos) break;

        size_t id_end = content.find("\"", id_start + 1);
        if (id_end == std::string::npos) break;

        std::string id = content.substr(id_start + 1, id_end - id_start - 1);

        // Find record object start
        size_t record_start = content.find("{", id_end);
        if (record_start == std::string::npos) break;

        // Find matching closing brace
        int brace_count = 1;
        size_t record_end = record_start + 1;
        while (record_end < content.length() && brace_count > 0) {
            if (content[record_end] == '{') brace_count++;
            else if (content[record_end] == '}') brace_count--;
            record_end++;
        }

        if (brace_count == 0) {
            std::string record_json = content.substr(record_start, record_end - record_start);
            Record record = record_from_json_simple(record_json);
            records[id] = record;
        }

        pos = record_end;
    }

    return true;
}

bool JsonFileBackend::save_collection(const std::string& collection, const std::map<std::string, Record>& records) const {
    std::string path = get_collection_path(collection);

    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    file << "{";
    bool first = true;
    for (const auto& [id, record] : records) {
        if (!first) file << ",";
        file << std::quoted(id) << ":" << record_to_json(record);
        first = false;
    }
    file << "}";

    return true;
}

bool JsonFileBackend::matches_condition(const Record& record, const QueryCondition& condition) const {
    auto it = record.fields.find(condition.field);
    if (it == record.fields.end()) {
        return false;
    }

    const auto& field_value = it->second;
    const auto& condition_value = condition.value;

    // Simplified matching - in a real implementation, we'd handle all type combinations
    if (std::holds_alternative<std::string>(field_value) &&
        std::holds_alternative<std::string>(condition_value)) {

        const auto& str_field = std::get<std::string>(field_value);
        const auto& str_condition = std::get<std::string>(condition_value);

        switch (condition.op) {
            case QueryOperator::EQUALS:
                return str_field == str_condition;
            case QueryOperator::CONTAINS:
                return str_field.find(str_condition) != std::string::npos;
            case QueryOperator::STARTS_WITH:
                return str_field.rfind(str_condition, 0) == 0;
            default:
                return false;
        }
    }

    if (std::holds_alternative<int64_t>(field_value) &&
        std::holds_alternative<int64_t>(condition_value)) {

        const auto& num_field = std::get<int64_t>(field_value);
        const auto& num_condition = std::get<int64_t>(condition_value);

        switch (condition.op) {
            case QueryOperator::EQUALS:
                return num_field == num_condition;
            case QueryOperator::LESS_THAN:
                return num_field < num_condition;
            case QueryOperator::GREATER_THAN:
                return num_field > num_condition;
            default:
                return false;
        }
    }

    return false;
}

// PersistenceManager implementation
PersistenceManager::PersistenceManager(std::unique_ptr<PersistenceBackend> backend)
    : backend_(std::move(backend)) {}

bool PersistenceManager::save_document(const Document& document) {
    if (!ensure_collection(collections::DOCUMENTS)) {
        return false;
    }

    Record record;
    record.id = document.id;
    record.fields["content"] = document.content;

    // Save metadata
    for (const auto& [key, value] : document.metadata) {
        record.fields["meta_" + key] = value;
    }

    return backend_->save_record(collections::DOCUMENTS, record);
}

bool PersistenceManager::update_document(const Document& document) {
    // Update is the same as save for this implementation
    return save_document(document);
}

bool PersistenceManager::delete_document(const std::string& id) {
    return backend_->delete_record(collections::DOCUMENTS, id);
}

std::vector<Document> PersistenceManager::query_documents(const Query& query) {
    auto records = backend_->query_records(collections::DOCUMENTS, query);
    std::vector<Document> documents;

    for (const auto& record : records) {
        Document doc;
        doc.id = record.id;

        if (record.fields.find("content") != record.fields.end() &&
            std::holds_alternative<std::string>(record.fields.at("content"))) {
            doc.content = std::get<std::string>(record.fields.at("content"));
        }

        // Restore metadata
        for (const auto& [key, value] : record.fields) {
            if (key.find("meta_") == 0) {
                std::string meta_key = key.substr(5);
                if (std::holds_alternative<std::string>(value)) {
                    doc.metadata[meta_key] = std::get<std::string>(value);
                }
            }
        }

        documents.push_back(doc);
    }

    return documents;
}

size_t PersistenceManager::count_documents(const Query& query) {
    return backend_->count_records(collections::DOCUMENTS, query);
}

bool PersistenceManager::save_config(const RetrievalConfig& config, const std::string& config_id) {
    // Placeholder implementation
    if (!ensure_collection(collections::CONFIGS)) {
        return false;
    }

    Record record;
    record.id = config_id;
    record.fields["type"] = std::string("retrieval_config");

    return backend_->save_record(collections::CONFIGS, record);
}

std::optional<RetrievalConfig> PersistenceManager::get_config(const std::string& config_id) {
    // Placeholder implementation
    auto record = backend_->get_record(collections::CONFIGS, config_id);
    if (!record) {
        return std::nullopt;
    }

    RetrievalConfig config;
    // Basic configuration restoration would go here
    return config;
}

bool PersistenceManager::delete_config(const std::string& config_id) {
    return backend_->delete_record(collections::CONFIGS, config_id);
}

std::vector<std::string> PersistenceManager::list_configs() {
    auto records = backend_->query_records(collections::CONFIGS, Query{});
    std::vector<std::string> configs;

    for (const auto& record : records) {
        configs.push_back(record.id);
    }

    return configs;
}

bool PersistenceManager::save_index_metadata(const std::string& index_id, const std::map<std::string, FieldValue>& metadata) {
    if (!ensure_collection(collections::INDEXES)) {
        return false;
    }

    Record record;
    record.id = index_id;
    record.fields = metadata;
    record.fields["index_id"] = index_id;

    return backend_->save_record(collections::INDEXES, record);
}

std::optional<std::map<std::string, FieldValue>> PersistenceManager::get_index_metadata(const std::string& index_id) {
    auto record = backend_->get_record(collections::INDEXES, index_id);
    if (!record) {
        return std::nullopt;
    }

    return record->fields;
}

bool PersistenceManager::delete_index_metadata(const std::string& index_id) {
    return backend_->delete_record(collections::INDEXES, index_id);
}

std::vector<std::string> PersistenceManager::list_indexes() {
    auto records = backend_->query_records(collections::INDEXES, Query{});
    std::vector<std::string> indexes;

    for (const auto& record : records) {
        indexes.push_back(record.id);
    }

    return indexes;
}

bool PersistenceManager::save_embedding_cache(const std::string& text, const std::vector<double>& embedding) {
    if (!ensure_collection(collections::EMBEDDINGS)) {
        return false;
    }

    Record record;
    record.id = std::to_string(std::hash<std::string>{}(text)); // Use hash as ID
    record.fields["text"] = text;
    record.fields["embedding"] = embedding;

    return backend_->save_record(collections::EMBEDDINGS, record);
}

std::optional<std::vector<double>> PersistenceManager::get_embedding_cache(const std::string& text) {
    std::string id = std::to_string(std::hash<std::string>{}(text));
    auto record = backend_->get_record(collections::EMBEDDINGS, id);

    if (!record) {
        return std::nullopt;
    }

    if (record->fields.find("embedding") != record->fields.end() &&
        std::holds_alternative<std::vector<double>>(record->fields.at("embedding"))) {
        return std::get<std::vector<double>>(record->fields.at("embedding"));
    }

    return std::nullopt;
}

bool PersistenceManager::clear_embedding_cache() {
    return backend_->drop_collection(collections::EMBEDDINGS) &&
           backend_->create_collection(collections::EMBEDDINGS);
}

bool PersistenceManager::begin_transaction() {
    transaction_active_ = true;
    return backend_->begin_transaction();
}

bool PersistenceManager::commit_transaction() {
    if (transaction_active_) {
        transaction_active_ = false;
        return backend_->commit_transaction();
    }
    return false;
}

bool PersistenceManager::rollback_transaction() {
    if (transaction_active_) {
        transaction_active_ = false;
        return backend_->rollback_transaction();
    }
    return false;
}

bool PersistenceManager::backup_all(const std::string& backup_path) {
    return backend_->backup(backup_path);
}

bool PersistenceManager::restore_all(const std::string& backup_path) {
    return backend_->restore(backup_path);
}

std::optional<Document> PersistenceManager::get_document(const std::string& id) {
    auto record = backend_->get_record(collections::DOCUMENTS, id);
    if (!record) {
        return std::nullopt;
    }

    Document document;
    document.id = record->id;

    if (record->fields.find("content") != record->fields.end() &&
        std::holds_alternative<std::string>(record->fields.at("content"))) {
        document.content = std::get<std::string>(record->fields.at("content"));
    }

    // Restore metadata
    for (const auto& [key, value] : record->fields) {
        if (key.find("meta_") == 0) {
            std::string meta_key = key.substr(5); // Remove "meta_" prefix
            if (std::holds_alternative<std::string>(value)) {
                document.metadata[meta_key] = std::get<std::string>(value);
            }
        }
    }

    return document;
}

bool PersistenceManager::ensure_collection(const std::string& collection) {
    return backend_->create_collection(collection) ||
           std::find(backend_->list_collections().begin(),
                    backend_->list_collections().end(),
                    collection) != backend_->list_collections().end();
}

bool PersistenceManager::is_healthy() const {
    return backend_->is_healthy();
}

std::map<std::string, std::string> PersistenceManager::get_stats() const {
    return backend_->get_stats();
}

std::vector<std::string> PersistenceManager::list_collections() const {
    return backend_->list_collections();
}

} // namespace persistence
} // namespace langchain