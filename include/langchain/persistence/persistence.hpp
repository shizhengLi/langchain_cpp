#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <map>
#include <variant>
#include <optional>
#include <fstream>
#include <mutex>
#include <chrono>
#include "../core/types.hpp"
#include "../core/config.hpp"

namespace langchain {
namespace persistence {

// Data types for persistence
using FieldValue = std::variant<
    std::string,
    int64_t,
    double,
    bool,
    std::vector<std::string>,
    std::vector<double>,
    std::chrono::system_clock::time_point
>;

// Record structure for database operations
struct Record {
    std::string id;
    std::map<std::string, FieldValue> fields;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;

    Record() : created_at(std::chrono::system_clock::now()),
               updated_at(std::chrono::system_clock::now()) {}
};

// Query operations
enum class QueryOperator {
    EQUALS,
    NOT_EQUALS,
    LESS_THAN,
    LESS_THAN_OR_EQUAL,
    GREATER_THAN,
    GREATER_THAN_OR_EQUAL,
    CONTAINS,
    STARTS_WITH,
    ENDS_WITH,
    IN
};

struct QueryCondition {
    std::string field;
    QueryOperator op;
    FieldValue value;

    QueryCondition(std::string f, QueryOperator o, FieldValue v)
        : field(std::move(f)), op(o), value(std::move(v)) {}
};

struct Query {
    std::vector<QueryCondition> conditions;
    std::string order_by;
    bool ascending = true;
    size_t limit = 100;
    size_t offset = 0;
};

// Persistence interface
class PersistenceBackend {
public:
    virtual ~PersistenceBackend() = default;

    // Basic CRUD operations
    virtual bool save_record(const std::string& collection, const Record& record) = 0;
    virtual std::optional<Record> get_record(const std::string& collection, const std::string& id) = 0;
    virtual bool update_record(const std::string& collection, const std::string& id, const Record& record) = 0;
    virtual bool delete_record(const std::string& collection, const std::string& id) = 0;

    // Query operations
    virtual std::vector<Record> query_records(const std::string& collection, const Query& query) = 0;
    virtual size_t count_records(const std::string& collection, const Query& query) = 0;

    // Collection operations
    virtual bool create_collection(const std::string& collection) = 0;
    virtual bool drop_collection(const std::string& collection) = 0;
    virtual std::vector<std::string> list_collections() = 0;

    // Transaction support
    virtual bool begin_transaction() = 0;
    virtual bool commit_transaction() = 0;
    virtual bool rollback_transaction() = 0;

    // Backup and restore
    virtual bool backup(const std::string& backup_path) = 0;
    virtual bool restore(const std::string& backup_path) = 0;

    // Health and status
    virtual bool is_healthy() const = 0;
    virtual std::map<std::string, std::string> get_stats() const = 0;
};

// JSON file-based persistence backend
class JsonFileBackend : public PersistenceBackend {
private:
    std::string base_path_;
    mutable std::mutex mutex_;
    std::map<std::string, std::string> file_cache_;

    std::string get_collection_path(const std::string& collection) const;
    std::string record_to_json(const Record& record) const;
    Record record_from_json_simple(const std::string& json_str) const;
    bool load_collection(const std::string& collection, std::map<std::string, Record>& records) const;
    bool save_collection(const std::string& collection, const std::map<std::string, Record>& records) const;
    bool matches_condition(const Record& record, const QueryCondition& condition) const;

public:
    explicit JsonFileBackend(const std::string& base_path);
    ~JsonFileBackend() override = default;

    // CRUD operations
    bool save_record(const std::string& collection, const Record& record) override;
    std::optional<Record> get_record(const std::string& collection, const std::string& id) override;
    bool update_record(const std::string& collection, const std::string& id, const Record& record) override;
    bool delete_record(const std::string& collection, const std::string& id) override;

    // Query operations
    std::vector<Record> query_records(const std::string& collection, const Query& query) override;
    size_t count_records(const std::string& collection, const Query& query) override;

    // Collection operations
    bool create_collection(const std::string& collection) override;
    bool drop_collection(const std::string& collection) override;
    std::vector<std::string> list_collections() override;

    // Transaction support (simplified for file backend)
    bool begin_transaction() override;
    bool commit_transaction() override;
    bool rollback_transaction() override;

    // Backup and restore
    bool backup(const std::string& backup_path) override;
    bool restore(const std::string& backup_path) override;

    // Health and status
    bool is_healthy() const override;
    std::map<std::string, std::string> get_stats() const override;
};

// Persistence manager - high-level interface
class PersistenceManager {
private:
    std::unique_ptr<PersistenceBackend> backend_;
    mutable std::mutex mutex_;
    bool transaction_active_ = false;

public:
    explicit PersistenceManager(std::unique_ptr<PersistenceBackend> backend);
    ~PersistenceManager() = default;

    // High-level document operations
    bool save_document(const Document& document);
    std::optional<Document> get_document(const std::string& id);
    bool update_document(const Document& document);
    bool delete_document(const std::string& id);
    std::vector<Document> query_documents(const Query& query);
    size_t count_documents(const Query& query);

    // Configuration persistence
    bool save_config(const RetrievalConfig& config, const std::string& config_id);
    std::optional<RetrievalConfig> get_config(const std::string& config_id);
    bool delete_config(const std::string& config_id);
    std::vector<std::string> list_configs();

    // Index persistence
    bool save_index_metadata(const std::string& index_id, const std::map<std::string, FieldValue>& metadata);
    std::optional<std::map<std::string, FieldValue>> get_index_metadata(const std::string& index_id);
    bool delete_index_metadata(const std::string& index_id);
    std::vector<std::string> list_indexes();

    // Embedding cache persistence
    bool save_embedding_cache(const std::string& text, const std::vector<double>& embedding);
    std::optional<std::vector<double>> get_embedding_cache(const std::string& text);
    bool clear_embedding_cache();

    // Transaction management
    bool begin_transaction();
    bool commit_transaction();
    bool rollback_transaction();

    // Backup operations
    bool backup_all(const std::string& backup_path);
    bool restore_all(const std::string& backup_path);

    // Health and statistics
    bool is_healthy() const;
    std::map<std::string, std::string> get_stats() const;

    // Collection management
    bool ensure_collection(const std::string& collection);
    std::vector<std::string> list_collections() const;
};

// Utility functions
std::string field_value_to_string(const FieldValue& value);
FieldValue string_to_field_value(const std::string& str, const std::string& type_hint);
std::string generate_unique_id();
std::chrono::system_clock::time_point current_time();

// Collections used by the system
namespace collections {
    constexpr const char* DOCUMENTS = "documents";
    constexpr const char* CONFIGS = "configs";
    constexpr const char* INDEXES = "indexes";
    constexpr const char* EMBEDDINGS = "embeddings";
    constexpr const char* METADATA = "metadata";
}

} // namespace persistence
} // namespace langchain