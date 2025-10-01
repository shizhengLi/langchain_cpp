#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <chrono>
#include <functional>
#include <mutex>
#include <atomic>
#include <cstdint>

namespace langchain {
namespace security {

// Forward declarations
struct User;
struct Role;
struct Permission;
struct Session;
struct SecurityConfig;

/**
 * @brief User role enumeration
 */
enum class UserRole {
    ADMIN,
    USER,
    GUEST,
    CUSTOM
};

/**
 * @brief Permission types
 */
enum class PermissionType {
    READ,
    WRITE,
    DELETE,
    EXECUTE,
    ADMIN
};

/**
 * @brief Authentication methods
 */
enum class AuthMethod {
    PASSWORD,
    TOKEN,
    API_KEY,
    OAUTH,
    LDAP
};

/**
 * @brief Security policy levels
 */
enum class SecurityLevel {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL
};

/**
 * @brief User credentials
 */
struct Credentials {
    std::string username;
    std::string password_hash;
    std::string salt;
    AuthMethod method;
    std::unordered_map<std::string, std::string> metadata;

    Credentials() : method(AuthMethod::PASSWORD) {}
};

/**
 * @brief Security permission
 */
struct Permission {
    std::string id;
    std::string resource;
    PermissionType type;
    std::optional<std::string> condition;

    Permission(std::string id, std::string resource, PermissionType type)
        : id(std::move(id)), resource(std::move(resource)), type(type) {}
};

/**
 * @brief User role with permissions
 */
struct Role {
    std::string id;
    std::string name;
    std::string description;
    std::vector<Permission> permissions;
    SecurityLevel level;

    Role() : level(SecurityLevel::MEDIUM) {}

    Role(std::string id, std::string name, SecurityLevel level = SecurityLevel::MEDIUM)
        : id(std::move(id)), name(std::move(name)), level(level) {}
};

/**
 * @brief User account
 */
struct User {
    std::string id;
    std::string username;
    std::string email;
    Credentials credentials;
    std::vector<Role> roles;
    std::unordered_map<std::string, std::string> metadata;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point last_login;
    bool active;
    bool locked;

    User() : created_at(std::chrono::system_clock::now()),
             last_login(std::chrono::system_clock::now()),
             active(true), locked(false) {}
};

/**
 * @brief Authentication session
 */
struct Session {
    std::string id;
    std::string user_id;
    std::string token;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point expires_at;
    std::string ip_address;
    std::string user_agent;
    bool active;

    Session() : created_at(std::chrono::system_clock::now()),
               active(true) {}
};

/**
 * @brief Security configuration
 */
struct SecurityConfig {
    SecurityLevel level;
    bool enable_encryption;
    bool require_authentication;
    bool session_timeout_enabled;
    std::chrono::seconds session_timeout;
    size_t max_login_attempts;
    std::chrono::seconds lockout_duration;
    bool enable_audit_logging;
    bool require_2fa;

    SecurityConfig() : level(SecurityLevel::MEDIUM),
                      enable_encryption(true),
                      require_authentication(true),
                      session_timeout_enabled(true),
                      session_timeout(std::chrono::seconds(3600)),
                      max_login_attempts(5),
                      lockout_duration(std::chrono::seconds(900)),
                      enable_audit_logging(true),
                      require_2fa(false) {}
};

/**
 * @brief Audit log entry
 */
struct AuditLog {
    std::string id;
    std::chrono::system_clock::time_point timestamp;
    std::string user_id;
    std::string action;
    std::string resource;
    std::string ip_address;
    std::string result;
    std::unordered_map<std::string, std::string> details;

    AuditLog() : timestamp(std::chrono::system_clock::now()) {}
};

/**
 * @brief Authentication interface
 */
class AuthenticationService {
public:
    virtual ~AuthenticationService() = default;

    // User management
    virtual bool create_user(const User& user) = 0;
    virtual std::optional<User> get_user(const std::string& user_id) = 0;
    virtual bool update_user(const User& user) = 0;
    virtual bool delete_user(const std::string& user_id) = 0;

    // Authentication
    virtual std::optional<Session> authenticate(const std::string& username,
                                             const std::string& password,
                                             const std::string& ip_address = "",
                                             const std::string& user_agent = "") = 0;
    virtual bool validate_session(const std::string& session_token) = 0;
    virtual std::optional<Session> get_session(const std::string& session_token) = 0;
    virtual bool logout(const std::string& session_token) = 0;

    // Password management
    virtual std::string hash_password(const std::string& password, const std::string& salt) = 0;
    virtual std::string generate_salt() = 0;
    virtual bool verify_password(const std::string& password, const std::string& hash, const std::string& salt) = 0;

    // Token management
    virtual std::string generate_token() = 0;
    virtual bool is_token_valid(const std::string& token) = 0;
};

/**
 * @brief Authorization interface
 */
class AuthorizationService {
public:
    virtual ~AuthorizationService() = default;

    // Role management
    virtual bool create_role(const Role& role) = 0;
    virtual std::optional<Role> get_role(const std::string& role_id) = 0;
    virtual bool update_role(const Role& role) = 0;
    virtual bool delete_role(const std::string& role_id) = 0;

    // Permission checking
    virtual bool has_permission(const std::string& user_id,
                              const std::string& resource,
                              PermissionType permission) = 0;
    virtual bool has_permission(const Session& session,
                              const std::string& resource,
                              PermissionType permission) = 0;

    // Role assignment
    virtual bool assign_role(const std::string& user_id, const std::string& role_id) = 0;
    virtual bool revoke_role(const std::string& user_id, const std::string& role_id) = 0;
    virtual std::vector<Role> get_user_roles(const std::string& user_id) = 0;
};

/**
 * @brief Encryption interface
 */
class EncryptionService {
public:
    virtual ~EncryptionService() = default;

    // Data encryption
    virtual std::string encrypt(const std::string& plaintext, const std::string& key) = 0;
    virtual std::string decrypt(const std::string& ciphertext, const std::string& key) = 0;

    // Key management
    virtual std::string generate_key() = 0;
    virtual std::string derive_key(const std::string& password, const std::string& salt) = 0;

    // Hash functions
    virtual std::string hash(const std::string& data) = 0;
    virtual std::string hmac(const std::string& data, const std::string& key) = 0;
};

/**
 * @brief Audit logging interface
 */
class AuditService {
public:
    virtual ~AuditService() = default;

    // Logging
    virtual bool log_event(const AuditLog& log) = 0;
    virtual std::vector<AuditLog> get_logs(const std::string& user_id = "",
                                         std::chrono::system_clock::time_point start = {},
                                         std::chrono::system_clock::time_point end = {}) = 0;
    virtual bool clear_old_logs(std::chrono::system_clock::time_point before) = 0;

    // Filtering and searching
    virtual std::vector<AuditLog> search_logs(const std::string& query) = 0;
    virtual std::vector<AuditLog> get_failed_login_attempts(const std::string& user_id = "") = 0;
};

/**
 * @brief Main security manager
 */
class SecurityManager {
private:
    std::unique_ptr<AuthenticationService> auth_service_;
    std::unique_ptr<AuthorizationService> authz_service_;
    std::unique_ptr<EncryptionService> encryption_service_;
    std::unique_ptr<AuditService> audit_service_;
    SecurityConfig config_;
    mutable std::mutex mutex_;

public:
    explicit SecurityManager(const SecurityConfig& config = SecurityConfig{});
    ~SecurityManager() = default;

    // Initialize services
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

    // Encryption
    std::string encrypt_data(const std::string& data);
    std::string decrypt_data(const std::string& encrypted_data);

    // User management
    bool create_user(const User& user);
    std::optional<User> get_user(const std::string& user_id);
    bool update_user(const User& user);
    bool delete_user(const std::string& user_id);

    // Configuration
    void update_config(const SecurityConfig& config);
    SecurityConfig get_config() const;

    // Health check
    bool is_healthy() const;
    std::unordered_map<std::string, std::string> get_stats() const;
};

/**
 * @brief Security middleware for request handling
 */
class SecurityMiddleware {
private:
    SecurityManager* security_manager_;
    std::vector<std::string> public_paths_;
    std::unordered_map<std::string, std::vector<PermissionType>> path_permissions_;

public:
    explicit SecurityMiddleware(SecurityManager* manager);

    // Path configuration
    void add_public_path(const std::string& path);
    void set_path_permissions(const std::string& path, const std::vector<PermissionType>& permissions);

    // Request processing
    bool authenticate_request(const std::string& path, const std::string& session_token);
    bool authorize_request(const std::string& path, const std::string& session_token, PermissionType permission);

    // Utility
    std::optional<std::string> extract_token_from_header(const std::string& auth_header);
    std::string generate_csrf_token();
    bool validate_csrf_token(const std::string& token);
};

/**
 * @brief Utility functions
 */
namespace utils {
    std::string generate_uuid();
    std::string generate_random_string(size_t length);
    std::string secure_compare(const std::string& a, const std::string& b);
    bool is_valid_email(const std::string& email);
    bool is_strong_password(const std::string& password);
    std::string sanitize_input(const std::string& input);
}

} // namespace security
} // namespace langchain