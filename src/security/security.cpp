#include "langchain/security/security.hpp"
#include "langchain/utils/logging.hpp"
#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <regex>
#include <openssl/sha.h>
#include <openssl/aes.h>
#include <openssl/rand.h>
#include <openssl/hmac.h>
#include <fstream>
#include <filesystem>

namespace langchain {
namespace security {

// Utility functions implementation
namespace utils {

std::string generate_uuid() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    std::stringstream ss;
    for (int i = 0; i < 32; ++i) {
        if (i == 8 || i == 12 || i == 16 || i == 20) {
            ss << "-";
        }
        ss << std::hex << dis(gen);
    }
    return ss.str();
}

std::string generate_random_string(size_t length) {
    const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, chars.length() - 1);

    std::string result;
    result.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        result += chars[dis(gen)];
    }
    return result;
}

std::string secure_compare(const std::string& a, const std::string& b) {
    if (a.length() != b.length()) {
        return "false";
    }

    int result = 0;
    for (size_t i = 0; i < a.length(); ++i) {
        result |= a[i] ^ b[i];
    }
    return result == 0 ? "true" : "false";
}

bool is_valid_email(const std::string& email) {
    std::regex email_regex(R"(^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$)");
    return std::regex_match(email, email_regex);
}

bool is_strong_password(const std::string& password) {
    if (password.length() < 8) return false;

    bool has_upper = false, has_lower = false, has_digit = false, has_special = false;
    for (char c : password) {
        if (isupper(c)) has_upper = true;
        else if (islower(c)) has_lower = true;
        else if (isdigit(c)) has_digit = true;
        else if (ispunct(c)) has_special = true;
    }

    return has_upper && has_lower && has_digit && has_special;
}

std::string sanitize_input(const std::string& input) {
    std::string result;
    result.reserve(input.length());

    for (char c : input) {
        if (isprint(c) && c != '<' && c != '>' && c != '&' && c != '"' && c != '\'') {
            result += c;
        }
    }

    return result;
}

} // namespace utils

// Simple in-memory authentication service
class InMemoryAuthenticationService : public AuthenticationService {
private:
    std::unordered_map<std::string, User> users_;
    std::unordered_map<std::string, Session> sessions_;
    mutable std::mutex mutex_;

public:
    bool create_user(const User& user) override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (users_.find(user.username) != users_.end()) {
            return false;
        }

        users_[user.username] = user;
        return true;
    }

    std::optional<User> get_user(const std::string& user_id) override {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = users_.find(user_id);
        return (it != users_.end()) ? std::make_optional(it->second) : std::nullopt;
    }

    bool update_user(const User& user) override {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = users_.find(user.username);
        if (it == users_.end()) {
            return false;
        }

        it->second = user;
        return true;
    }

    bool delete_user(const std::string& user_id) override {
        std::lock_guard<std::mutex> lock(mutex_);

        return users_.erase(user_id) > 0;
    }

    std::optional<Session> authenticate(const std::string& username,
                                      const std::string& password,
                                      const std::string& ip_address,
                                      const std::string& user_agent) override {
        std::lock_guard<std::mutex> lock(mutex_);

        auto user_it = users_.find(username);
        if (user_it == users_.end()) {
            return std::nullopt;
        }

        const User& user = user_it->second;
        if (!user.active || user.locked) {
            return std::nullopt;
        }

        if (!verify_password(password, user.credentials.password_hash, user.credentials.salt)) {
            return std::nullopt;
        }

        // Create session
        Session session;
        session.id = utils::generate_uuid();
        session.user_id = user.id;
        session.token = generate_token();
        session.expires_at = std::chrono::system_clock::now() + std::chrono::hours(1);
        session.ip_address = ip_address;
        session.user_agent = user_agent;

        sessions_[session.token] = session;

        // Update user last login
        const_cast<User&>(user).last_login = std::chrono::system_clock::now();

        return session;
    }

    bool validate_session(const std::string& session_token) override {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = sessions_.find(session_token);
        if (it == sessions_.end()) {
            return false;
        }

        const Session& session = it->second;
        if (!session.active || session.expires_at < std::chrono::system_clock::now()) {
            sessions_.erase(it);
            return false;
        }

        return true;
    }

    std::optional<Session> get_session(const std::string& session_token) override {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = sessions_.find(session_token);
        if (it == sessions_.end()) {
            return std::nullopt;
        }

        const Session& session = it->second;
        if (!session.active || session.expires_at < std::chrono::system_clock::now()) {
            sessions_.erase(it);
            return std::nullopt;
        }

        return session;
    }

    bool logout(const std::string& session_token) override {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = sessions_.find(session_token);
        if (it == sessions_.end()) {
            return false;
        }

        sessions_.erase(it);
        return true;
    }

    std::string hash_password(const std::string& password, const std::string& salt) override {
        std::string salted_password = salt + password;
        unsigned char hash[SHA256_DIGEST_LENGTH];

        SHA256(reinterpret_cast<const unsigned char*>(salted_password.c_str()),
               salted_password.length(), hash);

        std::stringstream ss;
        for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
        }

        return ss.str();
    }

    std::string generate_salt() override {
        return utils::generate_random_string(32);
    }

    bool verify_password(const std::string& password, const std::string& hash, const std::string& salt) override {
        std::string computed_hash = hash_password(password, salt);
        return utils::secure_compare(computed_hash, hash) == "true";
    }

    std::string generate_token() override {
        return utils::generate_random_string(64);
    }

    bool is_token_valid(const std::string& token) override {
        return !token.empty() && token.length() >= 32;
    }
};

// Simple in-memory authorization service
class InMemoryAuthorizationService : public AuthorizationService {
private:
    std::unordered_map<std::string, Role> roles_;
    std::unordered_map<std::string, std::vector<std::string>> user_roles_; // user_id -> role_ids
    mutable std::mutex mutex_;

public:
    bool create_role(const Role& role) override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (roles_.find(role.id) != roles_.end()) {
            return false;
        }

        roles_[role.id] = role;
        return true;
    }

    std::optional<Role> get_role(const std::string& role_id) override {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = roles_.find(role_id);
        return (it != roles_.end()) ? std::make_optional(it->second) : std::nullopt;
    }

    bool update_role(const Role& role) override {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = roles_.find(role.id);
        if (it == roles_.end()) {
            return false;
        }

        it->second = role;
        return true;
    }

    bool delete_role(const std::string& role_id) override {
        std::lock_guard<std::mutex> lock(mutex_);

        return roles_.erase(role_id) > 0;
    }

    bool has_permission(const std::string& user_id,
                       const std::string& resource,
                       PermissionType permission) override {
        std::lock_guard<std::mutex> lock(mutex_);

        auto role_it = user_roles_.find(user_id);
        if (role_it == user_roles_.end()) {
            return false;
        }

        for (const std::string& role_id : role_it->second) {
            auto role_data_it = roles_.find(role_id);
            if (role_data_it != roles_.end()) {
                const Role& role = role_data_it->second;
                for (const Permission& perm : role.permissions) {
                    if (perm.resource == resource && perm.type == permission) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    bool has_permission(const Session& session,
                       const std::string& resource,
                       PermissionType permission) override {
        return has_permission(session.user_id, resource, permission);
    }

    bool assign_role(const std::string& user_id, const std::string& role_id) override {
        std::lock_guard<std::mutex> lock(mutex_);

        auto role_it = roles_.find(role_id);
        if (role_it == roles_.end()) {
            return false;
        }

        user_roles_[user_id].push_back(role_id);
        return true;
    }

    bool revoke_role(const std::string& user_id, const std::string& role_id) override {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = user_roles_.find(user_id);
        if (it == user_roles_.end()) {
            return false;
        }

        auto& roles = it->second;
        auto role_it = std::find(roles.begin(), roles.end(), role_id);
        if (role_it != roles.end()) {
            roles.erase(role_it);
            return true;
        }

        return false;
    }

    std::vector<Role> get_user_roles(const std::string& user_id) override {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<Role> result;
        auto it = user_roles_.find(user_id);
        if (it != user_roles_.end()) {
            for (const std::string& role_id : it->second) {
                auto role_it = roles_.find(role_id);
                if (role_it != roles_.end()) {
                    result.push_back(role_it->second);
                }
            }
        }

        return result;
    }
};

// Simple encryption service using OpenSSL
class OpenSSLEncryptionService : public EncryptionService {
public:
    std::string encrypt(const std::string& plaintext, const std::string& key) override {
        if (key.length() != 32) { // AES-256 requires 32-byte key
            return "";
        }

        // Generate random IV
        unsigned char iv[16];
        if (RAND_bytes(iv, 16) != 1) {
            return "";
        }

        // Prepare output buffer
        std::string ciphertext;
        ciphertext.resize(16 + plaintext.length() + AES_BLOCK_SIZE); // IV + encrypted data + padding

        AES_KEY aes_key;
        if (AES_set_encrypt_key(reinterpret_cast<const unsigned char*>(key.c_str()), 256, &aes_key) != 0) {
            return "";
        }

        // Copy IV to output
        std::copy(iv, iv + 16, ciphertext.begin());

        // Encrypt
        int num_bytes = 0;
        AES_cbc_encrypt(reinterpret_cast<const unsigned char*>(plaintext.c_str()),
                       reinterpret_cast<unsigned char*>(&ciphertext[16]),
                       plaintext.length(),
                       &aes_key,
                       iv,
                       AES_ENCRYPT);

        // Resize to actual encrypted length
        ciphertext.resize(16 + ((plaintext.length() + AES_BLOCK_SIZE) / AES_BLOCK_SIZE) * AES_BLOCK_SIZE);

        return ciphertext;
    }

    std::string decrypt(const std::string& ciphertext, const std::string& key) override {
        if (key.length() != 32 || ciphertext.length() < 16) {
            return "";
        }

        // Extract IV
        unsigned char iv[16];
        std::copy(ciphertext.begin(), ciphertext.begin() + 16, iv);

        // Prepare output buffer
        std::string plaintext;
        plaintext.resize(ciphertext.length() - 16);

        AES_KEY aes_key;
        if (AES_set_decrypt_key(reinterpret_cast<const unsigned char*>(key.c_str()), 256, &aes_key) != 0) {
            return "";
        }

        // Decrypt
        AES_cbc_encrypt(reinterpret_cast<const unsigned char*>(&ciphertext[16]),
                       reinterpret_cast<unsigned char*>(&plaintext[0]),
                       plaintext.length(),
                       &aes_key,
                       iv,
                       AES_DECRYPT);

        // Remove padding
        if (!plaintext.empty()) {
            size_t padding_length = plaintext.back();
            if (padding_length <= plaintext.length()) {
                plaintext.resize(plaintext.length() - padding_length);
            }
        }

        return plaintext;
    }

    std::string generate_key() override {
        return utils::generate_random_string(32);
    }

    std::string derive_key(const std::string& password, const std::string& salt) override {
        std::string salted_password = salt + password;
        unsigned char hash[SHA256_DIGEST_LENGTH];

        SHA256(reinterpret_cast<const unsigned char*>(salted_password.c_str()),
               salted_password.length(), hash);

        return std::string(reinterpret_cast<char*>(hash), SHA256_DIGEST_LENGTH);
    }

    std::string hash(const std::string& data) override {
        unsigned char hash[SHA256_DIGEST_LENGTH];

        SHA256(reinterpret_cast<const unsigned char*>(data.c_str()),
               data.length(), hash);

        std::stringstream ss;
        for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
        }

        return ss.str();
    }

    std::string hmac(const std::string& data, const std::string& key) override {
        unsigned char hmac_result[32];

        HMAC(EVP_sha256(),
             reinterpret_cast<const unsigned char*>(key.c_str()),
             key.length(),
             reinterpret_cast<const unsigned char*>(data.c_str()),
             data.length(),
             hmac_result,
             nullptr);

        std::stringstream ss;
        for (int i = 0; i < 32; ++i) {
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hmac_result[i]);
        }

        return ss.str();
    }
};

// Simple in-memory audit service
class InMemoryAuditService : public AuditService {
private:
    std::vector<AuditLog> logs_;
    mutable std::mutex mutex_;

public:
    bool log_event(const AuditLog& log) override {
        std::lock_guard<std::mutex> lock(mutex_);

        AuditLog new_log = log;
        new_log.id = utils::generate_uuid();
        new_log.timestamp = std::chrono::system_clock::now();

        logs_.push_back(new_log);
        return true;
    }

    std::vector<AuditLog> get_logs(const std::string& user_id,
                                  std::chrono::system_clock::time_point start,
                                  std::chrono::system_clock::time_point end) override {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<AuditLog> result;
        for (const AuditLog& log : logs_) {
            bool matches = true;

            if (!user_id.empty() && log.user_id != user_id) {
                matches = false;
            }

            if (start != std::chrono::system_clock::time_point{} && log.timestamp < start) {
                matches = false;
            }

            if (end != std::chrono::system_clock::time_point{} && log.timestamp > end) {
                matches = false;
            }

            if (matches) {
                result.push_back(log);
            }
        }

        return result;
    }

    bool clear_old_logs(std::chrono::system_clock::time_point before) override {
        std::lock_guard<std::mutex> lock(mutex_);

        logs_.erase(
            std::remove_if(logs_.begin(), logs_.end(),
                          [before](const AuditLog& log) {
                              return log.timestamp < before;
                          }),
            logs_.end()
        );

        return true;
    }

    std::vector<AuditLog> search_logs(const std::string& query) override {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<AuditLog> result;
        for (const AuditLog& log : logs_) {
            if (log.action.find(query) != std::string::npos ||
                log.resource.find(query) != std::string::npos ||
                log.result.find(query) != std::string::npos) {
                result.push_back(log);
            }
        }

        return result;
    }

    std::vector<AuditLog> get_failed_login_attempts(const std::string& user_id) override {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<AuditLog> result;
        for (const AuditLog& log : logs_) {
            if (log.action == "LOGIN" && log.result == "FAILED") {
                if (user_id.empty() || log.user_id == user_id) {
                    result.push_back(log);
                }
            }
        }

        return result;
    }
};

// SecurityManager implementation
SecurityManager::SecurityManager(const SecurityConfig& config)
    : config_(config) {
}

bool SecurityManager::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);

    try {
        auth_service_ = std::make_unique<InMemoryAuthenticationService>();
        authz_service_ = std::make_unique<InMemoryAuthorizationService>();
        encryption_service_ = std::make_unique<OpenSSLEncryptionService>();
        audit_service_ = std::make_unique<InMemoryAuditService>();

        // Create default roles
        Role admin_role("admin", "Administrator", SecurityLevel::HIGH);
        admin_role.permissions.emplace_back("admin_all", "*", PermissionType::ADMIN);
        authz_service_->create_role(admin_role);

        Role user_role("user", "User", SecurityLevel::MEDIUM);
        user_role.permissions.emplace_back("user_read", "documents", PermissionType::READ);
        user_role.permissions.emplace_back("user_write", "documents", PermissionType::WRITE);
        authz_service_->create_role(user_role);

        LOG_INFO("Security manager initialized successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to initialize security manager: " + std::string(e.what()));
        return false;
    }
}

void SecurityManager::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);

    auth_service_.reset();
    authz_service_.reset();
    encryption_service_.reset();
    audit_service_.reset();

    LOG_INFO("Security manager shutdown completed");
}

std::optional<Session> SecurityManager::login(const std::string& username,
                                             const std::string& password,
                                             const std::string& ip_address,
                                             const std::string& user_agent) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!config_.require_authentication) {
        return std::nullopt;
    }

    auto session = auth_service_->authenticate(username, password, ip_address, user_agent);

    if (session) {
        AuditLog log;
        log.user_id = session->user_id;
        log.action = "LOGIN";
        log.resource = "authentication";
        log.ip_address = ip_address;
        log.result = "SUCCESS";
        audit_service_->log_event(log);

        LOG_INFO("User " + username + " logged in successfully from " + ip_address);
    } else {
        AuditLog log;
        log.user_id = username; // Use username as user_id for failed attempts
        log.action = "LOGIN";
        log.resource = "authentication";
        log.ip_address = ip_address;
        log.result = "FAILED";
        audit_service_->log_event(log);

        LOG_WARN("Failed login attempt for user " + username + " from " + ip_address);
    }

    return session;
}

bool SecurityManager::logout(const std::string& session_token) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto session = auth_service_->get_session(session_token);
    bool result = auth_service_->logout(session_token);

    if (result && session) {
        AuditLog log;
        log.user_id = session->user_id;
        log.action = "LOGOUT";
        log.resource = "authentication";
        log.result = "SUCCESS";
        audit_service_->log_event(log);

        LOG_INFO("User logged out successfully");
    }

    return result;
}

bool SecurityManager::validate_session(const std::string& session_token) {
    std::lock_guard<std::mutex> lock(mutex_);

    return auth_service_->validate_session(session_token);
}

std::optional<User> SecurityManager::get_current_user(const std::string& session_token) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto session = auth_service_->get_session(session_token);
    if (!session) {
        return std::nullopt;
    }

    return auth_service_->get_user(session->user_id);
}

bool SecurityManager::check_permission(const std::string& session_token,
                                      const std::string& resource,
                                      PermissionType permission) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto session = auth_service_->get_session(session_token);
    if (!session) {
        return false;
    }

    bool has_permission = authz_service_->has_permission(*session, resource, permission);

    AuditLog log;
    log.user_id = session->user_id;
    log.action = "AUTHORIZE";
    log.resource = resource;
    log.result = has_permission ? "SUCCESS" : "DENIED";
    audit_service_->log_event(log);

    return has_permission;
}

std::string SecurityManager::encrypt_data(const std::string& data) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!config_.enable_encryption) {
        return data;
    }

    std::string key = encryption_service_->derive_key("default_key", "langchain_salt");
    return encryption_service_->encrypt(data, key);
}

std::string SecurityManager::decrypt_data(const std::string& encrypted_data) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!config_.enable_encryption) {
        return encrypted_data;
    }

    std::string key = encryption_service_->derive_key("default_key", "langchain_salt");
    return encryption_service_->decrypt(encrypted_data, key);
}

bool SecurityManager::create_user(const User& user) {
    std::lock_guard<std::mutex> lock(mutex_);

    User new_user = user;
    new_user.id = utils::generate_uuid();
    new_user.credentials.salt = auth_service_->generate_salt();
    new_user.credentials.password_hash = auth_service_->hash_password(
        user.credentials.password_hash, new_user.credentials.salt);

    bool result = auth_service_->create_user(new_user);

    if (result) {
        AuditLog log;
        log.user_id = new_user.id;
        log.action = "CREATE_USER";
        log.resource = "user_management";
        log.result = "SUCCESS";
        audit_service_->log_event(log);
    }

    return result;
}

std::optional<User> SecurityManager::get_user(const std::string& user_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    return auth_service_->get_user(user_id);
}

bool SecurityManager::update_user(const User& user) {
    std::lock_guard<std::mutex> lock(mutex_);

    bool result = auth_service_->update_user(user);

    if (result) {
        AuditLog log;
        log.user_id = user.id;
        log.action = "UPDATE_USER";
        log.resource = "user_management";
        log.result = "SUCCESS";
        audit_service_->log_event(log);
    }

    return result;
}

bool SecurityManager::delete_user(const std::string& user_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    bool result = auth_service_->delete_user(user_id);

    if (result) {
        AuditLog log;
        log.user_id = user_id;
        log.action = "DELETE_USER";
        log.resource = "user_management";
        log.result = "SUCCESS";
        audit_service_->log_event(log);
    }

    return result;
}

void SecurityManager::update_config(const SecurityConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);

    config_ = config;

    AuditLog log;
    log.action = "UPDATE_CONFIG";
    log.resource = "security_config";
    log.result = "SUCCESS";
    audit_service_->log_event(log);
}

SecurityConfig SecurityManager::get_config() const {
    std::lock_guard<std::mutex> lock(mutex_);

    return config_;
}

bool SecurityManager::is_healthy() const {
    std::lock_guard<std::mutex> lock(mutex_);

    return auth_service_ && authz_service_ && encryption_service_ && audit_service_;
}

std::unordered_map<std::string, std::string> SecurityManager::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::unordered_map<std::string, std::string> stats;
    stats["status"] = is_healthy() ? "healthy" : "unhealthy";
    stats["security_level"] = std::to_string(static_cast<int>(config_.level));
    stats["encryption_enabled"] = config_.enable_encryption ? "true" : "false";
    stats["authentication_required"] = config_.require_authentication ? "true" : "false";

    return stats;
}

// SecurityMiddleware implementation
SecurityMiddleware::SecurityMiddleware(SecurityManager* manager)
    : security_manager_(manager) {
}

void SecurityMiddleware::add_public_path(const std::string& path) {
    public_paths_.push_back(path);
}

void SecurityMiddleware::set_path_permissions(const std::string& path,
                                            const std::vector<PermissionType>& permissions) {
    path_permissions_[path] = permissions;
}

bool SecurityMiddleware::authenticate_request(const std::string& path,
                                            const std::string& session_token) {
    // Check if path is public
    for (const std::string& public_path : public_paths_) {
        if (path.find(public_path) == 0) {
            return true;
        }
    }

    // Validate session
    return security_manager_->validate_session(session_token);
}

bool SecurityMiddleware::authorize_request(const std::string& path,
                                         const std::string& session_token,
                                         PermissionType permission) {
    // Check if path is public
    for (const std::string& public_path : public_paths_) {
        if (path.find(public_path) == 0) {
            return true;
        }
    }

    // Check permission
    return security_manager_->check_permission(session_token, path, permission);
}

std::optional<std::string> SecurityMiddleware::extract_token_from_header(const std::string& auth_header) {
    if (auth_header.empty()) {
        return std::nullopt;
    }

    const std::string bearer_prefix = "Bearer ";
    if (auth_header.substr(0, bearer_prefix.length()) == bearer_prefix) {
        return auth_header.substr(bearer_prefix.length());
    }

    return std::nullopt;
}

std::string SecurityMiddleware::generate_csrf_token() {
    return utils::generate_random_string(32);
}

bool SecurityMiddleware::validate_csrf_token(const std::string& token) {
    return !token.empty() && token.length() >= 16;
}

} // namespace security
} // namespace langchain