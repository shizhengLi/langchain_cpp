#include "langchain/langchain.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace langchain;

int main() {
    std::cout << "=== LangChain++ Security Example ===" << std::endl;

    try {
        // 1. User Authentication
        std::cout << "\n=== User Authentication ===" << std::endl;

        security::AuthenticationManager auth_manager;
        auth_manager.set_password_requirement("minimum_length", 8);
        auth_manager.set_password_requirement("require_uppercase", true);
        auth_manager.set_password_requirement("require_lowercase", true);
        auth_manager.set_password_requirement("require_numbers", true);

        std::cout << "Configured authentication with password requirements" << std::endl;

        // Create users
        std::vector<std::pair<std::string, std::string>> users = {
            {"alice", "SecurePass123!"},
            {"bob", "MyPassword456@"},
            {"admin", "AdminPass789#"}
        };

        for (const auto& [username, password] : users) {
            auto user = auth_manager.create_user(username, password);
            if (user) {
                std::cout << "Created user: " << username << " (ID: " << user->id << ")" << std::endl;
            } else {
                std::cout << "Failed to create user: " << username << std::endl;
            }
        }

        // Test authentication
        std::cout << "\nTesting user authentication:" << std::endl;

        std::vector<std::pair<std::string, std::string>> login_attempts = {
            {"alice", "SecurePass123!"},     // Correct password
            {"bob", "WrongPassword"},        // Wrong password
            {"admin", "AdminPass789#"},      // Correct password
            {"nonexistent", "SomePass"},     // Non-existent user
            {"alice", "SecurePass123"}       // Missing special character
        };

        for (const auto& [username, password] : login_attempts) {
            auto auth_result = auth_manager.authenticate(username, password);
            std::cout << "Login '" << username << "': "
                     << (auth_result.success ? "SUCCESS" : "FAILED");

            if (!auth_result.success) {
                std::cout << " (" << auth_result.error_message << ")";
            } else {
                std::cout << " (Token: " << auth_result.session_token.substr(0, 8) << "...)";
            }
            std::cout << std::endl;
        }

        // Test password validation
        std::cout << "\nTesting password validation:" << std::endl;

        std::vector<std::string> test_passwords = {
            "short",           // Too short
            "nocapitals123!",  // No uppercase
            "NOLOWERCASE123!", // No lowercase
            "NoNumbers!",      // No numbers
            "ValidPass123!"    // Valid
        };

        for (const auto& password : test_passwords) {
            bool is_valid = auth_manager.validate_password(password);
            std::cout << "Password '" << password << "': "
                     << (is_valid ? "VALID" : "INVALID") << std::endl;
        }

        // Get authenticated user
        auto alice_auth = auth_manager.authenticate("alice", "SecurePass123!");
        security::User* current_user = nullptr;
        if (alice_auth.success) {
            current_user = auth_manager.get_user_by_token(alice_auth.session_token);
            if (current_user) {
                std::cout << "\nCurrent user: " << current_user->username
                         << " (ID: " << current_user->id << ")" << std::endl;
            }
        }

        // 2. Role-based Authorization
        std::cout << "\n=== Role-based Authorization ===" << std::endl;

        security::AuthorizationManager authz_manager;

        // Define roles and permissions
        authz_manager.create_role("admin", {
            security::Permission::READ_ALL,
            security::Permission::WRITE_ALL,
            security::Permission::DELETE_ALL,
            security::Permission::MANAGE_USERS,
            security::Permission::VIEW_METRICS
        });

        authz_manager.create_role("developer", {
            security::Permission::READ_OWN,
            security::Permission::WRITE_OWN,
            security::Permission::READ_ALL,
            security::Permission::EXECUTE_CHAINS,
            security::Permission::VIEW_METRICS
        });

        authz_manager.create_role("user", {
            security::Permission::READ_OWN,
            security::Permission::WRITE_OWN,
            security::Permission::EXECUTE_CHAINS
        });

        authz_manager.create_role("guest", {
            security::Permission::READ_OWN
        });

        std::cout << "Created roles: admin, developer, user, guest" << std::endl;

        // Assign roles to users
        if (current_user) {
            authz_manager.assign_role(current_user->id, "admin");
            std::cout << "Assigned 'admin' role to user '" << current_user->username << "'" << std::endl;
        }

        // Get other users and assign roles
        auto bob_user = auth_manager.get_user("bob");
        if (bob_user) {
            authz_manager.assign_role(bob_user->id, "developer");
            std::cout << "Assigned 'developer' role to user 'bob'" << std::endl;
        }

        // Test authorization
        std::cout << "\nTesting authorization:" << std::endl;

        std::vector<std::pair<std::string, security::Permission>> permission_tests = {
            {"alice", security::Permission::DELETE_ALL},
            {"alice", security::Permission::MANAGE_USERS},
            {"bob", security::Permission::WRITE_ALL},
            {"bob", security::Permission::VIEW_METRICS},
            {"alice", security::Permission::READ_OWN},
            {"bob", security::Permission::DELETE_ALL}  // Should fail for developer
        };

        for (const auto& [username, permission] : permission_tests) {
            auto user = auth_manager.get_user(username);
            if (user) {
                bool has_permission = authz_manager.has_permission(user->id, permission);
                std::cout << "User '" << username << "' - Permission '"
                         << security::permission_to_string(permission) << "': "
                         << (has_permission ? "GRANTED" : "DENIED") << std::endl;
            }
        }

        // Test resource-based authorization
        std::cout << "\nTesting resource-based authorization:" << std::endl;

        std::string resource_id = "chain_123";
        std::string owner_id = bob_user ? bob_user->id : "unknown";

        // Test different users accessing the resource
        std::vector<std::string> test_users = {"alice", "bob", "nonexistent"};

        for (const auto& username : test_users) {
            auto test_user = auth_manager.get_user(username);
            if (test_user) {
                bool can_read = authz_manager.can_access_resource(test_user->id, resource_id, owner_id, security::Permission::READ_OWN);
                bool can_write = authz_manager.can_access_resource(test_user->id, resource_id, owner_id, security::Permission::WRITE_OWN);
                bool can_delete = authz_manager.can_access_resource(test_user->id, resource_id, owner_id, security::Permission::DELETE_ALL);

                std::cout << "User '" << username << "' accessing resource '" << resource_id << "': "
                         << "R:" << (can_read ? "✓" : "✗") << " "
                         << "W:" << (can_write ? "✓" : "✗") << " "
                         << "D:" << (can_delete ? "✓" : "✗") << std::endl;
            }
        }

        // 3. Data Encryption and Decryption
        std::cout << "\n=== Data Encryption ===" << std::endl;

        security::EncryptionManager encryption_manager;

        // Generate encryption key
        std::string encryption_key = encryption_manager.generate_key();
        std::cout << "Generated encryption key (" << encryption_key.length() << " characters)" << std::endl;

        // Test data encryption
        std::vector<std::string> sensitive_data = {
            "OpenAI API Key: sk-1234567890abcdef",
            "Database password: SecretDBPass123!",
            "User private information: john.doe@email.com",
            "Secret conversation about confidential business strategy",
            "Personal health information and medical records"
        };

        std::cout << "\nEncrypting sensitive data:" << std::endl;

        std::vector<std::pair<std::string, std::string>> encrypted_data;

        for (const auto& data : sensitive_data) {
            auto encrypted = encryption_manager.encrypt(data, encryption_key);
            encrypted_data.push_back({data, encrypted});

            std::cout << "Original: " << data.substr(0, 30) << "..." << std::endl;
            std::cout << "Encrypted: " << encrypted.substr(0, 50) << "..." << std::endl;
            std::cout << "Size ratio: " << std::fixed << std::setprecision(2)
                     << (double)encrypted.length() / data.length() << "x" << std::endl;
            std::cout << std::endl;
        }

        // Test decryption
        std::cout << "Testing decryption:" << std::endl;

        for (const auto& [original, encrypted] : encrypted_data) {
            auto decrypted = encryption_manager.decrypt(encrypted, encryption_key);
            bool success = (decrypted == original);

            std::cout << "Decryption: " << (success ? "SUCCESS" : "FAILED") << std::endl;
            if (success) {
                std::cout << "Verified: " << decrypted.substr(0, 30) << "..." << std::endl;
            }
            std::cout << std::endl;
        }

        // Test with wrong key
        std::cout << "Testing decryption with wrong key:" << std::endl;
        std::string wrong_key = encryption_manager.generate_key();
        auto failed_decryption = encryption_manager.decrypt(encrypted_data[0].second, wrong_key);
        bool decryption_failed = failed_decryption.empty();
        std::cout << "Decryption with wrong key: " << (decryption_failed ? "PROPERLY FAILED" : "UNEXPECTEDLY SUCCEEDED") << std::endl;

        // 4. Secure Session Management
        std::cout << "\n=== Secure Session Management ===" << std::endl;

        security::SessionManager session_manager;
        session_manager.set_session_timeout(std::chrono::minutes(30));
        session_manager.set_max_sessions_per_user(3);

        std::cout << "Configured session manager (30 min timeout, max 3 sessions per user)" << std::endl;

        // Create sessions for different users
        std::vector<std::string> session_tokens;

        if (current_user) {
            for (int i = 0; i < 2; ++i) {
                auto session = session_manager.create_session(current_user->id, "web_browser");
                if (session) {
                    session_tokens.push_back(session->token);
                    std::cout << "Created session for '" << current_user->username << "': "
                             << session->token.substr(0, 12) << "..." << std::endl;
                }
            }
        }

        if (bob_user) {
            auto session = session_manager.create_session(bob_user->id, "mobile_app");
            if (session) {
                session_tokens.push_back(session->token);
                std::cout << "Created session for 'bob': " << session->token.substr(0, 12) << "..." << std::endl;
            }
        }

        // Test session validation
        std::cout << "\nTesting session validation:" << std::endl;

        for (const auto& token : session_tokens) {
            bool is_valid = session_manager.is_session_valid(token);
            auto session = session_manager.get_session(token);

            std::cout << "Session " << token.substr(0, 12) << "...: "
                     << (is_valid ? "VALID" : "INVALID");

            if (session && is_valid) {
                std::cout << " (User: " << session->user_id
                         << ", Last seen: " << std::chrono::duration_cast<std::chrono::minutes>(
                             std::chrono::system_clock::now() - session->last_activity).count()
                         << " min ago)";
            }
            std::cout << std::endl;
        }

        // Test session refresh
        std::cout << "\nTesting session refresh:" << std::endl;
        if (!session_tokens.empty()) {
            auto refreshed_token = session_manager.refresh_session(session_tokens[0]);
            if (!refreshed_token.empty()) {
                std::cout << "Session refreshed: " << refreshed_token.substr(0, 12) << "..." << std::endl;

                // Old token should be invalid
                bool old_token_valid = session_manager.is_session_valid(session_tokens[0]);
                std::cout << "Old token validity: " << (old_token_valid ? "STILL VALID" : "INVALIDATED") << std::endl;
            }
        }

        // Test session cleanup
        std::cout << "\nTesting session cleanup:" << std::endl;
        size_t sessions_before = session_manager.active_session_count();
        session_manager.cleanup_expired_sessions();
        size_t sessions_after = session_manager.active_session_count();
        std::cout << "Active sessions before cleanup: " << sessions_before << std::endl;
        std::cout << "Active sessions after cleanup: " << sessions_after << std::endl;

        // 5. Input Validation and Sanitization
        std::cout << "\n=== Input Validation ===" << std::endl;

        security::InputValidator input_validator;

        // Test various types of input validation
        std::vector<std::pair<std::string, std::string>> validation_tests = {
            {"email", "user@example.com"},
            {"email", "invalid-email"},
            {"username", "alice123"},
            {"username", "user with spaces"},
            {"api_key", "sk-1234567890abcdef"},
            {"api_key", "short"},
            {"text", "Normal text content"},
            {"text", "Text with <script>alert('xss')</script> tags"},
            {"query", "SELECT * FROM users WHERE id = 1"},
            {"query", "'; DROP TABLE users; --"}
        };

        for (const auto& [type, input] : validation_tests) {
            security::ValidationResult result = input_validator.validate(input, type);
            std::cout << "Validating " << type << " '" << input.substr(0, 20) << "': "
                     << (result.is_valid ? "VALID" : "INVALID");

            if (!result.is_valid && !result.error_message.empty()) {
                std::cout << " (" << result.error_message << ")";
            }
            std::cout << std::endl;
        }

        // Test input sanitization
        std::cout << "\nTesting input sanitization:" << std::endl;

        std::vector<std::string> unsafe_inputs = {
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "javascript:void(0)",
            "<img src=x onerror=alert('xss')>",
            "../../etc/passwd"
        };

        for (const auto& unsafe : unsafe_inputs) {
            std::string sanitized = input_validator.sanitize(unsafe);
            std::cout << "Unsafe:  " << unsafe << std::endl;
            std::cout << "Safe:    " << sanitized << std::endl;
            std::cout << "Changed: " << (unsafe != sanitized ? "YES" : "NO") << std::endl << std::endl;
        }

        // 6. Audit Logging
        std::cout << "=== Audit Logging ===" << std::endl;

        security::AuditLogger audit_logger;
        audit_logger.set_log_file("/tmp/langchain_audit.log");
        audit_logger.enable_file_logging(true);
        audit_logger.enable_console_logging(false);

        std::cout << "Configured audit logger" << std::endl;

        // Log various security events
        if (current_user) {
            audit_logger.log_authentication_event(current_user->id, true, "web_login");
            audit_logger.log_authorization_event(current_user->id, "chain_123", security::Permission::READ_ALL, true);
            audit_logger.log_data_access_event(current_user->id, "document_456", "read", true);
            audit_logger.log_security_event("admin_login", current_user->id, "Administrator logged in", security::LogLevel::INFO);
            audit_logger.log_security_event("failed_permission", bob_user ? bob_user->id : "unknown", "User attempted unauthorized access", security::LogLevel::WARNING);
        }

        // Log encryption operations
        audit_logger.log_encryption_event("data_encrypted", "Sensitive user data encrypted", true);
        audit_logger.log_encryption_event("data_decrypted", "Authorized data access", true);

        std::cout << "Logged " << audit_logger.event_count() << " security events" << std::endl;

        // Get recent security events
        auto recent_events = audit_logger.get_recent_events(5);
        std::cout << "\nRecent security events:" << std::endl;
        for (const auto& event : recent_events) {
            std::cout << "  [" << security::log_level_to_string(event.level) << "] "
                     << event.event_type << " (User: " << event.user_id << ")" << std::endl;
        }

        // 7. Security Metrics and Monitoring
        std::cout << "\n=== Security Metrics ===" << std::endl;

        security::SecurityMetrics metrics;

        // Collect security statistics
        metrics.record_authentication_attempt(true);
        metrics.record_authentication_attempt(false);
        metrics.record_authentication_attempt(false);
        metrics.record_authorization_check(true);
        metrics.record_authorization_check(false);
        metrics.record_security_event("suspicious_activity");
        metrics.record_encryption_operation("encrypt");
        metrics.record_encryption_operation("decrypt");

        // Display metrics
        std::cout << "Security Statistics:" << std::endl;
        std::cout << "  Authentication attempts: " << metrics.get_authentication_attempts() << std::endl;
        std::cout << "  Success rate: " << std::fixed << std::setprecision(1)
                 << metrics.get_authentication_success_rate() * 100 << "%" << std::endl;
        std::cout << "  Authorization checks: " << metrics.get_authorization_checks() << std::endl;
        std::cout << "  Security events: " << metrics.get_security_events() << std::endl;
        std::cout << "  Encryption operations: " << metrics.get_encryption_operations() << std::endl;

        // Get security recommendations
        auto recommendations = metrics.get_security_recommendations();
        if (!recommendations.empty()) {
            std::cout << "\nSecurity Recommendations:" << std::endl;
            for (const auto& rec : recommendations) {
                std::cout << "  - " << rec << std::endl;
            }
        }

        // 8. Integration with LLM Chains
        std::cout << "\n=== Security Integration with LLM Chains ===" << std::endl;

        // Create secure LLM chain
        const char* api_key = std::getenv("OPENAI_API_KEY");
        if (!api_key) {
            api_key = "mock-api-key";
        }

        auto llm = std::make_shared<llm::OpenAILLM>(api_key);
        llm->configure(config);

        auto secure_template = std::make_shared<prompts::PromptTemplate>(
            "You are a secure AI assistant. Current user: {user} (Role: {role}).\n"
            "Provide helpful responses while maintaining security protocols.\n\n"
            "User: {query}\nAssistant:",
            {{"user", "role", "query"}}
        );

        chains::LLMChain secure_chain(llm, secure_template);

        // Test secure chain with authorization
        if (current_user && authz_manager.has_permission(current_user->id, security::Permission::EXECUTE_CHAINS)) {
            std::string user_role = authz_manager.get_user_role(current_user->id);

            chains::ChainInput secure_input = {
                {"user", current_user->username},
                {"role", user_role},
                {"query": "What are the best security practices for AI systems?"}
            };

            std::cout << "Executing secure chain for user '" << current_user->username << "' with role '" << user_role << "'" << std::endl;

            auto secure_output = secure_chain.run(secure_input);
            std::string response = std::any_cast<std::string>(secure_output["text"]);

            std::cout << "Secure AI Response: " << response.substr(0, 200) << "..." << std::endl;

            // Log the chain execution
            audit_logger.log_data_access_event(current_user->id, "secure_chain", "execute", true);
        }

        // Cleanup
        std::cout << "\n=== Cleanup ===" << std::endl;

        // Cleanup sessions
        for (const auto& token : session_tokens) {
            session_manager.invalidate_session(token);
        }
        std::cout << "Invalidated all test sessions" << std::endl;

        // Save audit log
        bool audit_saved = audit_logger.save_logs();
        std::cout << "Audit log " << (audit_saved ? "saved successfully" : "save failed") << std::endl;

        std::cout << "\n=== Security Example completed successfully! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}