# Test configuration
add_executable(unit_tests)

# Test source files
target_sources(unit_tests PRIVATE
    unit_tests/test_core_types.cpp
    unit_tests/test_core_config.cpp
    unit_tests/test_memory_pool.cpp
    unit_tests/test_thread_pool.cpp
    unit_tests/test_logging.cpp
    unit_tests/test_simd_ops.cpp
    unit_tests/test_base_retriever.cpp
    unit_tests/test_text_processor.cpp
    unit_tests/test_inverted_index_retriever.cpp
    unit_tests/test_bm25_retriever.cpp
    unit_tests/test_bm25_simd_tfidf.cpp
    unit_tests/main.cpp
)

# Include directories
target_include_directories(unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CATCH2_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(unit_tests PRIVATE
    langchain_cpp
    Catch2::Catch2WithMain
    Threads::Threads
)

# Enable testing
add_test(NAME UnitTests COMMAND unit_tests)

# Set test properties
set_tests_properties(UnitTests PROPERTIES
    TIMEOUT 300
    PASS_REGULAR_EXPRESSION "All tests passed"
)