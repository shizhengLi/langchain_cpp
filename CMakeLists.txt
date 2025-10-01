cmake_minimum_required(VERSION 3.20)
project(langchain_cpp VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()


# Include directories
include_directories(include)

# Find required packages
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)

# Third-party libraries (header-only)
add_subdirectory(third_party)

# Source files
set(SOURCES
    src/core/types.cpp
    src/core/config.cpp
    src/utils/memory_pool.cpp
    src/utils/thread_pool.cpp
    src/utils/logging.cpp
    src/utils/simd_ops.cpp
    src/text/text_processor.cpp
    src/retrievers/inverted_index_retriever.cpp
    src/retrievers/bm25_retriever.cpp
    src/vectorstores/simple_vector_store.cpp
  src/retrievers/hybrid_retriever.cpp
  src/llm/base_llm.cpp
  src/llm/openai_llm.cpp
  src/chains/llm_chain.cpp
  src/chains/sequential_chain.cpp
  src/prompts/prompt_template.cpp
  src/memory/memory.cpp
  src/metrics/metrics.cpp
  src/distributed/distributed_processing.cpp
  src/persistence/persistence.cpp
  src/security/security.cpp
)

# Header files
set(HEADERS
    include/langchain/langchain.hpp
    include/langchain/core/base.hpp
    include/langchain/core/types.hpp
    include/langchain/core/config.hpp
    include/langchain/utils/memory_pool.hpp
    include/langchain/utils/thread_pool.hpp
    include/langchain/utils/logging.hpp
    include/langchain/utils/simd_ops.hpp
    include/langchain/retrievers/base_retriever.hpp
    include/langchain/retrievers/inverted_index_retriever.hpp
    include/langchain/retrievers/bm25_retriever.hpp
    include/langchain/text/text_processor.hpp
    include/langchain/vectorstores/simple_vector_store.hpp
  include/langchain/retrievers/hybrid_retriever.hpp
  include/langchain/llm/base_llm.hpp
  include/langchain/llm/openai_llm.hpp
  include/langchain/chains/base_chain.hpp
  include/langchain/chains/llm_chain.hpp
  include/langchain/chains/sequential_chain.hpp
  include/langchain/prompts/prompt_template.hpp
  include/langchain/memory/memory.hpp
  include/langchain/metrics/metrics.hpp
  include/langchain/distributed/distributed_processing.hpp
  include/langchain/persistence/persistence.hpp
  include/langchain/security/security.hpp
)

# Create main library
add_library(langchain_cpp ${SOURCES} ${HEADERS})
target_link_libraries(langchain_cpp Threads::Threads CURL::libcurl OpenSSL::SSL OpenSSL::Crypto)
target_include_directories(langchain_cpp PUBLIC include)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(langchain_cpp PRIVATE
        -O3
        -march=native
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -flto
    )
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(langchain_cpp PRIVATE
        /O2
        /permissive-
        /W4
        /WX
    )
endif()

# Enable testing
option(ENABLE_TESTING "Enable tests" ON)
if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS langchain_cpp
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/langchain
    DESTINATION include
)

# Examples
option(BUILD_EXAMPLES "Build examples" OFF)
if(BUILD_EXAMPLES AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt)
    add_subdirectory(examples)
endif()

# Documentation
option(BUILD_DOCS "Build documentation" OFF)
if(BUILD_DOCS)
    add_subdirectory(docs)
endif()

# Benchmarks
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()