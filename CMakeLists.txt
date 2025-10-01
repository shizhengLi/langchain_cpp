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

# Include directories
include_directories(include)

# Find required packages
find_package(Threads REQUIRED)

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
)

# Create main library
add_library(langchain_cpp ${SOURCES} ${HEADERS})
target_link_libraries(langchain_cpp Threads::Threads)
target_include_directories(langchain_cpp PUBLIC include)

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
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
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