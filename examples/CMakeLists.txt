cmake_minimum_required(VERSION 3.20)

# Examples for LangChain++ implementation

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# 1. Basic Retrieval Example
add_executable(basic_retrieval basic_retrieval.cpp)
target_link_libraries(basic_retrieval langchain_cpp)
# C++20 is already set by parent project

# 2. Production Setup Example (currently working)
add_executable(production_setup production_setup.cpp)
target_link_libraries(production_setup langchain_cpp)

# Temporarily commented out examples that need updates
# add_executable(advanced_retrieval advanced_retrieval.cpp)
# target_link_libraries(advanced_retrieval langchain_cpp)

# add_executable(llm_integration llm_integration.cpp)
# target_link_libraries(llm_integration langchain_cpp)

# add_executable(chain_composition chain_composition.cpp)
# target_link_libraries(chain_composition langchain_cpp)

# add_executable(memory_systems memory_systems.cpp)
# target_link_libraries(memory_systems langchain_cpp)

# add_executable(security_example security_example.cpp)
# target_link_libraries(security_example langchain_cpp)

# add_executable(performance_example performance_example.cpp)
# target_link_libraries(performance_example langchain_cpp)

# Set output directory
set_target_properties(
    basic_retrieval production_setup
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
)

# Create a custom target for building all examples
add_custom_target(examples
    DEPENDS
        basic_retrieval production_setup
    COMMENT "Building all examples"
)