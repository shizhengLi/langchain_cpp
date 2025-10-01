cmake_minimum_required(VERSION 3.20)

# Examples for LangChain++ implementation
add_executable(basic_example basic_example.cpp)
target_link_libraries(basic_example langchain_cpp)