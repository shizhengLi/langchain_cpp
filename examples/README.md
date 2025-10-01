# LangChain++ Examples

This directory contains practical examples demonstrating how to use various features of the LangChain++ C++ library.

## Available Examples

### 1. Basic Retrieval (`basic_retrieval.cpp`)
- Setting up an inverted index retriever
- Adding documents and building index
- Performing basic search operations

### 2. Advanced Retrieval (`advanced_retrieval.cpp`)
- Multiple retrieval strategies (BM25, Vector, Hybrid)
- Configuration and customization
- Performance comparison

### 3. LLM Integration (`llm_integration.cpp`)
- OpenAI API integration
- Streaming responses
- Error handling and retry logic

### 4. Chain Composition (`chain_composition.cpp`)
- Creating LLM chains
- Prompt template usage
- Sequential and parallel chains

### 5. Memory Systems (`memory_systems.cpp`)
- Conversation memory
- Long-term memory management
- Memory-backed chains

### 6. Security Implementation (`security_example.cpp`)
- User authentication and authorization
- Data encryption and decryption
- Secure session management

### 7. Performance Optimization (`performance_example.cpp`)
- Concurrent processing
- Memory optimization
- Performance monitoring

### 8. Production Setup (`production_setup.cpp`)
- Monitoring and metrics
- Distributed processing
- Error handling and logging

## Building Examples

### Quick Build (from project root)

```bash
# Navigate to project root (if you're not already there)
cd https://github.com/shizhengLi/langchain_cpp

# Clean build (remove old CMake files from root and build directory)
rm -rf build CMakeCache.txt CMakeFiles cmake_install.cmake Makefile CTestTestfile.cmake

# Create build directory and configure
mkdir build && cd build
# Note: Tests are temporarily disabled (-DENABLE_TESTING=OFF) because they require additional test files
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON -DENABLE_TESTING=OFF

# Build the entire project with all examples
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Alternatively, build only examples after main library is built
make examples

# Or build specific working examples
make basic_retrieval
make production_setup
```

### Running Examples

Working examples are built to the `build/examples/` directory:

```bash
# Navigate to build directory (if you're not already there)
cd https://github.com/shizhengLi/langchain_cpp/build

# Run working examples
./examples/basic_retrieval        # Basic document retrieval and search
./examples/production_setup       # Production server setup (runs until Ctrl+C)
```

**Note:** Currently only `basic_retrieval` and `production_setup` examples are fully functional. Other examples (advanced_retrieval, llm_integration, etc.) are temporarily disabled pending API updates.

### Prerequisites

- **C++20 compatible compiler**: GCC 11+, Clang 14+, or MSVC 2022 17.6+
- **CMake 3.20+**
- **OpenAI API key** (required for LLM examples): Set as environment variable `OPENAI_API_KEY`
- **System dependencies**: See main README for complete dependency list
- **Optional dependencies**: Redis, PostgreSQL (for advanced examples)

### Environment Setup

For LLM-based examples, set up your API key:

```bash
# Export your OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-openai-api-key-here"
```

### Getting Started

1. **Start with `basic_retrieval.cpp`** - Demonstrates core document indexing and retrieval concepts
2. **Try `advanced_retrieval.cpp`** - Shows BM25, Vector, and Hybrid retrieval strategies
3. **Explore `llm_integration.cpp`** - OpenAI API integration with streaming and error handling
4. **Move to `chain_composition.cpp`** - Complex chain orchestration patterns
5. **Check `performance_example.cpp`** - Optimization techniques and monitoring
6. **Review `production_setup.cpp`** - Production-ready server implementation

### Troubleshooting

**Build Issues:**
```bash
# Clean rebuild if遇到问题
rm -rf build/
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON -DENABLE_TESTING=OFF
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
```

**Runtime Issues:**
- Ensure OpenAI API key is set for LLM examples
- Check that all dependencies are installed
- Run with debug output: `./examples/basic_retrieval --debug`

**Memory Issues:**
- Use smaller document sets for initial testing
- Adjust memory limits in your environment if needed

## Contributing

Feel free to contribute new examples! Please follow the existing code style and include comprehensive documentation.