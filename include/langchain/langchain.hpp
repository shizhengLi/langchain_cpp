#pragma once

/**
 * @file langchain.hpp
 * @brief Main header for LangChain++ library
 *
 * This is the primary include file for the LangChain++ library.
 * Include this header to access all LangChain++ functionality.
 */

// Core components
#include "core/base.hpp"
#include "core/types.hpp"
#include "core/config.hpp"

// Utility components
#include "utils/memory_pool.hpp"
#include "utils/thread_pool.hpp"
#include "utils/logging.hpp"
#include "utils/simd_ops.hpp"

// Forward declarations for main components
namespace langchain {

// Retrieval system
template<typename Config> class DocumentRetriever;
template<typename Config> class VectorRetriever;
template<typename Config> class EnsembleRetriever;

// LLM interfaces
class BaseLLM;
class OpenAILLM;
class MockLLM;

// Memory system
class ConversationBufferMemory;
class SummaryMemory;

// Chain system
template<typename Config> class LLMChain;
template<typename Config> class RetrievalChain;

// Agent system
class ConversationalAgent;

// Tool system
class BaseTool;

} // namespace langchain

/**
 * @namespace langchain
 * @brief Main namespace for LangChain++ library
 *
 * The langchain namespace contains all components and utilities
 * for building LLM applications with high performance.
 */
namespace langchain {}