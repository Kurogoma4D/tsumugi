//! tmg-llm: LLM communication layer (OpenAI-compatible API, SSE streaming).

pub mod client;
pub mod error;
pub mod pool;
pub mod pool_config;
pub mod prompt_tool_calling;
pub mod tokenize;
pub mod types;

pub use client::{ChatStream, LlmClient, LlmClientConfig};
pub use error::LlmError;
pub use pool::{EndpointHealth, LlmPool};
pub use pool_config::{LoadBalanceStrategy, PoolConfig, PoolConfigError, ValidationReport};
pub use prompt_tool_calling::{
    ParseError, ToolCallingMode, build_tool_calling_prompt, format_compressed_tool_defs,
    parse_tool_calls,
};
pub use tokenize::{
    TokenizeFailureHook, count_tokens_or_estimate, estimate_tokens_heuristic,
    set_tokenize_failure_hook,
};
pub use types::{
    ChatMessage, ChatRequest, ChatResponse, FunctionCall, FunctionDefinition, Role, StreamEvent,
    TokenizeResponse, ToolCall, ToolCallAccumulator, ToolCallIndexError, ToolDefinition, ToolKind,
};
