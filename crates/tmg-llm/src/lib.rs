//! tmg-llm: LLM communication layer (OpenAI-compatible API, SSE streaming).

pub mod client;
pub mod error;
pub mod types;

pub use client::{ChatStream, LlmClient, LlmClientConfig};
pub use error::LlmError;
pub use types::{
    ChatMessage, ChatRequest, ChatResponse, FunctionCall, FunctionDefinition, Role, StreamEvent,
    ToolCall, ToolCallAccumulator, ToolCallIndexError, ToolDefinition, ToolKind,
};
