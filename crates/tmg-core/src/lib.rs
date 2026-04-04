//! tmg-core: Core domain types, traits, and orchestration logic for tsumugi.

pub mod agent_loop;
pub mod context;
pub mod error;
pub mod message;
pub mod prompt;

pub use agent_loop::{AgentLoop, StreamSink};
pub use context::{
    ContextCompressor, ContextConfig, TokenCounter, format_context_usage, truncate_tool_result,
};
pub use error::CoreError;
pub use message::Message;

// Re-export ToolCallingMode from tmg-llm for downstream consumers.
pub use tmg_llm::ToolCallingMode;
