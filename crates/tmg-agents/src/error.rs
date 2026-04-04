//! Error types for the agents crate.

/// Errors produced by the subagent system.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// An error originating from the LLM communication layer.
    #[error("llm error: {0}")]
    Llm(#[from] tmg_llm::LlmError),

    /// An error originating from tool execution.
    #[error("tool error: {0}")]
    Tool(#[from] tmg_tools::ToolError),

    /// An unknown agent type was requested.
    #[error("unknown agent type: {name}")]
    UnknownAgentType {
        /// The agent type name that was not recognized.
        name: String,
    },

    /// A subagent attempted to spawn another subagent (nesting is forbidden).
    #[error("subagent nesting is not allowed: spawn_agent cannot be called from a subagent")]
    NestingForbidden,

    /// The subagent was cancelled via `CancellationToken`.
    #[error("subagent cancelled")]
    Cancelled,

    /// A task join error from `JoinSet`.
    #[error("subagent task join error: {message}")]
    JoinError {
        /// Description of the join error.
        message: String,
    },

    /// JSON serialization/deserialization error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}
