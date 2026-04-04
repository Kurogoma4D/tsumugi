//! Error types for the core crate.

/// Errors produced by the core orchestration layer.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// An error originating from the LLM communication layer.
    #[error("llm error: {0}")]
    Llm(#[from] tmg_llm::LlmError),

    /// An error originating from tool execution.
    #[error("tool error: {0}")]
    Tool(#[from] tmg_tools::ToolError),

    /// An I/O error (e.g. reading prompt files).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// The agent loop was cancelled via `CancellationToken`.
    #[error("agent loop cancelled")]
    Cancelled,
}
