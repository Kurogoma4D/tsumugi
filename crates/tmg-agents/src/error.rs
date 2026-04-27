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

    /// The escalator subagent was requested but disabled by operator
    /// configuration (`[harness.escalator] disable = true`, SPEC §9.10).
    /// The auto-promotion gate (issue #37) treats this as "do not
    /// escalate" rather than as a hard error.
    #[error("escalator disabled")]
    EscalatorDisabled,

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

    /// TOML parsing error for custom agent definitions.
    #[error("toml parse error in {path}: {source}")]
    TomlParse {
        /// Path to the TOML file.
        path: String,
        /// The underlying TOML deserialization error.
        #[source]
        source: toml::de::Error,
    },

    /// TOML serialization error.
    #[error("toml serialization error: {reason}")]
    TomlSerialize {
        /// Description of the serialization failure.
        reason: String,
    },

    /// A custom agent definition is invalid (missing required fields, etc.).
    #[error("invalid custom agent in {path}: {reason}")]
    InvalidCustomAgent {
        /// Path to the TOML file.
        path: String,
        /// Description of the validation issue.
        reason: String,
    },

    /// An I/O error occurred during agent discovery.
    #[error("{context}: {source}")]
    Io {
        /// Description of what was being done when the error occurred.
        context: String,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}
