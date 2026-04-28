//! Error types for the tools crate.

/// The maximum length for tool output before truncation.
pub const MAX_OUTPUT_LENGTH: usize = 128 * 1024;

/// Errors that can occur during tool execution.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// The requested tool was not found in the registry.
    #[error("tool not found: {name}")]
    NotFound {
        /// The name that was looked up.
        name: String,
    },

    /// Invalid parameters were supplied to the tool.
    #[error("invalid parameters: {message}")]
    InvalidParams {
        /// A human-readable description of what was wrong.
        message: String,
    },

    /// A path traversal attack was detected.
    #[error("path traversal rejected: {path}")]
    PathTraversal {
        /// The offending path string.
        path: String,
    },

    /// An I/O error occurred during tool execution.
    #[error("{context}: {source}")]
    Io {
        /// Description of what the tool was doing when the error occurred.
        context: String,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// A regex compilation error.
    #[error("invalid regex pattern: {source}")]
    Regex {
        /// The underlying regex error.
        #[from]
        source: regex::Error,
    },

    /// A timeout was exceeded (e.g. `shell_exec`).
    #[error("command timed out after {seconds}s")]
    Timeout {
        /// The timeout duration in seconds.
        seconds: u64,
    },

    /// JSON serialization/deserialization error.
    #[error("json error: {context}: {source}")]
    Json {
        /// Description of what was being serialized.
        context: String,
        /// The underlying `serde_json` error.
        #[source]
        source: serde_json::Error,
    },

    /// A sandbox policy denied the operation.
    ///
    /// Returned when a tool's pre-execution check against the active
    /// [`SandboxContext`](tmg_sandbox::SandboxContext) fails — e.g. a
    /// workspace-external write under `WorkspaceWrite` mode, or any
    /// write under `ReadOnly` mode.
    #[error("sandbox denied operation: {source}")]
    Sandbox {
        /// The underlying sandbox error.
        #[from]
        source: tmg_sandbox::SandboxError,
    },
}

impl ToolError {
    /// Create an I/O error with context.
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }

    /// Create an invalid-params error.
    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self::InvalidParams {
            message: message.into(),
        }
    }

    /// Create a JSON error with context.
    pub fn json(context: impl Into<String>, source: serde_json::Error) -> Self {
        Self::Json {
            context: context.into(),
            source,
        }
    }
}
