/// Errors produced by the LLM communication layer.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// Failed to connect to the LLM server (e.g. server not running).
    #[error("connection failed: {0}")]
    ConnectionFailed(String),

    /// The request timed out.
    #[error("request timed out")]
    Timeout,

    /// The server returned a non-success HTTP status.
    #[error("server error (HTTP {status}): {body}")]
    ServerError {
        /// HTTP status code.
        status: u16,
        /// Response body text.
        body: String,
    },

    /// Failed to parse the server response.
    #[error("invalid response: {0}")]
    InvalidResponse(String),

    /// The SSE stream produced an error event.
    #[error("stream error: {0}")]
    StreamError(String),

    /// The request was cancelled via `CancellationToken`.
    #[error("request cancelled")]
    Cancelled,

    /// An HTTP-level error from reqwest that doesn't fit other variants.
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    /// The connection pool has no endpoints configured.
    #[error("connection pool has no endpoints configured")]
    PoolEmpty,

    /// All endpoints in the pool are down (health checks failed).
    #[error("all endpoints in the pool are unreachable")]
    AllEndpointsDown,
}

impl LlmError {
    /// Returns `true` if this error indicates the server is unreachable.
    pub fn is_connection_failure(&self) -> bool {
        matches!(self, Self::ConnectionFailed(_))
    }
}
