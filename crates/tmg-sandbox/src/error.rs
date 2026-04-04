//! Error types for the sandbox crate.

use std::path::PathBuf;

/// Errors that can occur during sandbox setup or enforcement.
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    /// Failed to apply Landlock filesystem restrictions.
    #[error("failed to apply landlock rules: {reason}")]
    Landlock {
        /// A human-readable description of what went wrong.
        reason: String,
    },

    /// Failed to create a network namespace for isolation.
    #[error("failed to create network namespace: {source}")]
    NetworkNamespace {
        /// The underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// Failed to apply iptables rules for domain allowlisting.
    #[error("failed to apply iptables rule: {reason}")]
    Iptables {
        /// A human-readable description of the failure.
        reason: String,
    },

    /// A filesystem access was denied by the sandbox.
    #[error("access denied: {path} is outside the sandbox")]
    AccessDenied {
        /// The path that was rejected.
        path: PathBuf,
    },

    /// Failed to adjust OOM score for a child process.
    #[error("failed to adjust OOM score: {source}")]
    OomAdjust {
        /// The underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// A spawned process exceeded the configured timeout.
    #[error("process timed out after {seconds}s and was killed")]
    Timeout {
        /// The timeout duration in seconds.
        seconds: u64,
    },

    /// An I/O error occurred during sandbox operations.
    #[error("{context}: {source}")]
    Io {
        /// Description of what the sandbox was doing when the error occurred.
        context: String,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// DNS resolution failed for an allowed domain.
    #[error("DNS resolution failed for domain '{domain}': {source}")]
    DnsResolution {
        /// The domain that could not be resolved.
        domain: String,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// The sandbox feature is not supported on this platform.
    #[error("sandbox is not supported on this platform; running without restrictions")]
    Unsupported,
}

impl SandboxError {
    /// Create an I/O error with context.
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }
}
