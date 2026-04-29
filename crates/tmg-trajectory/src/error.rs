//! Error types for the trajectory crate.
//!
//! The crate's surface follows the workspace convention: a single
//! [`TrajectoryError`] enum covers every failure mode (I/O, JSON
//! serialisation, regex compilation in user-supplied extra patterns,
//! and archive bundling). Construction sites add `#[from]`-driven
//! conversions so most call sites only need a `?`.

use std::path::PathBuf;

use thiserror::Error;

/// Errors emitted by the trajectory recorder, exporter, and bundle
/// command paths.
#[derive(Debug, Error)]
pub enum TrajectoryError {
    /// I/O failure (open, write, mkdir, rename).
    ///
    /// The wrapped path identifies the file or directory the operation
    /// was targeting; the inner [`std::io::Error`] preserves the
    /// platform error code.
    #[error("I/O error at {path}: {source}")]
    Io {
        /// File or directory the operation targeted.
        path: PathBuf,
        /// The underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// JSON serialisation / deserialisation failure.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// User-supplied `redact_extra_patterns` did not compile.
    #[error("invalid extra redaction pattern {pattern:?}: {source}")]
    InvalidRedactionPattern {
        /// The bad pattern, surfaced verbatim so operators can locate
        /// it in their config.
        pattern: String,
        /// Underlying regex compile error.
        #[source]
        source: regex::Error,
    },

    /// Archive (tar / zstd) failure during `tmg trajectory bundle`.
    #[error("bundle error at {path}: {message}")]
    Bundle {
        /// Path the archive op was operating on (the output archive,
        /// or a member it was trying to read).
        path: PathBuf,
        /// Free-form context message.
        message: String,
    },

    /// Configuration value rejected at validation time.
    #[error("invalid trajectory config: {0}")]
    InvalidConfig(String),
}

impl TrajectoryError {
    /// Build an [`TrajectoryError::Io`] from a path and an
    /// [`std::io::Error`]. Mirrors the helper pattern used in
    /// [`tmg_harness::HarnessError`].
    #[must_use]
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// Build an [`TrajectoryError::Bundle`] error.
    #[must_use]
    pub fn bundle(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        Self::Bundle {
            path: path.into(),
            message: message.into(),
        }
    }
}
