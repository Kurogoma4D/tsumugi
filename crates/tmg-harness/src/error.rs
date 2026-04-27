//! Error types for the harness layer.
//!
//! Errors are surfaced from [`RunStore`](crate::store::RunStore) and
//! [`RunRunner`](crate::runner::RunRunner) operations. Each variant
//! captures enough context (path, run id) to be actionable when
//! propagated to the CLI.

use std::path::PathBuf;

/// Errors that can occur when manipulating runs and sessions.
#[derive(Debug, thiserror::Error)]
pub enum HarnessError {
    /// I/O error while accessing a run directory or `run.toml`.
    #[error("I/O error at {path}: {source}")]
    Io {
        /// Path being accessed when the error occurred.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// Failed to serialize a [`Run`](crate::run::Run) to TOML.
    #[error("failed to serialize run.toml at {path}: {source}")]
    Serialize {
        /// Target file path.
        path: PathBuf,
        /// Underlying serialization error.
        source: toml::ser::Error,
    },

    /// Failed to deserialize `run.toml` from disk.
    #[error("failed to parse run.toml at {path}: {source}")]
    Deserialize {
        /// Source file path.
        path: PathBuf,
        /// Underlying deserialization error.
        source: toml::de::Error,
    },

    /// A requested run could not be found.
    #[error("run not found: {run_id}")]
    RunNotFound {
        /// The run id that was looked up.
        run_id: String,
    },

    /// The run id stored in `run.toml` did not match the directory it
    /// was loaded from. This indicates a corrupted or hand-edited run
    /// directory; we refuse to load rather than silently use the wrong
    /// id.
    #[error("run id mismatch: expected {expected}, found {found} in run.toml")]
    IdMismatch {
        /// Run id derived from the directory name (what the caller asked for).
        expected: String,
        /// Run id actually present inside the loaded `run.toml`.
        found: String,
    },

    /// Failed to serialize a [`Session`](crate::session::Session) to JSON.
    #[error("failed to serialize session at {path}: {source}")]
    SessionSerialize {
        /// Target file path.
        path: PathBuf,
        /// Underlying serialization error.
        source: serde_json::Error,
    },

    /// Failed to deserialize a `session_NNN.json` from disk.
    #[error("failed to parse session log at {path}: {source}")]
    SessionDeserialize {
        /// Source file path.
        path: PathBuf,
        /// Underlying deserialization error.
        source: serde_json::Error,
    },

    /// `RunRunner::end_session` was called with a [`SessionHandle`] that
    /// does not match the currently-active [`Session`]. This typically
    /// indicates a bug in the caller (e.g. an out-of-order
    /// `begin_session` / `end_session` pairing); we refuse to persist
    /// rather than silently overwriting the active session with the
    /// wrong index.
    ///
    /// [`SessionHandle`]: crate::session::SessionHandle
    /// [`Session`]: crate::session::Session
    #[error(
        "session handle mismatch: end_session called with index {actual} but active session is index {expected}"
    )]
    SessionMismatch {
        /// Index of the session that is actually active.
        expected: u32,
        /// Index carried by the handle the caller passed in.
        actual: u32,
    },
}

impl HarnessError {
    /// Construct an [`HarnessError::Io`] from a path and `std::io::Error`.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// Construct a [`HarnessError::SessionSerialize`].
    pub fn session_serialize(path: impl Into<PathBuf>, source: serde_json::Error) -> Self {
        Self::SessionSerialize {
            path: path.into(),
            source,
        }
    }

    /// Construct a [`HarnessError::SessionDeserialize`].
    pub fn session_deserialize(path: impl Into<PathBuf>, source: serde_json::Error) -> Self {
        Self::SessionDeserialize {
            path: path.into(),
            source,
        }
    }
}
