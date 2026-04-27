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
}

impl HarnessError {
    /// Construct an [`HarnessError::Io`] from a path and `std::io::Error`.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
