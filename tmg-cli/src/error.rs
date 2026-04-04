//! Structured error types for configuration loading.

use std::path::PathBuf;

/// Errors that can occur during configuration loading and validation.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The specified configuration file was not found.
    #[error("configuration file not found: {path}")]
    NotFound {
        /// Path that was looked up.
        path: PathBuf,
    },

    /// I/O error while reading a configuration file.
    #[error("failed to read configuration file {path}: {source}")]
    Io {
        /// Path of the file that caused the error.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// TOML parse error.
    #[error("failed to parse configuration file {path}: {source}")]
    Parse {
        /// Path of the file that caused the error.
        path: PathBuf,
        /// Underlying TOML deserialization error.
        source: toml::de::Error,
    },

    /// A configuration field has an invalid value.
    #[error("invalid value for {field}: {value:?} ({reason})")]
    InvalidValue {
        /// Dotted field path (e.g. `llm.endpoint`).
        field: String,
        /// The invalid value.
        value: String,
        /// Human-readable reason.
        reason: String,
    },
}
