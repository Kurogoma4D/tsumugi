//! Error type for the tmg-search crate.

use std::path::PathBuf;

/// Errors produced by the search index and ingest paths.
///
/// The crate uses `thiserror` 2.x — one variant per failure shape so
/// callers can distinguish "DB couldn't be opened" from "the schema
/// version is incompatible" without string matching.
#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    /// Generic I/O failure with the path that was being touched.
    #[error("io error at {path}: {source}")]
    Io {
        /// The file or directory that was being accessed.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// `SQLite` returned an error.
    #[error("sqlite error: {context}: {source}")]
    Sqlite {
        /// What the search code was doing when sqlite failed (e.g.
        /// `"opening state.db"`).
        context: String,
        /// The underlying rusqlite error.
        #[source]
        source: rusqlite::Error,
    },

    /// JSON serialization or deserialization failed.
    #[error("json error: {context}: {source}")]
    Json {
        /// What the search code was doing.
        context: String,
        /// Underlying `serde_json` error.
        #[source]
        source: serde_json::Error,
    },

    /// A search query parameter was invalid (e.g. `limit < 1`).
    #[error("invalid query parameter: {message}")]
    InvalidQuery {
        /// Human-readable description.
        message: String,
    },
}

impl SearchError {
    /// Build an [`SearchError::Io`] from a path + source pair.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// Build an [`SearchError::Sqlite`] with a context string.
    pub fn sqlite(context: impl Into<String>, source: rusqlite::Error) -> Self {
        Self::Sqlite {
            context: context.into(),
            source,
        }
    }

    /// Build an [`SearchError::Json`] with a context string.
    pub fn json(context: impl Into<String>, source: serde_json::Error) -> Self {
        Self::Json {
            context: context.into(),
            source,
        }
    }

    /// Build an [`SearchError::InvalidQuery`].
    pub fn invalid_query(message: impl Into<String>) -> Self {
        Self::InvalidQuery {
            message: message.into(),
        }
    }
}
