//! Error types for the memory crate.

/// Errors that can occur when working with the memory store.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    /// Memory entry name failed validation (empty, contains slashes, etc.).
    #[error("invalid memory name {name:?}: {reason}")]
    InvalidName {
        /// The offending name.
        name: String,
        /// Why it was rejected.
        reason: String,
    },

    /// A memory entry with the requested name already exists.
    #[error("memory entry already exists: {name}")]
    AlreadyExists {
        /// The name of the conflicting entry.
        name: String,
    },

    /// No memory entry with the requested name was found.
    #[error("memory entry not found: {name}")]
    NotFound {
        /// The name that was looked up.
        name: String,
    },

    /// The frontmatter on a memory file was missing or malformed.
    #[error("invalid frontmatter at {path}: {reason}")]
    InvalidFrontmatter {
        /// File path that failed to parse.
        path: String,
        /// What was wrong.
        reason: String,
    },

    /// The `type` value on a memory entry was outside the allowed vocabulary.
    #[error(
        "invalid memory type {value:?} at {path}: must be one of user, feedback, project, reference"
    )]
    InvalidType {
        /// File path with the bad type.
        path: String,
        /// The offending value.
        value: String,
    },

    /// An I/O error occurred during memory store operations.
    #[error("{context}: {source}")]
    Io {
        /// What was being attempted.
        context: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// A YAML serialization or deserialization error.
    #[error("yaml error at {path}: {source}")]
    Yaml {
        /// File path involved.
        path: String,
        /// Underlying YAML error.
        #[source]
        source: serde_yml::Error,
    },
}

impl MemoryError {
    /// Convenience: construct an [`Self::Io`] with context.
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }

    /// Convenience: construct an [`Self::InvalidName`].
    pub fn invalid_name(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidName {
            name: name.into(),
            reason: reason.into(),
        }
    }

    /// Convenience: construct an [`Self::InvalidFrontmatter`].
    pub fn invalid_frontmatter(path: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidFrontmatter {
            path: path.into(),
            reason: reason.into(),
        }
    }
}
