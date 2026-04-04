//! Error types for the skills crate.

/// Errors that can occur during skill discovery, parsing, or loading.
#[derive(Debug, thiserror::Error)]
pub enum SkillError {
    /// An I/O error occurred while reading a skill file.
    #[error("{context}: {source}")]
    Io {
        /// Description of what was being done when the error occurred.
        context: String,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// The SKILL.md frontmatter is missing or malformed.
    #[error("invalid frontmatter in {path}: {reason}")]
    InvalidFrontmatter {
        /// Path to the skill file.
        path: String,
        /// Description of the parsing issue.
        reason: String,
    },

    /// YAML deserialization failed.
    #[error("yaml parse error in {path}: {source}")]
    YamlParse {
        /// Path to the skill file.
        path: String,
        /// The underlying YAML error.
        #[source]
        source: serde_yml::Error,
    },

    /// The requested skill was not found.
    #[error("skill not found: {name}")]
    NotFound {
        /// The skill name that was looked up.
        name: String,
    },
}

impl SkillError {
    /// Create an I/O error with context.
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }

    /// Create an invalid frontmatter error.
    pub fn invalid_frontmatter(path: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidFrontmatter {
            path: path.into(),
            reason: reason.into(),
        }
    }
}
