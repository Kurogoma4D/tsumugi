//! Path validation utilities for preventing path traversal attacks.

use std::path::{Component, Path, PathBuf};

use crate::error::ToolError;

/// Validate that `path` does not contain path traversal components (e.g. `..`).
///
/// Returns the path as-is on success. This function rejects paths containing
/// `..` components (which could be used to escape a sandboxed directory) but
/// does **not** canonicalize or resolve symlinks. Symlink resolution is
/// delegated to the sandbox layer.
pub fn validate_path(path: impl AsRef<Path>) -> Result<PathBuf, ToolError> {
    let path = path.as_ref();

    // Reject any `..` components.
    for component in path.components() {
        if let Component::ParentDir = component {
            return Err(ToolError::PathTraversal {
                path: path.display().to_string(),
            });
        }
    }

    Ok(path.to_path_buf())
}

#[expect(clippy::unwrap_used, reason = "test assertions")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_absolute_path() {
        let result = validate_path("/tmp/test/file.txt");
        assert!(result.is_ok());
    }

    #[test]
    fn valid_relative_path() {
        let result = validate_path("src/main.rs");
        assert!(result.is_ok());
    }

    #[test]
    fn rejects_parent_dir() {
        let result = validate_path("/tmp/../etc/passwd");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::PathTraversal { .. }));
    }

    #[test]
    fn rejects_relative_parent_dir() {
        let result = validate_path("../secret");
        assert!(result.is_err());
    }

    #[test]
    fn allows_current_dir_component() {
        let result = validate_path("./src/main.rs");
        assert!(result.is_ok());
    }
}
