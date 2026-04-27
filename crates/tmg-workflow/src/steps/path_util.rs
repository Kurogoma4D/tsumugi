//! Shared path-validation helper for step handlers.
//!
//! Mirrors `tmg_tools::path_util::validate_path` but is reproduced here
//! because that module is private to `tmg-tools`. Re-exporting it would
//! widen `tmg-tools`'s public API for a single helper; the duplication
//! is intentional and tested below.
//!
//! Both `write_file` and `agent` (via `inject_files`) use this helper
//! to reject any path containing a `..` traversal component before
//! handing the path to the sandbox boundary.

use std::path::{Component, Path};

use crate::error::{Result, WorkflowError};

/// Validate that `path` does not contain `..` (traversal) components.
///
/// # Errors
///
/// Returns [`WorkflowError::InvalidPath`] when any component of the
/// supplied path is [`Component::ParentDir`].
pub(crate) fn validate_path(path: &Path) -> Result<()> {
    for component in path.components() {
        if matches!(component, Component::ParentDir) {
            return Err(WorkflowError::InvalidPath {
                path: path.display().to_string(),
                reason: "contains '..' (traversal not allowed)".to_owned(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_parent_dir() {
        let result = validate_path(Path::new("../outside.txt"));
        assert!(matches!(result, Err(WorkflowError::InvalidPath { .. })));
    }

    #[test]
    fn rejects_nested_parent_dir() {
        let result = validate_path(Path::new("a/b/../../etc/passwd"));
        assert!(matches!(result, Err(WorkflowError::InvalidPath { .. })));
    }

    #[test]
    fn allows_nested_path() {
        assert!(validate_path(Path::new("src/lib.rs")).is_ok());
    }

    #[test]
    fn allows_absolute_path() {
        assert!(validate_path(Path::new("/tmp/x.txt")).is_ok());
    }
}
