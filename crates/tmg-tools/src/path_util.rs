//! Path validation utilities for preventing path traversal attacks.

use std::path::{Component, Path, PathBuf};

use tmg_sandbox::SandboxContext;

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

/// Resolve a tool-supplied path against the sandbox workspace.
///
/// If `path` is already absolute, it is returned unchanged (after the
/// `..`-component check performed by [`validate_path`]). Relative paths
/// are joined with the [`SandboxContext`]'s workspace so the result is
/// always absolute and can be fed to
/// [`SandboxContext::check_path_access`] /
/// [`SandboxContext::check_write_access`] without ambiguity.
///
/// Tools should call this in preference to [`validate_path`] alone so
/// the sandbox sees the same path the OS will see.
pub fn validate_and_resolve(
    path: impl AsRef<Path>,
    ctx: &SandboxContext,
) -> Result<PathBuf, ToolError> {
    let path = validate_path(path)?;
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(ctx.workspace().join(path))
    }
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

    #[test]
    fn validate_and_resolve_keeps_absolute_paths() {
        let ctx = SandboxContext::test_default();
        let result = validate_and_resolve("/tmp/test/file.txt", &ctx).unwrap();
        assert_eq!(result, PathBuf::from("/tmp/test/file.txt"));
    }

    #[test]
    fn validate_and_resolve_joins_relative_paths_to_workspace() {
        let ctx = SandboxContext::test_default();
        let result = validate_and_resolve("src/main.rs", &ctx).unwrap();
        assert!(result.is_absolute());
        assert!(result.ends_with("src/main.rs"));
    }

    #[test]
    fn validate_and_resolve_rejects_traversal() {
        let ctx = SandboxContext::test_default();
        let result = validate_and_resolve("../escape.txt", &ctx);
        assert!(matches!(result, Err(ToolError::PathTraversal { .. })));
    }
}
