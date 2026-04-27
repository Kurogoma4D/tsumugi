//! `write_file` step handler.
//!
//! Resolves the path/content templates, performs traversal-rejection
//! via `tmg_tools` style validation (we hand-implement the same `..`
//! check here so this crate doesn't depend on the *internal* `path_util`
//! module of `tmg-tools`), and writes the file under the workspace.

use std::path::{Component, Path, PathBuf};

use serde_json::json;

use tmg_sandbox::SandboxContext;

use crate::def::StepResult;
use crate::error::{Result, WorkflowError};
use crate::expr;

/// Validate that `path` does not contain `..` (traversal) components.
///
/// Mirrors `tmg_tools::path_util::validate_path` but is reproduced
/// here because that module is private to `tmg-tools`. Re-exporting it
/// would widen `tmg-tools`'s public API for a single helper; the
/// duplication is intentional and tested below.
fn validate_path(path: &Path) -> Result<()> {
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

/// Execute a `write_file` step.
pub(crate) async fn execute(
    sandbox: &SandboxContext,
    path_template: &str,
    content_template: &str,
    ctx: &expr::ExprContext<'_>,
) -> Result<StepResult> {
    let path_str = expr::eval_string(path_template, ctx)?;
    let content = expr::eval_string(content_template, ctx)?;

    let rel_or_abs = PathBuf::from(&path_str);
    validate_path(&rel_or_abs)?;

    // Resolve to an absolute path under the workspace when relative.
    let target = if rel_or_abs.is_absolute() {
        rel_or_abs.clone()
    } else {
        sandbox.workspace().join(&rel_or_abs)
    };

    sandbox.check_write_access(&target)?;

    if let Some(parent) = target.parent() {
        if !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                WorkflowError::io(
                    format!("creating parent directory for {}", target.display()),
                    e,
                )
            })?;
        }
    }

    tokio::fs::write(&target, &content)
        .await
        .map_err(|e| WorkflowError::io(format!("writing {}", target.display()), e))?;

    let display_path = path_str.clone();
    Ok(StepResult {
        output: json!({"path": display_path}),
        exit_code: 0,
        stdout: String::new(),
        stderr: String::new(),
        changed_files: vec![display_path],
    })
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
    fn allows_nested_path() {
        assert!(validate_path(Path::new("src/lib.rs")).is_ok());
    }

    #[test]
    fn allows_absolute_path() {
        assert!(validate_path(Path::new("/tmp/x.txt")).is_ok());
    }
}
