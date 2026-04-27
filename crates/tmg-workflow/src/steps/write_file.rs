//! `write_file` step handler.
//!
//! Resolves the path/content templates, performs traversal-rejection
//! via the shared [`super::path_util::validate_path`] helper, and
//! writes the file under the workspace.

use std::path::PathBuf;

use serde_json::json;

use tmg_sandbox::SandboxContext;

use crate::def::StepResult;
use crate::error::{Result, WorkflowError};
use crate::expr;
use crate::steps::path_util::validate_path;

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

// Path-validation tests live in `super::path_util::tests` now that the
// helper is shared with the agent step's `inject_files` handling.
