//! Shell-step handler.
//!
//! Runs a shell command through [`tmg_sandbox::SandboxContext::run_command`].
//! The sandbox layer enforces the per-command timeout and OOM-score
//! adjustment; the workflow engine simply chooses *which* timeout to
//! use (per-step override or `[workflow] default_shell_timeout`).
//!
//! ## Per-step timeout caveat
//!
//! `SandboxContext::run_command` reads its timeout from the sandbox's
//! own [`tmg_sandbox::SandboxConfig::timeout_secs`]. The shell handler
//! therefore creates a *new* `SandboxContext` configured for the
//! resolved timeout when the workflow step's timeout differs from the
//! shared sandbox default. This is intentional: the alternative —
//! mutating the shared sandbox — would race other concurrent steps
//! once parallel execution lands (#40). The sandbox's filesystem
//! `activate()` work is not duplicated because we copy the
//! mode/workspace and do not call `activate()` again on the per-step
//! clone (Landlock rules already apply process-wide).

use std::time::Duration;

use serde_json::json;

use tmg_sandbox::{SandboxConfig, SandboxContext};

use crate::def::StepResult;
use crate::error::{Result, WorkflowError};
use crate::expr;

/// Execute a shell step.
pub(crate) async fn execute(
    sandbox: &SandboxContext,
    default_timeout: Duration,
    command_template: &str,
    timeout: Option<Duration>,
    ctx: &expr::ExprContext<'_>,
) -> Result<StepResult> {
    let command = expr::eval_string(command_template, ctx)?;

    let resolved_timeout = timeout.unwrap_or(default_timeout);

    // Build a per-step sandbox view with the resolved timeout. We
    // intentionally avoid mutating the shared sandbox to keep
    // concurrent step execution (a future enhancement, #40) safe.
    let cfg = SandboxConfig::new(sandbox.workspace())
        .with_mode(sandbox.mode())
        .with_timeout(resolved_timeout.as_secs());
    let local_sandbox = SandboxContext::new(cfg);

    let output = local_sandbox.run_command(&command).await?;

    let exit_code = output.exit_code;
    let value = json!({
        "stdout": output.stdout,
        "stderr": output.stderr,
        "exit_code": exit_code,
    });
    let mut result = StepResult {
        output: value,
        exit_code,
        stdout: output.stdout,
        stderr: output.stderr,
        changed_files: Vec::new(),
    };

    // The sandbox layer surfaces a non-zero exit as a successful
    // CommandOutput rather than an error — we mirror that here so
    // workflows can branch on `exit_code` via `when`.
    if exit_code < 0 {
        // Sandboxed runner uses -1 for "could not spawn"; flag this
        // explicitly as a step failure so the engine emits StepFailed.
        result
            .stderr
            .push_str("\n[tmg-workflow] sandbox reported exit_code < 0 (process did not spawn)");
        return Err(WorkflowError::StepFailed {
            step_id: "<shell>".to_owned(),
            message: format!("command did not spawn: {command}"),
        });
    }

    Ok(result)
}
