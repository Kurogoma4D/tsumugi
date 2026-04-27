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
//! `activate()` work is not duplicated because we clone the existing
//! config and do not call `activate()` again on the per-step clone
//! (Landlock rules already apply process-wide).
//!
//! The clone preserves *every* field of [`tmg_sandbox::SandboxConfig`]
//! (mode, workspace, `allowed_domains`, `oom_score_adj`, ...) and only
//! overrides `timeout_secs` with the resolved per-step value. A
//! previous version of this file rebuilt the config from scratch via
//! `SandboxConfig::new(...)` and silently dropped `allowed_domains` /
//! `oom_score_adj`; the regression test in this module guards against
//! that recurring.

use std::time::Duration;

use serde_json::json;

use tmg_sandbox::{SandboxConfig, SandboxContext};

use crate::def::StepResult;
use crate::error::{Result, WorkflowError};
use crate::expr;

/// Build the per-step sandbox config: clone the parent's full config
/// and override only `timeout_secs`.
///
/// Factored out as a free function so the regression test can verify
/// the clone preserves every knob (`allowed_domains`, `oom_score_adj`,
/// `mode`, `workspace`, ...) without actually executing a command.
fn per_step_config(parent: &SandboxConfig, resolved_timeout: Duration) -> SandboxConfig {
    let mut cfg = parent.clone();
    cfg.timeout_secs = resolved_timeout.as_secs();
    cfg
}

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
    //
    // Clone the *full* SandboxConfig so all knobs (allowed_domains,
    // oom_score_adj, mode, workspace, ...) carry over to the per-step
    // sandbox. Only `timeout_secs` is overridden with the resolved
    // per-step value.
    let cfg = per_step_config(sandbox.config(), resolved_timeout);
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

#[cfg(test)]
mod tests {
    use super::*;
    use tmg_sandbox::SandboxMode;

    /// Regression: the per-step sandbox clone must preserve every
    /// `SandboxConfig` field (`mode`, `workspace`, `allowed_domains`,
    /// `oom_score_adj`, ...) and only override `timeout_secs`.
    ///
    /// A previous version rebuilt the config from scratch via
    /// `SandboxConfig::new(...)` and dropped `allowed_domains` /
    /// `oom_score_adj`.
    #[test]
    fn per_step_config_preserves_all_fields() {
        let parent = SandboxConfig::new("/tmp/ws")
            .with_mode(SandboxMode::ReadOnly)
            .with_allowed_domains(vec!["api.example.com".to_owned()])
            .with_timeout(15);
        // OOM score is not on the public builder, set directly.
        let mut parent = parent;
        parent.oom_score_adj = 750;

        let cfg = per_step_config(&parent, Duration::from_secs(120));

        // Only `timeout_secs` differs.
        assert_eq!(cfg.timeout_secs, 120);
        // Everything else is preserved.
        assert_eq!(cfg.mode, SandboxMode::ReadOnly);
        assert_eq!(cfg.workspace, parent.workspace);
        assert_eq!(cfg.allowed_domains, vec!["api.example.com".to_owned()]);
        assert_eq!(cfg.oom_score_adj, 750);
    }
}
