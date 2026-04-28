//! `shell_exec` tool: command execution under the sandbox context.

use std::fmt::Write;

use tmg_sandbox::{SandboxContext, SandboxError};

use crate::error::ToolError;
use crate::types::{Tool, ToolResult};

/// Execute a shell command under the active [`SandboxContext`].
///
/// All command execution flows through
/// [`SandboxContext::run_command`], which applies the configured
/// timeout, OOM-score adjustment, and (on Linux) network namespace
/// restrictions in one place.
pub struct ShellExecTool;

impl Tool for ShellExecTool {
    fn name(&self) -> &'static str {
        "shell_exec"
    }

    fn description(&self) -> &'static str {
        "Execute a shell command and return its output. The command runs through the \
         configured sandbox (timeout, OOM-score adjustment, network policy). \
         A per-call `timeout_secs` may shorten the sandbox-wide default."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute."
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds. Default and upper bound are inherited from the sandbox configuration; values larger than the sandbox-configured maximum are clamped down to that maximum (the sandbox owns the process-budget ceiling)."
                }
            },
            "required": ["command"],
            "additionalProperties": false
        })
    }

    fn execute<'a>(
        &'a self,
        params: serde_json::Value,
        ctx: &'a SandboxContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + 'a>,
    > {
        Box::pin(self.execute_inner(params, ctx))
    }
}

impl ShellExecTool {
    async fn execute_inner(
        &self,
        params: serde_json::Value,
        ctx: &SandboxContext,
    ) -> Result<ToolResult, ToolError> {
        let Some(command) = params.get("command").and_then(serde_json::Value::as_str) else {
            return Err(ToolError::invalid_params(
                "missing required parameter: command",
            ));
        };

        // Allow the LLM to shorten the sandbox-wide default timeout
        // for a single call, but never extend it (the sandbox owns
        // the process-budget contract).
        // [`SandboxContext::run_command_with_timeout`] clamps the
        // requested value down to the sandbox-configured maximum, so
        // the per-call path stays allocation-free (no
        // [`SandboxConfig`] clone) while preserving the upper bound.
        let per_call_timeout = params
            .get("timeout_secs")
            .and_then(serde_json::Value::as_u64);

        let result = match per_call_timeout {
            Some(t) => ctx.run_command_with_timeout(command, t).await,
            None => ctx.run_command(command).await,
        };

        match result {
            Ok(output) => {
                let mut text = String::new();
                let _ = writeln!(text, "Exit code: {}", output.exit_code);

                if !output.stdout.is_empty() {
                    let _ = write!(text, "\n--- stdout ---\n{}", output.stdout);
                }
                if !output.stderr.is_empty() {
                    let _ = write!(text, "\n--- stderr ---\n{}", output.stderr);
                }

                if output.success() {
                    Ok(ToolResult::success(text))
                } else {
                    Ok(ToolResult::error(text))
                }
            }
            Err(SandboxError::Timeout { seconds }) => Err(ToolError::Timeout { seconds }),
            Err(e) => Err(ToolError::Sandbox { source: e }),
        }
    }
}

#[expect(clippy::unwrap_used, reason = "test assertions")]
#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> SandboxContext {
        SandboxContext::test_default()
    }

    #[tokio::test]
    async fn exec_simple_command() {
        let tool = ShellExecTool;
        let sandbox = ctx();
        let result = tool
            .execute(serde_json::json!({ "command": "echo hello" }), &sandbox)
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.output.contains("hello"));
        assert!(result.output.contains("Exit code: 0"));
    }

    #[tokio::test]
    async fn exec_failing_command() {
        let tool = ShellExecTool;
        let sandbox = ctx();
        let result = tool
            .execute(serde_json::json!({ "command": "exit 1" }), &sandbox)
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.output.contains("Exit code: 1"));
    }

    #[tokio::test]
    async fn exec_timeout() {
        let tool = ShellExecTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({
                    "command": "sleep 60",
                    "timeout_secs": 1
                }),
                &sandbox,
            )
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn exec_stderr_output() {
        let tool = ShellExecTool;
        let sandbox = ctx();
        let result = tool
            .execute(serde_json::json!({ "command": "echo error >&2" }), &sandbox)
            .await
            .unwrap();

        assert!(!result.is_error); // exit code 0
        assert!(result.output.contains("stderr"));
        assert!(result.output.contains("error"));
    }

    #[tokio::test]
    async fn exec_missing_command_param() {
        let tool = ShellExecTool;
        let sandbox = ctx();
        let result = tool.execute(serde_json::json!({}), &sandbox).await;
        assert!(result.is_err());
    }
}
