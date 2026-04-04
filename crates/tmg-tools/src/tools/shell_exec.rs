//! `shell_exec` tool: command execution with timeout support.

use std::fmt::Write;
use std::time::Duration;

use crate::error::ToolError;
use crate::types::{Tool, ToolResult};

/// Default timeout in seconds for shell commands.
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Execute a shell command with timeout support.
pub struct ShellExecTool;

impl Tool for ShellExecTool {
    fn name(&self) -> &'static str {
        "shell_exec"
    }

    fn description(&self) -> &'static str {
        "Execute a shell command and return its output. The command is run via `sh -c`. \
         A timeout can be specified (default: 30 seconds)."
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
                    "description": "Maximum execution time in seconds (default: 30)."
                }
            },
            "required": ["command"],
            "additionalProperties": false
        })
    }

    fn execute(
        &self,
        params: serde_json::Value,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + '_>,
    > {
        Box::pin(self.execute_inner(params))
    }
}

impl ShellExecTool {
    async fn execute_inner(&self, params: serde_json::Value) -> Result<ToolResult, ToolError> {
        let Some(command) = params.get("command").and_then(serde_json::Value::as_str) else {
            return Err(ToolError::invalid_params(
                "missing required parameter: command",
            ));
        };

        let timeout_secs = params
            .get("timeout_secs")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(DEFAULT_TIMEOUT_SECS);

        let timeout = Duration::from_secs(timeout_secs);

        let child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| ToolError::io("spawning shell command", e))?;

        let wait_result = tokio::time::timeout(timeout, child.wait_with_output()).await;

        match wait_result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
                let exit_code = output.status.code().unwrap_or(-1);

                let mut text = String::new();
                let _ = writeln!(text, "Exit code: {exit_code}");

                if !stdout.is_empty() {
                    let _ = write!(text, "\n--- stdout ---\n{stdout}");
                }
                if !stderr.is_empty() {
                    let _ = write!(text, "\n--- stderr ---\n{stderr}");
                }

                if output.status.success() {
                    Ok(ToolResult::success(text))
                } else {
                    Ok(ToolResult::error(text))
                }
            }
            Ok(Err(e)) => Err(ToolError::io("waiting for command", e)),
            Err(_) => {
                // Timeout exceeded. The child process was consumed by
                // `wait_with_output`, which handles cleanup internally.
                Err(ToolError::Timeout {
                    seconds: timeout_secs,
                })
            }
        }
    }
}

#[expect(clippy::unwrap_used, reason = "test assertions")]
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn exec_simple_command() {
        let tool = ShellExecTool;
        let result = tool
            .execute(serde_json::json!({ "command": "echo hello" }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.output.contains("hello"));
        assert!(result.output.contains("Exit code: 0"));
    }

    #[tokio::test]
    async fn exec_failing_command() {
        let tool = ShellExecTool;
        let result = tool
            .execute(serde_json::json!({ "command": "exit 1" }))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.output.contains("Exit code: 1"));
    }

    #[tokio::test]
    async fn exec_timeout() {
        let tool = ShellExecTool;
        let result = tool
            .execute(serde_json::json!({
                "command": "sleep 60",
                "timeout_secs": 1
            }))
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn exec_stderr_output() {
        let tool = ShellExecTool;
        let result = tool
            .execute(serde_json::json!({ "command": "echo error >&2" }))
            .await
            .unwrap();

        assert!(!result.is_error); // exit code 0
        assert!(result.output.contains("stderr"));
        assert!(result.output.contains("error"));
    }

    #[tokio::test]
    async fn exec_missing_command_param() {
        let tool = ShellExecTool;
        let result = tool.execute(serde_json::json!({})).await;
        assert!(result.is_err());
    }
}
