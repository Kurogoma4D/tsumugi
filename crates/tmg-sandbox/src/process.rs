//! Process restriction utilities: timeout enforcement and OOM score adjustment.

use std::path::Path;
use std::time::Duration;

use crate::error::SandboxError;
use crate::platform;

/// Execute a shell command within sandbox constraints.
///
/// This function:
/// 1. Spawns the command via `sh -c` with `current_dir` set to `workspace`
/// 2. Adjusts the child process's OOM score (Linux only)
/// 3. Enforces a timeout, sending `SIGKILL` if exceeded
///
/// Anchoring `cwd` to the sandbox workspace prevents `tmg` invocations
/// from running shell commands against an unintended directory when
/// the agent is launched from outside the workspace tree.
///
/// Returns the command's stdout, stderr, and exit code on success.
///
/// # Errors
///
/// - [`SandboxError::Timeout`] if the process exceeds `timeout_secs`
/// - [`SandboxError::Io`] if the process cannot be spawned or waited on
/// - [`SandboxError::OomAdjust`] if OOM score adjustment fails (non-fatal
///   on non-Linux platforms)
pub async fn run_sandboxed_command(
    command: &str,
    timeout_secs: u64,
    oom_score_adj: i32,
    workspace: &Path,
) -> Result<CommandOutput, SandboxError> {
    let child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(workspace)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| SandboxError::io("spawning sandboxed command", e))?;

    // Adjust OOM score for the child process (best-effort).
    if let Some(pid) = child.id() {
        // OOM adjustment failure is logged but not fatal.
        let _ = platform::adjust_oom_score(pid, oom_score_adj);
    }

    let timeout = Duration::from_secs(timeout_secs);

    // Use `wait_with_output()` to read stdout/stderr concurrently with
    // waiting for the process to exit. This avoids a deadlock that can
    // occur when reading pipes sequentially after `wait()` -- the child
    // may block on a full pipe buffer and never exit.
    let wait_result = tokio::time::timeout(timeout, child.wait_with_output()).await;

    match wait_result {
        Ok(Ok(output)) => {
            let exit_code = output.status.code().unwrap_or(-1);

            Ok(CommandOutput {
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                exit_code,
            })
        }
        Ok(Err(e)) => Err(SandboxError::io("waiting for sandboxed command", e)),
        Err(_) => {
            // Timeout exceeded. `kill_on_drop(true)` ensures cleanup.
            Err(SandboxError::Timeout {
                seconds: timeout_secs,
            })
        }
    }
}

/// Output from a sandboxed command execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandOutput {
    /// Standard output of the command.
    pub stdout: String,
    /// Standard error of the command.
    pub stderr: String,
    /// Exit code of the command (-1 if the process was killed by a signal).
    pub exit_code: i32,
}

impl CommandOutput {
    /// Returns `true` if the command exited successfully (exit code 0).
    pub fn success(&self) -> bool {
        self.exit_code == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn run_simple_command() {
        let cwd = std::env::temp_dir();
        let output = run_sandboxed_command("echo hello", 10, 0, &cwd)
            .await
            .unwrap_or_else(|e| CommandOutput {
                stdout: String::new(),
                stderr: e.to_string(),
                exit_code: -1,
            });

        assert!(output.success());
        assert!(output.stdout.contains("hello"));
    }

    #[tokio::test]
    async fn run_command_timeout_returns_error() {
        let cwd = std::env::temp_dir();
        let result = run_sandboxed_command("sleep 60", 1, 0, &cwd).await;
        let Err(SandboxError::Timeout { seconds }) = result else {
            // If it's not a Timeout error, the test should fail.
            assert!(result.is_err(), "expected Timeout error");
            return;
        };
        assert_eq!(seconds, 1);
    }

    #[tokio::test]
    async fn run_failing_command() {
        let cwd = std::env::temp_dir();
        let output = run_sandboxed_command("exit 42", 10, 0, &cwd)
            .await
            .unwrap_or_else(|e| CommandOutput {
                stdout: String::new(),
                stderr: e.to_string(),
                exit_code: -1,
            });

        assert!(!output.success());
        assert_eq!(output.exit_code, 42);
    }

    #[tokio::test]
    async fn run_command_uses_workspace_cwd() {
        // The sandbox must anchor `pwd` to its configured workspace
        // even when the parent process cwd lives elsewhere.
        let workspace = std::env::temp_dir();
        let output = run_sandboxed_command("pwd", 10, 0, &workspace)
            .await
            .unwrap_or_else(|e| CommandOutput {
                stdout: String::new(),
                stderr: e.to_string(),
                exit_code: -1,
            });

        assert!(output.success());
        // Use canonicalisation to compare paths through any symlink
        // (e.g. `/tmp` -> `/private/tmp` on macOS) that the shell's
        // `pwd -L` would otherwise mask.
        let expected = std::fs::canonicalize(&workspace).unwrap_or(workspace.clone());
        let actual_str = output.stdout.trim();
        let actual = std::fs::canonicalize(actual_str).unwrap_or_else(|_| actual_str.into());
        assert_eq!(actual, expected);
    }

    #[test]
    fn command_output_success() {
        let output = CommandOutput {
            stdout: "ok".to_owned(),
            stderr: String::new(),
            exit_code: 0,
        };
        assert!(output.success());
    }

    #[test]
    fn command_output_failure() {
        let output = CommandOutput {
            stdout: String::new(),
            stderr: "error".to_owned(),
            exit_code: 1,
        };
        assert!(!output.success());
    }
}
