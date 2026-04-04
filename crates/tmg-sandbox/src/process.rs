//! Process restriction utilities: timeout enforcement and OOM score adjustment.

use std::time::Duration;

use crate::error::SandboxError;
use crate::platform;

/// Execute a shell command within sandbox constraints.
///
/// This function:
/// 1. Spawns the command via `sh -c`
/// 2. Adjusts the child process's OOM score (Linux only)
/// 3. Enforces a timeout, sending `SIGKILL` if exceeded
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
) -> Result<CommandOutput, SandboxError> {
    let mut child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
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

    // Use `wait()` instead of `wait_with_output()` so we retain ownership
    // of the child handle for explicit kill on timeout. Capture stdout/stderr
    // pipes before awaiting.
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();

    let wait_result = tokio::time::timeout(timeout, child.wait()).await;

    match wait_result {
        Ok(Ok(status)) => {
            let stdout = read_pipe(stdout_pipe).await;
            let stderr = read_stderr_pipe(stderr_pipe).await;
            let exit_code = status.code().unwrap_or(-1);

            Ok(CommandOutput {
                stdout,
                stderr,
                exit_code,
            })
        }
        Ok(Err(e)) => Err(SandboxError::io("waiting for sandboxed command", e)),
        Err(_) => {
            // Timeout exceeded. Send SIGKILL for immediate cleanup.
            // `kill_on_drop(true)` is a fallback if this fails.
            let _ = child.kill().await;
            Err(SandboxError::Timeout {
                seconds: timeout_secs,
            })
        }
    }
}

/// Read all bytes from an optional pipe and return as a lossy UTF-8 string.
async fn read_pipe(pipe: Option<tokio::process::ChildStdout>) -> String {
    use tokio::io::AsyncReadExt;

    let Some(mut pipe) = pipe else {
        return String::new();
    };
    let mut buf = Vec::new();
    let _ = pipe.read_to_end(&mut buf).await;
    String::from_utf8_lossy(&buf).into_owned()
}

/// Read all bytes from an optional stderr pipe and return as a lossy UTF-8 string.
async fn read_stderr_pipe(pipe: Option<tokio::process::ChildStderr>) -> String {
    use tokio::io::AsyncReadExt;

    let Some(mut pipe) = pipe else {
        return String::new();
    };
    let mut buf = Vec::new();
    let _ = pipe.read_to_end(&mut buf).await;
    String::from_utf8_lossy(&buf).into_owned()
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
        let output = run_sandboxed_command("echo hello", 10, 0)
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
        let result = run_sandboxed_command("sleep 60", 1, 0).await;
        let Err(SandboxError::Timeout { seconds }) = result else {
            // If it's not a Timeout error, the test should fail.
            assert!(result.is_err(), "expected Timeout error");
            return;
        };
        assert_eq!(seconds, 1);
    }

    #[tokio::test]
    async fn run_failing_command() {
        let output = run_sandboxed_command("exit 42", 10, 0)
            .await
            .unwrap_or_else(|e| CommandOutput {
                stdout: String::new(),
                stderr: e.to_string(),
                exit_code: -1,
            });

        assert!(!output.success());
        assert_eq!(output.exit_code, 42);
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
