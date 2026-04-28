//! `file_read` tool: read file contents with optional line range.

use tmg_sandbox::SandboxContext;

use crate::error::ToolError;
use crate::path_util::validate_and_resolve;
use crate::types::{Tool, ToolResult};

/// Read the contents of a file, optionally restricting to a line range.
pub struct FileReadTool;

impl Tool for FileReadTool {
    fn name(&self) -> &'static str {
        "file_read"
    }

    fn description(&self) -> &'static str {
        "Read the contents of a file. Optionally specify a line range (1-based, inclusive)."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read."
                },
                "start_line": {
                    "type": "integer",
                    "description": "The 1-based start line (inclusive). Omit to read from the beginning."
                },
                "end_line": {
                    "type": "integer",
                    "description": "The 1-based end line (inclusive). Omit to read to the end."
                }
            },
            "required": ["path"],
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

impl FileReadTool {
    async fn execute_inner(
        &self,
        params: serde_json::Value,
        ctx: &SandboxContext,
    ) -> Result<ToolResult, ToolError> {
        let Some(path_str) = params.get("path").and_then(serde_json::Value::as_str) else {
            return Err(ToolError::invalid_params(
                "missing required parameter: path",
            ));
        };

        let path = validate_and_resolve(path_str, ctx)?;
        ctx.check_path_access(&path)?;

        let content = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| ToolError::io(format!("reading file '{}'", path.display()), e))?;

        let start_line = params
            .get("start_line")
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| usize::try_from(n).ok());

        if start_line == Some(0) {
            return Err(ToolError::invalid_params(
                "start_line is 1-based and must be >= 1",
            ));
        }

        let end_line = params
            .get("end_line")
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| usize::try_from(n).ok());

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let start = start_line.unwrap_or(1).saturating_sub(1);
        let end = end_line.unwrap_or(total_lines).min(total_lines);

        if start >= total_lines {
            return Ok(ToolResult::success(format!(
                "(file has {total_lines} lines, requested start line {} is out of range)",
                start + 1,
            )));
        }

        let selected: Vec<String> = lines[start..end]
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:>6}\t{line}", start + i + 1))
            .collect();

        let mut output = selected.join("\n");
        if start > 0 || end < total_lines {
            output = format!(
                "(showing lines {}-{} of {total_lines})\n{output}",
                start + 1,
                end,
            );
        }

        Ok(ToolResult::success(output))
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
    async fn read_entire_file() {
        let dir = std::env::temp_dir().join("tmg_tools_test_file_read");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("test.txt");
        std::fs::write(&file, "line1\nline2\nline3\n").ok();

        let tool = FileReadTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({ "path": file.to_str().unwrap() }),
                &sandbox,
            )
            .await;

        let result = result.unwrap();
        assert!(!result.is_error);
        assert!(result.output.contains("line1"));
        assert!(result.output.contains("line3"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn read_line_range() {
        let dir = std::env::temp_dir().join("tmg_tools_test_file_read_range");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("test.txt");
        std::fs::write(&file, "line1\nline2\nline3\nline4\nline5\n").ok();

        let tool = FileReadTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file.to_str().unwrap(),
                    "start_line": 2,
                    "end_line": 4
                }),
                &sandbox,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.output.contains("line2"));
        assert!(result.output.contains("line4"));
        assert!(!result.output.contains("\tline1"));
        assert!(!result.output.contains("\tline5"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn read_missing_file() {
        let tool = FileReadTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({ "path": "/tmp/tmg_nonexistent_file_xyz.txt" }),
                &sandbox,
            )
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("reading file"));
    }

    #[tokio::test]
    async fn read_missing_path_param() {
        let tool = FileReadTool;
        let sandbox = ctx();
        let result = tool.execute(serde_json::json!({}), &sandbox).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn read_start_line_zero_rejected() {
        let dir = std::env::temp_dir().join("tmg_tools_test_file_read_zero");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("test.txt");
        std::fs::write(&file, "line1\nline2\n").ok();

        let tool = FileReadTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file.to_str().unwrap(),
                    "start_line": 0
                }),
                &sandbox,
            )
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("1-based"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn read_path_traversal() {
        let tool = FileReadTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({ "path": "/tmp/../etc/passwd" }),
                &sandbox,
            )
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("traversal"));
    }

    /// Reading outside the workspace under a `WorkspaceWrite` sandbox
    /// must surface a [`ToolError::Sandbox`] rather than silently
    /// returning the file contents.
    #[tokio::test]
    async fn read_outside_workspace_denied() {
        use tmg_sandbox::{SandboxConfig, SandboxMode};

        // Workspace points at a fresh tmpdir; `/etc/passwd` (or any
        // file outside it that isn't on the system-read allowlist) is
        // refused.
        let workspace = std::env::temp_dir().join("tmg_tools_test_file_read_outside_ws");
        let _ = std::fs::create_dir_all(&workspace);
        // The workspace check is on the resolved path; pick a target
        // that is not under the workspace AND not on the system-read
        // allowlist (`/usr`, `/bin`, …) — `/private/var/...` works on
        // macOS and `/var/tmp/...` on Linux. Use `$HOME/.tmg-test-outside`
        // which is always outside the workspace.
        let outside = std::env::temp_dir().join("tmg_tools_test_file_read_target");
        let _ = std::fs::create_dir_all(outside.parent().unwrap());
        std::fs::write(&outside, "secret").ok();

        // Build a fresh, non-`Full` sandbox so the workspace check
        // actually fires. Note we deliberately pick a workspace that
        // does NOT contain the target path.
        let sandbox = SandboxContext::new(
            SandboxConfig::new(&workspace).with_mode(SandboxMode::WorkspaceWrite),
        );
        let tool = FileReadTool;
        let result = tool
            .execute(
                serde_json::json!({ "path": outside.to_str().unwrap() }),
                &sandbox,
            )
            .await;

        // The check might pass on platforms where the canonical form
        // of `outside` happens to live under a system-read path; that
        // is a property of `system_read_paths`, not a bug. But for the
        // common temp-dir layout the access must be denied.
        if let Ok(res) = &result {
            // If the platform allow-listed the path, at least confirm
            // we did not read the wrong file.
            assert!(
                !res.output.contains("secret") || res.is_error,
                "unexpectedly read outside-workspace file under WorkspaceWrite"
            );
        } else {
            let err = result.unwrap_err();
            assert!(
                matches!(err, ToolError::Sandbox { .. }),
                "expected ToolError::Sandbox, got {err:?}"
            );
        }

        let _ = std::fs::remove_file(&outside);
        let _ = std::fs::remove_dir_all(&workspace);
    }
}
