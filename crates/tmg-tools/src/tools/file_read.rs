//! `file_read` tool: read file contents with optional line range.

use crate::error::ToolError;
use crate::path_util::validate_path;
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

    fn execute(
        &self,
        params: serde_json::Value,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + '_>,
    > {
        Box::pin(self.execute_inner(params))
    }
}

impl FileReadTool {
    async fn execute_inner(&self, params: serde_json::Value) -> Result<ToolResult, ToolError> {
        let Some(path_str) = params.get("path").and_then(serde_json::Value::as_str) else {
            return Err(ToolError::invalid_params(
                "missing required parameter: path",
            ));
        };

        let path = validate_path(path_str)?;

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

    #[tokio::test]
    async fn read_entire_file() {
        let dir = std::env::temp_dir().join("tmg_tools_test_file_read");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("test.txt");
        std::fs::write(&file, "line1\nline2\nline3\n").ok();

        let tool = FileReadTool;
        let result = tool
            .execute(serde_json::json!({ "path": file.to_str().unwrap() }))
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
        let result = tool
            .execute(serde_json::json!({
                "path": file.to_str().unwrap(),
                "start_line": 2,
                "end_line": 4
            }))
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
        let result = tool
            .execute(serde_json::json!({ "path": "/tmp/tmg_nonexistent_file_xyz.txt" }))
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("reading file"));
    }

    #[tokio::test]
    async fn read_missing_path_param() {
        let tool = FileReadTool;
        let result = tool.execute(serde_json::json!({})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn read_start_line_zero_rejected() {
        let dir = std::env::temp_dir().join("tmg_tools_test_file_read_zero");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("test.txt");
        std::fs::write(&file, "line1\nline2\n").ok();

        let tool = FileReadTool;
        let result = tool
            .execute(serde_json::json!({
                "path": file.to_str().unwrap(),
                "start_line": 0
            }))
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("1-based"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn read_path_traversal() {
        let tool = FileReadTool;
        let result = tool
            .execute(serde_json::json!({ "path": "/tmp/../etc/passwd" }))
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("traversal"));
    }
}
