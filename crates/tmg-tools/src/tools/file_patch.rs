//! `file_patch` tool: search/replace partial file editing.

use tmg_sandbox::SandboxContext;

use crate::error::ToolError;
use crate::path_util::validate_and_resolve;
use crate::types::{Tool, ToolResult};

/// Apply a search/replace patch to a file.
pub struct FilePatchTool;

impl Tool for FilePatchTool {
    fn name(&self) -> &'static str {
        "file_patch"
    }

    fn description(&self) -> &'static str {
        "Apply a search/replace edit to a file. The `old_string` must match exactly \
         one location in the file. It will be replaced with `new_string`."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to edit."
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to search for in the file. Must match exactly once."
                },
                "new_string": {
                    "type": "string",
                    "description": "The string to replace `old_string` with. Use empty string to delete."
                }
            },
            "required": ["path", "old_string", "new_string"],
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

impl FilePatchTool {
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
        let Some(old_string) = params.get("old_string").and_then(serde_json::Value::as_str) else {
            return Err(ToolError::invalid_params(
                "missing required parameter: old_string",
            ));
        };
        let Some(new_string) = params.get("new_string").and_then(serde_json::Value::as_str) else {
            return Err(ToolError::invalid_params(
                "missing required parameter: new_string",
            ));
        };

        let path = validate_and_resolve(path_str, ctx)?;
        // Patching is a write: gate it on `check_write_access` so
        // ReadOnly subagents cannot rewrite files. We deliberately
        // skip the explicit `check_path_access` call here: the only
        // paths it would additionally permit are the system-read
        // allowlist (e.g. `/usr`, `/bin`), and writing into those
        // would be denied by `check_write_access` anyway. So skipping
        // the read check loses no enforcement and avoids a redundant
        // round-trip through `normalize_path`.
        ctx.check_write_access(&path)?;

        let content = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| ToolError::io(format!("reading file '{}'", path.display()), e))?;

        let match_count = content.matches(old_string).count();

        if match_count == 0 {
            return Ok(ToolResult::error(format!(
                "old_string not found in '{}'",
                path.display()
            )));
        }

        if match_count > 1 {
            return Ok(ToolResult::error(format!(
                "old_string matches {match_count} locations in '{}'. It must match exactly once.",
                path.display()
            )));
        }

        let new_content = content.replacen(old_string, new_string, 1);

        tokio::fs::write(&path, &new_content)
            .await
            .map_err(|e| ToolError::io(format!("writing file '{}'", path.display()), e))?;

        Ok(ToolResult::success(format!(
            "Successfully patched '{}'",
            path.display()
        )))
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
    async fn patch_single_match() {
        let dir = std::env::temp_dir().join("tmg_tools_test_file_patch");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("test.txt");
        std::fs::write(&file, "hello world\nfoo bar\n").ok();

        let tool = FilePatchTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file.to_str().unwrap(),
                    "old_string": "foo bar",
                    "new_string": "baz qux"
                }),
                &sandbox,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        let content = std::fs::read_to_string(&file).unwrap();
        assert_eq!(content, "hello world\nbaz qux\n");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn patch_no_match() {
        let dir = std::env::temp_dir().join("tmg_tools_test_file_patch_no_match");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("test.txt");
        std::fs::write(&file, "hello world\n").ok();

        let tool = FilePatchTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file.to_str().unwrap(),
                    "old_string": "nonexistent",
                    "new_string": "replacement"
                }),
                &sandbox,
            )
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.output.contains("not found"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn patch_multiple_matches() {
        let dir = std::env::temp_dir().join("tmg_tools_test_file_patch_multi");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("test.txt");
        std::fs::write(&file, "aaa\naaa\naaa\n").ok();

        let tool = FilePatchTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file.to_str().unwrap(),
                    "old_string": "aaa",
                    "new_string": "bbb"
                }),
                &sandbox,
            )
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.output.contains("3 locations"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn patch_missing_file() {
        let tool = FilePatchTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({
                    "path": "/tmp/tmg_nonexistent_patch_xyz.txt",
                    "old_string": "a",
                    "new_string": "b"
                }),
                &sandbox,
            )
            .await;
        assert!(result.is_err());
    }
}
