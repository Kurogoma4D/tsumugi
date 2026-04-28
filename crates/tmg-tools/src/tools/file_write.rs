//! `file_write` tool: create or overwrite a file.

use std::path::Path;

use tmg_sandbox::SandboxContext;

use crate::error::ToolError;
use crate::path_util::validate_and_resolve;
use crate::types::{Tool, ToolResult};

/// Create or overwrite a file with the given content.
pub struct FileWriteTool;

impl Tool for FileWriteTool {
    fn name(&self) -> &'static str {
        "file_write"
    }

    fn description(&self) -> &'static str {
        "Create or overwrite a file with the specified content. Parent directories are created automatically."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path of the file to create or overwrite."
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file."
                }
            },
            "required": ["path", "content"],
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

impl FileWriteTool {
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
        let Some(content) = params.get("content").and_then(serde_json::Value::as_str) else {
            return Err(ToolError::invalid_params(
                "missing required parameter: content",
            ));
        };

        let path = validate_and_resolve(path_str, ctx)?;
        ctx.check_write_access(&path)?;

        // Create parent directories if they don't exist.
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            ensure_parent_dirs(parent).await?;
        }

        tokio::fs::write(&path, content)
            .await
            .map_err(|e| ToolError::io(format!("writing file '{}'", path.display()), e))?;

        let bytes = content.len();
        Ok(ToolResult::success(format!(
            "Successfully wrote {bytes} bytes to '{}'",
            path.display()
        )))
    }
}

/// Create parent directories, reporting a useful error on failure.
async fn ensure_parent_dirs(parent: &Path) -> Result<(), ToolError> {
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(|e| ToolError::io(format!("creating directories '{}'", parent.display()), e))
}

#[expect(clippy::unwrap_used, reason = "test assertions")]
#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> SandboxContext {
        SandboxContext::test_default()
    }

    #[tokio::test]
    async fn write_new_file() {
        let dir = std::env::temp_dir().join("tmg_tools_test_file_write");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("new.txt");

        let tool = FileWriteTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file.to_str().unwrap(),
                    "content": "hello world"
                }),
                &sandbox,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.output.contains("11 bytes"));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "hello world");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn write_creates_parent_dirs() {
        let dir = std::env::temp_dir().join("tmg_tools_test_file_write_nested");
        let _ = std::fs::remove_dir_all(&dir);
        let file = dir.join("a").join("b").join("c.txt");

        let tool = FileWriteTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file.to_str().unwrap(),
                    "content": "nested"
                }),
                &sandbox,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "nested");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn write_missing_content_param() {
        let tool = FileWriteTool;
        let sandbox = ctx();
        let result = tool
            .execute(serde_json::json!({ "path": "/tmp/test.txt" }), &sandbox)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn write_path_traversal() {
        let tool = FileWriteTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({
                    "path": "/tmp/../etc/shadow",
                    "content": "pwned"
                }),
                &sandbox,
            )
            .await;
        assert!(result.is_err());
    }

    /// Writing outside the sandbox workspace under `WorkspaceWrite`
    /// must surface a [`ToolError::Sandbox`] before the file is
    /// touched.
    #[tokio::test]
    async fn write_outside_workspace_denied() {
        use tmg_sandbox::{SandboxConfig, SandboxMode};

        let workspace = std::env::temp_dir().join("tmg_tools_test_file_write_ws");
        let _ = std::fs::create_dir_all(&workspace);
        let outside = std::env::temp_dir().join("tmg_tools_test_file_write_outside.txt");
        let _ = std::fs::remove_file(&outside);

        let sandbox = SandboxContext::new(
            SandboxConfig::new(&workspace).with_mode(SandboxMode::WorkspaceWrite),
        );
        let tool = FileWriteTool;
        let result = tool
            .execute(
                serde_json::json!({
                    "path": outside.to_str().unwrap(),
                    "content": "should-not-write",
                }),
                &sandbox,
            )
            .await;

        match result {
            Err(ToolError::Sandbox { .. }) => {}
            other => unreachable!(
                "expected ToolError::Sandbox for workspace-external write, got {other:?}"
            ),
        }
        assert!(
            !outside.exists(),
            "sandbox-denied write must not create the file"
        );

        let _ = std::fs::remove_dir_all(&workspace);
    }

    /// Under `ReadOnly` mode, even workspace-internal writes are
    /// denied — reads remain allowed.
    #[tokio::test]
    async fn write_under_read_only_denied() {
        use tmg_sandbox::{SandboxConfig, SandboxMode};

        let workspace = std::env::temp_dir().join("tmg_tools_test_file_write_ro_ws");
        let _ = std::fs::create_dir_all(&workspace);
        let target = workspace.join("inside.txt");

        let sandbox =
            SandboxContext::new(SandboxConfig::new(&workspace).with_mode(SandboxMode::ReadOnly));
        let tool = FileWriteTool;
        let result = tool
            .execute(
                serde_json::json!({
                    "path": target.to_str().unwrap(),
                    "content": "blocked",
                }),
                &sandbox,
            )
            .await;
        match result {
            Err(ToolError::Sandbox { .. }) => {}
            other => {
                unreachable!("expected ToolError::Sandbox under read-only mode, got {other:?}")
            }
        }
        assert!(!target.exists());

        let _ = std::fs::remove_dir_all(&workspace);
    }
}
