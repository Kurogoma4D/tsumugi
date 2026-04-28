//! `list_dir` tool: directory tree display with optional depth limit.

use std::fmt::Write;
use std::path::{Path, PathBuf};

use tmg_sandbox::SandboxContext;

use crate::error::ToolError;
use crate::path_util::validate_and_resolve;
use crate::types::{Tool, ToolResult};

/// Default maximum depth for directory traversal.
const DEFAULT_MAX_DEPTH: usize = 3;

/// Maximum number of entries to display before truncating.
const MAX_ENTRIES: usize = 1000;

/// List the contents of a directory as a tree.
pub struct ListDirTool;

impl Tool for ListDirTool {
    fn name(&self) -> &'static str {
        "list_dir"
    }

    fn description(&self) -> &'static str {
        "List the contents of a directory in a tree format. Optionally specify a depth limit."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the directory to list."
                },
                "depth": {
                    "type": "integer",
                    "description": "Maximum depth to recurse (default: 3). Use 1 for immediate children only."
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

impl ListDirTool {
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

        let max_depth = params
            .get("depth")
            .and_then(serde_json::Value::as_u64)
            .and_then(|d| usize::try_from(d).ok())
            .unwrap_or(DEFAULT_MAX_DEPTH);

        let path = validate_and_resolve(path_str, ctx)?;
        ctx.check_path_access(&path)?;

        // Use blocking I/O for the directory walk since std::fs::read_dir is
        // synchronous and we may read many entries.
        let tree = tokio::task::spawn_blocking(move || build_tree(&path, max_depth))
            .await
            .map_err(|e| {
                ToolError::io("directory listing task panicked", std::io::Error::other(e))
            })??;

        Ok(ToolResult::success(tree))
    }
}

/// Build a tree representation of the directory.
fn build_tree(root: &Path, max_depth: usize) -> Result<String, ToolError> {
    let mut entries = Vec::new();
    let mut count = 0;
    let mut truncated = false;

    collect_entries(root, 0, max_depth, &mut entries, &mut count, &mut truncated)?;

    let mut output = format!("{}/\n", root.display());
    for (indent, name, is_dir) in &entries {
        let prefix = "  ".repeat(*indent);
        let suffix = if *is_dir { "/" } else { "" };
        let _ = writeln!(output, "{prefix}{name}{suffix}");
    }

    if truncated {
        let _ = writeln!(output, "... (truncated at {MAX_ENTRIES} entries)");
    }

    Ok(output)
}

/// Recursively collect directory entries.
fn collect_entries(
    dir: &Path,
    depth: usize,
    max_depth: usize,
    entries: &mut Vec<(usize, String, bool)>,
    count: &mut usize,
    truncated: &mut bool,
) -> Result<(), ToolError> {
    if depth >= max_depth || *truncated {
        return Ok(());
    }

    let read_dir = std::fs::read_dir(dir)
        .map_err(|e| ToolError::io(format!("reading directory '{}'", dir.display()), e))?;

    let mut children: Vec<PathBuf> = Vec::new();
    for entry in read_dir {
        let entry =
            entry.map_err(|e| ToolError::io(format!("reading entry in '{}'", dir.display()), e))?;
        children.push(entry.path());
    }

    // Sort entries for deterministic output.
    children.sort();

    for child in &children {
        if *count >= MAX_ENTRIES {
            *truncated = true;
            return Ok(());
        }

        let name = child
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();

        // Skip hidden files/directories.
        if name.starts_with('.') {
            continue;
        }

        let is_dir = child.is_dir();
        entries.push((depth + 1, name, is_dir));
        *count += 1;

        if is_dir {
            collect_entries(child, depth + 1, max_depth, entries, count, truncated)?;
        }
    }

    Ok(())
}

#[expect(clippy::unwrap_used, reason = "test assertions")]
#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> SandboxContext {
        SandboxContext::test_default()
    }

    #[tokio::test]
    async fn list_dir_basic() {
        let dir = std::env::temp_dir().join("tmg_tools_test_list_dir");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).ok();
        std::fs::write(dir.join("file.txt"), "content").ok();
        std::fs::write(dir.join("sub").join("nested.txt"), "nested").ok();

        let tool = ListDirTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({ "path": dir.to_str().unwrap() }),
                &sandbox,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.output.contains("file.txt"));
        assert!(result.output.contains("sub/"));
        assert!(result.output.contains("nested.txt"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn list_dir_depth_1() {
        let dir = std::env::temp_dir().join("tmg_tools_test_list_dir_depth");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub").join("deep")).ok();
        std::fs::write(dir.join("top.txt"), "top").ok();
        std::fs::write(dir.join("sub").join("deep").join("deep.txt"), "deep").ok();

        let tool = ListDirTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({
                    "path": dir.to_str().unwrap(),
                    "depth": 1
                }),
                &sandbox,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.output.contains("top.txt"));
        assert!(result.output.contains("sub/"));
        // deep.txt should NOT appear at depth 1.
        assert!(!result.output.contains("deep.txt"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn list_dir_nonexistent() {
        let tool = ListDirTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({ "path": "/tmp/tmg_nonexistent_dir_xyz" }),
                &sandbox,
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn list_dir_hidden_files_skipped() {
        let dir = std::env::temp_dir().join("tmg_tools_test_list_dir_hidden");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join(".hidden"), "hidden").ok();
        std::fs::write(dir.join("visible.txt"), "visible").ok();

        let tool = ListDirTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({ "path": dir.to_str().unwrap() }),
                &sandbox,
            )
            .await
            .unwrap();

        assert!(!result.output.contains(".hidden"));
        assert!(result.output.contains("visible.txt"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
