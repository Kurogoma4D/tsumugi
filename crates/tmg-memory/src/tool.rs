//! [`MemoryTool`]: the agent-facing CRUD tool over a [`MemoryStore`].

use std::pin::Pin;
use std::str::FromStr as _;
use std::sync::Arc;

use serde_json::Value as JsonValue;
use tmg_sandbox::SandboxContext;
use tmg_tools::error::ToolError;
use tmg_tools::types::{Tool, ToolResult};

use crate::budget::{BudgetReport, capacity_nudge};
use crate::entry::MemoryType;
use crate::error::MemoryError;
use crate::store::MemoryStore;

/// Name of the tool registered in the [`tmg_tools::ToolRegistry`].
pub const MEMORY_TOOL_NAME: &str = "memory";

/// Agent-facing CRUD tool over a [`MemoryStore`].
///
/// All write operations target the project layer; the global layer is
/// read-only from the LLM's perspective. The tool generates file paths
/// internally so the LLM cannot inject arbitrary write targets.
pub struct MemoryTool {
    store: Arc<MemoryStore>,
}

impl MemoryTool {
    /// Construct a tool around an existing store.
    #[must_use]
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self { store }
    }
}

impl Tool for MemoryTool {
    fn name(&self) -> &'static str {
        MEMORY_TOOL_NAME
    }

    fn description(&self) -> &'static str {
        "Project-scoped persistent memory. Use `read` to inspect a topic before starting work, \
         and `add` / `update` / `remove` to curate. Index lives at `.tsumugi/memory/MEMORY.md` \
         and is included in the system prompt."
    }

    fn parameters_schema(&self) -> JsonValue {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "update", "remove", "read"],
                    "description": "Which CRUD operation to perform."
                },
                "name": {
                    "type": "string",
                    "description": "Topic name (ASCII alphanumerics + `_` / `-`). Becomes <name>.md."
                },
                "type": {
                    "type": "string",
                    "enum": ["user", "feedback", "project", "reference"],
                    "description": "Categorical tag; required for `add`, optional for `update`."
                },
                "description": {
                    "type": "string",
                    "description": "One-line summary written into MEMORY.md. Required for `add`."
                },
                "content": {
                    "type": "string",
                    "description": "Full body text. Required for `add`; optional for `update`."
                }
            },
            "required": ["action", "name"],
            "additionalProperties": false
        })
    }

    fn execute<'a>(
        &'a self,
        params: JsonValue,
        ctx: &'a SandboxContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult, ToolError>> + Send + 'a>> {
        // The store I/O is synchronous (small files; the agent loop
        // already runs us inside a tokio task). Wrap the result in a
        // ready-future so the Tool trait stays uniform across async
        // and sync implementations.
        let result = self.execute_inner(&params, ctx);
        Box::pin(std::future::ready(result))
    }
}

impl MemoryTool {
    fn execute_inner(
        &self,
        params: &JsonValue,
        ctx: &SandboxContext,
    ) -> Result<ToolResult, ToolError> {
        let Some(action) = params.get("action").and_then(JsonValue::as_str) else {
            return Err(ToolError::invalid_params(
                "missing required parameter: action",
            ));
        };
        let Some(name) = params.get("name").and_then(JsonValue::as_str) else {
            return Err(ToolError::invalid_params(
                "missing required parameter: name",
            ));
        };

        match action {
            "read" => self.handle_read(name),
            "add" => {
                let kind_str = params
                    .get("type")
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| ToolError::invalid_params("missing required parameter: type"))?;
                let kind = MemoryType::from_str(kind_str)
                    .map_err(|v| ToolError::invalid_params(format!("invalid type: {v}")))?;
                let description = params
                    .get("description")
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| {
                        ToolError::invalid_params("missing required parameter: description")
                    })?;
                let content = params
                    .get("content")
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| {
                        ToolError::invalid_params("missing required parameter: content")
                    })?;
                self.handle_add(name, kind, description, content, ctx)
            }
            "update" => {
                let kind = match params.get("type").and_then(JsonValue::as_str) {
                    Some(s) => Some(
                        MemoryType::from_str(s)
                            .map_err(|v| ToolError::invalid_params(format!("invalid type: {v}")))?,
                    ),
                    None => None,
                };
                let description = params.get("description").and_then(JsonValue::as_str);
                let content = params.get("content").and_then(JsonValue::as_str);
                if description.is_none() && content.is_none() && kind.is_none() {
                    return Err(ToolError::invalid_params(
                        "update requires at least one of: type, description, content",
                    ));
                }
                self.handle_update(name, kind, description, content, ctx)
            }
            "remove" => self.handle_remove(name, ctx),
            other => Err(ToolError::invalid_params(format!(
                "unknown action {other:?}; expected add | update | remove | read"
            ))),
        }
    }

    fn handle_read(&self, name: &str) -> Result<ToolResult, ToolError> {
        match self.store.read(name) {
            Ok((entry, scope)) => {
                let body = entry.body;
                let header = format!(
                    "[memory:{}] {} ({}) — {}\n\n",
                    scope.as_str(),
                    entry.frontmatter.name,
                    entry.frontmatter.kind,
                    entry.frontmatter.description,
                );
                Ok(ToolResult::success(header + &body))
            }
            Err(MemoryError::NotFound { .. }) => {
                Ok(ToolResult::error(format!("no memory entry named {name:?}")))
            }
            Err(MemoryError::InvalidName { reason, .. }) => Err(ToolError::invalid_params(reason)),
            Err(other) => Err(map_memory_error(other)),
        }
    }

    fn handle_add(
        &self,
        name: &str,
        kind: MemoryType,
        description: &str,
        content: &str,
        ctx: &SandboxContext,
    ) -> Result<ToolResult, ToolError> {
        // Sandbox check — the project memory directory must be writable
        // under the active sandbox policy. The path is generated by the
        // store, so the LLM cannot influence the destination.
        let target_dir = self.store.project_dir();
        ctx.check_write_access(target_dir)?;

        match self.store.add(name, kind, description, content) {
            Ok(report) => Ok(success_with_budget(
                format!("added memory entry {name:?} ({kind}). Index updated."),
                &report,
                self.store.budget(),
                content.chars().count(),
            )),
            Err(MemoryError::AlreadyExists { .. }) => Ok(ToolResult::error(format!(
                "memory entry {name:?} already exists; use action=update to modify it"
            ))),
            Err(MemoryError::InvalidName { reason, .. }) => Err(ToolError::invalid_params(reason)),
            Err(other) => Err(map_memory_error(other)),
        }
    }

    fn handle_update(
        &self,
        name: &str,
        kind: Option<MemoryType>,
        description: Option<&str>,
        content: Option<&str>,
        ctx: &SandboxContext,
    ) -> Result<ToolResult, ToolError> {
        ctx.check_write_access(self.store.project_dir())?;
        match self.store.update(name, kind, description, content) {
            Ok(report) => {
                let chars = content.map_or(0, |c| c.chars().count());
                Ok(success_with_budget(
                    format!("updated memory entry {name:?}."),
                    &report,
                    self.store.budget(),
                    chars,
                ))
            }
            Err(MemoryError::NotFound { .. }) => Ok(ToolResult::error(format!(
                "no project-layer memory entry named {name:?}"
            ))),
            Err(MemoryError::InvalidName { reason, .. }) => Err(ToolError::invalid_params(reason)),
            Err(other) => Err(map_memory_error(other)),
        }
    }

    fn handle_remove(&self, name: &str, ctx: &SandboxContext) -> Result<ToolResult, ToolError> {
        ctx.check_write_access(self.store.project_dir())?;
        match self.store.remove(name) {
            Ok(report) => Ok(success_with_budget(
                format!("removed memory entry {name:?}."),
                &report,
                self.store.budget(),
                0,
            )),
            Err(MemoryError::NotFound { .. }) => Ok(ToolResult::error(format!(
                "no project-layer memory entry named {name:?}"
            ))),
            Err(MemoryError::InvalidName { reason, .. }) => Err(ToolError::invalid_params(reason)),
            Err(other) => Err(map_memory_error(other)),
        }
    }
}

fn success_with_budget(
    base_message: String,
    report: &BudgetReport,
    budget: &crate::budget::MemoryBudget,
    body_chars: usize,
) -> ToolResult {
    use std::fmt::Write as _;
    let mut msg = base_message;
    if body_chars > budget.entry_max_chars {
        // Best-effort: writing into a `String` cannot fail.
        let _ = write!(
            msg,
            " warning: body is {} chars (limit {}). Consider summarising.",
            body_chars, budget.entry_max_chars,
        );
    }
    if let Some(nudge) = capacity_nudge(report, budget) {
        msg.push_str(&nudge);
    }
    ToolResult::success(msg)
}

fn map_memory_error(err: MemoryError) -> ToolError {
    match err {
        MemoryError::Io { context, source } => ToolError::Io { context, source },
        MemoryError::Yaml { path, source } => ToolError::Io {
            context: format!("yaml at {path}"),
            source: std::io::Error::other(source.to_string()),
        },
        other => ToolError::Io {
            context: "memory store".to_owned(),
            source: std::io::Error::other(other.to_string()),
        },
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;
    use crate::budget::MemoryBudget;

    fn make_store(root: &std::path::Path) -> Arc<MemoryStore> {
        Arc::new(MemoryStore::with_dirs(
            root.join(".tsumugi").join("memory"),
            None,
            MemoryBudget::default(),
        ))
    }

    #[tokio::test]
    async fn add_then_read_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = make_store(dir.path());
        let tool = MemoryTool::new(store);
        let ctx = SandboxContext::test_default();

        let res = tool
            .execute(
                serde_json::json!({
                    "action": "add",
                    "name": "topic_one",
                    "type": "project",
                    "description": "first topic",
                    "content": "topic body content",
                }),
                &ctx,
            )
            .await
            .expect("add ok");
        assert!(!res.is_error, "add error: {}", res.output);

        let res = tool
            .execute(
                serde_json::json!({
                    "action": "read",
                    "name": "topic_one",
                }),
                &ctx,
            )
            .await
            .expect("read ok");
        assert!(!res.is_error, "read error: {}", res.output);
        assert!(res.output.contains("topic body content"));
        assert!(res.output.contains("[memory:project]"));
    }

    #[tokio::test]
    async fn add_duplicate_returns_error_result() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = make_store(dir.path());
        let tool = MemoryTool::new(store);
        let ctx = SandboxContext::test_default();

        let _ = tool
            .execute(
                serde_json::json!({
                    "action": "add",
                    "name": "dup",
                    "type": "user",
                    "description": "d",
                    "content": "c",
                }),
                &ctx,
            )
            .await
            .expect("first add");
        let res = tool
            .execute(
                serde_json::json!({
                    "action": "add",
                    "name": "dup",
                    "type": "user",
                    "description": "d",
                    "content": "c",
                }),
                &ctx,
            )
            .await
            .expect("second add (returns error result)");
        assert!(res.is_error);
        assert!(res.output.contains("already exists"));
    }

    #[tokio::test]
    async fn update_partial_description_only() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = make_store(dir.path());
        let tool = MemoryTool::new(store);
        let ctx = SandboxContext::test_default();

        let _ = tool
            .execute(
                serde_json::json!({
                    "action": "add",
                    "name": "t",
                    "type": "feedback",
                    "description": "old",
                    "content": "old body",
                }),
                &ctx,
            )
            .await
            .expect("add");
        let _ = tool
            .execute(
                serde_json::json!({
                    "action": "update",
                    "name": "t",
                    "description": "new",
                }),
                &ctx,
            )
            .await
            .expect("update");
        let res = tool
            .execute(
                serde_json::json!({
                    "action": "read",
                    "name": "t",
                }),
                &ctx,
            )
            .await
            .expect("read");
        assert!(res.output.contains("new"));
        assert!(res.output.contains("old body"));
    }

    #[tokio::test]
    async fn remove_drops_entry() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = make_store(dir.path());
        let tool = MemoryTool::new(store);
        let ctx = SandboxContext::test_default();
        let _ = tool
            .execute(
                serde_json::json!({
                    "action": "add",
                    "name": "t",
                    "type": "feedback",
                    "description": "d",
                    "content": "c",
                }),
                &ctx,
            )
            .await
            .expect("add");
        let res = tool
            .execute(
                serde_json::json!({
                    "action": "remove",
                    "name": "t",
                }),
                &ctx,
            )
            .await
            .expect("remove");
        assert!(!res.is_error);
        let res = tool
            .execute(
                serde_json::json!({
                    "action": "read",
                    "name": "t",
                }),
                &ctx,
            )
            .await
            .expect("read missing");
        assert!(res.is_error);
    }

    #[tokio::test]
    async fn capacity_warning_surfaces_in_result() {
        let dir = tempfile::tempdir().expect("tempdir");
        let small = MemoryBudget {
            index_max_lines: 200,
            entry_max_chars: 600,
            total_files_limit: 1,
        };
        let store = Arc::new(MemoryStore::with_dirs(
            dir.path().join(".tsumugi").join("memory"),
            None,
            small,
        ));
        let tool = MemoryTool::new(store);
        let ctx = SandboxContext::test_default();
        // First add fills the budget; second add exceeds it.
        let _ = tool
            .execute(
                serde_json::json!({
                    "action": "add",
                    "name": "a",
                    "type": "user",
                    "description": "d",
                    "content": "c",
                }),
                &ctx,
            )
            .await
            .expect("first");
        let res = tool
            .execute(
                serde_json::json!({
                    "action": "add",
                    "name": "b",
                    "type": "user",
                    "description": "d",
                    "content": "c",
                }),
                &ctx,
            )
            .await
            .expect("second");
        assert!(!res.is_error, "expected success with warning, got error");
        assert!(
            res.output.contains("near capacity"),
            "expected capacity nudge, got: {}",
            res.output
        );
    }

    #[tokio::test]
    async fn unknown_action_is_invalid_params() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = make_store(dir.path());
        let tool = MemoryTool::new(store);
        let ctx = SandboxContext::test_default();
        let err = tool
            .execute(
                serde_json::json!({
                    "action": "wat",
                    "name": "t",
                }),
                &ctx,
            )
            .await
            .expect_err("invalid action");
        assert!(matches!(err, ToolError::InvalidParams { .. }));
    }
}
