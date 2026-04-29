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

/// Callback fired after every successful write (`add` / `update` /
/// `remove`) with the freshly-rendered merged index. The agent loop
/// installs one of these so the in-memory copy of the index stays in
/// sync with disk and a "memory updated" hint can be surfaced in the
/// TUI Activity Pane.
///
/// The closure must be `Send + Sync` so it can be moved into the
/// async tool dispatch path.
pub type MemoryRefreshCallback = Arc<dyn Fn(&str) + Send + Sync>;

/// Agent-facing CRUD tool over a [`MemoryStore`].
///
/// All write operations target the project layer; the global layer is
/// read-only from the LLM's perspective. The tool generates file paths
/// internally so the LLM cannot inject arbitrary write targets.
pub struct MemoryTool {
    store: Arc<MemoryStore>,
    /// Optional refresh callback fired after every successful write.
    /// `None` means the tool runs without notifying anyone, which is
    /// the right default for code paths that don't have an
    /// `AgentLoop` to update (e.g. unit tests).
    refresh: Option<MemoryRefreshCallback>,
}

impl MemoryTool {
    /// Construct a tool around an existing store, with no refresh
    /// hook installed.
    #[must_use]
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self {
            store,
            refresh: None,
        }
    }

    /// Construct a tool with a refresh callback that fires after every
    /// successful write. The callback receives the freshly-rendered
    /// merged index payload (matching what the system prompt saw at
    /// startup).
    ///
    /// Used by the CLI startup wiring to keep [`tmg_core::AgentLoop`]'s
    /// in-memory index up to date and to emit a "memory updated"
    /// stream event into the TUI Activity Pane (issue #4 of PR #76).
    #[must_use]
    pub fn with_refresh(store: Arc<MemoryStore>, refresh: MemoryRefreshCallback) -> Self {
        Self {
            store,
            refresh: Some(refresh),
        }
    }

    /// Fire the refresh callback (if any) with the latest merged
    /// index. Best-effort: a stale read is silently ignored so a
    /// transient I/O error never aborts a successful write.
    fn fire_refresh(&self) {
        let Some(cb) = self.refresh.as_ref() else {
            return;
        };
        match self.store.read_merged_index() {
            Ok(index) => cb(&index),
            Err(e) => {
                tracing::debug!(error = %e, "memory refresh: read_merged_index failed; skipping callback");
            }
        }
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

        // Validate body length BEFORE the store write so an oversized
        // entry never persists. Issue #9 of PR #76: previously the
        // length check fired after `store.add(...)` and the entry was
        // already on disk by the time the warning surfaced.
        let body_chars = content.chars().count();
        let budget = self.store.budget();
        if body_chars > budget.entry_max_chars {
            return Ok(ToolResult::error(format!(
                "body is {body_chars} chars (limit {}). Trim or split the entry before adding.",
                budget.entry_max_chars,
            )));
        }

        match self.store.add(name, kind, description, content) {
            Ok(report) => {
                self.fire_refresh();
                Ok(success_with_budget(
                    format!("added memory entry {name:?} ({kind}). Index updated."),
                    &report,
                    self.store.budget(),
                    body_chars,
                ))
            }
            Err(MemoryError::AlreadyExists { .. }) => Ok(ToolResult::error(format!(
                "memory entry {name:?} already exists; use action=update to modify it"
            ))),
            Err(MemoryError::InvalidName { reason, .. }) => Err(ToolError::invalid_params(reason)),
            Err(MemoryError::InvalidDescription { reason }) => {
                Err(ToolError::invalid_params(reason))
            }
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
        // Same budget gate as `add`: reject before mutating disk.
        let body_chars = content.map_or(0, |c| c.chars().count());
        let budget = self.store.budget();
        if let Some(_c) = content
            && body_chars > budget.entry_max_chars
        {
            return Ok(ToolResult::error(format!(
                "body is {body_chars} chars (limit {}). Trim or split the entry before updating.",
                budget.entry_max_chars,
            )));
        }
        match self.store.update(name, kind, description, content) {
            Ok(report) => {
                self.fire_refresh();
                Ok(success_with_budget(
                    format!("updated memory entry {name:?}."),
                    &report,
                    self.store.budget(),
                    body_chars,
                ))
            }
            Err(MemoryError::NotFound { .. }) => Ok(ToolResult::error(format!(
                "no project-layer memory entry named {name:?}"
            ))),
            Err(MemoryError::InvalidName { reason, .. }) => Err(ToolError::invalid_params(reason)),
            Err(MemoryError::InvalidDescription { reason }) => {
                Err(ToolError::invalid_params(reason))
            }
            Err(other) => Err(map_memory_error(other)),
        }
    }

    fn handle_remove(&self, name: &str, ctx: &SandboxContext) -> Result<ToolResult, ToolError> {
        ctx.check_write_access(self.store.project_dir())?;
        match self.store.remove(name) {
            Ok(report) => {
                self.fire_refresh();
                Ok(success_with_budget(
                    format!("removed memory entry {name:?}."),
                    &report,
                    self.store.budget(),
                    0,
                ))
            }
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
    _body_chars: usize,
) -> ToolResult {
    let mut msg = base_message;
    // Body-length check now lives in `handle_add` / `handle_update`
    // (before the write). Here we only surface the index/file-count
    // capacity nudge.
    if let Some(nudge) = capacity_nudge(report, budget) {
        msg.push_str(&nudge);
    }
    ToolResult::success(msg)
}

/// Map a [`MemoryError`] to the right [`ToolError`] / [`ToolResult`]
/// variant. Issue #8 of PR #76: lossy mapping (everything → `ToolError::Io`)
/// produced misleading error messages and lost variant info.
///
/// Callers handle [`MemoryError::AlreadyExists`] and
/// [`MemoryError::NotFound`] inline because those map to a soft
/// `ToolResult::error` so the LLM can adapt; everything else flows
/// through this function.
fn map_memory_error(err: MemoryError) -> ToolError {
    match err {
        MemoryError::Io { context, source } => ToolError::Io { context, source },
        // Frontmatter / YAML / Type / Description failures are user-
        // input errors from the LLM's perspective, not I/O failures.
        MemoryError::InvalidFrontmatter { path, reason } => {
            ToolError::invalid_params(format!("invalid frontmatter at {path}: {reason}"))
        }
        MemoryError::InvalidType { path, value } => ToolError::invalid_params(format!(
            "invalid memory type {value:?} at {path}: must be one of user, feedback, project, reference",
        )),
        MemoryError::InvalidDescription { reason } => ToolError::invalid_params(reason),
        MemoryError::Yaml { path, source } => {
            ToolError::invalid_params(format!("yaml error at {path}: {source}"))
        }
        // The handlers route AlreadyExists / NotFound / InvalidName
        // before calling here; if any reach this fallthrough it is a
        // bug, but surface a useful message anyway.
        other @ (MemoryError::AlreadyExists { .. }
        | MemoryError::NotFound { .. }
        | MemoryError::InvalidName { .. }) => ToolError::invalid_params(other.to_string()),
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

    /// Regression test for issue #4 of PR #76: after a successful
    /// `add` / `update` / `remove`, the registered refresh callback
    /// fires with the freshly-rendered merged index so the agent loop
    /// can re-inject it into the system prompt.
    #[tokio::test]
    async fn refresh_callback_fires_after_successful_writes() {
        use std::sync::Mutex;
        let dir = tempfile::tempdir().expect("tempdir");
        let store = make_store(dir.path());
        let received: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let received_clone = Arc::clone(&received);
        let cb: super::MemoryRefreshCallback = Arc::new(move |index: &str| {
            received_clone.lock().expect("lock").push(index.to_owned());
        });
        let tool = MemoryTool::with_refresh(Arc::clone(&store), cb);
        let ctx = SandboxContext::test_default();

        // add
        let _ = tool
            .execute(
                serde_json::json!({
                    "action": "add",
                    "name": "topic_one",
                    "type": "project",
                    "description": "first",
                    "content": "body",
                }),
                &ctx,
            )
            .await
            .expect("add");
        // update
        let _ = tool
            .execute(
                serde_json::json!({
                    "action": "update",
                    "name": "topic_one",
                    "description": "renamed",
                }),
                &ctx,
            )
            .await
            .expect("update");
        // remove
        let _ = tool
            .execute(
                serde_json::json!({
                    "action": "remove",
                    "name": "topic_one",
                }),
                &ctx,
            )
            .await
            .expect("remove");

        let received = received.lock().expect("lock");
        assert_eq!(
            received.len(),
            3,
            "expected 3 callback invocations (one per write), got {}: {received:?}",
            received.len(),
        );
        // First snapshot: contains topic_one with "first" desc.
        assert!(received[0].contains("topic_one"));
        assert!(received[0].contains("first"));
        // Second snapshot: still contains topic_one but renamed.
        assert!(received[1].contains("topic_one"));
        assert!(received[1].contains("renamed"));
        // Third snapshot: topic_one removed.
        assert!(!received[2].contains("topic_one"));
    }

    /// Regression test for issue #1 of PR #76: `add(name="MEMORY")`
    /// must be rejected at validation time so the index file cannot
    /// be overwritten by an entry that shares its stem.
    #[tokio::test]
    async fn add_with_reserved_name_is_invalid_params() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = make_store(dir.path());
        let tool = MemoryTool::new(store);
        let ctx = SandboxContext::test_default();
        let err = tool
            .execute(
                serde_json::json!({
                    "action": "add",
                    "name": "MEMORY",
                    "type": "project",
                    "description": "should fail",
                    "content": "body",
                }),
                &ctx,
            )
            .await
            .expect_err("MEMORY must be rejected");
        assert!(
            matches!(err, ToolError::InvalidParams { .. }),
            "expected InvalidParams, got {err:?}",
        );
    }

    /// Regression test for issue #9 of PR #76: oversized bodies must
    /// be rejected BEFORE the store write so the entry file never
    /// lands on disk in an over-budget state.
    #[tokio::test]
    async fn oversized_body_is_rejected_before_write() {
        let dir = tempfile::tempdir().expect("tempdir");
        let small = MemoryBudget {
            index_max_lines: 200,
            entry_max_chars: 10,
            total_files_limit: 50,
        };
        let store = Arc::new(MemoryStore::with_dirs(
            dir.path().join(".tsumugi").join("memory"),
            None,
            small,
        ));
        let tool = MemoryTool::new(Arc::clone(&store));
        let ctx = SandboxContext::test_default();
        let res = tool
            .execute(
                serde_json::json!({
                    "action": "add",
                    "name": "big",
                    "type": "project",
                    "description": "d",
                    "content": "this body is way over the ten-character limit",
                }),
                &ctx,
            )
            .await
            .expect("call");
        assert!(res.is_error, "oversized body must be a soft error result");
        // The entry file must NOT exist on disk.
        let entry_path = dir.path().join(".tsumugi").join("memory").join("big.md");
        assert!(
            !entry_path.exists(),
            "oversized entry file should not have been written",
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
