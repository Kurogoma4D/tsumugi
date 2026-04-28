//! `progress_append` tool: append a single bullet to `progress.md`.

use std::sync::Arc;

use tmg_sandbox::SandboxContext;
use tmg_tools::{Tool, ToolError, ToolResult};
use tokio::sync::Mutex;

use crate::runner::RunRunner;

/// Tool that appends one entry to the run's `progress.md`.
///
/// Holds an `Arc<Mutex<RunRunner>>` rather than the [`ProgressLog`]
/// directly so callers wiring the registry only need one shared handle
/// to the runner; mutual exclusion with `session_summary_save` etc. is
/// taken care of by the same lock.
///
/// [`ProgressLog`]: crate::artifacts::ProgressLog
pub struct ProgressAppendTool {
    runner: Arc<Mutex<RunRunner>>,
}

impl ProgressAppendTool {
    /// Construct a new `progress_append` tool over the given runner.
    pub fn new(runner: Arc<Mutex<RunRunner>>) -> Self {
        Self { runner }
    }
}

impl Tool for ProgressAppendTool {
    fn name(&self) -> &'static str {
        "progress_append"
    }

    fn description(&self) -> &'static str {
        "Append a single bullet to the current run's progress.md timeline. \
         Use this to record what was tried, what worked, and what was decided \
         so future sessions have a quick recap. The entry is appended under \
         the current session header; previous content cannot be overwritten."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "entry": {
                    "type": "string",
                    "description": "A short progress note. One bullet per call."
                }
            },
            "required": ["entry"],
            "additionalProperties": false
        })
    }

    fn execute<'a>(
        &'a self,
        params: serde_json::Value,
        _ctx: &'a SandboxContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + 'a>,
    > {
        // The sandbox parameter is currently advisory: this tool only
        // appends to `progress.md` via the runner-owned log, which
        // lives under the workspace by construction.
        let runner = Arc::clone(&self.runner);
        Box::pin(async move {
            let Some(entry) = params.get("entry").and_then(serde_json::Value::as_str) else {
                return Err(ToolError::invalid_params(
                    "missing required parameter: entry",
                ));
            };
            let trimmed = entry.trim();
            if trimmed.is_empty() {
                return Err(ToolError::invalid_params("`entry` must not be empty"));
            }

            let guard = runner.lock().await;
            // Mirror `session_summary_save`: refuse to append if there
            // is no active session. Without this guard the entry is
            // written under whatever session header happens to be
            // newest in `progress.md`, which is misleading.
            if guard.active_session().is_none() {
                return Err(ToolError::invalid_params("no active session"));
            }
            guard
                .progress_log()
                .append_entry(trimmed)
                .map_err(|e| ToolError::io("appending to progress.md", to_io_error(&e)))?;
            Ok(ToolResult::success("ok"))
        })
    }
}

/// Convert a [`HarnessError`](crate::error::HarnessError) into an
/// `io::Error` for surfacing through [`ToolError::io`]. Tool errors do
/// not have a dedicated `Harness` variant and adding one would force
/// every tool consumer to learn about harness internals; the textual
/// `Display` representation is sufficient for the LLM.
fn to_io_error(err: &crate::error::HarnessError) -> std::io::Error {
    std::io::Error::other(err.to_string())
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use crate::store::RunStore;

    fn make_runner() -> (tempfile::TempDir, Arc<Mutex<RunRunner>>) {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let store = Arc::new(RunStore::new(tmp.path().join("runs")));
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut runner = RunRunner::new(run, store);
        let _ = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        (tmp, Arc::new(Mutex::new(runner)))
    }

    #[tokio::test]
    async fn append_appends_one_bullet() {
        let (_tmp, runner) = make_runner();
        let tool = ProgressAppendTool::new(Arc::clone(&runner));
        let ctx = SandboxContext::test_default();
        let res = tool
            .execute(serde_json::json!({ "entry": "first note" }), &ctx)
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(!res.is_error);

        let path = runner.lock().await.progress_log().path().to_path_buf();
        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
        assert!(content.contains("- first note\n"), "{content}");
    }

    #[tokio::test]
    async fn missing_entry_param_is_error() {
        let (_tmp, runner) = make_runner();
        let tool = ProgressAppendTool::new(runner);
        let ctx = SandboxContext::test_default();
        let res = tool.execute(serde_json::json!({}), &ctx).await;
        assert!(matches!(res, Err(ToolError::InvalidParams { .. })));
    }

    #[tokio::test]
    async fn empty_entry_is_error() {
        let (_tmp, runner) = make_runner();
        let tool = ProgressAppendTool::new(runner);
        let ctx = SandboxContext::test_default();
        let res = tool
            .execute(serde_json::json!({ "entry": "   \n" }), &ctx)
            .await;
        assert!(matches!(res, Err(ToolError::InvalidParams { .. })));
    }

    /// `progress_append` mirrors `session_summary_save` in refusing to
    /// run when there is no active session. This catches mis-wirings
    /// where the tool is registered before `begin_session()` is called.
    #[tokio::test]
    async fn no_active_session_is_error() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let store = Arc::new(RunStore::new(tmp.path().join("runs")));
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));
        // Note: no `begin_session()` call.
        let runner = Arc::new(Mutex::new(RunRunner::new(run, store)));

        let tool = ProgressAppendTool::new(runner);
        let ctx = SandboxContext::test_default();
        let res = tool
            .execute(serde_json::json!({ "entry": "should fail" }), &ctx)
            .await;
        assert!(
            matches!(res, Err(ToolError::InvalidParams { ref message }) if message.contains("no active session")),
            "expected `no active session` error, got {res:?}",
        );
    }

    /// API-level append-only: the tool exposes only `execute` with an
    /// `entry` parameter. There is no `clear`, `truncate`, or
    /// `overwrite` API surface, and calling `execute` repeatedly only
    /// grows the file.
    #[tokio::test]
    async fn cannot_overwrite_via_tool_api() {
        let (_tmp, runner) = make_runner();
        let tool = ProgressAppendTool::new(Arc::clone(&runner));

        let path = runner.lock().await.progress_log().path().to_path_buf();
        let initial = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));

        let ctx = SandboxContext::test_default();
        for i in 0..10 {
            tool.execute(serde_json::json!({ "entry": format!("note {i}") }), &ctx)
                .await
                .unwrap_or_else(|e| panic!("{e}"));
        }

        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
        // The original initial header must still appear as a prefix; the
        // tool cannot replace it.
        assert!(
            content.starts_with(&initial),
            "tool was able to rewrite the file prefix; before:\n{initial}\nafter:\n{content}"
        );
        // Length is monotonically increasing.
        assert!(content.len() > initial.len());
        // Each entry is present exactly once.
        for i in 0..10 {
            let needle = format!("- note {i}\n");
            assert_eq!(
                content.matches(&needle).count(),
                1,
                "expected exactly one occurrence of `{needle}`"
            );
        }
    }

    /// Disk-level append-only: even if a caller bypassed the tool API
    /// and used the `ProgressLog` directly, every public mutator opens
    /// the file with `append(true)` so the leading bytes are never
    /// rewritten.
    #[tokio::test]
    async fn cannot_overwrite_at_disk_level() {
        let (_tmp, runner) = make_runner();
        let path = runner.lock().await.progress_log().path().to_path_buf();
        let before = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));

        // Write 100 entries directly via the runner's progress log.
        for i in 0..100_u32 {
            runner
                .lock()
                .await
                .progress_log()
                .append_entry(&format!("entry {i}"))
                .unwrap_or_else(|e| panic!("{e}"));
        }

        let after = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
        assert!(after.starts_with(&before), "disk-level overwrite detected");
    }
}
