//! `session_summary_save` tool: persist a summary string into the
//! active session's `session_NNN.json`.

use std::sync::Arc;

use tmg_sandbox::SandboxContext;
use tmg_tools::{Tool, ToolError, ToolResult};
use tokio::sync::Mutex;

use crate::runner::RunRunner;

/// Tool that updates the active session's `summary` field and persists
/// it to `session_NNN.json`.
pub struct SessionSummarySaveTool {
    runner: Arc<Mutex<RunRunner>>,
}

impl SessionSummarySaveTool {
    /// Construct the tool over the given runner.
    pub fn new(runner: Arc<Mutex<RunRunner>>) -> Self {
        Self { runner }
    }
}

impl Tool for SessionSummarySaveTool {
    fn name(&self) -> &'static str {
        "session_summary_save"
    }

    fn description(&self) -> &'static str {
        "Persist a human-readable summary of the current session. \
         Call this once near the end of the session — its content is \
         saved to session_NNN.json and surfaced as `progress_summary` \
         in future session_bootstrap calls."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "The session summary text."
                }
            },
            "required": ["summary"],
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
        // mutates the run's session log via the runner, which lives
        // under the workspace by construction. Accepting it keeps the
        // [`Tool`] signature uniform.
        let runner = Arc::clone(&self.runner);
        Box::pin(async move {
            let Some(summary) = params.get("summary").and_then(serde_json::Value::as_str) else {
                return Err(ToolError::invalid_params(
                    "missing required parameter: summary",
                ));
            };

            let mut guard = runner.lock().await;
            let Some(session) = guard.active_session_mut() else {
                return Err(ToolError::invalid_params(
                    "no active session to save a summary for",
                ));
            };
            summary.clone_into(&mut session.summary);
            guard.save_active_session().map_err(|e| {
                ToolError::io(
                    "saving session_NNN.json",
                    std::io::Error::other(e.to_string()),
                )
            })?;
            Ok(ToolResult::success("ok"))
        })
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use crate::session::{Session, SessionEndTrigger};
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
    async fn save_summary_persists_to_session_log() {
        let (_tmp, runner) = make_runner();
        let tool = SessionSummarySaveTool::new(Arc::clone(&runner));
        let ctx = SandboxContext::test_default();
        let res = tool
            .execute(serde_json::json!({ "summary": "wrote a thing" }), &ctx)
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(!res.is_error);

        let session_path = {
            let guard = runner.lock().await;
            guard.session_log().session_path(1)
        };
        let raw = std::fs::read_to_string(&session_path).unwrap_or_else(|e| panic!("{e}"));
        assert!(raw.contains("\"summary\": \"wrote a thing\""), "{raw}");
    }

    /// SPEC §9.4 round-trip: every documented field is present after a
    /// save, and the file deserialises back into an equal `Session`.
    #[tokio::test]
    async fn session_log_matches_spec_schema() {
        let (_tmp, runner) = make_runner();

        // Set every field on the live session before saving.
        {
            let mut guard = runner.lock().await;
            let session = guard
                .active_session_mut()
                .unwrap_or_else(|| panic!("active session"));
            session.tool_calls_count = 12;
            session.files_modified.push("src/foo.rs".to_owned());
            session.files_modified.push("src/bar.rs".to_owned());
            session.context_usage_peak = 0.62;
            session.next_session_hint = Some("look at integration tests".to_owned());
        }

        let tool = SessionSummarySaveTool::new(Arc::clone(&runner));
        let ctx = SandboxContext::test_default();
        tool.execute(
            serde_json::json!({ "summary": "Implemented foo and bar" }),
            &ctx,
        )
        .await
        .unwrap_or_else(|e| panic!("{e}"));

        // End the active session so `trigger` and `ended_at` are
        // populated, then re-save to flush them to disk before checking
        // the schema.
        {
            let mut guard = runner.lock().await;
            if let Some(s) = guard.active_session_mut() {
                s.end(SessionEndTrigger::Completed);
            }
            guard
                .save_active_session()
                .unwrap_or_else(|e| panic!("{e}"));
        }

        let path = runner.lock().await.session_log().session_path(1);
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
        for key in [
            "session_number",
            "started_at",
            "ended_at",
            "trigger",
            "summary",
            "tool_calls_count",
            "files_modified",
            "features_marked_passing",
            "context_usage_peak",
            "next_session_hint",
        ] {
            assert!(raw.contains(key), "missing key `{key}` in {raw}");
        }

        let loaded: Session = {
            let guard = runner.lock().await;
            guard
                .session_log()
                .load(1)
                .unwrap_or_else(|e| panic!("{e}"))
                .unwrap_or_else(|| panic!("session 1"))
        };
        assert_eq!(loaded.summary, "Implemented foo and bar");
        assert_eq!(loaded.tool_calls_count, 12);
        assert_eq!(loaded.files_modified, vec!["src/foo.rs", "src/bar.rs"]);
        assert!((loaded.context_usage_peak - 0.62).abs() < f64::EPSILON);
        assert_eq!(
            loaded.next_session_hint.as_deref(),
            Some("look at integration tests")
        );
        assert_eq!(loaded.end_trigger, Some(SessionEndTrigger::Completed));
    }

    #[tokio::test]
    async fn missing_summary_param_is_error() {
        let (_tmp, runner) = make_runner();
        let tool = SessionSummarySaveTool::new(runner);
        let ctx = SandboxContext::test_default();
        let res = tool.execute(serde_json::json!({}), &ctx).await;
        assert!(matches!(res, Err(ToolError::InvalidParams { .. })));
    }
}
