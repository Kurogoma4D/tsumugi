//! `feature_list_read` tool: return the full `features.json`.
//!
//! Registered **only** for harnessed runs. The companion
//! [`FeatureListMarkPassingTool`](super::FeatureListMarkPassingTool) is
//! the only writer; this tool is read-only and never mutates the file.

use std::sync::Arc;

use tmg_sandbox::SandboxContext;
use tmg_tools::{Tool, ToolError, ToolResult};
use tokio::sync::Mutex;

use crate::runner::RunRunner;

/// Tool that returns the full contents of the run's `features.json`.
///
/// Holds an `Arc<Mutex<RunRunner>>` so the lock is shared with the
/// other Run-scoped tools; this is consistent with
/// `progress_append` / `session_summary_save`.
pub struct FeatureListReadTool {
    runner: Arc<Mutex<RunRunner>>,
}

impl FeatureListReadTool {
    /// Construct the tool over the given runner.
    pub fn new(runner: Arc<Mutex<RunRunner>>) -> Self {
        Self { runner }
    }
}

impl Tool for FeatureListReadTool {
    fn name(&self) -> &'static str {
        "feature_list_read"
    }

    fn description(&self) -> &'static str {
        "Return the full features.json for the current harnessed run, \
         including every feature's id, category, description, steps, and \
         current `passes` flag. Use this to discover what work remains; \
         use `feature_list_mark_passing` to mark a feature passing once \
         its acceptance criteria are met."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        })
    }

    fn execute<'a>(
        &'a self,
        _params: serde_json::Value,
        _ctx: &'a SandboxContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + 'a>,
    > {
        // The sandbox parameter is currently advisory: this tool only
        // reads the run-internal `features.json` via the runner.
        let runner = Arc::clone(&self.runner);
        Box::pin(async move {
            let features_handle = {
                let guard = runner.lock().await;
                guard.features().clone()
            };
            let features = features_handle.read().map_err(|e| {
                ToolError::io(
                    "reading features.json",
                    std::io::Error::other(e.to_string()),
                )
            })?;
            let json = serde_json::to_string_pretty(&features)
                .map_err(|e| ToolError::json("serializing features.json", e))?;
            Ok(ToolResult::success(json))
        })
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use crate::run::RunScope;
    use crate::store::RunStore;

    fn make_harnessed_runner_with_features(
        features_json: &str,
    ) -> (tempfile::TempDir, Arc<Mutex<RunRunner>>) {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let store = Arc::new(RunStore::new(tmp.path().join("runs")));
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let mut run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));
        run.scope = RunScope::harnessed("test-workflow", Some(10));
        run.workflow_id = Some("test-workflow".to_owned());
        store.save(&run).unwrap_or_else(|e| panic!("{e}"));

        let features_path = store.features_file(&run.id);
        std::fs::write(&features_path, features_json).unwrap_or_else(|e| panic!("{e}"));

        let mut runner = RunRunner::new(run, store);
        let _ = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        (tmp, Arc::new(Mutex::new(runner)))
    }

    fn sample_json() -> &'static str {
        r#"{
  "features": [
    {
      "id": "feat-001",
      "category": "auth",
      "description": "Login",
      "steps": ["A", "B"],
      "passes": false
    }
  ]
}"#
    }

    #[tokio::test]
    async fn returns_full_features_json() {
        let (_tmp, runner) = make_harnessed_runner_with_features(sample_json());
        let tool = FeatureListReadTool::new(runner);
        let ctx = SandboxContext::test_default();
        let res = tool
            .execute(serde_json::json!({}), &ctx)
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(!res.is_error);
        let parsed: serde_json::Value =
            serde_json::from_str(&res.output).unwrap_or_else(|e| panic!("{e}"));
        let features = parsed
            .get("features")
            .and_then(serde_json::Value::as_array)
            .unwrap_or_else(|| panic!("expected `features` array in {parsed}"));
        assert_eq!(features.len(), 1);
        assert_eq!(
            features[0].get("id").and_then(serde_json::Value::as_str),
            Some("feat-001")
        );
    }

    #[tokio::test]
    async fn missing_file_surfaces_io_error() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let store = Arc::new(RunStore::new(tmp.path().join("runs")));
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));
        // Note: features.json is intentionally not created.
        let mut runner = RunRunner::new(run, store);
        let _ = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        let runner = Arc::new(Mutex::new(runner));

        let tool = FeatureListReadTool::new(runner);
        let ctx = SandboxContext::test_default();
        let res = tool.execute(serde_json::json!({}), &ctx).await;
        assert!(matches!(res, Err(ToolError::Io { .. })), "got {res:?}");
    }
}
