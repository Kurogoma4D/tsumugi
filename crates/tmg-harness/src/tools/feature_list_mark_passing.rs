//! `feature_list_mark_passing` tool: flip a feature's `passes` flag to
//! `true`.
//!
//! Registered **only** for harnessed runs. Schema-violating mutations
//! (adding / removing entries, modifying `id`/`category`/`description`/
//! `steps`) are impossible through this tool — the underlying
//! [`FeatureList::mark_passing`](crate::artifacts::FeatureList::mark_passing)
//! accepts a single `feature_id` parameter and refuses unknown ids.

use std::sync::Arc;

use tmg_tools::{Tool, ToolError, ToolResult};
use tokio::sync::Mutex;

use crate::error::HarnessError;
use crate::runner::RunRunner;

/// Tool that marks a feature as passing in `features.json`.
pub struct FeatureListMarkPassingTool {
    runner: Arc<Mutex<RunRunner>>,
}

impl FeatureListMarkPassingTool {
    /// Construct the tool over the given runner.
    pub fn new(runner: Arc<Mutex<RunRunner>>) -> Self {
        Self { runner }
    }
}

impl Tool for FeatureListMarkPassingTool {
    fn name(&self) -> &'static str {
        "feature_list_mark_passing"
    }

    fn description(&self) -> &'static str {
        "Mark a single feature as passing in features.json by id. \
         Only the `passes` field of the given feature is modified; all \
         other fields and the order of entries are preserved. Unknown \
         ids are rejected. Idempotent: calling on an already-passing \
         feature is a no-op."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "feature_id": {
                    "type": "string",
                    "description": "The id of the feature to mark passing (e.g. \"feat-001\")."
                }
            },
            "required": ["feature_id"],
            "additionalProperties": false
        })
    }

    fn execute(
        &self,
        params: serde_json::Value,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + '_>,
    > {
        let runner = Arc::clone(&self.runner);
        Box::pin(async move {
            let Some(feature_id) = params.get("feature_id").and_then(serde_json::Value::as_str)
            else {
                return Err(ToolError::invalid_params(
                    "missing required parameter: feature_id",
                ));
            };
            let trimmed = feature_id.trim();
            if trimmed.is_empty() {
                return Err(ToolError::invalid_params("`feature_id` must not be empty"));
            }

            // Take a cheap clone of the `FeatureList` handle so we can
            // touch the disk without holding the runner lock. The
            // session-side side-effect (`features_marked_passing`) is
            // intentionally deferred until after the on-disk write
            // succeeds: recording the id before the write would leave
            // the session log claiming the feature was marked even when
            // the underlying file was not modified (unknown id, I/O
            // error). See the `unknown_id_*` regression tests below.
            let features_handle = {
                let guard = runner.lock().await;
                guard.features().clone()
            };

            match features_handle.mark_passing(trimmed) {
                Ok(()) => {
                    // Re-acquire the lock to record the side-effect.
                    // Done in a second critical section because the
                    // file-write above can block long enough that we do
                    // not want it to hold the runner mutex.
                    let mut guard = runner.lock().await;
                    if let Some(session) = guard.active_session_mut()
                        && !session.features_marked_passing.iter().any(|x| x == trimmed)
                    {
                        session.features_marked_passing.push(trimmed.to_owned());
                    }
                    Ok(ToolResult::success(format!(
                        "marked feature `{trimmed}` as passing"
                    )))
                }
                Err(HarnessError::UnknownFeatureId { feature_id }) => Ok(ToolResult::error(
                    format!("unknown feature id: {feature_id}"),
                )),
                Err(other) => Err(ToolError::io(
                    "marking feature passing",
                    std::io::Error::other(other.to_string()),
                )),
            }
        })
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use crate::run::RunScope;
    use crate::store::RunStore;

    fn sample_json() -> &'static str {
        r#"{
  "features": [
    {
      "id": "feat-001",
      "category": "auth",
      "description": "Login",
      "steps": ["A", "B"],
      "passes": false
    },
    {
      "id": "feat-002",
      "category": "auth",
      "description": "Logout",
      "steps": ["C"],
      "passes": false
    }
  ]
}"#
    }

    fn make_runner_with_features(json: &str) -> (tempfile::TempDir, Arc<Mutex<RunRunner>>) {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let store = Arc::new(RunStore::new(tmp.path().join("runs")));
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let mut run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));
        run.scope = RunScope::Harnessed {
            workflow_id: "wf".to_owned(),
            max_sessions: None,
        };
        store.save(&run).unwrap_or_else(|e| panic!("{e}"));

        std::fs::write(store.features_file(&run.id), json).unwrap_or_else(|e| panic!("{e}"));

        let mut runner = RunRunner::new(run, store);
        let _ = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        (tmp, Arc::new(Mutex::new(runner)))
    }

    #[tokio::test]
    async fn happy_path_flips_only_target_feature() {
        let (_tmp, runner) = make_runner_with_features(sample_json());
        let tool = FeatureListMarkPassingTool::new(Arc::clone(&runner));
        let res = tool
            .execute(serde_json::json!({ "feature_id": "feat-001" }))
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(!res.is_error);

        let features = {
            let guard = runner.lock().await;
            guard.features().read().unwrap_or_else(|e| panic!("{e}"))
        };
        assert!(features.features[0].passes);
        assert!(!features.features[1].passes);
    }

    #[tokio::test]
    async fn updates_active_session_features_marked_passing() {
        let (_tmp, runner) = make_runner_with_features(sample_json());
        let tool = FeatureListMarkPassingTool::new(Arc::clone(&runner));
        tool.execute(serde_json::json!({ "feature_id": "feat-001" }))
            .await
            .unwrap_or_else(|e| panic!("{e}"));

        let guard = runner.lock().await;
        let session = guard
            .active_session()
            .unwrap_or_else(|| panic!("active session"));
        assert_eq!(session.features_marked_passing, vec!["feat-001".to_owned()]);
    }

    #[tokio::test]
    async fn duplicate_calls_do_not_double_record_in_session() {
        let (_tmp, runner) = make_runner_with_features(sample_json());
        let tool = FeatureListMarkPassingTool::new(Arc::clone(&runner));
        for _ in 0..3 {
            tool.execute(serde_json::json!({ "feature_id": "feat-001" }))
                .await
                .unwrap_or_else(|e| panic!("{e}"));
        }
        let guard = runner.lock().await;
        let session = guard
            .active_session()
            .unwrap_or_else(|| panic!("active session"));
        assert_eq!(session.features_marked_passing, vec!["feat-001".to_owned()]);
    }

    #[tokio::test]
    async fn unknown_id_returns_tool_error_result() {
        let (_tmp, runner) = make_runner_with_features(sample_json());
        let tool = FeatureListMarkPassingTool::new(Arc::clone(&runner));
        let res = tool
            .execute(serde_json::json!({ "feature_id": "feat-999" }))
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(res.is_error, "expected is_error=true, got {res:?}");
        assert!(res.output.contains("feat-999"), "{}", res.output);

        // The side-effect on the active session must NOT have been
        // applied: the on-disk write was rejected, so the session log
        // would otherwise lie about features being marked. Regression
        // test for the ordering bug between the runner-side bookkeeping
        // and the file write.
        let guard = runner.lock().await;
        let session = guard
            .active_session()
            .unwrap_or_else(|| panic!("active session"));
        assert!(
            session.features_marked_passing.is_empty(),
            "features_marked_passing should remain empty after a failed mark; got {:?}",
            session.features_marked_passing,
        );
    }

    #[tokio::test]
    async fn missing_param_is_invalid_params() {
        let (_tmp, runner) = make_runner_with_features(sample_json());
        let tool = FeatureListMarkPassingTool::new(runner);
        let res = tool.execute(serde_json::json!({})).await;
        assert!(matches!(res, Err(ToolError::InvalidParams { .. })));
    }

    #[tokio::test]
    async fn empty_param_is_invalid_params() {
        let (_tmp, runner) = make_runner_with_features(sample_json());
        let tool = FeatureListMarkPassingTool::new(runner);
        let res = tool
            .execute(serde_json::json!({ "feature_id": "  " }))
            .await;
        assert!(matches!(res, Err(ToolError::InvalidParams { .. })));
    }

    /// Schema invariance through the tool API: the only public way to
    /// invoke the underlying `mark_passing` is by passing a `feature_id`.
    /// There is no way to add/remove features or alter any other field.
    #[tokio::test]
    async fn schema_invariance_via_tool_api() {
        let (_tmp, runner) = make_runner_with_features(sample_json());
        let tool = FeatureListMarkPassingTool::new(Arc::clone(&runner));

        let before = {
            let guard = runner.lock().await;
            guard.features().read().unwrap_or_else(|e| panic!("{e}"))
        };

        // Try smuggling extra fields in the params.
        tool.execute(serde_json::json!({
            "feature_id": "feat-001",
            "evil": "smuggled",
            "passes": false,
        }))
        .await
        .unwrap_or_else(|e| panic!("{e}"));

        let after = {
            let guard = runner.lock().await;
            guard.features().read().unwrap_or_else(|e| panic!("{e}"))
        };
        // Same number of features.
        assert_eq!(before.features.len(), after.features.len());
        // Same ids in the same order.
        let before_ids: Vec<&str> = before.features.iter().map(|f| f.id.as_str()).collect();
        let after_ids: Vec<&str> = after.features.iter().map(|f| f.id.as_str()).collect();
        assert_eq!(before_ids, after_ids);
        // Categories / descriptions / steps unchanged.
        for (b, a) in before.features.iter().zip(after.features.iter()) {
            assert_eq!(b.category, a.category);
            assert_eq!(b.description, a.description);
            assert_eq!(b.steps, a.steps);
        }
    }
}
