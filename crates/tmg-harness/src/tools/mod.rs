//! Run-scoped [`Tool`](tmg_tools::Tool) implementations.
//!
//! These tools differ from those in [`tmg_tools::default_registry`]
//! because they hold a reference to the active [`RunRunner`]: they
//! mutate the run's `progress.md` / active `Session` rather than
//! interacting with the workspace filesystem directly.
//!
//! Wiring: the CLI registers them on the global [`ToolRegistry`] after
//! creating the [`RunRunner`]; they are intentionally **not** included
//! in `default_registry()` so a Run-less code path (e.g. `--prompt`
//! one-shot mode) does not accidentally surface them to the LLM.
//!
//! ## Scope-gated tools
//!
//! Two tools are gated to harnessed runs only:
//!
//! - [`FeatureListReadTool`] — return the full `features.json`.
//! - [`FeatureListMarkPassingTool`] — flip a feature's `passes` to
//!   `true`.
//!
//! The CLI uses [`register_run_tools`] to register the appropriate set
//! based on `RunRunner::scope()`: ad-hoc runs see the three core
//! tools, harnessed runs additionally see the two feature tools.

mod feature_list_mark_passing;
mod feature_list_read;
mod progress_append;
mod run_tool_provider;
mod session_bootstrap;
mod session_summary_save;

use std::sync::Arc;

use tmg_tools::ToolRegistry;
use tokio::sync::Mutex;

pub use feature_list_mark_passing::FeatureListMarkPassingTool;
pub use feature_list_read::FeatureListReadTool;
pub use progress_append::ProgressAppendTool;
pub use run_tool_provider::RunRunnerToolProvider;
pub use session_bootstrap::{
    BootstrapPayload, InitScriptStatus, SessionBootstrapTool, SmokeTestResult,
};
pub use session_summary_save::SessionSummarySaveTool;

use crate::runner::RunRunner;

/// Register the Run-scoped tools onto `registry` according to the
/// runner's scope.
///
/// Always registered:
///
/// - [`ProgressAppendTool`]
/// - [`SessionBootstrapTool`]
/// - [`SessionSummarySaveTool`]
///
/// Registered only when `runner.scope()` is
/// [`RunScope::Harnessed`](crate::run::RunScope::Harnessed):
///
/// - [`FeatureListReadTool`]
/// - [`FeatureListMarkPassingTool`]
///
/// This single helper is the authoritative wiring point used by both
/// the CLI and the unit tests in this crate, ensuring the
/// "harnessed-only" promise is upheld in one place rather than
/// sprinkled across callers.
pub async fn register_run_tools(registry: &mut ToolRegistry, runner: Arc<Mutex<RunRunner>>) {
    let is_harnessed = runner.lock().await.is_harnessed();

    registry.register(ProgressAppendTool::new(Arc::clone(&runner)));
    registry.register(SessionBootstrapTool::new(Arc::clone(&runner)));
    registry.register(SessionSummarySaveTool::new(Arc::clone(&runner)));

    if is_harnessed {
        registry.register(FeatureListReadTool::new(Arc::clone(&runner)));
        registry.register(FeatureListMarkPassingTool::new(Arc::clone(&runner)));
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use crate::run::RunScope;
    use crate::store::RunStore;

    fn make_runner_with_scope(scope: &RunScope) -> (tempfile::TempDir, Arc<Mutex<RunRunner>>) {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let store = Arc::new(RunStore::new(tmp.path().join("runs")));
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let mut run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));
        run.scope = scope.clone();
        if let RunScope::Harnessed { workflow_id, .. } = scope {
            run.workflow_id = Some(workflow_id.clone());
        }
        store.save(&run).unwrap_or_else(|e| panic!("{e}"));

        let mut runner = RunRunner::new(run, store);
        let _ = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        (tmp, Arc::new(Mutex::new(runner)))
    }

    fn registered_names(registry: &ToolRegistry) -> Vec<String> {
        registry.tool_names().map(str::to_owned).collect()
    }

    /// Acceptance: ad-hoc registration does NOT include the
    /// `feature_list_*` tools.
    #[tokio::test]
    async fn ad_hoc_registration_omits_feature_tools() {
        let (_tmp, runner) = make_runner_with_scope(&RunScope::AdHoc);
        let mut registry = ToolRegistry::new();
        register_run_tools(&mut registry, runner).await;

        let names = registered_names(&registry);
        assert!(names.contains(&"progress_append".to_owned()));
        assert!(names.contains(&"session_bootstrap".to_owned()));
        assert!(names.contains(&"session_summary_save".to_owned()));
        assert!(!names.contains(&"feature_list_read".to_owned()));
        assert!(!names.contains(&"feature_list_mark_passing".to_owned()));
    }

    /// Acceptance: harnessed registration DOES include the
    /// `feature_list_*` tools.
    #[tokio::test]
    async fn harnessed_registration_includes_feature_tools() {
        let scope = RunScope::Harnessed {
            workflow_id: "test-wf".to_owned(),
            max_sessions: None,
        };
        let (_tmp, runner) = make_runner_with_scope(&scope);
        let mut registry = ToolRegistry::new();
        register_run_tools(&mut registry, runner).await;

        let names = registered_names(&registry);
        assert!(names.contains(&"progress_append".to_owned()));
        assert!(names.contains(&"feature_list_read".to_owned()));
        assert!(names.contains(&"feature_list_mark_passing".to_owned()));
    }
}
