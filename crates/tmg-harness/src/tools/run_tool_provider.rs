//! Implementation of [`RunToolProvider`] backed by an
//! `Arc<Mutex<RunRunner>>`.
//!
//! This is the wiring point that lets the `tmg-agents` crate stay
//! independent of `tmg-harness`: the provider trait lives in
//! `tmg-agents`, and `tmg-harness` supplies an implementation that
//! constructs the Run-aware tools (`progress_append`,
//! `feature_list_read`, `feature_list_mark_passing`) on demand from a
//! shared `RunRunner` handle.
//!
//! The provider is scope-aware: `feature_list_*` tools are only
//! registered when the active run is harnessed, mirroring the policy
//! enforced by [`crate::tools::register_run_tools`].

use std::sync::Arc;

use tmg_agents::RunToolProvider;
use tmg_tools::ToolRegistry;
use tokio::sync::Mutex;

use crate::run::RunScope;
use crate::runner::RunRunner;
use crate::tools::{FeatureListMarkPassingTool, FeatureListReadTool, ProgressAppendTool};

/// `RunToolProvider` that constructs Run-aware tools from a shared
/// `RunRunner` handle.
///
/// Cloning is cheap: only the `Arc` to the runner and the cached
/// scope flag are stored.
///
/// **Scope policy:** the cached `is_harnessed` flag is captured at
/// construction time. `progress_append` is registered for both ad-hoc
/// and harnessed runs; `feature_list_read` and
/// `feature_list_mark_passing` are registered **only** for harnessed
/// runs. This mirrors the behaviour of
/// [`crate::tools::register_run_tools`] so the main agent and any
/// spawned subagents see the same scope-gated tool set.
#[derive(Clone)]
pub struct RunRunnerToolProvider {
    runner: Arc<Mutex<RunRunner>>,
    is_harnessed: bool,
}

impl RunRunnerToolProvider {
    /// Construct a provider from a shared [`RunRunner`] handle.
    ///
    /// The harnessed/ad-hoc flag is captured eagerly so that
    /// `register_run_tool` (which is synchronous) does not have to
    /// acquire the runner lock on the hot path.
    #[must_use]
    pub async fn new(runner: Arc<Mutex<RunRunner>>) -> Self {
        let is_harnessed = matches!(runner.lock().await.scope(), RunScope::Harnessed { .. });
        Self {
            runner,
            is_harnessed,
        }
    }

    /// Construct a provider with an explicit `is_harnessed` flag.
    ///
    /// Prefer [`Self::new`] in production code; this constructor is
    /// useful in tests where the runner's scope is known at the call
    /// site without paying for an async lock.
    #[must_use]
    pub fn with_scope(runner: Arc<Mutex<RunRunner>>, is_harnessed: bool) -> Self {
        Self {
            runner,
            is_harnessed,
        }
    }

    /// Whether this provider was constructed against a harnessed run.
    #[must_use]
    pub fn is_harnessed(&self) -> bool {
        self.is_harnessed
    }
}

impl RunToolProvider for RunRunnerToolProvider {
    fn register_run_tool(&self, registry: &mut ToolRegistry, name: &str) -> bool {
        match name {
            "progress_append" => {
                registry.register(ProgressAppendTool::new(Arc::clone(&self.runner)));
                true
            }
            "feature_list_read" if self.is_harnessed => {
                registry.register(FeatureListReadTool::new(Arc::clone(&self.runner)));
                true
            }
            "feature_list_mark_passing" if self.is_harnessed => {
                registry.register(FeatureListMarkPassingTool::new(Arc::clone(&self.runner)));
                true
            }
            // Anything else (including the harnessed-only tools on an
            // ad-hoc run) is intentionally not registered: a `false`
            // return tells the caller to skip the name silently.
            _ => false,
        }
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

    /// On a harnessed run the provider registers all three Run-aware
    /// tools when asked.
    #[tokio::test]
    async fn harnessed_provider_registers_all_run_aware_tools() {
        let scope = RunScope::Harnessed {
            workflow_id: "wf".to_owned(),
            max_sessions: None,
        };
        let (_tmp, runner) = make_runner_with_scope(&scope);
        let provider = RunRunnerToolProvider::new(runner).await;

        let mut registry = ToolRegistry::new();
        assert!(provider.register_run_tool(&mut registry, "progress_append"));
        assert!(provider.register_run_tool(&mut registry, "feature_list_read"));
        assert!(provider.register_run_tool(&mut registry, "feature_list_mark_passing"));

        assert!(registry.get("progress_append").is_some());
        assert!(registry.get("feature_list_read").is_some());
        assert!(registry.get("feature_list_mark_passing").is_some());
    }

    /// On an ad-hoc run the provider refuses to register
    /// `feature_list_*` tools (returns `false`) but still registers
    /// `progress_append`. This mirrors the policy enforced by
    /// [`crate::tools::register_run_tools`].
    #[tokio::test]
    async fn ad_hoc_provider_skips_feature_list_tools() {
        let (_tmp, runner) = make_runner_with_scope(&RunScope::AdHoc);
        let provider = RunRunnerToolProvider::new(runner).await;

        let mut registry = ToolRegistry::new();
        assert!(provider.register_run_tool(&mut registry, "progress_append"));
        assert!(!provider.register_run_tool(&mut registry, "feature_list_read"));
        assert!(!provider.register_run_tool(&mut registry, "feature_list_mark_passing"));

        assert!(registry.get("progress_append").is_some());
        assert!(registry.get("feature_list_read").is_none());
        assert!(registry.get("feature_list_mark_passing").is_none());
    }

    /// Unknown names always return `false`.
    #[tokio::test]
    async fn unknown_name_returns_false() {
        let scope = RunScope::Harnessed {
            workflow_id: "wf".to_owned(),
            max_sessions: None,
        };
        let (_tmp, runner) = make_runner_with_scope(&scope);
        let provider = RunRunnerToolProvider::new(runner).await;

        let mut registry = ToolRegistry::new();
        assert!(!provider.register_run_tool(&mut registry, "file_read"));
        assert!(!provider.register_run_tool(&mut registry, "totally_unknown"));
        assert_eq!(registry.tool_names().count(), 0);
    }
}
