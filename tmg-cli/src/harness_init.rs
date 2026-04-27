//! Startup integration with the [`tmg_harness`] crate.
//!
//! Encapsulates the run-resolution logic that runs at the start of
//! [`run_tui`](crate::run_tui): given a [`HarnessConfig`] and a
//! workspace path, return a [`Run`] that the CLI can hand to the
//! [`RunRunner`]. The function resumes the most recent unfinished run
//! when configured to and falls back to creating a fresh ad-hoc run
//! otherwise.
//!
//! Keeping this separate from `main.rs` makes the policy unit-testable
//! without spinning up an `AgentLoop` or a TUI.

use std::sync::Arc;

use tmg_harness::{HarnessError, Run, RunStore};

use crate::config::HarnessConfig;

/// Resolve the active [`Run`] at startup, persisting any state changes
/// through the supplied [`RunStore`].
///
/// Behaviour:
///
/// - When `harness.auto_resume_on_start` is `true`, ask the store for
///   the most recent resumable run. If one exists, load it and return.
/// - Otherwise (no resumable run, or auto-resume disabled), create a
///   new ad-hoc run rooted at `workspace_path`.
///
/// In both cases the run's `last_session_at` and `session_count` are
/// updated and persisted via the [`RunRunner`](tmg_harness::RunRunner)
/// elsewhere; this function only handles the load-vs-create choice.
pub fn resolve_startup_run(
    harness: &HarnessConfig,
    store: &Arc<RunStore>,
    workspace_path: std::path::PathBuf,
) -> Result<Run, HarnessError> {
    if harness.auto_resume_on_start {
        if let Some(summary) = store.latest_resumable()? {
            return store.load(&summary.id);
        }
    }
    store.create_ad_hoc(workspace_path, None)
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use tmg_harness::{RunStatus, RunStore};

    fn make_store() -> (tempfile::TempDir, Arc<RunStore>) {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let store = Arc::new(RunStore::new(tmp.path().join("runs")));
        (tmp, store)
    }

    #[test]
    fn auto_resume_true_with_existing_run_resumes_it() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));

        // Seed an existing run that is still in Running status.
        let seeded = store
            .create_ad_hoc(workspace.clone(), None)
            .unwrap_or_else(|e| panic!("{e}"));

        let cfg = HarnessConfig {
            runs_dir: store.runs_dir().to_path_buf(),
            auto_resume_on_start: true,
        };
        let resolved =
            resolve_startup_run(&cfg, &store, workspace.clone()).unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(resolved.id, seeded.id, "should resume the existing run");
    }

    #[test]
    fn auto_resume_false_creates_new_run_even_when_run_exists() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));

        let seeded = store
            .create_ad_hoc(workspace.clone(), None)
            .unwrap_or_else(|e| panic!("{e}"));

        let cfg = HarnessConfig {
            runs_dir: store.runs_dir().to_path_buf(),
            auto_resume_on_start: false,
        };
        let resolved =
            resolve_startup_run(&cfg, &store, workspace.clone()).unwrap_or_else(|e| panic!("{e}"));

        assert_ne!(
            resolved.id, seeded.id,
            "should create a new run instead of resuming"
        );
        // Both runs should exist now.
        let listed = store.list().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(listed.len(), 2);
    }

    #[test]
    fn auto_resume_true_skips_terminal_runs() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));

        let mut completed = store
            .create_ad_hoc(workspace.clone(), None)
            .unwrap_or_else(|e| panic!("{e}"));
        completed.status = RunStatus::Completed;
        store.save(&completed).unwrap_or_else(|e| panic!("{e}"));

        let cfg = HarnessConfig {
            runs_dir: store.runs_dir().to_path_buf(),
            auto_resume_on_start: true,
        };
        let resolved =
            resolve_startup_run(&cfg, &store, workspace.clone()).unwrap_or_else(|e| panic!("{e}"));

        assert_ne!(
            resolved.id, completed.id,
            "should not resume a Completed run"
        );
    }

    #[test]
    fn auto_resume_true_creates_run_when_none_exist() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));

        let cfg = HarnessConfig {
            runs_dir: store.runs_dir().to_path_buf(),
            auto_resume_on_start: true,
        };
        let resolved =
            resolve_startup_run(&cfg, &store, workspace.clone()).unwrap_or_else(|e| panic!("{e}"));

        // The run should be persisted and listed.
        let listed = store.list().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, resolved.id);
    }
}
