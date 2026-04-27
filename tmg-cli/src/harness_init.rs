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
///   the most recent resumable run **whose `workspace_path` matches**
///   the current `workspace_path`. If one exists, load it and return.
/// - Otherwise (no matching resumable run, or auto-resume disabled),
///   create a new ad-hoc run rooted at `workspace_path`.
///
/// Filtering by `workspace_path` prevents resuming a run that belongs
/// to a different project when several projects share the same
/// configured `runs_dir`.
///
/// In both cases the run's `last_session_at` and `session_count` are
/// updated and persisted via the [`RunRunner`](tmg_harness::RunRunner)
/// elsewhere; this function only handles the load-vs-create choice.
pub fn resolve_startup_run(
    harness: &HarnessConfig,
    store: &Arc<RunStore>,
    workspace_path: std::path::PathBuf,
) -> Result<Run, HarnessError> {
    if harness.auto_resume_on_start
        && let Some(summary) = store.latest_resumable(Some(&workspace_path))?
    {
        return store.load(&summary.id);
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
            bootstrap_max_tokens: tmg_harness::DEFAULT_BOOTSTRAP_MAX_TOKENS,
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
            bootstrap_max_tokens: tmg_harness::DEFAULT_BOOTSTRAP_MAX_TOKENS,
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
            bootstrap_max_tokens: tmg_harness::DEFAULT_BOOTSTRAP_MAX_TOKENS,
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
            bootstrap_max_tokens: tmg_harness::DEFAULT_BOOTSTRAP_MAX_TOKENS,
        };
        let resolved =
            resolve_startup_run(&cfg, &store, workspace.clone()).unwrap_or_else(|e| panic!("{e}"));

        // The run should be persisted and listed.
        let listed = store.list().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, resolved.id);
    }

    #[test]
    fn auto_resume_skips_runs_for_other_workspace() {
        let (tmp, store) = make_store();
        let workspace_a = tmp.path().join("workspace-a");
        let workspace_b = tmp.path().join("workspace-b");
        std::fs::create_dir_all(&workspace_a).unwrap_or_else(|e| panic!("{e}"));
        std::fs::create_dir_all(&workspace_b).unwrap_or_else(|e| panic!("{e}"));

        // Seed a resumable run for workspace_a.
        let run_a = store
            .create_ad_hoc(workspace_a.clone(), None)
            .unwrap_or_else(|e| panic!("{e}"));

        let cfg = HarnessConfig {
            runs_dir: store.runs_dir().to_path_buf(),
            auto_resume_on_start: true,
            bootstrap_max_tokens: tmg_harness::DEFAULT_BOOTSTRAP_MAX_TOKENS,
        };

        // Starting in workspace_b should NOT resume run_a; instead a
        // fresh ad-hoc run should be created with workspace_path = b.
        let resolved = resolve_startup_run(&cfg, &store, workspace_b.clone())
            .unwrap_or_else(|e| panic!("{e}"));

        assert_ne!(resolved.id, run_a.id, "must not resume across workspaces");
        assert_eq!(resolved.workspace_path, workspace_b);

        // Both runs should exist now.
        let listed = store.list().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(listed.len(), 2);
    }

    /// Integration-style test exercising the resume flow end-to-end:
    /// `resolve_startup_run` -> `RunRunner::new` -> `begin_session`,
    /// across two simulated launches in the same workspace.
    #[test]
    fn resume_flow_increments_session_count_on_second_launch() {
        use tmg_harness::{RunRunner, SessionEndTrigger};

        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));

        let cfg = HarnessConfig {
            runs_dir: store.runs_dir().to_path_buf(),
            auto_resume_on_start: true,
            bootstrap_max_tokens: tmg_harness::DEFAULT_BOOTSTRAP_MAX_TOKENS,
        };

        // First launch: create a fresh run, begin and end one session,
        // capture the assigned RunId.
        let first_run =
            resolve_startup_run(&cfg, &store, workspace.clone()).unwrap_or_else(|e| panic!("{e}"));
        let first_id = first_run.id.clone();
        let mut runner_one = RunRunner::new(first_run, Arc::clone(&store));
        let handle_one = runner_one.begin_session().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(runner_one.run().session_count, 1);
        runner_one
            .end_session(&handle_one, SessionEndTrigger::Completed)
            .unwrap_or_else(|e| panic!("{e}"));

        // Second launch: resume the same run (same workspace, status
        // is still Running so it should be picked up by
        // latest_resumable).
        let second_run =
            resolve_startup_run(&cfg, &store, workspace.clone()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            second_run.id, first_id,
            "second launch should resume the same RunId"
        );
        assert_eq!(
            second_run.session_count, 1,
            "session_count from prior launch must be persisted"
        );

        let mut runner_two = RunRunner::new(second_run, Arc::clone(&store));
        let handle_two = runner_two.begin_session().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(handle_two.index, 2);
        assert_eq!(
            runner_two.run().session_count,
            2,
            "session_count must increment to 2 on resume"
        );
        assert_eq!(runner_two.run().id, first_id);
    }
}
