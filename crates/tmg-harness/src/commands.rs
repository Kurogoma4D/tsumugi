//! Shared CLI/TUI command logic.
//!
//! These functions encapsulate the run-management operations exposed
//! by the [`tmg run`](https://github.com/Kurogoma4D/tsumugi/issues/43)
//! subcommand family and the matching TUI slash-commands (issue #46).
//! Both call sites use the same building blocks here so the on-disk
//! state transitions stay consistent regardless of which entry point
//! drove them.
//!
//! The functions take borrowed handles to the [`RunStore`] /
//! [`RunRunner`] rather than owning them; this keeps the helpers
//! callable from inside an `Arc<Mutex<RunRunner>>` guard or from a
//! one-shot CLI invocation that constructs a runner just to flip a
//! single field.
//!
//! See SPEC §9.8 for the canonical command reference.
//!
//! # Reading status detail
//!
//! [`StatusReport`] bundles the structured fields the CLI's
//! `tmg run status` formats. Higher-level surfaces (TUI, JSON
//! exporters) can render the same struct without re-parsing
//! `progress.md` themselves.

use crate::artifacts::FeatureList;
use crate::error::HarnessError;
use crate::run::{Run, RunId, RunScope, RunStatus, RunSummary};
use crate::runner::RunRunner;
use crate::session::{Session, SessionEndTrigger};
use crate::store::RunStore;

/// Snapshot of a single run for the `tmg run status` command.
///
/// Combines the stored [`Run`] record with the tail of `progress.md`
/// (if any), the most recent persisted [`Session`] (if any), and the
/// harnessed-only feature pass/fail histogram. Higher-level surfaces
/// can render this structure without re-reading any artifact files.
#[derive(Debug, Clone)]
pub struct StatusReport {
    /// Full run record (scope, status, session counts).
    pub run: Run,
    /// Last `n` lines of `progress.md`, joined with `\n`. Empty when
    /// the file does not exist or has no content.
    pub progress_tail: String,
    /// Number of progress lines actually returned.
    pub progress_tail_lines: usize,
    /// Most recent persisted session, when one exists.
    pub last_session: Option<Session>,
    /// Feature pass/fail histogram for harnessed runs. `None` for
    /// ad-hoc or when `features.json` is missing/malformed.
    pub feature_histogram: Option<FeatureHistogram>,
}

/// Pass/fail histogram for a harnessed run's `features.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureHistogram {
    /// Total number of features in `features.json`.
    pub total: usize,
    /// Number of features whose `passes` field is `true`.
    pub passing: usize,
}

impl FeatureHistogram {
    /// Number of failing features (`total - passing`).
    #[must_use]
    pub fn failing(&self) -> usize {
        self.total.saturating_sub(self.passing)
    }
}

/// Return all runs as [`RunSummary`] entries (newest first).
///
/// This is a thin wrapper over [`RunStore::list`] so the CLI / TUI
/// share one entry point.
pub fn list(store: &RunStore) -> Result<Vec<RunSummary>, HarnessError> {
    store.list()
}

/// Build a [`StatusReport`] for the given run.
///
/// `progress_tail_lines` controls how many lines of `progress.md` to
/// include in the report (the most recent N). The CLI defaults to
/// twenty.
pub fn status(
    store: &RunStore,
    run_id: &RunId,
    progress_tail_lines: usize,
) -> Result<StatusReport, HarnessError> {
    let run = store.load(run_id)?;

    let progress_path = store.progress_file(run_id);
    let progress_tail = match std::fs::read_to_string(&progress_path) {
        Ok(c) => tail_lines(&c, progress_tail_lines),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(HarnessError::io(&progress_path, e)),
    };
    let actual_lines = if progress_tail.is_empty() {
        0
    } else {
        progress_tail.lines().count()
    };

    let last_session = if run.session_count == 0 {
        None
    } else {
        let session_dir = store.session_log_dir(run_id);
        let log = crate::artifacts::SessionLog::new(session_dir);
        log.load(run.session_count)?
    };

    let feature_histogram = match &run.scope {
        RunScope::Harnessed { .. } => {
            let features_path = store.features_file(run_id);
            if features_path.exists() {
                let list = FeatureList::new(features_path);
                match list.read() {
                    Ok(features) => {
                        let total = features.features.len();
                        let passing = features.features.iter().filter(|f| f.passes).count();
                        Some(FeatureHistogram { total, passing })
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "failed to read features.json for status report"
                        );
                        None
                    }
                }
            } else {
                None
            }
        }
        RunScope::AdHoc => None,
    };

    Ok(StatusReport {
        run,
        progress_tail,
        progress_tail_lines: actual_lines,
        last_session,
        feature_histogram,
    })
}

/// Mark the run as paused.
///
/// Flips `run.toml` status to [`RunStatus::Paused`]. The runner
/// remains usable; a follow-up `tmg run resume` will pick it back up
/// because [`RunStatus::Paused`] is treated as resumable.
///
/// **Refuses if a TUI is currently attached** to the same run (see
/// [`crate::tui_sentinel`]): an in-memory runner inside the TUI would
/// otherwise silently overwrite the `Paused` status on its next
/// persist. Outside a TUI session the operation is best-effort — the
/// CLI surfaces the in-TUI alternative in the returned error.
///
/// # Errors
///
/// - [`HarnessError::TuiAttached`] when a live TUI sentinel is
///   detected for this run.
/// - [`HarnessError::Io`] / [`HarnessError::Serialize`] when persisting
///   `run.toml` fails.
pub fn pause(runner: &mut RunRunner) -> Result<(), HarnessError> {
    refuse_if_tui_attached(runner)?;
    runner.set_status(RunStatus::Paused)
}

/// Mark the run as failed with the supplied reason.
///
/// Flips `run.toml` status to [`RunStatus::Failed`] with the given
/// message. The CLI's `tmg run abort` command uses
/// `"user aborted"` as the reason; other call sites can pass a more
/// specific string.
///
/// **Refuses if a TUI is currently attached** to the same run; see
/// [`pause`] for the rationale and [`HarnessError::TuiAttached`] for
/// the surfaced error.
///
/// # Errors
///
/// - [`HarnessError::TuiAttached`] when a live TUI sentinel is
///   detected for this run.
/// - [`HarnessError::Io`] / [`HarnessError::Serialize`] when persisting
///   `run.toml` fails.
pub fn abort(runner: &mut RunRunner, reason: impl Into<String>) -> Result<(), HarnessError> {
    refuse_if_tui_attached(runner)?;
    runner.set_status(RunStatus::Failed {
        reason: reason.into(),
    })
}

/// Surface [`HarnessError::TuiAttached`] when the run's TUI sentinel
/// reports a live process. Used by `pause` / `abort` to short-circuit
/// before mutating `run.toml`.
fn refuse_if_tui_attached(runner: &RunRunner) -> Result<(), HarnessError> {
    let run_dir = runner.store().run_dir(runner.run_id());
    if let Some(pid) = crate::tui_sentinel::read_live_pid(&run_dir)? {
        return Err(HarnessError::TuiAttached {
            run_id: runner.run_id().as_str().to_owned(),
            pid,
        });
    }
    Ok(())
}

/// Force the run to harnessed scope without going through the
/// auto-promotion gate.
///
/// Unlike [`RunRunner::escalate_to_harnessed`], this does not write
/// `features.json` / `init.sh` (they are assumed to either already
/// exist from a prior promotion or be supplied by the user out-of-
/// band). The on-disk `run.toml` flips to harnessed and the runner's
/// in-memory state is updated; the caller is responsible for
/// re-installing a [`crate::tools::RunRunnerToolProvider`] so the
/// LLM sees the harnessed-only tools.
///
/// **Pre-condition**: `features.json` must exist in the run's
/// workspace. The doc invariant on
/// [`crate::store::RunStore::upgrade_to_harnessed`] is that the
/// initializer subagent has already populated the artifacts; this
/// helper checks the file is on disk before flipping the on-disk
/// scope so a typo'd `tmg run upgrade <id>` outside the TUI does not
/// end up with a harnessed `run.toml` that points at non-existent
/// artifacts.
///
/// **Idempotent**: when the run is already harnessed, the call
/// returns `Ok(())` immediately without touching `run.toml` or the
/// in-memory runner. This avoids re-stamping `upgraded_at` /
/// `upgraded_from_session` / `upgrade_reason` every time the user
/// re-runs `tmg run upgrade` against an already-harnessed run.
///
/// # Errors
///
/// - [`HarnessError::Precondition`] when `features.json` is missing
///   from the run's workspace directory.
/// - [`HarnessError::Io`] / [`HarnessError::Serialize`] for write
///   failures on `run.toml`.
pub fn upgrade(runner: &mut RunRunner) -> Result<(), HarnessError> {
    if runner.is_harnessed() {
        // True no-op: re-saving `run.toml` with the same scope would
        // reset `upgraded_at` and the upgrade-reason metadata, which
        // is misleading after a real prior promotion. Leave the
        // record untouched.
        return Ok(());
    }
    let session_count = runner.run().session_count;
    let store = runner.store().clone();
    let features_path = store.features_file(runner.run_id());
    if !features_path.exists() {
        return Err(HarnessError::Precondition {
            message: format!(
                "cannot upgrade: features.json not found at {}; \
                 run the initializer first or upgrade from inside a TUI session",
                features_path.display(),
            ),
        });
    }
    let run = store.upgrade_to_harnessed(runner.run_id(), session_count, "manual /run upgrade")?;
    runner.replace_run(run);
    Ok(())
}

/// Demote a harnessed run back to ad-hoc.
///
/// The on-disk `features.json` / `init.sh` artifacts are preserved
/// (see [`RunStore::downgrade_to_ad_hoc`] for rationale). Only the
/// `run.toml` scope flips, so the LLM stops seeing harnessed-only
/// tools after the caller re-installs a fresh
/// [`crate::tools::RunRunnerToolProvider`].
///
/// Idempotent: when the run is already ad-hoc, the call is a no-op
/// (still re-saves `run.toml`).
pub fn downgrade(runner: &mut RunRunner) -> Result<(), HarnessError> {
    let store = runner.store().clone();
    let run = store.downgrade_to_ad_hoc(runner.run_id())?;
    runner.replace_run(run);
    Ok(())
}

/// Trigger a manual session rotation.
///
/// Equivalent to [`RunRunner::end_session_with_rotation`] with
/// [`SessionEndTrigger::UserNewSession`]. The current session is
/// closed and a fresh successor session is started.
pub fn new_session(runner: &mut RunRunner) -> Result<(), HarnessError> {
    runner
        .end_session_with_rotation(SessionEndTrigger::UserNewSession)
        .map(|_started| ())
}

/// Return the last `n` lines of `s` joined with `\n`.
///
/// `n == 0` returns an empty string. Lines beyond the tail are
/// dropped. The return value never has a trailing `\n` so the CLI
/// can append its own terminator.
fn tail_lines(s: &str, n: usize) -> String {
    if n == 0 || s.is_empty() {
        return String::new();
    }
    let lines: Vec<&str> = s.lines().collect();
    let take_from = lines.len().saturating_sub(n);
    lines[take_from..].join("\n")
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn make_runner_ad_hoc() -> (tempfile::TempDir, Arc<RunStore>, RunRunner) {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let store = Arc::new(RunStore::new(tmp.path().join("runs")));
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));
        let runner = RunRunner::new(run, Arc::clone(&store));
        (tmp, store, runner)
    }

    /// Plant a minimal valid `features.json` so `upgrade` clears its
    /// pre-condition check.
    fn write_minimal_features(store: &RunStore, run_id: &crate::run::RunId) {
        let path = store.features_file(run_id);
        std::fs::write(
            &path,
            r#"{"features":[{"id":"f0","category":"test","description":"d","steps":[],"passes":false}]}"#,
        )
        .unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn pause_flips_status_to_paused() {
        let (_tmp, store, mut runner) = make_runner_ad_hoc();
        let id = runner.run_id().clone();
        pause(&mut runner).unwrap_or_else(|e| panic!("{e}"));
        let reloaded = store.load(&id).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(reloaded.status, RunStatus::Paused);
    }

    #[test]
    fn abort_flips_status_to_failed() {
        let (_tmp, store, mut runner) = make_runner_ad_hoc();
        let id = runner.run_id().clone();
        abort(&mut runner, "user aborted").unwrap_or_else(|e| panic!("{e}"));
        let reloaded = store.load(&id).unwrap_or_else(|e| panic!("{e}"));
        match reloaded.status {
            RunStatus::Failed { reason } => assert_eq!(reason, "user aborted"),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    /// `pause` refuses with [`HarnessError::TuiAttached`] when the
    /// run's `.tui-pid` sentinel reports a live process. Use the
    /// current process's PID so the liveness probe definitely
    /// resolves to "alive".
    #[test]
    fn pause_refuses_when_tui_sentinel_is_live() {
        let (_tmp, store, mut runner) = make_runner_ad_hoc();
        let id = runner.run_id().clone();
        let run_dir = store.run_dir(&id);
        crate::tui_sentinel::write(&run_dir).unwrap_or_else(|e| panic!("{e}"));

        match pause(&mut runner) {
            Err(HarnessError::TuiAttached { run_id, pid }) => {
                assert_eq!(run_id, id.as_str());
                assert_eq!(pid, std::process::id());
            }
            other => panic!("expected TuiAttached, got {other:?}"),
        }
        // The on-disk status must NOT have been flipped to Paused.
        let reloaded = store.load(&id).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(reloaded.status, RunStatus::Running);
    }

    /// Same guard for `abort`.
    #[test]
    fn abort_refuses_when_tui_sentinel_is_live() {
        let (_tmp, store, mut runner) = make_runner_ad_hoc();
        let id = runner.run_id().clone();
        let run_dir = store.run_dir(&id);
        crate::tui_sentinel::write(&run_dir).unwrap_or_else(|e| panic!("{e}"));

        match abort(&mut runner, "user aborted") {
            Err(HarnessError::TuiAttached { .. }) => {}
            other => panic!("expected TuiAttached, got {other:?}"),
        }
        let reloaded = store.load(&id).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(reloaded.status, RunStatus::Running);
    }

    /// Once the sentinel is cleared, `pause` proceeds normally.
    #[test]
    fn pause_proceeds_after_tui_sentinel_cleared() {
        let (_tmp, store, mut runner) = make_runner_ad_hoc();
        let id = runner.run_id().clone();
        let run_dir = store.run_dir(&id);
        crate::tui_sentinel::write(&run_dir).unwrap_or_else(|e| panic!("{e}"));
        crate::tui_sentinel::clear(&run_dir).unwrap_or_else(|e| panic!("{e}"));

        pause(&mut runner).unwrap_or_else(|e| panic!("{e}"));
        let reloaded = store.load(&id).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(reloaded.status, RunStatus::Paused);
    }

    #[test]
    fn upgrade_flips_scope_to_harnessed() {
        let (_tmp, store, mut runner) = make_runner_ad_hoc();
        let id = runner.run_id().clone();
        write_minimal_features(&store, &id);
        upgrade(&mut runner).unwrap_or_else(|e| panic!("{e}"));
        let reloaded = store.load(&id).unwrap_or_else(|e| panic!("{e}"));
        assert!(matches!(reloaded.scope, RunScope::Harnessed { .. }));
        assert!(runner.is_harnessed());
    }

    #[test]
    fn upgrade_without_features_file_returns_precondition_error() {
        let (_tmp, _store, mut runner) = make_runner_ad_hoc();
        let result = upgrade(&mut runner);
        match result {
            Err(HarnessError::Precondition { message }) => {
                assert!(
                    message.contains("features.json not found"),
                    "expected precondition message about features.json, got {message:?}",
                );
            }
            other => panic!("expected Precondition error, got {other:?}"),
        }
        assert!(
            !runner.is_harnessed(),
            "scope must remain ad-hoc when precondition fails",
        );
    }

    #[test]
    fn upgrade_already_harnessed_is_noop_and_preserves_metadata() {
        let (_tmp, store, mut runner) = make_runner_ad_hoc();
        let id = runner.run_id().clone();
        write_minimal_features(&store, &id);

        // First upgrade: stamps the upgrade metadata.
        upgrade(&mut runner).unwrap_or_else(|e| panic!("{e}"));
        let first = store.load(&id).unwrap_or_else(|e| panic!("{e}"));
        let first_upgraded_at = match &first.scope {
            RunScope::Harnessed { upgraded_at, .. } => *upgraded_at,
            RunScope::AdHoc => panic!("expected harnessed after first upgrade"),
        };
        // Sleep so a subsequent re-stamp would observably differ.
        std::thread::sleep(std::time::Duration::from_millis(20));

        // Second upgrade: must be a no-op. The on-disk record must
        // not have its upgrade timestamp re-stamped.
        upgrade(&mut runner).unwrap_or_else(|e| panic!("{e}"));
        let second = store.load(&id).unwrap_or_else(|e| panic!("{e}"));
        match &second.scope {
            RunScope::Harnessed { upgraded_at, .. } => assert_eq!(*upgraded_at, first_upgraded_at),
            RunScope::AdHoc => panic!("expected harnessed after second upgrade"),
        }
    }

    #[test]
    fn downgrade_flips_scope_back_to_ad_hoc() {
        let (_tmp, store, mut runner) = make_runner_ad_hoc();
        let id = runner.run_id().clone();
        write_minimal_features(&store, &id);
        upgrade(&mut runner).unwrap_or_else(|e| panic!("{e}"));
        assert!(runner.is_harnessed());

        downgrade(&mut runner).unwrap_or_else(|e| panic!("{e}"));
        let reloaded = store.load(&id).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(reloaded.scope, RunScope::AdHoc);
        assert!(!runner.is_harnessed());
    }

    #[test]
    fn downgrade_preserves_features_file_on_disk() {
        let (_tmp, store, mut runner) = make_runner_ad_hoc();
        let id = runner.run_id().clone();
        write_minimal_features(&store, &id);
        upgrade(&mut runner).unwrap_or_else(|e| panic!("{e}"));

        // Overwrite the planted features.json with a richer payload so
        // we can verify it's preserved across the downgrade.
        let features_path = store.features_file(&id);
        std::fs::write(
            &features_path,
            r#"{"features":[{"id":"f1","category":"test","description":"x","steps":[],"passes":true}]}"#,
        )
        .unwrap_or_else(|e| panic!("{e}"));

        downgrade(&mut runner).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            features_path.exists(),
            "features.json should NOT be deleted on downgrade",
        );
    }

    #[test]
    fn list_returns_summaries() {
        let (_tmp, store, _runner) = make_runner_ad_hoc();
        let summaries = list(&store).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(summaries.len(), 1);
    }

    #[test]
    fn status_includes_progress_tail() {
        let (_tmp, store, runner) = make_runner_ad_hoc();
        let id = runner.run_id().clone();
        // Append some content so the tail has lines to return.
        let progress = store.progress_file(&id);
        std::fs::write(
            &progress,
            "# Progress Log\n\n## Session #1 (2024-01-01) [ad-hoc]\n- did stuff\n",
        )
        .unwrap_or_else(|e| panic!("{e}"));

        let report = status(&store, &id, 10).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            report.progress_tail.contains("did stuff"),
            "expected progress tail, got {:?}",
            report.progress_tail,
        );
        assert!(report.feature_histogram.is_none());
    }

    #[test]
    fn status_returns_feature_histogram_for_harnessed() {
        let (_tmp, store, mut runner) = make_runner_ad_hoc();
        let id = runner.run_id().clone();
        write_minimal_features(&store, &id);
        upgrade(&mut runner).unwrap_or_else(|e| panic!("{e}"));

        // Overwrite the planted features.json with the richer payload
        // the histogram test wants to assert against.
        let features_path = store.features_file(&id);
        std::fs::write(
            &features_path,
            r#"{"features":[
              {"id":"f1","category":"a","description":"x","steps":[],"passes":true},
              {"id":"f2","category":"a","description":"y","steps":[],"passes":false}
            ]}"#,
        )
        .unwrap_or_else(|e| panic!("{e}"));

        let report = status(&store, &id, 5).unwrap_or_else(|e| panic!("{e}"));
        let hist = report
            .feature_histogram
            .unwrap_or_else(|| panic!("expected histogram"));
        assert_eq!(hist.total, 2);
        assert_eq!(hist.passing, 1);
        assert_eq!(hist.failing(), 1);
    }

    #[test]
    fn tail_lines_returns_last_n() {
        let s = "a\nb\nc\nd\n";
        assert_eq!(tail_lines(s, 2), "c\nd");
        assert_eq!(tail_lines(s, 10), "a\nb\nc\nd");
        assert_eq!(tail_lines(s, 0), "");
        assert_eq!(tail_lines("", 5), "");
    }

    #[tokio::test]
    async fn new_session_rotates_active_session() {
        let (_tmp, _store, mut runner) = make_runner_ad_hoc();
        let _ = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(runner.run().session_count, 1);

        new_session(&mut runner).unwrap_or_else(|e| panic!("{e}"));

        // After rotation, session_count should be incremented.
        assert_eq!(runner.run().session_count, 2);
        assert!(runner.active_session().is_some());
    }
}
