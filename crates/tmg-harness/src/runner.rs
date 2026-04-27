//! Run runner used by the CLI startup sequence.
//!
//! [`RunRunner`] owns the active [`Run`] plus the artifact handles
//! ([`ProgressLog`], [`SessionLog`]) that the Run-scoped tools mutate.
//! It is shared across the CLI / TUI / tools as `Arc<Mutex<RunRunner>>`
//! so multiple async tasks can update the active session safely.
//!
//! Responsibilities:
//!
//! - hold the active [`Run`] and a reference-counted [`RunStore`] so
//!   that session lifecycle updates can be persisted from anywhere in
//!   the CLI / TUI.
//! - bump `session_count` and `last_session_at` on
//!   [`begin_session`](RunRunner::begin_session), append a
//!   `## Session #N` header to `progress.md`, and persist `run.toml`.
//! - on [`end_session`](RunRunner::end_session), close the active
//!   [`Session`], write `session_NNN.json` via [`SessionLog::save`],
//!   then re-persist `run.toml`.
//! - expose mutators ([`active_session_mut`](RunRunner::active_session_mut),
//!   [`save_active_session`](RunRunner::save_active_session)) so the
//!   `progress_append` / `session_summary_save` tools can update the
//!   live session record from inside the agent loop.
//!
//! Workflow integration and escalation are out of scope for this issue
//! and tracked separately.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;

use crate::artifacts::{FeatureList, InitScript, ProgressLog, SessionLog};
use crate::error::HarnessError;
use crate::run::{Run, RunScope, RunSummary};
use crate::session::{Session, SessionEndTrigger, SessionHandle};
use crate::store::RunStore;

/// Default max tokens for `session_bootstrap` output before truncation.
pub const DEFAULT_BOOTSTRAP_MAX_TOKENS: usize = 4096;

/// Default wall-clock budget for one harnessed session.
///
/// Mirrors the CLI's `[harness] default_session_timeout` default
/// (`30m`); used as a fallback when the runner has not been
/// configured with a value from `tsumugi.toml`.
pub const DEFAULT_SESSION_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// Run runner wrapping a [`Run`] plus its artifacts.
///
/// **Concurrency:** like [`RunStore`], this runner assumes the owning
/// process is the only writer to the underlying `runs_dir`. There is
/// no inter-process locking around session begin/end; running multiple
/// `tmg` processes against the same `runs_dir` can race on
/// `session_count` updates. A locking layer is tracked as a follow-up.
#[derive(Debug)]
pub struct RunRunner {
    run: Run,
    store: Arc<RunStore>,
    progress: ProgressLog,
    session_log: SessionLog,
    /// Handle to the harnessed-only `features.json`.
    ///
    /// Constructed eagerly from the store's path layout for both ad-hoc
    /// and harnessed runs; the underlying file only exists for harnessed
    /// runs whose initializer has populated it. Callers must check
    /// [`scope`](Self::scope) before assuming the file exists.
    features: FeatureList,
    /// Handle to the harnessed-only `init.sh`.
    ///
    /// Same eager-construction contract as [`features`](Self::features).
    init_script: InitScript,
    /// Active in-flight session, populated between
    /// [`begin_session`](RunRunner::begin_session) and
    /// [`end_session`](RunRunner::end_session).
    active_session: Option<Session>,
    /// Token budget for `session_bootstrap` output.
    ///
    /// Reads from `[harness] bootstrap_max_tokens` in `tsumugi.toml`.
    /// `0` disables truncation entirely; see
    /// [`set_bootstrap_max_tokens`](Self::set_bootstrap_max_tokens) for
    /// the full semantics.
    bootstrap_max_tokens: usize,
    /// Wall-clock budget for one harnessed session.
    ///
    /// Reads from `[harness] default_session_timeout` in `tsumugi.toml`
    /// (humantime string, e.g. `"30m"`). Used by the harnessed
    /// `session_bootstrap` to cap `init.sh` execution so the inner
    /// sandbox timer matches the outer `tokio::time::timeout` deadline.
    /// Defaults to [`DEFAULT_SESSION_TIMEOUT`].
    default_session_timeout: Duration,
}

impl RunRunner {
    /// Construct a runner over the given run and store, deriving artifact
    /// handles from the store's directory layout.
    #[must_use]
    pub fn new(run: Run, store: Arc<RunStore>) -> Self {
        let progress = ProgressLog::new(store.progress_file(&run.id));
        let session_log = SessionLog::new(store.session_log_dir(&run.id));
        let features = FeatureList::new(store.features_file(&run.id));
        let init_script = InitScript::new(store.init_script_file(&run.id));
        Self {
            run,
            store,
            progress,
            session_log,
            features,
            init_script,
            active_session: None,
            bootstrap_max_tokens: DEFAULT_BOOTSTRAP_MAX_TOKENS,
            default_session_timeout: DEFAULT_SESSION_TIMEOUT,
        }
    }

    /// Set the `bootstrap_max_tokens` budget.
    ///
    /// **Semantics:**
    /// - `0` means **no truncation** — the bootstrap payload is emitted
    ///   in full regardless of size. Use this only when you trust the
    ///   inputs (e.g. tests or short-lived projects).
    /// - any value `>= 1` is treated as a hard budget; older
    ///   `progress.md` sessions and tail-end git log lines are shed
    ///   until the serialized payload's estimated token count fits.
    ///
    /// Defaults to [`DEFAULT_BOOTSTRAP_MAX_TOKENS`].
    pub fn set_bootstrap_max_tokens(&mut self, n: usize) {
        self.bootstrap_max_tokens = n;
    }

    /// Return the current `bootstrap_max_tokens` budget.
    ///
    /// A return value of `0` indicates "no truncation"; any value
    /// `>= 1` is the hard token budget enforced by `session_bootstrap`.
    /// See [`set_bootstrap_max_tokens`](Self::set_bootstrap_max_tokens)
    /// for full semantics.
    #[must_use]
    pub fn bootstrap_max_tokens(&self) -> usize {
        self.bootstrap_max_tokens
    }

    /// Set the per-session wall-clock budget.
    ///
    /// This is the value plumbed into `session_bootstrap`'s `init.sh`
    /// execution so that the inner sandbox timer (`with_timeout`) and
    /// the outer `tokio::time::timeout` share the same deadline.
    /// Defaults to [`DEFAULT_SESSION_TIMEOUT`].
    pub fn set_default_session_timeout(&mut self, timeout: Duration) {
        self.default_session_timeout = timeout;
    }

    /// Return the current per-session wall-clock budget.
    ///
    /// Read by `session_bootstrap` to bound `init.sh` execution; see
    /// [`set_default_session_timeout`](Self::set_default_session_timeout)
    /// for the contract.
    #[must_use]
    pub fn default_session_timeout(&self) -> Duration {
        self.default_session_timeout
    }

    /// Borrow the active run.
    #[must_use]
    pub fn run(&self) -> &Run {
        &self.run
    }

    /// Borrow the active run mutably.
    pub fn run_mut(&mut self) -> &mut Run {
        &mut self.run
    }

    /// Lightweight summary of the active run, for header display etc.
    #[must_use]
    pub fn summary(&self) -> RunSummary {
        RunSummary::from_run(&self.run)
    }

    /// Borrow the run's [`ProgressLog`].
    #[must_use]
    pub fn progress_log(&self) -> &ProgressLog {
        &self.progress
    }

    /// Borrow the run's [`SessionLog`].
    #[must_use]
    pub fn session_log(&self) -> &SessionLog {
        &self.session_log
    }

    /// Borrow the run's [`FeatureList`] handle.
    ///
    /// The underlying file (`features.json`) only exists for harnessed
    /// runs after the `initializer` subagent has created it. Use
    /// [`scope`](Self::scope) to gate access at the call site.
    #[must_use]
    pub fn features(&self) -> &FeatureList {
        &self.features
    }

    /// Borrow the run's [`InitScript`] handle.
    ///
    /// Like [`features`](Self::features), the underlying file
    /// (`init.sh`) only exists for harnessed runs.
    #[must_use]
    pub fn init_script(&self) -> &InitScript {
        &self.init_script
    }

    /// Borrow the run's [`RunScope`].
    ///
    /// Convenience method for tools that need to gate registration or
    /// behaviour on whether the run is ad-hoc or harnessed.
    #[must_use]
    pub fn scope(&self) -> &RunScope {
        &self.run.scope
    }

    /// Whether the active run is harnessed.
    #[must_use]
    pub fn is_harnessed(&self) -> bool {
        matches!(self.run.scope, RunScope::Harnessed { .. })
    }

    /// Borrow the workspace path for this run.
    #[must_use]
    pub fn workspace_path(&self) -> &std::path::Path {
        &self.run.workspace_path
    }

    /// Workspace path as an owned `PathBuf` for tools that want to
    /// move the value into a spawned task.
    #[must_use]
    pub fn workspace_path_owned(&self) -> PathBuf {
        self.run.workspace_path.clone()
    }

    /// Borrow the active session, if any.
    #[must_use]
    pub fn active_session(&self) -> Option<&Session> {
        self.active_session.as_ref()
    }

    /// Borrow the active session mutably (used by Run-scoped tools to
    /// update `tool_calls_count`, `files_modified`, etc.).
    pub fn active_session_mut(&mut self) -> Option<&mut Session> {
        self.active_session.as_mut()
    }

    /// Persist the active session to `session_NNN.json` without ending
    /// it. Used by `session_summary_save` so the saved summary is
    /// durable even if the process is killed before `end_session`.
    pub fn save_active_session(&self) -> Result<(), HarnessError> {
        if let Some(session) = self.active_session.as_ref() {
            self.session_log.save(session)?;
        }
        Ok(())
    }

    /// Begin a new session.
    ///
    /// - Increments `session_count` and stamps `last_session_at` on the run.
    /// - Persists `run.toml`.
    /// - Appends a `## Session #N (timestamp) [scope]` header to
    ///   `progress.md`.
    /// - Writes the initial `session_NNN.json` so external readers see a
    ///   consistent record (with `ended_at = None`) immediately.
    pub fn begin_session(&mut self) -> Result<SessionHandle, HarnessError> {
        let now = Utc::now();
        self.run.session_count = self.run.session_count.saturating_add(1);
        self.run.last_session_at = Some(now);
        self.store.save(&self.run)?;

        self.progress
            .append_session_header(self.run.session_count, &self.run.scope, now)?;

        let mut session = Session::begin(self.run.session_count);
        // begin() stamps with Utc::now(); harmonise so both timestamps match.
        session.started_at = now;
        // Persist initial record (ended_at still None so readers can tell
        // it's the live session).
        self.session_log.save(&session)?;

        let handle = SessionHandle {
            index: session.index,
        };
        self.active_session = Some(session);
        Ok(handle)
    }

    /// End the given session, marking the trigger, persisting the final
    /// `session_NNN.json` and the run record.
    ///
    /// If `handle.index` does not match the currently-active session,
    /// returns [`HarnessError::SessionMismatch`] **without** mutating
    /// the active session or persisting any state. This avoids silently
    /// overwriting the active session with an incorrect index, which
    /// would corrupt `session_NNN.json` for the real active session.
    ///
    /// The active session is also left untouched (not consumed) so the
    /// caller can recover by retrying with the correct handle, e.g.
    /// the one returned by [`begin_session`](Self::begin_session).
    pub fn end_session(
        &mut self,
        handle: &SessionHandle,
        trigger: SessionEndTrigger,
    ) -> Result<(), HarnessError> {
        // Validate the handle against the active session before taking
        // it out of `self`, so a mismatched handle is rejected without
        // disturbing in-memory state.
        let Some(session_ref) = self.active_session.as_ref() else {
            tracing::warn!(
                handle_index = handle.index,
                "end_session called without an active session"
            );
            self.store.save(&self.run)?;
            return Ok(());
        };
        if session_ref.index != handle.index {
            return Err(HarnessError::SessionMismatch {
                expected: session_ref.index,
                actual: handle.index,
            });
        }

        // Indices match — take the session out and finalize it.
        if let Some(mut session) = self.active_session.take() {
            session.end(trigger);
            self.session_log.save(&session)?;
        }
        self.store.save(&self.run)?;
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;

    fn make_runner() -> (tempfile::TempDir, RunRunner) {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let store = Arc::new(RunStore::new(tmp.path().join("runs")));
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));
        let runner = RunRunner::new(run, store);
        (tmp, runner)
    }

    #[test]
    fn begin_session_increments_count() {
        let (_tmp, mut runner) = make_runner();
        assert_eq!(runner.run().session_count, 0);
        let handle = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(handle.index, 1);
        assert_eq!(runner.run().session_count, 1);
        assert!(runner.run().last_session_at.is_some());
        assert!(runner.active_session().is_some());
    }

    #[test]
    fn begin_session_appends_progress_header() {
        let (_tmp, mut runner) = make_runner();
        let _ = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        let progress =
            std::fs::read_to_string(runner.progress_log().path()).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            progress.starts_with("# Progress Log\n"),
            "progress.md missing init header: {progress}"
        );
        assert!(
            progress.contains("## Session #1 ("),
            "progress.md missing session header: {progress}"
        );
        assert!(progress.contains("[ad-hoc]"), "{progress}");
    }

    #[test]
    fn end_session_writes_session_log_json() {
        let (_tmp, mut runner) = make_runner();
        let handle = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        runner
            .end_session(&handle, SessionEndTrigger::Completed)
            .unwrap_or_else(|e| panic!("{e}"));
        let entries = runner
            .session_log()
            .list()
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].index, 1);

        let session = runner
            .session_log()
            .load(1)
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_else(|| panic!("session 1 should exist"));
        assert_eq!(session.end_trigger, Some(SessionEndTrigger::Completed));
        assert!(session.ended_at.is_some());
    }

    #[test]
    fn end_session_persists_run() {
        let (_tmp, mut runner) = make_runner();
        let handle = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        runner
            .end_session(&handle, SessionEndTrigger::Completed)
            .unwrap_or_else(|e| panic!("{e}"));
        let store = Arc::clone(&runner.store);
        let id = runner.run().id.clone();
        let loaded = store.load(&id).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(loaded.session_count, 1);
        assert_eq!(loaded.workspace_path, runner.run().workspace_path);
        assert_eq!(loaded.id, id);
    }

    #[test]
    fn end_session_mismatched_handle_returns_error_without_persisting() {
        let (_tmp, mut runner) = make_runner();
        let real_handle = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        let bad_handle = SessionHandle {
            index: real_handle.index + 7,
        };

        let result = runner.end_session(&bad_handle, SessionEndTrigger::Completed);
        assert!(
            matches!(
                result,
                Err(HarnessError::SessionMismatch { expected, actual })
                    if expected == real_handle.index && actual == bad_handle.index
            ),
            "expected SessionMismatch, got {result:?}",
        );

        // Active session still alive: the real handle still works.
        assert!(
            runner.active_session().is_some(),
            "active session must not be dropped on mismatch",
        );
        runner
            .end_session(&real_handle, SessionEndTrigger::Completed)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(runner.active_session().is_none());
    }

    #[test]
    fn save_active_session_persists_summary_without_ending() {
        let (_tmp, mut runner) = make_runner();
        let _ = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        if let Some(s) = runner.active_session_mut() {
            s.summary = "halfway summary".to_owned();
        }
        runner
            .save_active_session()
            .unwrap_or_else(|e| panic!("{e}"));
        let session = runner
            .session_log()
            .load(1)
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_else(|| panic!("session 1 should exist"));
        assert_eq!(session.summary, "halfway summary");
        assert!(
            session.ended_at.is_none(),
            "save_active_session must not end the session"
        );
    }
}
