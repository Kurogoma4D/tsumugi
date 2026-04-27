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

use chrono::Utc;
use tmg_core::TokenCounter;

use crate::artifacts::{ProgressLog, SessionLog};
use crate::error::HarnessError;
use crate::run::{Run, RunSummary};
use crate::session::{Session, SessionEndTrigger, SessionHandle};
use crate::store::RunStore;

/// Default max tokens for `session_bootstrap` output before truncation.
pub const DEFAULT_BOOTSTRAP_MAX_TOKENS: usize = 4096;

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
    /// Active in-flight session, populated between
    /// [`begin_session`](RunRunner::begin_session) and
    /// [`end_session`](RunRunner::end_session).
    active_session: Option<Session>,
    /// Token budget for `session_bootstrap` output.
    ///
    /// Reads from `[harness] bootstrap_max_tokens` in `tsumugi.toml`.
    bootstrap_max_tokens: usize,
    /// Optional token counter used by `session_bootstrap` to budget its
    /// output. When `None`, falls back to
    /// [`TokenCounter::estimate_tokens`].
    token_counter: Option<Arc<TokenCounter>>,
}

impl RunRunner {
    /// Construct a runner over the given run and store, deriving artifact
    /// handles from the store's directory layout.
    #[must_use]
    pub fn new(run: Run, store: Arc<RunStore>) -> Self {
        let progress = ProgressLog::new(store.progress_file(&run.id));
        let session_log = SessionLog::new(store.session_log_dir(&run.id));
        Self {
            run,
            store,
            progress,
            session_log,
            active_session: None,
            bootstrap_max_tokens: DEFAULT_BOOTSTRAP_MAX_TOKENS,
            token_counter: None,
        }
    }

    /// Set the `bootstrap_max_tokens` budget. Defaults to
    /// [`DEFAULT_BOOTSTRAP_MAX_TOKENS`].
    pub fn set_bootstrap_max_tokens(&mut self, n: usize) {
        self.bootstrap_max_tokens = n;
    }

    /// Return the current `bootstrap_max_tokens` budget.
    #[must_use]
    pub fn bootstrap_max_tokens(&self) -> usize {
        self.bootstrap_max_tokens
    }

    /// Attach a [`TokenCounter`] for accurate output sizing in
    /// `session_bootstrap`.
    pub fn set_token_counter(&mut self, counter: Arc<TokenCounter>) {
        self.token_counter = Some(counter);
    }

    /// Borrow the optional [`TokenCounter`].
    #[must_use]
    pub fn token_counter(&self) -> Option<&Arc<TokenCounter>> {
        self.token_counter.as_ref()
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
    pub fn end_session(
        &mut self,
        handle: &SessionHandle,
        trigger: SessionEndTrigger,
    ) -> Result<(), HarnessError> {
        if let Some(mut session) = self.active_session.take() {
            if session.index != handle.index {
                tracing::warn!(
                    handle_index = handle.index,
                    active_index = session.index,
                    "end_session handle does not match active session; persisting active session anyway"
                );
            }
            session.end(trigger);
            self.session_log.save(&session)?;
        } else {
            tracing::warn!(
                handle_index = handle.index,
                "end_session called without an active session"
            );
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
