//! Minimal run runner used by the CLI startup sequence.
//!
//! The eventual `RunRunner` will own a workflow engine, escalator, and
//! `AgentLoop` driver. This first slice owns just enough state to:
//!
//! - hold the active [`Run`] and a reference-counted [`RunStore`] so
//!   that session lifecycle updates can be persisted from anywhere in
//!   the CLI / TUI.
//! - bump `session_count` and `last_session_at` on
//!   [`begin_session`](RunRunner::begin_session) and persist on both
//!   begin and [`end_session`](RunRunner::end_session).
//!
//! Workflow integration, bootstrap, session log writing, and
//! escalation are out of scope for this issue and tracked separately.

use std::sync::Arc;

use chrono::Utc;

use crate::error::HarnessError;
use crate::run::{Run, RunSummary};
use crate::session::{SessionEndTrigger, SessionHandle};
use crate::store::RunStore;

/// Minimal harness runner wrapping a [`Run`] and a [`RunStore`].
#[derive(Debug)]
pub struct RunRunner {
    run: Run,
    store: Arc<RunStore>,
}

impl RunRunner {
    /// Construct a runner over the given run and store.
    #[must_use]
    pub fn new(run: Run, store: Arc<RunStore>) -> Self {
        Self { run, store }
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

    /// Begin a new session. Increments `session_count`, stamps
    /// `last_session_at`, and persists the run.
    pub fn begin_session(&mut self) -> Result<SessionHandle, HarnessError> {
        self.run.session_count = self.run.session_count.saturating_add(1);
        self.run.last_session_at = Some(Utc::now());
        self.store.save(&self.run)?;
        Ok(SessionHandle {
            index: self.run.session_count,
        })
    }

    /// End the given session. Currently a no-op beyond persisting any
    /// pending state. The `trigger` argument is reserved for the
    /// session log writer in the follow-up issue.
    pub fn end_session(
        &mut self,
        _handle: SessionHandle,
        _trigger: SessionEndTrigger,
    ) -> Result<(), HarnessError> {
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
    }

    #[test]
    fn end_session_persists_run() {
        let (_tmp, mut runner) = make_runner();
        let handle = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        runner
            .end_session(handle, SessionEndTrigger::Completed)
            .unwrap_or_else(|e| panic!("{e}"));
        let store = Arc::clone(&runner.store);
        let id = runner.run().id.clone();
        let loaded = store.load(&id).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(loaded.session_count, 1);
        assert_eq!(loaded.workspace_path, runner.run().workspace_path);
        assert_eq!(loaded.id, id);
    }
}
