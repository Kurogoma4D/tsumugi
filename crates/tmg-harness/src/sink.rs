//! [`StreamSink`] decorator that writes per-session statistics back to
//! the active [`RunRunner`].
//!
//! The agent loop drives a single [`StreamSink`] per turn; wrapping the
//! TUI / stdout sink with [`HarnessStreamSink`] gives the harness a
//! lightweight observation hook without needing a second event channel.
//! The decorator forwards every callback to the inner sink first
//! (preserving existing behaviour) and then updates the active session
//! record on a best-effort basis. Failures while updating the session
//! are logged as warnings so a transient lock contention (e.g. tools
//! holding the runner lock) cannot abort an in-flight LLM turn.

use std::sync::Arc;

use tmg_core::{CoreError, StreamSink};
use tokio::sync::Mutex;

use crate::runner::RunRunner;

/// Tool names whose successful execution corresponds to a write to the
/// workspace. We use a name-based heuristic rather than introspection so
/// that custom tools that follow the same naming convention are picked
/// up automatically.
const FILE_MODIFYING_TOOLS: &[&str] = &["file_write", "file_patch"];

/// Wraps an inner [`StreamSink`] and updates the active session's
/// statistics on every event.
///
/// The runner is held behind `Arc<Mutex<...>>` so the same lock can be
/// shared with the Run-scoped tools (`progress_append`,
/// `session_summary_save`); using a try-lock here keeps the streaming
/// path non-blocking.
pub struct HarnessStreamSink<S> {
    inner: S,
    runner: Arc<Mutex<RunRunner>>,
}

impl<S> HarnessStreamSink<S> {
    /// Construct a new harness sink wrapping `inner`.
    pub fn new(inner: S, runner: Arc<Mutex<RunRunner>>) -> Self {
        Self { inner, runner }
    }

    /// Apply `update` to the active session if the runner lock can be
    /// acquired immediately. Logs a warning otherwise; never blocks.
    fn with_active_session<F>(&self, update: F)
    where
        F: FnOnce(&mut crate::session::Session),
    {
        match self.runner.try_lock() {
            Ok(mut guard) => {
                if let Some(session) = guard.active_session_mut() {
                    update(session);
                }
            }
            Err(_) => {
                tracing::trace!(
                    "harness sink could not acquire runner lock; session stats skipped for one event"
                );
            }
        }
    }
}

impl<S: StreamSink> StreamSink for HarnessStreamSink<S> {
    fn on_thinking(&mut self, token: &str) -> Result<(), CoreError> {
        self.inner.on_thinking(token)
    }

    fn on_token(&mut self, token: &str) -> Result<(), CoreError> {
        self.inner.on_token(token)
    }

    fn on_done(&mut self) -> Result<(), CoreError> {
        self.inner.on_done()
    }

    fn on_tool_call(&mut self, name: &str, arguments: &str) -> Result<(), CoreError> {
        self.with_active_session(|s| {
            s.tool_calls_count = s.tool_calls_count.saturating_add(1);
        });
        self.inner.on_tool_call(name, arguments)
    }

    fn on_tool_result(
        &mut self,
        name: &str,
        output: &str,
        is_error: bool,
    ) -> Result<(), CoreError> {
        if !is_error && FILE_MODIFYING_TOOLS.contains(&name) {
            // Best-effort: the tool's `output` does not currently encode
            // a structured "modified path" field, so we record the tool
            // name as a coarse marker. Refining this once the file
            // tools surface paths is tracked as a follow-up.
            let marker = format!("(via {name})");
            self.with_active_session(|s| {
                if !s.files_modified.contains(&marker) {
                    s.files_modified.push(marker);
                }
            });
        }
        self.inner.on_tool_result(name, output, is_error)
    }

    fn on_warning(&mut self, message: &str) -> Result<(), CoreError> {
        self.inner.on_warning(message)
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use crate::store::RunStore;

    struct NullSink;

    impl StreamSink for NullSink {
        fn on_token(&mut self, _token: &str) -> Result<(), CoreError> {
            Ok(())
        }
    }

    fn make_runner() -> (tempfile::TempDir, Arc<Mutex<RunRunner>>) {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let store = Arc::new(RunStore::new(tmp.path().join("runs")));
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut runner = RunRunner::new(run, store);
        let _ = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        (tmp, Arc::new(Mutex::new(runner)))
    }

    #[tokio::test]
    async fn on_tool_call_increments_count() {
        let (_tmp, runner) = make_runner();
        let mut sink = HarnessStreamSink::new(NullSink, Arc::clone(&runner));
        sink.on_tool_call("file_read", "{}")
            .unwrap_or_else(|e| panic!("{e}"));
        sink.on_tool_call("shell_exec", "{}")
            .unwrap_or_else(|e| panic!("{e}"));
        let guard = runner.lock().await;
        let session = guard
            .active_session()
            .unwrap_or_else(|| panic!("active session"));
        assert_eq!(session.tool_calls_count, 2);
    }

    #[tokio::test]
    async fn on_tool_result_records_file_modifications() {
        let (_tmp, runner) = make_runner();
        let mut sink = HarnessStreamSink::new(NullSink, Arc::clone(&runner));
        sink.on_tool_result("file_write", "ok", false)
            .unwrap_or_else(|e| panic!("{e}"));
        sink.on_tool_result("file_patch", "ok", false)
            .unwrap_or_else(|e| panic!("{e}"));
        // Errored writes should not count.
        sink.on_tool_result("file_write", "boom", true)
            .unwrap_or_else(|e| panic!("{e}"));
        // Non-write tools should not count.
        sink.on_tool_result("file_read", "ok", false)
            .unwrap_or_else(|e| panic!("{e}"));

        let guard = runner.lock().await;
        let session = guard
            .active_session()
            .unwrap_or_else(|| panic!("active session"));
        assert!(
            session
                .files_modified
                .iter()
                .any(|m| m.contains("file_write")),
            "missing file_write: {:?}",
            session.files_modified
        );
        assert!(
            session
                .files_modified
                .iter()
                .any(|m| m.contains("file_patch")),
            "missing file_patch: {:?}",
            session.files_modified
        );
        assert_eq!(
            session.files_modified.len(),
            2,
            "should dedupe by tool name: {:?}",
            session.files_modified
        );
    }
}
