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
use tmg_skills::TurnOutcomeRecorder;
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
    /// Optional per-turn outcome recorder used by the autonomous skill
    /// pipeline (issue #54). When present, every `on_tool_result`
    /// pushes a [`tmg_skills::ToolCallOutcome`] into the recorder so
    /// the per-turn observer can build a
    /// [`tmg_skills::TurnSummary`] with per-call outcomes (rather
    /// than the aggregate count [`tmg_core::TurnSummary`] carries).
    /// `None` is the default, so non-skills callers pay nothing.
    skill_outcome_recorder: Option<Arc<TurnOutcomeRecorder>>,
}

impl<S> HarnessStreamSink<S> {
    /// Construct a new harness sink wrapping `inner`.
    pub fn new(inner: S, runner: Arc<Mutex<RunRunner>>) -> Self {
        Self {
            inner,
            runner,
            skill_outcome_recorder: None,
        }
    }

    /// Install the [`TurnOutcomeRecorder`] used by the autonomous skill
    /// pipeline (issue #54). The recorder is called from
    /// `on_tool_result` with the tool name and a `success` flag derived
    /// from `is_error`.
    #[must_use]
    pub fn with_skill_outcome_recorder(mut self, recorder: Arc<TurnOutcomeRecorder>) -> Self {
        self.skill_outcome_recorder = Some(recorder);
        self
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

    fn on_tool_call(
        &mut self,
        call_id: &str,
        name: &str,
        arguments: &str,
    ) -> Result<(), CoreError> {
        self.with_active_session(|s| {
            s.tool_calls_count = s.tool_calls_count.saturating_add(1);
        });
        // Issue #54: capture `use_skill` invocations so the per-turn
        // observer can drive [`SkillsRuntime::record_use_skill_outcome`]
        // with the actual skill name. We record here (on the call,
        // before the result) so a failed `use_skill` lookup still
        // counts as an attempt — the post-turn observer downgrades it
        // to a Failure outcome based on `turn_errored`.
        if let Some(recorder) = self.skill_outcome_recorder.as_ref() {
            if name == "use_skill" {
                if let Some(skill_name) = extract_skill_name(arguments) {
                    recorder.record_use_skill(&skill_name);
                }
            }
        }
        self.inner.on_tool_call(call_id, name, arguments)
    }

    fn on_tool_result(
        &mut self,
        call_id: &str,
        name: &str,
        output: &str,
        is_error: bool,
    ) -> Result<(), CoreError> {
        // Issue #54: record per-call outcome for the skill emergence
        // signals. We do this BEFORE the file-modification accounting
        // so a panic in `extract_modified_path` (theoretically
        // unreachable) cannot drop the outcome.
        if let Some(recorder) = self.skill_outcome_recorder.as_ref() {
            recorder.record_call(name, !is_error);
            // The `memory add` tool is the cue for the
            // `UserCorrection` trigger. We treat *any* successful
            // `memory` tool call as feedback memory written; the more
            // precise "only feedback-kind add" check would require
            // peeking inside the JSON, which is fragile. For the
            // signal collector this conservative over-trigger is
            // preferable to an under-trigger.
            if !is_error && name == "memory" {
                recorder.mark_feedback_memory_written();
            }
        }
        if !is_error && FILE_MODIFYING_TOOLS.contains(&name) {
            // Try to extract the actual modified path. Tools may emit
            // either structured JSON (`{"path": "..."}` /
            // `{"file_path": "..."}`) or a free-form success string
            // such as `Successfully wrote N bytes to '<path>'`. Fall
            // back to the coarse `(via {name})` marker only when no
            // path is recoverable.
            let entry = extract_modified_path(output).unwrap_or_else(|| format!("(via {name})"));
            self.with_active_session(|s| {
                if !s.files_modified.contains(&entry) {
                    s.files_modified.push(entry);
                }
            });
        }
        self.inner.on_tool_result(call_id, name, output, is_error)
    }

    fn on_tool_result_compressed(
        &mut self,
        call_id: &str,
        name: &str,
        symbol_count: usize,
    ) -> Result<(), CoreError> {
        self.inner
            .on_tool_result_compressed(call_id, name, symbol_count)
    }

    fn on_warning(&mut self, message: &str) -> Result<(), CoreError> {
        self.inner.on_warning(message)
    }
}

/// Pull `skill_name` out of a `use_skill` tool-call's JSON arguments.
///
/// Returns `None` when the arguments are not valid JSON, the
/// `skill_name` field is missing or non-string, or the value is empty.
/// The caller treats `None` as "not a recoverable `use_skill` call" and
/// drops the use-skill metric for that round.
fn extract_skill_name(arguments: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(arguments).ok()?;
    let raw = value.get("skill_name")?.as_str()?.trim();
    if raw.is_empty() {
        None
    } else {
        Some(raw.to_owned())
    }
}

/// Try to recover the modified file path from a tool's textual output.
///
/// The lookup tries, in order:
///
/// 1. Parse `output` as JSON and read the `path` or `file_path` string
///    field.
/// 2. Match the canonical success messages emitted by `file_write`
///    (`Successfully wrote N bytes to '<path>'`) and `file_patch`
///    (`Successfully patched '<path>'`).
///
/// Returns `None` when no path can be extracted; callers fall back to
/// a `(via {name})` marker.
fn extract_modified_path(output: &str) -> Option<String> {
    // 1. Structured JSON shape — used by future file tools that emit
    //    `{"path": "..."}` / `{"file_path": "..."}`.
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(output) {
        for key in ["path", "file_path"] {
            if let Some(s) = value.get(key).and_then(serde_json::Value::as_str) {
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_owned());
                }
            }
        }
    }

    // 2. Free-form success strings emitted by today's `file_write` /
    //    `file_patch` tools. Both put the path inside single quotes
    //    near the end of the line; pulling the *last* `'…'` segment is
    //    robust to wording differences.
    extract_quoted_path(output)
}

/// Return the contents of the last `'…'`-quoted span in `s`, if any.
///
/// Matches both `file_write`'s "...to '<path>'" and `file_patch`'s
/// "...patched '<path>'" wording without requiring a regex dependency.
fn extract_quoted_path(s: &str) -> Option<String> {
    let close = s.rfind('\'')?;
    let open = s[..close].rfind('\'')?;
    let candidate = &s[open + 1..close];
    if candidate.is_empty() {
        None
    } else {
        Some(candidate.to_owned())
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
        sink.on_tool_call("call_1", "file_read", "{}")
            .unwrap_or_else(|e| panic!("{e}"));
        sink.on_tool_call("call_2", "shell_exec", "{}")
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
        // Free-form success strings emitted by today's tools. The sink
        // pulls the path out of the trailing single-quoted segment.
        sink.on_tool_result(
            "call_1",
            "file_write",
            "Successfully wrote 11 bytes to '/tmp/foo.rs'",
            false,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        sink.on_tool_result(
            "call_2",
            "file_patch",
            "Successfully patched '/tmp/bar.rs'",
            false,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        // Errored writes should not count.
        sink.on_tool_result(
            "call_3",
            "file_write",
            "Successfully wrote 0 bytes to '/tmp/baz'",
            true,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        // Non-write tools should not count.
        sink.on_tool_result("call_4", "file_read", "ok", false)
            .unwrap_or_else(|e| panic!("{e}"));

        let guard = runner.lock().await;
        let session = guard
            .active_session()
            .unwrap_or_else(|| panic!("active session"));
        assert!(
            session.files_modified.contains(&"/tmp/foo.rs".to_owned()),
            "missing file_write path: {:?}",
            session.files_modified
        );
        assert!(
            session.files_modified.contains(&"/tmp/bar.rs".to_owned()),
            "missing file_patch path: {:?}",
            session.files_modified
        );
        assert_eq!(
            session.files_modified.len(),
            2,
            "should dedupe per path: {:?}",
            session.files_modified
        );
    }

    /// Path extraction succeeds for unstructured `file_write` output and
    /// dedupes when the same file is written twice within one session.
    #[tokio::test]
    async fn on_tool_result_dedupes_repeated_paths() {
        let (_tmp, runner) = make_runner();
        let mut sink = HarnessStreamSink::new(NullSink, Arc::clone(&runner));
        sink.on_tool_result(
            "call_1",
            "file_write",
            "Successfully wrote 5 bytes to '/tmp/dupe.rs'",
            false,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        sink.on_tool_result(
            "call_2",
            "file_write",
            "Successfully wrote 17 bytes to '/tmp/dupe.rs'",
            false,
        )
        .unwrap_or_else(|e| panic!("{e}"));

        let guard = runner.lock().await;
        let session = guard
            .active_session()
            .unwrap_or_else(|| panic!("active session"));
        assert_eq!(session.files_modified, vec!["/tmp/dupe.rs".to_owned()]);
    }

    /// Structured JSON outputs (the future shape of file tools) are
    /// preferred over the free-form regex when both are present.
    #[tokio::test]
    async fn on_tool_result_extracts_path_from_json() {
        let (_tmp, runner) = make_runner();
        let mut sink = HarnessStreamSink::new(NullSink, Arc::clone(&runner));
        sink.on_tool_result(
            "call_1",
            "file_write",
            r#"{"path": "/tmp/json_path.rs", "bytes": 42}"#,
            false,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        sink.on_tool_result(
            "call_2",
            "file_patch",
            r#"{"file_path": "/tmp/file_path_key.rs"}"#,
            false,
        )
        .unwrap_or_else(|e| panic!("{e}"));

        let guard = runner.lock().await;
        let session = guard
            .active_session()
            .unwrap_or_else(|| panic!("active session"));
        assert!(
            session
                .files_modified
                .contains(&"/tmp/json_path.rs".to_owned()),
            "{:?}",
            session.files_modified,
        );
        assert!(
            session
                .files_modified
                .contains(&"/tmp/file_path_key.rs".to_owned()),
            "{:?}",
            session.files_modified,
        );
    }

    /// When neither a JSON shape nor the canonical text format matches,
    /// the sink falls back to the coarse `(via {name})` marker so the
    /// session still records that *something* was modified.
    #[tokio::test]
    async fn on_tool_result_falls_back_when_path_unrecoverable() {
        let (_tmp, runner) = make_runner();
        let mut sink = HarnessStreamSink::new(NullSink, Arc::clone(&runner));
        sink.on_tool_result("call_1", "file_write", "no path here, sorry", false)
            .unwrap_or_else(|e| panic!("{e}"));
        let guard = runner.lock().await;
        let session = guard
            .active_session()
            .unwrap_or_else(|| panic!("active session"));
        assert_eq!(session.files_modified, vec!["(via file_write)".to_owned()]);
    }

    #[test]
    fn extract_quoted_path_works() {
        assert_eq!(
            extract_quoted_path("Successfully wrote 11 bytes to '/tmp/foo.rs'"),
            Some("/tmp/foo.rs".to_owned()),
        );
        assert_eq!(
            extract_quoted_path("Successfully patched '/tmp/bar.rs'"),
            Some("/tmp/bar.rs".to_owned()),
        );
        assert_eq!(extract_quoted_path("no quotes"), None);
        assert_eq!(extract_quoted_path("only one '"), None);
        assert_eq!(extract_quoted_path("empty ''"), None);
    }

    #[test]
    fn extract_modified_path_prefers_json_over_text() {
        // When both keys are present in JSON, `path` wins (it's listed
        // first in the lookup order).
        let json = r#"{"path": "/json", "file_path": "/other"}"#;
        assert_eq!(extract_modified_path(json), Some("/json".to_owned()));
    }

    #[test]
    fn extract_modified_path_handles_text_fallback() {
        let text = "Successfully wrote 11 bytes to '/tmp/x.rs'";
        assert_eq!(extract_modified_path(text), Some("/tmp/x.rs".to_owned()));
    }

    /// Mirrors the production composition used by the TUI: an inner
    /// channel-style forwarding sink wrapped with `HarnessStreamSink`.
    /// After fake tool events flow through the wrapper, the runner's
    /// active session sees both `tool_calls_count` and `files_modified`
    /// updated, and the inner sink continues to receive every callback.
    #[tokio::test]
    async fn wrapping_inner_sink_updates_runner_and_forwards() {
        use std::sync::Mutex as StdMutex;

        // A test sink that records every callback so we can assert
        // forwarding semantics (no events swallowed by the decorator).
        #[derive(Default)]
        struct RecordingSink {
            events: Arc<StdMutex<Vec<String>>>,
        }
        impl StreamSink for RecordingSink {
            fn on_token(&mut self, token: &str) -> Result<(), CoreError> {
                self.events
                    .lock()
                    .unwrap_or_else(|e| panic!("{e}"))
                    .push(format!("token:{token}"));
                Ok(())
            }
            fn on_tool_call(
                &mut self,
                _call_id: &str,
                name: &str,
                _args: &str,
            ) -> Result<(), CoreError> {
                self.events
                    .lock()
                    .unwrap_or_else(|e| panic!("{e}"))
                    .push(format!("call:{name}"));
                Ok(())
            }
            fn on_tool_result(
                &mut self,
                _call_id: &str,
                name: &str,
                _output: &str,
                _is_error: bool,
            ) -> Result<(), CoreError> {
                self.events
                    .lock()
                    .unwrap_or_else(|e| panic!("{e}"))
                    .push(format!("result:{name}"));
                Ok(())
            }
        }

        let (_tmp, runner) = make_runner();
        let events = Arc::new(StdMutex::new(Vec::new()));
        let inner = RecordingSink {
            events: Arc::clone(&events),
        };
        let mut wrapped = HarnessStreamSink::new(inner, Arc::clone(&runner));

        wrapped.on_token("hello").unwrap_or_else(|e| panic!("{e}"));
        wrapped
            .on_tool_call("call_1", "file_write", "{}")
            .unwrap_or_else(|e| panic!("{e}"));
        wrapped
            .on_tool_result(
                "call_1",
                "file_write",
                "Successfully wrote 5 bytes to '/tmp/wired.rs'",
                false,
            )
            .unwrap_or_else(|e| panic!("{e}"));

        // Inner sink must have received every event in order.
        let recorded = events.lock().unwrap_or_else(|e| panic!("{e}")).clone();
        assert_eq!(
            recorded,
            vec![
                "token:hello".to_owned(),
                "call:file_write".to_owned(),
                "result:file_write".to_owned(),
            ],
        );

        // And the decorator must have updated the runner's active session.
        let guard = runner.lock().await;
        let session = guard
            .active_session()
            .unwrap_or_else(|| panic!("active session"));
        assert_eq!(session.tool_calls_count, 1);
        assert_eq!(session.files_modified, vec!["/tmp/wired.rs".to_owned()]);
    }
}
