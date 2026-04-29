//! [`TrajectorySink`] trait and the [`TrajectoryStreamSink`] adapter
//! that wires a [`Recorder`] into the agent loop's
//! [`tmg_core::StreamSink`] tee chain.
//!
//! ## Why two abstractions?
//!
//! - [`TrajectorySink`] is the **public interface** integrators code
//!   against. It accepts a [`crate::record::TrajectoryRecord`] and
//!   nothing else: useful for unit-testing recorder consumers, for
//!   the export path (which mints records out of `session_log/*.json`
//!   rather than from a live LLM stream), and for any future consumer
//!   that wants to plug in alongside or instead of a recorder.
//!
//! - [`TrajectoryStreamSink`] is the **agent-loop wire-up**. It
//!   implements [`tmg_core::StreamSink`] so the existing tee chain
//!   (`TeeStreamSink<TrajectoryStreamSink<Inner>>`) works without
//!   changes to `tmg-core`. The wire-up translates `on_token` /
//!   `on_tool_call` / `on_tool_result` callbacks into the right
//!   [`crate::record::TrajectoryRecord`] variants.
//!
//! Both abstractions accept an optional recorder so callers can use
//! the same wiring whether or not `[trajectory] enabled` is set —
//! the disabled case is a no-op.

use std::sync::{Arc, Mutex};

use tmg_core::{CoreError, StreamSink};

use crate::error::TrajectoryError;
use crate::record::{ToolCallRecord, TrajectoryRecord};
use crate::recorder::Recorder;

/// Public interface for trajectory consumers.
///
/// Each `append` must be cheap: implementations are called from
/// inside the agent loop's hot path, and a slow / blocking sink
/// would back-pressure the LLM stream.
pub trait TrajectorySink: Send + Sync {
    /// Append a fully-formed record. Implementations decide whether
    /// to redact, drop, or persist; the caller treats the call as
    /// fire-and-forget.
    fn append(&self, record: TrajectoryRecord);
}

/// [`TrajectorySink`] backed by a [`Recorder`].
///
/// Errors from the recorder are logged via `tracing::warn!` and
/// swallowed: a transient I/O failure on the trajectory path must
/// not abort the live conversation.
#[derive(Debug, Clone)]
pub struct RecorderSink {
    recorder: Arc<Recorder>,
}

impl RecorderSink {
    /// Wrap a [`Recorder`] in a [`TrajectorySink`].
    #[must_use]
    pub fn new(recorder: Arc<Recorder>) -> Self {
        Self { recorder }
    }

    /// Borrow the wrapped recorder so callers can drive it directly
    /// (e.g. for a `record_meta` call at session start, before any
    /// stream events have fired).
    #[must_use]
    pub fn recorder(&self) -> Arc<Recorder> {
        Arc::clone(&self.recorder)
    }
}

impl TrajectorySink for RecorderSink {
    fn append(&self, record: TrajectoryRecord) {
        if let Err(e) = self.recorder.append_raw(&record) {
            tracing::warn!(error = %e, "trajectory append failed");
        }
    }
}

/// Wraps an inner [`StreamSink`] and forwards every event into a
/// [`Recorder`] (when present), buffering per-turn assistant state so
/// the trajectory captures the canonical
/// `(content, thinking, tool_calls)` triple as one record per round.
///
/// # Buffering rationale
///
/// The agent loop streams text/thinking incrementally and only knows
/// the full assistant text + tool calls at `on_done`. We buffer the
/// streaming deltas inside a [`Mutex`] until the round closes, then
/// emit one [`crate::record::AssistantRecord`] per round and
/// [`crate::record::ToolResultRecord`] per `on_tool_result` callback.
/// This mirrors the on-disk format requested in issue #55:
/// the JSONL has one assistant record per LLM round, not one per
/// token.
///
/// # Tool-call lookup
///
/// `on_tool_call` fires once per call before dispatch. We park the
/// `(id, name, arguments)` triple in an internal map keyed by
/// `call_id` so the matching `on_tool_result` can recover the tool
/// name even when a sink is given the result first (rare, but
/// possible if the inner sink reorders).
pub struct TrajectoryStreamSink<S> {
    inner: S,
    recorder: Option<Arc<Recorder>>,
    buf: Mutex<TurnBuffer>,
}

#[derive(Default)]
struct TurnBuffer {
    /// Accumulated reasoning tokens for the current round.
    thinking: String,
    /// Accumulated text content tokens for the current round.
    content: String,
    /// Tool calls emitted this round (`on_tool_call` events).
    tool_calls: Vec<ToolCallRecord>,
}

impl TurnBuffer {
    fn take(&mut self) -> Self {
        Self {
            thinking: std::mem::take(&mut self.thinking),
            content: std::mem::take(&mut self.content),
            tool_calls: std::mem::take(&mut self.tool_calls),
        }
    }
}

/// Lock the per-turn buffer, recovering from poison.
///
/// A poisoned mutex means a previous holder panicked while owning the
/// buffer. The buffer itself is plain [`String`] / [`Vec`] state with
/// no broken invariants we know about, so dropping events on the
/// floor is the wrong default — that would cause the trajectory to
/// silently miss tokens / tool calls. Instead we surface a warning
/// and recover the inner state with [`std::sync::PoisonError::into_inner`]
/// so the sink keeps observing the conversation.
fn lock_turn_buffer(m: &Mutex<TurnBuffer>) -> std::sync::MutexGuard<'_, TurnBuffer> {
    m.lock().unwrap_or_else(|poison| {
        tracing::warn!("trajectory turn buffer mutex poisoned; recovering");
        poison.into_inner()
    })
}

impl<S> TrajectoryStreamSink<S> {
    /// Create a tee that forwards every event to `inner` and (when
    /// `recorder` is `Some`) writes a corresponding trajectory record.
    pub fn new(inner: S, recorder: Option<Arc<Recorder>>) -> Self {
        Self {
            inner,
            recorder,
            buf: Mutex::new(TurnBuffer::default()),
        }
    }

    /// Borrow the inner sink. Useful for tests that want to inspect
    /// the wrapped sink's state.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Drop the wrapper and return the inner sink.
    pub fn into_inner(self) -> S {
        self.inner
    }

    /// Borrow the optional recorder so callers can drive it directly
    /// (e.g. to write a `MetaRecord` at session start or a
    /// `FeedbackRecord` from a slash-command path that does not flow
    /// through `StreamSink`).
    #[must_use]
    pub fn recorder(&self) -> Option<Arc<Recorder>> {
        self.recorder.as_ref().map(Arc::clone)
    }

    fn record_assistant(&self, buf: &TurnBuffer) {
        let Some(recorder) = self.recorder.as_ref() else {
            return;
        };
        // Skip empty assistant rounds (e.g. tool-only rounds with no
        // text and no thinking carry no information beyond the
        // tool_calls themselves; we keep them so the assistant ->
        // tool_result -> assistant flow is preserved in trajectory
        // form even when the model only emits tool calls).
        if buf.content.is_empty() && buf.thinking.is_empty() && buf.tool_calls.is_empty() {
            return;
        }
        let thinking = if buf.thinking.is_empty() {
            None
        } else {
            Some(buf.thinking.as_str())
        };
        if let Err(e) = recorder.record_assistant(&buf.content, thinking, &buf.tool_calls) {
            tracing::warn!(error = %e, "trajectory assistant record failed");
        }
    }
}

// NOTE: any new method added to `tmg_core::StreamSink` MUST be
// implemented (and forwarded) here as well — otherwise the trajectory
// recorder will silently miss the new event kind and the on-disk
// JSONL becomes incomplete relative to the live conversation. The
// `on_*` methods below cover every event the agent loop currently
// emits; adding a new kind is a deliberate change to both `tmg-core`
// and this impl.
impl<S: StreamSink> StreamSink for TrajectoryStreamSink<S> {
    fn on_thinking(&mut self, token: &str) -> Result<(), CoreError> {
        if self.recorder.is_some() {
            let mut g = lock_turn_buffer(&self.buf);
            g.thinking.push_str(token);
        }
        self.inner.on_thinking(token)
    }

    fn on_token(&mut self, token: &str) -> Result<(), CoreError> {
        if self.recorder.is_some() {
            let mut g = lock_turn_buffer(&self.buf);
            g.content.push_str(token);
        }
        self.inner.on_token(token)
    }

    fn on_done(&mut self) -> Result<(), CoreError> {
        // Take the buffer first, write the assistant record, then
        // forward `on_done` so any state the inner sink mutates does
        // not race with our recorder.
        if self.recorder.is_some() {
            let taken = lock_turn_buffer(&self.buf).take();
            self.record_assistant(&taken);
        }
        self.inner.on_done()
    }

    fn on_tool_call(
        &mut self,
        call_id: &str,
        name: &str,
        arguments: &str,
    ) -> Result<(), CoreError> {
        if self.recorder.is_some() {
            let mut g = lock_turn_buffer(&self.buf);
            // Try to parse the arguments as JSON; if the model emitted
            // something malformed, keep the raw string so the record
            // still carries the literal payload for offline triage.
            let args_val = serde_json::from_str::<serde_json::Value>(arguments)
                .unwrap_or_else(|_| serde_json::Value::String(arguments.to_owned()));
            g.tool_calls.push(ToolCallRecord {
                id: call_id.to_owned(),
                name: name.to_owned(),
                arguments: args_val,
            });
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
        if let Some(recorder) = self.recorder.as_ref()
            && let Err(e) = recorder.record_tool_result(call_id, name, output, is_error)
        {
            tracing::warn!(error = %e, "trajectory tool_result record failed");
        }
        self.inner.on_tool_result(call_id, name, output, is_error)
    }

    fn on_tool_result_compressed(
        &mut self,
        call_id: &str,
        name: &str,
        symbol_count: usize,
    ) -> Result<(), CoreError> {
        // Compression metadata is informational only; we do not emit
        // a trajectory record for it because the post-compression
        // payload was already captured in `on_tool_result`.
        self.inner
            .on_tool_result_compressed(call_id, name, symbol_count)
    }

    fn on_warning(&mut self, message: &str) -> Result<(), CoreError> {
        // Surfaces (memory swap, auto-compression failure) are not
        // training signal; forward to the inner sink only.
        self.inner.on_warning(message)
    }
}

/// Convenience: a no-op [`TrajectorySink`] used in tests and in CLI
/// branches where the recorder is disabled but a sink type is still
/// required.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoOpSink;

impl TrajectorySink for NoOpSink {
    fn append(&self, _record: TrajectoryRecord) {}
}

/// Drive a [`TrajectorySink`] to flush any internal buffers.
///
/// Implementors that batch writes can override this; the default is
/// a no-op so tests can call `flush` unconditionally.
///
/// # Errors
///
/// Returns [`TrajectoryError`] when the recorder fails to flush.
pub fn flush_sink(sink: &dyn AnyRecorder) -> Result<(), TrajectoryError> {
    sink.flush()
}

/// Internal trait used by [`flush_sink`]; lets the CLI poke at the
/// recorder without leaking the concrete type.
pub trait AnyRecorder {
    /// Flush the underlying writer.
    ///
    /// # Errors
    /// Returns [`TrajectoryError`] when the writer fails to flush.
    fn flush(&self) -> Result<(), TrajectoryError>;
}

impl AnyRecorder for Recorder {
    fn flush(&self) -> Result<(), TrajectoryError> {
        Recorder::flush(self)
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use crate::config::TrajectoryConfig;
    use tempfile::TempDir;

    /// A no-op inner sink for the `TrajectoryStreamSink` tests.
    struct InnerSink;
    impl StreamSink for InnerSink {
        fn on_token(&mut self, _t: &str) -> Result<(), CoreError> {
            Ok(())
        }
    }

    #[test]
    fn stream_sink_emits_one_assistant_record_per_round() {
        let dir = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let path = dir.path().join("s.jsonl");
        let rec = Arc::new(
            Recorder::create(&path, TrajectoryConfig::default()).unwrap_or_else(|e| panic!("{e}")),
        );
        let mut sink = TrajectoryStreamSink::new(InnerSink, Some(Arc::clone(&rec)));
        sink.on_thinking("plan: ").unwrap_or_else(|e| panic!("{e}"));
        sink.on_thinking("read file")
            .unwrap_or_else(|e| panic!("{e}"));
        sink.on_token("Reading ").unwrap_or_else(|e| panic!("{e}"));
        sink.on_token("Cargo.toml")
            .unwrap_or_else(|e| panic!("{e}"));
        sink.on_tool_call("call_1", "file_read", r#"{"path":"Cargo.toml"}"#)
            .unwrap_or_else(|e| panic!("{e}"));
        sink.on_done().unwrap_or_else(|e| panic!("{e}"));
        sink.on_tool_result("call_1", "file_read", "[workspace]", false)
            .unwrap_or_else(|e| panic!("{e}"));
        // Second round (post tool result).
        sink.on_token("done").unwrap_or_else(|e| panic!("{e}"));
        sink.on_done().unwrap_or_else(|e| panic!("{e}"));
        rec.flush().unwrap_or_else(|e| panic!("{e}"));

        let body = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
        let lines: Vec<&str> = body.lines().collect();
        // 1 assistant (with tool_call) + 1 tool_result + 1 assistant.
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert!(lines[0].contains(r#""type":"assistant""#), "{}", lines[0]);
        assert!(lines[0].contains("Reading Cargo.toml"), "{}", lines[0]);
        // include_thinking=false by default -> thinking stripped.
        assert!(!lines[0].contains("plan: read file"), "{}", lines[0]);
        assert!(lines[0].contains("file_read"), "{}", lines[0]);
        assert!(lines[1].contains(r#""type":"tool_result""#), "{}", lines[1]);
        assert!(lines[2].contains(r#""type":"assistant""#), "{}", lines[2]);
        assert!(lines[2].contains("done"), "{}", lines[2]);
    }

    /// When the recorder is `None`, the sink writes nothing.
    #[test]
    fn disabled_sink_writes_nothing() {
        let mut sink: TrajectoryStreamSink<InnerSink> = TrajectoryStreamSink::new(InnerSink, None);
        sink.on_token("ignored").unwrap_or_else(|e| panic!("{e}"));
        sink.on_done().unwrap_or_else(|e| panic!("{e}"));
        // Nothing to assert beyond "no panic, no I/O" — the empty
        // recorder slot guarantees no file is touched.
    }
}
