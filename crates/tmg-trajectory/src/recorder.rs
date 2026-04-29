//! [`Recorder`]: JSONL trajectory writer.
//!
//! A [`Recorder`] owns one open file under
//! `.tsumugi/runs/<run-id>/trajectories/session_NNN.jsonl` and exposes
//! an `append`-style API. The implementation is deliberately
//! synchronous: writing one JSON line at a time is comparable in cost
//! to a single `tracing::warn!`, so spawning a background task adds
//! plumbing overhead without measurable benefit. When integrators
//! want fire-and-forget semantics they wrap the recorder in an
//! [`std::sync::Arc`] and drive it from inside the agent loop's
//! existing tokio task — the recorder takes `&self` everywhere via an
//! internal mutex, so the wrapper is shareable across sinks.
//!
//! # Trajectory tee pattern
//!
//! Issue #55 specifies the recorder must integrate alongside the
//! existing `--event-log` `TeeStreamSink` rather than spawn its own
//! second tee. The CLI wires this by using
//! [`TrajectoryStreamSink`](crate::sink::TrajectoryStreamSink), which
//! wraps an inner sink and also forwards to a `Recorder`. The same
//! sink composes with the event log: `TeeStreamSink<TrajectorySink<Inner>>`
//! gives one tee chain, three observers.
//!
//! # Disabled vs. enabled
//!
//! When `[trajectory] enabled = false`, the CLI must skip
//! [`Recorder::create`] altogether so no file is opened and no
//! records are written. The recorder itself does not consult the
//! enable flag — it is the integrator's responsibility to gate
//! construction. See [`crate::sink::TrajectoryStreamSink`] for the
//! optional-recorder convenience wrapper that no-ops when the
//! recorder is `None`.

use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use serde::Serialize;
use tracing::warn;

use crate::config::{ToolResultMode, TrajectoryConfig};
use crate::error::TrajectoryError;
use crate::record::{
    AssistantRecord, FeedbackRecord, MetaRecord, SystemRecord, ToolCallRecord, ToolResultRecord,
    TrajectoryRecord, UserRecord, VerdictRecord,
};
use crate::redact::Redactor;

/// Directory name (under each run dir) where trajectories live.
pub const TRAJECTORIES_DIRNAME: &str = "trajectories";

/// Construct the path of a session trajectory file under
/// `<run_dir>/<output_dir>/session_NNN.jsonl`.
#[must_use]
pub fn trajectory_path(run_dir: &Path, output_dir: &str, session_index: u32) -> PathBuf {
    run_dir
        .join(output_dir)
        .join(format!("session_{session_index:03}.jsonl"))
}

/// JSONL recorder for one session.
///
/// Construction opens the file in append mode; the directory is
/// created on demand. Appending a record acquires the internal
/// mutex, serialises the record, writes it as one line, and flushes
/// the file directly — every record is followed by a `flush()` call
/// so a process kill does not lose the tail. We deliberately use the
/// bare [`File`] (no [`std::io::BufWriter`]): per-line flushing makes
/// any buffer pointless and keeps the writer one syscall per record.
/// Errors are surfaced via the public methods so callers can decide
/// whether to abort or warn-and-continue (the typical agent-loop
/// path warns).
pub struct Recorder {
    inner: Mutex<RecorderState>,
    config: TrajectoryConfig,
    redactor: Redactor,
}

struct RecorderState {
    /// Raw file handle. We do not wrap in [`std::io::BufWriter`]
    /// because [`Recorder::write_line`] flushes after every record
    /// (durability matters more than throughput for trajectory
    /// recording), so a buffer would never amortise.
    writer: File,
    path: PathBuf,
}

impl Recorder {
    /// Open or create the trajectory file at `path` in append mode.
    ///
    /// `config` controls runtime knobs (whether to emit `thinking`,
    /// how to render tool results); `extra_patterns` is the
    /// pre-compiled extras for the redactor — passing them at
    /// construction time means we surface a regex-compile failure at
    /// recorder creation rather than first-write.
    ///
    /// # Errors
    ///
    /// Returns [`TrajectoryError::Io`] when the parent directory cannot
    /// be created or the file cannot be opened.
    pub fn create(
        path: impl AsRef<Path>,
        config: TrajectoryConfig,
    ) -> Result<Self, TrajectoryError> {
        let redactor = Redactor::with_extras(config.redact_extra_patterns.iter())?;
        Self::with_redactor(path, config, redactor)
    }

    /// Like [`Self::create`] but reuses an already-built [`Redactor`].
    /// Used by the export path so a single redactor is reused across
    /// all reconstructed sessions.
    pub fn with_redactor(
        path: impl AsRef<Path>,
        config: TrajectoryConfig,
        redactor: Redactor,
    ) -> Result<Self, TrajectoryError> {
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent() {
            std::fs::create_dir_all(parent).map_err(|e| TrajectoryError::io(parent, e))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path_ref)
            .map_err(|e| TrajectoryError::io(path_ref, e))?;
        Ok(Self {
            inner: Mutex::new(RecorderState {
                writer: file,
                path: path_ref.to_path_buf(),
            }),
            config,
            redactor,
        })
    }

    /// Borrow the path of the trajectory file. Useful for logging /
    /// diagnostics.
    #[must_use]
    pub fn path(&self) -> PathBuf {
        // Take a short lock and clone — the path never mutates after
        // construction so this is wait-free in the common case.
        match self.inner.lock() {
            Ok(g) => g.path.clone(),
            Err(p) => p.into_inner().path.clone(),
        }
    }

    /// Borrow the active configuration.
    #[must_use]
    pub fn config(&self) -> &TrajectoryConfig {
        &self.config
    }

    /// Append a fully-formed [`TrajectoryRecord`] verbatim. Callers
    /// that go through this method are responsible for redaction; the
    /// dedicated `record_*` helpers below redact transparently.
    ///
    /// # Errors
    ///
    /// Returns [`TrajectoryError`] on serialisation or I/O failure.
    pub fn append_raw(&self, record: &TrajectoryRecord) -> Result<(), TrajectoryError> {
        self.write_line(record)
    }

    /// Append a [`MetaRecord`] (no redaction needed — `MetaRecord`
    /// fields are all integer / enum / timestamp).
    ///
    /// # Errors
    /// Returns [`TrajectoryError`] on serialisation or I/O failure.
    pub fn record_meta(&self, meta: MetaRecord) -> Result<(), TrajectoryError> {
        self.write_line(&TrajectoryRecord::Meta(meta))
    }

    /// Append a [`SystemRecord`] with redaction applied.
    ///
    /// # Errors
    /// Returns [`TrajectoryError`] on serialisation or I/O failure.
    pub fn record_system(&self, content: &str) -> Result<(), TrajectoryError> {
        self.write_line(&TrajectoryRecord::System(SystemRecord {
            content: self.redactor.apply(content),
        }))
    }

    /// Append a [`UserRecord`] with redaction applied.
    ///
    /// # Errors
    /// Returns [`TrajectoryError`] on serialisation or I/O failure.
    pub fn record_user(&self, content: &str) -> Result<(), TrajectoryError> {
        self.write_line(&TrajectoryRecord::User(UserRecord {
            content: self.redactor.apply(content),
        }))
    }

    /// Append an [`AssistantRecord`].
    ///
    /// `thinking` is dropped when [`TrajectoryConfig::include_thinking`]
    /// is `false`. `content` and `thinking` are redacted; tool-call
    /// argument values are redacted by serialising the arguments
    /// `serde_json::Value` and re-parsing — strings inside the JSON
    /// tree get the same redactor pass.
    ///
    /// # Errors
    /// Returns [`TrajectoryError`] on serialisation or I/O failure.
    pub fn record_assistant(
        &self,
        content: &str,
        thinking: Option<&str>,
        tool_calls: &[ToolCallRecord],
    ) -> Result<(), TrajectoryError> {
        let thinking = if self.config.include_thinking {
            thinking.map(|t| self.redactor.apply(t))
        } else {
            None
        };
        let redacted_calls: Vec<ToolCallRecord> = tool_calls
            .iter()
            .map(|tc| ToolCallRecord {
                id: tc.id.clone(),
                name: tc.name.clone(),
                arguments: redact_value(&tc.arguments, &self.redactor),
            })
            .collect();
        self.write_line(&TrajectoryRecord::Assistant(AssistantRecord {
            content: self.redactor.apply(content),
            thinking,
            tool_calls: redacted_calls,
        }))
    }

    /// Append a [`ToolResultRecord`].
    ///
    /// The `output` payload is rendered according to
    /// [`TrajectoryConfig::include_tool_results`]:
    ///
    /// - [`ToolResultMode::Full`] — verbatim, no redaction. Use only
    ///   when the operator has accepted the full disclosure trade-off.
    /// - [`ToolResultMode::Redacted`] — pass through the redactor (the
    ///   default).
    /// - [`ToolResultMode::SummaryOnly`] — replaced by a one-line
    ///   `"<N chars, ok|err>"` placeholder. Useful for shipping
    ///   trajectories where the conversation flow matters but the
    ///   actual tool payloads are sensitive.
    ///
    /// # Errors
    /// Returns [`TrajectoryError`] on serialisation or I/O failure.
    pub fn record_tool_result(
        &self,
        tool_call_id: &str,
        tool_name: &str,
        output: &str,
        is_error: bool,
    ) -> Result<(), TrajectoryError> {
        let rendered = match self.config.include_tool_results {
            ToolResultMode::Full => output.to_owned(),
            ToolResultMode::Redacted => self.redactor.apply(output),
            ToolResultMode::SummaryOnly => {
                let tag = if is_error { "err" } else { "ok" };
                format!("<{} chars, {}>", output.chars().count(), tag)
            }
        };
        self.write_line(&TrajectoryRecord::ToolResult(ToolResultRecord {
            tool_call_id: tool_call_id.to_owned(),
            tool_name: tool_name.to_owned(),
            output: rendered,
            is_error,
        }))
    }

    /// Append a [`FeedbackRecord`] with redaction applied to `content`.
    ///
    /// # Errors
    /// Returns [`TrajectoryError`] on serialisation or I/O failure.
    pub fn record_feedback(&self, source: &str, content: &str) -> Result<(), TrajectoryError> {
        self.write_line(&TrajectoryRecord::Feedback(FeedbackRecord {
            source: source.to_owned(),
            content: self.redactor.apply(content),
        }))
    }

    /// Append a [`VerdictRecord`].
    ///
    /// # Errors
    /// Returns [`TrajectoryError`] on serialisation or I/O failure.
    pub fn record_verdict(
        &self,
        outcome: &str,
        feature_marked_passing: Vec<String>,
    ) -> Result<(), TrajectoryError> {
        self.write_line(&TrajectoryRecord::Verdict(VerdictRecord {
            outcome: outcome.to_owned(),
            feature_marked_passing,
        }))
    }

    /// Flush the underlying file. Useful for tests; production
    /// integrators don't need to call this — [`Self::write_line`]
    /// already flushes after every record, and the file is closed on
    /// drop.
    ///
    /// # Errors
    /// Returns [`TrajectoryError::Io`] on flush failure.
    pub fn flush(&self) -> Result<(), TrajectoryError> {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let path = guard.path.clone();
        guard
            .writer
            .flush()
            .map_err(|e| TrajectoryError::io(path, e))
    }

    fn write_line<T: Serialize>(&self, value: &T) -> Result<(), TrajectoryError> {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => {
                // Mutex poisoning means a previous writer panicked;
                // the file handle is still valid, so we recover the
                // inner state and continue. Surface a warning so the
                // operator can investigate the original panic.
                warn!("trajectory recorder mutex poisoned; recovering");
                p.into_inner()
            }
        };
        let path = guard.path.clone();
        let line = serde_json::to_string(value).map_err(TrajectoryError::Json)?;
        guard
            .writer
            .write_all(line.as_bytes())
            .map_err(|e| TrajectoryError::io(&path, e))?;
        guard
            .writer
            .write_all(b"\n")
            .map_err(|e| TrajectoryError::io(&path, e))?;
        // Flush per record so a process kill does not lose the tail.
        // This matches `EventLogWriter::write_event`.
        guard
            .writer
            .flush()
            .map_err(|e| TrajectoryError::io(&path, e))?;
        Ok(())
    }
}

impl std::fmt::Debug for Recorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recorder")
            .field("path", &self.path())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// Recursively redact every string node inside a JSON value. Numbers /
/// booleans / nulls pass through unchanged. Maps and arrays preserve
/// key order so trajectory diffs stay stable.
fn redact_value(value: &serde_json::Value, redactor: &Redactor) -> serde_json::Value {
    match value {
        serde_json::Value::String(s) => serde_json::Value::String(redactor.apply(s)),
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(|v| redact_value(v, redactor)).collect())
        }
        serde_json::Value::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            for (k, v) in map {
                out.insert(k.clone(), redact_value(v, redactor));
            }
            serde_json::Value::Object(out)
        }
        // Numbers / booleans / nulls have no redactable content.
        other => other.clone(),
    }
}

/// Best-effort timestamp helper used by the recorder's `MetaRecord`
/// builders so the integrator does not need to import chrono on every
/// call site.
#[must_use]
pub fn now_utc() -> DateTime<Utc> {
    Utc::now()
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use crate::record::{MetaRecord, ToolCallRecord};
    use tempfile::TempDir;

    fn read_lines(path: &Path) -> Vec<String> {
        let body = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{e}"));
        body.lines().map(str::to_owned).collect()
    }

    #[test]
    fn appends_jsonl_when_enabled() {
        let dir = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let path = dir.path().join("session_001.jsonl");
        let rec =
            Recorder::create(&path, TrajectoryConfig::default()).unwrap_or_else(|e| panic!("{e}"));
        rec.record_meta(MetaRecord {
            run_id: "abc12345".into(),
            session_num: 1,
            model: "qwen3.5-4b".into(),
            scope: "ad-hoc".into(),
            started_at: Utc::now(),
            ended_at: None,
            outcome: None,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        rec.record_system("you are tsumugi")
            .unwrap_or_else(|e| panic!("{e}"));
        rec.record_user("read Cargo.toml")
            .unwrap_or_else(|e| panic!("{e}"));
        rec.flush().unwrap_or_else(|e| panic!("{e}"));
        drop(rec);

        let lines = read_lines(&path);
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert!(lines[0].contains(r#""type":"meta""#));
        assert!(lines[1].contains(r#""type":"system""#));
        assert!(lines[2].contains(r#""type":"user""#));
    }

    /// Redaction fires on user / assistant / `tool_result` content.
    #[test]
    fn redacts_secrets_across_record_types() {
        let dir = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let path = dir.path().join("s.jsonl");
        let rec =
            Recorder::create(&path, TrajectoryConfig::default()).unwrap_or_else(|e| panic!("{e}"));
        rec.record_user("API key sk-abcdefghijklmnopqrstuvwxyz1234567890ABCD")
            .unwrap_or_else(|e| panic!("{e}"));
        rec.record_assistant("ghp_abcdefghijklmnopqrstuvwxyz0123456789AB", None, &[])
            .unwrap_or_else(|e| panic!("{e}"));
        rec.record_tool_result("call_1", "file_read", "AKIAIOSFODNN7EXAMPLE in env", false)
            .unwrap_or_else(|e| panic!("{e}"));
        rec.flush().unwrap_or_else(|e| panic!("{e}"));
        drop(rec);

        let body = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
        for needle in [
            "sk-abcdefghijkl",
            "ghp_abcdefghijkl",
            "AKIAIOSFODNN7EXAMPLE",
        ] {
            assert!(!body.contains(needle), "leaked {needle}: {body}");
        }
        assert!(body.contains("[REDACTED]"), "{body}");
    }

    /// `include_thinking = false` strips `thinking` from the assistant
    /// record entirely (no `null`, no `""`).
    #[test]
    fn include_thinking_false_strips_block() {
        let dir = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let path = dir.path().join("s.jsonl");
        let cfg = TrajectoryConfig {
            include_thinking: false,
            ..TrajectoryConfig::default()
        };
        let rec = Recorder::create(&path, cfg).unwrap_or_else(|e| panic!("{e}"));
        rec.record_assistant("hello", Some("internal reasoning"), &[])
            .unwrap_or_else(|e| panic!("{e}"));
        rec.flush().unwrap_or_else(|e| panic!("{e}"));
        drop(rec);

        let body = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
        assert!(!body.contains("internal reasoning"), "{body}");
        assert!(!body.contains("\"thinking\""), "{body}");
    }

    /// `include_thinking = true` keeps the block AND redacts secrets
    /// inside it.
    #[test]
    fn include_thinking_true_redacts_block() {
        let dir = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let path = dir.path().join("s.jsonl");
        let cfg = TrajectoryConfig {
            include_thinking: true,
            ..TrajectoryConfig::default()
        };
        let rec = Recorder::create(&path, cfg).unwrap_or_else(|e| panic!("{e}"));
        rec.record_assistant(
            "hello",
            Some("plan: use sk-abcdefghijklmnopqrstuvwxyz1234567890ABCD"),
            &[],
        )
        .unwrap_or_else(|e| panic!("{e}"));
        rec.flush().unwrap_or_else(|e| panic!("{e}"));
        drop(rec);

        let body = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
        assert!(body.contains("\"thinking\""), "{body}");
        assert!(!body.contains("sk-abcdefghijkl"), "{body}");
    }

    /// `include_tool_results = "summary_only"` produces a compact
    /// `"<N chars, ok>"` placeholder.
    #[test]
    fn summary_only_produces_compact_form() {
        let dir = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let path = dir.path().join("s.jsonl");
        let cfg = TrajectoryConfig {
            include_tool_results: ToolResultMode::SummaryOnly,
            ..TrajectoryConfig::default()
        };
        let rec = Recorder::create(&path, cfg).unwrap_or_else(|e| panic!("{e}"));
        let big = "x".repeat(5_000);
        rec.record_tool_result("c1", "file_read", &big, false)
            .unwrap_or_else(|e| panic!("{e}"));
        rec.flush().unwrap_or_else(|e| panic!("{e}"));
        drop(rec);

        let body = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
        // Body must NOT contain the full payload.
        assert!(!body.contains(&"x".repeat(100)), "{body}");
        // Must contain the placeholder.
        assert!(body.contains("<5000 chars, ok>"), "{body}");
    }

    /// `include_tool_results = "full"` keeps the payload verbatim
    /// (intentionally bypasses redaction).
    #[test]
    fn full_mode_keeps_payload_verbatim() {
        let dir = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let path = dir.path().join("s.jsonl");
        let cfg = TrajectoryConfig {
            include_tool_results: ToolResultMode::Full,
            ..TrajectoryConfig::default()
        };
        let rec = Recorder::create(&path, cfg).unwrap_or_else(|e| panic!("{e}"));
        rec.record_tool_result(
            "c1",
            "file_read",
            "secret token=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ab",
            false,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        rec.flush().unwrap_or_else(|e| panic!("{e}"));
        drop(rec);

        let body = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
        // Full mode does NOT redact.
        assert!(body.contains("token=ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "{body}");
    }

    /// Tool-call argument values get the redactor pass, but field
    /// keys are preserved.
    #[test]
    fn tool_call_arguments_are_redacted() {
        let dir = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let path = dir.path().join("s.jsonl");
        let rec =
            Recorder::create(&path, TrajectoryConfig::default()).unwrap_or_else(|e| panic!("{e}"));
        let args = serde_json::json!({
            "path": "/etc/secrets",
            "auth": "token=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789xx",
        });
        let calls = vec![ToolCallRecord {
            id: "c1".into(),
            name: "file_read".into(),
            arguments: args,
        }];
        rec.record_assistant("", None, &calls)
            .unwrap_or_else(|e| panic!("{e}"));
        rec.flush().unwrap_or_else(|e| panic!("{e}"));
        drop(rec);

        let body = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
        assert!(!body.contains("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "{body}");
        assert!(body.contains("/etc/secrets"), "{body}");
    }

    /// `extras` configured via `redact_extra_patterns` apply on top of
    /// the shared base set.
    #[test]
    fn extra_patterns_applied() {
        let dir = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let path = dir.path().join("s.jsonl");
        let cfg = TrajectoryConfig {
            redact_extra_patterns: vec!["INTERNAL-[A-Z]{4}-[0-9]{4}".into()],
            ..TrajectoryConfig::default()
        };
        let rec = Recorder::create(&path, cfg).unwrap_or_else(|e| panic!("{e}"));
        rec.record_user("see ticket INTERNAL-ABCD-1234 today")
            .unwrap_or_else(|e| panic!("{e}"));
        rec.flush().unwrap_or_else(|e| panic!("{e}"));
        drop(rec);

        let body = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
        assert!(!body.contains("INTERNAL-ABCD-1234"), "{body}");
        assert!(body.contains("[REDACTED]"), "{body}");
    }

    #[test]
    fn trajectory_path_format() {
        let p = trajectory_path(Path::new("/runs/abc"), "trajectories", 42);
        assert_eq!(p, PathBuf::from("/runs/abc/trajectories/session_042.jsonl"));
    }
}
