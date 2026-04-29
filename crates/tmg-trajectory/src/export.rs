//! Reconstruct trajectories from existing on-disk artifacts.
//!
//! When the `[trajectory]` feature was enabled mid-project, earlier
//! runs only have `session_log/session_NNN.json` (and, if the user ran
//! with `--event-log`, the corresponding event log) on disk. This
//! module rebuilds an approximate trajectory from those artifacts.
//!
//! ## Granularity tradeoff
//!
//! - When **only** `session_log/*.json` is present, the export
//!   collapses each session into a `meta` + `system` (placeholder) +
//!   `user` (the recorded summary) + `verdict` quadruple. This is
//!   marked "summary-only" in the meta record's `outcome` field so
//!   downstream converters can decide whether to drop it.
//!
//! - When the matching `event_log/*.jsonl` (issue #50 / #52) is
//!   present, the export replays the event stream into per-round
//!   `assistant` + `tool_call` + `tool_result` records. The event
//!   log does not carry the original user message, so we still
//!   borrow `session.summary` for one synthetic `user` record at the
//!   top of the file.
//!
//! Either way the reconstructed file is a *close approximation*, not
//! a verbatim copy of what the live recorder would have produced.
//! That is acceptable for the SFT/RL training use-case where coarse
//! conversation flow is more useful than no signal at all.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::Deserialize;
use tmg_harness::{Run, RunStore, SessionLog};

use crate::config::TrajectoryConfig;
use crate::error::TrajectoryError;
use crate::record::{
    MetaRecord, SystemRecord, ToolCallRecord, ToolResultRecord, TrajectoryRecord, UserRecord,
    VerdictRecord,
};
use crate::recorder::{Recorder, trajectory_path};
use crate::redact::Redactor;

/// Filter for `tmg trajectory export` selecting which runs to process.
///
/// Mirrors the CLI flags: `--run <id>`, `--all`, `--since <ISO8601>`.
#[derive(Debug, Clone, Default)]
pub struct ExportFilter {
    /// Specific run id to export. `None` means "use the other knobs".
    pub run_id: Option<String>,
    /// When `true`, every run in the store is exported.
    pub all: bool,
    /// Lower bound on `Run::created_at`. When `Some` and `all = false`,
    /// only runs newer than this are exported.
    pub since: Option<DateTime<Utc>>,
}

/// Outcome of a single export run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportSummary {
    /// Run id we exported for.
    pub run_id: String,
    /// Number of session trajectories written.
    pub sessions_exported: u32,
    /// `true` when the export found at least one `session_log` entry
    /// without an `event_log` neighbour and used summary-only fallback.
    pub had_summary_only_fallback: bool,
}

/// Top-level entry point for `tmg trajectory export`.
///
/// `out_dir` is the destination root: each exported session is written
/// under `<out_dir>/<run-id>/session_NNN.jsonl`. When `out_dir` is
/// `None`, the CLI default is `<run_dir>/<config.output_dir>/` (i.e.
/// the same location the live recorder would use), so re-export is
/// idempotent.
///
/// # Errors
///
/// Returns [`TrajectoryError`] on filesystem / JSON failure or
/// configuration errors.
pub fn export(
    store: &RunStore,
    filter: &ExportFilter,
    out_dir: Option<&Path>,
    config: &TrajectoryConfig,
) -> Result<Vec<ExportSummary>, TrajectoryError> {
    let runs = collect_runs(store, filter)?;
    let redactor = Redactor::with_extras(config.redact_extra_patterns.iter())?;
    let mut summaries = Vec::with_capacity(runs.len());
    for run in runs {
        let run_dir = store.run_dir(&run.id);
        let dest = match out_dir {
            Some(d) => d.join(run.id.as_str()),
            None => config.resolve_output_dir(&run_dir),
        };
        let summary = export_one_run(&run, &run_dir, &dest, config, &redactor)?;
        summaries.push(summary);
    }
    Ok(summaries)
}

fn collect_runs(store: &RunStore, filter: &ExportFilter) -> Result<Vec<Run>, TrajectoryError> {
    if let Some(id) = &filter.run_id {
        let parsed = tmg_harness::RunId::parse(id.clone())
            .map_err(|e| TrajectoryError::InvalidConfig(format!("invalid run id {id:?}: {e}")))?;
        let run = store
            .load(&parsed)
            .map_err(|e| TrajectoryError::InvalidConfig(format!("loading run {id:?}: {e}")))?;
        return Ok(vec![run]);
    }
    let summaries = store
        .list()
        .map_err(|e| TrajectoryError::InvalidConfig(format!("listing runs: {e}")))?;
    let mut out = Vec::with_capacity(summaries.len());
    for summary in summaries {
        if let Some(since) = filter.since
            && summary.created_at < since
        {
            continue;
        }
        match store.load(&summary.id) {
            Ok(run) => out.push(run),
            Err(e) => {
                tracing::warn!(run_id = %summary.id, error = %e, "skipping run that failed to load");
            }
        }
    }
    Ok(out)
}

fn export_one_run(
    run: &Run,
    run_dir: &Path,
    dest_dir: &Path,
    config: &TrajectoryConfig,
    redactor: &Redactor,
) -> Result<ExportSummary, TrajectoryError> {
    std::fs::create_dir_all(dest_dir).map_err(|e| TrajectoryError::io(dest_dir, e))?;
    let session_log = SessionLog::new(run_dir.join("session_log"));
    let entries = session_log
        .list()
        .map_err(|e| TrajectoryError::InvalidConfig(format!("listing session log: {e}")))?;
    let mut had_fallback = false;
    let mut count: u32 = 0;
    for entry in entries {
        let session = match session_log.load(entry.index) {
            Ok(Some(s)) => s,
            Ok(None) => continue,
            Err(e) => {
                tracing::warn!(index = entry.index, error = %e, "skipping unreadable session log");
                continue;
            }
        };
        let event_log_path = candidate_event_log(run_dir, entry.index);
        let traj_path = dest_dir.join(format!("session_{:03}.jsonl", entry.index));
        // Open per-session recorder. Reuse the redactor so we don't
        // recompile the regex set per session.
        let recorder = Recorder::with_redactor(&traj_path, config.clone(), redactor.clone())?;

        let scope_label = run.scope.label().to_owned();
        let model = "unknown".to_owned(); // session_log does not record the model name.
        recorder.record_meta(MetaRecord {
            run_id: run.id.as_str().to_owned(),
            session_num: session.index,
            model,
            scope: scope_label,
            started_at: session.started_at,
            ended_at: session.ended_at,
            outcome: session
                .end_trigger
                .as_ref()
                .map(super::record_outcome_label),
        })?;
        recorder.record_system(
            "(reconstructed from session_log; original system prompt unavailable)",
        )?;
        if !session.summary.is_empty() {
            recorder.record_user(&session.summary)?;
        }
        if let Some(path) = event_log_path.as_ref().filter(|p| p.exists()) {
            replay_event_log(path, &recorder)?;
        } else {
            had_fallback = true;
        }
        recorder.record_verdict(
            &session
                .end_trigger
                .as_ref()
                .map_or_else(|| "unknown".to_owned(), super::record_outcome_label),
            session.features_marked_passing.clone(),
        )?;
        recorder.flush()?;
        count = count.saturating_add(1);
    }
    Ok(ExportSummary {
        run_id: run.id.as_str().to_owned(),
        sessions_exported: count,
        had_summary_only_fallback: had_fallback,
    })
}

/// Best-effort lookup for an event-log neighbour of a session.
///
/// Convention: a TUI / CLI session writes its event log to a path
/// supplied by the user (`--event-log <path>`); when the path lives
/// inside the run directory we can find it. Searched names:
///
/// - `<run_dir>/event_log/session_NNN.jsonl`
/// - `<run_dir>/event_log_NNN.jsonl`
/// - `<run_dir>/events.jsonl` (single-file fallback for the most
///   recent session)
fn candidate_event_log(run_dir: &Path, session_index: u32) -> Option<PathBuf> {
    let candidates = [
        run_dir
            .join("event_log")
            .join(format!("session_{session_index:03}.jsonl")),
        run_dir.join(format!("event_log_{session_index:03}.jsonl")),
        run_dir.join("events.jsonl"),
    ];
    candidates.into_iter().find(|c| c.exists())
}

/// Minimal subset of [`tmg_core::EventLogWriter`]'s on-disk schema we
/// need to consume here. Mirrors the `EventRecord<'a>` type — but
/// stays decoupled so a future event-log evolution (e.g. new event
/// kinds) can be added without forcing this crate to rebuild against
/// `tmg-core`.
#[derive(Debug, Deserialize)]
struct EventEnvelope {
    event: EventEnvelopeKind,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum EventEnvelopeKind {
    #[serde(rename = "thinking")]
    Thinking { token: String },
    #[serde(rename = "token")]
    Token { token: String },
    #[serde(rename = "tool_call")]
    ToolCall { name: String, arguments: String },
    #[serde(rename = "tool_result")]
    ToolResult {
        name: String,
        output: String,
        is_error: bool,
    },
    #[serde(rename = "done")]
    Done,
    /// Catch-all so unknown event kinds (warnings, memory ops, pool
    /// selection events) parse without erroring; we just drop them.
    #[serde(other)]
    Other,
}

/// Replay an `--event-log` JSONL file into trajectory records.
///
/// # Tool-call / tool-result pairing
///
/// The event log does not carry call ids, so this function synthesises
/// sequential `export_call_N` ids and matches `tool_result` events to
/// the oldest pending `tool_call` of the same name (FIFO by tool
/// name). This works well in practice but is **not** guaranteed to
/// match the live recorder's ids — concurrent calls of the same tool
/// can be paired in a different order from the live agent loop.
///
/// When the event log is truncated (e.g. the process was killed
/// mid-turn), some `tool_call` events will have no matching
/// `tool_result`. This function emits a `tracing::warn!` at the end
/// of replay listing how many calls remained unmatched so callers can
/// detect the truncation case in their logs.
fn replay_event_log(path: &Path, recorder: &Recorder) -> Result<(), TrajectoryError> {
    let raw = std::fs::read_to_string(path).map_err(|e| TrajectoryError::io(path, e))?;
    let mut current_thinking = String::new();
    let mut current_text = String::new();
    let mut current_calls: Vec<ToolCallRecord> = Vec::new();
    // Keep a stable lookup of the most recent `tool_call` so the
    // next `tool_result` can be matched up with the corresponding
    // call_id. The event log does not carry call ids, so we
    // synthesise sequential ones.
    let mut next_call_id: u32 = 0;
    let mut pending_calls: Vec<(String, String)> = Vec::new(); // (id, name)

    for (line_no, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let env: EventEnvelope = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(line_no, error = %e, "skipping unparsable event_log line");
                continue;
            }
        };
        match env.event {
            EventEnvelopeKind::Thinking { token } => current_thinking.push_str(&token),
            EventEnvelopeKind::Token { token } => current_text.push_str(&token),
            EventEnvelopeKind::ToolCall { name, arguments } => {
                next_call_id = next_call_id.saturating_add(1);
                let id = format!("export_call_{next_call_id}");
                let args_val = serde_json::from_str::<serde_json::Value>(&arguments)
                    .unwrap_or_else(|_| serde_json::Value::String(arguments.clone()));
                current_calls.push(ToolCallRecord {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: args_val,
                });
                pending_calls.push((id, name));
            }
            EventEnvelopeKind::ToolResult {
                name,
                output,
                is_error,
            } => {
                // Match against the oldest pending call by name. This
                // is a heuristic — concurrent calls of the same name
                // are still paired in FIFO order which is good enough
                // for an offline reconstruction.
                let id = pending_calls
                    .iter()
                    .position(|(_, n)| n == &name)
                    .map_or_else(
                        || {
                            next_call_id = next_call_id.saturating_add(1);
                            format!("export_call_{next_call_id}")
                        },
                        |i| pending_calls.remove(i).0,
                    );
                recorder.record_tool_result(&id, &name, &output, is_error)?;
            }
            EventEnvelopeKind::Done => {
                // Flush the round.
                let thinking = if current_thinking.is_empty() {
                    None
                } else {
                    Some(current_thinking.as_str())
                };
                if !current_text.is_empty() || thinking.is_some() || !current_calls.is_empty() {
                    recorder.record_assistant(&current_text, thinking, &current_calls)?;
                }
                current_text.clear();
                current_thinking.clear();
                current_calls.clear();
            }
            EventEnvelopeKind::Other => {}
        }
    }
    // Flush any tail-end content that did not see a `done` event
    // (e.g. truncated event log).
    if !current_text.is_empty() || !current_thinking.is_empty() || !current_calls.is_empty() {
        let thinking = if current_thinking.is_empty() {
            None
        } else {
            Some(current_thinking.as_str())
        };
        recorder.record_assistant(&current_text, thinking, &current_calls)?;
    }
    // Surface truncation / orphaned-call situations so callers can
    // tell the difference between a clean replay and one where the
    // event log was cut short. The `export_call_N` ids in
    // `pending_calls` will never appear in any `tool_result` record
    // produced by this run.
    if !pending_calls.is_empty() {
        tracing::warn!(
            remaining = pending_calls.len(),
            "event log truncated; tool_calls without matching tool_results",
        );
    }
    Ok(())
}

/// Inventory entry returned by [`list_trajectories`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrajectoryEntry {
    /// Owning run id.
    pub run_id: String,
    /// 1-indexed session number.
    pub session_num: u32,
    /// Path to the JSONL file on disk.
    pub path: PathBuf,
    /// File size in bytes.
    pub size_bytes: u64,
}

/// Walk every run dir under `runs_dir`, listing trajectory files
/// already on disk (under each run's `<output_dir>`). Used by
/// `tmg trajectory list`.
///
/// # Errors
/// Returns [`TrajectoryError::Io`] when a directory cannot be read.
pub fn list_trajectories(
    runs_dir: &Path,
    config: &TrajectoryConfig,
) -> Result<Vec<TrajectoryEntry>, TrajectoryError> {
    let mut out = Vec::new();
    if !runs_dir.exists() {
        return Ok(out);
    }
    let read_dir = std::fs::read_dir(runs_dir).map_err(|e| TrajectoryError::io(runs_dir, e))?;
    for run_ent in read_dir {
        let run_ent = run_ent.map_err(|e| TrajectoryError::io(runs_dir, e))?;
        let run_path = run_ent.path();
        if !run_path.is_dir() {
            continue;
        }
        let run_id = match run_path.file_name().and_then(|n| n.to_str()) {
            Some(s) => s.to_owned(),
            None => continue,
        };
        let traj_dir = config.resolve_output_dir(&run_path);
        if !traj_dir.exists() {
            continue;
        }
        let session_files =
            std::fs::read_dir(&traj_dir).map_err(|e| TrajectoryError::io(&traj_dir, e))?;
        for sess_ent in session_files {
            let sess_ent = sess_ent.map_err(|e| TrajectoryError::io(&traj_dir, e))?;
            let path = sess_ent.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(idx) = parse_session_jsonl(name) else {
                continue;
            };
            let metadata = std::fs::metadata(&path).map_err(|e| TrajectoryError::io(&path, e))?;
            out.push(TrajectoryEntry {
                run_id: run_id.clone(),
                session_num: idx,
                path,
                size_bytes: metadata.len(),
            });
        }
    }
    out.sort_by(|a, b| {
        a.run_id
            .cmp(&b.run_id)
            .then_with(|| a.session_num.cmp(&b.session_num))
    });
    Ok(out)
}

fn parse_session_jsonl(name: &str) -> Option<u32> {
    let stripped = name.strip_prefix("session_")?.strip_suffix(".jsonl")?;
    stripped.parse::<u32>().ok()
}

/// Convenience constructor used by the CLI to write a freshly-minted
/// trajectory record without going through the streaming sink path.
/// Callers that already hold a [`Recorder`] can skip this helper.
///
/// # Errors
/// Returns [`TrajectoryError`] when the recorder cannot be constructed.
pub fn open_recorder_for_session(
    run_dir: &Path,
    session_index: u32,
    config: TrajectoryConfig,
) -> Result<Recorder, TrajectoryError> {
    let path = trajectory_path(run_dir, &config.output_dir, session_index);
    Recorder::create(path, config)
}

/// Push a [`TrajectoryRecord`] into a recorder by variant. Lets
/// callers write code in terms of the `TrajectoryRecord` enum without
/// caring which `record_*` helper to call.
///
/// # Errors
/// Returns [`TrajectoryError`] on serialisation or I/O failure.
pub fn record(recorder: &Recorder, record: TrajectoryRecord) -> Result<(), TrajectoryError> {
    match record {
        TrajectoryRecord::Meta(m) => recorder.record_meta(m),
        TrajectoryRecord::System(SystemRecord { content }) => recorder.record_system(&content),
        TrajectoryRecord::User(UserRecord { content }) => recorder.record_user(&content),
        TrajectoryRecord::Assistant(a) => {
            recorder.record_assistant(&a.content, a.thinking.as_deref(), &a.tool_calls)
        }
        TrajectoryRecord::ToolResult(ToolResultRecord {
            tool_call_id,
            tool_name,
            output,
            is_error,
        }) => recorder.record_tool_result(&tool_call_id, &tool_name, &output, is_error),
        TrajectoryRecord::Feedback(f) => recorder.record_feedback(&f.source, &f.content),
        TrajectoryRecord::Verdict(VerdictRecord {
            outcome,
            feature_marked_passing,
        }) => recorder.record_verdict(&outcome, feature_marked_passing),
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use tmg_harness::{Session, SessionEndTrigger};

    /// Round-trip a minimal session through the export path: write a
    /// `session_NNN.json`, run `export`, verify the produced JSONL
    /// has meta + system + user + verdict.
    #[test]
    fn export_summary_only_fallback() {
        let tmp = tempfile::TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let runs_dir = tmp.path().join("runs");
        std::fs::create_dir_all(&runs_dir).unwrap_or_else(|e| panic!("{e}"));
        let store = RunStore::new(&runs_dir);
        let run = store
            .create_ad_hoc(tmp.path().to_path_buf(), None)
            .unwrap_or_else(|e| panic!("{e}"));

        let log = SessionLog::new(store.session_log_dir(&run.id));
        let mut session = Session::begin(1);
        session.summary =
            "implemented foo and ran the API key sk-abcdefghijklmnopqrstuvwxyz1234567890ABCD"
                .into();
        session.end(SessionEndTrigger::Completed);
        log.save(&session).unwrap_or_else(|e| panic!("{e}"));

        let out_dir = tmp.path().join("out");
        let summaries = export(
            &store,
            &ExportFilter {
                run_id: Some(run.id.as_str().to_owned()),
                ..ExportFilter::default()
            },
            Some(&out_dir),
            &TrajectoryConfig::default(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].sessions_exported, 1);
        assert!(summaries[0].had_summary_only_fallback);

        let traj = std::fs::read_to_string(out_dir.join(run.id.as_str()).join("session_001.jsonl"))
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(traj.contains(r#""type":"meta""#), "{traj}");
        assert!(traj.contains(r#""type":"system""#), "{traj}");
        assert!(traj.contains(r#""type":"user""#), "{traj}");
        assert!(traj.contains(r#""type":"verdict""#), "{traj}");
        // Redaction must apply on the `user` record minted from the summary.
        assert!(!traj.contains("sk-abcdefghijkl"), "{traj}");
    }

    /// When an `event_log/session_NNN.jsonl` neighbour is present, the
    /// export replays it into per-round assistant records.
    #[test]
    fn export_replays_event_log() {
        let tmp = tempfile::TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let runs_dir = tmp.path().join("runs");
        std::fs::create_dir_all(&runs_dir).unwrap_or_else(|e| panic!("{e}"));
        let store = RunStore::new(&runs_dir);
        let run = store
            .create_ad_hoc(tmp.path().to_path_buf(), None)
            .unwrap_or_else(|e| panic!("{e}"));

        let log = SessionLog::new(store.session_log_dir(&run.id));
        let mut session = Session::begin(1);
        session.summary = "Read Cargo.toml".into();
        session.end(SessionEndTrigger::Completed);
        log.save(&session).unwrap_or_else(|e| panic!("{e}"));

        let event_log_dir = store.run_dir(&run.id).join("event_log");
        std::fs::create_dir_all(&event_log_dir).unwrap_or_else(|e| panic!("{e}"));
        let event_log_path = event_log_dir.join("session_001.jsonl");
        // Synthesised event log mimicking `EventLogWriter`'s schema.
        let event_log_body = r#"{"elapsed_ms":0,"event":{"type":"token","token":"Reading "}}
{"elapsed_ms":1,"event":{"type":"token","token":"Cargo.toml"}}
{"elapsed_ms":2,"event":{"type":"tool_call","name":"file_read","arguments":"{\"path\":\"Cargo.toml\"}"}}
{"elapsed_ms":3,"event":{"type":"done"}}
{"elapsed_ms":4,"event":{"type":"tool_result","name":"file_read","output":"[workspace]","is_error":false}}
{"elapsed_ms":5,"event":{"type":"token","token":"done"}}
{"elapsed_ms":6,"event":{"type":"done"}}
"#;
        std::fs::write(&event_log_path, event_log_body).unwrap_or_else(|e| panic!("{e}"));

        let out_dir = tmp.path().join("out");
        let _ = export(
            &store,
            &ExportFilter {
                run_id: Some(run.id.as_str().to_owned()),
                ..ExportFilter::default()
            },
            Some(&out_dir),
            &TrajectoryConfig::default(),
        )
        .unwrap_or_else(|e| panic!("{e}"));

        let traj = std::fs::read_to_string(out_dir.join(run.id.as_str()).join("session_001.jsonl"))
            .unwrap_or_else(|e| panic!("{e}"));
        // Two assistant rounds, one tool_result.
        let assistant_count = traj.matches(r#""type":"assistant""#).count();
        let tool_result_count = traj.matches(r#""type":"tool_result""#).count();
        assert_eq!(assistant_count, 2, "{traj}");
        assert_eq!(tool_result_count, 1, "{traj}");
    }
}
