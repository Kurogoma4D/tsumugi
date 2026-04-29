//! tmg-trajectory: structured RL/SFT-grade conversation logs.
//!
//! Records every turn of an agent session (system prompt, user input,
//! assistant response with optional reasoning + tool calls, tool
//! results, user-intervention feedback, and an end-of-session
//! verdict) as JSON Lines under
//! `.tsumugi/runs/<run-id>/trajectories/session_NNN.jsonl`.
//!
//! # `OpenAI` compatibility
//!
//! The on-disk schema is a *superset* of the `OpenAI` chat-completion
//! conversation format: each record's `type` field corresponds to a
//! standard `role` (or to the auxiliary `tool_result` role) so a
//! thin transform produces an `OpenAI`-compatible dataset without
//! re-reading the source-of-truth log. The exact mapping is
//! documented on [`record::TrajectoryRecord`].
//!
//! # Wiring into the agent loop
//!
//! The integrator (the CLI in this workspace) opens a [`Recorder`] at
//! session start, wraps the existing [`tmg_core::StreamSink`] in a
//! [`TrajectoryStreamSink`], and lets the tee chain do the rest. See
//! [`sink`] for the wire-up details.
//!
//! # Default OFF
//!
//! Issue #55 mandates trajectory recording is **opt-in**. The CLI is
//! responsible for honouring [`TrajectoryConfig::enabled`]; this
//! crate refuses nothing, but the integrator must skip recorder
//! construction when the config flag is `false`.
//!
//! # `.gitignore` recommendation
//!
//! Trajectory files are **not** intended for version control: they
//! contain raw conversation text and code diffs. Add the following
//! patterns to your project's `.gitignore`:
//!
//! ```text
//! # tsumugi trajectories (issue #55) — full conversation logs
//! .tsumugi/runs/*/trajectories/
//! .tsumugi/trajectories/
//! ```
//!
//! When `tmg init` adds a `.gitignore` template in a future revision
//! it should include these entries.

pub mod bundle;
pub mod config;
pub mod error;
pub mod export;
pub mod record;
pub mod recorder;
pub mod redact;
pub mod sink;

pub use config::{ToolResultMode, TrajectoryConfig};
pub use error::TrajectoryError;
pub use export::{
    ExportFilter, ExportSummary, TrajectoryEntry, export, list_trajectories,
    open_recorder_for_session,
};
pub use record::{
    AssistantRecord, FeedbackRecord, MetaRecord, SystemRecord, ToolCallRecord, ToolResultRecord,
    TrajectoryRecord, UserRecord, VerdictRecord,
};
pub use recorder::{Recorder, TRAJECTORIES_DIRNAME, now_utc, trajectory_path};
pub use redact::{REDACTION_TOKEN, Redactor, redact_secrets};
pub use sink::{NoOpSink, RecorderSink, TrajectorySink, TrajectoryStreamSink};

/// Convert a [`tmg_harness::SessionEndTrigger`] into the short label
/// used by the [`record::VerdictRecord::outcome`] / [`record::MetaRecord::outcome`]
/// fields.
///
/// Centralised here so the live-recording path (CLI) and the export
/// path (which has a `&Session` only) emit identical values.
#[must_use]
pub fn record_outcome_label(trigger: &tmg_harness::SessionEndTrigger) -> String {
    match trigger {
        tmg_harness::SessionEndTrigger::Completed => "completed".into(),
        tmg_harness::SessionEndTrigger::UserCancelled => "user_cancelled".into(),
        tmg_harness::SessionEndTrigger::Rotated { reason } => {
            format!("rotated: {reason}")
        }
        tmg_harness::SessionEndTrigger::Errored { message } => {
            format!("errored: {message}")
        }
        tmg_harness::SessionEndTrigger::UserExit => "user_exit".into(),
        tmg_harness::SessionEndTrigger::ContextRotation => "context_rotation".into(),
        tmg_harness::SessionEndTrigger::Timeout => "timeout".into(),
        tmg_harness::SessionEndTrigger::UserNewSession => "user_new_session".into(),
    }
}
