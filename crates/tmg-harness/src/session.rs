//! Session data model.
//!
//! A [`Session`] is one continuous interaction between the user (or a
//! workflow) and the agent. A [`Run`](crate::run::Run) accumulates one
//! or more sessions over its lifetime; the run's `session_count` is
//! incremented every time a new session begins.
//!
//! The on-disk JSON layout for a finished session is described in
//! SPEC §9.4 and persisted via
//! [`SessionLog::save`](crate::artifacts::SessionLog::save).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// One session within a [`Run`](crate::run::Run).
///
/// The serialized form follows the schema in SPEC §9.4 (`session_NNN.json`):
///
/// ```json
/// {
///   "session_number": 3,
///   "started_at": "2025-04-25T07:00:00Z",
///   "ended_at": "2025-04-25T07:42:13Z",
///   "trigger": { "kind": "completed" },
///   "summary": "Implemented foo and bar",
///   "tool_calls_count": 12,
///   "files_modified": ["src/foo.rs", "src/bar.rs"],
///   "features_marked_passing": [],
///   "context_usage_peak": 0.62,
///   "next_session_hint": "Look at integration tests"
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Session {
    /// Sequence number within the parent run (1-indexed).
    ///
    /// Serialized as `session_number` per SPEC §9.4.
    #[serde(rename = "session_number")]
    pub index: u32,

    /// When the session started.
    pub started_at: DateTime<Utc>,

    /// When the session ended, if it has.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<DateTime<Utc>>,

    /// Reason the session ended, when applicable.
    ///
    /// Serialized as `trigger` per SPEC §9.4.
    #[serde(default, rename = "trigger", skip_serializing_if = "Option::is_none")]
    pub end_trigger: Option<SessionEndTrigger>,

    /// Human-readable summary of what happened in this session.
    ///
    /// Updated by the `session_summary_save` tool.
    #[serde(default)]
    pub summary: String,

    /// Number of tool calls executed in this session.
    #[serde(default)]
    pub tool_calls_count: u32,

    /// Files modified during this session (paths relative to workspace,
    /// or absolute when outside the workspace).
    #[serde(default)]
    pub files_modified: Vec<String>,

    /// Feature ids that the agent flagged as newly-passing in this session.
    ///
    /// Empty for ad-hoc runs; populated by the harnessed scope (#34+).
    #[serde(default)]
    pub features_marked_passing: Vec<String>,

    /// Peak context usage observed during the session, as a fraction in
    /// `0.0..=1.0`. Defaults to `0.0` until plumbed through.
    #[serde(default)]
    pub context_usage_peak: f64,

    /// Optional hint for the next session, surfaced via
    /// [`SessionLog::last_hint`](crate::artifacts::SessionLog::last_hint).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_session_hint: Option<String>,
}

impl Session {
    /// Begin a new session at the given sequence number.
    #[must_use]
    pub fn begin(index: u32) -> Self {
        Self {
            index,
            started_at: Utc::now(),
            ended_at: None,
            end_trigger: None,
            summary: String::new(),
            tool_calls_count: 0,
            files_modified: Vec::new(),
            features_marked_passing: Vec::new(),
            context_usage_peak: 0.0,
            next_session_hint: None,
        }
    }

    /// Mark this session as ended with the given trigger.
    pub fn end(&mut self, trigger: SessionEndTrigger) {
        self.ended_at = Some(Utc::now());
        self.end_trigger = Some(trigger);
    }
}

/// Why a session ended. SPEC §9.4 / §9.5.
///
/// Issue #38 grew the enum with the four "session-boundary"
/// causes prescribed by SPEC §9.4 ([`UserExit`](Self::UserExit),
/// [`ContextRotation`](Self::ContextRotation),
/// [`Timeout`](Self::Timeout),
/// [`UserNewSession`](Self::UserNewSession)). The original
/// [`Completed`](Self::Completed) / [`UserCancelled`](Self::UserCancelled)
/// / [`Rotated`](Self::Rotated) / [`Errored`](Self::Errored) variants
/// are preserved for backwards-compatibility with on-disk
/// `session_NNN.json` files written by earlier versions and for
/// callers that still want a richer trigger payload (e.g.
/// [`Errored`](Self::Errored) carries a message).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionEndTrigger {
    /// Normal completion (user closed the TUI / workflow finished).
    Completed,
    /// Cancelled by the user (e.g. Ctrl-C).
    UserCancelled,
    /// The harness rotated to a new session (e.g. context exhaustion).
    ///
    /// Carries a free-form reason. Prefer the
    /// [`ContextRotation`](Self::ContextRotation) variant for the
    /// canonical SPEC §9.4 force-rotate path; this richer variant is
    /// kept for cases where the caller wants to attribute the rotation
    /// to a specific subsystem (e.g. `"manual /compact"`).
    Rotated {
        /// Reason for rotation, surfaced to the operator.
        reason: String,
    },
    /// An error terminated the session.
    Errored {
        /// Error message.
        message: String,
    },
    /// SPEC §9.4 case 1: the user ended the run (closed the CLI / TUI
    /// without resuming). Distinct from
    /// [`UserCancelled`](Self::UserCancelled) in that the run is not
    /// expected to spawn a successor session.
    UserExit,
    /// SPEC §9.4 case 2: the harness's force-rotate fired because
    /// `context_usage` exceeded
    /// [`HarnessConfig::context_force_rotate_threshold`](crate::runner::DEFAULT_CONTEXT_FORCE_ROTATE_THRESHOLD).
    /// The runner immediately begins a new session after persisting
    /// the old one.
    ContextRotation,
    /// SPEC §9.4 case 3: the per-session wall-clock budget configured
    /// via [`HarnessConfig::default_session_timeout`] elapsed without
    /// the session completing. Fired by the watchdog spawned in
    /// [`RunRunner::begin_session`](crate::runner::RunRunner::begin_session).
    Timeout,
    /// SPEC §9.4 case 4: the user explicitly rolled to a new session
    /// (e.g. via the `/run new-session` command).
    UserNewSession,
}

/// Opaque handle returned by `RunRunner::begin_session` and consumed by
/// `RunRunner::end_session`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionHandle {
    /// Session sequence number within the run.
    pub index: u32,
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;

    #[test]
    fn begin_session_sets_started() {
        let session = Session::begin(1);
        assert_eq!(session.index, 1);
        assert!(session.ended_at.is_none());
        assert!(session.end_trigger.is_none());
        assert!(session.summary.is_empty());
        assert_eq!(session.tool_calls_count, 0);
        assert!(session.files_modified.is_empty());
        assert!(session.features_marked_passing.is_empty());
    }

    #[test]
    fn end_session_records_trigger() {
        let mut session = Session::begin(1);
        session.end(SessionEndTrigger::Completed);
        assert!(session.ended_at.is_some());
        assert_eq!(session.end_trigger, Some(SessionEndTrigger::Completed));
    }

    #[test]
    fn session_round_trip_json_matches_spec_schema() {
        let mut session = Session::begin(3);
        session.summary = "Implemented foo and bar".to_owned();
        session.tool_calls_count = 12;
        session.files_modified.push("src/foo.rs".to_owned());
        session.files_modified.push("src/bar.rs".to_owned());
        session.context_usage_peak = 0.62;
        session.next_session_hint = Some("Look at integration tests".to_owned());
        session.end(SessionEndTrigger::Completed);

        let json = serde_json::to_string(&session).unwrap_or_else(|e| panic!("{e}"));
        // SPEC §9.4 keys must be present.
        assert!(json.contains("\"session_number\":3"), "{json}");
        assert!(json.contains("\"trigger\":"), "{json}");
        assert!(json.contains("\"summary\":\"Implemented foo and bar\""));
        assert!(json.contains("\"tool_calls_count\":12"));
        assert!(json.contains("\"files_modified\":[\"src/foo.rs\",\"src/bar.rs\"]"));
        assert!(json.contains("\"features_marked_passing\":[]"));
        assert!(json.contains("\"context_usage_peak\":0.62"));
        assert!(json.contains("\"next_session_hint\":\"Look at integration tests\""));

        // Round-trip back through serde.
        let parsed: Session = serde_json::from_str(&json).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(parsed, session);
    }

    #[test]
    fn session_round_trip_toml() {
        let mut session = Session::begin(2);
        session.end(SessionEndTrigger::Rotated {
            reason: "context full".to_owned(),
        });
        let toml = toml::to_string(&session).unwrap_or_else(|e| panic!("{e}"));
        let parsed: Session = toml::from_str(&toml).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(session, parsed);
    }
}
