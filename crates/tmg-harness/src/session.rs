//! Session data model.
//!
//! A [`Session`] is one continuous interaction between the user (or a
//! workflow) and the agent. A [`Run`](crate::run::Run) accumulates one
//! or more sessions over its lifetime; the run's `session_count` is
//! incremented every time a new session begins.
//!
//! This minimal version of the type does not yet persist session logs
//! (the file `session_log.jsonl` is introduced in a follow-up issue).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// One session within a [`Run`](crate::run::Run).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Session {
    /// Sequence number within the parent run (1-indexed).
    pub index: u32,
    /// When the session started.
    pub started_at: DateTime<Utc>,
    /// When the session ended, if it has.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<DateTime<Utc>>,
    /// Reason the session ended, when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_trigger: Option<SessionEndTrigger>,
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
        }
    }

    /// Mark this session as ended with the given trigger.
    pub fn end(&mut self, trigger: SessionEndTrigger) {
        self.ended_at = Some(Utc::now());
        self.end_trigger = Some(trigger);
    }
}

/// Why a session ended. SPEC §9.5.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionEndTrigger {
    /// Normal completion (user closed the TUI / workflow finished).
    Completed,
    /// Cancelled by the user (e.g. Ctrl-C).
    UserCancelled,
    /// The harness rotated to a new session (e.g. context exhaustion).
    Rotated {
        /// Reason for rotation, surfaced to the operator.
        reason: String,
    },
    /// An error terminated the session.
    Errored {
        /// Error message.
        message: String,
    },
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
    }

    #[test]
    fn end_session_records_trigger() {
        let mut session = Session::begin(1);
        session.end(SessionEndTrigger::Completed);
        assert!(session.ended_at.is_some());
        assert_eq!(session.end_trigger, Some(SessionEndTrigger::Completed));
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
