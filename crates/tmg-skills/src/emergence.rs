//! Emergence triggers for autonomous skill creation.
//!
//! After every `AgentLoop::turn` the harness calls
//! [`SignalCollector::record_turn`] with the just-completed turn's
//! summary. The collector evaluates four trigger families (see
//! [`TriggerKind`]) and exposes a [`SkillCandidacySignal`] when any
//! fires. The signal is a *candidate*, not a decision: the runtime
//! still has to spawn `skill_critic` to decide whether the procedure
//! is worth bottling up as a Skill.
//!
//! ## Hard limits
//!
//! - At most **one** auto-generated skill per session
//!   ([`SignalCollector::record_auto_generated`] flips the gate).
//! - The session-wide auto-creation gate can be flipped off entirely
//!   via [`SignalCollector::disable_auto`] (driven by
//!   `/skills disable-auto`).
//!
//! Trigger detection is local to this struct so the harness side
//! collapses to "feed each turn in, react when a signal pops out".

use std::ops::Range;

use serde::{Deserialize, Serialize};

/// A summary of one completed agent turn — the input the collector
/// needs to decide whether any trigger should fire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnSummary {
    /// Sequential turn index in the current session (0-based).
    pub turn_index: usize,

    /// Names of every tool call attempted on this turn, paired with
    /// whether each call succeeded.
    pub tool_calls: Vec<ToolCallOutcome>,

    /// Whether the user wrote a new feedback memory entry on this turn
    /// (issue #52). True is the cue for the "user correction" trigger.
    pub feedback_memory_written: bool,

    /// Whether the turn ended with an error (any `is_error` tool result
    /// or a top-level loop error). Used by the "successful complex
    /// task" trigger to gate on a clean ending.
    pub turn_errored: bool,
}

/// Per-tool-call outcome inside a turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallOutcome {
    /// The tool name invoked (`file_read`, `shell_exec`, ...).
    pub tool: String,
    /// Whether the call returned a successful result.
    pub success: bool,
}

/// The trigger family that produced a [`SkillCandidacySignal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriggerKind {
    /// Five or more consecutive successful tool calls and the closing
    /// turn produced no error. The classic "this looks like a recipe"
    /// signal.
    SuccessfulComplexTask,

    /// The same tool failed twice in a row, then a different tool (or
    /// the same tool with different parameters — we use the rougher
    /// "different tool name" proxy here) succeeded. Captures recovery
    /// patterns worth memorialising.
    ErrorRecovery,

    /// Triggered the turn the user wrote a new feedback memory entry.
    UserCorrection,

    /// Triggered explicitly by `/skill capture`. Bypasses the
    /// per-session limit because the user is the one asking for it.
    Manual,
}

/// A skill-candidacy signal handed off to `skill_critic` for a
/// final create/skip verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillCandidacySignal {
    /// Why this signal fired.
    pub kind: TriggerKind,

    /// Range of turn indices the candidate procedure spans (inclusive
    /// start, exclusive end).
    pub turn_range: Range<usize>,

    /// Number of tool calls observed across `turn_range`.
    pub tool_call_count: u32,
}

/// Minimum number of consecutive successful tool calls required to fire
/// [`TriggerKind::SuccessfulComplexTask`]. Per the issue: "5+
/// consecutive `tool_calls` succeed".
const SUCCESSFUL_COMPLEX_THRESHOLD: u32 = 5;

/// Stateful collector that observes turns and emits signals.
///
/// The collector is `Send + Sync` so the harness can park it inside an
/// `Arc<Mutex<...>>` shared across the agent loop and the slash-command
/// dispatch path (the `/skill capture` route).
#[derive(Debug)]
pub struct SignalCollector {
    /// Remaining auto-generation budget (decremented on
    /// `record_auto_generated`).
    auto_remaining: u32,

    /// Whether `/skills disable-auto` was issued for the session.
    auto_disabled: bool,

    /// Running count of consecutive successful tool calls (used by
    /// [`TriggerKind::SuccessfulComplexTask`]).
    consecutive_successes: u32,

    /// Turn index where the current success streak began.
    success_streak_start: Option<usize>,

    /// Cumulative tool-call count tracked for the current streak.
    success_streak_tool_calls: u32,

    /// State machine bookkeeping for [`TriggerKind::ErrorRecovery`]:
    /// the last failing tool name and how many times in a row it failed
    /// on prior turns.
    last_failing_tool: Option<String>,
    consecutive_same_failures: u32,
    error_recovery_anchor_turn: Option<usize>,
}

impl SignalCollector {
    /// Create a new collector with the given per-session auto-creation
    /// budget. The issue prescribes 1 (one).
    #[must_use]
    pub fn new(max_auto_per_session: u32) -> Self {
        Self {
            auto_remaining: max_auto_per_session,
            auto_disabled: false,
            consecutive_successes: 0,
            success_streak_start: None,
            success_streak_tool_calls: 0,
            last_failing_tool: None,
            consecutive_same_failures: 0,
            error_recovery_anchor_turn: None,
        }
    }

    /// Whether autonomous skill creation is still permitted on this
    /// session (budget remaining and not user-disabled).
    #[must_use]
    pub fn auto_creation_allowed(&self) -> bool {
        !self.auto_disabled && self.auto_remaining > 0
    }

    /// Mark that auto-creation was just used by the harness.
    pub fn record_auto_generated(&mut self) {
        self.auto_remaining = self.auto_remaining.saturating_sub(1);
    }

    /// Disable auto-creation for the rest of the session.
    pub fn disable_auto(&mut self) {
        self.auto_disabled = true;
    }

    /// Whether auto-creation has been disabled by the user.
    #[must_use]
    pub fn auto_disabled(&self) -> bool {
        self.auto_disabled
    }

    /// Remaining auto-creation budget (for diagnostics / TUI display).
    #[must_use]
    pub fn auto_remaining(&self) -> u32 {
        self.auto_remaining
    }

    /// Manually trigger a candidacy signal (driven by `/skill
    /// capture`). The manual signal bypasses the per-session limit but
    /// still respects the [`Self::auto_disabled`] gate (a user who
    /// asked to halt auto-creation almost certainly didn't intend the
    /// next `/skill capture` to override that decision; we treat them
    /// as a single switch).
    pub fn manual_capture(&self, turn_index: usize, tool_call_count: u32) -> SkillCandidacySignal {
        SkillCandidacySignal {
            kind: TriggerKind::Manual,
            turn_range: turn_index..(turn_index + 1),
            tool_call_count,
        }
    }

    /// Record a turn and return any auto-fire signal it produced.
    ///
    /// The collector mutates internal streak state regardless of
    /// whether the auto-gate is open; we still want to *observe*
    /// patterns even when the user has disabled auto-creation, because
    /// that data may feed future telemetry. The gate only governs
    /// *firing* (`Some(_)` return).
    pub fn record_turn(&mut self, turn: &TurnSummary) -> Option<SkillCandidacySignal> {
        let mut signal: Option<SkillCandidacySignal> = None;

        // -- ErrorRecovery: detect a same-tool double failure followed
        //    by a successful different-tool call within this turn or
        //    on the next turn after the second failure.
        if let Some(s) = self.try_error_recovery(turn) {
            signal = Some(s);
        }

        // -- Streak update for SuccessfulComplexTask.
        for call in &turn.tool_calls {
            if call.success {
                if self.consecutive_successes == 0 {
                    self.success_streak_start = Some(turn.turn_index);
                }
                self.consecutive_successes = self.consecutive_successes.saturating_add(1);
                self.success_streak_tool_calls = self.success_streak_tool_calls.saturating_add(1);
            } else {
                self.consecutive_successes = 0;
                self.success_streak_start = None;
                self.success_streak_tool_calls = 0;
            }
        }

        if !turn.turn_errored
            && self.consecutive_successes >= SUCCESSFUL_COMPLEX_THRESHOLD
            && signal.is_none()
        {
            let start = self.success_streak_start.unwrap_or(turn.turn_index);
            signal = Some(SkillCandidacySignal {
                kind: TriggerKind::SuccessfulComplexTask,
                turn_range: start..(turn.turn_index + 1),
                tool_call_count: self.success_streak_tool_calls,
            });
            // Reset so the same streak doesn't fire again on the next
            // turn.
            self.consecutive_successes = 0;
            self.success_streak_start = None;
            self.success_streak_tool_calls = 0;
        }

        // -- UserCorrection: dominate other signals when present.
        if turn.feedback_memory_written {
            signal = Some(SkillCandidacySignal {
                kind: TriggerKind::UserCorrection,
                turn_range: turn.turn_index..(turn.turn_index + 1),
                tool_call_count: u32::try_from(turn.tool_calls.len()).unwrap_or(u32::MAX),
            });
        }

        // Gate the actual fire on the auto-creation budget. We still
        // return the signal shape so callers can introspect the
        // collector's view (e.g. the TUI surfacing "we *would* have
        // generated a skill but auto is disabled").
        if let Some(sig) = &signal {
            if !self.auto_creation_allowed() {
                // Suppress: leave the streak counters reset; do not
                // emit.
                return None;
            }
            return Some(sig.clone());
        }
        None
    }

    /// [`TriggerKind::ErrorRecovery`] state machine: look for two
    /// consecutive failures of the same tool followed by a different
    /// tool succeeding.
    fn try_error_recovery(&mut self, turn: &TurnSummary) -> Option<SkillCandidacySignal> {
        let mut recovery: Option<SkillCandidacySignal> = None;

        for call in &turn.tool_calls {
            if call.success {
                // Recovery only counts if a different tool succeeded
                // *after* a same-tool failure pair.
                if let (Some(failing), Some(anchor)) = (
                    self.last_failing_tool.as_deref(),
                    self.error_recovery_anchor_turn,
                ) && self.consecutive_same_failures >= 2
                    && failing != call.tool
                {
                    recovery = Some(SkillCandidacySignal {
                        kind: TriggerKind::ErrorRecovery,
                        turn_range: anchor..(turn.turn_index + 1),
                        tool_call_count: self.consecutive_same_failures + 1,
                    });
                }
                self.last_failing_tool = None;
                self.consecutive_same_failures = 0;
                self.error_recovery_anchor_turn = None;
            } else {
                let same_as_last = self
                    .last_failing_tool
                    .as_deref()
                    .is_some_and(|t| t == call.tool);
                if same_as_last {
                    self.consecutive_same_failures =
                        self.consecutive_same_failures.saturating_add(1);
                } else {
                    self.last_failing_tool = Some(call.tool.clone());
                    self.consecutive_same_failures = 1;
                    self.error_recovery_anchor_turn = Some(turn.turn_index);
                }
            }
        }

        recovery
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "tests assert with expect for clarity; the workspace policy denies it in production code"
)]
mod tests {
    use super::*;

    fn turn(idx: usize, calls: &[(&str, bool)], errored: bool) -> TurnSummary {
        TurnSummary {
            turn_index: idx,
            tool_calls: calls
                .iter()
                .map(|(tool, ok)| ToolCallOutcome {
                    tool: (*tool).to_owned(),
                    success: *ok,
                })
                .collect(),
            feedback_memory_written: false,
            turn_errored: errored,
        }
    }

    #[test]
    fn successful_complex_task_after_five_calls() {
        let mut c = SignalCollector::new(1);
        // Five successful calls in one turn ending cleanly.
        let t = turn(
            0,
            &[
                ("file_read", true),
                ("file_read", true),
                ("file_read", true),
                ("file_read", true),
                ("shell_exec", true),
            ],
            false,
        );
        let signal = c.record_turn(&t).expect("signal expected");
        assert_eq!(signal.kind, TriggerKind::SuccessfulComplexTask);
        assert_eq!(signal.tool_call_count, 5);
    }

    #[test]
    fn error_recovery_fires_after_double_failure_then_success() {
        let mut c = SignalCollector::new(1);
        // Turn 0: the same tool fails twice.
        let signal = c.record_turn(&turn(
            0,
            &[("shell_exec", false), ("shell_exec", false)],
            true,
        ));
        assert!(signal.is_none());
        // Turn 1: a different tool succeeds. Recovery fires.
        let signal = c.record_turn(&turn(1, &[("file_read", true)], false));
        let s = signal.expect("recovery signal");
        assert_eq!(s.kind, TriggerKind::ErrorRecovery);
        assert!(s.turn_range.start == 0 && s.turn_range.end == 2);
    }

    #[test]
    fn user_correction_takes_precedence() {
        let mut c = SignalCollector::new(1);
        let t = TurnSummary {
            turn_index: 3,
            tool_calls: vec![ToolCallOutcome {
                tool: "memory_write".to_owned(),
                success: true,
            }],
            feedback_memory_written: true,
            turn_errored: false,
        };
        let signal = c.record_turn(&t).expect("signal");
        assert_eq!(signal.kind, TriggerKind::UserCorrection);
    }

    #[test]
    fn manual_capture_emits_signal() {
        let c = SignalCollector::new(1);
        let signal = c.manual_capture(7, 3);
        assert_eq!(signal.kind, TriggerKind::Manual);
        assert_eq!(signal.turn_range, 7..8);
        assert_eq!(signal.tool_call_count, 3);
    }

    #[test]
    fn auto_disabled_suppresses_signals() {
        let mut c = SignalCollector::new(1);
        c.disable_auto();
        let t = TurnSummary {
            turn_index: 0,
            tool_calls: vec![ToolCallOutcome {
                tool: "x".to_owned(),
                success: true,
            }],
            feedback_memory_written: true,
            turn_errored: false,
        };
        assert!(c.record_turn(&t).is_none());
    }

    #[test]
    fn per_session_budget_caps_auto_creation() {
        let mut c = SignalCollector::new(1);
        assert!(c.auto_creation_allowed());
        c.record_auto_generated();
        assert!(!c.auto_creation_allowed());

        // Even a clear UserCorrection signal is suppressed once the
        // budget is exhausted.
        let t = TurnSummary {
            turn_index: 0,
            tool_calls: vec![],
            feedback_memory_written: true,
            turn_errored: false,
        };
        assert!(c.record_turn(&t).is_none());
    }

    #[test]
    fn streak_resets_on_failure() {
        let mut c = SignalCollector::new(1);
        // Four successes, then a failure mid-turn -- should NOT fire
        // because the streak is reset.
        let t = turn(
            0,
            &[
                ("a", true),
                ("a", true),
                ("a", true),
                ("a", true),
                ("a", false),
                ("a", true),
            ],
            true,
        );
        assert!(c.record_turn(&t).is_none());
    }
}
