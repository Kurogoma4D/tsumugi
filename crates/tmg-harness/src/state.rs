//! Per-session aggregate state observed by the auto-promotion gate.
//!
//! [`SessionState`] is updated after every [`AgentLoop::turn`] via the
//! optional [`TurnObserver`] callback installed by the harness wire-up.
//! Its purpose is to feed
//! [`EscalationEvaluator::detect_signals`](crate::escalation::EscalationEvaluator::detect_signals)
//! the SPEC §9.10 trigger inputs without poking through the live agent
//! loop or the run's persisted `Session` record.
//!
//! Only the fields enumerated in SPEC §9.10 are tracked here; richer
//! aggregates (token-counter spike detection, file-modification
//! frequency curves) are out of scope and tracked separately.
//!
//! [`AgentLoop::turn`]: tmg_core::AgentLoop::turn

use std::collections::BTreeMap;

use crate::run::RunScope;

/// One turn's worth of metrics collected by the harness after every
/// [`AgentLoop::turn`].
///
/// The harness aggregates these into the long-lived [`SessionState`].
/// Producing a [`TurnSummary`] is intentionally cheap: the agent loop
/// already carries the underlying counters, and the harness simply
/// snapshots them.
///
/// [`AgentLoop::turn`]: tmg_core::AgentLoop::turn
#[derive(Debug, Clone, Default)]
pub struct TurnSummary {
    /// Total tokens estimated for the conversation history *after* this
    /// turn. Used to derive `context_usage` against the configured max
    /// context budget.
    pub tokens_used: usize,

    /// Number of tool calls dispatched during this turn (across all
    /// rounds within the turn).
    pub tool_calls: u32,

    /// Files the harness sink observed as modified during this turn
    /// (unique workspace-relative paths).
    pub files_modified: Vec<String>,

    /// Approximate diff line count produced during this turn. The CLI
    /// wires this from `git diff --shortstat` against the workspace
    /// HEAD before and after the turn; an undermined provider may
    /// leave this as `0`.
    pub diff_lines: u32,

    /// Most recent user-input message text. The keyword detector sees
    /// this verbatim; whitespace and casing handling is the
    /// detector's responsibility.
    pub user_message: String,
}

/// Snapshot of trigger inputs evaluated by
/// [`EscalationEvaluator::detect_signals`](crate::escalation::EscalationEvaluator::detect_signals).
///
/// All numeric fields default to `0`; `last_user_input_size_signal`
/// defaults to `false`. Updates flow through
/// [`SessionState::observe`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionState {
    /// Active run scope. Auto-promotion is only considered when this
    /// is [`RunScope::AdHoc`].
    pub scope: RunScope,

    /// Whether the most recent user input contained two or more
    /// scale-indicating phrases (Japanese / English dictionary lookup).
    pub last_user_input_size_signal: bool,

    /// Latest context window utilisation in `0.0..=1.0`. Computed by
    /// the harness wire-up from `tokens_used / max_context_tokens`.
    pub context_usage: f32,

    /// Number of tool calls observed in the most recent turn that have
    /// not (yet) been finalised as a written-back tool result. The
    /// CLI's signal collection treats this as the count of `tool_calls`
    /// in the most recent turn — i.e. concurrency depth at turn-end.
    pub pending_subtasks: u32,

    /// Cumulative diff lines accumulated across the session.
    pub session_diff_lines: u32,

    /// Number of times the same user message has repeated within the
    /// session. Reset every time the message changes.
    pub workflow_loop_count: u32,

    /// Maximum number of edits any single file has accumulated within
    /// the session.
    pub same_file_edit_count: u32,

    /// Most recent user message recorded; used to detect a workflow
    /// loop (same prompt repeated).
    last_user_message: String,

    /// Per-file edit tally used to derive
    /// [`Self::same_file_edit_count`].
    file_edits: BTreeMap<String, u32>,
}

impl SessionState {
    /// Construct a fresh state seeded with the run's scope.
    #[must_use]
    pub fn new(scope: RunScope) -> Self {
        Self {
            scope,
            ..Self::default()
        }
    }

    /// Replace the run scope (used after auto-promotion succeeds).
    pub fn set_scope(&mut self, scope: RunScope) {
        self.scope = scope;
    }

    /// Update the state with metrics from one [`TurnSummary`].
    ///
    /// `max_context_tokens` is the active LLM context budget; passing
    /// `0` clamps `context_usage` to `0.0` (i.e. the budget is
    /// effectively unbounded).
    pub fn observe(&mut self, summary: &TurnSummary, max_context_tokens: usize) {
        // Context usage is `tokens_used / max_context_tokens`, clamped to
        // [0, 1]. A zero budget short-circuits to 0 so the auto-promotion
        // gate cannot fire for misconfigured runs.
        if max_context_tokens == 0 {
            self.context_usage = 0.0;
        } else {
            // Casting through f64 keeps the division precise for usize
            // values that exceed f32's mantissa.
            #[expect(
                clippy::cast_precision_loss,
                reason = "intermediate f64 ratio is intentionally lossy when capped to [0,1]"
            )]
            let ratio = summary.tokens_used as f64 / max_context_tokens as f64;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "ratio is clamped to [0,1] which fits comfortably in f32"
            )]
            {
                self.context_usage = ratio.clamp(0.0, 1.0) as f32;
            }
        }

        self.pending_subtasks = summary.tool_calls;
        self.session_diff_lines = self.session_diff_lines.saturating_add(summary.diff_lines);

        // Workflow-loop detection: same prompt repeated bumps the
        // counter; otherwise reset to 1 for the new prompt. Empty
        // messages are ignored to avoid spurious increments from the
        // harness pushing a "tick" with an empty user_message.
        if !summary.user_message.is_empty() {
            if summary.user_message == self.last_user_message {
                self.workflow_loop_count = self.workflow_loop_count.saturating_add(1);
            } else {
                self.workflow_loop_count = 1;
                self.last_user_message.clone_from(&summary.user_message);
            }
            self.last_user_input_size_signal =
                crate::escalation::keywords::detect_size_signal(&summary.user_message);
        }

        // Per-file edit tally; surface the max as `same_file_edit_count`.
        for path in &summary.files_modified {
            let counter = self.file_edits.entry(path.clone()).or_insert(0);
            *counter = counter.saturating_add(1);
            if *counter > self.same_file_edit_count {
                self.same_file_edit_count = *counter;
            }
        }
    }

    /// Reset the per-turn signals after an auto-promotion succeeds so
    /// the same conditions do not immediately re-trigger.
    ///
    /// The [`Self::last_user_message`] tracker is also cleared so that
    /// the first post-promotion turn starts a fresh workflow-loop
    /// streak rather than inheriting the pre-promotion prompt's
    /// counter.
    pub fn reset_after_promotion(&mut self) {
        self.last_user_input_size_signal = false;
        self.workflow_loop_count = 0;
        self.same_file_edit_count = 0;
        self.session_diff_lines = 0;
        self.file_edits.clear();
        self.last_user_message.clear();
    }

    /// Reset the per-session signals across a session-rotation
    /// boundary (SPEC §9.4) so the successor session starts with
    /// clean state.
    ///
    /// Conceptually this clears every signal that was scoped to "the
    /// previous session's turn observations": the size-keyword latch,
    /// the workflow-loop counter, the per-file edit tally, and the
    /// session-cumulative diff line count. The active run scope is
    /// preserved (the rotation does not flip ad-hoc <-> harnessed),
    /// and [`context_usage`](Self::context_usage) is reset so the
    /// fresh session starts at `0.0` until the first new turn fires
    /// `observe`.
    pub fn reset_for_new_session(&mut self) {
        self.last_user_input_size_signal = false;
        self.context_usage = 0.0;
        self.pending_subtasks = 0;
        self.session_diff_lines = 0;
        self.workflow_loop_count = 0;
        self.same_file_edit_count = 0;
        self.last_user_message.clear();
        self.file_edits.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observe_updates_context_usage_clamped() {
        let mut state = SessionState::new(RunScope::AdHoc);
        let summary_a = TurnSummary {
            tokens_used: 4096,
            ..Default::default()
        };
        state.observe(&summary_a, 8192);
        assert!((state.context_usage - 0.5).abs() < 1e-6);

        let summary_b = TurnSummary {
            tokens_used: 16_000,
            ..Default::default()
        };
        state.observe(&summary_b, 8192);
        assert!((state.context_usage - 1.0).abs() < 1e-6);

        let summary_c = TurnSummary {
            tokens_used: 100,
            ..Default::default()
        };
        state.observe(&summary_c, 0);
        assert!(state.context_usage.abs() < 1e-6);
    }

    #[test]
    fn observe_tracks_pending_subtasks() {
        let mut state = SessionState::new(RunScope::AdHoc);
        let summary = TurnSummary {
            tool_calls: 4,
            ..Default::default()
        };
        state.observe(&summary, 1000);
        assert_eq!(state.pending_subtasks, 4);
    }

    #[test]
    fn observe_accumulates_diff_lines() {
        let mut state = SessionState::new(RunScope::AdHoc);
        let summary = TurnSummary {
            diff_lines: 100,
            ..Default::default()
        };
        state.observe(&summary, 1000);
        state.observe(&summary, 1000);
        assert_eq!(state.session_diff_lines, 200);
    }

    #[test]
    fn workflow_loop_detects_repeats() {
        let mut state = SessionState::new(RunScope::AdHoc);
        let summary = TurnSummary {
            user_message: "do the thing".to_owned(),
            ..Default::default()
        };
        state.observe(&summary, 1000);
        assert_eq!(state.workflow_loop_count, 1);
        state.observe(&summary, 1000);
        assert_eq!(state.workflow_loop_count, 2);

        let other = TurnSummary {
            user_message: "different".to_owned(),
            ..Default::default()
        };
        state.observe(&other, 1000);
        assert_eq!(state.workflow_loop_count, 1);
    }

    #[test]
    fn same_file_edit_tracks_max() {
        let mut state = SessionState::new(RunScope::AdHoc);
        let summary_a = TurnSummary {
            files_modified: vec!["a.rs".to_owned(), "b.rs".to_owned()],
            ..Default::default()
        };
        state.observe(&summary_a, 1000);
        assert_eq!(state.same_file_edit_count, 1);

        let summary_b = TurnSummary {
            files_modified: vec!["a.rs".to_owned()],
            ..Default::default()
        };
        state.observe(&summary_b, 1000);
        state.observe(&summary_b, 1000);
        assert_eq!(state.same_file_edit_count, 3);
    }

    #[test]
    fn reset_after_promotion_clears_per_turn_signals() {
        let mut state = SessionState::new(RunScope::AdHoc);
        let summary = TurnSummary {
            user_message: "アプリ全体 をフルスクラッチで作ってOAuth対応も".to_owned(),
            files_modified: vec!["a.rs".to_owned()],
            diff_lines: 500,
            ..Default::default()
        };
        state.observe(&summary, 1000);
        state.observe(&summary, 1000);
        assert!(state.last_user_input_size_signal);
        assert!(state.session_diff_lines > 0);

        state.reset_after_promotion();
        assert!(!state.last_user_input_size_signal);
        assert_eq!(state.session_diff_lines, 0);
        assert_eq!(state.workflow_loop_count, 0);
        assert_eq!(state.same_file_edit_count, 0);
    }

    /// Post-promotion observation must start cleanly: the
    /// `last_user_message` tracker is cleared so the first turn after
    /// promotion does NOT count as a repeat of the pre-promotion
    /// prompt.
    #[test]
    fn reset_after_promotion_clears_last_user_message_tracker() {
        let mut state = SessionState::new(RunScope::AdHoc);
        let prompt = TurnSummary {
            user_message: "rebuild everything".to_owned(),
            ..Default::default()
        };
        state.observe(&prompt, 1000);
        state.observe(&prompt, 1000);
        assert_eq!(state.workflow_loop_count, 2);

        state.reset_after_promotion();

        // The same prompt arrives again post-promotion — the loop
        // counter must reset to 1, not 3, because the tracker was
        // cleared.
        state.observe(&prompt, 1000);
        assert_eq!(
            state.workflow_loop_count, 1,
            "post-promotion observation must start a fresh streak",
        );
    }
}
