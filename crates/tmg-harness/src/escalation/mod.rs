//! Auto-promotion evaluation pipeline.
//!
//! The harness invokes [`EscalationEvaluator`] after every
//! [`AgentLoop::turn`] to decide whether the active ad-hoc run should
//! be promoted to a harnessed run. The pipeline is deliberately
//! cheap-on-the-hot-path:
//!
//! 1. [`EscalationEvaluator::detect_signals`] is a synchronous,
//!    allocation-light comparison against the SPEC §9.10 trigger
//!    table. The harness calls this on every turn end.
//! 2. When at least one signal fires, the harness spawns
//!    [`EscalationEvaluator::evaluate`] on a `tokio::task` so the
//!    LLM-side escalator round-trip never blocks the UI.
//! 3. If the escalator returns `escalate=true`, the harness invokes
//!    [`RunRunner::escalate_to_harnessed`](crate::runner::RunRunner::escalate_to_harnessed)
//!    to drive the SPEC §9.3 7-step promotion.
//!
//! The evaluator owns an optional [`EscalatorLauncher`] so test code
//! can substitute a mock instead of needing a live llama-server. The
//! launcher abstraction also gives the future "in-process escalator"
//! integration a place to slot in without rewriting the evaluator.
//!
//! [`AgentLoop::turn`]: tmg_core::AgentLoop::turn

pub mod config;
pub mod keywords;
pub mod launcher;

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use config::{EscalationConfig, EscalationConfigError};
pub use launcher::{EscalatorLauncher, SubagentEscalatorLauncher};

use tmg_agents::{EscalatorVerdict, parse_verdict};

use crate::state::SessionState;

/// One firing signal from the SPEC §9.10 trigger table.
///
/// Signals are surfaced individually so the eventual escalator prompt
/// can describe **which** rule tripped, not just that *some* rule did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EscalationSignal {
    /// The most recent user prompt mentioned two or more
    /// scale-indicating phrases (see [`keywords::detect_size_signal`]).
    UserInputSize,
    /// Context window utilisation is above
    /// `context_pressure_threshold` AND there are at least
    /// `context_pressure_pending_subtasks` tool calls in flight.
    ContextPressure {
        /// Observed context usage in `0.0..=1.0`.
        usage: f32,
        /// Number of pending subtasks at the time of measurement.
        pending: u32,
    },
    /// Cumulative session diff is above `diff_size_threshold`.
    DiffSize {
        /// Observed cumulative diff lines.
        lines: u32,
    },
    /// The same user prompt has repeated at least
    /// `repetition_workflow_threshold` times this session.
    WorkflowLoop {
        /// Observed repetition count.
        count: u32,
    },
    /// Some single file has been edited at least
    /// `repetition_file_edit_threshold` times this session.
    SameFileEdit {
        /// Observed per-file edit count (max across all files).
        count: u32,
    },
}

impl EscalationSignal {
    /// Short human label for the signal, used in escalator prompts and
    /// `progress.md` records.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::UserInputSize => "user-input-size",
            Self::ContextPressure { .. } => "context-pressure",
            Self::DiffSize { .. } => "diff-size",
            Self::WorkflowLoop { .. } => "workflow-loop",
            Self::SameFileEdit { .. } => "same-file-edit",
        }
    }
}

/// Outcome of an [`EscalationEvaluator::evaluate`] call.
#[derive(Debug, Clone, PartialEq)]
pub enum EscalationDecision {
    /// Escalator returned `escalate=false`; the harness must take no
    /// action.
    Skip {
        /// Justification reported by the escalator subagent.
        reason: String,
    },
    /// Escalator returned `escalate=true`; the harness should drive
    /// the SPEC §9.3 7-step promotion using the supplied reason and
    /// optional feature-count estimate.
    Escalate {
        /// Justification reported by the escalator subagent.
        reason: String,
        /// Optional `estimated_features` field from the verdict.
        estimated_features: Option<u32>,
    },
}

impl EscalationDecision {
    /// Whether this decision asks the harness to promote.
    #[must_use]
    pub fn should_escalate(&self) -> bool {
        matches!(self, Self::Escalate { .. })
    }

    /// Human-readable reason carried by either variant.
    #[must_use]
    pub fn reason(&self) -> &str {
        match self {
            Self::Skip { reason } | Self::Escalate { reason, .. } => reason,
        }
    }
}

/// Errors raised by [`EscalationEvaluator::evaluate`].
#[derive(Debug, Error)]
pub enum EscalationError {
    /// The escalator subagent is disabled or unavailable. The harness
    /// treats this as "do not escalate" rather than a hard failure.
    #[error("escalator unavailable: {0}")]
    Unavailable(String),

    /// The escalator subagent returned an error.
    #[error("escalator subagent error: {0}")]
    Subagent(String),

    /// The escalator's response could not be parsed as a strict
    /// [`EscalatorVerdict`].
    #[error("failed to parse escalator verdict: {0}")]
    Parse(#[from] tmg_agents::EscalatorParseError),
}

/// Auto-promotion evaluator.
///
/// Holds the [`EscalationConfig`] thresholds and an optional
/// [`EscalatorLauncher`]. When the launcher is `None`, the evaluator
/// behaves as if the escalator were disabled — every signal is skipped
/// without blocking. This shape lets test code construct an evaluator
/// that exercises [`Self::detect_signals`] without involving a real
/// LLM client.
pub struct EscalationEvaluator {
    config: EscalationConfig,
    launcher: Option<Arc<dyn EscalatorLauncher>>,
}

impl EscalationEvaluator {
    /// Construct an evaluator with the given config and optional
    /// launcher.
    ///
    /// Pass `launcher = None` to disable the LLM-side round-trip; the
    /// evaluator will return [`EscalationError::Unavailable`] from
    /// [`Self::evaluate`] and [`Self::detect_signals`] still works.
    #[must_use]
    pub fn new(config: EscalationConfig, launcher: Option<Arc<dyn EscalatorLauncher>>) -> Self {
        Self { config, launcher }
    }

    /// Borrow the active config.
    #[must_use]
    pub fn config(&self) -> &EscalationConfig {
        &self.config
    }

    /// Borrow the active launcher (if any).
    #[must_use]
    pub fn launcher(&self) -> Option<&Arc<dyn EscalatorLauncher>> {
        self.launcher.as_ref()
    }

    /// Evaluate the SPEC §9.10 trigger table against `state`.
    ///
    /// Returns the empty vec when:
    ///
    /// - `enabled = false` in the active config, or
    /// - the active run is **not** [`RunScope::AdHoc`](crate::run::RunScope::AdHoc),
    /// - or no individual condition is satisfied.
    ///
    /// Otherwise returns one [`EscalationSignal`] per matching rule.
    /// The harness should hand the resulting vec to [`Self::evaluate`].
    #[must_use]
    pub fn detect_signals(&self, state: &SessionState) -> Vec<EscalationSignal> {
        if !self.config.enabled {
            return Vec::new();
        }
        if !matches!(state.scope, crate::run::RunScope::AdHoc) {
            return Vec::new();
        }

        let mut signals = Vec::new();

        if state.last_user_input_size_signal {
            signals.push(EscalationSignal::UserInputSize);
        }
        if state.context_usage > self.config.context_pressure_threshold
            && state.pending_subtasks >= self.config.context_pressure_pending_subtasks
        {
            signals.push(EscalationSignal::ContextPressure {
                usage: state.context_usage,
                pending: state.pending_subtasks,
            });
        }
        if state.session_diff_lines > self.config.diff_size_threshold {
            signals.push(EscalationSignal::DiffSize {
                lines: state.session_diff_lines,
            });
        }
        if state.workflow_loop_count >= self.config.repetition_workflow_threshold {
            signals.push(EscalationSignal::WorkflowLoop {
                count: state.workflow_loop_count,
            });
        }
        if state.same_file_edit_count >= self.config.repetition_file_edit_threshold {
            signals.push(EscalationSignal::SameFileEdit {
                count: state.same_file_edit_count,
            });
        }

        signals
    }

    /// Drive the escalator subagent and parse its verdict.
    ///
    /// `signals` and `recent_summary` are passed to the launcher
    /// verbatim; the launcher renders them into the escalator's task
    /// prompt. The returned verdict's `escalate` flag determines the
    /// [`EscalationDecision`] variant.
    ///
    /// # Errors
    ///
    /// - [`EscalationError::Unavailable`] when no launcher is
    ///   installed or the launcher reports the escalator as disabled.
    /// - [`EscalationError::Subagent`] for other launcher failures.
    /// - [`EscalationError::Parse`] when the verdict JSON is malformed.
    pub async fn evaluate(
        &self,
        signals: Vec<EscalationSignal>,
        recent_summary: &str,
    ) -> Result<EscalationDecision, EscalationError> {
        let Some(launcher) = self.launcher.as_ref() else {
            return Err(EscalationError::Unavailable(
                "no escalator launcher installed".to_owned(),
            ));
        };

        let verdict_json = launcher.invoke(&signals, recent_summary).await?;
        let verdict: EscalatorVerdict = parse_verdict(&verdict_json)?;
        Ok(if verdict.escalate {
            EscalationDecision::Escalate {
                reason: verdict.reason,
                estimated_features: verdict.estimated_features,
            }
        } else {
            EscalationDecision::Skip {
                reason: verdict.reason,
            }
        })
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use crate::run::RunScope;
    use crate::state::SessionState;

    fn cfg() -> EscalationConfig {
        EscalationConfig::default()
    }

    /// SPEC §9.10 trigger 1: user-input keyword signal.
    #[test]
    fn detects_user_input_size() {
        let evaluator = EscalationEvaluator::new(cfg(), None);
        let mut state = SessionState::new(RunScope::AdHoc);
        state.last_user_input_size_signal = true;

        let signals = evaluator.detect_signals(&state);
        assert!(
            signals.contains(&EscalationSignal::UserInputSize),
            "expected UserInputSize, got {signals:?}"
        );
    }

    /// SPEC §9.10 trigger 2: context pressure (BOTH usage above threshold AND pending subtasks).
    #[test]
    fn detects_context_pressure() {
        let evaluator = EscalationEvaluator::new(cfg(), None);
        let mut state = SessionState::new(RunScope::AdHoc);
        state.context_usage = 0.8;
        state.pending_subtasks = 5;

        let signals = evaluator.detect_signals(&state);
        assert!(
            signals
                .iter()
                .any(|s| matches!(s, EscalationSignal::ContextPressure { .. })),
            "expected ContextPressure, got {signals:?}"
        );

        // Just usage but no pending subtasks → does not fire.
        let mut state = SessionState::new(RunScope::AdHoc);
        state.context_usage = 0.8;
        state.pending_subtasks = 1;
        let signals = evaluator.detect_signals(&state);
        assert!(
            !signals
                .iter()
                .any(|s| matches!(s, EscalationSignal::ContextPressure { .. })),
            "should not fire without pending subtasks: {signals:?}"
        );
    }

    /// SPEC §9.10 trigger 3: diff size.
    #[test]
    fn detects_diff_size() {
        let evaluator = EscalationEvaluator::new(cfg(), None);
        let mut state = SessionState::new(RunScope::AdHoc);
        state.session_diff_lines = 600; // > default 500

        let signals = evaluator.detect_signals(&state);
        assert!(
            signals
                .iter()
                .any(|s| matches!(s, EscalationSignal::DiffSize { .. })),
            "expected DiffSize, got {signals:?}"
        );
    }

    /// SPEC §9.10 trigger 4: workflow loop.
    #[test]
    fn detects_workflow_loop() {
        let evaluator = EscalationEvaluator::new(cfg(), None);
        let mut state = SessionState::new(RunScope::AdHoc);
        state.workflow_loop_count = 3; // == default 3

        let signals = evaluator.detect_signals(&state);
        assert!(
            signals
                .iter()
                .any(|s| matches!(s, EscalationSignal::WorkflowLoop { .. })),
            "expected WorkflowLoop, got {signals:?}"
        );
    }

    /// SPEC §9.10 trigger 5: same-file edit.
    #[test]
    fn detects_same_file_edit() {
        let evaluator = EscalationEvaluator::new(cfg(), None);
        let mut state = SessionState::new(RunScope::AdHoc);
        state.same_file_edit_count = 5; // == default 5

        let signals = evaluator.detect_signals(&state);
        assert!(
            signals
                .iter()
                .any(|s| matches!(s, EscalationSignal::SameFileEdit { .. })),
            "expected SameFileEdit, got {signals:?}"
        );
    }

    /// `enabled = false` short-circuits the entire signal collection
    /// path: no signals are reported even when every condition is met.
    #[test]
    fn disabled_short_circuits_signal_collection() {
        let mut config = cfg();
        config.enabled = false;
        let evaluator = EscalationEvaluator::new(config, None);

        let mut state = SessionState::new(RunScope::AdHoc);
        state.last_user_input_size_signal = true;
        state.context_usage = 0.95;
        state.pending_subtasks = 10;
        state.session_diff_lines = 9999;
        state.workflow_loop_count = 99;
        state.same_file_edit_count = 99;

        assert!(evaluator.detect_signals(&state).is_empty());
    }

    /// Harnessed runs are not evaluated even with full thresholds.
    #[test]
    fn harnessed_scope_skipped() {
        let evaluator = EscalationEvaluator::new(cfg(), None);
        let mut state = SessionState::new(RunScope::harnessed("wf", None));
        state.last_user_input_size_signal = true;
        state.session_diff_lines = 9999;
        state.same_file_edit_count = 99;

        assert!(evaluator.detect_signals(&state).is_empty());
    }

    #[tokio::test]
    async fn evaluate_without_launcher_returns_unavailable() {
        let evaluator = EscalationEvaluator::new(cfg(), None);
        let result = evaluator
            .evaluate(vec![EscalationSignal::UserInputSize], "summary")
            .await;
        let err = match result {
            Ok(decision) => panic!("must fail, got {decision:?}"),
            Err(e) => e,
        };
        assert!(matches!(err, EscalationError::Unavailable(_)), "{err:?}");
    }

    /// Mock launcher returning a positive verdict drives the
    /// `Escalate` decision branch and threads the verdict's reason
    /// and `estimated_features` through unchanged.
    #[tokio::test]
    async fn evaluate_escalate_branch() {
        let launcher = Arc::new(launcher::testing::MockLauncher::with_response(
            r#"{"escalate":true,"reason":"spans crates","estimated_features":36}"#,
        ));
        let evaluator = EscalationEvaluator::new(cfg(), Some(launcher));

        let decision = evaluator
            .evaluate(vec![EscalationSignal::UserInputSize], "summary")
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        match decision {
            EscalationDecision::Escalate {
                reason,
                estimated_features,
            } => {
                assert_eq!(reason, "spans crates");
                assert_eq!(estimated_features, Some(36));
            }
            EscalationDecision::Skip { .. } => panic!("expected Escalate, got Skip"),
        }
    }

    #[tokio::test]
    async fn evaluate_skip_branch() {
        let launcher = Arc::new(launcher::testing::MockLauncher::with_response(
            r#"{"escalate":false,"reason":"too small"}"#,
        ));
        let evaluator = EscalationEvaluator::new(cfg(), Some(launcher));

        let decision = evaluator
            .evaluate(vec![EscalationSignal::DiffSize { lines: 600 }], "summary")
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(!decision.should_escalate());
        assert_eq!(decision.reason(), "too small");
    }

    /// Concurrent invocation: two evaluations on the same evaluator
    /// must both succeed without contention. (Higher-level
    /// "no double-promote" gating is enforced inside `RunRunner`,
    /// tested separately.)
    #[tokio::test]
    async fn concurrent_evaluations_independent() {
        let launcher = Arc::new(launcher::testing::MockLauncher::with_response(
            r#"{"escalate":false,"reason":"ok"}"#,
        ));
        let evaluator = Arc::new(EscalationEvaluator::new(cfg(), Some(launcher)));

        let e1 = Arc::clone(&evaluator);
        let e2 = Arc::clone(&evaluator);
        let h1 = tokio::spawn(async move {
            e1.evaluate(vec![EscalationSignal::UserInputSize], "summary one")
                .await
        });
        let h2 = tokio::spawn(async move {
            e2.evaluate(
                vec![EscalationSignal::DiffSize { lines: 600 }],
                "summary two",
            )
            .await
        });

        let r1 = h1
            .await
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_else(|e| panic!("{e}"));
        let r2 = h2
            .await
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(r1.reason(), "ok");
        assert_eq!(r2.reason(), "ok");
    }
}
