//! Configuration for the auto-promotion gate.
//!
//! Mirrors the `[harness.escalation]` section described in SPEC §10.1
//! and consumed by [`EscalationEvaluator`](super::EscalationEvaluator).
//! The struct lives in [`tmg_harness`] (rather than the CLI) because
//! the evaluator itself enforces the validation invariants — keeping
//! the config close to its consumer makes the contract harder to drift.

use thiserror::Error;

/// Validated configuration for [`EscalationEvaluator`](super::EscalationEvaluator).
///
/// All thresholds are required to be non-zero positive values; the
/// `context_pressure_threshold` is additionally constrained to
/// `0.0 < x <= 1.0`. The CLI's `[harness.escalation]` partial-config
/// converter (in `tmg-cli`) calls [`Self::validated`] and surfaces any
/// violation as a config-load error so misconfigurations are caught
/// before any subagent spawns.
#[derive(Debug, Clone, PartialEq)]
pub struct EscalationConfig {
    /// Master switch. When `false`,
    /// [`EscalationEvaluator::detect_signals`](super::EscalationEvaluator::detect_signals)
    /// returns an empty vec and the harness short-circuits the gate
    /// entirely — manual `/run upgrade` (future issue) is the only
    /// way to escalate.
    pub enabled: bool,

    /// Cumulative session diff threshold above which the diff-size
    /// signal fires.
    pub diff_size_threshold: u32,

    /// Context utilisation threshold (in `0.0..=1.0`) above which the
    /// context-pressure signal becomes eligible. The signal also
    /// requires [`Self::context_pressure_pending_subtasks`] subtasks
    /// to be in flight — both conditions are required.
    pub context_pressure_threshold: f32,

    /// Minimum tool-call count required to qualify as "pending
    /// subtasks" for the context-pressure signal.
    pub context_pressure_pending_subtasks: u32,

    /// Workflow-loop count at or above which the repetition signal
    /// fires.
    pub repetition_workflow_threshold: u32,

    /// Per-file edit count at or above which the file-edit signal
    /// fires.
    pub repetition_file_edit_threshold: u32,
}

impl EscalationConfig {
    /// Construct a fully-validated config.
    ///
    /// # Errors
    ///
    /// Returns [`EscalationConfigError`] when any threshold is zero or
    /// when `context_pressure_threshold` is outside `(0.0, 1.0]`.
    pub fn validated(
        enabled: bool,
        diff_size_threshold: u32,
        context_pressure_threshold: f32,
        context_pressure_pending_subtasks: u32,
        repetition_workflow_threshold: u32,
        repetition_file_edit_threshold: u32,
    ) -> Result<Self, EscalationConfigError> {
        if diff_size_threshold == 0 {
            return Err(EscalationConfigError::ZeroThreshold {
                field: "diff_size_threshold",
            });
        }
        if !(context_pressure_threshold > 0.0 && context_pressure_threshold <= 1.0) {
            return Err(EscalationConfigError::ContextThresholdOutOfRange {
                value: context_pressure_threshold,
            });
        }
        if context_pressure_pending_subtasks == 0 {
            return Err(EscalationConfigError::ZeroThreshold {
                field: "context_pressure_pending_subtasks",
            });
        }
        if repetition_workflow_threshold == 0 {
            return Err(EscalationConfigError::ZeroThreshold {
                field: "repetition_workflow_threshold",
            });
        }
        if repetition_file_edit_threshold == 0 {
            return Err(EscalationConfigError::ZeroThreshold {
                field: "repetition_file_edit_threshold",
            });
        }
        Ok(Self {
            enabled,
            diff_size_threshold,
            context_pressure_threshold,
            context_pressure_pending_subtasks,
            repetition_workflow_threshold,
            repetition_file_edit_threshold,
        })
    }
}

impl Default for EscalationConfig {
    /// Defaults match the SPEC §10.1 example values.
    ///
    /// The defaults are chosen to be conservative: the auto-promotion
    /// gate fires only when the session is genuinely pushing against
    /// the ad-hoc envelope.
    fn default() -> Self {
        Self {
            enabled: true,
            diff_size_threshold: 500,
            context_pressure_threshold: 0.6,
            context_pressure_pending_subtasks: 3,
            repetition_workflow_threshold: 3,
            repetition_file_edit_threshold: 5,
        }
    }
}

/// Validation errors raised by [`EscalationConfig::validated`].
#[derive(Debug, Error, PartialEq)]
pub enum EscalationConfigError {
    /// A non-zero positive integer field was set to zero.
    #[error("[harness.escalation] {field} must be a positive integer (>= 1)")]
    ZeroThreshold {
        /// The offending field name.
        field: &'static str,
    },

    /// `context_pressure_threshold` was out of the expected `(0.0, 1.0]`
    /// range.
    #[error("[harness.escalation] context_pressure_threshold must be in (0.0, 1.0], got {value}")]
    ContextThresholdOutOfRange {
        /// The offending value.
        value: f32,
    },
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;

    #[test]
    fn default_passes_validation() {
        let cfg = EscalationConfig::default();
        let validated = EscalationConfig::validated(
            cfg.enabled,
            cfg.diff_size_threshold,
            cfg.context_pressure_threshold,
            cfg.context_pressure_pending_subtasks,
            cfg.repetition_workflow_threshold,
            cfg.repetition_file_edit_threshold,
        )
        .unwrap_or_else(|e| panic!("default config must validate: {e}"));
        assert_eq!(validated, cfg);
    }

    #[test]
    fn rejects_zero_diff_size_threshold() {
        let result = EscalationConfig::validated(true, 0, 0.6, 3, 3, 5);
        let Err(err) = result else {
            panic!("zero diff_size_threshold must fail");
        };
        assert!(
            matches!(
                err,
                EscalationConfigError::ZeroThreshold {
                    field: "diff_size_threshold"
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn rejects_context_threshold_out_of_range() {
        let result = EscalationConfig::validated(true, 500, 0.0, 3, 3, 5);
        let Err(err) = result else {
            panic!("0.0 must fail");
        };
        assert!(
            matches!(
                err,
                EscalationConfigError::ContextThresholdOutOfRange { .. }
            ),
            "got {err:?}"
        );
        let result = EscalationConfig::validated(true, 500, 1.5, 3, 3, 5);
        let Err(err) = result else {
            panic!("1.5 must fail");
        };
        assert!(
            matches!(
                err,
                EscalationConfigError::ContextThresholdOutOfRange { .. }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn accepts_threshold_at_boundary() {
        let cfg = EscalationConfig::validated(true, 1, 1.0, 1, 1, 1)
            .unwrap_or_else(|e| panic!("boundary values must validate: {e}"));
        assert!((cfg.context_pressure_threshold - 1.0).abs() < f32::EPSILON);
    }
}
