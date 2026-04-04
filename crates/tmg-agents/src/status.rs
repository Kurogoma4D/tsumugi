//! Subagent status with type-level state transitions.
//!
//! Uses an enum with explicit transition methods to prevent invalid
//! state changes at the type level. The `transition_to_running`,
//! `complete`, and `fail` methods consume the status value and return
//! the new status, enforcing valid transitions.

/// The lifecycle status of a subagent instance.
///
/// Valid transitions:
/// - `Pending` -> `Running`
/// - `Running` -> `Completed`
/// - `Running` -> `Failed`
/// - `Running` -> `Cancelled`
///
/// Invalid transitions (e.g. `Pending` -> `Completed`) are prevented
/// by the typed transition methods which check the current variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubagentStatus {
    /// The subagent has been created but not yet started.
    Pending,
    /// The subagent is currently executing.
    Running,
    /// The subagent completed successfully with a result.
    Completed {
        /// The subagent's final output text.
        result: String,
    },
    /// The subagent failed with an error.
    Failed {
        /// The error message.
        error: String,
    },
    /// The subagent was cancelled before completion.
    Cancelled,
}

impl SubagentStatus {
    /// Transition from `Pending` to `Running`.
    ///
    /// Returns `None` if the current status is not `Pending`.
    pub fn transition_to_running(self) -> Option<Self> {
        match self {
            Self::Pending => Some(Self::Running),
            _ => None,
        }
    }

    /// Transition from `Running` to `Completed` with a result.
    ///
    /// Returns `None` if the current status is not `Running`.
    pub fn complete(self, result: String) -> Option<Self> {
        match self {
            Self::Running => Some(Self::Completed { result }),
            _ => None,
        }
    }

    /// Transition from `Running` to `Failed` with an error message.
    ///
    /// Returns `None` if the current status is not `Running`.
    pub fn fail(self, error: String) -> Option<Self> {
        match self {
            Self::Running => Some(Self::Failed { error }),
            _ => None,
        }
    }

    /// Transition from `Running` to `Cancelled`.
    ///
    /// Returns `None` if the current status is not `Running`.
    pub fn cancel(self) -> Option<Self> {
        match self {
            Self::Running => Some(Self::Cancelled),
            _ => None,
        }
    }

    /// Whether this status represents a terminal state (completed,
    /// failed, or cancelled).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed { .. } | Self::Failed { .. } | Self::Cancelled
        )
    }

    /// A short human-readable label for the status.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed { .. } => "completed",
            Self::Failed { .. } => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

impl std::fmt::Display for SubagentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Running => write!(f, "running"),
            Self::Completed { result } => {
                let preview = if result.len() > 80 {
                    format!("{}...", &result[..77])
                } else {
                    result.clone()
                };
                write!(f, "completed: {preview}")
            }
            Self::Failed { error } => write!(f, "failed: {error}"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_to_running() {
        let status = SubagentStatus::Pending;
        let status = status.transition_to_running();
        assert_eq!(status, Some(SubagentStatus::Running));
    }

    #[test]
    fn running_to_completed() {
        let status = SubagentStatus::Running;
        let status = status.complete("done".to_owned());
        assert_eq!(
            status,
            Some(SubagentStatus::Completed {
                result: "done".to_owned()
            })
        );
    }

    #[test]
    fn running_to_failed() {
        let status = SubagentStatus::Running;
        let status = status.fail("oops".to_owned());
        assert_eq!(
            status,
            Some(SubagentStatus::Failed {
                error: "oops".to_owned()
            })
        );
    }

    #[test]
    fn running_to_cancelled() {
        let status = SubagentStatus::Running;
        let status = status.cancel();
        assert_eq!(status, Some(SubagentStatus::Cancelled));
    }

    #[test]
    fn invalid_pending_to_completed() {
        let status = SubagentStatus::Pending;
        let result = status.complete("done".to_owned());
        assert!(result.is_none());
    }

    #[test]
    fn invalid_completed_to_running() {
        let status = SubagentStatus::Completed {
            result: "done".to_owned(),
        };
        let result = status.transition_to_running();
        assert!(result.is_none());
    }

    #[test]
    fn invalid_failed_to_completed() {
        let status = SubagentStatus::Failed {
            error: "err".to_owned(),
        };
        let result = status.complete("done".to_owned());
        assert!(result.is_none());
    }

    #[test]
    fn is_terminal() {
        assert!(!SubagentStatus::Pending.is_terminal());
        assert!(!SubagentStatus::Running.is_terminal());
        assert!(
            SubagentStatus::Completed {
                result: String::new()
            }
            .is_terminal()
        );
        assert!(
            SubagentStatus::Failed {
                error: String::new()
            }
            .is_terminal()
        );
        assert!(SubagentStatus::Cancelled.is_terminal());
    }

    #[test]
    fn label_values() {
        assert_eq!(SubagentStatus::Pending.label(), "pending");
        assert_eq!(SubagentStatus::Running.label(), "running");
        assert_eq!(
            SubagentStatus::Completed {
                result: String::new()
            }
            .label(),
            "completed"
        );
        assert_eq!(
            SubagentStatus::Failed {
                error: String::new()
            }
            .label(),
            "failed"
        );
        assert_eq!(SubagentStatus::Cancelled.label(), "cancelled");
    }
}
