//! Subagent status with enum-based state transitions.
//!
//! Uses `&mut self` methods that return `bool` to indicate success,
//! preventing invalid state changes while allowing ergonomic in-place
//! mutation without requiring `clone()`.

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
    /// Returns `true` if the transition was applied, `false` if the
    /// current status is not `Pending`.
    pub fn transition_to_running(&mut self) -> bool {
        if matches!(self, Self::Pending) {
            *self = Self::Running;
            true
        } else {
            false
        }
    }

    /// Transition from `Running` to `Completed` with a result.
    ///
    /// Returns `true` if the transition was applied, `false` if the
    /// current status is not `Running`.
    pub fn complete(&mut self, result: String) -> bool {
        if matches!(self, Self::Running) {
            *self = Self::Completed { result };
            true
        } else {
            false
        }
    }

    /// Transition from `Running` to `Failed` with an error message.
    ///
    /// Returns `true` if the transition was applied, `false` if the
    /// current status is not `Running`.
    pub fn fail(&mut self, error: String) -> bool {
        if matches!(self, Self::Running) {
            *self = Self::Failed { error };
            true
        } else {
            false
        }
    }

    /// Transition from `Running` to `Cancelled`.
    ///
    /// Returns `true` if the transition was applied, `false` if the
    /// current status is not `Running`.
    pub fn cancel(&mut self) -> bool {
        if matches!(self, Self::Running) {
            *self = Self::Cancelled;
            true
        } else {
            false
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

/// Truncate a string to at most `max_chars` characters, returning a
/// `&str` slice. This is safe for multi-byte UTF-8 (CJK, emoji, etc.)
/// because it uses `char_indices` to find the correct byte boundary.
pub fn truncate_str(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => &s[..byte_idx],
        None => s,
    }
}

impl std::fmt::Display for SubagentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Running => write!(f, "running"),
            Self::Completed { result } => {
                let preview = if result.chars().count() > 80 {
                    format!("{}...", truncate_str(result, 77))
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
        let mut status = SubagentStatus::Pending;
        assert!(status.transition_to_running());
        assert_eq!(status, SubagentStatus::Running);
    }

    #[test]
    fn running_to_completed() {
        let mut status = SubagentStatus::Running;
        assert!(status.complete("done".to_owned()));
        assert_eq!(
            status,
            SubagentStatus::Completed {
                result: "done".to_owned()
            }
        );
    }

    #[test]
    fn running_to_failed() {
        let mut status = SubagentStatus::Running;
        assert!(status.fail("oops".to_owned()));
        assert_eq!(
            status,
            SubagentStatus::Failed {
                error: "oops".to_owned()
            }
        );
    }

    #[test]
    fn running_to_cancelled() {
        let mut status = SubagentStatus::Running;
        assert!(status.cancel());
        assert_eq!(status, SubagentStatus::Cancelled);
    }

    #[test]
    fn invalid_pending_to_completed() {
        let mut status = SubagentStatus::Pending;
        assert!(!status.complete("done".to_owned()));
        assert_eq!(status, SubagentStatus::Pending);
    }

    #[test]
    fn invalid_completed_to_running() {
        let mut status = SubagentStatus::Completed {
            result: "done".to_owned(),
        };
        assert!(!status.transition_to_running());
    }

    #[test]
    fn invalid_failed_to_completed() {
        let mut status = SubagentStatus::Failed {
            error: "err".to_owned(),
        };
        assert!(!status.complete("done".to_owned()));
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

    #[test]
    fn truncate_str_ascii() {
        assert_eq!(truncate_str("hello world", 5), "hello");
        assert_eq!(truncate_str("hi", 10), "hi");
    }

    #[test]
    fn truncate_str_multibyte() {
        // CJK characters are 3 bytes each in UTF-8.
        let cjk = "あいうえお";
        assert_eq!(truncate_str(cjk, 3), "あいう");
        assert_eq!(truncate_str(cjk, 10), cjk);
    }

    #[test]
    fn truncate_str_emoji() {
        let emoji = "😀😁😂🤣😃";
        assert_eq!(truncate_str(emoji, 2), "😀😁");
    }

    #[test]
    fn display_completed_truncates_long_result() {
        let long = "あ".repeat(100);
        let display = SubagentStatus::Completed { result: long }.to_string();
        assert!(display.starts_with("completed: "));
        assert!(display.ends_with("..."));
        // Should not panic on multi-byte truncation.
    }
}
