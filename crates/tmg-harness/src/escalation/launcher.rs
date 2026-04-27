//! Pluggable launcher abstraction for the escalator subagent.
//!
//! The evaluator deliberately accepts an
//! [`EscalatorLauncher`] trait object instead of a
//! [`SubagentManager`](tmg_agents::SubagentManager) directly so test
//! code can substitute a mock launcher without spinning up a real LLM
//! client. The production implementation
//! [`SubagentEscalatorLauncher`] dispatches through the manager and
//! formats the prompt on the way in.

use std::sync::Arc;

use tmg_agents::{AgentError, AgentKind, AgentType, SubagentConfig, SubagentManager};
use tokio::sync::Mutex;

use crate::escalation::{EscalationError, EscalationSignal};

/// Abstraction over "ask the escalator subagent for a verdict".
///
/// Implementations return the **raw** JSON string that the escalator
/// emitted; the evaluator parses it via
/// [`tmg_agents::parse_verdict`]. Returning the raw string keeps the
/// trait object-safe (no associated types) and lets the test launcher
/// stay completely synchronous internally.
pub trait EscalatorLauncher: Send + Sync {
    /// Render `signals` + `recent_summary` into a task prompt, spawn
    /// the escalator subagent, and return its final output.
    ///
    /// # Errors
    ///
    /// Implementations should surface
    /// [`EscalationError::Unavailable`] when the escalator is
    /// disabled, and [`EscalationError::Subagent`] for any other
    /// runtime failure (LLM I/O, task join, parser, etc.).
    fn invoke<'a>(
        &'a self,
        signals: &'a [EscalationSignal],
        recent_summary: &'a str,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<String, EscalationError>> + Send + 'a>,
    >;
}

/// Production launcher backed by [`SubagentManager`].
///
/// The launcher serializes the access through `Arc<Mutex<...>>` so the
/// existing TUI lock topology is preserved (`SubagentManager` is already
/// shared this way for `spawn_agent` and the live-summaries panel).
pub struct SubagentEscalatorLauncher {
    manager: Arc<Mutex<SubagentManager>>,
}

impl SubagentEscalatorLauncher {
    /// Construct a launcher around the shared manager handle.
    #[must_use]
    pub fn new(manager: Arc<Mutex<SubagentManager>>) -> Self {
        Self { manager }
    }

    /// Render `signals` + `recent_summary` into the escalator's task
    /// prompt.
    ///
    /// The wording follows SPEC §9.10's expectation that the prompt is
    /// strictly a JSON object with a `signals` array and a
    /// `recent_summary` string. The escalator system prompt knows how
    /// to parse it; we deliberately keep the wrapper minimal so the
    /// model is not led toward one verdict over the other.
    fn render_prompt(signals: &[EscalationSignal], recent_summary: &str) -> String {
        // Use serde_json::to_string so the rendered prompt is parseable
        // round-trip; the escalator system prompt expects strict JSON.
        let payload = serde_json::json!({
            "signals": signals,
            "recent_summary": recent_summary,
        });
        // Pretty-printing is fine: the escalator system prompt is
        // tolerant of whitespace.
        serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_owned())
    }
}

impl EscalatorLauncher for SubagentEscalatorLauncher {
    fn invoke<'a>(
        &'a self,
        signals: &'a [EscalationSignal],
        recent_summary: &'a str,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<String, EscalationError>> + Send + 'a>,
    > {
        Box::pin(async move {
            let prompt = Self::render_prompt(signals, recent_summary);
            let config = SubagentConfig {
                agent_kind: AgentKind::Builtin(AgentType::Escalator),
                task: prompt,
                background: false,
            };

            let mut manager = self.manager.lock().await;
            let id = match manager.spawn(config).await {
                Ok(id) => id,
                Err(AgentError::EscalatorDisabled) => {
                    return Err(EscalationError::Unavailable(
                        "escalator disabled by [harness.escalator] config".to_owned(),
                    ));
                }
                Err(other) => return Err(EscalationError::Subagent(other.to_string())),
            };
            manager
                .wait_for(id)
                .await
                .map_err(|e| EscalationError::Subagent(e.to_string()))
        })
    }
}

#[cfg(test)]
pub mod testing {
    //! Test-only mock launcher.
    //!
    //! Reachable from the crate's unit tests via
    //! `crate::escalation::launcher::testing`. If integration tests
    //! ever need access from a separate crate, promote this to a
    //! `test-util` feature; for now the harness's own unit tests are
    //! the only consumer.

    use super::{EscalationError, EscalationSignal, EscalatorLauncher};

    /// Mock launcher that returns a canned response string. The
    /// response is meant to be a JSON string that
    /// [`tmg_agents::parse_verdict`] can consume.
    pub struct MockLauncher {
        /// The canned response served on every `invoke` call.
        pub response: String,
    }

    impl EscalatorLauncher for MockLauncher {
        fn invoke<'a>(
            &'a self,
            _signals: &'a [EscalationSignal],
            _recent_summary: &'a str,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<String, EscalationError>> + Send + 'a>,
        > {
            let response = self.response.clone();
            Box::pin(async move { Ok(response) })
        }
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;

    #[test]
    fn render_prompt_is_well_formed_json() {
        let signals = vec![
            EscalationSignal::UserInputSize,
            EscalationSignal::DiffSize { lines: 600 },
        ];
        let prompt = SubagentEscalatorLauncher::render_prompt(&signals, "did stuff");
        let parsed: serde_json::Value = serde_json::from_str(&prompt)
            .unwrap_or_else(|e| panic!("rendered prompt must be valid JSON: {e}"));
        assert!(parsed.get("signals").is_some());
        assert!(parsed.get("recent_summary").is_some());
    }
}
