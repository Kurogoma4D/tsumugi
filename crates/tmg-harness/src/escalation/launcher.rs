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
        //
        // The `expect` is sound: `serde_json::to_string_pretty` only
        // fails on (a) `Serialize` implementations that error or (b)
        // non-string-keyed maps. Our payload is a `Value` literal
        // built from `EscalationSignal` (a derive-Serialize enum with
        // string-keyed variants) and `&str`, so the call is
        // statically infallible.
        #[expect(
            clippy::expect_used,
            reason = "rendered payload is a serde_json::Value built from serializable inputs; serialization is infallible"
        )]
        let rendered = serde_json::to_string_pretty(&payload)
            .expect("escalator prompt payload is always serializable");
        rendered
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

    /// Canned launcher response for [`MockLauncher`].
    ///
    /// Tests construct one of these to exercise either the happy path
    /// (`Ok(json)`) or one of the documented error variants without
    /// spinning up a real LLM client. `Ok(json)` is consumed by
    /// [`tmg_agents::parse_verdict`]; `Err` is propagated by the
    /// evaluator unchanged.
    pub enum MockResponse {
        /// Raw JSON string returned by a successful invoke.
        Ok(String),
        /// Pre-built [`EscalationError`] returned by a failing invoke.
        Err(EscalationError),
    }

    impl MockResponse {
        /// Convenience constructor for the common case of a happy
        /// path JSON string.
        pub fn ok(json: impl Into<String>) -> Self {
            Self::Ok(json.into())
        }

        fn clone_for_invoke(&self) -> Result<String, EscalationError> {
            match self {
                Self::Ok(s) => Ok(s.clone()),
                Self::Err(e) => Err(match e {
                    EscalationError::Unavailable(msg) => EscalationError::Unavailable(msg.clone()),
                    EscalationError::Subagent(msg) => EscalationError::Subagent(msg.clone()),
                    // `Parse` carries an opaque `EscalatorParseError`
                    // that does not implement `Clone`; tests that need
                    // a parse error should use the `Ok(invalid_json)`
                    // path so the evaluator constructs a real
                    // `Parse` error from the malformed input.
                    EscalationError::Parse(_) => EscalationError::Subagent(
                        "parse error placeholder; use Ok(malformed_json) instead".to_owned(),
                    ),
                }),
            }
        }
    }

    /// Mock launcher that returns a canned response on every call.
    ///
    /// The canned response can be either a happy-path JSON string
    /// (consumed by [`tmg_agents::parse_verdict`]) or an
    /// [`EscalationError`]; this lets tests exercise the evaluator's
    /// failure-path handling without a real LLM client.
    pub struct MockLauncher {
        /// Response produced on every `invoke` call.
        pub response: MockResponse,
    }

    impl MockLauncher {
        /// Convenience constructor for the (common) happy-path case.
        pub fn with_response(json: impl Into<String>) -> Self {
            Self {
                response: MockResponse::ok(json),
            }
        }

        /// Convenience constructor for a launcher that always returns
        /// the given error.
        pub fn with_error(err: EscalationError) -> Self {
            Self {
                response: MockResponse::Err(err),
            }
        }
    }

    impl From<String> for MockResponse {
        fn from(s: String) -> Self {
            Self::Ok(s)
        }
    }

    impl EscalatorLauncher for MockLauncher {
        fn invoke<'a>(
            &'a self,
            _signals: &'a [EscalationSignal],
            _recent_summary: &'a str,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<String, EscalationError>> + Send + 'a>,
        > {
            let result = self.response.clone_for_invoke();
            Box::pin(async move { result })
        }
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use crate::escalation::EscalationEvaluator;
    use crate::escalation::config::EscalationConfig;
    use std::sync::Arc;

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

    /// Failure path: launcher returns [`EscalationError::Subagent`].
    /// The evaluator must surface it unchanged.
    #[tokio::test]
    async fn mock_launcher_subagent_error_propagates() {
        let launcher = Arc::new(testing::MockLauncher::with_error(
            EscalationError::Subagent("LLM timed out".to_owned()),
        ));
        let evaluator = EscalationEvaluator::new(EscalationConfig::default(), Some(launcher));
        let result = evaluator
            .evaluate(vec![EscalationSignal::UserInputSize], "summary")
            .await;
        match result {
            Err(EscalationError::Subagent(msg)) => assert_eq!(msg, "LLM timed out"),
            other => panic!("expected Subagent error, got {other:?}"),
        }
    }

    /// Failure path: launcher returns malformed JSON. The evaluator
    /// must wrap the parse failure as [`EscalationError::Parse`].
    #[tokio::test]
    async fn mock_launcher_malformed_json_yields_parse_error() {
        let launcher = Arc::new(testing::MockLauncher::with_response("{not valid json"));
        let evaluator = EscalationEvaluator::new(EscalationConfig::default(), Some(launcher));
        let result = evaluator
            .evaluate(vec![EscalationSignal::UserInputSize], "summary")
            .await;
        assert!(
            matches!(result, Err(EscalationError::Parse(_))),
            "expected Parse error, got {result:?}",
        );
    }

    /// Failure path: launcher returns
    /// [`EscalationError::Unavailable`]. The evaluator must surface
    /// it unchanged so callers can treat it as "do not escalate"
    /// rather than a hard failure.
    #[tokio::test]
    async fn mock_launcher_unavailable_propagates() {
        let launcher = Arc::new(testing::MockLauncher::with_error(
            EscalationError::Unavailable("escalator disabled".to_owned()),
        ));
        let evaluator = EscalationEvaluator::new(EscalationConfig::default(), Some(launcher));
        let result = evaluator
            .evaluate(vec![EscalationSignal::UserInputSize], "summary")
            .await;
        match result {
            Err(EscalationError::Unavailable(msg)) => assert_eq!(msg, "escalator disabled"),
            other => panic!("expected Unavailable error, got {other:?}"),
        }
    }
}
