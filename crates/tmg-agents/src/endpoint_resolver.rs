//! Centralised `(endpoint, model)` resolution for subagent spawns.
//!
//! Replaces the ad-hoc precedence ladder previously implemented inline
//! in `SubagentManager`. Issue #50 moved the rules into this module so
//! the SPEC §10.1 / §9.10 / §9.3 precedence is one self-contained,
//! well-tested unit:
//!
//! 1. `AgentKind::Custom`: the `CustomAgentDef`'s `endpoint` /
//!    `model` win when set (non-empty after the `from_strings`
//!    normalisation). Missing fields fall back to the next layer.
//! 2. `AgentKind::Builtin(AgentType::Escalator)`: the
//!    [`EscalatorOverrides`] win when set. Missing fields fall through
//!    to step 4 (the escalator never routes through the pool — see
//!    SPEC §9.10).
//! 3. Every other `AgentKind::Builtin`: the [`tmg_llm::LlmPool`] picks
//!    an endpoint when one is configured with two or more endpoints
//!    (single-endpoint pools are skipped so the `main` fallback
//!    always wins). The model is always inherited from `main` because
//!    the pool only varies the URL.
//! 4. Final fallback: the main `(endpoint, model)` pair.
//!
//! The resolver is `Send + Sync` so the
//! [`SubagentManager`](crate::manager::SubagentManager) can hand a
//! `&self` reference to the spawn closure without an extra
//! `Arc<Mutex<_>>`. Pool selection is async (the pool's read lock
//! lives in tokio); the resolver therefore exposes
//! [`EndpointResolver::resolve`] as an `async fn`.

use std::sync::Arc;

use tmg_llm::LlmPool;

use crate::config::{AgentKind, AgentType};
use crate::manager::EscalatorOverrides;

/// The provenance label written to the event log when an endpoint is
/// resolved. Centralised here so the `--event-log` reader has a stable
/// vocabulary to filter on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionSource {
    /// A [`CustomAgentDef`] supplied an explicit endpoint and/or model.
    Custom,
    /// An [`EscalatorOverrides`] entry took precedence (escalator only).
    EscalatorOverride,
    /// A multi-endpoint [`LlmPool`] picked an endpoint.
    Pool,
    /// The main `(endpoint, model)` pair was used (default fallback).
    Main,
}

impl ResolutionSource {
    /// The string emitted into `event_log` records.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            ResolutionSource::Custom => "custom",
            ResolutionSource::EscalatorOverride => "escalator_override",
            ResolutionSource::Pool => "pool",
            ResolutionSource::Main => "main",
        }
    }
}

/// One resolved `(endpoint, model)` pair plus the source that
/// produced it.
///
/// The endpoint is intentionally kept as `String` rather than `Url` to
/// avoid pulling the `url` crate as a direct dependency: every
/// downstream caller (`LlmClientConfig`, `LlmPool`, the event log)
/// already accepts a `String`. SPEC-compliant URL validation lives in
/// [`tmg_llm::PoolConfig::validate`] and the CLI config validator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedEndpoint {
    /// The base URL that the spawn should route to.
    pub endpoint: String,
    /// The model name to embed in chat-completion requests.
    pub model: String,
    /// Which precedence rule produced this pair.
    pub source: ResolutionSource,
}

/// Resolves the `(endpoint, model)` pair for a subagent spawn.
///
/// Constructed once per [`SubagentManager`](crate::manager::SubagentManager)
/// and consulted on every [`SubagentManager::spawn`] /
/// [`SubagentManager::spawn_with_notify`] call.
#[derive(Debug, Clone)]
pub struct EndpointResolver {
    main_endpoint: String,
    main_model: String,
    pool: Option<Arc<LlmPool>>,
    escalator_override: EscalatorOverrides,
}

impl EndpointResolver {
    /// Build a resolver with only the main `(endpoint, model)` pair.
    /// Equivalent to "no pool, no escalator overrides".
    #[must_use]
    pub fn new(main_endpoint: impl Into<String>, main_model: impl Into<String>) -> Self {
        Self {
            main_endpoint: main_endpoint.into(),
            main_model: main_model.into(),
            pool: None,
            escalator_override: EscalatorOverrides::default(),
        }
    }

    /// Install (or replace) the [`LlmPool`] used at step 3 of the
    /// precedence ladder. A single-endpoint pool is silently skipped
    /// because the resolver would just produce the same URL the
    /// `main` fallback does.
    #[must_use]
    pub fn with_pool(mut self, pool: Option<Arc<LlmPool>>) -> Self {
        self.pool = pool;
        self
    }

    /// Install (or replace) the [`EscalatorOverrides`] consulted at
    /// step 2 of the precedence ladder.
    #[must_use]
    pub fn with_escalator_overrides(mut self, overrides: EscalatorOverrides) -> Self {
        self.escalator_override = overrides;
        self
    }

    /// Replace the [`EscalatorOverrides`] in-place.
    ///
    /// Useful for callers that already own a mutable reference to the
    /// resolver (e.g. [`SubagentManager::set_escalator_overrides`])
    /// and want to avoid cloning the resolver just to swap one field.
    pub fn set_escalator_overrides(&mut self, overrides: EscalatorOverrides) {
        self.escalator_override = overrides;
    }

    /// Borrow the main endpoint URL so the manager can decide whether
    /// to reuse its shared `LlmClient` (when the resolved URL matches)
    /// or construct a per-spawn client.
    #[must_use]
    pub fn main_endpoint(&self) -> &str {
        &self.main_endpoint
    }

    /// Borrow the main model name for the same reason as
    /// [`Self::main_endpoint`].
    #[must_use]
    pub fn main_model(&self) -> &str {
        &self.main_model
    }

    /// Borrow the currently-installed escalator overrides. The
    /// manager reads `disabled` to short-circuit before allocating a
    /// subagent ID.
    #[must_use]
    pub fn escalator_overrides(&self) -> &EscalatorOverrides {
        &self.escalator_override
    }

    /// Resolve the `(endpoint, model)` pair for the supplied
    /// [`AgentKind`].
    ///
    /// See the module docs for the precedence rules. This is `async`
    /// because step 3 may need to acquire the pool's read lock.
    ///
    /// Allocates three `String`s on every call (one per resolved
    /// field). The cost is dwarfed by the spawn / LLM-client
    /// construction the caller does next, so the allocation is
    /// intentionally not optimised away.
    #[must_use = "the resolved endpoint is the only output; ignoring it would make the call pointless"]
    pub async fn resolve(&self, kind: &AgentKind) -> ResolvedEndpoint {
        // Step 1: custom agent overrides.
        if let AgentKind::Custom(def) = kind {
            // Custom agents may set either field independently. We
            // resolve them separately so a custom agent that only sets
            // `endpoint` still picks up `main_model`.
            let endpoint = def
                .endpoint()
                .map_or_else(|| self.main_endpoint.clone(), str::to_owned);
            let model = def
                .model()
                .map_or_else(|| self.main_model.clone(), str::to_owned);
            // Source is `Custom` whenever at least one field came from
            // the def; this matches the operator's mental model of
            // "the custom agent overrode the main config".
            let any_override = def.endpoint().is_some() || def.model().is_some();
            let source = if any_override {
                ResolutionSource::Custom
            } else {
                ResolutionSource::Main
            };
            tracing::debug!(
                agent = %kind.name(),
                endpoint = %endpoint,
                model = %model,
                source = source.as_str(),
                "endpoint resolved",
            );
            return ResolvedEndpoint {
                endpoint,
                model,
                source,
            };
        }

        // Step 2: escalator overrides win for the escalator builtin.
        if matches!(kind, AgentKind::Builtin(AgentType::Escalator)) {
            let endpoint = self
                .escalator_override
                .endpoint
                .clone()
                .unwrap_or_else(|| self.main_endpoint.clone());
            let model = self
                .escalator_override
                .model
                .clone()
                .unwrap_or_else(|| self.main_model.clone());
            let source = if self.escalator_override.endpoint.is_some()
                || self.escalator_override.model.is_some()
            {
                ResolutionSource::EscalatorOverride
            } else {
                ResolutionSource::Main
            };
            tracing::debug!(
                agent = %kind.name(),
                endpoint = %endpoint,
                model = %model,
                source = source.as_str(),
                "endpoint resolved",
            );
            return ResolvedEndpoint {
                endpoint,
                model,
                source,
            };
        }

        // Step 3: try the pool for any other builtin. We only consult
        // multi-endpoint pools so a single-endpoint pool reuses the
        // main fallback (avoids redundant logging and a needless lock
        // acquisition).
        if let Some(pool) = self.pool.as_ref() {
            match pool.pick_endpoint_url().await {
                Ok(Some(picked)) => {
                    tracing::debug!(
                        agent = %kind.name(),
                        endpoint = %picked,
                        model = %self.main_model,
                        source = ResolutionSource::Pool.as_str(),
                        "endpoint resolved",
                    );
                    return ResolvedEndpoint {
                        endpoint: picked,
                        model: self.main_model.clone(),
                        source: ResolutionSource::Pool,
                    };
                }
                Ok(None) => {
                    // Single-endpoint pool: fall through to main.
                }
                Err(e) => {
                    // Pool selection failure (e.g. all endpoints
                    // unhealthy) — log and fall back to main so the
                    // subagent still has a chance to run.
                    tracing::warn!(
                        agent = %kind.name(),
                        error = %e,
                        "pool endpoint selection failed; falling back to main",
                    );
                }
            }
        }

        // Step 4: main fallback.
        tracing::debug!(
            agent = %kind.name(),
            endpoint = %self.main_endpoint,
            model = %self.main_model,
            source = ResolutionSource::Main.as_str(),
            "endpoint resolved",
        );
        ResolvedEndpoint {
            endpoint: self.main_endpoint.clone(),
            model: self.main_model.clone(),
            source: ResolutionSource::Main,
        }
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
#[expect(clippy::panic, reason = "test assertions")]
#[expect(
    clippy::similar_names,
    reason = "tests intentionally pair `resolver` (the under-test value) with `resolved` (its output) for readability"
)]
mod tests {
    use super::*;
    use tmg_llm::{LoadBalanceStrategy, PoolConfig};

    fn main_resolver() -> EndpointResolver {
        EndpointResolver::new("http://main:8080", "main-model")
    }

    #[tokio::test]
    async fn builtin_resolves_to_main_with_no_pool_no_overrides() {
        let resolver = main_resolver();
        let resolved = resolver
            .resolve(&AgentKind::Builtin(AgentType::Explore))
            .await;
        assert_eq!(resolved.endpoint, "http://main:8080");
        assert_eq!(resolved.model, "main-model");
        assert_eq!(resolved.source, ResolutionSource::Main);
    }

    #[tokio::test]
    async fn escalator_uses_overrides_when_set() {
        let resolver = main_resolver().with_escalator_overrides(EscalatorOverrides::from_strings(
            "http://escalator.invalid".to_owned(),
            "lite".to_owned(),
            false,
        ));
        let resolved = resolver
            .resolve(&AgentKind::Builtin(AgentType::Escalator))
            .await;
        assert_eq!(resolved.endpoint, "http://escalator.invalid");
        assert_eq!(resolved.model, "lite");
        assert_eq!(resolved.source, ResolutionSource::EscalatorOverride);
    }

    #[tokio::test]
    async fn escalator_falls_back_to_main_when_overrides_empty() {
        let resolver = main_resolver().with_escalator_overrides(EscalatorOverrides::from_strings(
            String::new(),
            String::new(),
            false,
        ));
        let resolved = resolver
            .resolve(&AgentKind::Builtin(AgentType::Escalator))
            .await;
        assert_eq!(resolved.endpoint, "http://main:8080");
        assert_eq!(resolved.model, "main-model");
        assert_eq!(resolved.source, ResolutionSource::Main);
    }

    #[tokio::test]
    async fn escalator_does_not_route_through_pool() {
        // Even when a multi-endpoint pool is installed, the escalator
        // must not be load-balanced (SPEC §9.10 cost-control).
        let pool = Arc::new(
            LlmPool::new(
                &PoolConfig {
                    endpoints: vec![
                        "http://pool-a:8080".to_owned(),
                        "http://pool-b:8080".to_owned(),
                    ],
                    strategy: LoadBalanceStrategy::RoundRobin,
                },
                "main-model",
            )
            .expect("pool"),
        );
        let resolver = main_resolver().with_pool(Some(pool));
        let resolved = resolver
            .resolve(&AgentKind::Builtin(AgentType::Escalator))
            .await;
        assert_eq!(resolved.endpoint, "http://main:8080");
        assert_eq!(resolved.source, ResolutionSource::Main);
    }

    #[tokio::test]
    async fn builtin_routes_through_multi_endpoint_pool() {
        let pool = Arc::new(
            LlmPool::new(
                &PoolConfig {
                    endpoints: vec![
                        "http://pool-a:8080".to_owned(),
                        "http://pool-b:8080".to_owned(),
                    ],
                    strategy: LoadBalanceStrategy::RoundRobin,
                },
                "main-model",
            )
            .expect("pool"),
        );
        let resolver = main_resolver().with_pool(Some(pool));
        let resolved = resolver
            .resolve(&AgentKind::Builtin(AgentType::Explore))
            .await;
        // The pool starts with all endpoints in `Unknown` health, so
        // the round-robin counter picks `pool-a` first.
        assert_eq!(resolved.endpoint, "http://pool-a:8080");
        assert_eq!(resolved.model, "main-model");
        assert_eq!(resolved.source, ResolutionSource::Pool);
    }

    #[tokio::test]
    async fn builtin_skips_single_endpoint_pool() {
        // A single-endpoint pool is a no-op: it would just yield the
        // same URL as the main fallback, so the resolver should record
        // the main source rather than `pool` to keep event-log
        // semantics meaningful.
        let pool = Arc::new(
            LlmPool::new(&PoolConfig::single("http://main:8080"), "main-model").expect("pool"),
        );
        let resolver = main_resolver().with_pool(Some(pool));
        let resolved = resolver
            .resolve(&AgentKind::Builtin(AgentType::Worker))
            .await;
        assert_eq!(resolved.endpoint, "http://main:8080");
        assert_eq!(resolved.source, ResolutionSource::Main);
    }

    #[tokio::test]
    async fn custom_agent_overrides_main() {
        let toml = r#"
name = "reviewer"
description = "test"
instructions = "do things"
endpoint = "http://custom:7777"
model = "custom-model"

[tools]
allow = ["file_read"]
"#;
        let def = crate::custom::CustomAgentDef::from_toml(toml, "test.toml")
            .unwrap_or_else(|e| panic!("{e}"));
        let kind = AgentKind::Custom(Arc::new(def));
        let resolver = main_resolver();
        let resolved = resolver.resolve(&kind).await;
        assert_eq!(resolved.endpoint, "http://custom:7777");
        assert_eq!(resolved.model, "custom-model");
        assert_eq!(resolved.source, ResolutionSource::Custom);
    }

    #[tokio::test]
    async fn custom_agent_with_partial_override_inherits_main_for_missing_field() {
        let toml = r#"
name = "reviewer"
description = "test"
instructions = "do things"
endpoint = "http://custom:7777"

[tools]
allow = ["file_read"]
"#;
        let def = crate::custom::CustomAgentDef::from_toml(toml, "test.toml")
            .unwrap_or_else(|e| panic!("{e}"));
        let kind = AgentKind::Custom(Arc::new(def));
        let resolver = main_resolver();
        let resolved = resolver.resolve(&kind).await;
        assert_eq!(resolved.endpoint, "http://custom:7777");
        assert_eq!(resolved.model, "main-model");
        assert_eq!(resolved.source, ResolutionSource::Custom);
    }

    /// Two sequential resolves on a 2-endpoint pool must yield two
    /// different URLs, proving the round-robin counter advances on
    /// every call. Regression for issue #50 review: previously only
    /// single-call resolves were tested.
    #[tokio::test]
    async fn pool_round_robin_advances_across_two_resolves() {
        let pool = Arc::new(
            LlmPool::new(
                &PoolConfig {
                    endpoints: vec![
                        "http://pool-a:8080".to_owned(),
                        "http://pool-b:8080".to_owned(),
                    ],
                    strategy: LoadBalanceStrategy::RoundRobin,
                },
                "main-model",
            )
            .expect("pool"),
        );
        let resolver = main_resolver().with_pool(Some(pool));
        let kind = AgentKind::Builtin(AgentType::Worker);
        let first = resolver.resolve(&kind).await;
        let second = resolver.resolve(&kind).await;
        assert_eq!(first.source, ResolutionSource::Pool);
        assert_eq!(second.source, ResolutionSource::Pool);
        assert_ne!(
            first.endpoint, second.endpoint,
            "round-robin must advance: first={}, second={}",
            first.endpoint, second.endpoint,
        );
    }

    /// Concurrent resolves on a 3-endpoint pool must each pick a
    /// distinct URL when the counter is incremented atomically. The
    /// `tokio::join!` schedules both futures on the same runtime, so
    /// any non-atomic increment would surface as a duplicate URL with
    /// reasonable frequency under stress; the deterministic assertion
    /// here is the strongest guarantee we can make without adding a
    /// fuzz harness.
    #[tokio::test]
    async fn pool_round_robin_handles_concurrent_resolves() {
        let pool = Arc::new(
            LlmPool::new(
                &PoolConfig {
                    endpoints: vec![
                        "http://pool-a:8080".to_owned(),
                        "http://pool-b:8080".to_owned(),
                        "http://pool-c:8080".to_owned(),
                    ],
                    strategy: LoadBalanceStrategy::RoundRobin,
                },
                "main-model",
            )
            .expect("pool"),
        );
        let resolver = main_resolver().with_pool(Some(pool));
        let kind = AgentKind::Builtin(AgentType::Worker);

        // Three concurrent resolves should produce a permutation of
        // the three endpoint URLs (the round-robin counter starts at 0
        // and increments atomically). We collect into a `HashSet` to
        // assert uniqueness; any drop in the count proves the
        // `AtomicUsize` did not behave under contention.
        let (a, b, c) = tokio::join!(
            resolver.resolve(&kind),
            resolver.resolve(&kind),
            resolver.resolve(&kind),
        );
        let urls: std::collections::HashSet<String> =
            [a.endpoint, b.endpoint, c.endpoint].into_iter().collect();
        assert_eq!(
            urls.len(),
            3,
            "concurrent resolves must each pick a distinct URL; got {urls:?}",
        );
        assert!(urls.contains("http://pool-a:8080"));
        assert!(urls.contains("http://pool-b:8080"));
        assert!(urls.contains("http://pool-c:8080"));
    }

    #[tokio::test]
    async fn custom_agent_overrides_take_precedence_over_pool() {
        // SPEC §10.1: custom-agent overrides bypass the pool entirely.
        let pool = Arc::new(
            LlmPool::new(
                &PoolConfig {
                    endpoints: vec![
                        "http://pool-a:8080".to_owned(),
                        "http://pool-b:8080".to_owned(),
                    ],
                    strategy: LoadBalanceStrategy::RoundRobin,
                },
                "main-model",
            )
            .expect("pool"),
        );
        let toml = r#"
name = "reviewer"
description = "test"
instructions = "do things"
endpoint = "http://custom:7777"
model = "custom-model"

[tools]
allow = ["file_read"]
"#;
        let def = crate::custom::CustomAgentDef::from_toml(toml, "test.toml")
            .unwrap_or_else(|e| panic!("{e}"));
        let kind = AgentKind::Custom(Arc::new(def));
        let resolver = main_resolver().with_pool(Some(pool));
        let resolved = resolver.resolve(&kind).await;
        assert_eq!(resolved.endpoint, "http://custom:7777");
        assert_eq!(resolved.model, "custom-model");
        assert_eq!(resolved.source, ResolutionSource::Custom);
    }
}
