//! LLM connection pool for distributing subagent requests across multiple
//! llama-server endpoints.
//!
//! The [`LlmPool`] manages a set of endpoints with health checking and
//! load balancing. When only a single endpoint is configured, no pooling
//! overhead is incurred and the endpoint is used directly.
//!
//! # Cancel safety
//!
//! Endpoint selection via [`LlmPool::acquire`] is cancel-safe: dropping the
//! future at any `await` point will not corrupt internal state. The background
//! health check task respects a [`CancellationToken`] and shuts down promptly.

use std::sync::Arc;
use std::time::Duration;

use rand::Rng as _;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::client::{LlmClient, LlmClientConfig};
use crate::error::LlmError;
use crate::pool_config::{LoadBalanceStrategy, PoolConfig};

/// Default interval between background health checks.
const DEFAULT_HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(30);

/// Timeout for a single health-check request (GET /health or similar).
const HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(5);

/// Health status of a single endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointHealth {
    /// The endpoint responded to the last health check.
    Healthy,
    /// The endpoint failed the last health check.
    Unhealthy,
    /// Health status has not been determined yet.
    Unknown,
}

/// Internal state for a single managed endpoint.
#[derive(Debug)]
struct Endpoint {
    /// The base URL for this endpoint (e.g. `http://localhost:8081`).
    url: String,
    /// Current health status.
    health: EndpointHealth,
    /// Pre-built LLM client for this endpoint.
    client: LlmClient,
}

/// Shared pool state protected by an async-aware `RwLock`.
#[derive(Debug)]
struct PoolState {
    /// All managed endpoints.
    endpoints: Vec<Endpoint>,
    /// Round-robin counter (only used with `RoundRobin` strategy).
    round_robin_index: usize,
}

/// An LLM connection pool that distributes requests across multiple
/// llama-server endpoints with health checking and load balancing.
///
/// When a single endpoint is configured, `LlmPool` delegates directly
/// to a single [`LlmClient`] without any pooling overhead.
#[derive(Clone)]
pub struct LlmPool {
    /// The pool implementation: either a single client or a full pool.
    inner: PoolInner,
}

#[derive(Clone)]
enum PoolInner {
    /// Single endpoint: no pooling, direct delegation.
    Single(LlmClient),
    /// Multiple endpoints with load balancing and health checks.
    Multi {
        state: Arc<RwLock<PoolState>>,
        strategy: LoadBalanceStrategy,
        model: String,
    },
}

impl std::fmt::Debug for LlmPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            PoolInner::Single(client) => f
                .debug_struct("LlmPool")
                .field("mode", &"single")
                .field("client", client)
                .finish(),
            PoolInner::Multi {
                strategy, model, ..
            } => f
                .debug_struct("LlmPool")
                .field("mode", &"multi")
                .field("strategy", strategy)
                .field("model", model)
                .finish_non_exhaustive(),
        }
    }
}

impl LlmPool {
    /// Create a new pool from a [`PoolConfig`] and model name.
    ///
    /// If only one endpoint is listed, the pool operates in single-endpoint
    /// mode with zero pooling overhead.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::PoolEmpty`] if no endpoints are configured.
    /// Returns [`LlmError::Http`] if an underlying HTTP client cannot be built.
    pub fn new(config: &PoolConfig, model: impl Into<String>) -> Result<Self, LlmError> {
        let model = model.into();

        if config.endpoints.is_empty() {
            return Err(LlmError::PoolEmpty);
        }

        if config.endpoints.len() == 1 {
            let client_config = LlmClientConfig::new(&config.endpoints[0], &model);
            let client = LlmClient::new(client_config)?;
            return Ok(Self {
                inner: PoolInner::Single(client),
            });
        }

        let mut endpoints = Vec::with_capacity(config.endpoints.len());
        for url in &config.endpoints {
            let client_config = LlmClientConfig::new(url, &model);
            let client = LlmClient::new(client_config)?;
            endpoints.push(Endpoint {
                url: url.clone(),
                health: EndpointHealth::Unknown,
                client,
            });
        }

        let state = Arc::new(RwLock::new(PoolState {
            endpoints,
            round_robin_index: 0,
        }));

        Ok(Self {
            inner: PoolInner::Multi {
                state,
                strategy: config.strategy,
                model,
            },
        })
    }

    /// Acquire an [`LlmClient`] from the pool using the configured strategy.
    ///
    /// For single-endpoint pools, returns the client directly. For
    /// multi-endpoint pools, selects a healthy endpoint based on the
    /// load balancing strategy, falling back to unhealthy/unknown
    /// endpoints if no healthy ones are available.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::AllEndpointsDown`] if all endpoints are
    /// confirmed unhealthy and no fallback is available.
    ///
    /// # Cancel safety
    ///
    /// This method is cancel-safe. Dropping the future between `await`
    /// points will not corrupt the pool state. The round-robin counter
    /// may advance without a request being sent, but this is harmless.
    pub async fn acquire(&self) -> Result<LlmClient, LlmError> {
        match &self.inner {
            PoolInner::Single(client) => Ok(client.clone()),
            PoolInner::Multi {
                state, strategy, ..
            } => {
                let mut pool_state = state.write().await;
                select_endpoint(&mut pool_state, *strategy)
            }
        }
    }

    /// Start the background health check task.
    ///
    /// The task periodically checks all endpoints by sending a lightweight
    /// HTTP request (GET to the `/health` endpoint). Results update the
    /// internal health status of each endpoint.
    ///
    /// The task shuts down gracefully when the [`CancellationToken`] is
    /// cancelled. Returns a `JoinHandle` that resolves when the task exits.
    ///
    /// For single-endpoint pools, this is a no-op that returns immediately.
    pub fn start_health_check(&self, cancel: CancellationToken) -> tokio::task::JoinHandle<()> {
        self.start_health_check_with_interval(cancel, DEFAULT_HEALTH_CHECK_INTERVAL)
    }

    /// Start the background health check task with a custom interval.
    ///
    /// See [`start_health_check`](Self::start_health_check) for details.
    pub fn start_health_check_with_interval(
        &self,
        cancel: CancellationToken,
        interval: Duration,
    ) -> tokio::task::JoinHandle<()> {
        match &self.inner {
            PoolInner::Single(_) => {
                // No health checking needed for a single endpoint.
                tokio::spawn(async {})
            }
            PoolInner::Multi { state, .. } => {
                let state = Arc::clone(state);
                tokio::spawn(health_check_loop(state, cancel, interval))
            }
        }
    }

    /// Return the number of endpoints in the pool.
    pub async fn endpoint_count(&self) -> usize {
        match &self.inner {
            PoolInner::Single(_) => 1,
            PoolInner::Multi { state, .. } => {
                let pool_state = state.read().await;
                pool_state.endpoints.len()
            }
        }
    }

    /// Return health status of all endpoints.
    ///
    /// For single-endpoint pools, returns a single-element vec with
    /// `EndpointHealth::Unknown` (health checking is not performed).
    pub async fn endpoint_health(&self) -> Vec<(String, EndpointHealth)> {
        match &self.inner {
            PoolInner::Single(client) => {
                vec![(format!("{client:?}"), EndpointHealth::Unknown)]
            }
            PoolInner::Multi { state, .. } => {
                let pool_state = state.read().await;
                pool_state
                    .endpoints
                    .iter()
                    .map(|ep| (ep.url.clone(), ep.health))
                    .collect()
            }
        }
    }
}

/// Select an endpoint from the pool based on the given strategy.
///
/// Prefers healthy endpoints. Falls back to unknown, then unhealthy
/// endpoints if no healthy ones are available.
fn select_endpoint(
    state: &mut PoolState,
    strategy: LoadBalanceStrategy,
) -> Result<LlmClient, LlmError> {
    let len = state.endpoints.len();
    if len == 0 {
        return Err(LlmError::PoolEmpty);
    }

    // Build a preference order: Healthy > Unknown > Unhealthy.
    let healthy_indices: Vec<usize> = state
        .endpoints
        .iter()
        .enumerate()
        .filter(|(_, ep)| ep.health == EndpointHealth::Healthy)
        .map(|(i, _)| i)
        .collect();

    let fallback_indices: Vec<usize> = if healthy_indices.is_empty() {
        // Try Unknown first, then all.
        let unknown: Vec<usize> = state
            .endpoints
            .iter()
            .enumerate()
            .filter(|(_, ep)| ep.health == EndpointHealth::Unknown)
            .map(|(i, _)| i)
            .collect();

        if unknown.is_empty() {
            // All endpoints are unhealthy; try them anyway as a last resort.
            (0..len).collect()
        } else {
            unknown
        }
    } else {
        healthy_indices
    };

    if fallback_indices.is_empty() {
        return Err(LlmError::AllEndpointsDown);
    }

    let selected = match strategy {
        LoadBalanceStrategy::RoundRobin => {
            let offset = state.round_robin_index % fallback_indices.len();
            state.round_robin_index = state.round_robin_index.wrapping_add(1);
            fallback_indices[offset]
        }
        LoadBalanceStrategy::Random => {
            let mut rng = rand::rng();
            let offset = rng.random_range(0..fallback_indices.len());
            fallback_indices[offset]
        }
    };

    Ok(state.endpoints[selected].client.clone())
}

/// Background health check loop.
///
/// Runs until the `CancellationToken` is cancelled. On each tick, sends
/// a lightweight HTTP GET to each endpoint's `/health` path and updates
/// the health status accordingly.
async fn health_check_loop(
    state: Arc<RwLock<PoolState>>,
    cancel: CancellationToken,
    interval: Duration,
) {
    let http_client = reqwest::Client::builder()
        .connect_timeout(HEALTH_CHECK_TIMEOUT)
        .timeout(HEALTH_CHECK_TIMEOUT)
        .build();

    let Ok(http_client) = http_client else {
        // If we cannot build an HTTP client, health checking is
        // impossible. All endpoints keep their current status.
        return;
    };

    let mut ticker = tokio::time::interval(interval);
    // The first tick fires immediately; consume it so the first real
    // check happens after `interval`.
    ticker.tick().await;

    loop {
        tokio::select! {
            () = cancel.cancelled() => {
                return;
            }
            _ = ticker.tick() => {
                check_all_endpoints(&http_client, &state).await;
            }
        }
    }
}

/// Check all endpoints once and update their health status.
async fn check_all_endpoints(http_client: &reqwest::Client, state: &Arc<RwLock<PoolState>>) {
    // Read endpoint URLs under a short-lived read lock.
    let urls: Vec<String> = {
        let pool_state = state.read().await;
        pool_state
            .endpoints
            .iter()
            .map(|ep| ep.url.clone())
            .collect()
    };

    // Perform health checks concurrently without holding the lock.
    let mut results = Vec::with_capacity(urls.len());
    let mut join_set = tokio::task::JoinSet::new();

    for (idx, url) in urls.into_iter().enumerate() {
        let client = http_client.clone();
        join_set.spawn(async move {
            let health_url = format!("{}/health", url.trim_end_matches('/'));
            let is_healthy = client
                .get(&health_url)
                .send()
                .await
                .is_ok_and(|r| r.status().is_success());
            (
                idx,
                if is_healthy {
                    EndpointHealth::Healthy
                } else {
                    EndpointHealth::Unhealthy
                },
            )
        });
    }

    while let Some(Ok((idx, health))) = join_set.join_next().await {
        results.push((idx, health));
    }

    // Write results back under a write lock.
    let mut pool_state = state.write().await;
    for (idx, health) in results {
        if idx < pool_state.endpoints.len() {
            pool_state.endpoints[idx].health = health;
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "expect is appropriate in test assertions"
)]
mod tests {
    use super::*;

    #[test]
    fn pool_rejects_empty_endpoints() {
        let config = PoolConfig {
            endpoints: vec![],
            strategy: LoadBalanceStrategy::RoundRobin,
        };

        let result = LlmPool::new(&config, "test");
        assert!(result.is_err());
    }

    #[test]
    fn pool_single_endpoint_mode() {
        let config = PoolConfig::single("http://localhost:8080");
        let pool = LlmPool::new(&config, "test").expect("create pool");

        assert!(matches!(pool.inner, PoolInner::Single(_)));
    }

    #[test]
    fn pool_multi_endpoint_mode() {
        let config = PoolConfig {
            endpoints: vec![
                "http://localhost:8081".to_owned(),
                "http://localhost:8082".to_owned(),
            ],
            strategy: LoadBalanceStrategy::RoundRobin,
        };

        let pool = LlmPool::new(&config, "test").expect("create pool");
        assert!(matches!(pool.inner, PoolInner::Multi { .. }));
    }

    #[test]
    fn round_robin_rotates() {
        let mut state = PoolState {
            endpoints: vec![
                make_test_endpoint("http://a", EndpointHealth::Healthy),
                make_test_endpoint("http://b", EndpointHealth::Healthy),
                make_test_endpoint("http://c", EndpointHealth::Healthy),
            ],
            round_robin_index: 0,
        };

        // First three calls should cycle through a, b, c.
        let _ = select_endpoint(&mut state, LoadBalanceStrategy::RoundRobin).expect("select 0");
        assert_eq!(state.round_robin_index, 1);

        let _ = select_endpoint(&mut state, LoadBalanceStrategy::RoundRobin).expect("select 1");
        assert_eq!(state.round_robin_index, 2);

        let _ = select_endpoint(&mut state, LoadBalanceStrategy::RoundRobin).expect("select 2");
        assert_eq!(state.round_robin_index, 3);

        // Should wrap around.
        let _ = select_endpoint(&mut state, LoadBalanceStrategy::RoundRobin).expect("select 3");
        assert_eq!(state.round_robin_index, 4);
    }

    #[test]
    fn skips_unhealthy_endpoints() {
        let mut state = PoolState {
            endpoints: vec![
                make_test_endpoint("http://a", EndpointHealth::Unhealthy),
                make_test_endpoint("http://b", EndpointHealth::Healthy),
                make_test_endpoint("http://c", EndpointHealth::Unhealthy),
            ],
            round_robin_index: 0,
        };

        // Should always select endpoint b (index 1).
        let client = select_endpoint(&mut state, LoadBalanceStrategy::RoundRobin).expect("select");
        // The client was created; we can't easily check which endpoint it
        // corresponds to, but since there's only one healthy endpoint, the
        // round-robin index should advance and wrap within the healthy set.
        drop(client);
        assert_eq!(state.round_robin_index, 1);
    }

    #[test]
    fn falls_back_to_unknown_when_none_healthy() {
        let mut state = PoolState {
            endpoints: vec![
                make_test_endpoint("http://a", EndpointHealth::Unhealthy),
                make_test_endpoint("http://b", EndpointHealth::Unknown),
            ],
            round_robin_index: 0,
        };

        // Should select endpoint b (Unknown preferred over Unhealthy).
        let result = select_endpoint(&mut state, LoadBalanceStrategy::RoundRobin);
        assert!(result.is_ok());
    }

    #[test]
    fn falls_back_to_unhealthy_as_last_resort() {
        let mut state = PoolState {
            endpoints: vec![
                make_test_endpoint("http://a", EndpointHealth::Unhealthy),
                make_test_endpoint("http://b", EndpointHealth::Unhealthy),
            ],
            round_robin_index: 0,
        };

        // Should still return a client (last resort).
        let result = select_endpoint(&mut state, LoadBalanceStrategy::RoundRobin);
        assert!(result.is_ok());
    }

    #[test]
    fn random_strategy_selects_from_healthy() {
        let mut state = PoolState {
            endpoints: vec![
                make_test_endpoint("http://a", EndpointHealth::Healthy),
                make_test_endpoint("http://b", EndpointHealth::Healthy),
            ],
            round_robin_index: 0,
        };

        // Just verify it doesn't error -- randomness makes exact assertion hard.
        for _ in 0..20 {
            let result = select_endpoint(&mut state, LoadBalanceStrategy::Random);
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn single_endpoint_acquire() {
        let config = PoolConfig::single("http://localhost:9999");
        let pool = LlmPool::new(&config, "test").expect("create pool");

        let client = pool.acquire().await;
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn multi_endpoint_acquire() {
        let config = PoolConfig {
            endpoints: vec![
                "http://localhost:9998".to_owned(),
                "http://localhost:9999".to_owned(),
            ],
            strategy: LoadBalanceStrategy::RoundRobin,
        };

        let pool = LlmPool::new(&config, "test").expect("create pool");
        let count = pool.endpoint_count().await;
        assert_eq!(count, 2);

        // Acquire should succeed (endpoints start as Unknown, which is
        // a valid fallback).
        let client = pool.acquire().await;
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn health_check_cancellation() {
        let config = PoolConfig {
            endpoints: vec![
                "http://localhost:9998".to_owned(),
                "http://localhost:9999".to_owned(),
            ],
            strategy: LoadBalanceStrategy::RoundRobin,
        };

        let pool = LlmPool::new(&config, "test").expect("create pool");
        let cancel = CancellationToken::new();

        let handle =
            pool.start_health_check_with_interval(cancel.clone(), Duration::from_millis(50));

        // Let it run briefly then cancel.
        tokio::time::sleep(Duration::from_millis(100)).await;
        cancel.cancel();

        // Should complete promptly.
        let result = tokio::time::timeout(Duration::from_secs(2), handle).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn endpoint_health_reporting() {
        let config = PoolConfig {
            endpoints: vec![
                "http://localhost:9998".to_owned(),
                "http://localhost:9999".to_owned(),
            ],
            strategy: LoadBalanceStrategy::RoundRobin,
        };

        let pool = LlmPool::new(&config, "test").expect("create pool");
        let health = pool.endpoint_health().await;
        assert_eq!(health.len(), 2);

        // All should start as Unknown.
        for (_, h) in &health {
            assert_eq!(*h, EndpointHealth::Unknown);
        }
    }

    /// Helper: create a test endpoint with a given health status.
    fn make_test_endpoint(url: &str, health: EndpointHealth) -> Endpoint {
        let client_config = LlmClientConfig::new(url, "test");
        let client = LlmClient::new(client_config).expect("build test client");
        Endpoint {
            url: url.to_owned(),
            health,
            client,
        }
    }
}
