//! Configuration types for the LLM connection pool.
//!
//! These types map to the `[llm.subagent_pool]` section of `tsumugi.toml`.

use serde::{Deserialize, Serialize};

/// Load balancing strategy for distributing requests across endpoints.
///
/// Designed as an enum for extensibility: future strategies such as
/// `LeastConnections` can be added without breaking changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadBalanceStrategy {
    /// Distribute requests in round-robin order across healthy endpoints.
    #[default]
    RoundRobin,

    /// Select a random healthy endpoint for each request.
    Random,
}

/// Configuration for the subagent connection pool.
///
/// Corresponds to the `[llm.subagent_pool]` section of `tsumugi.toml`.
///
/// # Example TOML
///
/// ```toml
/// [llm.subagent_pool]
/// endpoints = ["http://localhost:8081", "http://localhost:8082"]
/// strategy = "round_robin"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// List of llama-server endpoint URLs for subagent requests.
    pub endpoints: Vec<String>,

    /// Load balancing strategy to use when selecting an endpoint.
    #[serde(default)]
    pub strategy: LoadBalanceStrategy,
}

impl PoolConfig {
    /// Create a pool config with a single endpoint (no actual pooling).
    pub fn single(endpoint: impl Into<String>) -> Self {
        Self {
            endpoints: vec![endpoint.into()],
            strategy: LoadBalanceStrategy::RoundRobin,
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
    fn deserialize_pool_config_round_robin() {
        let toml_str = r#"
            endpoints = ["http://localhost:8081", "http://localhost:8082"]
            strategy = "round_robin"
        "#;

        let config: PoolConfig = toml::from_str(toml_str).expect("parse pool config");
        assert_eq!(config.endpoints.len(), 2);
        assert_eq!(config.strategy, LoadBalanceStrategy::RoundRobin);
    }

    #[test]
    fn deserialize_pool_config_random() {
        let toml_str = r#"
            endpoints = ["http://localhost:8081"]
            strategy = "random"
        "#;

        let config: PoolConfig = toml::from_str(toml_str).expect("parse pool config");
        assert_eq!(config.strategy, LoadBalanceStrategy::Random);
    }

    #[test]
    fn deserialize_pool_config_default_strategy() {
        let toml_str = r#"
            endpoints = ["http://localhost:8081"]
        "#;

        let config: PoolConfig = toml::from_str(toml_str).expect("parse pool config");
        assert_eq!(config.strategy, LoadBalanceStrategy::RoundRobin);
    }

    #[test]
    fn single_endpoint_config() {
        let config = PoolConfig::single("http://localhost:8080");
        assert_eq!(config.endpoints.len(), 1);
        assert_eq!(config.endpoints[0], "http://localhost:8080");
    }
}
