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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolConfig {
    /// List of llama-server endpoint URLs for subagent requests.
    pub endpoints: Vec<String>,

    /// Load balancing strategy to use when selecting an endpoint.
    #[serde(default)]
    pub strategy: LoadBalanceStrategy,
}

/// Errors from [`PoolConfig::validate`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PoolConfigError {
    /// The endpoints list is empty.
    #[error("endpoints list must not be empty")]
    EmptyEndpoints,

    /// An endpoint string is empty or whitespace-only.
    #[error("endpoint at index {index} is empty")]
    EmptyEndpointString {
        /// Zero-based index of the offending entry.
        index: usize,
    },

    /// An endpoint string is not a valid URL.
    #[error("endpoint at index {index} is not a valid URL: {url:?}")]
    MalformedUrl {
        /// Zero-based index of the offending entry.
        index: usize,
        /// The invalid URL string.
        url: String,
    },
}

/// Outcome of [`PoolConfig::validate_relaxed`].
///
/// Captures the operator-friendly invariants for `[llm.subagent_pool]`:
/// an empty `endpoints` list is **not** an error (it is the explicit
/// way to disable the pool), and duplicate URLs are deduped with a
/// warning rather than rejected. Malformed URLs and empty / whitespace
/// strings are still hard errors so a typo is reported at load time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    /// Endpoints after dedupe, preserving the original order of the
    /// first occurrence.
    pub deduped_endpoints: Vec<String>,
    /// Duplicate URLs that were collapsed (each entry appears once
    /// even if it was repeated more than twice).
    pub duplicates: Vec<String>,
    /// `true` when `endpoints` was empty after deserialization,
    /// signalling the pool is disabled and the caller should fall back
    /// to the main endpoint.
    pub disabled: bool,
}

impl PoolConfig {
    /// Create a pool config with a single endpoint (no actual pooling).
    pub fn single(endpoint: impl Into<String>) -> Self {
        Self {
            endpoints: vec![endpoint.into()],
            strategy: LoadBalanceStrategy::RoundRobin,
        }
    }

    /// Validate the pool configuration.
    ///
    /// Checks for:
    /// - Empty endpoints list
    /// - Empty or whitespace-only endpoint strings
    /// - Malformed URLs (must start with `http://` or `https://`)
    ///
    /// # Errors
    ///
    /// Returns the first validation error found.
    pub fn validate(&self) -> Result<(), PoolConfigError> {
        if self.endpoints.is_empty() {
            return Err(PoolConfigError::EmptyEndpoints);
        }

        for (index, url) in self.endpoints.iter().enumerate() {
            let trimmed = url.trim();
            if trimmed.is_empty() {
                return Err(PoolConfigError::EmptyEndpointString { index });
            }

            if !trimmed.starts_with("http://") && !trimmed.starts_with("https://") {
                return Err(PoolConfigError::MalformedUrl {
                    index,
                    url: url.clone(),
                });
            }
        }

        Ok(())
    }

    /// Operator-friendly validation that matches the SPEC §10.1
    /// `[llm.subagent_pool]` policy:
    ///
    /// - An empty `endpoints` list is **not** an error: it means "pool
    ///   disabled, route every subagent to the main endpoint".
    /// - Duplicate URLs are deduped (first occurrence wins) and
    ///   surfaced through [`ValidationReport::duplicates`] so the
    ///   caller can warn / log without aborting the run.
    /// - Empty / whitespace-only strings and malformed URLs remain
    ///   hard errors — these almost always indicate a typo and should
    ///   surface at config-load time rather than at first request.
    ///
    /// The strategy enum is already validated by `serde` at
    /// deserialize time, so out-of-range values cannot reach this
    /// path.
    ///
    /// # Errors
    ///
    /// Returns [`PoolConfigError::EmptyEndpointString`] or
    /// [`PoolConfigError::MalformedUrl`] for the first offending entry.
    /// Empty endpoints lists never produce an error here.
    pub fn validate_relaxed(&self) -> Result<ValidationReport, PoolConfigError> {
        if self.endpoints.is_empty() {
            return Ok(ValidationReport {
                deduped_endpoints: Vec::new(),
                duplicates: Vec::new(),
                disabled: true,
            });
        }

        for (index, url) in self.endpoints.iter().enumerate() {
            let trimmed = url.trim();
            if trimmed.is_empty() {
                return Err(PoolConfigError::EmptyEndpointString { index });
            }
            if !trimmed.starts_with("http://") && !trimmed.starts_with("https://") {
                return Err(PoolConfigError::MalformedUrl {
                    index,
                    url: url.clone(),
                });
            }
        }

        // Stable dedupe: preserve the first occurrence's position so
        // the round-robin order matches the operator's intent.
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut deduped: Vec<String> = Vec::with_capacity(self.endpoints.len());
        let mut duplicates: Vec<String> = Vec::new();
        for url in &self.endpoints {
            if seen.insert(url.clone()) {
                deduped.push(url.clone());
            } else if !duplicates.contains(url) {
                duplicates.push(url.clone());
            }
        }

        Ok(ValidationReport {
            deduped_endpoints: deduped,
            duplicates,
            disabled: false,
        })
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

    #[test]
    fn validate_ok() {
        let config = PoolConfig {
            endpoints: vec![
                "http://localhost:8081".to_owned(),
                "https://example.com".to_owned(),
            ],
            strategy: LoadBalanceStrategy::RoundRobin,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_empty_endpoints() {
        let config = PoolConfig {
            endpoints: vec![],
            strategy: LoadBalanceStrategy::RoundRobin,
        };
        assert_eq!(config.validate(), Err(PoolConfigError::EmptyEndpoints));
    }

    #[test]
    fn validate_empty_string() {
        let config = PoolConfig {
            endpoints: vec!["http://ok".to_owned(), "  ".to_owned()],
            strategy: LoadBalanceStrategy::RoundRobin,
        };
        assert!(matches!(
            config.validate(),
            Err(PoolConfigError::EmptyEndpointString { index: 1 })
        ));
    }

    #[test]
    fn validate_malformed_url() {
        let config = PoolConfig {
            endpoints: vec!["not-a-url".to_owned()],
            strategy: LoadBalanceStrategy::RoundRobin,
        };
        assert!(matches!(
            config.validate(),
            Err(PoolConfigError::MalformedUrl { index: 0, .. })
        ));
    }

    #[test]
    fn validate_relaxed_empty_endpoints_disables_pool() {
        let config = PoolConfig {
            endpoints: vec![],
            strategy: LoadBalanceStrategy::RoundRobin,
        };
        let report = config.validate_relaxed().expect("relaxed validate ok");
        assert!(report.disabled, "empty endpoints must disable the pool");
        assert!(report.deduped_endpoints.is_empty());
        assert!(report.duplicates.is_empty());
    }

    #[test]
    fn validate_relaxed_dedupes_duplicates() {
        let config = PoolConfig {
            endpoints: vec![
                "http://a:8080".to_owned(),
                "http://b:8080".to_owned(),
                "http://a:8080".to_owned(),
                "http://c:8080".to_owned(),
                "http://b:8080".to_owned(),
            ],
            strategy: LoadBalanceStrategy::RoundRobin,
        };
        let report = config.validate_relaxed().expect("relaxed validate ok");
        assert!(!report.disabled);
        assert_eq!(
            report.deduped_endpoints,
            vec![
                "http://a:8080".to_owned(),
                "http://b:8080".to_owned(),
                "http://c:8080".to_owned(),
            ],
        );
        assert_eq!(
            report.duplicates,
            vec!["http://a:8080".to_owned(), "http://b:8080".to_owned()],
        );
    }

    #[test]
    fn validate_relaxed_rejects_malformed_url() {
        let config = PoolConfig {
            endpoints: vec!["http://ok".to_owned(), "not-a-url".to_owned()],
            strategy: LoadBalanceStrategy::RoundRobin,
        };
        assert!(matches!(
            config.validate_relaxed(),
            Err(PoolConfigError::MalformedUrl { index: 1, .. })
        ));
    }

    #[test]
    fn validate_relaxed_rejects_empty_string() {
        let config = PoolConfig {
            endpoints: vec!["http://ok".to_owned(), "  ".to_owned()],
            strategy: LoadBalanceStrategy::RoundRobin,
        };
        assert!(matches!(
            config.validate_relaxed(),
            Err(PoolConfigError::EmptyEndpointString { index: 1 })
        ));
    }
}
