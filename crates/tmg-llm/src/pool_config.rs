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

impl LoadBalanceStrategy {
    /// Stable string representation used in `--event-log` records and
    /// other external sinks. The strings match the `serde` rename
    /// (`snake_case`) so the log vocabulary lines up with the TOML
    /// surface. Issue #50.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            LoadBalanceStrategy::RoundRobin => "round_robin",
            LoadBalanceStrategy::Random => "random",
        }
    }
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
///
/// # Dedupe semantics
///
/// Duplicate detection treats two URLs as equal when their normalised
/// form matches. The normaliser:
///
/// 1. trims trailing `/` characters so `http://x:8080` and
///    `http://x:8080/` collapse,
/// 2. trims leading and trailing whitespace,
/// 3. lowercases the scheme and host (e.g. `HTTP://Example.COM` and
///    `http://example.com` collapse).
///
/// The original (un-normalised) string is preserved in
/// [`Self::deduped_endpoints`] so the caller's logs and the pool's
/// runtime behaviour use exactly the URL the operator typed. Only the
/// dedupe key is normalised. Lowercasing of path / query is **not**
/// performed because llama-server treats those as case-sensitive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    /// Endpoints after dedupe, preserving the original order of the
    /// first occurrence and the operator's original casing /
    /// punctuation.
    pub deduped_endpoints: Vec<String>,
    /// Duplicate URLs that were collapsed (each entry appears once
    /// even if it was repeated more than twice). Reports the first
    /// non-canonical occurrence for each collision so logs show the
    /// operator the form they typed.
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
        // the round-robin order matches the operator's intent. The
        // dedupe *key* is the normalised URL (see
        // [`normalize_for_dedupe`]); we keep the original string in
        // `deduped_endpoints` so the live pool routes to exactly the
        // URL the operator typed. The `seen_duplicates` set keeps the
        // duplicate-detection O(n) (the prior `Vec::contains` made it
        // O(n²)).
        let mut seen: std::collections::HashSet<String> =
            std::collections::HashSet::with_capacity(self.endpoints.len());
        let mut seen_duplicates: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut deduped: Vec<String> = Vec::with_capacity(self.endpoints.len());
        let mut duplicates: Vec<String> = Vec::new();
        for url in &self.endpoints {
            let key = normalize_for_dedupe(url);
            if seen.insert(key.clone()) {
                deduped.push(url.clone());
            } else if seen_duplicates.insert(key) {
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

/// Normalise a URL string for dedupe-key comparison.
///
/// Applied only to the dedupe key, never to the stored string. The
/// rules are documented on [`ValidationReport`]; in summary: trim
/// surrounding whitespace, drop trailing `/`, and lowercase the scheme
/// + host so trivially-different spellings collapse.
fn normalize_for_dedupe(url: &str) -> String {
    let trimmed = url.trim().trim_end_matches('/');
    // Split scheme://rest into ("scheme", "rest"). If the URL is
    // malformed (no `://`), fall back to lower-casing the whole thing
    // so dedupe still catches whitespace-only variants. Malformed URLs
    // are caught by the validator above, so this branch is defensive.
    if let Some((scheme, rest)) = trimmed.split_once("://") {
        // Within `rest`, the host is everything up to the next `/` or
        // `?` (whichever comes first). Lowercasing only the host
        // preserves path / query case sensitivity.
        let host_end = rest.find(['/', '?']);
        let (host, tail) = host_end.map_or((rest, ""), |idx| rest.split_at(idx));
        let mut out = String::with_capacity(trimmed.len());
        out.push_str(&scheme.to_ascii_lowercase());
        out.push_str("://");
        out.push_str(&host.to_ascii_lowercase());
        out.push_str(tail);
        out
    } else {
        trimmed.to_ascii_lowercase()
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

    /// Issue #50 review: dedupe must collapse `http://x:8080` and
    /// `http://x:8080/` so the operator does not get a "two-endpoint"
    /// pool that is really one endpoint pointed at twice. The
    /// scheme/host casing is also normalised so a trivially-cased
    /// variant collapses with its lowercase twin. The first
    /// non-canonical occurrence is recorded once in `duplicates` even
    /// when several spellings collide on the same key.
    #[test]
    fn validate_relaxed_dedupes_normalised_variants() {
        let config = PoolConfig {
            endpoints: vec![
                "http://x:8080".to_owned(),
                "http://x:8080/".to_owned(),
                "http://X:8080/".to_owned(),
                "http://y:8080".to_owned(),
            ],
            strategy: LoadBalanceStrategy::RoundRobin,
        };
        let report = config.validate_relaxed().expect("relaxed validate ok");
        assert!(!report.disabled);
        assert_eq!(
            report.deduped_endpoints,
            vec!["http://x:8080".to_owned(), "http://y:8080".to_owned()],
            "dedupe must collapse trailing-slash and case variants while \
             preserving the first occurrence's spelling",
        );
        // All three duplicate spellings map to the same normalised key
        // (`http://x:8080`), so the first non-canonical spelling is
        // recorded exactly once.
        assert_eq!(report.duplicates, vec!["http://x:8080/".to_owned()]);
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
