//! Sandbox configuration.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::mode::SandboxMode;

/// Default timeout in seconds for shell command execution.
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Default OOM score adjustment for child processes.
///
/// A higher score makes the process more likely to be killed by the
/// kernel OOM killer. 500 is moderately aggressive.
const DEFAULT_OOM_SCORE_ADJ: i32 = 500;

/// Configuration for the sandbox environment.
///
/// This is typically deserialized from the `[sandbox]` section of
/// `tsumugi.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// The workspace directory that the agent operates within.
    ///
    /// This directory receives read (and optionally write) access
    /// depending on the [`SandboxMode`].
    pub workspace: PathBuf,

    /// The sandbox operating mode.
    #[serde(default)]
    pub mode: SandboxMode,

    /// Domains allowed for outbound network access.
    ///
    /// When non-empty on Linux, a network namespace is created and
    /// iptables rules restrict outbound traffic to only these domains
    /// (resolved to IP addresses at sandbox creation time).
    ///
    /// This is the legacy top-level form. New configurations should
    /// prefer the `[sandbox.network]` table (see [`NetworkConfig`])
    /// which also exposes the `strict` toggle. When both are present,
    /// the union of `allowed_domains` is used.
    #[serde(default)]
    pub allowed_domains: Vec<String>,

    /// Network restriction settings (`[sandbox.network]` in TOML).
    ///
    /// Carries `allowed_domains` and the `strict` capability-failure
    /// toggle as described in SPEC §6.2.
    #[serde(default)]
    pub network: NetworkConfig,

    /// Maximum execution time in seconds for shell commands.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,

    /// OOM score adjustment for child processes (Linux only).
    ///
    /// Range: -1000 to 1000. Higher values make the process more
    /// likely to be killed by the OOM killer.
    #[serde(default = "default_oom_score_adj")]
    pub oom_score_adj: i32,
}

/// `[sandbox.network]` settings (SPEC §6.2).
///
/// Controls how the agent enforces outbound network restrictions on
/// Linux. On non-Linux platforms these fields are still accepted but
/// ignored (the platform fallback emits a warning).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Domains allowed for outbound network access.
    ///
    /// Resolved once at sandbox creation time and installed as
    /// per-IP `iptables` `ACCEPT` rules. Re-resolution on TTL expiry
    /// is not yet implemented (tracked as future work).
    #[serde(default)]
    pub allowed_domains: Vec<String>,

    /// When `true`, missing `CAP_NET_ADMIN` (or any other failure to
    /// install the ACL) causes sandbox activation to fail. When
    /// `false` (the default), the agent emits a warning and continues
    /// with no network restriction.
    #[serde(default)]
    pub strict: bool,
}

const fn default_timeout_secs() -> u64 {
    DEFAULT_TIMEOUT_SECS
}

const fn default_oom_score_adj() -> i32 {
    DEFAULT_OOM_SCORE_ADJ
}

impl SandboxConfig {
    /// Create a new sandbox configuration with the given workspace path.
    ///
    /// All other settings use their defaults (workspace-write mode,
    /// no allowed domains, 30s timeout, OOM score 500).
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self {
            workspace: workspace.into(),
            mode: SandboxMode::default(),
            allowed_domains: Vec::new(),
            network: NetworkConfig::default(),
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            oom_score_adj: DEFAULT_OOM_SCORE_ADJ,
        }
    }

    /// Set the sandbox mode.
    #[must_use]
    pub fn with_mode(mut self, mode: SandboxMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the allowed domains for network access.
    #[must_use]
    pub fn with_allowed_domains(mut self, domains: Vec<String>) -> Self {
        self.allowed_domains = domains;
        self
    }

    /// Set the timeout for shell commands.
    #[must_use]
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_secs = seconds;
        self
    }

    /// Set the structured `[sandbox.network]` configuration.
    #[must_use]
    pub fn with_network(mut self, network: NetworkConfig) -> Self {
        self.network = network;
        self
    }

    /// Return the effective list of allowed domains, merging the legacy
    /// top-level [`Self::allowed_domains`] with the structured
    /// [`NetworkConfig::allowed_domains`]. Duplicates are removed while
    /// preserving order (legacy entries first, then any new entries from
    /// the structured form).
    #[must_use]
    pub fn effective_allowed_domains(&self) -> Vec<String> {
        let mut merged = self.allowed_domains.clone();
        for d in &self.network.allowed_domains {
            if !merged.contains(d) {
                merged.push(d.clone());
            }
        }
        merged
    }

    /// Whether strict network enforcement is enabled.
    #[must_use]
    pub fn network_strict(&self) -> bool {
        self.network.strict
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = SandboxConfig::new("/tmp/workspace");
        assert_eq!(config.workspace, PathBuf::from("/tmp/workspace"));
        assert_eq!(config.mode, SandboxMode::WorkspaceWrite);
        assert!(config.allowed_domains.is_empty());
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.oom_score_adj, 500);
    }

    #[test]
    fn builder_methods() {
        let config = SandboxConfig::new("/tmp/ws")
            .with_mode(SandboxMode::ReadOnly)
            .with_allowed_domains(vec!["example.com".to_owned()])
            .with_timeout(60);

        assert_eq!(config.mode, SandboxMode::ReadOnly);
        assert_eq!(config.allowed_domains, vec!["example.com"]);
        assert_eq!(config.timeout_secs, 60);
    }

    #[test]
    fn serde_roundtrip() {
        let config = SandboxConfig::new("/tmp/ws")
            .with_mode(SandboxMode::ReadOnly)
            .with_allowed_domains(vec!["api.example.com".to_owned()]);

        let toml_str =
            toml::to_string(&config).unwrap_or_else(|_| String::from("serialize failed"));
        let deserialized: SandboxConfig =
            toml::from_str(&toml_str).unwrap_or_else(|_| SandboxConfig::new("/fallback"));

        assert_eq!(deserialized.mode, SandboxMode::ReadOnly);
        assert_eq!(deserialized.allowed_domains, vec!["api.example.com"]);
    }

    #[test]
    fn network_section_defaults_are_lenient() {
        let config = SandboxConfig::new("/tmp/ws");
        assert!(config.network.allowed_domains.is_empty());
        assert!(!config.network.strict);
    }

    #[test]
    fn effective_allowed_domains_merges_legacy_and_network() {
        let mut config = SandboxConfig::new("/tmp/ws")
            .with_allowed_domains(vec!["crates.io".to_owned(), "shared.com".to_owned()]);
        config.network.allowed_domains =
            vec!["registry.npmjs.org".to_owned(), "shared.com".to_owned()];

        let merged = config.effective_allowed_domains();
        assert_eq!(
            merged,
            vec!["crates.io", "shared.com", "registry.npmjs.org"]
        );
    }

    #[test]
    fn network_strict_toggle_propagates() {
        let mut config = SandboxConfig::new("/tmp/ws");
        assert!(!config.network_strict());
        config.network.strict = true;
        assert!(config.network_strict());
    }

    #[test]
    fn network_section_round_trips_through_toml() {
        let toml_str = "
            workspace = \"/tmp/ws\"
            mode = \"workspace_write\"
            timeout_secs = 30
            oom_score_adj = 500

            [network]
            allowed_domains = [\"crates.io\"]
            strict = true
        ";
        let parsed: SandboxConfig =
            toml::from_str(toml_str).unwrap_or_else(|_| SandboxConfig::new("/fallback"));
        assert_eq!(parsed.network.allowed_domains, vec!["crates.io"]);
        assert!(parsed.network.strict);
    }
}
