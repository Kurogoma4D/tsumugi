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
    #[serde(default)]
    pub allowed_domains: Vec<String>,

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
}
