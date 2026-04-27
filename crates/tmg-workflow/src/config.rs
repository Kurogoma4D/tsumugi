//! Workflow runtime configuration.
//!
//! This is the *engine-facing* shape of the `[workflow]` section in
//! `tsumugi.toml`. The CLI's own config layer (`tmg-cli`) maps its
//! partial / mergeable representation into this struct before handing
//! it to the engine.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// `[workflow]` configuration consumed by [`crate::engine::WorkflowEngine`]
/// and [`crate::discovery::discover_workflows`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkflowConfig {
    /// Extra workflow discovery directories beyond the standard
    /// `<project>/.tsumugi/workflows/` and
    /// `~/.config/tsumugi/workflows/` locations.
    #[serde(default)]
    pub discovery_paths: Vec<PathBuf>,

    /// Default timeout for `shell` steps that omit their own `timeout`.
    ///
    /// Stored as a humantime-style string in TOML (`"30s"`, `"2m"`).
    /// Defaults to 30 seconds.
    #[serde(default = "default_shell_timeout", with = "humantime_serde")]
    pub default_shell_timeout: Duration,

    /// Default model name used for `agent` steps that don't specify
    /// `model`. Empty string means "inherit from `[llm].model`".
    #[serde(default)]
    pub default_agent_model: String,

    /// Maximum number of `parallel` step branches that may run at
    /// once. Reserved for issue #40; the executor in this crate
    /// validates the value at config time but does not yet act on it.
    #[serde(default = "default_max_parallel_agents")]
    pub max_parallel_agents: u32,
}

impl WorkflowConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an `InvalidWorkflow` error when a numeric knob is
    /// outside its supported range.
    pub fn validate(&self) -> Result<(), crate::error::WorkflowError> {
        if self.default_shell_timeout.as_secs() == 0
            && self.default_shell_timeout.subsec_nanos() == 0
        {
            return Err(crate::error::WorkflowError::invalid_workflow(
                "[workflow]",
                "default_shell_timeout must be at least 1 second",
            ));
        }
        if self.max_parallel_agents == 0 {
            return Err(crate::error::WorkflowError::invalid_workflow(
                "[workflow]",
                "max_parallel_agents must be a positive integer (>= 1)",
            ));
        }
        Ok(())
    }
}

fn default_shell_timeout() -> Duration {
    Duration::from_secs(30)
}

const fn default_max_parallel_agents() -> u32 {
    2
}

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            discovery_paths: Vec::new(),
            default_shell_timeout: default_shell_timeout(),
            default_agent_model: String::new(),
            max_parallel_agents: default_max_parallel_agents(),
        }
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions")]
mod tests {
    use super::*;

    #[test]
    fn default_validates() {
        WorkflowConfig::default()
            .validate()
            .unwrap_or_else(|e| panic!("default config should validate: {e}"));
    }

    #[test]
    fn rejects_zero_timeout() {
        let cfg = WorkflowConfig {
            default_shell_timeout: Duration::ZERO,
            ..WorkflowConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_zero_parallel() {
        let cfg = WorkflowConfig {
            max_parallel_agents: 0,
            ..WorkflowConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn parses_from_toml() {
        // Use TOML to mirror how tsumugi.toml stores the section.
        let toml_str = r#"
discovery_paths = ["/custom/workflows"]
default_shell_timeout = "1m"
default_agent_model = ""
max_parallel_agents = 4
"#;
        let cfg: WorkflowConfig = toml::from_str(toml_str).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(cfg.default_shell_timeout, Duration::from_secs(60));
        assert_eq!(cfg.max_parallel_agents, 4);
        assert_eq!(cfg.discovery_paths.len(), 1);
    }

    #[test]
    fn rejects_unknown_field() {
        let toml_str = r"
discovery_paths = []
unknown_knob = 1
";
        let result: std::result::Result<WorkflowConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }
}
