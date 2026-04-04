//! Configuration loading and merging for tsumugi.
//!
//! This module implements the `tsumugi.toml` configuration file loading,
//! merging across multiple sources, and validation. The configuration
//! priority (highest to lowest) is:
//!
//! 1. CLI arguments
//! 2. Environment variables (`TMG_*` prefix)
//! 3. Project-local config (`.tsumugi/tsumugi.toml`)
//! 4. Global config (`~/.config/tsumugi/tsumugi.toml`)
//! 5. Built-in defaults

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;

// ---------------------------------------------------------------------------
// Section types
// ---------------------------------------------------------------------------

/// LLM connection settings from `[llm]` section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmConfig {
    /// llama-server endpoint URL.
    #[serde(default = "default_endpoint")]
    pub endpoint: String,

    /// Model name to use in requests.
    #[serde(default = "default_model")]
    pub model: String,

    /// Maximum context window tokens.
    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: usize,

    /// Fraction of max context at which compression auto-triggers (0.0..=1.0).
    #[serde(default = "default_compression_threshold")]
    pub compression_threshold: f64,

    /// Maximum tokens allowed in a single tool result before truncation.
    #[serde(default = "default_max_tool_result_tokens")]
    pub max_tool_result_tokens: usize,

    /// Tool calling mode: `native`, `prompt_based`, or `auto`.
    #[serde(default)]
    pub tool_calling: String,
}

fn default_endpoint() -> String {
    "http://localhost:8080".to_owned()
}

fn default_model() -> String {
    "default".to_owned()
}

const fn default_max_context_tokens() -> usize {
    8192
}

const fn default_compression_threshold() -> f64 {
    0.8
}

const fn default_max_tool_result_tokens() -> usize {
    4096
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            endpoint: default_endpoint(),
            model: default_model(),
            max_context_tokens: default_max_context_tokens(),
            compression_threshold: default_compression_threshold(),
            max_tool_result_tokens: default_max_tool_result_tokens(),
            tool_calling: String::new(),
        }
    }
}

/// Sandbox settings from `[sandbox]` section.
///
/// This mirrors the fields of [`tmg_sandbox::SandboxConfig`] but uses
/// `Option` wrappers so that partial TOML sections can be deserialized
/// and merged independently.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SandboxConfigSection {
    /// Sandbox operating mode: `read_only`, `workspace_write`, or `full`.
    #[serde(default)]
    pub mode: Option<String>,

    /// Domains allowed for outbound network access.
    #[serde(default)]
    pub allowed_domains: Option<Vec<String>>,

    /// Maximum execution time in seconds for shell commands.
    #[serde(default)]
    pub timeout_secs: Option<u64>,

    /// OOM score adjustment for child processes (Linux only).
    #[serde(default)]
    pub oom_score_adj: Option<i32>,
}

/// TUI display settings from `[tui]` section.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct TuiConfig {
    /// Whether to show the token usage indicator in the header.
    #[serde(default)]
    pub show_token_usage: Option<bool>,
}

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

/// Root configuration, corresponding to the entire `tsumugi.toml` file.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct TsumugiConfig {
    /// LLM connection settings.
    #[serde(default)]
    pub llm: LlmConfig,

    /// Sandbox settings.
    #[serde(default)]
    pub sandbox: SandboxConfigSection,

    /// TUI display settings.
    #[serde(default)]
    pub tui: TuiConfig,

    /// Skill discovery settings.
    #[serde(default)]
    pub skills: tmg_skills::SkillsConfig,
}

// ---------------------------------------------------------------------------
// Merging
// ---------------------------------------------------------------------------

impl LlmConfig {
    /// Merge `other` into `self`. Non-default fields in `other` take
    /// precedence.
    fn merge_from(&mut self, other: &Self) {
        let defaults = Self::default();

        if other.endpoint != defaults.endpoint {
            self.endpoint.clone_from(&other.endpoint);
        }
        if other.model != defaults.model {
            self.model.clone_from(&other.model);
        }
        if other.max_context_tokens != defaults.max_context_tokens {
            self.max_context_tokens = other.max_context_tokens;
        }
        if (other.compression_threshold - defaults.compression_threshold).abs() > f64::EPSILON {
            self.compression_threshold = other.compression_threshold;
        }
        if other.max_tool_result_tokens != defaults.max_tool_result_tokens {
            self.max_tool_result_tokens = other.max_tool_result_tokens;
        }
        if !other.tool_calling.is_empty() {
            self.tool_calling.clone_from(&other.tool_calling);
        }
    }
}

impl SandboxConfigSection {
    /// Merge `other` into `self`. `Some` fields in `other` take precedence.
    fn merge_from(&mut self, other: &Self) {
        if other.mode.is_some() {
            self.mode.clone_from(&other.mode);
        }
        if other.allowed_domains.is_some() {
            self.allowed_domains.clone_from(&other.allowed_domains);
        }
        if other.timeout_secs.is_some() {
            self.timeout_secs = other.timeout_secs;
        }
        if other.oom_score_adj.is_some() {
            self.oom_score_adj = other.oom_score_adj;
        }
    }
}

impl TuiConfig {
    /// Merge `other` into `self`. `Some` fields in `other` take precedence.
    fn merge_from(&mut self, other: &Self) {
        if other.show_token_usage.is_some() {
            self.show_token_usage = other.show_token_usage;
        }
    }
}

impl TsumugiConfig {
    /// Merge `other` into `self`. Fields explicitly set in `other` take
    /// precedence.
    pub fn merge_from(&mut self, other: &Self) {
        self.llm.merge_from(&other.llm);
        self.sandbox.merge_from(&other.sandbox);
        self.tui.merge_from(&other.tui);
        // For skills, non-empty discovery_paths in other replaces self.
        if !other.skills.discovery_paths.is_empty() {
            self.skills
                .discovery_paths
                .clone_from(&other.skills.discovery_paths);
        }
        // Explicit false overrides default true.
        if !other.skills.compat_claude {
            self.skills.compat_claude = false;
        }
        if !other.skills.compat_agent_skills {
            self.skills.compat_agent_skills = false;
        }
    }

    /// Apply environment variable overrides (`TMG_*` prefix).
    ///
    /// This accepts an environment lookup function so that tests can
    /// inject values without modifying the process environment (which is
    /// `unsafe` in Edition 2024).
    pub fn apply_env_overrides(&mut self, env_fn: &dyn Fn(&str) -> Option<String>) {
        if let Some(v) = env_fn("TMG_LLM_ENDPOINT") {
            self.llm.endpoint = v;
        }
        if let Some(v) = env_fn("TMG_LLM_MODEL") {
            self.llm.model = v;
        }
        if let Some(v) = env_fn("TMG_LLM_MAX_CONTEXT_TOKENS") {
            if let Ok(n) = v.parse::<usize>() {
                self.llm.max_context_tokens = n;
            }
        }
        if let Some(v) = env_fn("TMG_LLM_COMPRESSION_THRESHOLD") {
            if let Ok(n) = v.parse::<f64>() {
                self.llm.compression_threshold = n;
            }
        }
        if let Some(v) = env_fn("TMG_LLM_MAX_TOOL_RESULT_TOKENS") {
            if let Ok(n) = v.parse::<usize>() {
                self.llm.max_tool_result_tokens = n;
            }
        }
        if let Some(v) = env_fn("TMG_LLM_TOOL_CALLING") {
            self.llm.tool_calling = v;
        }
        if let Some(v) = env_fn("TMG_SANDBOX_MODE") {
            self.sandbox.mode = Some(v);
        }
        if let Some(v) = env_fn("TMG_SANDBOX_TIMEOUT_SECS") {
            if let Ok(n) = v.parse::<u64>() {
                self.sandbox.timeout_secs = Some(n);
            }
        }
    }

    /// Validate the configuration, returning a structured error on failure.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate endpoint is a valid URL.
        let endpoint = self.llm.endpoint.trim();
        if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
            return Err(ConfigError::InvalidValue {
                field: "llm.endpoint".to_owned(),
                value: self.llm.endpoint.clone(),
                reason: "must start with http:// or https://".to_owned(),
            });
        }

        // Validate model is not empty.
        if self.llm.model.trim().is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "llm.model".to_owned(),
                value: self.llm.model.clone(),
                reason: "must not be empty".to_owned(),
            });
        }

        // Validate compression_threshold is in range.
        if !(0.0..=1.0).contains(&self.llm.compression_threshold) {
            return Err(ConfigError::InvalidValue {
                field: "llm.compression_threshold".to_owned(),
                value: self.llm.compression_threshold.to_string(),
                reason: "must be between 0.0 and 1.0".to_owned(),
            });
        }

        // Validate tool_calling mode if specified.
        if !self.llm.tool_calling.is_empty() {
            let valid = ["native", "prompt_based", "auto"];
            if !valid.contains(&self.llm.tool_calling.as_str()) {
                return Err(ConfigError::InvalidValue {
                    field: "llm.tool_calling".to_owned(),
                    value: self.llm.tool_calling.clone(),
                    reason: "must be one of: native, prompt_based, auto".to_owned(),
                });
            }
        }

        // Validate sandbox mode if specified.
        if let Some(ref mode) = self.sandbox.mode {
            let valid = ["read_only", "workspace_write", "full"];
            if !valid.contains(&mode.as_str()) {
                return Err(ConfigError::InvalidValue {
                    field: "sandbox.mode".to_owned(),
                    value: mode.clone(),
                    reason: "must be one of: read_only, workspace_write, full".to_owned(),
                });
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// File loading
// ---------------------------------------------------------------------------

/// Load a single `tsumugi.toml` file from the given path.
///
/// Returns `Ok(None)` if the file does not exist, `Err` if it exists
/// but cannot be read or parsed.
fn load_toml_file(path: &Path) -> Result<Option<TsumugiConfig>, ConfigError> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(ConfigError::Io {
                path: path.to_owned(),
                source: e,
            });
        }
    };

    let config: TsumugiConfig = toml::from_str(&content).map_err(|e| ConfigError::Parse {
        path: path.to_owned(),
        source: e,
    })?;

    Ok(Some(config))
}

/// Resolve the global config path (`~/.config/tsumugi/tsumugi.toml`).
fn global_config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("tsumugi").join("tsumugi.toml"))
}

/// Resolve the project-local config path (`.tsumugi/tsumugi.toml`
/// relative to the current directory).
fn project_config_path() -> Option<PathBuf> {
    std::env::current_dir()
        .ok()
        .map(|d| d.join(".tsumugi").join("tsumugi.toml"))
}

/// Load configuration from all sources and merge them according to
/// the priority chain.
///
/// If `config_path` is `Some`, only that file is loaded (no global or
/// project-local discovery). Environment variable overrides are always
/// applied.
pub fn load_config(config_path: Option<&Path>) -> Result<TsumugiConfig, ConfigError> {
    load_config_with_env(config_path, &|key| std::env::var(key).ok())
}

/// Load configuration with a custom environment lookup function.
///
/// This is the testable core of [`load_config`].
pub fn load_config_with_env(
    config_path: Option<&Path>,
    env_fn: &dyn Fn(&str) -> Option<String>,
) -> Result<TsumugiConfig, ConfigError> {
    let mut config = TsumugiConfig::default();

    if let Some(path) = config_path {
        // Explicit --config path: only load that file.
        let Some(file_config) = load_toml_file(path)? else {
            return Err(ConfigError::NotFound {
                path: path.to_owned(),
            });
        };
        config = file_config;
    } else {
        // Load global config.
        if let Some(global_path) = global_config_path() {
            if let Some(global) = load_toml_file(&global_path)? {
                config = global;
            }
        }

        // Merge project-local config (higher priority).
        if let Some(project_path) = project_config_path() {
            if let Some(project) = load_toml_file(&project_path)? {
                config.merge_from(&project);
            }
        }
    }

    // Apply environment variable overrides.
    config.apply_env_overrides(env_fn);

    Ok(config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let config = TsumugiConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn merge_project_overrides_global() {
        let mut global = TsumugiConfig::default();
        global.llm.endpoint = "http://global:8080".to_owned();
        global.llm.model = "global-model".to_owned();

        let mut project = TsumugiConfig::default();
        project.llm.endpoint = "http://project:9090".to_owned();
        // model stays default, should not override global

        global.merge_from(&project);

        assert_eq!(global.llm.endpoint, "http://project:9090");
        assert_eq!(global.llm.model, "global-model");
    }

    #[test]
    fn env_overrides_apply() {
        let mut config = TsumugiConfig::default();
        let env_fn = |key: &str| -> Option<String> {
            match key {
                "TMG_LLM_ENDPOINT" => Some("http://env:1234".to_owned()),
                "TMG_LLM_MODEL" => Some("env-model".to_owned()),
                "TMG_LLM_MAX_CONTEXT_TOKENS" => Some("16384".to_owned()),
                "TMG_SANDBOX_MODE" => Some("read_only".to_owned()),
                _ => None,
            }
        };

        config.apply_env_overrides(&env_fn);

        assert_eq!(config.llm.endpoint, "http://env:1234");
        assert_eq!(config.llm.model, "env-model");
        assert_eq!(config.llm.max_context_tokens, 16384);
        assert_eq!(config.sandbox.mode.as_deref(), Some("read_only"));
    }

    #[test]
    fn invalid_endpoint_rejected() {
        let mut config = TsumugiConfig::default();
        config.llm.endpoint = "not-a-url".to_owned();

        let Err(err) = config.validate() else {
            panic!("expected validation to fail for invalid endpoint");
        };
        assert!(
            matches!(err, ConfigError::InvalidValue { ref field, .. } if field == "llm.endpoint"),
            "expected InvalidValue for llm.endpoint, got {err:?}"
        );
    }

    #[test]
    fn empty_model_rejected() {
        let mut config = TsumugiConfig::default();
        config.llm.model = "  ".to_owned();

        let Err(err) = config.validate() else {
            panic!("expected validation to fail for empty model");
        };
        assert!(
            matches!(err, ConfigError::InvalidValue { ref field, .. } if field == "llm.model"),
            "expected InvalidValue for llm.model, got {err:?}"
        );
    }

    #[test]
    fn invalid_compression_threshold_rejected() {
        let mut config = TsumugiConfig::default();
        config.llm.compression_threshold = 1.5;

        let Err(err) = config.validate() else {
            panic!("expected validation to fail for invalid threshold");
        };
        assert!(
            matches!(err, ConfigError::InvalidValue { ref field, .. } if field == "llm.compression_threshold"),
            "expected InvalidValue for compression_threshold, got {err:?}"
        );
    }

    #[test]
    fn invalid_tool_calling_rejected() {
        let mut config = TsumugiConfig::default();
        config.llm.tool_calling = "invalid".to_owned();

        let Err(err) = config.validate() else {
            panic!("expected validation to fail for invalid tool_calling");
        };
        assert!(
            matches!(err, ConfigError::InvalidValue { ref field, .. } if field == "llm.tool_calling"),
            "expected InvalidValue for llm.tool_calling, got {err:?}"
        );
    }

    #[test]
    fn invalid_sandbox_mode_rejected() {
        let mut config = TsumugiConfig::default();
        config.sandbox.mode = Some("bad".to_owned());

        let Err(err) = config.validate() else {
            panic!("expected validation to fail for invalid sandbox mode");
        };
        assert!(
            matches!(err, ConfigError::InvalidValue { ref field, .. } if field == "sandbox.mode"),
            "expected InvalidValue for sandbox.mode, got {err:?}"
        );
    }

    #[test]
    fn load_nonexistent_explicit_path_is_error() {
        let result = load_config_with_env(Some(Path::new("/nonexistent/tsumugi.toml")), &|_| None);
        assert!(matches!(result, Err(ConfigError::NotFound { .. })));
    }

    #[test]
    fn load_from_toml_string() {
        let toml_str = r#"
[llm]
endpoint = "http://custom:1234"
model = "my-model"
max_context_tokens = 4096

[sandbox]
mode = "read_only"
timeout_secs = 60

[tui]
show_token_usage = true

[skills]
discovery_paths = ["/extra"]
compat_claude = false
"#;
        let config: TsumugiConfig = toml::from_str(toml_str).unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(config.llm.endpoint, "http://custom:1234");
        assert_eq!(config.llm.model, "my-model");
        assert_eq!(config.llm.max_context_tokens, 4096);
        assert_eq!(config.sandbox.mode.as_deref(), Some("read_only"));
        assert_eq!(config.sandbox.timeout_secs, Some(60));
        assert_eq!(config.tui.show_token_usage, Some(true));
        assert!(!config.skills.compat_claude);
    }

    #[test]
    fn load_with_env_overrides_full_chain() {
        // Simulate: global has endpoint, env overrides model.
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let config_path = dir.path().join("tsumugi.toml");
        std::fs::write(
            &config_path,
            "[llm]\nendpoint = \"http://file:8080\"\nmodel = \"file-model\"\n",
        )
        .unwrap_or_else(|e| panic!("{e}"));

        let env_fn = |key: &str| -> Option<String> {
            if key == "TMG_LLM_MODEL" {
                Some("env-model".to_owned())
            } else {
                None
            }
        };

        let config =
            load_config_with_env(Some(&config_path), &env_fn).unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(config.llm.endpoint, "http://file:8080");
        assert_eq!(config.llm.model, "env-model");
    }

    #[test]
    fn parse_error_includes_path() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let config_path = dir.path().join("bad.toml");
        std::fs::write(&config_path, "[[invalid toml").unwrap_or_else(|e| panic!("{e}"));

        let result = load_config_with_env(Some(&config_path), &|_| None);
        assert!(
            matches!(result, Err(ConfigError::Parse { ref path, .. }) if path == &config_path),
            "expected Parse error with path, got {result:?}"
        );
    }

    #[test]
    fn sandbox_section_merge() {
        let mut base = TsumugiConfig::default();
        base.sandbox.timeout_secs = Some(30);
        base.sandbox.mode = Some("workspace_write".to_owned());

        let mut overlay = TsumugiConfig::default();
        overlay.sandbox.mode = Some("read_only".to_owned());
        // timeout_secs is None, should not override

        base.merge_from(&overlay);

        assert_eq!(base.sandbox.mode.as_deref(), Some("read_only"));
        assert_eq!(base.sandbox.timeout_secs, Some(30));
    }

    #[test]
    fn tui_section_merge() {
        let mut base = TsumugiConfig::default();
        base.tui.show_token_usage = Some(false);

        let mut overlay = TsumugiConfig::default();
        overlay.tui.show_token_usage = Some(true);

        base.merge_from(&overlay);
        assert_eq!(base.tui.show_token_usage, Some(true));
    }

    #[test]
    fn valid_tool_calling_modes_accepted() {
        for mode in ["native", "prompt_based", "auto"] {
            let mut config = TsumugiConfig::default();
            config.llm.tool_calling = mode.to_owned();
            assert!(config.validate().is_ok(), "mode {mode} should be valid");
        }
    }

    #[test]
    fn valid_sandbox_modes_accepted() {
        for mode in ["read_only", "workspace_write", "full"] {
            let mut config = TsumugiConfig::default();
            config.sandbox.mode = Some(mode.to_owned());
            assert!(config.validate().is_ok(), "mode {mode} should be valid");
        }
    }

    #[test]
    fn empty_tool_calling_is_valid() {
        let config = TsumugiConfig::default();
        // Default empty tool_calling should pass validation (means "auto").
        assert!(config.validate().is_ok());
    }
}
