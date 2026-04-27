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
use tmg_llm::ToolCallingMode;
use tmg_sandbox::SandboxMode;

use crate::error::ConfigError;

// ---------------------------------------------------------------------------
// Partial types for deserialization / merging
// ---------------------------------------------------------------------------

/// Partial LLM config used for deserialization from TOML.
///
/// All fields are `Option` so we can distinguish "not set" from
/// "explicitly set to the default value".
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PartialLlmConfig {
    pub endpoint: Option<String>,
    pub model: Option<String>,
    pub max_context_tokens: Option<usize>,
    pub compression_threshold: Option<f64>,
    pub max_tool_result_tokens: Option<usize>,
    pub tool_calling: Option<ToolCallingMode>,
}

impl PartialLlmConfig {
    /// Merge `other` into `self`. `Some` fields in `other` take precedence.
    fn merge_from(&mut self, other: &Self) {
        if other.endpoint.is_some() {
            self.endpoint.clone_from(&other.endpoint);
        }
        if other.model.is_some() {
            self.model.clone_from(&other.model);
        }
        if other.max_context_tokens.is_some() {
            self.max_context_tokens = other.max_context_tokens;
        }
        if other.compression_threshold.is_some() {
            self.compression_threshold = other.compression_threshold;
        }
        if other.max_tool_result_tokens.is_some() {
            self.max_tool_result_tokens = other.max_tool_result_tokens;
        }
        if other.tool_calling.is_some() {
            self.tool_calling = other.tool_calling;
        }
    }

    /// Convert to final `LlmConfig`, filling in defaults for unset fields.
    fn into_final(self) -> LlmConfig {
        LlmConfig {
            endpoint: self.endpoint.unwrap_or_else(default_endpoint),
            model: self.model.unwrap_or_else(default_model),
            max_context_tokens: self
                .max_context_tokens
                .unwrap_or(default_max_context_tokens()),
            compression_threshold: self
                .compression_threshold
                .unwrap_or(default_compression_threshold()),
            max_tool_result_tokens: self
                .max_tool_result_tokens
                .unwrap_or(default_max_tool_result_tokens()),
            tool_calling: self.tool_calling.unwrap_or_default(),
        }
    }
}

/// Partial skills config used for deserialization from TOML.
///
/// Boolean fields use `Option<bool>` so that an explicit `false` in a
/// project-local config can override a global `true`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PartialSkillsConfig {
    #[serde(default)]
    pub discovery_paths: Vec<PathBuf>,
    pub compat_claude: Option<bool>,
    pub compat_agent_skills: Option<bool>,
}

impl PartialSkillsConfig {
    /// Merge `other` into `self`. `Some` fields and non-empty paths in
    /// `other` take precedence.
    fn merge_from(&mut self, other: &Self) {
        if !other.discovery_paths.is_empty() {
            self.discovery_paths.clone_from(&other.discovery_paths);
        }
        if other.compat_claude.is_some() {
            self.compat_claude = other.compat_claude;
        }
        if other.compat_agent_skills.is_some() {
            self.compat_agent_skills = other.compat_agent_skills;
        }
    }

    /// Convert to final `SkillsConfig`, filling in defaults for unset fields.
    fn into_final(self) -> tmg_skills::SkillsConfig {
        tmg_skills::SkillsConfig {
            discovery_paths: self.discovery_paths,
            compat_claude: self.compat_claude.unwrap_or(true),
            compat_agent_skills: self.compat_agent_skills.unwrap_or(true),
        }
    }
}

/// Partial harness config used for deserialization from TOML.
///
/// All fields are `Option` so we can distinguish "not set" from
/// "explicitly set to the default value", matching the
/// [`PartialLlmConfig`] convention.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PartialHarnessConfig {
    pub runs_dir: Option<PathBuf>,
    pub auto_resume_on_start: Option<bool>,
}

impl PartialHarnessConfig {
    /// Merge `other` into `self`. `Some` fields in `other` take precedence.
    fn merge_from(&mut self, other: &Self) {
        if other.runs_dir.is_some() {
            self.runs_dir.clone_from(&other.runs_dir);
        }
        if other.auto_resume_on_start.is_some() {
            self.auto_resume_on_start = other.auto_resume_on_start;
        }
    }

    /// Convert to final `HarnessConfig`, filling in defaults for unset
    /// fields.
    fn into_final(self) -> HarnessConfig {
        HarnessConfig {
            runs_dir: self.runs_dir.unwrap_or_else(default_runs_dir),
            auto_resume_on_start: self
                .auto_resume_on_start
                .unwrap_or(default_auto_resume_on_start()),
        }
    }
}

/// Partial top-level config used for deserialization and merging.
///
/// Deserialized from TOML, merged across sources, then converted to
/// the final [`TsumugiConfig`] via [`into_final`](Self::into_final).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PartialTsumugiConfig {
    #[serde(default)]
    pub llm: PartialLlmConfig,
    #[serde(default)]
    pub sandbox: SandboxConfigSection,
    #[serde(default)]
    pub tui: TuiConfig,
    #[serde(default)]
    pub skills: PartialSkillsConfig,
    #[serde(default)]
    pub harness: PartialHarnessConfig,
}

impl PartialTsumugiConfig {
    /// Merge `other` into `self`. Fields explicitly set in `other` take
    /// precedence.
    fn merge_from(&mut self, other: &Self) {
        self.llm.merge_from(&other.llm);
        self.sandbox.merge_from(&other.sandbox);
        self.tui.merge_from(&other.tui);
        self.skills.merge_from(&other.skills);
        self.harness.merge_from(&other.harness);
    }

    /// Apply environment variable overrides (`TMG_*` prefix).
    ///
    /// Returns an error if a numeric environment variable is present but
    /// cannot be parsed.
    fn apply_env_overrides(
        &mut self,
        env_fn: &dyn Fn(&str) -> Option<String>,
    ) -> Result<(), ConfigError> {
        if let Some(v) = env_fn("TMG_LLM_ENDPOINT") {
            self.llm.endpoint = Some(v);
        }
        if let Some(v) = env_fn("TMG_LLM_MODEL") {
            self.llm.model = Some(v);
        }
        if let Some(v) = env_fn("TMG_LLM_MAX_CONTEXT_TOKENS") {
            let n = v.parse::<usize>().map_err(|_| ConfigError::InvalidValue {
                field: "TMG_LLM_MAX_CONTEXT_TOKENS".to_owned(),
                value: v,
                reason: "must be a non-negative integer".to_owned(),
            })?;
            self.llm.max_context_tokens = Some(n);
        }
        if let Some(v) = env_fn("TMG_LLM_COMPRESSION_THRESHOLD") {
            let n = v.parse::<f64>().map_err(|_| ConfigError::InvalidValue {
                field: "TMG_LLM_COMPRESSION_THRESHOLD".to_owned(),
                value: v,
                reason: "must be a floating-point number".to_owned(),
            })?;
            self.llm.compression_threshold = Some(n);
        }
        if let Some(v) = env_fn("TMG_LLM_MAX_TOOL_RESULT_TOKENS") {
            let n = v.parse::<usize>().map_err(|_| ConfigError::InvalidValue {
                field: "TMG_LLM_MAX_TOOL_RESULT_TOKENS".to_owned(),
                value: v,
                reason: "must be a non-negative integer".to_owned(),
            })?;
            self.llm.max_tool_result_tokens = Some(n);
        }
        if let Some(v) = env_fn("TMG_LLM_TOOL_CALLING") {
            let mode = v
                .parse::<ToolCallingMode>()
                .map_err(|_| ConfigError::InvalidValue {
                    field: "TMG_LLM_TOOL_CALLING".to_owned(),
                    value: v,
                    reason: "must be one of: native, prompt_based, auto".to_owned(),
                })?;
            self.llm.tool_calling = Some(mode);
        }
        if let Some(v) = env_fn("TMG_SANDBOX_MODE") {
            let mode = v
                .parse::<SandboxMode>()
                .map_err(|_| ConfigError::InvalidValue {
                    field: "TMG_SANDBOX_MODE".to_owned(),
                    value: v,
                    reason: "must be one of: read_only, workspace_write, full".to_owned(),
                })?;
            self.sandbox.mode = Some(mode);
        }
        if let Some(v) = env_fn("TMG_SANDBOX_TIMEOUT_SECS") {
            let n = v.parse::<u64>().map_err(|_| ConfigError::InvalidValue {
                field: "TMG_SANDBOX_TIMEOUT_SECS".to_owned(),
                value: v,
                reason: "must be a non-negative integer".to_owned(),
            })?;
            self.sandbox.timeout_secs = Some(n);
        }
        if let Some(v) = env_fn("TMG_HARNESS_RUNS_DIR") {
            self.harness.runs_dir = Some(PathBuf::from(v));
        }
        if let Some(v) = env_fn("TMG_HARNESS_AUTO_RESUME_ON_START") {
            let parsed = parse_bool(&v).map_err(|()| ConfigError::InvalidValue {
                field: "TMG_HARNESS_AUTO_RESUME_ON_START".to_owned(),
                value: v,
                reason: "must be one of: true, false, 1, 0".to_owned(),
            })?;
            self.harness.auto_resume_on_start = Some(parsed);
        }
        Ok(())
    }

    /// Convert to final [`TsumugiConfig`], filling defaults for unset fields.
    fn into_final(self) -> TsumugiConfig {
        TsumugiConfig {
            llm: self.llm.into_final(),
            sandbox: self.sandbox,
            tui: self.tui,
            skills: self.skills.into_final(),
            harness: self.harness.into_final(),
        }
    }
}

/// Parse a `bool` from a string, accepting common truthy/falsy spellings.
fn parse_bool(value: &str) -> Result<bool, ()> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(()),
    }
}

// ---------------------------------------------------------------------------
// Section types (final, validated)
// ---------------------------------------------------------------------------

/// LLM connection settings from `[llm]` section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmConfig {
    /// llama-server endpoint URL.
    pub endpoint: String,

    /// Model name to use in requests.
    pub model: String,

    /// Maximum context window tokens.
    pub max_context_tokens: usize,

    /// Fraction of max context at which compression auto-triggers (0.0..=1.0).
    pub compression_threshold: f64,

    /// Maximum tokens allowed in a single tool result before truncation.
    pub max_tool_result_tokens: usize,

    /// Tool calling mode.
    pub tool_calling: ToolCallingMode,
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
            tool_calling: ToolCallingMode::default(),
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
    /// Sandbox operating mode.
    #[serde(default)]
    pub mode: Option<SandboxMode>,

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

/// Harness settings from `[harness]` section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarnessConfig {
    /// Directory where run state is persisted (relative to cwd).
    pub runs_dir: PathBuf,

    /// Whether to automatically resume the most recent unfinished run on
    /// startup, instead of creating a new ad-hoc run.
    pub auto_resume_on_start: bool,
}

fn default_runs_dir() -> PathBuf {
    PathBuf::from(".tsumugi").join("runs")
}

const fn default_auto_resume_on_start() -> bool {
    true
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            runs_dir: default_runs_dir(),
            auto_resume_on_start: default_auto_resume_on_start(),
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level config (final, validated)
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

    /// Harness settings (run persistence and resume policy).
    #[serde(default)]
    pub harness: HarnessConfig,
}

// ---------------------------------------------------------------------------
// Merging (SandboxConfigSection, TuiConfig -- still needed for partial merges)
// ---------------------------------------------------------------------------

impl SandboxConfigSection {
    /// Merge `other` into `self`. `Some` fields in `other` take precedence.
    fn merge_from(&mut self, other: &Self) {
        if other.mode.is_some() {
            self.mode = other.mode;
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

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

impl TsumugiConfig {
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

        // No need to validate tool_calling or sandbox.mode -- they are
        // already strongly typed enums that reject invalid values at
        // deserialization time.

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
fn load_toml_file(path: &Path) -> Result<Option<PartialTsumugiConfig>, ConfigError> {
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

    let config: PartialTsumugiConfig =
        toml::from_str(&content).map_err(|e| ConfigError::Parse {
            path: path.to_owned(),
            source: e,
        })?;

    Ok(Some(config))
}

/// Resolve the global config path (`~/.config/tsumugi/tsumugi.toml`).
///
/// Returns `None` if the platform config directory cannot be determined
/// (e.g. `$HOME` is not set). This is intentional: the caller falls
/// through to the next source in the priority chain.
fn global_config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("tsumugi").join("tsumugi.toml"))
}

/// Resolve the project-local config path (`.tsumugi/tsumugi.toml`
/// relative to the current directory).
///
/// Returns `None` if the current working directory cannot be determined
/// (e.g. it has been deleted). This is intentional: the caller falls
/// through to the next source in the priority chain.
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
    let mut config = PartialTsumugiConfig::default();

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
    config.apply_env_overrides(env_fn)?;

    Ok(config.into_final())
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
        let mut global = PartialTsumugiConfig::default();
        global.llm.endpoint = Some("http://global:8080".to_owned());
        global.llm.model = Some("global-model".to_owned());

        let mut project = PartialTsumugiConfig::default();
        project.llm.endpoint = Some("http://project:9090".to_owned());
        // model stays None, should not override global

        global.merge_from(&project);
        let final_config = global.into_final();

        assert_eq!(final_config.llm.endpoint, "http://project:9090");
        assert_eq!(final_config.llm.model, "global-model");
    }

    #[test]
    fn merge_explicitly_set_default_value_still_overrides() {
        // Fix #2: explicitly setting a value equal to the default must still
        // override the other layer's non-default value.
        let mut global = PartialTsumugiConfig::default();
        global.llm.max_context_tokens = Some(16384);

        let mut project = PartialTsumugiConfig::default();
        // Explicitly set back to the default value of 8192.
        project.llm.max_context_tokens = Some(default_max_context_tokens());

        global.merge_from(&project);
        let final_config = global.into_final();

        assert_eq!(
            final_config.llm.max_context_tokens,
            default_max_context_tokens()
        );
    }

    #[test]
    fn skills_compat_override_false_over_true() {
        // Fix #1: project-local `compat_claude = false` must override global `true`.
        let mut global = PartialTsumugiConfig::default();
        global.skills.compat_claude = Some(true);
        global.skills.compat_agent_skills = Some(true);

        let mut project = PartialTsumugiConfig::default();
        project.skills.compat_claude = Some(false);
        // compat_agent_skills not set in project, should remain true.

        global.merge_from(&project);
        let final_config = global.into_final();

        assert!(!final_config.skills.compat_claude);
        assert!(final_config.skills.compat_agent_skills);
    }

    #[test]
    fn skills_compat_override_true_over_false() {
        // Fix #1: project-local `compat_claude = true` must override global `false`.
        let mut global = PartialTsumugiConfig::default();
        global.skills.compat_claude = Some(false);

        let mut project = PartialTsumugiConfig::default();
        project.skills.compat_claude = Some(true);

        global.merge_from(&project);
        let final_config = global.into_final();

        assert!(final_config.skills.compat_claude);
    }

    #[test]
    fn env_overrides_apply() {
        let mut config = PartialTsumugiConfig::default();
        let env_fn = |key: &str| -> Option<String> {
            match key {
                "TMG_LLM_ENDPOINT" => Some("http://env:1234".to_owned()),
                "TMG_LLM_MODEL" => Some("env-model".to_owned()),
                "TMG_LLM_MAX_CONTEXT_TOKENS" => Some("16384".to_owned()),
                "TMG_SANDBOX_MODE" => Some("read_only".to_owned()),
                _ => None,
            }
        };

        config
            .apply_env_overrides(&env_fn)
            .unwrap_or_else(|e| panic!("{e}"));
        let final_config = config.into_final();

        assert_eq!(final_config.llm.endpoint, "http://env:1234");
        assert_eq!(final_config.llm.model, "env-model");
        assert_eq!(final_config.llm.max_context_tokens, 16384);
        assert_eq!(final_config.sandbox.mode, Some(SandboxMode::ReadOnly));
    }

    #[test]
    fn env_override_bad_numeric_returns_error() {
        // Fix #3: unparseable numeric env vars must produce errors.
        let mut config = PartialTsumugiConfig::default();
        let env_fn = |key: &str| -> Option<String> {
            if key == "TMG_LLM_MAX_CONTEXT_TOKENS" {
                Some("not_a_number".to_owned())
            } else {
                None
            }
        };

        let result = config.apply_env_overrides(&env_fn);
        assert!(
            matches!(
                result,
                Err(ConfigError::InvalidValue { ref field, .. })
                    if field == "TMG_LLM_MAX_CONTEXT_TOKENS"
            ),
            "expected InvalidValue error, got {result:?}"
        );
    }

    #[test]
    fn env_override_bad_tool_calling_returns_error() {
        let mut config = PartialTsumugiConfig::default();
        let env_fn = |key: &str| -> Option<String> {
            if key == "TMG_LLM_TOOL_CALLING" {
                Some("invalid_mode".to_owned())
            } else {
                None
            }
        };

        let result = config.apply_env_overrides(&env_fn);
        assert!(
            matches!(
                result,
                Err(ConfigError::InvalidValue { ref field, .. })
                    if field == "TMG_LLM_TOOL_CALLING"
            ),
            "expected InvalidValue error, got {result:?}"
        );
    }

    #[test]
    fn env_override_bad_sandbox_mode_returns_error() {
        let mut config = PartialTsumugiConfig::default();
        let env_fn = |key: &str| -> Option<String> {
            if key == "TMG_SANDBOX_MODE" {
                Some("invalid_mode".to_owned())
            } else {
                None
            }
        };

        let result = config.apply_env_overrides(&env_fn);
        assert!(
            matches!(
                result,
                Err(ConfigError::InvalidValue { ref field, .. })
                    if field == "TMG_SANDBOX_MODE"
            ),
            "expected InvalidValue error, got {result:?}"
        );
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
tool_calling = "native"

[sandbox]
mode = "read_only"
timeout_secs = 60

[tui]
show_token_usage = true

[skills]
discovery_paths = ["/extra"]
compat_claude = false
"#;
        let partial: PartialTsumugiConfig =
            toml::from_str(toml_str).unwrap_or_else(|e| panic!("{e}"));
        let config = partial.into_final();

        assert_eq!(config.llm.endpoint, "http://custom:1234");
        assert_eq!(config.llm.model, "my-model");
        assert_eq!(config.llm.max_context_tokens, 4096);
        assert_eq!(config.llm.tool_calling, ToolCallingMode::Native);
        assert_eq!(config.sandbox.mode, Some(SandboxMode::ReadOnly));
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
        let mut base = PartialTsumugiConfig::default();
        base.sandbox.timeout_secs = Some(30);
        base.sandbox.mode = Some(SandboxMode::WorkspaceWrite);

        let mut overlay = PartialTsumugiConfig::default();
        overlay.sandbox.mode = Some(SandboxMode::ReadOnly);
        // timeout_secs is None, should not override

        base.merge_from(&overlay);
        let final_config = base.into_final();

        assert_eq!(final_config.sandbox.mode, Some(SandboxMode::ReadOnly));
        assert_eq!(final_config.sandbox.timeout_secs, Some(30));
    }

    #[test]
    fn tui_section_merge() {
        let mut base = PartialTsumugiConfig::default();
        base.tui.show_token_usage = Some(false);

        let mut overlay = PartialTsumugiConfig::default();
        overlay.tui.show_token_usage = Some(true);

        base.merge_from(&overlay);
        let final_config = base.into_final();
        assert_eq!(final_config.tui.show_token_usage, Some(true));
    }

    #[test]
    fn valid_tool_calling_modes_accepted() {
        for mode in [
            ToolCallingMode::Native,
            ToolCallingMode::PromptBased,
            ToolCallingMode::Auto,
        ] {
            let mut config = TsumugiConfig::default();
            config.llm.tool_calling = mode;
            assert!(config.validate().is_ok(), "mode {mode} should be valid");
        }
    }

    #[test]
    fn valid_sandbox_modes_accepted() {
        for mode in [
            SandboxMode::ReadOnly,
            SandboxMode::WorkspaceWrite,
            SandboxMode::Full,
        ] {
            let mut config = TsumugiConfig::default();
            config.sandbox.mode = Some(mode);
            assert!(config.validate().is_ok(), "mode {mode} should be valid");
        }
    }

    #[test]
    fn default_tool_calling_is_auto() {
        let config = TsumugiConfig::default();
        assert_eq!(config.llm.tool_calling, ToolCallingMode::Auto);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn harness_defaults() {
        let config = TsumugiConfig::default();
        assert_eq!(config.harness.runs_dir, default_runs_dir());
        assert!(config.harness.auto_resume_on_start);
    }

    #[test]
    fn harness_section_round_trip() {
        let toml_str = r#"
[harness]
runs_dir = "/tmp/tsumugi-runs"
auto_resume_on_start = false
"#;
        let partial: PartialTsumugiConfig =
            toml::from_str(toml_str).unwrap_or_else(|e| panic!("{e}"));
        let config = partial.into_final();
        assert_eq!(config.harness.runs_dir, PathBuf::from("/tmp/tsumugi-runs"));
        assert!(!config.harness.auto_resume_on_start);
    }

    #[test]
    fn harness_section_merge_overrides() {
        let mut base = PartialTsumugiConfig::default();
        base.harness.runs_dir = Some(PathBuf::from(".tsumugi/runs"));
        base.harness.auto_resume_on_start = Some(true);

        let mut overlay = PartialTsumugiConfig::default();
        overlay.harness.auto_resume_on_start = Some(false);
        // runs_dir stays None: should not override.

        base.merge_from(&overlay);
        let final_config = base.into_final();

        assert_eq!(
            final_config.harness.runs_dir,
            PathBuf::from(".tsumugi/runs")
        );
        assert!(!final_config.harness.auto_resume_on_start);
    }

    #[test]
    fn harness_env_overrides_apply() {
        let mut config = PartialTsumugiConfig::default();
        let env_fn = |key: &str| -> Option<String> {
            match key {
                "TMG_HARNESS_RUNS_DIR" => Some("/var/tsumugi".to_owned()),
                "TMG_HARNESS_AUTO_RESUME_ON_START" => Some("false".to_owned()),
                _ => None,
            }
        };
        config
            .apply_env_overrides(&env_fn)
            .unwrap_or_else(|e| panic!("{e}"));
        let final_config = config.into_final();
        assert_eq!(final_config.harness.runs_dir, PathBuf::from("/var/tsumugi"));
        assert!(!final_config.harness.auto_resume_on_start);
    }

    #[test]
    fn harness_env_override_invalid_bool_returns_error() {
        let mut config = PartialTsumugiConfig::default();
        let env_fn = |key: &str| -> Option<String> {
            if key == "TMG_HARNESS_AUTO_RESUME_ON_START" {
                Some("maybe".to_owned())
            } else {
                None
            }
        };
        let result = config.apply_env_overrides(&env_fn);
        assert!(matches!(
            result,
            Err(ConfigError::InvalidValue { ref field, .. })
                if field == "TMG_HARNESS_AUTO_RESUME_ON_START"
        ));
    }
}
