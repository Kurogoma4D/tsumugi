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
///
/// `deny_unknown_fields` is set so a typo in `tsumugi.toml` (for
/// example `smoke_test_evry_n_sessions`) is reported at load time
/// rather than silently ignored. The other partial config sections
/// stay permissive for now so unrelated tools can write extra keys
/// without breaking config loading; the harness section is the most
/// likely place for typos given how recently the knobs were added.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PartialHarnessConfig {
    pub runs_dir: Option<PathBuf>,
    pub auto_resume_on_start: Option<bool>,
    pub bootstrap_max_tokens: Option<usize>,
    pub smoke_test_every_n_sessions: Option<u32>,
    pub default_max_sessions: Option<u32>,
    /// Wall-clock budget for one session in the harnessed runner.
    ///
    /// Stored as a string in TOML (`"30m"`, `"2h30m"`, ...). Parsed via
    /// [`humantime::parse_duration`] in [`Self::into_final`]; an
    /// invalid value produces a [`ConfigError::InvalidValue`] at
    /// resolution time so config errors surface as soon as
    /// [`load_config`] runs rather than only when the harness tries to
    /// schedule a session.
    pub default_session_timeout: Option<String>,
    /// `[harness.escalator]` overrides for the scope-escalation
    /// subagent (SPEC §10.1 / §9.10 / issue #36).
    #[serde(default)]
    pub escalator: PartialEscalatorConfig,
}

/// Partial `[harness.escalator]` config used for deserialization from
/// TOML.
///
/// All three fields are optional; missing values (or empty strings on
/// `endpoint` / `model`) mean "inherit from the main `[llm]` section"
/// when the value is later handed to
/// [`tmg_agents::EscalatorOverrides::from_strings`].
///
/// Mirrors the `deny_unknown_fields` policy of [`PartialHarnessConfig`]
/// so a typo in `[harness.escalator]` is rejected at load time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PartialEscalatorConfig {
    /// Override endpoint URL for escalator spawns. Empty string or
    /// absent means "inherit `[llm] endpoint`".
    pub endpoint: Option<String>,

    /// Override model name for escalator spawns. Empty string or
    /// absent means "inherit `[llm] model`".
    pub model: Option<String>,

    /// When `true`, the harness rejects requests for the escalator
    /// subagent (SPEC §9.10 cost management). The auto-promotion gate
    /// (issue #37) treats a disabled escalator as "do not escalate".
    pub disable: Option<bool>,
}

impl PartialEscalatorConfig {
    /// Merge `other` into `self`. `Some(_)` fields in `other` win,
    /// mirroring the rest of the partial-config merge story. Empty
    /// strings are preserved so the same TOML round-trips cleanly;
    /// the empty-string-as-inherit convention is resolved exactly
    /// once when the partial is converted to the final
    /// [`EscalatorConfig`] by the consumer.
    fn merge_from(&mut self, other: &Self) {
        if other.endpoint.is_some() {
            self.endpoint.clone_from(&other.endpoint);
        }
        if other.model.is_some() {
            self.model.clone_from(&other.model);
        }
        if other.disable.is_some() {
            self.disable = other.disable;
        }
    }

    /// Resolve into the final [`EscalatorConfig`]. No validation is
    /// required here -- empty strings are valid (they simply mean
    /// "inherit") and the disable flag is a plain bool.
    fn into_final(self) -> EscalatorConfig {
        EscalatorConfig {
            endpoint: self.endpoint.unwrap_or_default(),
            model: self.model.unwrap_or_default(),
            disable: self.disable.unwrap_or(false),
        }
    }
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
        if other.bootstrap_max_tokens.is_some() {
            self.bootstrap_max_tokens = other.bootstrap_max_tokens;
        }
        if other.smoke_test_every_n_sessions.is_some() {
            self.smoke_test_every_n_sessions = other.smoke_test_every_n_sessions;
        }
        if other.default_max_sessions.is_some() {
            self.default_max_sessions = other.default_max_sessions;
        }
        if other.default_session_timeout.is_some() {
            self.default_session_timeout
                .clone_from(&other.default_session_timeout);
        }
        self.escalator.merge_from(&other.escalator);
    }

    /// Convert to final `HarnessConfig`, filling in defaults for unset
    /// fields.
    ///
    /// Validates that the three positive-integer / non-zero-duration
    /// knobs are strictly greater than zero. A zero
    /// `smoke_test_every_n_sessions` would mean "smoke-test on every
    /// 0-th session" (undefined / divide-by-zero downstream); a zero
    /// `default_max_sessions` would prevent any session from running;
    /// a zero `default_session_timeout` would race the inner sandbox
    /// timer against the outer `tokio::time::timeout` and always fire
    /// immediately.
    fn into_final(self) -> Result<HarnessConfig, ConfigError> {
        let default_session_timeout = match self.default_session_timeout.as_deref() {
            Some(raw) => parse_humantime_duration("harness.default_session_timeout", raw)?,
            None => default_session_timeout(),
        };

        let smoke_test_every_n_sessions = self
            .smoke_test_every_n_sessions
            .unwrap_or_else(default_smoke_test_every_n_sessions);
        if smoke_test_every_n_sessions == 0 {
            return Err(ConfigError::InvalidValue {
                field: "harness.smoke_test_every_n_sessions".to_owned(),
                value: smoke_test_every_n_sessions.to_string(),
                reason: "must be a positive integer (>= 1)".to_owned(),
            });
        }

        let default_max_sessions = self
            .default_max_sessions
            .unwrap_or_else(default_default_max_sessions);
        if default_max_sessions == 0 {
            return Err(ConfigError::InvalidValue {
                field: "harness.default_max_sessions".to_owned(),
                value: default_max_sessions.to_string(),
                reason: "must be a positive integer (>= 1)".to_owned(),
            });
        }

        if default_session_timeout.is_zero() {
            return Err(ConfigError::InvalidValue {
                field: "harness.default_session_timeout".to_owned(),
                value: humantime::format_duration(default_session_timeout).to_string(),
                reason: "must be a non-zero duration".to_owned(),
            });
        }

        Ok(HarnessConfig {
            runs_dir: self.runs_dir.unwrap_or_else(default_runs_dir),
            auto_resume_on_start: self
                .auto_resume_on_start
                .unwrap_or(default_auto_resume_on_start()),
            bootstrap_max_tokens: self
                .bootstrap_max_tokens
                .unwrap_or_else(default_bootstrap_max_tokens),
            smoke_test_every_n_sessions,
            default_max_sessions,
            default_session_timeout,
            escalator: self.escalator.into_final(),
        })
    }
}

/// Parse a humantime-style duration string (e.g. `"30m"`, `"2h"`) into
/// a [`std::time::Duration`].
///
/// Surfaced as a free function so env-override and TOML-conversion
/// paths share the same error formatting.
fn parse_humantime_duration(field: &str, value: &str) -> Result<std::time::Duration, ConfigError> {
    humantime::parse_duration(value)
        .map(|d| std::time::Duration::from_secs(d.as_secs()))
        .map_err(|e| ConfigError::InvalidValue {
            field: field.to_owned(),
            value: value.to_owned(),
            reason: e.to_string(),
        })
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
    #[expect(
        clippy::too_many_lines,
        reason = "linear list of env-var overrides; splitting into helpers would obscure the one-knob-per-block layout"
    )]
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
        if let Some(v) = env_fn("TMG_HARNESS_BOOTSTRAP_MAX_TOKENS") {
            let n = v.parse::<usize>().map_err(|_| ConfigError::InvalidValue {
                field: "TMG_HARNESS_BOOTSTRAP_MAX_TOKENS".to_owned(),
                value: v,
                reason: "must be a non-negative integer".to_owned(),
            })?;
            self.harness.bootstrap_max_tokens = Some(n);
        }
        if let Some(v) = env_fn("TMG_HARNESS_SMOKE_TEST_EVERY_N_SESSIONS") {
            let n = v.parse::<u32>().map_err(|_| ConfigError::InvalidValue {
                field: "TMG_HARNESS_SMOKE_TEST_EVERY_N_SESSIONS".to_owned(),
                value: v.clone(),
                reason: "must be a non-negative integer".to_owned(),
            })?;
            if n == 0 {
                return Err(ConfigError::InvalidValue {
                    field: "TMG_HARNESS_SMOKE_TEST_EVERY_N_SESSIONS".to_owned(),
                    value: v,
                    reason: "must be a positive integer (>= 1)".to_owned(),
                });
            }
            self.harness.smoke_test_every_n_sessions = Some(n);
        }
        if let Some(v) = env_fn("TMG_HARNESS_DEFAULT_MAX_SESSIONS") {
            let n = v.parse::<u32>().map_err(|_| ConfigError::InvalidValue {
                field: "TMG_HARNESS_DEFAULT_MAX_SESSIONS".to_owned(),
                value: v.clone(),
                reason: "must be a non-negative integer".to_owned(),
            })?;
            if n == 0 {
                return Err(ConfigError::InvalidValue {
                    field: "TMG_HARNESS_DEFAULT_MAX_SESSIONS".to_owned(),
                    value: v,
                    reason: "must be a positive integer (>= 1)".to_owned(),
                });
            }
            self.harness.default_max_sessions = Some(n);
        }
        if let Some(v) = env_fn("TMG_HARNESS_DEFAULT_SESSION_TIMEOUT") {
            // Validate now so a bad string is reported with a precise
            // env-var name; the parsed value is re-derived in
            // `into_final` so the storage shape stays a `String`.
            let parsed = parse_humantime_duration("TMG_HARNESS_DEFAULT_SESSION_TIMEOUT", &v)?;
            if parsed.is_zero() {
                return Err(ConfigError::InvalidValue {
                    field: "TMG_HARNESS_DEFAULT_SESSION_TIMEOUT".to_owned(),
                    value: v,
                    reason: "must be a non-zero duration".to_owned(),
                });
            }
            self.harness.default_session_timeout = Some(v);
        }
        if let Some(v) = env_fn("TMG_HARNESS_ESCALATOR_ENDPOINT") {
            self.harness.escalator.endpoint = Some(v);
        }
        if let Some(v) = env_fn("TMG_HARNESS_ESCALATOR_MODEL") {
            self.harness.escalator.model = Some(v);
        }
        if let Some(v) = env_fn("TMG_HARNESS_ESCALATOR_DISABLE") {
            let parsed = parse_bool(&v).map_err(|()| ConfigError::InvalidValue {
                field: "TMG_HARNESS_ESCALATOR_DISABLE".to_owned(),
                value: v,
                reason: "must be one of: true, false, 1, 0".to_owned(),
            })?;
            self.harness.escalator.disable = Some(parsed);
        }
        Ok(())
    }

    /// Convert to final [`TsumugiConfig`], filling defaults for unset fields.
    fn into_final(self) -> Result<TsumugiConfig, ConfigError> {
        Ok(TsumugiConfig {
            llm: self.llm.into_final(),
            sandbox: self.sandbox,
            tui: self.tui,
            skills: self.skills.into_final(),
            harness: self.harness.into_final()?,
        })
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

/// `[harness.escalator]` settings (SPEC §10.1 / §9.10 / issue #36).
///
/// Empty `endpoint` / `model` mean "inherit the main `[llm]`
/// endpoint / model" when handed to
/// [`tmg_agents::EscalatorOverrides::from_strings`]. The struct itself
/// keeps the raw strings so a TOML round-trip preserves the operator's
/// exact configuration.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct EscalatorConfig {
    /// Override endpoint for escalator spawns. Empty string means
    /// "inherit `[llm] endpoint`".
    #[serde(default)]
    pub endpoint: String,

    /// Override model for escalator spawns. Empty string means
    /// "inherit `[llm] model`".
    #[serde(default)]
    pub model: String,

    /// When `true`, escalator spawns are rejected with
    /// `AgentError::EscalatorDisabled`.
    #[serde(default)]
    pub disable: bool,
}

/// Harness settings from `[harness]` section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessConfig {
    /// Directory where run state is persisted (relative to cwd).
    pub runs_dir: PathBuf,

    /// Whether to automatically resume the most recent unfinished run on
    /// startup, instead of creating a new ad-hoc run.
    pub auto_resume_on_start: bool,

    /// Maximum tokens for the `session_bootstrap` payload before the
    /// tool truncates older progress sessions / git log entries to fit.
    ///
    /// Set to `0` to disable truncation entirely (the payload is
    /// emitted in full regardless of size). Any value `>= 1` is a hard
    /// budget; older `progress.md` sessions and tail-end git log lines
    /// are shed until the serialized payload fits.
    pub bootstrap_max_tokens: usize,

    /// Frequency at which `session_bootstrap` invokes the tester
    /// subagent for a smoke-test pass.
    ///
    /// Interpretation: run the tester every Nth session (1 = every
    /// session). The tester subagent itself is not yet implemented;
    /// this knob is wired ahead of time so CLI configuration is stable
    /// when the tester lands.
    pub smoke_test_every_n_sessions: u32,

    /// Default cap on the number of sessions in a harnessed run that
    /// does not specify its own `max_sessions`.
    pub default_max_sessions: u32,

    /// Default wall-clock budget for one session in a harnessed run.
    ///
    /// Stored as a [`std::time::Duration`] internally; written in
    /// `tsumugi.toml` as a humantime string (e.g. `"30m"`, `"1h30m"`).
    #[serde(with = "humantime_serde")]
    pub default_session_timeout: std::time::Duration,

    /// `[harness.escalator]` overrides for the scope-escalation
    /// subagent.
    #[serde(default)]
    pub escalator: EscalatorConfig,
}

fn default_runs_dir() -> PathBuf {
    PathBuf::from(".tsumugi").join("runs")
}

const fn default_auto_resume_on_start() -> bool {
    true
}

fn default_bootstrap_max_tokens() -> usize {
    tmg_harness::DEFAULT_BOOTSTRAP_MAX_TOKENS
}

const fn default_smoke_test_every_n_sessions() -> u32 {
    1
}

const fn default_default_max_sessions() -> u32 {
    50
}

fn default_session_timeout() -> std::time::Duration {
    std::time::Duration::from_secs(30 * 60)
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            runs_dir: default_runs_dir(),
            auto_resume_on_start: default_auto_resume_on_start(),
            bootstrap_max_tokens: default_bootstrap_max_tokens(),
            smoke_test_every_n_sessions: default_smoke_test_every_n_sessions(),
            default_max_sessions: default_default_max_sessions(),
            default_session_timeout: default_session_timeout(),
            escalator: EscalatorConfig::default(),
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

    config.into_final()
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
        let final_config = global.into_final().unwrap_or_else(|e| panic!("{e}"));

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
        let final_config = global.into_final().unwrap_or_else(|e| panic!("{e}"));

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
        let final_config = global.into_final().unwrap_or_else(|e| panic!("{e}"));

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
        let final_config = global.into_final().unwrap_or_else(|e| panic!("{e}"));

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
        let final_config = config.into_final().unwrap_or_else(|e| panic!("{e}"));

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
        let config = partial.into_final().unwrap_or_else(|e| panic!("{e}"));

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
        let final_config = base.into_final().unwrap_or_else(|e| panic!("{e}"));

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
        let final_config = base.into_final().unwrap_or_else(|e| panic!("{e}"));
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
        assert_eq!(
            config.harness.bootstrap_max_tokens,
            tmg_harness::DEFAULT_BOOTSTRAP_MAX_TOKENS
        );
        assert_eq!(config.harness.smoke_test_every_n_sessions, 1);
        assert_eq!(config.harness.default_max_sessions, 50);
        assert_eq!(
            config.harness.default_session_timeout,
            std::time::Duration::from_secs(30 * 60),
        );
    }

    #[test]
    fn harness_default_session_timeout_round_trips_via_humantime_serde() {
        let toml_str = r#"
[harness]
default_session_timeout = "45m"
"#;
        let partial: PartialTsumugiConfig =
            toml::from_str(toml_str).unwrap_or_else(|e| panic!("{e}"));
        let config = partial.into_final().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            config.harness.default_session_timeout,
            std::time::Duration::from_secs(45 * 60),
        );
    }

    #[test]
    fn harness_default_session_timeout_invalid_string_is_error() {
        let toml_str = r#"
[harness]
default_session_timeout = "not-a-duration"
"#;
        let partial: PartialTsumugiConfig =
            toml::from_str(toml_str).unwrap_or_else(|e| panic!("{e}"));
        let result = partial.into_final();
        assert!(matches!(
            result,
            Err(ConfigError::InvalidValue { ref field, .. })
                if field == "harness.default_session_timeout"
        ));
    }

    #[test]
    fn harness_smoke_test_every_n_sessions_round_trip() {
        let toml_str = r"
[harness]
smoke_test_every_n_sessions = 3
default_max_sessions = 100
";
        let partial: PartialTsumugiConfig =
            toml::from_str(toml_str).unwrap_or_else(|e| panic!("{e}"));
        let config = partial.into_final().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(config.harness.smoke_test_every_n_sessions, 3);
        assert_eq!(config.harness.default_max_sessions, 100);
    }

    #[test]
    fn harness_env_overrides_new_fields_apply() {
        let mut config = PartialTsumugiConfig::default();
        let env_fn = |key: &str| -> Option<String> {
            match key {
                "TMG_HARNESS_SMOKE_TEST_EVERY_N_SESSIONS" => Some("5".to_owned()),
                "TMG_HARNESS_DEFAULT_MAX_SESSIONS" => Some("75".to_owned()),
                "TMG_HARNESS_DEFAULT_SESSION_TIMEOUT" => Some("2h".to_owned()),
                _ => None,
            }
        };
        config
            .apply_env_overrides(&env_fn)
            .unwrap_or_else(|e| panic!("{e}"));
        let final_config = config.into_final().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(final_config.harness.smoke_test_every_n_sessions, 5);
        assert_eq!(final_config.harness.default_max_sessions, 75);
        assert_eq!(
            final_config.harness.default_session_timeout,
            std::time::Duration::from_secs(2 * 60 * 60),
        );
    }

    #[test]
    fn harness_env_override_invalid_session_timeout_returns_error() {
        let mut config = PartialTsumugiConfig::default();
        let env_fn = |key: &str| -> Option<String> {
            if key == "TMG_HARNESS_DEFAULT_SESSION_TIMEOUT" {
                Some("not-humantime".to_owned())
            } else {
                None
            }
        };
        let result = config.apply_env_overrides(&env_fn);
        assert!(matches!(
            result,
            Err(ConfigError::InvalidValue { ref field, .. })
                if field == "TMG_HARNESS_DEFAULT_SESSION_TIMEOUT"
        ));
    }

    #[test]
    fn harness_section_round_trip() {
        let toml_str = r#"
[harness]
runs_dir = "/tmp/tsumugi-runs"
auto_resume_on_start = false
bootstrap_max_tokens = 1024
"#;
        let partial: PartialTsumugiConfig =
            toml::from_str(toml_str).unwrap_or_else(|e| panic!("{e}"));
        let config = partial.into_final().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(config.harness.runs_dir, PathBuf::from("/tmp/tsumugi-runs"));
        assert!(!config.harness.auto_resume_on_start);
        assert_eq!(config.harness.bootstrap_max_tokens, 1024);
    }

    #[test]
    fn harness_section_merge_overrides() {
        let mut base = PartialTsumugiConfig::default();
        base.harness.runs_dir = Some(PathBuf::from(".tsumugi/runs"));
        base.harness.auto_resume_on_start = Some(true);
        base.harness.bootstrap_max_tokens = Some(2048);

        let mut overlay = PartialTsumugiConfig::default();
        overlay.harness.auto_resume_on_start = Some(false);
        overlay.harness.bootstrap_max_tokens = Some(8192);
        // runs_dir stays None: should not override.

        base.merge_from(&overlay);
        let final_config = base.into_final().unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(
            final_config.harness.runs_dir,
            PathBuf::from(".tsumugi/runs")
        );
        assert!(!final_config.harness.auto_resume_on_start);
        assert_eq!(final_config.harness.bootstrap_max_tokens, 8192);
    }

    #[test]
    fn harness_env_overrides_apply() {
        let mut config = PartialTsumugiConfig::default();
        let env_fn = |key: &str| -> Option<String> {
            match key {
                "TMG_HARNESS_RUNS_DIR" => Some("/var/tsumugi".to_owned()),
                "TMG_HARNESS_AUTO_RESUME_ON_START" => Some("false".to_owned()),
                "TMG_HARNESS_BOOTSTRAP_MAX_TOKENS" => Some("12345".to_owned()),
                _ => None,
            }
        };
        config
            .apply_env_overrides(&env_fn)
            .unwrap_or_else(|e| panic!("{e}"));
        let final_config = config.into_final().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(final_config.harness.runs_dir, PathBuf::from("/var/tsumugi"));
        assert!(!final_config.harness.auto_resume_on_start);
        assert_eq!(final_config.harness.bootstrap_max_tokens, 12345);
    }

    #[test]
    fn harness_env_override_invalid_bootstrap_max_tokens_returns_error() {
        let mut config = PartialTsumugiConfig::default();
        let env_fn = |key: &str| -> Option<String> {
            if key == "TMG_HARNESS_BOOTSTRAP_MAX_TOKENS" {
                Some("not-a-number".to_owned())
            } else {
                None
            }
        };
        let result = config.apply_env_overrides(&env_fn);
        assert!(matches!(
            result,
            Err(ConfigError::InvalidValue { ref field, .. })
                if field == "TMG_HARNESS_BOOTSTRAP_MAX_TOKENS"
        ));
    }

    #[test]
    fn harness_zero_smoke_test_every_n_sessions_rejected() {
        let toml_str = r"
[harness]
smoke_test_every_n_sessions = 0
";
        let partial: PartialTsumugiConfig =
            toml::from_str(toml_str).unwrap_or_else(|e| panic!("{e}"));
        let result = partial.into_final();
        assert!(
            matches!(
                result,
                Err(ConfigError::InvalidValue { ref field, .. })
                    if field == "harness.smoke_test_every_n_sessions"
            ),
            "expected InvalidValue for smoke_test_every_n_sessions, got {result:?}"
        );
    }

    #[test]
    fn harness_zero_default_max_sessions_rejected() {
        let toml_str = r"
[harness]
default_max_sessions = 0
";
        let partial: PartialTsumugiConfig =
            toml::from_str(toml_str).unwrap_or_else(|e| panic!("{e}"));
        let result = partial.into_final();
        assert!(
            matches!(
                result,
                Err(ConfigError::InvalidValue { ref field, .. })
                    if field == "harness.default_max_sessions"
            ),
            "expected InvalidValue for default_max_sessions, got {result:?}"
        );
    }

    #[test]
    fn harness_zero_default_session_timeout_rejected() {
        let toml_str = r#"
[harness]
default_session_timeout = "0s"
"#;
        let partial: PartialTsumugiConfig =
            toml::from_str(toml_str).unwrap_or_else(|e| panic!("{e}"));
        let result = partial.into_final();
        assert!(
            matches!(
                result,
                Err(ConfigError::InvalidValue { ref field, .. })
                    if field == "harness.default_session_timeout"
            ),
            "expected InvalidValue for default_session_timeout, got {result:?}"
        );
    }

    #[test]
    fn harness_env_zero_smoke_test_every_n_sessions_rejected() {
        let mut config = PartialTsumugiConfig::default();
        let env_fn = |key: &str| -> Option<String> {
            if key == "TMG_HARNESS_SMOKE_TEST_EVERY_N_SESSIONS" {
                Some("0".to_owned())
            } else {
                None
            }
        };
        let result = config.apply_env_overrides(&env_fn);
        assert!(
            matches!(
                result,
                Err(ConfigError::InvalidValue { ref field, .. })
                    if field == "TMG_HARNESS_SMOKE_TEST_EVERY_N_SESSIONS"
            ),
            "expected InvalidValue, got {result:?}"
        );
    }

    #[test]
    fn harness_env_zero_default_max_sessions_rejected() {
        let mut config = PartialTsumugiConfig::default();
        let env_fn = |key: &str| -> Option<String> {
            if key == "TMG_HARNESS_DEFAULT_MAX_SESSIONS" {
                Some("0".to_owned())
            } else {
                None
            }
        };
        let result = config.apply_env_overrides(&env_fn);
        assert!(
            matches!(
                result,
                Err(ConfigError::InvalidValue { ref field, .. })
                    if field == "TMG_HARNESS_DEFAULT_MAX_SESSIONS"
            ),
            "expected InvalidValue, got {result:?}"
        );
    }

    #[test]
    fn harness_env_zero_default_session_timeout_rejected() {
        let mut config = PartialTsumugiConfig::default();
        let env_fn = |key: &str| -> Option<String> {
            if key == "TMG_HARNESS_DEFAULT_SESSION_TIMEOUT" {
                Some("0s".to_owned())
            } else {
                None
            }
        };
        let result = config.apply_env_overrides(&env_fn);
        assert!(
            matches!(
                result,
                Err(ConfigError::InvalidValue { ref field, .. })
                    if field == "TMG_HARNESS_DEFAULT_SESSION_TIMEOUT"
            ),
            "expected InvalidValue, got {result:?}"
        );
    }

    /// Fix #7: a typo in `[harness]` should fail at load time rather
    /// than be silently ignored.
    #[test]
    fn harness_unknown_field_typo_is_rejected_at_load() {
        let toml_str = r"
[harness]
smoke_test_evry_n_sessions = 3
";
        let result: Result<PartialTsumugiConfig, _> = toml::from_str(toml_str);
        assert!(
            result.is_err(),
            "expected `deny_unknown_fields` to reject the typo, got {result:?}"
        );
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

    // ---- [harness.escalator] (issue #36) ---------------------------

    #[test]
    fn escalator_defaults_inherit_main() {
        let config = TsumugiConfig::default();
        assert_eq!(config.harness.escalator.endpoint, "");
        assert_eq!(config.harness.escalator.model, "");
        assert!(!config.harness.escalator.disable);
    }

    #[test]
    fn escalator_round_trip_from_toml() {
        let toml_str = r#"
[harness.escalator]
endpoint = "http://escalator.invalid"
model = "tiny"
disable = true
"#;
        let partial: PartialTsumugiConfig =
            toml::from_str(toml_str).unwrap_or_else(|e| panic!("{e}"));
        let final_config = partial.into_final().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            final_config.harness.escalator.endpoint,
            "http://escalator.invalid",
        );
        assert_eq!(final_config.harness.escalator.model, "tiny");
        assert!(final_config.harness.escalator.disable);
    }

    /// `endpoint = ""` is the documented "inherit main endpoint"
    /// sentinel; loading must succeed and surface the empty string so
    /// the spawn-side resolver can decide what to do.
    #[test]
    fn escalator_empty_endpoint_round_trips() {
        let toml_str = r#"
[harness.escalator]
endpoint = ""
model = ""
disable = false
"#;
        let partial: PartialTsumugiConfig =
            toml::from_str(toml_str).unwrap_or_else(|e| panic!("{e}"));
        let final_config = partial.into_final().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(final_config.harness.escalator.endpoint, "");
        assert_eq!(final_config.harness.escalator.model, "");
        assert!(!final_config.harness.escalator.disable);
    }

    #[test]
    fn escalator_partial_overrides() {
        let toml_str = r#"
[harness.escalator]
endpoint = "http://escalator.invalid"
"#;
        let partial: PartialTsumugiConfig =
            toml::from_str(toml_str).unwrap_or_else(|e| panic!("{e}"));
        let final_config = partial.into_final().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            final_config.harness.escalator.endpoint,
            "http://escalator.invalid",
        );
        assert_eq!(
            final_config.harness.escalator.model, "",
            "model not set in TOML must remain empty (inherit main)"
        );
        assert!(!final_config.harness.escalator.disable);
    }

    #[test]
    fn escalator_unknown_field_rejected_at_load() {
        // deny_unknown_fields parity with the rest of harness config.
        let toml_str = r#"
[harness.escalator]
endpoint = "http://x"
disabled = true
"#;
        let result: Result<PartialTsumugiConfig, _> = toml::from_str(toml_str);
        assert!(
            result.is_err(),
            "expected deny_unknown_fields to reject `disabled` (typo for `disable`), got {result:?}"
        );
    }

    #[test]
    fn escalator_env_overrides_apply() {
        let mut config = PartialTsumugiConfig::default();
        let env_fn = |key: &str| -> Option<String> {
            match key {
                "TMG_HARNESS_ESCALATOR_ENDPOINT" => Some("http://env.escalator".to_owned()),
                "TMG_HARNESS_ESCALATOR_MODEL" => Some("env-tiny".to_owned()),
                "TMG_HARNESS_ESCALATOR_DISABLE" => Some("true".to_owned()),
                _ => None,
            }
        };
        config
            .apply_env_overrides(&env_fn)
            .unwrap_or_else(|e| panic!("{e}"));
        let final_config = config.into_final().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            final_config.harness.escalator.endpoint,
            "http://env.escalator"
        );
        assert_eq!(final_config.harness.escalator.model, "env-tiny");
        assert!(final_config.harness.escalator.disable);
    }

    #[test]
    fn escalator_env_disable_invalid_returns_error() {
        let mut config = PartialTsumugiConfig::default();
        let env_fn = |key: &str| -> Option<String> {
            if key == "TMG_HARNESS_ESCALATOR_DISABLE" {
                Some("maybe".to_owned())
            } else {
                None
            }
        };
        let result = config.apply_env_overrides(&env_fn);
        assert!(matches!(
            result,
            Err(ConfigError::InvalidValue { ref field, .. })
                if field == "TMG_HARNESS_ESCALATOR_DISABLE"
        ));
    }

    #[test]
    fn escalator_env_overrides_win_over_toml() {
        let toml_str = r#"
[harness.escalator]
endpoint = "http://from-toml"
model = "toml-model"
disable = false
"#;
        let mut partial: PartialTsumugiConfig =
            toml::from_str(toml_str).unwrap_or_else(|e| panic!("{e}"));
        let env_fn = |key: &str| -> Option<String> {
            match key {
                "TMG_HARNESS_ESCALATOR_ENDPOINT" => Some("http://from-env".to_owned()),
                "TMG_HARNESS_ESCALATOR_DISABLE" => Some("true".to_owned()),
                _ => None,
            }
        };
        partial
            .apply_env_overrides(&env_fn)
            .unwrap_or_else(|e| panic!("{e}"));
        let final_config = partial.into_final().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(final_config.harness.escalator.endpoint, "http://from-env");
        // model not env-overridden -> keeps the TOML value.
        assert_eq!(final_config.harness.escalator.model, "toml-model");
        assert!(final_config.harness.escalator.disable);
    }

    #[test]
    fn escalator_partial_merge_overrides() {
        let mut base = PartialTsumugiConfig::default();
        base.harness.escalator.endpoint = Some("http://global".to_owned());
        base.harness.escalator.model = Some("global-model".to_owned());

        let mut overlay = PartialTsumugiConfig::default();
        overlay.harness.escalator.endpoint = Some("http://project".to_owned());
        overlay.harness.escalator.disable = Some(true);

        base.merge_from(&overlay);
        let final_config = base.into_final().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(final_config.harness.escalator.endpoint, "http://project");
        // overlay did not set `model`; global wins.
        assert_eq!(final_config.harness.escalator.model, "global-model");
        assert!(final_config.harness.escalator.disable);
    }
}
