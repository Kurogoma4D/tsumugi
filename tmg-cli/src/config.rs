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
use std::time::Duration;

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
    /// Token threshold above which `file_read` results are rewritten
    /// via tree-sitter signature extraction. Issue #49.
    pub signature_threshold_tokens: Option<usize>,
    pub tool_calling: Option<ToolCallingMode>,
    /// `[llm.subagent_pool]` configuration. SPEC §10.1 / issue #50.
    /// When present, the pool's endpoints are used to load-balance
    /// non-escalator subagent spawns. An empty `endpoints` list is
    /// **not** an error: it means "pool disabled".
    #[serde(default)]
    pub subagent_pool: Option<tmg_llm::PoolConfig>,
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
        if other.signature_threshold_tokens.is_some() {
            self.signature_threshold_tokens = other.signature_threshold_tokens;
        }
        if other.tool_calling.is_some() {
            self.tool_calling = other.tool_calling;
        }
        if other.subagent_pool.is_some() {
            self.subagent_pool.clone_from(&other.subagent_pool);
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
            signature_threshold_tokens: self
                .signature_threshold_tokens
                .unwrap_or(default_signature_threshold_tokens()),
            tool_calling: self.tool_calling.unwrap_or_default(),
            subagent_pool: self.subagent_pool,
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

/// Partial workflow config used for deserialization from TOML.
///
/// Mirrors [`tmg_workflow::WorkflowConfig`] but stores the timeout as
/// an optional string so partial layers don't bake in a default they
/// can never opt out of. Unknown fields are rejected to surface typos
/// in `tsumugi.toml` early.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PartialWorkflowConfig {
    #[serde(default)]
    pub discovery_paths: Vec<PathBuf>,
    pub default_shell_timeout: Option<u64>,
    pub default_agent_model: Option<String>,
    pub max_parallel_agents: Option<u32>,
}

impl PartialWorkflowConfig {
    fn merge_from(&mut self, other: &Self) {
        if !other.discovery_paths.is_empty() {
            self.discovery_paths.clone_from(&other.discovery_paths);
        }
        if other.default_shell_timeout.is_some() {
            self.default_shell_timeout = other.default_shell_timeout;
        }
        if other.default_agent_model.is_some() {
            self.default_agent_model
                .clone_from(&other.default_agent_model);
        }
        if other.max_parallel_agents.is_some() {
            self.max_parallel_agents = other.max_parallel_agents;
        }
    }

    fn into_final(self) -> Result<tmg_workflow::WorkflowConfig, ConfigError> {
        // Construct, then delegate range checks to the canonical
        // `WorkflowConfig::validate()` so the rules are defined in a
        // single place. Any validation failure is mapped to a
        // structured `ConfigError::InvalidValue` for the CLI's error
        // reporting.
        let timeout_secs = self.default_shell_timeout.unwrap_or(30);
        let max_parallel = self.max_parallel_agents.unwrap_or(2);
        let cfg = tmg_workflow::WorkflowConfig {
            discovery_paths: self.discovery_paths,
            default_shell_timeout: Duration::from_secs(timeout_secs),
            default_agent_model: self.default_agent_model.unwrap_or_default(),
            max_parallel_agents: max_parallel,
        };
        cfg.validate().map_err(|e| {
            // The validate() error pinpoints the offending knob in
            // its message; surface that wrapped as InvalidValue so
            // the CLI reports it like other config issues.
            let (field, value) = if e.to_string().contains("default_shell_timeout") {
                (
                    "workflow.default_shell_timeout".to_owned(),
                    timeout_secs.to_string(),
                )
            } else {
                (
                    "workflow.max_parallel_agents".to_owned(),
                    max_parallel.to_string(),
                )
            };
            ConfigError::InvalidValue {
                field,
                value,
                reason: "must be a positive integer (>= 1)".to_owned(),
            }
        })?;
        Ok(cfg)
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
    /// Maximum number of recent live `session_NNN.json` files
    /// preserved before
    /// [`SessionLog::compress_old_sessions`](tmg_harness::SessionLog::compress_old_sessions)
    /// aggregates older entries. Must be `>= 1` after merging.
    pub session_log_compress_after: Option<usize>,
    /// Context-usage threshold above which
    /// [`RunRunner::should_force_rotate`](tmg_harness::RunRunner::should_force_rotate)
    /// reports `true` (SPEC §2.3). Must be in `(0.0, 1.0]`.
    pub context_force_rotate_threshold: Option<f64>,
    /// `[harness.escalator]` overrides for the scope-escalation
    /// subagent (SPEC §10.1 / §9.10 / issue #36).
    #[serde(default)]
    pub escalator: PartialEscalatorConfig,
    /// `[harness.escalation]` thresholds for the auto-promotion gate
    /// (SPEC §10.1 / §9.10 / issue #37).
    #[serde(default)]
    pub escalation: PartialEscalationGateConfig,
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

/// Partial `[harness.escalation]` config used for deserialization
/// from TOML.
///
/// Mirrors the strict `deny_unknown_fields` policy of the rest of the
/// harness section so a typo in `tsumugi.toml` (e.g.
/// `repetiton_workflow_threshold`) is rejected at load time. Defaults
/// match the SPEC §10.1 example values.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PartialEscalationGateConfig {
    /// Master switch; when `false`, the auto-promotion gate is
    /// disabled entirely and `[harness.escalation]` thresholds are
    /// ignored.
    pub enabled: Option<bool>,
    /// Cumulative session diff threshold above which the diff-size
    /// signal fires.
    pub diff_size_threshold: Option<u32>,
    /// Context window utilisation threshold (in `0.0..=1.0`) for the
    /// context-pressure signal.
    pub context_pressure_threshold: Option<f32>,
    /// Minimum tool-call count required to qualify the
    /// context-pressure signal.
    pub context_pressure_pending_subtasks: Option<u32>,
    /// Workflow-loop count for the repetition-by-prompt signal.
    pub repetition_workflow_threshold: Option<u32>,
    /// Per-file edit count for the repetition-by-file signal.
    pub repetition_file_edit_threshold: Option<u32>,
}

impl PartialEscalationGateConfig {
    /// Merge `other` into `self`. `Some` fields in `other` take precedence.
    fn merge_from(&mut self, other: &Self) {
        if other.enabled.is_some() {
            self.enabled = other.enabled;
        }
        if other.diff_size_threshold.is_some() {
            self.diff_size_threshold = other.diff_size_threshold;
        }
        if other.context_pressure_threshold.is_some() {
            self.context_pressure_threshold = other.context_pressure_threshold;
        }
        if other.context_pressure_pending_subtasks.is_some() {
            self.context_pressure_pending_subtasks = other.context_pressure_pending_subtasks;
        }
        if other.repetition_workflow_threshold.is_some() {
            self.repetition_workflow_threshold = other.repetition_workflow_threshold;
        }
        if other.repetition_file_edit_threshold.is_some() {
            self.repetition_file_edit_threshold = other.repetition_file_edit_threshold;
        }
    }

    /// Resolve into the final [`tmg_harness::EscalationConfig`].
    ///
    /// Defaults match the SPEC §10.1 example values; validation is
    /// delegated to [`tmg_harness::EscalationConfig::validated`] so
    /// the gate's invariants live in one place (`tmg-harness`) and
    /// the CLI just translates errors into `ConfigError`.
    fn into_final(self) -> Result<tmg_harness::EscalationConfig, ConfigError> {
        let defaults = tmg_harness::EscalationConfig::default();
        let enabled = self.enabled.unwrap_or(defaults.enabled);
        let diff_size_threshold = self
            .diff_size_threshold
            .unwrap_or(defaults.diff_size_threshold);
        let context_pressure_threshold = self
            .context_pressure_threshold
            .unwrap_or(defaults.context_pressure_threshold);
        let context_pressure_pending_subtasks = self
            .context_pressure_pending_subtasks
            .unwrap_or(defaults.context_pressure_pending_subtasks);
        let repetition_workflow_threshold = self
            .repetition_workflow_threshold
            .unwrap_or(defaults.repetition_workflow_threshold);
        let repetition_file_edit_threshold = self
            .repetition_file_edit_threshold
            .unwrap_or(defaults.repetition_file_edit_threshold);

        tmg_harness::EscalationConfig::validated(
            enabled,
            diff_size_threshold,
            context_pressure_threshold,
            context_pressure_pending_subtasks,
            repetition_workflow_threshold,
            repetition_file_edit_threshold,
        )
        .map_err(|e| ConfigError::InvalidValue {
            field: "harness.escalation".to_owned(),
            value: format!(
                "diff={diff_size_threshold} ctx={context_pressure_threshold} \
                 pending={context_pressure_pending_subtasks} \
                 wf={repetition_workflow_threshold} fe={repetition_file_edit_threshold}",
            ),
            reason: e.to_string(),
        })
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
        if other.session_log_compress_after.is_some() {
            self.session_log_compress_after = other.session_log_compress_after;
        }
        if other.context_force_rotate_threshold.is_some() {
            self.context_force_rotate_threshold = other.context_force_rotate_threshold;
        }
        self.escalator.merge_from(&other.escalator);
        self.escalation.merge_from(&other.escalation);
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

        let session_log_compress_after = self
            .session_log_compress_after
            .unwrap_or_else(default_session_log_compress_after);
        if session_log_compress_after == 0 {
            return Err(ConfigError::InvalidValue {
                field: "harness.session_log_compress_after".to_owned(),
                value: session_log_compress_after.to_string(),
                reason: "must be a positive integer (>= 1)".to_owned(),
            });
        }

        let context_force_rotate_threshold = self
            .context_force_rotate_threshold
            .unwrap_or_else(default_context_force_rotate_threshold);
        if !context_force_rotate_threshold.is_finite()
            || context_force_rotate_threshold <= 0.0
            || context_force_rotate_threshold > 1.0
        {
            return Err(ConfigError::InvalidValue {
                field: "harness.context_force_rotate_threshold".to_owned(),
                value: context_force_rotate_threshold.to_string(),
                reason: "must be in the range (0.0, 1.0]".to_owned(),
            });
        }

        let escalation = self.escalation.into_final()?;

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
            session_log_compress_after,
            context_force_rotate_threshold,
            escalator: self.escalator.into_final(),
            escalation,
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
/// Partial memory config used for deserialization / merging. Each
/// field is `Option` so a partial layer can override one knob without
/// resetting the others to their defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PartialMemoryConfig {
    pub enabled: Option<bool>,
    pub project_dir: Option<PathBuf>,
    pub global_dir: Option<PathBuf>,
    pub index_max_lines: Option<usize>,
    pub entry_max_chars: Option<usize>,
    pub total_files_limit: Option<usize>,
}

impl PartialMemoryConfig {
    fn merge_from(&mut self, other: &Self) {
        if other.enabled.is_some() {
            self.enabled = other.enabled;
        }
        if other.project_dir.is_some() {
            self.project_dir.clone_from(&other.project_dir);
        }
        if other.global_dir.is_some() {
            self.global_dir.clone_from(&other.global_dir);
        }
        if other.index_max_lines.is_some() {
            self.index_max_lines = other.index_max_lines;
        }
        if other.entry_max_chars.is_some() {
            self.entry_max_chars = other.entry_max_chars;
        }
        if other.total_files_limit.is_some() {
            self.total_files_limit = other.total_files_limit;
        }
    }

    fn into_final(self) -> MemoryConfig {
        let defaults = MemoryConfig::default();
        MemoryConfig {
            enabled: self.enabled.unwrap_or(defaults.enabled),
            project_dir: self.project_dir.unwrap_or(defaults.project_dir),
            global_dir: self.global_dir.unwrap_or(defaults.global_dir),
            index_max_lines: self.index_max_lines.unwrap_or(defaults.index_max_lines),
            entry_max_chars: self.entry_max_chars.unwrap_or(defaults.entry_max_chars),
            total_files_limit: self.total_files_limit.unwrap_or(defaults.total_files_limit),
        }
    }
}

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
    #[serde(default)]
    pub workflow: PartialWorkflowConfig,
    #[serde(default)]
    pub memory: PartialMemoryConfig,
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
        self.workflow.merge_from(&other.workflow);
        self.memory.merge_from(&other.memory);
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
        if let Some(v) = env_fn("TMG_LLM_SIGNATURE_THRESHOLD_TOKENS") {
            let n = v.parse::<usize>().map_err(|_| ConfigError::InvalidValue {
                field: "TMG_LLM_SIGNATURE_THRESHOLD_TOKENS".to_owned(),
                value: v,
                reason: "must be a non-negative integer".to_owned(),
            })?;
            self.llm.signature_threshold_tokens = Some(n);
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
        if let Some(v) = env_fn("TMG_HARNESS_SESSION_LOG_COMPRESS_AFTER") {
            let n = v.parse::<usize>().map_err(|_| ConfigError::InvalidValue {
                field: "TMG_HARNESS_SESSION_LOG_COMPRESS_AFTER".to_owned(),
                value: v.clone(),
                reason: "must be a non-negative integer".to_owned(),
            })?;
            if n == 0 {
                return Err(ConfigError::InvalidValue {
                    field: "TMG_HARNESS_SESSION_LOG_COMPRESS_AFTER".to_owned(),
                    value: v,
                    reason: "must be a positive integer (>= 1)".to_owned(),
                });
            }
            self.harness.session_log_compress_after = Some(n);
        }
        if let Some(v) = env_fn("TMG_HARNESS_CONTEXT_FORCE_ROTATE_THRESHOLD") {
            let n = v.parse::<f64>().map_err(|_| ConfigError::InvalidValue {
                field: "TMG_HARNESS_CONTEXT_FORCE_ROTATE_THRESHOLD".to_owned(),
                value: v.clone(),
                reason: "must be a floating-point number".to_owned(),
            })?;
            if !n.is_finite() || n <= 0.0 || n > 1.0 {
                return Err(ConfigError::InvalidValue {
                    field: "TMG_HARNESS_CONTEXT_FORCE_ROTATE_THRESHOLD".to_owned(),
                    value: v,
                    reason: "must be in the range (0.0, 1.0]".to_owned(),
                });
            }
            self.harness.context_force_rotate_threshold = Some(n);
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
        // [harness.escalation] env overrides (issue #37). Each knob
        // accepts the same parse rules as its TOML counterpart;
        // validation is delegated to `EscalationConfig::validated`.
        if let Some(v) = env_fn("TMG_HARNESS_ESCALATION_ENABLED") {
            let parsed = parse_bool(&v).map_err(|()| ConfigError::InvalidValue {
                field: "TMG_HARNESS_ESCALATION_ENABLED".to_owned(),
                value: v,
                reason: "must be one of: true, false, 1, 0".to_owned(),
            })?;
            self.harness.escalation.enabled = Some(parsed);
        }
        if let Some(v) = env_fn("TMG_HARNESS_ESCALATION_DIFF_SIZE_THRESHOLD") {
            let n = v.parse::<u32>().map_err(|_| ConfigError::InvalidValue {
                field: "TMG_HARNESS_ESCALATION_DIFF_SIZE_THRESHOLD".to_owned(),
                value: v,
                reason: "must be a non-negative integer".to_owned(),
            })?;
            self.harness.escalation.diff_size_threshold = Some(n);
        }
        if let Some(v) = env_fn("TMG_HARNESS_ESCALATION_CONTEXT_PRESSURE_THRESHOLD") {
            let n = v.parse::<f32>().map_err(|_| ConfigError::InvalidValue {
                field: "TMG_HARNESS_ESCALATION_CONTEXT_PRESSURE_THRESHOLD".to_owned(),
                value: v,
                reason: "must be a floating-point number".to_owned(),
            })?;
            self.harness.escalation.context_pressure_threshold = Some(n);
        }
        if let Some(v) = env_fn("TMG_HARNESS_ESCALATION_CONTEXT_PRESSURE_PENDING_SUBTASKS") {
            let n = v.parse::<u32>().map_err(|_| ConfigError::InvalidValue {
                field: "TMG_HARNESS_ESCALATION_CONTEXT_PRESSURE_PENDING_SUBTASKS".to_owned(),
                value: v,
                reason: "must be a non-negative integer".to_owned(),
            })?;
            self.harness.escalation.context_pressure_pending_subtasks = Some(n);
        }
        if let Some(v) = env_fn("TMG_HARNESS_ESCALATION_REPETITION_WORKFLOW_THRESHOLD") {
            let n = v.parse::<u32>().map_err(|_| ConfigError::InvalidValue {
                field: "TMG_HARNESS_ESCALATION_REPETITION_WORKFLOW_THRESHOLD".to_owned(),
                value: v,
                reason: "must be a non-negative integer".to_owned(),
            })?;
            self.harness.escalation.repetition_workflow_threshold = Some(n);
        }
        if let Some(v) = env_fn("TMG_HARNESS_ESCALATION_REPETITION_FILE_EDIT_THRESHOLD") {
            let n = v.parse::<u32>().map_err(|_| ConfigError::InvalidValue {
                field: "TMG_HARNESS_ESCALATION_REPETITION_FILE_EDIT_THRESHOLD".to_owned(),
                value: v,
                reason: "must be a non-negative integer".to_owned(),
            })?;
            self.harness.escalation.repetition_file_edit_threshold = Some(n);
        }
        if let Some(v) = env_fn("TMG_WORKFLOW_DEFAULT_SHELL_TIMEOUT") {
            let n = v.parse::<u64>().map_err(|_| ConfigError::InvalidValue {
                field: "TMG_WORKFLOW_DEFAULT_SHELL_TIMEOUT".to_owned(),
                value: v.clone(),
                reason: "must be a non-negative integer".to_owned(),
            })?;
            if n == 0 {
                return Err(ConfigError::InvalidValue {
                    field: "TMG_WORKFLOW_DEFAULT_SHELL_TIMEOUT".to_owned(),
                    value: v,
                    reason: "must be a positive integer (>= 1)".to_owned(),
                });
            }
            self.workflow.default_shell_timeout = Some(n);
        }
        if let Some(v) = env_fn("TMG_WORKFLOW_DEFAULT_AGENT_MODEL") {
            self.workflow.default_agent_model = Some(v);
        }
        if let Some(v) = env_fn("TMG_WORKFLOW_MAX_PARALLEL_AGENTS") {
            let n = v.parse::<u32>().map_err(|_| ConfigError::InvalidValue {
                field: "TMG_WORKFLOW_MAX_PARALLEL_AGENTS".to_owned(),
                value: v.clone(),
                reason: "must be a non-negative integer".to_owned(),
            })?;
            if n == 0 {
                return Err(ConfigError::InvalidValue {
                    field: "TMG_WORKFLOW_MAX_PARALLEL_AGENTS".to_owned(),
                    value: v,
                    reason: "must be a positive integer (>= 1)".to_owned(),
                });
            }
            self.workflow.max_parallel_agents = Some(n);
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
            workflow: self.workflow.into_final()?,
            memory: self.memory.into_final(),
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

    /// Token threshold above which `file_read` results are rewritten
    /// via tree-sitter signature extraction (issue #49). When the
    /// estimated token count of a `file_read` output exceeds this
    /// value and the file extension is supported, the body stored in
    /// the agent loop's history is replaced with a structural summary
    /// instead of being kept verbatim or tail-truncated.
    pub signature_threshold_tokens: usize,

    /// Tool calling mode.
    pub tool_calling: ToolCallingMode,

    /// `[llm.subagent_pool]` configuration. SPEC §10.1 / issue #50.
    /// `None` means "no pool section in the TOML"; `Some` with empty
    /// endpoints means "pool disabled" (validated through
    /// [`tmg_llm::PoolConfig::validate_relaxed`]).
    #[serde(default)]
    pub subagent_pool: Option<tmg_llm::PoolConfig>,
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

const fn default_signature_threshold_tokens() -> usize {
    1500
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            endpoint: default_endpoint(),
            model: default_model(),
            max_context_tokens: default_max_context_tokens(),
            compression_threshold: default_compression_threshold(),
            max_tool_result_tokens: default_max_tool_result_tokens(),
            signature_threshold_tokens: default_signature_threshold_tokens(),
            tool_calling: ToolCallingMode::default(),
            subagent_pool: None,
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

    /// Structured `[sandbox.network]` settings (SPEC §6.2).
    ///
    /// Mirrors [`tmg_sandbox::NetworkConfig`] but lets partial TOML
    /// sections merge independently of the legacy top-level
    /// `allowed_domains`.
    #[serde(default)]
    pub network: Option<NetworkConfigSection>,

    /// Maximum execution time in seconds for shell commands.
    #[serde(default)]
    pub timeout_secs: Option<u64>,

    /// OOM score adjustment for child processes (Linux only).
    #[serde(default)]
    pub oom_score_adj: Option<i32>,
}

/// `[sandbox.network]` settings as exposed in `tsumugi.toml`.
///
/// Mirrors [`tmg_sandbox::NetworkConfig`] but uses `Option` wrappers so
/// that partial overlays (e.g. a user `tsumugi.toml` that only sets
/// `strict = true`) can merge cleanly with the project default.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct NetworkConfigSection {
    /// Domains allowed for outbound network access.
    #[serde(default)]
    pub allowed_domains: Option<Vec<String>>,

    /// When `true`, missing `CAP_NET_ADMIN` causes activation to fail.
    /// When unset or `false`, the agent emits a warning and continues
    /// without network restriction.
    #[serde(default)]
    pub strict: Option<bool>,
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

    /// Maximum number of recent live `session_NNN.json` files preserved
    /// before
    /// [`SessionLog::compress_old_sessions`](tmg_harness::SessionLog::compress_old_sessions)
    /// aggregates older entries into `session_summaries.json`.
    pub session_log_compress_after: usize,

    /// Context-usage threshold above which
    /// [`RunRunner::should_force_rotate`](tmg_harness::RunRunner::should_force_rotate)
    /// returns `true` and the harness rotates to a fresh session
    /// (SPEC §2.3). Must be in `(0.0, 1.0]`.
    pub context_force_rotate_threshold: f64,

    /// `[harness.escalator]` overrides for the scope-escalation
    /// subagent.
    #[serde(default)]
    pub escalator: EscalatorConfig,

    /// `[harness.escalation]` thresholds for the auto-promotion gate
    /// (SPEC §9.10 / §10.1, issue #37). The validated
    /// [`tmg_harness::EscalationConfig`] enforces non-zero thresholds
    /// and `0.0 < context_pressure_threshold <= 1.0`.
    ///
    /// **Round-trip limitation:** this field is marked
    /// `#[serde(default, skip)]` because
    /// [`tmg_harness::EscalationConfig`] does not implement
    /// `Serialize` / `Deserialize` directly — its construction goes
    /// through `PartialEscalationGateConfig::into_final` for
    /// validation. Consequently, **a fully-resolved `HarnessConfig`
    /// cannot be serialized back to TOML without silently dropping
    /// the user's `[harness.escalation]` thresholds.** This is
    /// load-only today; any future feature that needs to round-trip
    /// the resolved config (e.g. `tmg config show` rendering the
    /// effective config) must derive `Serialize` / `Deserialize` on
    /// `tmg_harness::EscalationConfig` and remove this `skip`.
    ///
    /// TODO: add `Serialize` / `Deserialize` to
    /// `tmg_harness::EscalationConfig` once `tmg config show` lands.
    #[serde(default, skip)]
    pub escalation: tmg_harness::EscalationConfig,
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

const fn default_session_log_compress_after() -> usize {
    tmg_harness::DEFAULT_SESSION_LOG_COMPRESS_AFTER
}

const fn default_context_force_rotate_threshold() -> f64 {
    tmg_harness::DEFAULT_CONTEXT_FORCE_ROTATE_THRESHOLD as f64
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
            session_log_compress_after: default_session_log_compress_after(),
            context_force_rotate_threshold: default_context_force_rotate_threshold(),
            escalator: EscalatorConfig::default(),
            escalation: tmg_harness::EscalationConfig::default(),
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

    /// Workflow runtime settings (SPEC §8 / issue #39).
    #[serde(default)]
    pub workflow: tmg_workflow::WorkflowConfig,

    /// Memory layer settings (issue #52).
    #[serde(default)]
    pub memory: MemoryConfig,
}

/// `[memory]` section: project-scoped persistent memory configuration.
///
/// Mirrors [`tmg_memory::MemoryBudget`] for the capacity caps and adds
/// the directory overrides that let operators relocate the project /
/// global stores. An empty `global_dir` falls back to
/// `~/.config/tsumugi/memory/` via [`dirs::config_dir`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Master switch — `false` keeps the tool unregistered and skips
    /// the prompt injection.
    #[serde(default = "default_memory_enabled")]
    pub enabled: bool,
    /// Project memory directory, relative to the project root or
    /// absolute. Defaults to `.tsumugi/memory`.
    #[serde(default = "default_memory_project_dir")]
    pub project_dir: PathBuf,
    /// Global memory directory. Empty (or `""`) means "use
    /// `~/.config/tsumugi/memory`".
    #[serde(default)]
    pub global_dir: PathBuf,
    /// Maximum number of lines in `MEMORY.md`.
    #[serde(default = "default_memory_index_max_lines")]
    pub index_max_lines: usize,
    /// Maximum body character count per entry.
    #[serde(default = "default_memory_entry_max_chars")]
    pub entry_max_chars: usize,
    /// Maximum number of memory files (excluding the index).
    #[serde(default = "default_memory_total_files_limit")]
    pub total_files_limit: usize,
}

const fn default_memory_enabled() -> bool {
    true
}

fn default_memory_project_dir() -> PathBuf {
    PathBuf::from(".tsumugi/memory")
}

const fn default_memory_index_max_lines() -> usize {
    200
}

const fn default_memory_entry_max_chars() -> usize {
    600
}

const fn default_memory_total_files_limit() -> usize {
    50
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: default_memory_enabled(),
            project_dir: default_memory_project_dir(),
            global_dir: PathBuf::new(),
            index_max_lines: default_memory_index_max_lines(),
            entry_max_chars: default_memory_entry_max_chars(),
            total_files_limit: default_memory_total_files_limit(),
        }
    }
}

impl MemoryConfig {
    /// Resolve `project_dir` against `project_root` (relative paths
    /// only) and return the absolute path.
    #[must_use]
    pub fn resolve_project_dir(&self, project_root: &Path) -> PathBuf {
        if self.project_dir.is_absolute() {
            self.project_dir.clone()
        } else {
            project_root.join(&self.project_dir)
        }
    }

    /// Resolve `global_dir` to a concrete path. An empty path falls
    /// back to `~/.config/tsumugi/memory`. Returns `None` when neither
    /// the override nor the platform config dir is available.
    #[must_use]
    pub fn resolve_global_dir(&self) -> Option<PathBuf> {
        if self.global_dir.as_os_str().is_empty() {
            dirs::config_dir().map(|d| d.join("tsumugi").join("memory"))
        } else {
            Some(self.global_dir.clone())
        }
    }

    /// Build the [`tmg_memory::MemoryBudget`] from the configured caps.
    #[must_use]
    pub fn to_budget(&self) -> tmg_memory::MemoryBudget {
        tmg_memory::MemoryBudget {
            index_max_lines: self.index_max_lines,
            entry_max_chars: self.entry_max_chars,
            total_files_limit: self.total_files_limit,
        }
    }
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
        if let Some(other_net) = &other.network {
            let net = self
                .network
                .get_or_insert_with(NetworkConfigSection::default);
            net.merge_from(other_net);
        }
        if other.timeout_secs.is_some() {
            self.timeout_secs = other.timeout_secs;
        }
        if other.oom_score_adj.is_some() {
            self.oom_score_adj = other.oom_score_adj;
        }
    }

    /// Build a fully-resolved [`tmg_sandbox::SandboxConfig`] for the
    /// given `workspace`, applying defaults for fields the operator
    /// did not explicitly configure.
    ///
    /// Defaults follow [`tmg_sandbox::SandboxConfig::new`]
    /// (`WorkspaceWrite` mode, no allowed domains, 30 s timeout, 500
    /// OOM score). Operator overrides on this struct take precedence.
    #[must_use]
    pub fn to_sandbox_config(
        &self,
        workspace: impl Into<std::path::PathBuf>,
    ) -> tmg_sandbox::SandboxConfig {
        let mut cfg = tmg_sandbox::SandboxConfig::new(workspace);
        if let Some(mode) = self.mode {
            cfg = cfg.with_mode(mode);
        }
        if let Some(domains) = self.allowed_domains.clone() {
            cfg = cfg.with_allowed_domains(domains);
        }
        if let Some(net) = &self.network {
            let mut nc = tmg_sandbox::NetworkConfig::default();
            if let Some(domains) = net.allowed_domains.clone() {
                nc.allowed_domains = domains;
            }
            if let Some(strict) = net.strict {
                nc.strict = strict;
            }
            cfg = cfg.with_network(nc);
        }
        if let Some(timeout) = self.timeout_secs {
            cfg = cfg.with_timeout(timeout);
        }
        if let Some(oom) = self.oom_score_adj {
            cfg.oom_score_adj = oom;
        }
        cfg
    }
}

impl NetworkConfigSection {
    /// Merge `other` into `self`. `Some` fields in `other` take precedence.
    fn merge_from(&mut self, other: &Self) {
        if other.allowed_domains.is_some() {
            self.allowed_domains.clone_from(&other.allowed_domains);
        }
        if other.strict.is_some() {
            self.strict = other.strict;
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

        // Validate `[llm.subagent_pool]` (SPEC §10.1 / issue #50).
        // Uses the relaxed validator so:
        //   - empty `endpoints = []` is treated as "pool disabled",
        //   - duplicate URLs are deduped with a `tracing::warn!`,
        //   - malformed URLs / empty strings remain hard errors.
        if let Some(pool) = self.llm.subagent_pool.as_ref() {
            match pool.validate_relaxed() {
                Ok(report) => {
                    if !report.duplicates.is_empty() {
                        tracing::warn!(
                            duplicates = ?report.duplicates,
                            "[llm.subagent_pool] endpoints contained duplicate URLs; deduping",
                        );
                    }
                    if report.disabled {
                        tracing::debug!(
                            "[llm.subagent_pool] endpoints empty; subagent pool disabled",
                        );
                    }
                }
                Err(e) => {
                    return Err(ConfigError::InvalidValue {
                        field: "llm.subagent_pool".to_owned(),
                        value: format!("{:?}", pool.endpoints),
                        reason: e.to_string(),
                    });
                }
            }
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

    // -----------------------------------------------------------------
    // [harness.escalation] tests (issue #37)
    // -----------------------------------------------------------------

    #[test]
    fn escalation_section_defaults_to_spec_values() {
        let config = TsumugiConfig::default();
        assert!(config.harness.escalation.enabled);
        assert_eq!(config.harness.escalation.diff_size_threshold, 500);
        assert!((config.harness.escalation.context_pressure_threshold - 0.6).abs() < 1e-6);
        assert_eq!(
            config.harness.escalation.context_pressure_pending_subtasks,
            3
        );
        assert_eq!(config.harness.escalation.repetition_workflow_threshold, 3);
        assert_eq!(config.harness.escalation.repetition_file_edit_threshold, 5);
    }

    #[test]
    fn escalation_section_round_trips_via_toml() {
        let toml = r"
[harness.escalation]
enabled = false
diff_size_threshold = 1000
context_pressure_threshold = 0.8
context_pressure_pending_subtasks = 5
repetition_workflow_threshold = 4
repetition_file_edit_threshold = 7
";
        let partial: PartialTsumugiConfig = toml::from_str(toml).unwrap_or_else(|e| panic!("{e}"));
        let config = partial.into_final().unwrap_or_else(|e| panic!("{e}"));
        assert!(!config.harness.escalation.enabled);
        assert_eq!(config.harness.escalation.diff_size_threshold, 1000);
        assert!((config.harness.escalation.context_pressure_threshold - 0.8).abs() < 1e-6);
        assert_eq!(
            config.harness.escalation.context_pressure_pending_subtasks,
            5
        );
        assert_eq!(config.harness.escalation.repetition_workflow_threshold, 4);
        assert_eq!(config.harness.escalation.repetition_file_edit_threshold, 7);
    }

    #[test]
    fn escalation_unknown_field_is_rejected() {
        // Acceptance: typos in `[harness.escalation]` are caught at
        // load time thanks to `deny_unknown_fields`.
        let toml = r"
[harness.escalation]
enabled = true
diff_sze_threshold = 500
";
        let result: Result<PartialTsumugiConfig, toml::de::Error> = toml::from_str(toml);
        assert!(result.is_err(), "typo must be rejected");
        let msg = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(
            msg.contains("diff_sze_threshold") || msg.contains("unknown field"),
            "error must identify the typo, got: {msg}",
        );
    }

    #[test]
    fn escalation_zero_threshold_is_rejected() {
        let toml = r"
[harness.escalation]
diff_size_threshold = 0
";
        let partial: PartialTsumugiConfig = toml::from_str(toml).unwrap_or_else(|e| panic!("{e}"));
        let Err(err) = partial.into_final() else {
            panic!("zero threshold must be rejected");
        };
        assert!(matches!(err, ConfigError::InvalidValue { .. }), "{err:?}");
    }

    #[test]
    fn escalation_invalid_context_threshold_is_rejected() {
        let toml = r"
[harness.escalation]
context_pressure_threshold = 1.5
";
        let partial: PartialTsumugiConfig = toml::from_str(toml).unwrap_or_else(|e| panic!("{e}"));
        let Err(err) = partial.into_final() else {
            panic!("out-of-range context threshold must be rejected");
        };
        assert!(matches!(err, ConfigError::InvalidValue { .. }), "{err:?}");
    }

    #[test]
    fn escalation_env_overrides_apply() {
        let toml = r"
[harness.escalation]
enabled = true
diff_size_threshold = 500
";
        let mut partial: PartialTsumugiConfig =
            toml::from_str(toml).unwrap_or_else(|e| panic!("{e}"));

        let env_fn = |key: &str| -> Option<String> {
            match key {
                "TMG_HARNESS_ESCALATION_ENABLED" => Some("false".to_owned()),
                "TMG_HARNESS_ESCALATION_DIFF_SIZE_THRESHOLD" => Some("750".to_owned()),
                _ => None,
            }
        };
        partial
            .apply_env_overrides(&env_fn)
            .unwrap_or_else(|e| panic!("{e}"));

        let final_config = partial.into_final().unwrap_or_else(|e| panic!("{e}"));
        assert!(!final_config.harness.escalation.enabled);
        assert_eq!(final_config.harness.escalation.diff_size_threshold, 750);
    }

    /// Issue #38: `[harness] session_log_compress_after` and
    /// `[harness] context_force_rotate_threshold` parse from TOML and
    /// flow through to the validated `HarnessConfig`.
    #[test]
    fn force_rotate_and_compress_after_load_from_toml() {
        let toml = r"
[harness]
session_log_compress_after = 25
context_force_rotate_threshold = 0.9
";
        let partial: PartialTsumugiConfig = toml::from_str(toml).unwrap_or_else(|e| panic!("{e}"));
        let final_config = partial.into_final().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(final_config.harness.session_log_compress_after, 25);
        assert!((final_config.harness.context_force_rotate_threshold - 0.9).abs() < 1e-6);
    }

    /// `session_log_compress_after = 0` is rejected.
    #[test]
    fn session_log_compress_after_zero_rejected() {
        let toml = r"
[harness]
session_log_compress_after = 0
";
        let partial: PartialTsumugiConfig = toml::from_str(toml).unwrap_or_else(|e| panic!("{e}"));
        let Err(err) = partial.into_final() else {
            panic!("zero session_log_compress_after must be rejected");
        };
        assert!(
            matches!(
                err,
                ConfigError::InvalidValue { ref field, .. }
                    if field == "harness.session_log_compress_after"
            ),
            "got {err:?}",
        );
    }

    /// `context_force_rotate_threshold` outside `(0.0, 1.0]` is
    /// rejected.
    #[test]
    fn context_force_rotate_threshold_out_of_range_rejected() {
        for bad in ["0.0", "1.5", "-0.1"] {
            let toml = format!(
                r"
[harness]
context_force_rotate_threshold = {bad}
"
            );
            let partial: PartialTsumugiConfig =
                toml::from_str(&toml).unwrap_or_else(|e| panic!("{e}"));
            let Err(err) = partial.into_final() else {
                panic!("out-of-range threshold {bad} must be rejected");
            };
            assert!(
                matches!(
                    err,
                    ConfigError::InvalidValue { ref field, .. }
                        if field == "harness.context_force_rotate_threshold"
                ),
                "got {err:?} for {bad}",
            );
        }
    }

    /// Env-var overrides for the new knobs validate their inputs.
    #[test]
    fn force_rotate_env_overrides_apply() {
        let mut partial = PartialTsumugiConfig::default();
        let env_fn = |key: &str| -> Option<String> {
            match key {
                "TMG_HARNESS_SESSION_LOG_COMPRESS_AFTER" => Some("33".to_owned()),
                "TMG_HARNESS_CONTEXT_FORCE_ROTATE_THRESHOLD" => Some("0.85".to_owned()),
                _ => None,
            }
        };
        partial
            .apply_env_overrides(&env_fn)
            .unwrap_or_else(|e| panic!("{e}"));
        let final_config = partial.into_final().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(final_config.harness.session_log_compress_after, 33);
        assert!((final_config.harness.context_force_rotate_threshold - 0.85).abs() < 1e-6);
    }

    #[test]
    fn force_rotate_env_override_bad_value_rejected() {
        let mut partial = PartialTsumugiConfig::default();
        let env_fn = |key: &str| -> Option<String> {
            if key == "TMG_HARNESS_CONTEXT_FORCE_ROTATE_THRESHOLD" {
                Some("2.0".to_owned())
            } else {
                None
            }
        };
        let result = partial.apply_env_overrides(&env_fn);
        assert!(
            matches!(
                result,
                Err(ConfigError::InvalidValue { ref field, .. })
                    if field == "TMG_HARNESS_CONTEXT_FORCE_ROTATE_THRESHOLD"
            ),
            "got {result:?}",
        );
    }
}
