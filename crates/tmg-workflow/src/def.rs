//! Workflow type definitions (SPEC §8.10).
//!
//! These are the canonical, validated workflow representations consumed
//! by the engine. Raw YAML is parsed into these via [`crate::parse`].
//!
//! Control-flow steps (`Loop`, `Branch`, `Parallel`, `Group`, `Human`)
//! are explicitly out of scope for this iteration; see issue #40.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// The execution mode of a workflow.
///
/// `LongRunning` is a forward-looking placeholder for the long-running
/// session model in issue #42 — the engine in this crate currently
/// treats both modes identically, but parsing it here lets workflow YAML
/// stabilise its surface ahead of the executor work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowMode {
    /// Synchronous, run-to-completion workflow.
    #[default]
    Normal,
    /// Long-running, resumable workflow (placeholder for #42).
    LongRunning,
}

/// Definition of a single workflow input.
#[derive(Debug, Clone, PartialEq)]
pub struct InputDef {
    /// Declared type label (free-form string; SPEC keeps this loose so
    /// custom workflows can introduce semantic types like `"path"` or
    /// `"json"`). The engine performs no coercion based on this field.
    pub r#type: String,
    /// Optional default value used when the caller does not supply one.
    pub default: Option<serde_json::Value>,
    /// Whether the input must be supplied; when `false` a missing input
    /// without a default resolves to `null`.
    pub required: bool,
    /// Free-form human-readable description.
    pub description: Option<String>,
}

/// A single workflow step.
///
/// Only `Agent`, `Shell`, and `WriteFile` are supported in this iteration.
#[derive(Debug, Clone, PartialEq)]
pub enum StepDef {
    /// Spawn a fresh subagent conversation.
    Agent {
        /// Step identifier (`snake_case`).
        id: String,
        /// Subagent kind name (built-in or custom).
        subagent: String,
        /// Prompt template (`${{ ... }}` substitutions resolved at run time).
        prompt: String,
        /// Optional per-spawn model override.
        ///
        /// Currently advisory: the underlying [`tmg_agents::SubagentManager`]
        /// resolves endpoint/model from the manager defaults plus per-agent
        /// overrides on `CustomAgentDef`, with no per-spawn knob. We accept
        /// the field at parse time so future support is non-breaking.
        model: Option<String>,
        /// Optional `${{ ... }}` boolean expression; step is skipped when
        /// the expression evaluates to `false`.
        when: Option<String>,
        /// Files to read and inject into the agent prompt as a context
        /// block, preserving relative paths.
        inject_files: Vec<String>,
    },

    /// Run a shell command inside the sandbox.
    Shell {
        /// Step identifier.
        id: String,
        /// Shell command line (passed to `sh -c` by the sandbox layer).
        command: String,
        /// Optional command timeout. When unset, the engine falls back
        /// to `[workflow] default_shell_timeout`.
        timeout: Option<Duration>,
        /// Optional `${{ ... }}` boolean expression.
        when: Option<String>,
    },

    /// Write a file to disk (subject to sandbox write checks).
    WriteFile {
        /// Step identifier.
        id: String,
        /// Path template.
        path: String,
        /// File contents template.
        content: String,
    },
}

impl StepDef {
    /// Return the step's id.
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Agent { id, .. } | Self::Shell { id, .. } | Self::WriteFile { id, .. } => id,
        }
    }

    /// Return the step's kind label as used in progress events.
    #[must_use]
    pub fn step_type(&self) -> &'static str {
        match self {
            Self::Agent { .. } => "agent",
            Self::Shell { .. } => "shell",
            Self::WriteFile { .. } => "write_file",
        }
    }

    /// Return the step's `when` expression, if any.
    #[must_use]
    pub fn when(&self) -> Option<&str> {
        match self {
            Self::Agent { when, .. } | Self::Shell { when, .. } => when.as_deref(),
            // write_file does not currently support `when` per SPEC §8.4.
            Self::WriteFile { .. } => None,
        }
    }
}

/// A fully-validated workflow definition.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowDef {
    /// Unique workflow identifier (snake_case-ish).
    pub id: String,
    /// Optional human-readable description.
    pub description: Option<String>,
    /// Execution mode (placeholder beyond `Normal`).
    pub mode: WorkflowMode,
    /// Declared inputs (sorted by name via `BTreeMap`).
    pub inputs: BTreeMap<String, InputDef>,
    /// Ordered list of steps.
    pub steps: Vec<StepDef>,
    /// Output expressions (each value is a `${{ ... }}` template).
    pub outputs: BTreeMap<String, String>,
}

/// Result of executing a single step.
#[derive(Debug, Clone, PartialEq)]
pub struct StepResult {
    /// Structured output (for agent steps: parsed JSON or `{"text": "..."}`;
    /// for shell: `{"stdout": "...", "stderr": "...", "exit_code": N}`;
    /// for `write_file`: `{"path": "..."}`.
    pub output: serde_json::Value,
    /// Process exit code (0 for non-shell steps).
    pub exit_code: i32,
    /// Captured stdout (empty for non-shell steps).
    pub stdout: String,
    /// Captured stderr (empty for non-shell steps).
    pub stderr: String,
    /// Files written or otherwise modified by the step.
    pub changed_files: Vec<String>,
}

impl Default for StepResult {
    fn default() -> Self {
        Self {
            output: serde_json::Value::Null,
            exit_code: 0,
            stdout: String::new(),
            stderr: String::new(),
            changed_files: Vec::new(),
        }
    }
}

/// Final outputs produced by a workflow run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WorkflowOutputs {
    /// Each declared output mapped to its rendered string.
    pub values: BTreeMap<String, String>,
}

/// Lightweight metadata for a discovered workflow.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowMeta {
    /// Workflow id, equal to the file stem.
    pub id: String,
    /// Absolute path to the YAML file on disk.
    pub source_path: PathBuf,
    /// Optional description (parsed from the YAML).
    pub description: Option<String>,
}
