//! Workflow type definitions (SPEC §8.10).
//!
//! These are the canonical, validated workflow representations consumed
//! by the engine. Raw YAML is parsed into these via [`crate::parse`].
//!
//! Control-flow steps (`Loop`, `Branch`, `Parallel`, `Group`, `Human`)
//! were added in issue #40 and dispatch through [`crate::engine`].

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
#[non_exhaustive]
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
/// Leaf step types (`Agent`, `Shell`, `WriteFile`) execute directly;
/// control-flow types (`Loop`, `Branch`, `Parallel`, `Group`, `Human`)
/// recursively dispatch their children through the engine.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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

    /// Iterate `steps` until `until` evaluates to `true` or
    /// `max_iterations` is reached.
    ///
    /// The inner step ids are re-tagged per iteration as `id[1]`,
    /// `id[2]`, ... so each iteration's `StepResult` is addressable
    /// independently in `steps.<inner>[N]` lookups. The loop step's
    /// own `StepResult` carries either `{"max_iterations_reached":
    /// true}` (no `until` success) or `{"max_iterations_reached":
    /// false, "iterations": N}` on a successful exit.
    Loop {
        /// Step identifier.
        id: String,
        /// Maximum number of iterations to run.
        max_iterations: u32,
        /// `${{ ... }}` boolean expression evaluated *after* each
        /// iteration; iteration stops on `true`.
        until: String,
        /// Steps to execute on every iteration (in order).
        steps: Vec<StepDef>,
    },

    /// Evaluate each `(when, steps)` pair in order; execute the steps
    /// of the first pair whose `when` is truthy. If none match, run
    /// `default` (or no-op when absent).
    Branch {
        /// Step identifier.
        id: String,
        /// `(when_expression, steps)` pairs. The first matching pair
        /// wins; remaining pairs are not evaluated.
        conditions: Vec<(String, Vec<StepDef>)>,
        /// Optional fallback steps when no `conditions` match.
        default: Option<Vec<StepDef>>,
    },

    /// Spawn each child step concurrently. Honours the
    /// `[workflow] max_parallel_agents` cap (agent steps only); shell
    /// and `write_file` steps are not capped.
    Parallel {
        /// Step identifier.
        id: String,
        /// Steps to execute concurrently.
        steps: Vec<StepDef>,
    },

    /// Group multiple steps under a single failure policy.
    Group {
        /// Step identifier.
        id: String,
        /// What to do if any inner step fails.
        on_failure: FailurePolicy,
        /// Maximum retry count when `on_failure == Retry`. Ignored for
        /// other policies.
        max_retries: u32,
        /// Steps to execute as a group.
        steps: Vec<StepDef>,
    },

    /// Pause the workflow and ask the user (TUI / CLI) for a decision.
    ///
    /// The engine emits [`crate::progress::WorkflowProgress::HumanInputRequired`]
    /// and awaits a [`crate::progress::HumanResponse`]. `revise`
    /// rewinds workflow state to the snapshot taken before
    /// `revise_target`'s execution and re-runs the workflow from that
    /// step.
    Human {
        /// Step identifier.
        id: String,
        /// Prompt shown to the user.
        message: String,
        /// Optional `${{ ... }}` template rendered as additional
        /// context (e.g. `${{ steps.ux_design.output }}`).
        show: Option<String>,
        /// Allowed response keywords (e.g. `["approve", "reject", "revise"]`).
        options: Vec<String>,
        /// Step id to rewind to when the user picks `revise`. Required
        /// when `revise` is in `options`.
        revise_target: Option<String>,
    },

    /// Run another discovered workflow as a step. Used by the
    /// declarative-pipeline form (issue #41) where each `stages:` entry
    /// references a workflow id and may template its inputs against
    /// `${{ inputs.* }}` and `${{ stages.<id>.outputs.<key> }}`.
    ///
    /// ## Trade-off: separate variant vs. a `Pipeline` workflow type
    ///
    /// We add this as a new `StepDef` variant rather than introducing a
    /// dedicated top-level `Pipeline` type because:
    ///
    /// 1. The engine driver loop and snapshot/revise machinery are
    ///    already step-shaped — a pipeline just *is* a workflow whose
    ///    leaves are workflow refs.
    /// 2. It keeps `WorkflowDef` as the single canonical run target so
    ///    `run_workflow` can dispatch pipelines and ordinary workflows
    ///    uniformly without branching.
    /// 3. `loop_spec` slots into the existing iteration engine; a
    ///    separate type would have to re-implement looping.
    ///
    /// The cost is that `StepDef` grows another arm. The variant is
    /// `#[non_exhaustive]` (the parent enum already is) so future
    /// pipeline-only fields don't break callers.
    Workflow {
        /// Step identifier (the stage id in a pipeline).
        id: String,
        /// Target workflow id; resolved against the engine's
        /// workflow index at dispatch time.
        workflow_id: String,
        /// Templated inputs (each value is a `${{ ... }}` template
        /// string evaluated at dispatch time).
        inputs: BTreeMap<String, String>,
        /// Optional in-line loop wrapper. When present, the engine
        /// iterates the workflow step up to `max_iterations` times,
        /// evaluating `until` after each iteration and exiting early
        /// when the expression becomes truthy.
        loop_spec: Option<LoopSpec>,
    },
}

/// Inline loop specification for a [`StepDef::Workflow`] step.
///
/// Mirrors the shape of a [`StepDef::Loop`] body, but applies to a
/// single workflow invocation rather than an arbitrary inner step
/// list. We keep the two forms distinct so the pipeline grammar
/// (`stages: [{ workflow:..., loop:{ ... } }]`) reads naturally without
/// nesting an extra `loop` step type around every workflow step.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopSpec {
    /// Maximum iterations.
    pub max_iterations: u32,
    /// `${{ ... }}` boolean expression evaluated *after* each
    /// iteration. Iteration stops on truthy.
    pub until: String,
}

/// Failure handling policy for a [`StepDef::Group`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FailurePolicy {
    /// Propagate the inner failure upward and abort the workflow.
    #[default]
    Abort,
    /// Retry the entire group up to `max_retries` times before giving up.
    Retry,
    /// Log the failure but continue running subsequent steps.
    Continue,
}

impl StepDef {
    /// Return the step's id.
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Agent { id, .. }
            | Self::Shell { id, .. }
            | Self::WriteFile { id, .. }
            | Self::Loop { id, .. }
            | Self::Branch { id, .. }
            | Self::Parallel { id, .. }
            | Self::Group { id, .. }
            | Self::Human { id, .. }
            | Self::Workflow { id, .. } => id,
        }
    }

    /// Return the step's kind label as used in progress events.
    #[must_use]
    pub fn step_type(&self) -> &'static str {
        match self {
            Self::Agent { .. } => "agent",
            Self::Shell { .. } => "shell",
            Self::WriteFile { .. } => "write_file",
            Self::Loop { .. } => "loop",
            Self::Branch { .. } => "branch",
            Self::Parallel { .. } => "parallel",
            Self::Group { .. } => "group",
            Self::Human { .. } => "human",
            Self::Workflow { .. } => "workflow",
        }
    }

    /// Return the step's `when` expression, if any.
    #[must_use]
    pub fn when(&self) -> Option<&str> {
        match self {
            Self::Agent { when, .. } | Self::Shell { when, .. } => when.as_deref(),
            // write_file / control-flow / workflow steps do not
            // currently expose a `when` clause at the outer level —
            // control flow steps express conditional execution via
            // their own grammar (`until`, `conditions`, etc.),
            // `write_file` is intentionally always-on per SPEC §8.4,
            // and pipeline `workflow` steps wrap their conditional
            // logic via `loop_spec` or upstream `branch` stages.
            Self::WriteFile { .. }
            | Self::Loop { .. }
            | Self::Branch { .. }
            | Self::Parallel { .. }
            | Self::Group { .. }
            | Self::Human { .. }
            | Self::Workflow { .. } => None,
        }
    }

    /// Re-tag this step's id (and recursively, child step ids) by
    /// rewriting `id` to `f(id)`. Used by the loop runner to suffix
    /// inner steps as `id[1]`, `id[2]`, ... per iteration.
    ///
    /// The function is applied once per step node visited; child step
    /// ids are independent (each child sees its own original id).
    pub(crate) fn retag_ids(&mut self, f: &impl Fn(&str) -> String) {
        match self {
            Self::Agent { id, .. }
            | Self::Shell { id, .. }
            | Self::WriteFile { id, .. }
            | Self::Loop { id, .. }
            | Self::Branch { id, .. }
            | Self::Parallel { id, .. }
            | Self::Group { id, .. }
            | Self::Human { id, .. }
            | Self::Workflow { id, .. } => {
                *id = f(id);
            }
        }
        match self {
            Self::Loop { steps, .. } | Self::Parallel { steps, .. } | Self::Group { steps, .. } => {
                for s in steps {
                    s.retag_ids(f);
                }
            }
            Self::Branch {
                conditions,
                default,
                ..
            } => {
                for (_, steps) in conditions {
                    for s in steps {
                        s.retag_ids(f);
                    }
                }
                if let Some(default_steps) = default {
                    for s in default_steps {
                        s.retag_ids(f);
                    }
                }
            }
            Self::Agent { .. }
            | Self::Shell { .. }
            | Self::WriteFile { .. }
            | Self::Human { .. }
            | Self::Workflow { .. } => {}
        }
    }
}

/// A fully-validated workflow definition.
///
/// Marked `#[non_exhaustive]` because future SPEC additions are
/// expected to grow optional fields (e.g. per-workflow telemetry tags).
/// External crates must use [`WorkflowDef::new`] (and the builder
/// methods) instead of struct literals so growing the struct is not a
/// breaking change.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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
    /// Optional `init` phase (only valid for `mode: long_running`).
    ///
    /// SPEC §8.12: the `init` phase runs once per long-running run on
    /// the first attach. It typically invokes the `initializer` builtin
    /// subagent through an `agent` step to produce `features.json` /
    /// `init.sh` / `progress.md`. After [`InitPhase::steps`] complete,
    /// the executor flips the run to harnessed scope.
    pub init: Option<InitPhase>,
    /// Optional `iterate` phase (required for `mode: long_running`).
    ///
    /// SPEC §8.12: the `iterate` phase is the per-session loop. Each
    /// iteration begins a new harnessed session, resolves the
    /// [`IteratePhase::bootstrap`] items into a context block, runs
    /// [`IteratePhase::steps`], and evaluates [`IteratePhase::until`]
    /// to decide whether to stop.
    pub iterate: Option<IteratePhase>,
}

impl WorkflowDef {
    /// Build a new [`WorkflowDef`] with default fields.
    ///
    /// All optional fields (`description`, `mode`, `inputs`, `steps`,
    /// `outputs`, `init`, `iterate`) start empty. Use the builder
    /// methods or direct field assignment (within the crate) to
    /// populate them.
    #[must_use]
    pub fn new(id: String) -> Self {
        Self {
            id,
            description: None,
            mode: WorkflowMode::Normal,
            inputs: BTreeMap::new(),
            steps: Vec::new(),
            outputs: BTreeMap::new(),
            init: None,
            iterate: None,
        }
    }

    /// Set the workflow's [`WorkflowMode`].
    #[must_use]
    pub fn with_mode(mut self, mode: WorkflowMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the workflow description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the workflow's input declarations.
    #[must_use]
    pub fn with_inputs(mut self, inputs: BTreeMap<String, InputDef>) -> Self {
        self.inputs = inputs;
        self
    }

    /// Set the workflow's top-level steps.
    #[must_use]
    pub fn with_steps(mut self, steps: Vec<StepDef>) -> Self {
        self.steps = steps;
        self
    }

    /// Set the workflow's output templates.
    #[must_use]
    pub fn with_outputs(mut self, outputs: BTreeMap<String, String>) -> Self {
        self.outputs = outputs;
        self
    }

    /// Attach an `init:` phase.
    #[must_use]
    pub fn with_init(mut self, init: InitPhase) -> Self {
        self.init = Some(init);
        self
    }

    /// Attach an `iterate:` phase.
    #[must_use]
    pub fn with_iterate(mut self, iterate: IteratePhase) -> Self {
        self.iterate = Some(iterate);
        self
    }
}

/// `init:` phase configuration for a `mode: long_running` workflow.
///
/// Runs once per run, on the first attach (i.e. when the run is still
/// ad-hoc). The phase typically produces `features.json` / `init.sh` /
/// `progress.md` via the `initializer` builtin subagent and then the
/// executor escalates the run to harnessed scope.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct InitPhase {
    /// Optional named artifact paths exposed to expressions as
    /// `${{ artifacts.<name> }}`. The path is rendered as a string when
    /// referenced from a template.
    ///
    /// Example: `progress_file: ".tsumugi/runs/<id>/progress.md"`.
    pub artifacts: BTreeMap<String, PathBuf>,
    /// Steps to execute during initialisation.
    ///
    /// Step types are the standard set
    /// (`agent` / `shell` / `write_file` / `loop` / `branch` /
    /// `parallel` / `group` / `human`).
    pub steps: Vec<StepDef>,
}

/// `iterate:` phase configuration for a `mode: long_running` workflow.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct IteratePhase {
    /// Bootstrap items resolved at the start of each session and
    /// concatenated into the iterate step context.
    ///
    /// See [`BootstrapItem`] for the supported entry kinds.
    pub bootstrap: Vec<BootstrapItem>,
    /// Steps to execute every session.
    pub steps: Vec<StepDef>,
    /// `${{ ... }}` boolean expression evaluated after every session;
    /// when truthy the executor stops with [`crate::long_running::RunStatus::Completed`].
    pub until: String,
    /// Maximum number of sessions before the executor stops with
    /// [`crate::long_running::RunStatus::Exhausted`]. Must be `>= 1`.
    pub max_sessions: u32,
    /// Wall-clock budget for one session. When exceeded, the executor
    /// fires [`tmg_harness::SessionEndTrigger::Timeout`] via the
    /// existing watchdog path and continues with the next iteration.
    pub session_timeout: Duration,
}

/// One entry in a `bootstrap:` list.
///
/// Parsed from the YAML shorthand:
///
/// ```yaml
/// bootstrap:
///   - run: "ls -la"            # BootstrapItem::Run
///   - read: "src/lib.rs"       # BootstrapItem::Read
///   - smoke_test:              # BootstrapItem::SmokeTest
///       id: smoke
///       type: shell
///       command: "cargo test --lib"
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BootstrapItem {
    /// Run a shell command via the sandbox; capture stdout.
    Run(String),
    /// Read a file (path may use `${{ artifacts.* }}` references).
    Read(String),
    /// Execute a single step (typically a shell smoke test) and capture
    /// its stdout/stderr summary.
    SmokeTest(Box<StepDef>),
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
