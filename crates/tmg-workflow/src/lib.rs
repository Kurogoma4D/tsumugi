//! tmg-workflow: declarative YAML workflows for tsumugi (SPEC §8).
//!
//! This crate provides:
//!
//! - [`def`]: canonical workflow types ([`def::WorkflowDef`],
//!   [`def::StepDef`], [`def::StepResult`], ...).
//! - [`parse`]: YAML → [`def::WorkflowDef`] parser with strict id
//!   validation and friendly error messages.
//! - [`discovery`]: project / user / config-paths discovery
//!   (priority-ordered, deduplicated by id).
//! - [`expr`]: minimal `${{ ... }}` template / boolean / value
//!   evaluator scoped over `inputs`, `steps`, `config`, and `env`.
//! - [`engine::WorkflowEngine`]: workflow executor with full
//!   control-flow dispatch (`agent`, `shell`, `write_file`, `loop`,
//!   `branch`, `parallel`, `group`, `human`).
//! - [`progress`]: progress events emitted during execution.
//!
//! ## Out of scope (this iteration)
//!
//! The `run_workflow` tool is tracked separately in issue #41. The
//! `WorkflowMode::LongRunning` placeholder parses correctly but
//! executes identically to `Normal` until issue #42 lands. TUI
//! integration for `human` steps is tracked in issue #46.

pub mod config;
pub mod def;
pub mod discovery;
pub mod engine;
pub mod error;
pub mod expr;
pub mod long_running;
pub mod parse;
pub mod progress;
pub(crate) mod steps;
pub mod templates;
pub mod tools;

pub use config::WorkflowConfig;
pub use def::{
    BootstrapItem, FailurePolicy, InitPhase, InputDef, IteratePhase, LoopSpec, StepDef, StepResult,
    WorkflowDef, WorkflowMeta, WorkflowMode, WorkflowOutputs,
};
pub use discovery::discover_workflows;
pub use engine::{EngineExtras, WorkflowEngine, WorkflowIndex};
pub use error::{Result, WorkflowError};
pub use expr::{ArtifactResolver, ExprContext, eval_bool, eval_string, eval_value};
pub use long_running::{LongRunningExecutor, RunStatus};
pub use parse::{parse_workflow_file, parse_workflow_str};
pub use progress::{HumanResponder, HumanResponse, HumanResponseKind, WorkflowProgress};
pub use tools::{
    BackgroundRunsHandle, RunWorkflowTool, WorkflowRunId, WorkflowStatusTool,
    register_workflow_tools,
};
