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
//! - [`engine::WorkflowEngine`]: sequential executor for the
//!   `agent`, `shell`, and `write_file` step types.
//! - [`progress`]: progress events emitted during execution.
//!
//! ## Out of scope (this iteration)
//!
//! Control-flow steps (`loop`, `branch`, `parallel`, `group`, `human`)
//! and the `run_workflow` tool are tracked separately in issues #40 and
//! #41. The `WorkflowMode::LongRunning` placeholder parses correctly
//! but executes identically to `Normal` until issue #42 lands.

pub mod config;
pub mod def;
pub mod discovery;
pub mod engine;
pub mod error;
pub mod expr;
pub mod parse;
pub mod progress;
pub(crate) mod steps;

pub use config::WorkflowConfig;
pub use def::{
    InputDef, StepDef, StepResult, WorkflowDef, WorkflowMeta, WorkflowMode, WorkflowOutputs,
};
pub use discovery::discover_workflows;
pub use engine::WorkflowEngine;
pub use error::{Result, WorkflowError};
pub use expr::{ExprContext, eval_bool, eval_string, eval_value};
pub use parse::{parse_workflow_file, parse_workflow_str};
pub use progress::WorkflowProgress;
