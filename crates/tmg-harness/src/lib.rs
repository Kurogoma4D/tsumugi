//! tsumugi run/session harness.
//!
//! This crate introduces the unified "Run" model described in SPEC §9.
//! Every interaction with `tmg` happens within a [`Run`], whether
//! ad-hoc (interactive TUI session) or harnessed by a workflow.
//!
//! In addition to the [`Run`] / [`RunStore`] / [`RunRunner`] data plane
//! the crate provides:
//!
//! - artifact I/O for [`progress.md`](artifacts::ProgressLog) and
//!   [`session_NNN.json`](artifacts::SessionLog), plus the harnessed-
//!   only [`features.json`](artifacts::FeatureList) and
//!   [`init.sh`](artifacts::InitScript)
//! - Run-scoped tools that the LLM can invoke:
//!   [`ProgressAppendTool`], [`SessionBootstrapTool`],
//!   [`SessionSummarySaveTool`], plus the harnessed-only
//!   [`FeatureListReadTool`] / [`FeatureListMarkPassingTool`]
//! - a [`HarnessStreamSink`] decorator that updates the active
//!   session's stats from the existing
//!   [`StreamSink`](tmg_core::StreamSink) callbacks
//!
//! Forthcoming features (tracked as separate issues):
//!
//! - workflow engine integration and escalation evaluator
//! - `tmg run` CLI subcommand and richer TUI surface
//! - `initializer` subagent that generates `features.json` / `init.sh`
//! - `tester` subagent for the harnessed `smoke_test_result` field

pub mod artifacts;
pub mod error;
pub mod escalation;
pub mod run;
pub mod runner;
pub mod session;
pub mod sink;
pub mod state;
pub mod store;
pub mod tools;

pub use artifacts::{
    Feature, FeatureList, Features, FeaturesSummary, FeaturesSummaryEntry, InitScript,
    InitScriptError, InitScriptOutput, ProgressLog, SessionLog, SessionLogEntry,
};
pub use error::HarnessError;
pub use escalation::{
    EscalationConfig, EscalationConfigError, EscalationDecision, EscalationError,
    EscalationEvaluator, EscalationSignal, EscalatorLauncher, SubagentEscalatorLauncher,
};
pub use run::{Run, RunId, RunScope, RunStatus, RunSummary};
pub use runner::{DEFAULT_BOOTSTRAP_MAX_TOKENS, RunProgressEvent, RunProgressReceiver, RunRunner};
pub use session::{Session, SessionEndTrigger, SessionHandle};
pub use sink::HarnessStreamSink;
pub use state::{SessionState, TurnSummary};
pub use store::{
    FEATURES_FILENAME, INIT_SCRIPT_FILENAME, PROGRESS_FILENAME, RUN_FILENAME, RunStore,
    SESSION_LOG_DIRNAME, WORKSPACE_LINK,
};
pub use tools::{
    BootstrapPayload, FeatureListMarkPassingTool, FeatureListReadTool, InitScriptStatus,
    ProgressAppendTool, RunRunnerToolProvider, SessionBootstrapTool, SessionSummarySaveTool,
    SmokeTestResult, register_run_tools,
};
