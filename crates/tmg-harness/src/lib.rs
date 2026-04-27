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
//!   [`session_NNN.json`](artifacts::SessionLog)
//! - Run-scoped tools that the LLM can invoke:
//!   [`ProgressAppendTool`], [`SessionBootstrapTool`],
//!   [`SessionSummarySaveTool`]
//! - a [`HarnessStreamSink`] decorator that updates the active
//!   session's stats from the existing
//!   [`StreamSink`](tmg_core::StreamSink) callbacks
//!
//! Forthcoming features (tracked as separate issues):
//!
//! - `features.json` schema for harnessed runs (#34)
//! - workflow engine integration and escalation evaluator
//! - `tmg run` CLI subcommand and richer TUI surface

pub mod artifacts;
pub mod error;
pub mod run;
pub mod runner;
pub mod session;
pub mod sink;
pub mod store;
pub mod tools;

pub use artifacts::{ProgressLog, SessionLog, SessionLogEntry};
pub use error::HarnessError;
pub use run::{Run, RunId, RunScope, RunStatus, RunSummary};
pub use runner::{DEFAULT_BOOTSTRAP_MAX_TOKENS, RunRunner};
pub use session::{Session, SessionEndTrigger, SessionHandle};
pub use sink::HarnessStreamSink;
pub use store::{PROGRESS_FILENAME, RUN_FILENAME, RunStore, SESSION_LOG_DIRNAME, WORKSPACE_LINK};
pub use tools::{ProgressAppendTool, SessionBootstrapTool, SessionSummarySaveTool};
