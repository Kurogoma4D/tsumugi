//! tsumugi run/session harness.
//!
//! This crate introduces the unified "Run" model described in SPEC §9.
//! Every interaction with `tmg` happens within a [`Run`], whether
//! ad-hoc (interactive TUI session) or harnessed by a workflow.
//!
//! The crate is intentionally minimal in this first slice: it
//! provides the data model, a TOML-backed [`RunStore`], and a
//! [`RunRunner`] that the CLI startup sequence uses to attach the
//! existing `AgentLoop` to a persisted run.
//!
//! Forthcoming features (tracked as separate issues):
//!
//! - `progress.md` / `session_log.jsonl` writers
//! - `features.json` schema for harnessed runs
//! - workflow engine integration and escalation evaluator
//! - `tmg run` CLI subcommand and richer TUI surface

pub mod error;
pub mod run;
pub mod runner;
pub mod session;
pub mod store;

pub use error::HarnessError;
pub use run::{Run, RunId, RunScope, RunStatus, RunSummary};
pub use runner::RunRunner;
pub use session::{Session, SessionEndTrigger, SessionHandle};
pub use store::{RUN_FILENAME, RunStore, WORKSPACE_LINK};
