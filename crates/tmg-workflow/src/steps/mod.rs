//! Step handlers for the workflow engine.
//!
//! Each submodule implements one step type. Leaf handlers receive a
//! borrow of the engine's resources plus the resolved
//! [`crate::expr::ExprContext`] and return a [`crate::def::StepResult`].
//! Control-flow handlers ([`branch`], [`group`], [`loop_step`],
//! [`parallel`], [`human`]) accept the [`crate::engine::EngineCtx`] and
//! the shared `step_results` map and dispatch their children back into
//! [`crate::engine::dispatch_step`].

pub(crate) mod agent;
pub(crate) mod branch;
pub(crate) mod group;
pub(crate) mod human;
pub(crate) mod loop_step;
pub(crate) mod parallel;
pub(crate) mod path_util;
pub(crate) mod shell;
pub(crate) mod workflow;
pub(crate) mod write_file;
