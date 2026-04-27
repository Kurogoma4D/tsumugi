//! Step handlers for the workflow engine.
//!
//! Each submodule implements one step type. Handlers receive a borrow
//! of the engine's resources plus the resolved [`crate::expr::ExprContext`]
//! and return a [`crate::def::StepResult`].

pub(crate) mod agent;
pub(crate) mod shell;
pub(crate) mod write_file;
