//! Workflow progress events (SPEC §8.10).
//!
//! `WorkflowProgress` is the canonical message type emitted on the
//! channel passed to [`crate::engine::WorkflowEngine::run`]. The TUI
//! layer (issue #41) will consume this stream to render live progress.

use crate::def::{StepResult, WorkflowOutputs};

/// Progress event emitted by the workflow engine.
#[derive(Debug, Clone)]
pub enum WorkflowProgress {
    /// A step has started executing.
    StepStarted {
        /// Step id.
        step_id: String,
        /// Step type label (`"agent"`, `"shell"`, `"write_file"`).
        step_type: &'static str,
    },
    /// A streaming chunk of stdout / agent text.
    ///
    /// Reserved for future use; the current engine emits final
    /// `StepCompleted` events without intermediate streaming.
    StepOutput {
        /// Step id.
        step_id: String,
        /// Output chunk.
        chunk: String,
    },
    /// A step finished successfully.
    StepCompleted {
        /// Step id.
        step_id: String,
        /// Result captured from the step.
        result: StepResult,
    },
    /// A step failed; the engine aborts the run after emitting this.
    StepFailed {
        /// Step id.
        step_id: String,
        /// Error message.
        error: String,
    },
    /// The workflow finished and produced its final outputs.
    WorkflowCompleted {
        /// Final outputs (each declared output rendered as a string).
        outputs: WorkflowOutputs,
    },
}
