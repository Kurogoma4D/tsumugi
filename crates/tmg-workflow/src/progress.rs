//! Workflow progress events (SPEC §8.10).
//!
//! `WorkflowProgress` is the canonical message type emitted on the
//! channel passed to [`crate::engine::WorkflowEngine::run`]. The TUI
//! layer (issue #46) will consume this stream to render live progress.

use std::sync::Arc;

use tokio::sync::{Mutex, oneshot};

use crate::def::{StepResult, WorkflowOutputs};

/// Progress event emitted by the workflow engine.
///
/// ## Cloning trade-off for `HumanInputRequired`
///
/// The `response_tx` carried by [`WorkflowProgress::HumanInputRequired`]
/// is wrapped in `Arc<Mutex<Option<oneshot::Sender<HumanResponse>>>>`.
/// This is a deliberate trade-off so the entire enum can keep
/// `derive(Clone)` (TUI / CLI consumers may broadcast events to
/// multiple sinks). The downside is that the responder must `take()`
/// the inner `Sender` before sending; double-take returns `None`,
/// which the engine treats as "no responder will reply" and
/// short-circuits with a `StepFailed`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum WorkflowProgress {
    /// A step has started executing.
    StepStarted {
        /// Step id.
        step_id: String,
        /// Step type label (`"agent"`, `"shell"`, `"write_file"`,
        /// `"loop"`, `"branch"`, `"parallel"`, `"group"`, `"human"`).
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
    /// A step failed; the engine aborts the run after emitting this
    /// (unless wrapped in a `Group` with `on_failure: continue`).
    StepFailed {
        /// Step id.
        step_id: String,
        /// Error message.
        error: String,
    },
    /// A loop iteration just finished (or is about to begin) — emitted
    /// at the start of every iteration so consumers can render
    /// progress bars without inferring from inner step events.
    LoopIteration {
        /// The loop step's id.
        step_id: String,
        /// 1-based iteration counter.
        iteration: u32,
        /// Configured `max_iterations`.
        max: u32,
    },
    /// The engine is paused on a `human` step waiting for a response.
    ///
    /// The receiver should call
    /// `response_tx.lock().await.take()` to extract the one-shot
    /// sender, then `send(HumanResponse { ... })`. Double-take returns
    /// `None`; the engine treats that as a missing response and fails
    /// the step.
    HumanInputRequired {
        /// The `human` step's id.
        step_id: String,
        /// Prompt to display.
        message: String,
        /// Allowed response keywords.
        options: Vec<String>,
        /// Optional rendered `show:` payload.
        show: Option<String>,
        /// Optional `revise_target` declared on the human step. The TUI
        /// uses this to populate `HumanResponse::target` when the user
        /// picks `revise`; without it, a `revise` choice cannot be
        /// satisfied and the TUI surfaces an error.
        revise_target: Option<String>,
        /// One-shot reply channel (see type docs for the
        /// take-once contract).
        response_tx: HumanResponder,
    },
    /// The workflow finished and produced its final outputs.
    WorkflowCompleted {
        /// Final outputs (each declared output rendered as a string).
        outputs: WorkflowOutputs,
    },
}

/// Take-once `oneshot::Sender` for human-input replies.
///
/// Wrapped in `Arc<Mutex<Option<...>>>` so [`WorkflowProgress`] can
/// derive `Clone` while still respecting the underlying single-shot
/// channel semantics.
pub type HumanResponder = Arc<Mutex<Option<oneshot::Sender<HumanResponse>>>>;

/// Decision returned by a UI in response to
/// [`WorkflowProgress::HumanInputRequired`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HumanResponse {
    /// The high-level decision.
    pub kind: HumanResponseKind,
    /// Required only when `kind == HumanResponseKind::Revise`. The
    /// engine validates that this matches the human step's
    /// `revise_target` (when set) or one of the previously-executed
    /// step ids.
    pub target: Option<String>,
}

impl HumanResponse {
    /// Convenience constructor for an `approve` response.
    #[must_use]
    pub fn approve() -> Self {
        Self {
            kind: HumanResponseKind::Approve,
            target: None,
        }
    }

    /// Convenience constructor for a `reject` response.
    #[must_use]
    pub fn reject() -> Self {
        Self {
            kind: HumanResponseKind::Reject,
            target: None,
        }
    }

    /// Convenience constructor for a `revise` response targeting `step_id`.
    #[must_use]
    pub fn revise(step_id: impl Into<String>) -> Self {
        Self {
            kind: HumanResponseKind::Revise,
            target: Some(step_id.into()),
        }
    }
}

/// Kinds of decisions a human responder may return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HumanResponseKind {
    /// Continue past the human step.
    Approve,
    /// Abort the workflow.
    Reject,
    /// Rewind to (and re-execute from) `target`.
    Revise,
}
