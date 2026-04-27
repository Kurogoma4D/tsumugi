//! LLM-facing tool implementations for `tmg-workflow`.
//!
//! Two tools are exposed (issue #41):
//!
//! - [`RunWorkflowTool`] — start a workflow run, either inline
//!   (`background: false`) or detached (`background: true` returns a
//!   handle immediately).
//! - [`WorkflowStatusTool`] — query the status of a previously-started
//!   background run.
//!
//! Both tools share an [`Arc<Mutex<HashMap<WorkflowRunId, BackgroundRun>>>`]
//! so a `workflow_status` call sees runs started by `run_workflow` in
//! the same process. The CLI builds the map once and registers both
//! tools against it.

mod run_workflow;
mod types;
mod workflow_status;

pub use run_workflow::RunWorkflowTool;
// `register_workflow_tools_with_observer` is defined inline below;
// `register_workflow_tools` is also a free fn here. They are
// re-exported via `crate::lib`.
pub use types::{BackgroundRun, BackgroundRunsHandle, WorkflowRunId};
pub use workflow_status::WorkflowStatusTool;

use std::sync::Arc;

use tokio::sync::{Mutex, mpsc};

use crate::WorkflowEngine;
use crate::progress::WorkflowProgress;

/// Convenience helper: register `run_workflow` and `workflow_status`
/// against the supplied [`tmg_tools::ToolRegistry`].
///
/// The CLI calls this only when at least one workflow has been
/// discovered; an empty registration would expose tools that always
/// fail with "unknown workflow" and bloat the LLM tool list.
pub fn register_workflow_tools(
    registry: &mut tmg_tools::ToolRegistry,
    engine: &Arc<WorkflowEngine>,
    background_runs: &BackgroundRunsHandle,
) {
    register_workflow_tools_with_observer(registry, engine, background_runs, None);
}

/// Like [`register_workflow_tools`] but installs `observer` so the
/// foreground `run_workflow` path fans every [`WorkflowProgress`]
/// event out to the supplied channel as well as draining it
/// internally.
///
/// Sends to a closed/full observer channel are silently dropped — the
/// observer is best-effort, and a stalled TUI must never block engine
/// progress.
pub fn register_workflow_tools_with_observer(
    registry: &mut tmg_tools::ToolRegistry,
    engine: &Arc<WorkflowEngine>,
    background_runs: &BackgroundRunsHandle,
    observer: Option<mpsc::Sender<WorkflowProgress>>,
) {
    let Some(workflow_index) = engine.workflow_index_handle() else {
        // No index attached → nothing to dispatch. We still skip
        // registration to keep the LLM's view of the tool catalogue
        // honest.
        return;
    };
    if let Some(observer) = observer {
        registry.register(RunWorkflowTool::with_progress_observer(
            Arc::clone(engine),
            workflow_index,
            Arc::clone(background_runs),
            observer,
        ));
    } else {
        registry.register(RunWorkflowTool::new(
            Arc::clone(engine),
            workflow_index,
            Arc::clone(background_runs),
        ));
    }
    registry.register(WorkflowStatusTool::new(Arc::clone(background_runs)));
}

/// Build a fresh [`BackgroundRunsHandle`].
#[must_use]
pub fn new_background_runs() -> BackgroundRunsHandle {
    Arc::new(Mutex::new(std::collections::HashMap::new()))
}

/// Fire the cancellation token on every registered background run.
///
/// Intended for the host (e.g. the CLI) to call before runtime
/// shutdown so in-flight workflows have a chance to terminate
/// promptly rather than be aborted at the runtime boundary. Mirrors
/// [`RunWorkflowTool::cancel_all`] but does not require holding a
/// reference to the tool — the CLI registers the tool into a
/// `ToolRegistry` (a type-erasing container) and only the shared
/// `BackgroundRunsHandle` is left in scope.
///
/// Returns the number of run entries observed.
pub async fn cancel_all_background_runs(handle: &BackgroundRunsHandle) -> usize {
    let runs = handle.lock().await;
    let mut count = 0;
    for entry in runs.values() {
        entry.cancel.cancel();
        count += 1;
    }
    count
}
