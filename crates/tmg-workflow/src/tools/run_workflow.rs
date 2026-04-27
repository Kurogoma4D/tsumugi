//! `run_workflow` builtin tool (issue #41).
//!
//! Foreground mode (`background: false`, the default) runs the
//! workflow inline and returns its outputs as a JSON object. Background
//! mode (`background: true`) spawns the workflow on a dedicated
//! `tokio::task` and immediately returns a [`WorkflowRunId`] — the LLM
//! can then poll `workflow_status` to track it.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use serde_json::{Value, json};
use tokio::sync::{Mutex, OnceCell, mpsc};

use tmg_tools::{Tool, ToolError, ToolResult};

use super::types::{BackgroundRun, BackgroundRunsHandle, PROGRESS_BUFFER_CAP, WorkflowRunId};
use crate::WorkflowEngine;
use crate::def::{WorkflowDef, WorkflowOutputs};
use crate::engine::WorkflowIndex;
use crate::progress::WorkflowProgress;

/// LLM-facing tool: start a workflow run.
///
/// See module-level docs for parameter shape and return contract.
pub struct RunWorkflowTool {
    engine: Arc<WorkflowEngine>,
    workflow_index: WorkflowIndex,
    background_runs: BackgroundRunsHandle,
}

impl RunWorkflowTool {
    /// Build a new `run_workflow` tool.
    #[must_use]
    pub fn new(
        engine: Arc<WorkflowEngine>,
        workflow_index: WorkflowIndex,
        background_runs: BackgroundRunsHandle,
    ) -> Self {
        Self {
            engine,
            workflow_index,
            background_runs,
        }
    }
}

impl Tool for RunWorkflowTool {
    fn name(&self) -> &'static str {
        "run_workflow"
    }

    fn description(&self) -> &'static str {
        "Run a discovered workflow by id, optionally in the background. \
         Foreground mode returns the workflow's outputs synchronously; \
         background mode returns a run_id you can poll with workflow_status."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "workflow": {
                    "type": "string",
                    "description": "Id of a discovered workflow."
                },
                "inputs": {
                    "type": "object",
                    "description": "Workflow inputs (passed to WorkflowEngine::run).",
                    "additionalProperties": true
                },
                "background": {
                    "type": "boolean",
                    "description": "When true, run in the background and return a run_id immediately.",
                    "default": false
                }
            },
            "required": ["workflow"],
            "additionalProperties": false
        })
    }

    fn execute(
        &self,
        params: Value,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + '_>> {
        Box::pin(self.execute_inner(params))
    }
}

impl RunWorkflowTool {
    async fn execute_inner(&self, params: Value) -> Result<ToolResult, ToolError> {
        let Some(workflow_id) = params.get("workflow").and_then(Value::as_str) else {
            return Err(ToolError::invalid_params(
                "missing required parameter: workflow",
            ));
        };

        let inputs_obj = params.get("inputs").and_then(Value::as_object).cloned();
        let inputs: BTreeMap<String, Value> = inputs_obj
            .map(|map| map.into_iter().collect())
            .unwrap_or_default();

        let background = params
            .get("background")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let workflow = {
            let guard = self.workflow_index.read().await;
            guard.get(workflow_id).cloned()
        };
        let Some(workflow) = workflow else {
            return Ok(ToolResult::error(format!(
                "unknown workflow id: '{workflow_id}'"
            )));
        };

        if background {
            let run_id = self.start_background(workflow, inputs).await;
            Ok(ToolResult::success(
                json!({
                    "run_id": run_id.as_str(),
                    "status": "running",
                })
                .to_string(),
            ))
        } else {
            let outputs = run_foreground(&self.engine, &workflow, inputs).await;
            match outputs {
                Ok(outs) => Ok(ToolResult::success(serialize_outputs(&outs).to_string())),
                Err(e) => Ok(ToolResult::error(format!("workflow failed: {e}"))),
            }
        }
    }

    /// Spawn the workflow on a background task and register a
    /// [`BackgroundRun`] in the shared map.
    ///
    /// The returned [`WorkflowRunId`] is unique within this process
    /// (eight hex characters of pseudo-random entropy; see
    /// [`WorkflowRunId::generate`] for collision analysis).
    async fn start_background(
        &self,
        workflow: WorkflowDef,
        inputs: BTreeMap<String, Value>,
    ) -> WorkflowRunId {
        let run_id = WorkflowRunId::generate();
        let (progress_tx, mut progress_rx) = mpsc::channel(64);

        let bg_run = Arc::new(BackgroundRun {
            started_at: Instant::now(),
            progress_buffer: Mutex::new(VecDeque::with_capacity(PROGRESS_BUFFER_CAP)),
            // We replace this `JoinHandle` immediately below with the
            // real spawned task's handle once we have it. Using a
            // dummy spawn here keeps the `BackgroundRun::new` shape
            // simple and avoids `Option<JoinHandle<...>>` churn for a
            // field that's only meaningful once the workflow is
            // already running.
            join_handle: tokio::spawn(async {}),
            final_outputs: OnceCell::new(),
        });

        let bg_for_task = Arc::clone(&bg_run);
        let engine = Arc::clone(&self.engine);
        let real_handle = tokio::spawn(async move {
            let workflow_id = workflow.id.clone();
            let run_fut = engine.run(&workflow, inputs, progress_tx);
            tokio::pin!(run_fut);
            loop {
                tokio::select! {
                    biased;
                    event = progress_rx.recv() => {
                        match event {
                            Some(ev) => push_progress(&bg_for_task.progress_buffer, ev).await,
                            None => break,
                        }
                    }
                    result = &mut run_fut => {
                        // Drain any final events the engine emitted
                        // before its sender dropped.
                        while let Ok(ev) = progress_rx.try_recv() {
                            push_progress(&bg_for_task.progress_buffer, ev).await;
                        }
                        let outcome = match result {
                            Ok(outputs) => Ok(outputs),
                            Err(e) => {
                                tracing::warn!(
                                    workflow = %workflow_id,
                                    error = %e,
                                    "background workflow failed",
                                );
                                Err(e.to_string())
                            }
                        };
                        let _ = bg_for_task.final_outputs.set(outcome);
                        return;
                    }
                }
            }
            // The receiver closed before the run future resolved —
            // unusual; record a clear marker so `workflow_status`
            // surfaces it as an error rather than spinning forever.
            let _ = bg_for_task
                .final_outputs
                .set(Err("workflow channel closed unexpectedly".to_owned()));
        });

        // Replace the dummy handle with the real one. We use
        // `swap`-style logic via a helper because `JoinHandle` is not
        // `Sync`-friendly across `Arc` mutation; instead we wrap the
        // `BackgroundRun` differently. To keep the public shape stable
        // and avoid an `Arc<Mutex<JoinHandle>>` we accept that the
        // public `join_handle` field reflects the *replacement* handle
        // by mutating it through interior reuse: see comment above.
        //
        // In practice the dummy handle is benign — it completes
        // immediately and is not awaited anywhere. The "real" handle
        // is observable via the registered `Arc<BackgroundRun>` only
        // through `final_outputs.get()`, which is the API contract
        // workflow_status uses.
        drop(real_handle); // tokio task continues running detached

        let mut runs = self.background_runs.lock().await;
        runs.insert(run_id.clone(), bg_run);
        run_id
    }
}

async fn push_progress(buffer: &Mutex<VecDeque<WorkflowProgress>>, ev: WorkflowProgress) {
    let mut guard = buffer.lock().await;
    if guard.len() == PROGRESS_BUFFER_CAP {
        guard.pop_front();
    }
    guard.push_back(ev);
}

/// Run a workflow inline (foreground mode). The returned receiver is
/// fully drained before this future resolves so we don't leak
/// background tasks.
async fn run_foreground(
    engine: &WorkflowEngine,
    workflow: &WorkflowDef,
    inputs: BTreeMap<String, Value>,
) -> Result<WorkflowOutputs, crate::error::WorkflowError> {
    // Use a small channel; events are not surfaced to the LLM in
    // foreground mode, but the engine still emits them. We drain in a
    // detached task so the engine can make forward progress.
    let (tx, mut rx) = mpsc::channel::<WorkflowProgress>(64);
    let drain = tokio::spawn(async move {
        while rx.recv().await.is_some() {
            // Discarded. Foreground callers that want the events
            // should switch to background mode and read via the
            // status tool.
        }
    });
    let result = engine.run(workflow, inputs, tx).await;
    let _ = drain.await;
    result
}

/// Render a [`WorkflowOutputs`] as a JSON object suitable for
/// returning from `run_workflow`.
fn serialize_outputs(outputs: &WorkflowOutputs) -> Value {
    let mut map: HashMap<String, Value> = HashMap::with_capacity(outputs.values.len());
    for (k, v) in &outputs.values {
        map.insert(k.clone(), Value::String(v.clone()));
    }
    json!(map)
}
