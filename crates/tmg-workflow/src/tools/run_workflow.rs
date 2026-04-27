//! `run_workflow` builtin tool (issue #41).
//!
//! Foreground mode (`background: false`, the default) runs the
//! workflow inline and returns its outputs as a JSON object. Background
//! mode (`background: true`) spawns the workflow on a dedicated
//! `tokio::task` and immediately returns a [`WorkflowRunId`] — the LLM
//! can then poll `workflow_status` to track it.
//!
//! ## Background-run lifecycle and eviction
//!
//! Each background run is registered in the shared
//! [`BackgroundRunsHandle`] map. The map is periodically swept on every
//! `start_background` and `workflow_status` call; completed runs older
//! than [`BACKGROUND_RUN_RETENTION`] are dropped so a long-lived
//! process does not accumulate finished entries indefinitely. Running
//! entries are never evicted regardless of age.
//!
//! ## Cancellation
//!
//! Each background run owns a child [`CancellationToken`] derived from
//! the engine-wide token. [`RunWorkflowTool::cancel`] and
//! [`RunWorkflowTool::cancel_all`] expose this surface to host code
//! (e.g. the CLI's TUI shutdown path); the LLM is *not* given a
//! `cancel_workflow` tool in this issue. Firing the token causes the
//! spawned task's `select!` loop to record a `failed` outcome whose
//! error message references cancellation.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use serde_json::{Value, json};
use tokio::sync::{Mutex, OnceCell, mpsc};
use tokio_util::sync::CancellationToken;

use tmg_tools::{Tool, ToolError, ToolResult};

use super::types::{
    BACKGROUND_RUN_RETENTION, BackgroundRun, BackgroundRunsHandle, PROGRESS_BUFFER_CAP,
    WorkflowRunId, reap_completed_runs,
};
use crate::WorkflowEngine;
use crate::def::{WorkflowDef, WorkflowOutputs};
use crate::engine::WorkflowIndex;
use crate::progress::WorkflowProgress;

/// LLM-facing tool: start a workflow run.
///
/// See module-level docs for parameter shape, return contract,
/// background-run eviction policy, and the host-side cancel API.
pub struct RunWorkflowTool {
    engine: Arc<WorkflowEngine>,
    workflow_index: WorkflowIndex,
    background_runs: BackgroundRunsHandle,
    /// Optional observer that receives a clone of every
    /// [`WorkflowProgress`] event the *foreground* path observes.
    ///
    /// The host (e.g. the CLI's TUI startup) installs this so live
    /// progress events reach the TUI in addition to the foreground
    /// drain task. Background runs use the per-run progress buffer
    /// (queried via `workflow_status`) and are not fanned out here —
    /// the buffer already provides the durable history surface.
    progress_observer: Option<mpsc::Sender<WorkflowProgress>>,
}

impl RunWorkflowTool {
    /// Build a new `run_workflow` tool without a TUI progress observer.
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
            progress_observer: None,
        }
    }

    /// Build a new `run_workflow` tool that fans foreground progress
    /// events out to `observer` in addition to draining them
    /// internally. The CLI uses this to feed its TUI activity pane.
    ///
    /// Sends to a closed/full observer channel are silently dropped:
    /// the observer is best-effort, and a stalled TUI must never
    /// block engine progress.
    #[must_use]
    pub fn with_progress_observer(
        engine: Arc<WorkflowEngine>,
        workflow_index: WorkflowIndex,
        background_runs: BackgroundRunsHandle,
        observer: mpsc::Sender<WorkflowProgress>,
    ) -> Self {
        Self {
            engine,
            workflow_index,
            background_runs,
            progress_observer: Some(observer),
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
                    "description": "Map of input name to value, matching the workflow's declared inputs schema.",
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

    // The `Tool` trait still requires returning a boxed future for
    // dyn-compatibility (see `tmg_tools::types::Tool::execute`); when
    // the trait migrates to a native `async fn`, this can collapse to
    // a plain `async fn execute`.
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

        // `inputs` is optional: when absent we use an empty map. When
        // *present*, the value must be a JSON object — anything else
        // (string / array / number / bool / null) is a programming
        // error in the LLM's tool call rather than something we should
        // silently ignore.
        let inputs: BTreeMap<String, Value> = match params.get("inputs") {
            None => BTreeMap::new(),
            Some(Value::Object(map)) => map.clone().into_iter().collect(),
            Some(_) => {
                return Ok(ToolResult::error("inputs must be a JSON object"));
            }
        };

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
            let outputs = run_foreground(
                &self.engine,
                &workflow,
                inputs,
                self.progress_observer.clone(),
            )
            .await;
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
    /// (sixteen hex characters of pseudo-random entropy; see
    /// [`WorkflowRunId::generate`] for collision analysis). Before
    /// inserting the new entry we sweep the map of completed runs
    /// older than [`BACKGROUND_RUN_RETENTION`].
    async fn start_background(
        &self,
        workflow: WorkflowDef,
        inputs: BTreeMap<String, Value>,
    ) -> WorkflowRunId {
        let run_id = WorkflowRunId::generate();
        let (progress_tx, mut progress_rx) = mpsc::channel(64);

        // Pre-allocate the shared state so the spawn closure can
        // capture `Arc` clones; once `tokio::spawn` returns we hand
        // the real `JoinHandle` to the registered `BackgroundRun`.
        let progress_buffer = Arc::new(Mutex::new(VecDeque::with_capacity(PROGRESS_BUFFER_CAP)));
        let final_outputs: Arc<OnceCell<Result<WorkflowOutputs, String>>> =
            Arc::new(OnceCell::new());
        let cancel = CancellationToken::new();
        let started_at = Instant::now();

        let progress_buffer_for_task = Arc::clone(&progress_buffer);
        let final_outputs_for_task = Arc::clone(&final_outputs);
        let cancel_for_task = cancel.clone();
        let engine = Arc::clone(&self.engine);

        let cancel_for_select = cancel.clone();
        let join_handle = tokio::spawn(async move {
            let workflow_id = workflow.id.clone();
            let run_fut = engine.run_with_cancel(&workflow, inputs, progress_tx, cancel_for_task);
            tokio::pin!(run_fut);
            // Outcome of the run, set by exactly one branch of the
            // select! loop below. We use a local rather than writing
            // straight into `final_outputs` so the post-loop logic
            // can converge on a single set-once site.
            let outcome: Result<WorkflowOutputs, String>;
            loop {
                tokio::select! {
                    biased;
                    () = cancel_for_select.cancelled() => {
                        // External cancel observed. The engine itself
                        // may not honour the token at every await
                        // point (only `human` and inner control-flow
                        // steps do), so we record the cancellation
                        // outcome here and let the spawned run future
                        // unwind on its own. Any final progress
                        // events emitted before the engine sees the
                        // token are drained best-effort.
                        while let Ok(ev) = progress_rx.try_recv() {
                            push_progress(&progress_buffer_for_task, ev).await;
                        }
                        tracing::info!(
                            workflow = %workflow_id,
                            "background workflow cancelled",
                        );
                        outcome = Err("workflow cancelled".to_owned());
                        break;
                    }
                    event = progress_rx.recv() => {
                        if let Some(ev) = event {
                            push_progress(&progress_buffer_for_task, ev).await;
                        } else {
                            // Channel closed before the run future
                            // resolved. This is unusual but not
                            // fatal: we fall through and await the
                            // run future to capture its real
                            // outcome rather than guessing.
                            let result = (&mut run_fut).await;
                            outcome = match result {
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
                            break;
                        }
                    }
                    result = &mut run_fut => {
                        // Drain any final events the engine emitted
                        // before its sender dropped, *before* writing
                        // the outcome — both select arms then converge
                        // on the same post-loop completion path.
                        while let Ok(ev) = progress_rx.try_recv() {
                            push_progress(&progress_buffer_for_task, ev).await;
                        }
                        outcome = match result {
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
                        break;
                    }
                }
            }
            let _ = final_outputs_for_task.set(outcome);
        });

        // Construct the `BackgroundRun` *after* `tokio::spawn` so the
        // registered `join_handle` field reflects the real task —
        // `is_finished()` therefore reports correctly. The
        // `Arc<Mutex<...>>` / `Arc<OnceCell<...>>` clones below point
        // at the same allocations the spawn closure captured, so
        // reads through the registered run observe writes performed
        // by the task.
        let bg_run = Arc::new(BackgroundRun {
            started_at,
            progress_buffer: Arc::clone(&progress_buffer),
            final_outputs: Arc::clone(&final_outputs),
            join_handle: Mutex::new(Some(join_handle)),
            cancel,
        });

        let mut runs = self.background_runs.lock().await;
        let now = Instant::now();
        reap_completed_runs(&mut runs, BACKGROUND_RUN_RETENTION, now);
        runs.insert(run_id.clone(), bg_run);
        run_id
    }

    /// Fire the cancellation token for the background run with `run_id`.
    ///
    /// Returns `true` if a matching run was found, `false` otherwise.
    /// Looking up an already-finished run still returns `true` (the
    /// token fires harmlessly and the cancel is recorded as a no-op).
    /// This API is intended for host code (e.g. the CLI's TUI
    /// shutdown path); the LLM does not see it as a tool.
    pub async fn cancel(&self, run_id: &WorkflowRunId) -> bool {
        let runs = self.background_runs.lock().await;
        if let Some(entry) = runs.get(run_id) {
            entry.cancel.cancel();
            true
        } else {
            false
        }
    }

    /// Fire cancellation tokens for every registered background run.
    /// Used at process / TUI shutdown to give in-flight workflows a
    /// chance to terminate promptly rather than be aborted at the
    /// runtime boundary. Returns the number of runs that had their
    /// token fired (including already-completed ones, since the call
    /// is idempotent).
    pub async fn cancel_all(&self) -> usize {
        let runs = self.background_runs.lock().await;
        let mut count = 0;
        for entry in runs.values() {
            entry.cancel.cancel();
            count += 1;
        }
        count
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
///
/// When `observer` is `Some`, every event the engine emits is also
/// forwarded to the observer channel (typically wired to the TUI's
/// activity pane). Forwarding uses `try_send`: a stalled or full
/// observer must never block engine progress, so dropped events are
/// accepted as the cost of liveness.
async fn run_foreground(
    engine: &WorkflowEngine,
    workflow: &WorkflowDef,
    inputs: BTreeMap<String, Value>,
    observer: Option<mpsc::Sender<WorkflowProgress>>,
) -> Result<WorkflowOutputs, crate::error::WorkflowError> {
    // Use a small channel; events are not surfaced to the LLM in
    // foreground mode, but the engine still emits them. We drain in a
    // detached task so the engine can make forward progress.
    let (tx, mut rx) = mpsc::channel::<WorkflowProgress>(64);
    let drain = tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
            if let Some(obs) = observer.as_ref() {
                // Best-effort fan-out. `try_send` covers both the
                // `Closed` (TUI dropped its receiver) and `Full`
                // (TUI tick is slower than the engine) cases without
                // blocking the drain loop.
                let _ = obs.try_send(ev);
            }
            // The original drain semantics: events are discarded
            // beyond the optional fan-out. Foreground callers that
            // want a complete event log should use background mode
            // and `workflow_status`.
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
