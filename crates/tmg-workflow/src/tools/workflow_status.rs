//! `workflow_status` builtin tool (issue #41).
//!
//! Looks up a [`WorkflowRunId`] in the shared background-runs map and
//! returns a JSON status object:
//!
//! ```json
//! {
//!   "status": "running" | "completed" | "failed",
//!   "current_step": "verify_loop (3/5)",
//!   "elapsed": "2m34s",
//!   "outputs": <object | null>,
//!   "error": <string | null>
//! }
//! ```
//!
//! `current_step` is derived from the most recent
//! [`crate::progress::WorkflowProgress`] event in the run's bounded
//! buffer:
//!
//! - If the latest event is `LoopIteration`, the field reads
//!   `step_id (iter/max)`.
//! - Otherwise the latest `StepStarted` event's `step_id` is used
//!   bare.
//! - If the most recent terminal event for that step is
//!   `StepCompleted` / `StepFailed` (i.e. nothing has started since
//!   the step finished), `current_step` is `null` to avoid pretending
//!   a finished step is still running.
//! - Outside an active step (run finished, or no events buffered yet)
//!   the field is `null`.
//!
//! ## Eviction side effect
//!
//! Every call sweeps the runs map of completed entries older than
//! [`crate::tools::types::BACKGROUND_RUN_RETENTION`] before performing
//! its lookup. This keeps a long-lived process from accumulating
//! finished entries; running entries are never evicted.

use std::pin::Pin;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use tmg_sandbox::SandboxContext;
use tmg_tools::{Tool, ToolError, ToolResult};

use super::types::{
    BACKGROUND_RUN_RETENTION, BackgroundRun, BackgroundRunsHandle, WorkflowRunId,
    reap_completed_runs,
};
use crate::progress::WorkflowProgress;

/// LLM-facing tool: query the status of a previously-started
/// background workflow run.
pub struct WorkflowStatusTool {
    background_runs: BackgroundRunsHandle,
}

impl WorkflowStatusTool {
    /// Build a new `workflow_status` tool against the given runs map.
    #[must_use]
    pub fn new(background_runs: BackgroundRunsHandle) -> Self {
        Self { background_runs }
    }
}

impl Tool for WorkflowStatusTool {
    fn name(&self) -> &'static str {
        "workflow_status"
    }

    fn description(&self) -> &'static str {
        "Query the status of a background workflow run by run_id. \
         Returns the latest step, elapsed time, and (when finished) \
         the outputs or error message."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Run id returned by run_workflow."
                }
            },
            "required": ["run_id"],
            "additionalProperties": false
        })
    }

    // The `Tool` trait still requires returning a boxed future for
    // dyn-compatibility (see `tmg_tools::types::Tool::execute`); when
    // the trait migrates to a native `async fn`, this can collapse to
    // a plain `async fn execute`.
    fn execute<'a>(
        &'a self,
        params: Value,
        _ctx: &'a SandboxContext,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + 'a>> {
        // Status lookup is in-memory only, so the dispatch sandbox is
        // advisory here. Accepting it keeps the [`Tool`] signature
        // uniform across in-memory and on-disk tools.
        Box::pin(self.execute_inner(params))
    }
}

impl WorkflowStatusTool {
    async fn execute_inner(&self, params: Value) -> Result<ToolResult, ToolError> {
        let Some(run_id_str) = params.get("run_id").and_then(Value::as_str) else {
            return Err(ToolError::invalid_params(
                "missing required parameter: run_id",
            ));
        };
        // Reject malformed run ids early — anything that does not
        // match the canonical 16-char lowercase hex shape cannot be in
        // the map by construction, and surfacing "unknown run_id" for
        // a typo is more useful than a silent map miss.
        let Some(run_id) = WorkflowRunId::parse_lookup_key(run_id_str) else {
            return Ok(ToolResult::error(format!("unknown run_id: '{run_id_str}'")));
        };

        let bg_run = {
            let mut guard = self.background_runs.lock().await;
            // Sweep stale completed entries opportunistically so a
            // long-lived session does not accumulate finished runs.
            // Running entries are never evicted regardless of age.
            reap_completed_runs(&mut guard, BACKGROUND_RUN_RETENTION, Instant::now());
            guard.get(&run_id).cloned()
        };
        let Some(bg_run) = bg_run else {
            return Ok(ToolResult::error(format!("unknown run_id: '{run_id_str}'")));
        };

        let response = build_status(&bg_run).await;
        Ok(ToolResult::success(response.to_string()))
    }
}

/// Translate a [`BackgroundRun`] into the JSON status response.
async fn build_status(bg_run: &BackgroundRun) -> Value {
    let elapsed = format_duration(bg_run.started_at.elapsed());
    let final_outcome = bg_run.final_outputs.get();
    let buffer = bg_run.progress_buffer.lock().await;

    let current_step = derive_current_step(&buffer);

    match final_outcome {
        None => json!({
            "status": "running",
            "current_step": current_step,
            "elapsed": elapsed,
            "outputs": Value::Null,
            "error": Value::Null,
        }),
        Some(Ok(outputs)) => {
            let mut output_obj = serde_json::Map::new();
            for (k, v) in &outputs.values {
                output_obj.insert(k.clone(), Value::String(v.clone()));
            }
            json!({
                "status": "completed",
                // Once finished, there's no "current" step; surface
                // the last observed step name as a hint without the
                // iteration counter.
                "current_step": current_step,
                "elapsed": elapsed,
                "outputs": Value::Object(output_obj),
                "error": Value::Null,
            })
        }
        Some(Err(message)) => json!({
            "status": "failed",
            "current_step": current_step,
            "elapsed": elapsed,
            "outputs": Value::Null,
            "error": message.clone(),
        }),
    }
}

/// Derive the `current_step` field from the progress buffer.
///
/// Priority:
///
/// 1. The most recent `LoopIteration` event → `"<step_id> (i/max)"`
///    (loops emit iterations *during* their execution, so a trailing
///    `LoopIteration` always names the active step regardless of
///    inner-step `StepStarted` events emitted after it).
/// 2. The most recent `StepStarted` event whose step has not since
///    received a `StepCompleted` / `StepFailed` event for the same
///    id → `"<step_id>"`.
/// 3. None.
///
/// Returning `null` once the latest `StepStarted`'s step has
/// terminated avoids pretending a finished step is still running just
/// because no later step has begun. This matters between sequential
/// steps and at the very end of a successful run.
fn derive_current_step(buffer: &std::collections::VecDeque<WorkflowProgress>) -> Value {
    // First pass (newest-to-oldest): if a `LoopIteration` is the most
    // recent loop-related event we have, it wins outright.
    for ev in buffer.iter().rev() {
        if let WorkflowProgress::LoopIteration {
            step_id,
            iteration,
            max,
        } = ev
        {
            return Value::String(format!("{step_id} ({iteration}/{max})"));
        }
    }

    // Second pass: find the most recent `StepStarted` whose step has
    // not since terminated (no later `StepCompleted` / `StepFailed`
    // for the same id). A trailing termination event for the latest
    // started step suppresses it so we don't report a finished step
    // as current.
    let mut terminated_steps: Vec<&str> = Vec::new();
    for ev in buffer.iter().rev() {
        match ev {
            WorkflowProgress::StepCompleted { step_id, .. }
            | WorkflowProgress::StepFailed { step_id, .. } => {
                terminated_steps.push(step_id.as_str());
            }
            WorkflowProgress::StepStarted { step_id, .. } => {
                let id = step_id.as_str();
                if !terminated_steps.contains(&id) {
                    return Value::String(id.to_owned());
                }
                // Step started but later terminated; keep walking
                // for an earlier still-active step (rare; only with
                // overlapping control-flow events in the buffer).
            }
            _ => {}
        }
    }
    Value::Null
}

/// Format an elapsed [`Duration`] like `"2m34s"` / `"45s"` / `"310ms"`.
///
/// We bias toward minute/second granularity because workflows are
/// long-running by design; sub-second precision is only useful when
/// the run hasn't reached a one-second boundary yet.
fn format_duration(d: Duration) -> String {
    let total_secs = d.as_secs();
    if total_secs == 0 {
        return format!("{}ms", d.subsec_millis());
    }
    let mins = total_secs / 60;
    let secs = total_secs % 60;
    if mins == 0 {
        format!("{secs}s")
    } else {
        format!("{mins}m{secs:02}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::def::StepResult;
    use std::collections::VecDeque;

    #[test]
    fn format_duration_branches() {
        assert_eq!(format_duration(Duration::from_millis(310)), "310ms");
        assert_eq!(format_duration(Duration::from_secs(45)), "45s");
        assert_eq!(format_duration(Duration::from_secs(2 * 60 + 34)), "2m34s");
    }

    #[test]
    fn derive_current_step_prefers_recent_loop() {
        let mut q: VecDeque<WorkflowProgress> = VecDeque::new();
        q.push_back(WorkflowProgress::StepStarted {
            step_id: "verify_loop".to_owned(),
            step_type: "loop",
        });
        q.push_back(WorkflowProgress::LoopIteration {
            step_id: "verify_loop".to_owned(),
            iteration: 3,
            max: 5,
        });
        q.push_back(WorkflowProgress::StepStarted {
            step_id: "verify".to_owned(),
            step_type: "shell",
        });
        let v = derive_current_step(&q);
        // The loop iteration wins because we walk newest-to-oldest
        // and break on the first LoopIteration we see.
        assert_eq!(v, Value::String("verify_loop (3/5)".to_owned()));
    }

    #[test]
    fn derive_current_step_falls_back_to_step_started() {
        let mut q: VecDeque<WorkflowProgress> = VecDeque::new();
        q.push_back(WorkflowProgress::StepStarted {
            step_id: "build".to_owned(),
            step_type: "shell",
        });
        let v = derive_current_step(&q);
        assert_eq!(v, Value::String("build".to_owned()));
    }

    #[test]
    fn derive_current_step_empty() {
        let q: VecDeque<WorkflowProgress> = VecDeque::new();
        let v = derive_current_step(&q);
        assert_eq!(v, Value::Null);
    }

    /// A `StepCompleted` more recent than the latest `StepStarted`
    /// for the same id suppresses the started event so we don't
    /// pretend a finished step is still running.
    #[test]
    fn derive_current_step_suppresses_finished_step() {
        let mut q: VecDeque<WorkflowProgress> = VecDeque::new();
        q.push_back(WorkflowProgress::StepStarted {
            step_id: "build".to_owned(),
            step_type: "shell",
        });
        q.push_back(WorkflowProgress::StepCompleted {
            step_id: "build".to_owned(),
            result: StepResult::default(),
        });
        let v = derive_current_step(&q);
        assert_eq!(v, Value::Null);
    }

    /// Same idea for `StepFailed`.
    #[test]
    fn derive_current_step_suppresses_failed_step() {
        let mut q: VecDeque<WorkflowProgress> = VecDeque::new();
        q.push_back(WorkflowProgress::StepStarted {
            step_id: "verify".to_owned(),
            step_type: "shell",
        });
        q.push_back(WorkflowProgress::StepFailed {
            step_id: "verify".to_owned(),
            error: "boom".to_owned(),
        });
        let v = derive_current_step(&q);
        assert_eq!(v, Value::Null);
    }
}
