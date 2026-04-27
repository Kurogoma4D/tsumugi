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
//! - Outside an active step (run finished, or no events buffered yet)
//!   the field is `null`.

use std::pin::Pin;
use std::time::Duration;

use serde_json::{Value, json};

use tmg_tools::{Tool, ToolError, ToolResult};

use super::types::{BackgroundRun, BackgroundRunsHandle, WorkflowRunId};
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

    fn execute(
        &self,
        params: Value,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + '_>> {
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
        let run_id = WorkflowRunId::from_str_unchecked(run_id_str.to_owned());

        let bg_run = {
            let guard = self.background_runs.lock().await;
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
/// 1. The most recent `LoopIteration` event → `"<step_id> (i/max)"`.
/// 2. The most recent `StepStarted` event → `"<step_id>"`.
/// 3. None.
fn derive_current_step(buffer: &std::collections::VecDeque<WorkflowProgress>) -> Value {
    let mut last_step_started: Option<&str> = None;
    let mut last_loop: Option<(&str, u32, u32)> = None;
    for ev in buffer.iter().rev() {
        match ev {
            WorkflowProgress::LoopIteration {
                step_id,
                iteration,
                max,
            } => {
                last_loop = Some((step_id.as_str(), *iteration, *max));
                break;
            }
            WorkflowProgress::StepStarted { step_id, .. } => {
                if last_step_started.is_none() {
                    last_step_started = Some(step_id.as_str());
                }
            }
            _ => {}
        }
    }
    if let Some((id, iter, max)) = last_loop {
        return Value::String(format!("{id} ({iter}/{max})"));
    }
    if let Some(id) = last_step_started {
        return Value::String(id.to_owned());
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
}
