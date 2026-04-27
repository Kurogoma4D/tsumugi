//! `group` step handler (SPEC §8.4).
//!
//! Bundles multiple steps under one of three failure policies:
//!
//! - [`FailurePolicy::Abort`] (default): an inner failure propagates.
//! - [`FailurePolicy::Retry`]: re-run the *entire* group on inner
//!   failure, up to `max_retries` times.
//! - [`FailurePolicy::Continue`]: log the failure, mark the group's
//!   own `StepResult` as `failed`, and continue past the group.
//!
//! ## Retry semantics
//!
//! When retrying, the engine *restores* `step_results` to the
//! pre-group snapshot before each attempt. Without this, a partially
//! successful first attempt would leave inner-step results in place
//! that subsequent attempts could observe via `${{ steps.* }}` and
//! derive incorrect decisions from. The snapshot is captured locally
//! in this handler — the engine-level `snapshots` map is reserved for
//! revise rewinds across `human` boundaries.

use std::collections::BTreeMap;

use serde_json::json;

use crate::def::{FailurePolicy, StepDef, StepResult};
use crate::engine::{EngineCtx, StepOutcome, dispatch_step};
use crate::error::{Result, WorkflowError};
use crate::progress::WorkflowProgress;

#[expect(
    clippy::too_many_lines,
    reason = "linear retry/abort/continue dispatch; splitting would scatter the per-policy match arms across helpers"
)]
pub(crate) async fn execute(
    ctx: &EngineCtx,
    step: &StepDef,
    step_results: &mut BTreeMap<String, StepResult>,
) -> Result<StepOutcome> {
    let StepDef::Group {
        id,
        on_failure,
        max_retries,
        steps: body,
    } = step
    else {
        return Err(WorkflowError::StepFailed {
            step_id: step.id().to_owned(),
            message: "internal error: group::execute called with non-group step".to_owned(),
        });
    };

    let _ = ctx
        .progress_tx
        .send(WorkflowProgress::StepStarted {
            step_id: id.clone(),
            step_type: "group",
        })
        .await;

    let pre_snapshot = step_results.clone();

    let mut attempts: u32 = 0;
    let max_attempts = match on_failure {
        FailurePolicy::Retry => max_retries.saturating_add(1),
        FailurePolicy::Abort | FailurePolicy::Continue => 1,
    };

    let mut last_error: Option<WorkflowError> = None;

    'outer: while attempts < max_attempts {
        attempts += 1;
        // Restore pre-snapshot before each attempt (no-op on attempt 1).
        if attempts > 1 {
            *step_results = pre_snapshot.clone();
        }

        for inner in body {
            match dispatch_step(ctx, inner, step_results).await {
                Ok(StepOutcome::Completed) => {}
                Ok(StepOutcome::Revise { target }) => {
                    return Ok(StepOutcome::Revise { target });
                }
                Err(err) => {
                    tracing::debug!(group = id, attempt = attempts, %err, "group inner step failed");
                    last_error = Some(err);
                    continue 'outer;
                }
            }
        }
        // All inner steps passed — exit early.
        last_error = None;
        break;
    }

    match (last_error, on_failure) {
        (None, _) => {
            let result = StepResult {
                output: json!({"attempts": attempts, "ok": true}),
                exit_code: 0,
                stdout: String::new(),
                stderr: String::new(),
                changed_files: Vec::new(),
            };
            let _ = ctx
                .progress_tx
                .send(WorkflowProgress::StepCompleted {
                    step_id: id.clone(),
                    result: result.clone(),
                })
                .await;
            step_results.insert(id.clone(), result);
            Ok(StepOutcome::Completed)
        }
        (Some(err), FailurePolicy::Continue) => {
            tracing::warn!(group = id, %err, "group failed under on_failure: continue");
            // Restore the pre-group snapshot so subsequent steps don't
            // observe a half-applied failed group.
            *step_results = pre_snapshot;
            let result = StepResult {
                output: json!({
                    "attempts": attempts,
                    "ok": false,
                    "error": err.to_string(),
                }),
                exit_code: 1,
                stdout: String::new(),
                stderr: err.to_string(),
                changed_files: Vec::new(),
            };
            // Emit a `StepCompleted` (not `StepFailed`) — the *group*
            // succeeded by absorbing the failure, even though its
            // inner work did not. Callers that want to discriminate
            // can read `result.exit_code`.
            let _ = ctx
                .progress_tx
                .send(WorkflowProgress::StepCompleted {
                    step_id: id.clone(),
                    result: result.clone(),
                })
                .await;
            step_results.insert(id.clone(), result);
            Ok(StepOutcome::Completed)
        }
        (Some(err), FailurePolicy::Abort | FailurePolicy::Retry) => {
            let _ = ctx
                .progress_tx
                .send(WorkflowProgress::StepFailed {
                    step_id: id.clone(),
                    error: err.to_string(),
                })
                .await;
            Err(err)
        }
    }
}
