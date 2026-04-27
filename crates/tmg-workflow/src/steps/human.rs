//! `human` step handler (SPEC §8.4).
//!
//! Emits [`WorkflowProgress::HumanInputRequired`] with a one-shot
//! responder, awaits the [`HumanResponse`], and translates the
//! response into the appropriate [`StepOutcome`]:
//!
//! - `approve` -> `Completed` (workflow continues past this step).
//! - `reject` -> bubble up a [`WorkflowError::StepFailed`] so the
//!   engine aborts.
//! - `revise` -> [`StepOutcome::Revise`] carrying the target step id;
//!   the engine rewinds to the snapshot captured before that step.
//!
//! ## TUI integration
//!
//! The TUI / CLI consumer is responsible for `take()`-ing the
//! [`oneshot::Sender`] from the wrapped responder and `send`-ing a
//! [`HumanResponse`]. Tests in this crate drive the responder
//! directly (no TUI dependency).

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::json;
use tokio::sync::{Mutex, oneshot};

use crate::def::{StepDef, StepResult};
use crate::engine::{EngineCtx, StepOutcome, build_eval_ctx};
use crate::error::{Result, WorkflowError};
use crate::expr;
use crate::progress::{HumanResponseKind, WorkflowProgress};

#[expect(
    clippy::too_many_lines,
    reason = "linear approve/reject/revise dispatch; splitting would scatter the per-decision branches"
)]
pub(crate) async fn execute(
    ctx: &EngineCtx,
    step: &StepDef,
    step_results: &mut BTreeMap<String, StepResult>,
) -> Result<StepOutcome> {
    let StepDef::Human {
        id,
        message,
        show,
        options,
        revise_target,
    } = step
    else {
        return Err(WorkflowError::StepFailed {
            step_id: step.id().to_owned(),
            message: "internal error: human::execute called with non-human step".to_owned(),
        });
    };

    // NOTE: `StepStarted` is emitted by `engine::dispatch_step_inner`
    // for every step before delegating to the per-type handler;
    // emitting it again here would deliver duplicates.

    // Render the prompt and `show:` payload eagerly so the receiver
    // sees the substituted values, not the raw template.
    let stages_snapshot = ctx.stages.read().await.clone();
    let inner_ctx = build_eval_ctx(ctx, step_results, &stages_snapshot);
    let rendered_message = expr::eval_string(message, &inner_ctx)?;
    let rendered_show = match show {
        Some(template) => Some(expr::eval_string(template, &inner_ctx)?),
        None => None,
    };

    let (tx, rx) = oneshot::channel::<crate::progress::HumanResponse>();
    let responder = Arc::new(Mutex::new(Some(tx)));

    let _ = ctx
        .progress_tx
        .send(WorkflowProgress::HumanInputRequired {
            step_id: id.clone(),
            message: rendered_message,
            options: options.clone(),
            show: rendered_show,
            response_tx: Arc::clone(&responder),
        })
        .await;

    // Await the response. `recv` errors when the sender is dropped
    // without a value — surface that as a clear `StepFailed`. We also
    // honour the engine-level cancellation token so a workflow
    // never blocks forever when the UI consumer goes away without
    // responding.
    let response = tokio::select! {
        biased;
        () = ctx.cancel.cancelled() => {
            let err = WorkflowError::StepFailed {
                step_id: id.clone(),
                message: "human step cancelled".to_owned(),
            };
            let _ = ctx
                .progress_tx
                .send(WorkflowProgress::StepFailed {
                    step_id: id.clone(),
                    error: err.to_string(),
                })
                .await;
            return Err(err);
        }
        recv = rx => {
            recv.map_err(|_| WorkflowError::StepFailed {
                step_id: id.clone(),
                message: "human input channel was dropped before a response arrived".to_owned(),
            })?
        }
    };

    match response.kind {
        HumanResponseKind::Approve => {
            let result = StepResult {
                output: json!({"decision": "approve"}),
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
        HumanResponseKind::Reject => {
            let err = WorkflowError::StepFailed {
                step_id: id.clone(),
                message: "human reviewer rejected the workflow".to_owned(),
            };
            let _ = ctx
                .progress_tx
                .send(WorkflowProgress::StepFailed {
                    step_id: id.clone(),
                    error: err.to_string(),
                })
                .await;
            Err(err)
        }
        HumanResponseKind::Revise => {
            // Resolve the target. Prefer the response's explicit target,
            // falling back to the human step's declared `revise_target`.
            let target = response
                .target
                .clone()
                .or_else(|| revise_target.clone())
                .ok_or_else(|| WorkflowError::StepFailed {
                    step_id: id.clone(),
                    message: "human response 'revise' supplied no target and the human step has no revise_target"
                        .to_owned(),
                })?;
            // Optional sanity: if the human step *does* declare a
            // revise_target, restrict the response to that single id.
            // Without this a UI bug could rewind to an unintended step.
            if let Some(declared) = revise_target {
                if response.target.is_some() && response.target.as_ref() != Some(declared) {
                    return Err(WorkflowError::StepFailed {
                        step_id: id.clone(),
                        message: format!(
                            "revise target '{}' does not match declared revise_target '{declared}'",
                            response.target.as_deref().unwrap_or("<none>")
                        ),
                    });
                }
            }
            // Note: we intentionally do NOT insert a `StepCompleted` /
            // step_results entry for this human step on `revise` —
            // the engine will re-enter the workflow from `target` and
            // the human step itself will be re-run later.
            Ok(StepOutcome::Revise { target })
        }
    }
}
