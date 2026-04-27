//! `branch` step handler (SPEC §8.4).
//!
//! Evaluates each `(when, steps)` pair in order; the first pair whose
//! `when` is truthy has its `steps` executed. If none match, the
//! `default` body runs (when present); otherwise the branch is a
//! no-op. The branch's own `StepResult` records which arm fired.

use std::collections::BTreeMap;

use serde_json::json;

use crate::def::{StepDef, StepResult};
use crate::engine::{EngineCtx, StepOutcome, dispatch_step};
use crate::error::{Result, WorkflowError};
use crate::expr;
use crate::progress::WorkflowProgress;

pub(crate) async fn execute(
    ctx: &EngineCtx,
    step: &StepDef,
    step_results: &mut BTreeMap<String, StepResult>,
) -> Result<StepOutcome> {
    let StepDef::Branch {
        id,
        conditions,
        default,
    } = step
    else {
        return Err(WorkflowError::StepFailed {
            step_id: step.id().to_owned(),
            message: "internal error: branch::execute called with non-branch step".to_owned(),
        });
    };

    // NOTE: `StepStarted` is emitted by `engine::dispatch_step_inner`
    // for every step before delegating to the per-type handler;
    // emitting it again here would deliver duplicates.

    let mut chosen_arm: Option<usize> = None;

    let stages_snapshot = ctx.stages.read().await.clone();
    for (idx, (when_expr, _)) in conditions.iter().enumerate() {
        let inner_ctx =
            expr::ExprContext::new(&ctx.inputs, step_results, &ctx.config_json, &ctx.env)
                .with_stages(&stages_snapshot);
        let cond =
            expr::eval_bool(when_expr, &inner_ctx).map_err(|e| WorkflowError::StepFailed {
                step_id: id.clone(),
                message: format!("when-expression error in branch arm {idx}: {e}"),
            })?;
        if cond {
            chosen_arm = Some(idx);
            break;
        }
    }

    let body: Option<&Vec<StepDef>> = match chosen_arm {
        Some(idx) => Some(&conditions[idx].1),
        None => default.as_ref(),
    };

    let arm_label: String = match (chosen_arm, default) {
        (Some(idx), _) => format!("arm_{idx}"),
        (None, Some(_)) => "default".to_owned(),
        (None, None) => "noop".to_owned(),
    };

    if let Some(steps) = body {
        for inner in steps {
            // On failure, the inner leaf has already emitted its own
            // `StepFailed`; we additionally emit a `StepFailed` for
            // the *branch* so a consumer can render the branch as
            // failed and not just the leaf.
            let outcome = match dispatch_step(ctx, inner, step_results).await {
                Ok(outcome) => outcome,
                Err(err) => {
                    let _ = ctx
                        .progress_tx
                        .send(WorkflowProgress::StepFailed {
                            step_id: id.clone(),
                            error: err.to_string(),
                        })
                        .await;
                    return Err(err);
                }
            };
            if let StepOutcome::Revise { target } = outcome {
                return Ok(StepOutcome::Revise { target });
            }
        }
    }

    let result = StepResult {
        output: json!({
            "matched": arm_label,
            "matched_index": chosen_arm,
        }),
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
