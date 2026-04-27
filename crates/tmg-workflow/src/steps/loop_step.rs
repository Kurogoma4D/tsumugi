//! `loop` step handler (SPEC §8.4).
//!
//! Iterates the inner step list up to `max_iterations` times,
//! evaluating `until` after each iteration and exiting when it
//! becomes truthy. Inner step ids are *re-tagged* per iteration
//! (`id[1]`, `id[2]`, ...) so each iteration's [`StepResult`] is
//! independently addressable in `steps.<inner>[N]`-style lookups via
//! the `[]` indexing syntax.
//!
//! ## Re-tagging convention
//!
//! Given an inner step `verify`, iteration 1 stores its result under
//! `verify[1]`, iteration 2 under `verify[2]`, etc. The most recent
//! iteration's result is *also* stored under the bare id `verify` so
//! the loop's own `until:` expression can refer to
//! `steps.verify.exit_code` without iteration arithmetic. The mirror
//! is overwritten each iteration; only the most recent is preserved.
//!
//! ## Result shape
//!
//! - On `until` success: `output = {"max_iterations_reached": false,
//!   "iterations": N}` where N is the 1-based iteration count.
//! - On `max_iterations` reached without `until` succeeding: `output =
//!   {"max_iterations_reached": true, "iterations": max_iterations}`.

use std::collections::BTreeMap;

use serde_json::json;

use crate::def::{StepDef, StepResult};
use crate::engine::{EngineCtx, StepOutcome, build_eval_ctx, dispatch_step};
use crate::error::{Result, WorkflowError};
use crate::expr;
use crate::progress::WorkflowProgress;

pub(crate) async fn execute(
    ctx: &EngineCtx,
    step: &StepDef,
    step_results: &mut BTreeMap<String, StepResult>,
) -> Result<StepOutcome> {
    let StepDef::Loop {
        id,
        max_iterations,
        until,
        steps: body,
    } = step
    else {
        return Err(WorkflowError::StepFailed {
            step_id: step.id().to_owned(),
            message: "internal error: loop_step::execute called with non-loop step".to_owned(),
        });
    };

    // NOTE: `StepStarted` is emitted for *every* step (including
    // control-flow steps) by `engine::dispatch_step_inner` before this
    // handler is invoked. Emitting again here would surface as a
    // duplicate event to consumers. See engine.rs.

    let mut completed = 0u32;
    let mut hit_until = false;

    for iteration in 1..=*max_iterations {
        let _ = ctx
            .progress_tx
            .send(WorkflowProgress::LoopIteration {
                step_id: id.clone(),
                iteration,
                max: *max_iterations,
            })
            .await;

        for inner in body {
            // Clone + re-tag the inner step so its id reads `id[N]`.
            // We append `[N]` rather than mutating in place because
            // `StepDef::Branch`/`Loop`/etc. recursive children must
            // also be re-tagged consistently. `retag_ids` walks the
            // tree.
            let mut tagged = inner.clone();
            let suffix = format!("[{iteration}]");
            tagged.retag_ids(&|orig| format!("{orig}{suffix}"));

            // Dispatch the iteration-tagged step. Its result is stored
            // under the tagged id; we then mirror it under the bare
            // id (without the suffix) so the loop's `until:` can refer
            // to it naturally. On failure, the inner leaf has already
            // emitted its own `StepFailed`; we additionally emit a
            // `StepFailed` for the *loop* so a TUI consumer can show
            // the loop as failed and not just the leaf.
            let outcome = match dispatch_step(ctx, &tagged, step_results).await {
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
            // Loop bodies don't propagate Revise through to the engine
            // — a `human` inside a loop body that asks for revise
            // tries to rewind to a target outside the loop, which we
            // bubble up unchanged.
            if let StepOutcome::Revise { target } = outcome {
                let _ = ctx
                    .progress_tx
                    .send(WorkflowProgress::StepFailed {
                        step_id: id.clone(),
                        error: format!("loop interrupted by revise -> {target}"),
                    })
                    .await;
                return Ok(StepOutcome::Revise { target });
            }

            // Mirror the latest iteration result under the bare id.
            // We only mirror leaf steps — control-flow children
            // already write their own ids into `step_results` and the
            // parser forbids `ref:`-ing them, so they cannot reach
            // this branch in practice. Guard defensively anyway.
            //
            // When the tagged id was *not* written this iteration
            // (e.g. the inner leaf was skipped via `when=false`), we
            // remove the bare id rather than preserving iteration
            // N-1's value. Otherwise `until:` (which reads the bare
            // id) would observe stale data and could exit early.
            if matches!(
                inner,
                StepDef::Agent { .. } | StepDef::Shell { .. } | StepDef::WriteFile { .. }
            ) {
                let bare_id = inner.id();
                let tagged_id = format!("{bare_id}{suffix}");
                if let Some(latest) = step_results.get(&tagged_id).cloned() {
                    step_results.insert(bare_id.to_owned(), latest);
                } else {
                    step_results.remove(bare_id);
                }
            }
        }

        completed = iteration;

        // Evaluate `until` after the iteration body. Stages snapshot
        // is included so a pipeline-aware loop body can branch on
        // upstream stage outputs (issue #41).
        let stages_snapshot = ctx.stages.read().await.clone();
        let inner_ctx = build_eval_ctx(ctx, step_results, &stages_snapshot);
        let cond = expr::eval_bool(until, &inner_ctx).map_err(|e| WorkflowError::StepFailed {
            step_id: id.clone(),
            message: format!("until-expression error: {e}"),
        })?;
        if cond {
            hit_until = true;
            break;
        }
    }

    let result = StepResult {
        output: json!({
            "max_iterations_reached": !hit_until,
            "iterations": completed,
        }),
        exit_code: i32::from(!hit_until),
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
