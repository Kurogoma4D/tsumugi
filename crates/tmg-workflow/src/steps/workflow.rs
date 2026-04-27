//! `workflow` step handler (issue #41 — pipeline form).
//!
//! Each pipeline `stages:` entry materialises into a [`StepDef::Workflow`]
//! and is dispatched here. The handler:
//!
//! 1. Looks up the target workflow in the engine's
//!    [`crate::engine::WorkflowIndex`].
//! 2. Renders the templated `inputs:` map against the current
//!    [`crate::expr::ExprContext`] (with `stages` visible so a stage can
//!    forward upstream outputs).
//! 3. Calls [`crate::engine::run_nested_workflow`] to execute the
//!    target workflow with those inputs.
//! 4. Stores the result both as a [`StepResult`] (so `${{ steps.<id>.* }}`
//!    works) and as a [`crate::def::WorkflowOutputs`] entry under the
//!    stage id (so `${{ stages.<id>.outputs.<key> }}` works).
//!
//! When the step carries a `loop_spec`, the handler iterates the
//! invocation up to `max_iterations` times, re-evaluating `until` after
//! each call — this mirrors the [`StepDef::Loop`] semantics for a single
//! workflow target without wrapping it in an outer loop step.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::def::{LoopSpec, StepDef, StepResult, WorkflowOutputs};
use crate::engine::{EngineCtx, StepOutcome, build_eval_ctx, run_nested_workflow};
use crate::error::{Result, WorkflowError};
use crate::expr;
use crate::progress::WorkflowProgress;

pub(crate) async fn execute(
    ctx: &EngineCtx,
    step: &StepDef,
    step_results: &mut BTreeMap<String, StepResult>,
) -> Result<StepOutcome> {
    let StepDef::Workflow {
        id,
        workflow_id,
        inputs,
        loop_spec,
    } = step
    else {
        return Err(WorkflowError::StepFailed {
            step_id: step.id().to_owned(),
            message: "internal error: workflow::execute called with non-workflow step".to_owned(),
        });
    };

    // Resolve the target workflow from the engine's index.
    let Some(index) = &ctx.workflow_index else {
        return Err(WorkflowError::StepFailed {
            step_id: id.clone(),
            message: format!(
                "workflow step '{id}' references workflow '{workflow_id}' but no workflow index is attached to the engine; \
                 build the engine via `WorkflowEngine::with_workflow_index(...)` after discovery"
            ),
        });
    };
    let target = {
        let guard = index.read().await;
        guard.get(workflow_id).cloned()
    };
    let Some(target) = target else {
        return Err(WorkflowError::StepFailed {
            step_id: id.clone(),
            message: format!("workflow step '{id}' references unknown workflow '{workflow_id}'"),
        });
    };

    if let Some(spec) = loop_spec {
        run_with_loop(ctx, id, &target, inputs, spec, step_results).await
    } else {
        run_once(ctx, id, &target, inputs, step_results).await
    }
}

/// Dispatch a single (non-looped) `Workflow` step.
async fn run_once(
    ctx: &EngineCtx,
    stage_id: &str,
    target: &crate::def::WorkflowDef,
    inputs_template: &BTreeMap<String, String>,
    step_results: &mut BTreeMap<String, StepResult>,
) -> Result<StepOutcome> {
    let resolved_inputs = render_inputs(ctx, stage_id, inputs_template, step_results).await?;
    let outputs = run_nested_workflow(ctx, target, resolved_inputs).await?;
    record_stage_result(ctx, stage_id, &outputs, step_results).await;
    Ok(StepOutcome::Completed)
}

/// Dispatch a `Workflow` step wrapped in an inline `loop:` block.
///
/// We re-render `inputs` on every iteration so templates can refer to
/// the previous iteration's stage outputs (`${{ stages.<stage>.outputs.* }}`).
/// `until` is evaluated after each iteration body and stops the loop on
/// truthy.
async fn run_with_loop(
    ctx: &EngineCtx,
    stage_id: &str,
    target: &crate::def::WorkflowDef,
    inputs_template: &BTreeMap<String, String>,
    spec: &LoopSpec,
    step_results: &mut BTreeMap<String, StepResult>,
) -> Result<StepOutcome> {
    let mut completed = 0u32;
    let mut hit_until = false;
    for iteration in 1..=spec.max_iterations {
        let _ = ctx
            .progress_tx
            .send(WorkflowProgress::LoopIteration {
                step_id: stage_id.to_owned(),
                iteration,
                max: spec.max_iterations,
            })
            .await;

        let resolved_inputs = render_inputs(ctx, stage_id, inputs_template, step_results).await?;
        let outputs = run_nested_workflow(ctx, target, resolved_inputs).await?;
        record_stage_result(ctx, stage_id, &outputs, step_results).await;
        completed = iteration;

        // Evaluate `until` with stages visible so the typical pattern
        // `${{ stages.<id>.outputs.findings_count == 0 }}` works.
        let stages_snapshot = ctx.stages.read().await.clone();
        let inner_ctx = build_eval_ctx(ctx, step_results, &stages_snapshot);
        let cond =
            expr::eval_bool(&spec.until, &inner_ctx).map_err(|e| WorkflowError::StepFailed {
                step_id: stage_id.to_owned(),
                message: format!("until-expression error in loop: {e}"),
            })?;
        if cond {
            hit_until = true;
            break;
        }
    }

    // Loop wrapper bookkeeping mirrors `StepDef::Loop` so consumers
    // can spot a loop summary even on the workflow form.
    let summary = StepResult {
        output: json!({
            "max_iterations_reached": !hit_until,
            "iterations": completed,
            "workflow": target.id,
        }),
        exit_code: i32::from(!hit_until),
        stdout: String::new(),
        stderr: String::new(),
        changed_files: Vec::new(),
    };
    let _ = ctx
        .progress_tx
        .send(WorkflowProgress::StepCompleted {
            step_id: stage_id.to_owned(),
            result: summary.clone(),
        })
        .await;
    // We re-insert under the stage id so subsequent steps in the
    // outer workflow see the loop summary; the per-iteration
    // `record_stage_result` already wrote the most-recent
    // [`WorkflowOutputs`] into `ctx.stages`.
    step_results.insert(stage_id.to_owned(), summary);
    Ok(StepOutcome::Completed)
}

/// Render the templated input map against the current expression context.
async fn render_inputs(
    ctx: &EngineCtx,
    stage_id: &str,
    inputs_template: &BTreeMap<String, String>,
    step_results: &BTreeMap<String, StepResult>,
) -> Result<BTreeMap<String, Value>> {
    let stages_snapshot = ctx.stages.read().await.clone();
    let eval_ctx = build_eval_ctx(ctx, step_results, &stages_snapshot);
    let mut resolved: BTreeMap<String, Value> = BTreeMap::new();
    for (key, tmpl) in inputs_template {
        let rendered =
            expr::eval_string(tmpl, &eval_ctx).map_err(|e| WorkflowError::StepFailed {
                step_id: stage_id.to_owned(),
                message: format!("rendering inputs.{key}: {e}"),
            })?;
        resolved.insert(key.clone(), Value::String(rendered));
    }
    Ok(resolved)
}

/// Record a completed stage's outputs in both `ctx.stages` and
/// `step_results`.
///
/// The stage's [`StepResult::output`] is shaped as a JSON object with
/// each declared output as a string field — matching what `outputs:`
/// templates evaluate to, and giving `${{ steps.<stage>.output.<key> }}`
/// a natural shape.
async fn record_stage_result(
    ctx: &EngineCtx,
    stage_id: &str,
    outputs: &WorkflowOutputs,
    step_results: &mut BTreeMap<String, StepResult>,
) {
    {
        let mut guard = ctx.stages.write().await;
        guard.insert(stage_id.to_owned(), outputs.clone());
    }
    let mut output_obj = serde_json::Map::new();
    for (k, v) in &outputs.values {
        output_obj.insert(k.clone(), Value::String(v.clone()));
    }
    let result = StepResult {
        output: Value::Object(output_obj),
        exit_code: 0,
        stdout: String::new(),
        stderr: String::new(),
        changed_files: Vec::new(),
    };
    let _ = ctx
        .progress_tx
        .send(WorkflowProgress::StepCompleted {
            step_id: stage_id.to_owned(),
            result: result.clone(),
        })
        .await;
    step_results.insert(stage_id.to_owned(), result);
}
