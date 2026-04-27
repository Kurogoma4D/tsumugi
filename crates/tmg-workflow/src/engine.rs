//! Workflow executor with full control-flow support.
//!
//! `WorkflowEngine` owns the shared resources needed by step handlers
//! ([`tmg_llm::LlmPool`], [`tmg_sandbox::SandboxContext`],
//! [`tmg_tools::ToolRegistry`], [`tmg_agents::SubagentManager`]) and
//! drives the workflow steps recursively. Each leaf step (`agent`,
//! `shell`, `write_file`) executes through its dedicated handler in
//! [`crate::steps`]; control-flow steps (`loop`, `branch`, `parallel`,
//! `group`, `human`) dispatch back through the engine on their child
//! steps.
//!
//! ## Snapshot map for `revise` rewinds
//!
//! Before running every top-level step the engine snapshots the
//! *current* `step_results` map (a `BTreeMap<String, StepResult>`) and
//! stores it by step id in a side `snapshots` map. When a `human` step
//! returns `revise { target: <step_id> }`, the engine restores
//! `step_results = snapshots[<step_id>]` and re-enters dispatch from
//! the matching outer step.
//!
//! On every revise rewind we also *prune* `snapshots` of any entries
//! belonging to indices `>= target_idx`. Those steps are about to
//! re-run and will re-insert their own entries, so the prior snapshots
//! would otherwise pile up across rewinds. With the
//! pruning the snapshot map is bounded by `top_level_steps`
//! independent of how many rewinds happen — the safety budget of 64
//! rewinds is purely a runaway-revise guard, not a memory bound.
//!
//! The per-snapshot memory cost is `O(leaf_results_so_far)`. For
//! typical workflows (≤ tens of steps with small JSON outputs) this is
//! well under a megabyte and is *not* released until the engine
//! returns. Long-running workflows that emit large step outputs should
//! prefer placing `human` steps near the start so the snapshots stay
//! small.
//!
//! ## Cancellation safety
//!
//! Top-level `run` is cancel-safe at await points between leaf-step
//! dispatches. Parallel steps own a private [`CancellationToken`] that
//! the engine fires when any child fails so siblings observe the
//! cancellation deterministically.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use serde_json::Value;
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio_util::sync::CancellationToken;

use tmg_agents::SubagentManager;
use tmg_llm::LlmPool;
use tmg_sandbox::SandboxContext;
use tmg_tools::ToolRegistry;

use crate::config::WorkflowConfig;
use crate::def::{InputDef, StepDef, StepResult, WorkflowDef, WorkflowOutputs};
use crate::error::{Result, WorkflowError};
use crate::expr;
use crate::progress::WorkflowProgress;
use crate::steps;

/// Shared index of discovered workflows, keyed by id.
///
/// Owned by the [`WorkflowEngine`] and shared with the workflow tools
/// (`run_workflow`, `workflow_status`). The CLI populates the map once
/// after discovery; runtime additions are not required for the
/// pipeline form but the `RwLock` leaves the door open without a
/// breaking API change.
pub type WorkflowIndex = Arc<RwLock<HashMap<String, WorkflowDef>>>;

/// Shared, cheaply-cloned engine resources passed down through every
/// recursive dispatch.
///
/// We hold these by `Arc` (the engine owns the original `Arc`) so a
/// `parallel` step can `clone` cheap handles into spawned tasks.
#[derive(Clone)]
pub(crate) struct EngineCtx {
    /// Sandbox used for shell commands and `write_file` write checks.
    pub(crate) sandbox: Arc<SandboxContext>,
    /// Subagent manager used by agent steps.
    pub(crate) subagent_manager: Arc<Mutex<SubagentManager>>,
    /// Engine-level configuration (timeouts, default model, ...).
    pub(crate) config: WorkflowConfig,
    /// `tsumugi.toml`-derived configuration as JSON.
    pub(crate) config_json: Value,
    /// Snapshot of the process environment, captured once per run.
    pub(crate) env: Arc<BTreeMap<String, String>>,
    /// Workflow inputs, captured once per run.
    pub(crate) inputs: Arc<Value>,
    /// Progress sender. Cloned into spawned tasks when needed.
    pub(crate) progress_tx: mpsc::Sender<WorkflowProgress>,
    /// Semaphore capping concurrent agent steps. Counted only for
    /// agent-type leaves; `shell`/`write_file` run uncapped (issue #40
    /// SPEC: "agent ステップだけがカウント対象").
    pub(crate) agent_semaphore: Arc<tokio::sync::Semaphore>,
    /// Workflow-level cancellation token. Currently only observed by
    /// the `human` step (which would otherwise block forever waiting
    /// for a UI response that never comes). Reserved for broader
    /// graceful-shutdown plumbing in #42.
    pub(crate) cancel: CancellationToken,
    /// Shared workflow index (issue #41). The pipeline `workflow:`
    /// step looks up its target workflow here; `None` for engines that
    /// were built before workflow discovery (in which case any
    /// `workflow:` step fails with a clear error).
    pub(crate) workflow_index: Option<WorkflowIndex>,
    /// Per-stage outputs collected during a pipeline run. Populated
    /// only when the outer workflow uses `stages:`. Each completed
    /// `Workflow` step adds an entry keyed by its stage id.
    pub(crate) stages: Arc<RwLock<BTreeMap<String, WorkflowOutputs>>>,
    /// Recursion depth for `Workflow` step dispatch. Bumped on every
    /// nested `WorkflowEngine::run` call to prevent runaway recursion
    /// when a pipeline misconfigures stage references.
    pub(crate) workflow_depth: u32,
}

/// Workflow executor.
///
/// Constructed once per CLI run and reused across multiple workflow
/// invocations.
pub struct WorkflowEngine {
    /// Shared LLM pool (reserved for future use; agent steps currently
    /// route through [`SubagentManager`] which owns its own client).
    #[expect(
        dead_code,
        reason = "kept on the struct so the engine API matches SPEC §8.10 and so wiring can be re-used when the long-running mode lands (#42)"
    )]
    llm_pool: Arc<LlmPool>,
    sandbox: Arc<SandboxContext>,
    /// Tool registry. Reserved for built-in tool calls *from the engine*
    /// (currently unused; the `run_workflow` tool sits at the LLM
    /// surface, not inside the engine).
    #[expect(
        dead_code,
        reason = "kept on the struct so the engine API matches SPEC §8.10 and so future built-in tool dispatches from inside the engine remain a non-breaking change"
    )]
    tool_registry: Arc<ToolRegistry>,
    subagent_manager: Arc<Mutex<SubagentManager>>,
    config: WorkflowConfig,
    config_json: Value,
    /// Optional workflow index — populated by the CLI after
    /// discovery. The pipeline `workflow:` step reads from here to
    /// resolve its target. Engines built without an index reject any
    /// `workflow:` step with a clear error.
    workflow_index: Option<WorkflowIndex>,
}

/// Outcome of executing a single step (leaf or control-flow).
///
/// `Revise` short-circuits all the way back up to `run` so the engine
/// can rewind. The carrier is the *outer* target step id; the
/// rewinder restores the snapshot taken before that step ran.
#[derive(Debug)]
pub(crate) enum StepOutcome {
    /// Step completed; the result has already been written to the
    /// shared `step_results` map.
    Completed,
    /// Step is requesting a workflow-wide rewind to `target`.
    Revise { target: String },
}

/// Recursion depth ceiling for nested `Workflow` step dispatch.
///
/// Pipelines that reference each other in a cycle would loop forever
/// without this guard. The bound is intentionally loose (32) — real
/// pipelines compose at most a handful of layers in practice; anything
/// deeper than this is almost certainly a misconfiguration.
pub(crate) const MAX_WORKFLOW_DEPTH: u32 = 32;

/// Run a workflow as a *nested* invocation from inside a `Workflow`
/// step. Returns the final [`WorkflowOutputs`].
///
/// This re-implements the `WorkflowEngine::run` body inline (modulo
/// argument plumbing) so the nested call can:
///
/// - Inherit the *parent's* progress channel — the LLM and TUI see one
///   continuous event stream regardless of pipeline depth.
/// - Inherit the parent's cancellation token (cancelling the outer
///   workflow cancels every nested one).
/// - Bump `workflow_depth` so the recursion guard fires.
///
/// The nested call still gets its own `step_results` and `snapshots`
/// maps — stage isolation is the whole point of the pipeline form.
pub(crate) async fn run_nested_workflow(
    parent_ctx: &EngineCtx,
    workflow: &WorkflowDef,
    inputs: BTreeMap<String, Value>,
) -> Result<WorkflowOutputs> {
    if parent_ctx.workflow_depth >= MAX_WORKFLOW_DEPTH {
        return Err(WorkflowError::StepFailed {
            step_id: workflow.id.clone(),
            message: format!(
                "workflow recursion too deep (>{MAX_WORKFLOW_DEPTH} nested calls); check for cyclic pipeline references"
            ),
        });
    }

    let resolved_inputs = resolve_inputs(&workflow.inputs, inputs)?;
    let inputs_value = Value::Object(resolved_inputs);

    let nested_ctx = EngineCtx {
        sandbox: Arc::clone(&parent_ctx.sandbox),
        subagent_manager: Arc::clone(&parent_ctx.subagent_manager),
        config: parent_ctx.config.clone(),
        config_json: parent_ctx.config_json.clone(),
        env: Arc::clone(&parent_ctx.env),
        inputs: Arc::new(inputs_value),
        progress_tx: parent_ctx.progress_tx.clone(),
        agent_semaphore: Arc::clone(&parent_ctx.agent_semaphore),
        cancel: parent_ctx.cancel.clone(),
        workflow_index: parent_ctx.workflow_index.clone(),
        // Each nested run gets its own stages map. Pipeline outputs
        // visibility flows top-down via the parent's stages map (the
        // outer workflow renders its outputs after inner pipelines
        // complete); a nested run does not see its parent's stages
        // because the parent stages would change semantics
        // mid-iteration in the pipeline form. If we ever need
        // cross-level stage visibility we'll add it explicitly with a
        // borrow rather than a clone.
        stages: Arc::new(RwLock::new(BTreeMap::new())),
        workflow_depth: parent_ctx.workflow_depth + 1,
    };

    let mut step_results: BTreeMap<String, StepResult> = BTreeMap::new();
    let mut snapshots: BTreeMap<String, BTreeMap<String, StepResult>> = BTreeMap::new();
    let total_steps = workflow.steps.len();
    let mut idx: usize = 0;
    let mut rewind_budget: u32 = 64;
    while idx < total_steps {
        let step = &workflow.steps[idx];
        let step_id = step.id().to_owned();
        snapshots.insert(step_id.clone(), step_results.clone());
        let outcome = dispatch_step(&nested_ctx, step, &mut step_results).await?;
        match outcome {
            StepOutcome::Completed => idx += 1,
            StepOutcome::Revise { target } => {
                if rewind_budget == 0 {
                    return Err(WorkflowError::StepFailed {
                        step_id,
                        message: "revise rewind budget exhausted (>64 rewinds in one run)"
                            .to_owned(),
                    });
                }
                rewind_budget -= 1;
                let Some(target_idx) = workflow.steps.iter().position(|s| s.id() == target) else {
                    return Err(WorkflowError::StepFailed {
                        step_id,
                        message: format!("revise target '{target}' is not a top-level step"),
                    });
                };
                let Some(snap) = snapshots.get(&target).cloned() else {
                    return Err(WorkflowError::StepFailed {
                        step_id,
                        message: format!("no snapshot recorded for revise target '{target}'"),
                    });
                };
                step_results = snap;
                let stale: Vec<String> = workflow
                    .steps
                    .iter()
                    .skip(target_idx)
                    .map(|s| s.id().to_owned())
                    .collect();
                for sid in stale {
                    snapshots.remove(&sid);
                }
                idx = target_idx;
            }
        }
    }

    let stages_snapshot = nested_ctx.stages.read().await.clone();
    let mut output_values: BTreeMap<String, String> = BTreeMap::new();
    for (name, template) in &workflow.outputs {
        let inner_ctx = expr::ExprContext::new(
            &nested_ctx.inputs,
            &step_results,
            &nested_ctx.config_json,
            &nested_ctx.env,
        )
        .with_stages(&stages_snapshot);
        let rendered = expr::eval_string(template, &inner_ctx)?;
        output_values.insert(name.clone(), rendered);
    }
    let outputs = WorkflowOutputs {
        values: output_values,
    };
    let _ = nested_ctx
        .progress_tx
        .send(WorkflowProgress::WorkflowCompleted {
            outputs: outputs.clone(),
        })
        .await;
    Ok(outputs)
}

impl WorkflowEngine {
    /// Build a new engine.
    ///
    /// `config_json` is the JSON projection of the full `TsumugiConfig`
    /// (or any subset the caller wishes to expose). The engine does
    /// not introspect this value beyond passing it to the expression
    /// evaluator; supplying `Value::Null` is valid when no
    /// `${{ config.* }}` expressions are expected.
    pub fn new(
        llm_pool: Arc<LlmPool>,
        sandbox: Arc<SandboxContext>,
        tool_registry: Arc<ToolRegistry>,
        subagent_manager: Arc<Mutex<SubagentManager>>,
        config: WorkflowConfig,
        config_json: Value,
    ) -> Self {
        Self {
            llm_pool,
            sandbox,
            tool_registry,
            subagent_manager,
            config,
            config_json,
            workflow_index: None,
        }
    }

    /// Attach a [`WorkflowIndex`] so pipeline `workflow:` steps can
    /// resolve their targets. Returns `self` for builder-style chaining
    /// at construction time.
    #[must_use]
    pub fn with_workflow_index(mut self, index: WorkflowIndex) -> Self {
        self.workflow_index = Some(index);
        self
    }

    /// Return a clone of the workflow index handle, if any. Used by
    /// the workflow tools (`run_workflow`) to share the same map the
    /// engine uses internally.
    #[must_use]
    pub fn workflow_index_handle(&self) -> Option<WorkflowIndex> {
        self.workflow_index.clone()
    }

    /// Run the given workflow with the given input map.
    ///
    /// The supplied `progress_tx` receives one [`WorkflowProgress`]
    /// event per state transition. Events are sent best-effort: a
    /// receiver drop is *not* treated as an error so workflows can run
    /// to completion even when the consumer (e.g. TUI) has gone away.
    ///
    /// # Cancel safety
    ///
    /// This future is cancel-safe at await points between top-level
    /// steps. Spawned `parallel` children honour an internal
    /// `CancellationToken`; firing the token mid-dispatch will return
    /// promptly with a [`WorkflowError::StepFailed`].
    pub async fn run(
        &self,
        workflow: &WorkflowDef,
        inputs: BTreeMap<String, Value>,
        progress_tx: mpsc::Sender<WorkflowProgress>,
    ) -> Result<WorkflowOutputs> {
        // Resolve inputs (apply defaults / required check).
        let resolved_inputs = resolve_inputs(&workflow.inputs, inputs)?;
        let inputs_value = Value::Object(resolved_inputs);

        // Snapshot the environment once. Workflows must not see live
        // env mutation across steps — this matches the §8.7 isolation
        // ethos for agent contexts and gives deterministic re-runs.
        let env: BTreeMap<String, String> = std::env::vars().collect();

        // Cap agent concurrency at the configured ceiling. Permits are
        // acquired by agent leaf-step handlers; shell/write_file
        // leaves bypass the semaphore (per SPEC §8.4).
        let agent_semaphore = Arc::new(tokio::sync::Semaphore::new(
            self.config
                .max_parallel_agents
                .try_into()
                .unwrap_or(1)
                .max(1),
        ));

        let ctx = EngineCtx {
            sandbox: Arc::clone(&self.sandbox),
            subagent_manager: Arc::clone(&self.subagent_manager),
            config: self.config.clone(),
            config_json: self.config_json.clone(),
            env: Arc::new(env),
            inputs: Arc::new(inputs_value),
            progress_tx,
            agent_semaphore,
            cancel: CancellationToken::new(),
            workflow_index: self.workflow_index.clone(),
            stages: Arc::new(RwLock::new(BTreeMap::new())),
            workflow_depth: 0,
        };

        let mut step_results: BTreeMap<String, StepResult> = BTreeMap::new();
        // `snapshots[step_id]` holds the `step_results` map *as it was
        // immediately before* `step_id` started executing. On a
        // `revise` decision we restore this map and re-enter dispatch
        // at that step.
        let mut snapshots: BTreeMap<String, BTreeMap<String, StepResult>> = BTreeMap::new();

        // Main loop with revise-induced rewind support. We track the
        // index in workflow.steps so a revise can reset it; a defensive
        // `rewind_budget` prevents pathological human-loops.
        let total_steps = workflow.steps.len();
        let mut idx: usize = 0;
        let mut rewind_budget: u32 = 64;
        while idx < total_steps {
            let step = &workflow.steps[idx];
            let step_id = step.id().to_owned();
            // Snapshot before execution.
            snapshots.insert(step_id.clone(), step_results.clone());

            let outcome = dispatch_step(&ctx, step, &mut step_results).await?;

            match outcome {
                StepOutcome::Completed => idx += 1,
                StepOutcome::Revise { target } => {
                    if rewind_budget == 0 {
                        return Err(WorkflowError::StepFailed {
                            step_id: step_id.clone(),
                            message: "revise rewind budget exhausted (>64 rewinds in one run)"
                                .to_owned(),
                        });
                    }
                    rewind_budget -= 1;
                    let Some(target_idx) = workflow.steps.iter().position(|s| s.id() == target)
                    else {
                        return Err(WorkflowError::StepFailed {
                            step_id,
                            message: format!(
                                "revise target '{target}' is not a top-level step in this workflow"
                            ),
                        });
                    };
                    let Some(snap) = snapshots.get(&target).cloned() else {
                        return Err(WorkflowError::StepFailed {
                            step_id,
                            message: format!(
                                "no snapshot recorded for revise target '{target}' (target must have run already)"
                            ),
                        });
                    };
                    step_results = snap;
                    // Drop snapshots for steps at or beyond
                    // `target_idx`; they are about to re-run and will
                    // re-insert their own entries on the next loop
                    // turn. Without this, repeated revises across a
                    // long workflow would let `snapshots` grow
                    // unbounded.
                    let stale_ids: Vec<String> = workflow
                        .steps
                        .iter()
                        .skip(target_idx)
                        .map(|s| s.id().to_owned())
                        .collect();
                    for sid in stale_ids {
                        snapshots.remove(&sid);
                    }
                    idx = target_idx;
                }
            }
        }

        // Render outputs. Pipeline workflows (those that completed
        // `Workflow` steps) need the `stages` scope visible here so
        // `outputs:` templates can pick stage outputs back out.
        let stages_snapshot = ctx.stages.read().await.clone();
        let mut output_values: BTreeMap<String, String> = BTreeMap::new();
        for (name, template) in &workflow.outputs {
            let inner_ctx =
                expr::ExprContext::new(&ctx.inputs, &step_results, &ctx.config_json, &ctx.env)
                    .with_stages(&stages_snapshot);
            let rendered = expr::eval_string(template, &inner_ctx)?;
            output_values.insert(name.clone(), rendered);
        }
        let outputs = WorkflowOutputs {
            values: output_values,
        };
        let _ = ctx
            .progress_tx
            .send(WorkflowProgress::WorkflowCompleted {
                outputs: outputs.clone(),
            })
            .await;

        Ok(outputs)
    }
}

/// Dispatch a single step (leaf or control-flow) into the appropriate
/// handler. Updates `step_results` in place on success.
///
/// Returns a boxed `Send` future so recursive control-flow paths
/// (loop / branch / parallel / group) can `Box::pin` the recursion
/// without the compiler losing track of the `Send` bound across
/// `tokio::spawn` boundaries.
pub(crate) fn dispatch_step<'a>(
    ctx: &'a EngineCtx,
    step: &'a StepDef,
    step_results: &'a mut BTreeMap<String, StepResult>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<StepOutcome>> + Send + 'a>> {
    Box::pin(dispatch_step_inner(ctx, step, step_results))
}

#[expect(
    clippy::too_many_lines,
    reason = "linear per-variant dispatch; splitting would scatter the leaf-step matrix across helpers and obscure the StepOutcome flow"
)]
async fn dispatch_step_inner(
    ctx: &EngineCtx,
    step: &StepDef,
    step_results: &mut BTreeMap<String, StepResult>,
) -> Result<StepOutcome> {
    let step_id = step.id().to_owned();
    let step_type = step.step_type();

    // Evaluate `when` clause for leaf steps that support it.
    if let Some(expr_src) = step.when() {
        let stages_snapshot = ctx.stages.read().await.clone();
        let eval_ctx =
            expr::ExprContext::new(&ctx.inputs, step_results, &ctx.config_json, &ctx.env)
                .with_stages(&stages_snapshot);
        let cond = expr::eval_bool(expr_src, &eval_ctx).map_err(|e| WorkflowError::StepFailed {
            step_id: step_id.clone(),
            message: format!("when-expression error: {e}"),
        })?;
        if !cond {
            tracing::debug!(step_id, "skipping step: when=false");
            return Ok(StepOutcome::Completed);
        }
    }

    let _ = ctx
        .progress_tx
        .send(WorkflowProgress::StepStarted {
            step_id: step_id.clone(),
            step_type,
        })
        .await;

    let exec_result = match step {
        StepDef::Agent {
            id: _,
            subagent,
            prompt,
            model,
            when: _,
            inject_files,
        } => {
            // Acquire agent permit; held only for the duration of this
            // call. We must not hold the permit across recursive
            // control-flow boundaries, so an agent leaf is the only
            // place that takes one. Closing the semaphore is not
            // expected here, so `acquire().await` returning an error
            // would only happen if the engine itself dropped it — we
            // map that to a clear `StepFailed`.
            let permit = ctx
                .agent_semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|_| WorkflowError::StepFailed {
                    step_id: step_id.clone(),
                    message: "agent semaphore was closed".to_owned(),
                })?;
            let stages_snapshot = ctx.stages.read().await.clone();
            let eval_ctx =
                expr::ExprContext::new(&ctx.inputs, step_results, &ctx.config_json, &ctx.env)
                    .with_stages(&stages_snapshot);
            let result = steps::agent::execute(steps::agent::AgentStepArgs {
                subagent_manager: &ctx.subagent_manager,
                sandbox: &ctx.sandbox,
                step_id: &step_id,
                subagent_name: subagent,
                prompt_template: prompt,
                model: model.as_deref(),
                inject_files,
                ctx: &eval_ctx,
            })
            .await;
            drop(permit);
            result
        }
        StepDef::Shell {
            id: _,
            command,
            timeout,
            when: _,
        } => {
            let stages_snapshot = ctx.stages.read().await.clone();
            let eval_ctx =
                expr::ExprContext::new(&ctx.inputs, step_results, &ctx.config_json, &ctx.env)
                    .with_stages(&stages_snapshot);
            steps::shell::execute(
                &ctx.sandbox,
                ctx.config.default_shell_timeout,
                command,
                *timeout,
                &eval_ctx,
            )
            .await
        }
        StepDef::WriteFile {
            id: _,
            path,
            content,
        } => {
            let stages_snapshot = ctx.stages.read().await.clone();
            let eval_ctx =
                expr::ExprContext::new(&ctx.inputs, step_results, &ctx.config_json, &ctx.env)
                    .with_stages(&stages_snapshot);
            steps::write_file::execute(&ctx.sandbox, path, content, &eval_ctx).await
        }
        StepDef::Loop { .. } => {
            return retag_control_flow_error(
                steps::loop_step::execute(ctx, step, step_results).await,
                &step_id,
            );
        }
        StepDef::Branch { .. } => {
            return retag_control_flow_error(
                steps::branch::execute(ctx, step, step_results).await,
                &step_id,
            );
        }
        StepDef::Parallel { .. } => {
            return retag_control_flow_error(
                steps::parallel::execute(ctx, step, step_results).await,
                &step_id,
            );
        }
        StepDef::Group { .. } => {
            return retag_control_flow_error(
                steps::group::execute(ctx, step, step_results).await,
                &step_id,
            );
        }
        StepDef::Human { .. } => {
            return retag_control_flow_error(
                steps::human::execute(ctx, step, step_results).await,
                &step_id,
            );
        }
        StepDef::Workflow { .. } => {
            return retag_control_flow_error(
                steps::workflow::execute(ctx, step, step_results).await,
                &step_id,
            );
        }
    };

    // Re-tag every error returned by a leaf-step handler with the
    // current `step_id` so the returned `WorkflowError` and the
    // `StepFailed` progress event always agree.
    let exec_result = exec_result.map_err(|e| match e {
        WorkflowError::StepFailed { message, .. } => WorkflowError::StepFailed {
            step_id: step_id.clone(),
            message,
        },
        other => WorkflowError::StepFailed {
            step_id: step_id.clone(),
            message: other.to_string(),
        },
    });

    match exec_result {
        Ok(result) => {
            let _ = ctx
                .progress_tx
                .send(WorkflowProgress::StepCompleted {
                    step_id: step_id.clone(),
                    result: result.clone(),
                })
                .await;
            step_results.insert(step_id, result);
            Ok(StepOutcome::Completed)
        }
        Err(err) => {
            let _ = ctx
                .progress_tx
                .send(WorkflowProgress::StepFailed {
                    step_id: step_id.clone(),
                    error: err.to_string(),
                })
                .await;
            Err(err)
        }
    }
}

/// Re-tag an error returned by a control-flow handler so the propagated
/// `WorkflowError` carries the control-flow step's own id (matching
/// the `StepFailed` progress event the handler emitted on the same id
/// — see fix #2 in PR review for #64). This keeps the error stream and
/// the progress stream in agreement: the deepest leaf's `StepFailed`
/// already named the leaf, and the outer control-flow's `StepFailed`
/// names the outer step; the returned `WorkflowError` follows the
/// outer name.
fn retag_control_flow_error(result: Result<StepOutcome>, outer_id: &str) -> Result<StepOutcome> {
    result.map_err(|e| match e {
        WorkflowError::StepFailed { message, .. } => WorkflowError::StepFailed {
            step_id: outer_id.to_owned(),
            message,
        },
        other => WorkflowError::StepFailed {
            step_id: outer_id.to_owned(),
            message: other.to_string(),
        },
    })
}

/// Validate caller-supplied inputs against the workflow's declared
/// input list, applying defaults and rejecting required-but-missing
/// inputs.
fn resolve_inputs(
    declared: &BTreeMap<String, InputDef>,
    mut supplied: BTreeMap<String, Value>,
) -> Result<serde_json::Map<String, Value>> {
    let mut out: serde_json::Map<String, Value> = serde_json::Map::new();

    for (name, def) in declared {
        let value = if let Some(v) = supplied.remove(name) {
            v
        } else if let Some(default) = &def.default {
            default.clone()
        } else if def.required {
            return Err(WorkflowError::MissingInput { name: name.clone() });
        } else {
            Value::Null
        };
        out.insert(name.clone(), value);
    }

    // Permit extra inputs (forward compatibility): pass them through
    // unchanged so a workflow author can add an input without
    // breaking older callers that already supply it.
    for (name, value) in supplied {
        out.insert(name, value);
    }

    Ok(out)
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions")]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn resolve_inputs_applies_defaults() {
        let mut declared = BTreeMap::new();
        declared.insert(
            "name".to_owned(),
            InputDef {
                r#type: "string".to_owned(),
                default: Some(json!("anon")),
                required: false,
                description: None,
            },
        );
        let resolved =
            resolve_inputs(&declared, BTreeMap::new()).unwrap_or_else(|e| panic!("resolve: {e}"));
        assert_eq!(resolved.get("name"), Some(&json!("anon")));
    }

    #[test]
    fn resolve_inputs_rejects_missing_required() {
        let mut declared = BTreeMap::new();
        declared.insert(
            "x".to_owned(),
            InputDef {
                r#type: "string".to_owned(),
                default: None,
                required: true,
                description: None,
            },
        );
        let result = resolve_inputs(&declared, BTreeMap::new());
        assert!(matches!(result, Err(WorkflowError::MissingInput { .. })));
    }

    #[test]
    fn resolve_inputs_passes_through_extras() {
        let declared = BTreeMap::new();
        let mut supplied = BTreeMap::new();
        supplied.insert("extra".to_owned(), json!(1));
        let resolved =
            resolve_inputs(&declared, supplied).unwrap_or_else(|e| panic!("resolve: {e}"));
        assert_eq!(resolved.get("extra"), Some(&json!(1)));
    }
}
