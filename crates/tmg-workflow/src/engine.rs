//! Sequential workflow executor.
//!
//! `WorkflowEngine` owns the shared resources needed by step handlers
//! ([`tmg_llm::LlmPool`], [`tmg_sandbox::SandboxContext`],
//! [`tmg_tools::ToolRegistry`], [`tmg_agents::SubagentManager`]) and
//! drives the steps in declaration order. Per-step results are
//! captured into a `BTreeMap<String, StepResult>` and made available
//! to subsequent expressions via [`crate::expr::ExprContext`].
//!
//! This iteration is intentionally *sequential*: control-flow steps
//! (`loop`, `branch`, `parallel`, `group`, `human`) are out of scope
//! and tracked in issue #40. The engine returns at the first
//! [`crate::error::WorkflowError::StepFailed`] event after emitting
//! the corresponding progress message.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::Value;
use tokio::sync::{Mutex, mpsc};

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

/// Sequential workflow executor.
///
/// Constructed once per CLI run and reused across multiple workflow
/// invocations.
pub struct WorkflowEngine {
    /// Shared LLM pool (reserved for future use; agent steps currently
    /// route through [`SubagentManager`] which owns its own client).
    #[expect(
        dead_code,
        reason = "kept on the struct so the engine API matches SPEC §8.10 and so wiring can be re-used when parallel/long-running modes land (#40/#42)"
    )]
    llm_pool: Arc<LlmPool>,
    /// Sandbox used for shell commands and `write_file` write checks.
    sandbox: Arc<SandboxContext>,
    /// Tool registry. Reserved: built-in tool calls *from the engine*
    /// (e.g. `run_workflow`) will land in #41.
    #[expect(
        dead_code,
        reason = "kept on the struct so the engine API matches SPEC §8.10 and so the run_workflow tool (#41) can hook in without an API break"
    )]
    tool_registry: Arc<ToolRegistry>,
    /// Subagent manager used by agent steps.
    subagent_manager: Arc<Mutex<SubagentManager>>,
    /// Engine-level configuration (timeouts, default model, ...).
    config: WorkflowConfig,
    /// `tsumugi.toml`-derived configuration as JSON, exposed to
    /// expressions via the `config.*` scope.
    config_json: Value,
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
        }
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
    /// This future is cancel-safe at await points between steps; if
    /// dropped mid-step the underlying step handler may finish (e.g.
    /// the spawned subagent will continue until the manager's
    /// cancellation token fires). Callers that need to cancel
    /// promptly should fire the `CancellationToken` they passed to
    /// `SubagentManager::new` and `SandboxContext`.
    #[expect(
        clippy::too_many_lines,
        reason = "linear step-dispatch loop; splitting into helpers would obscure the per-step state machine"
    )]
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

        let mut steps_results: BTreeMap<String, StepResult> = BTreeMap::new();

        for step in &workflow.steps {
            let step_id = step.id().to_owned();
            let step_type = step.step_type();

            // Evaluate `when` clause, if any.
            if let Some(expr_src) = step.when() {
                let ctx =
                    expr::ExprContext::new(&inputs_value, &steps_results, &self.config_json, &env);
                let cond =
                    expr::eval_bool(expr_src, &ctx).map_err(|e| WorkflowError::StepFailed {
                        step_id: step_id.clone(),
                        message: format!("when-expression error: {e}"),
                    })?;
                if !cond {
                    tracing::debug!(step_id, "skipping step: when=false");
                    continue;
                }
            }

            // Best-effort: a closed receiver is not fatal.
            let _ = progress_tx
                .send(WorkflowProgress::StepStarted {
                    step_id: step_id.clone(),
                    step_type,
                })
                .await;

            let ctx =
                expr::ExprContext::new(&inputs_value, &steps_results, &self.config_json, &env);

            let exec_result = match step {
                StepDef::Agent {
                    id: _,
                    subagent,
                    prompt,
                    model,
                    when: _,
                    inject_files,
                } => {
                    steps::agent::execute(steps::agent::AgentStepArgs {
                        subagent_manager: &self.subagent_manager,
                        workspace: self.sandbox.workspace(),
                        step_id: &step_id,
                        subagent_name: subagent,
                        prompt_template: prompt,
                        model: model.as_deref(),
                        inject_files,
                        ctx: &ctx,
                    })
                    .await
                }
                StepDef::Shell {
                    id: _,
                    command,
                    timeout,
                    when: _,
                } => {
                    steps::shell::execute(
                        &self.sandbox,
                        self.config.default_shell_timeout,
                        command,
                        *timeout,
                        &ctx,
                    )
                    .await
                    // Tag the error with the step id for nicer messages.
                    .map_err(|e| match e {
                        WorkflowError::StepFailed { message, .. } => WorkflowError::StepFailed {
                            step_id: step_id.clone(),
                            message,
                        },
                        other => other,
                    })
                }
                StepDef::WriteFile {
                    id: _,
                    path,
                    content,
                } => steps::write_file::execute(&self.sandbox, path, content, &ctx).await,
            };

            match exec_result {
                Ok(result) => {
                    let _ = progress_tx
                        .send(WorkflowProgress::StepCompleted {
                            step_id: step_id.clone(),
                            result: result.clone(),
                        })
                        .await;
                    steps_results.insert(step_id, result);
                }
                Err(err) => {
                    let _ = progress_tx
                        .send(WorkflowProgress::StepFailed {
                            step_id: step_id.clone(),
                            error: err.to_string(),
                        })
                        .await;
                    return Err(err);
                }
            }
        }

        // Render outputs.
        let mut output_values: BTreeMap<String, String> = BTreeMap::new();
        for (name, template) in &workflow.outputs {
            let ctx =
                expr::ExprContext::new(&inputs_value, &steps_results, &self.config_json, &env);
            let rendered = expr::eval_string(template, &ctx)?;
            output_values.insert(name.clone(), rendered);
        }
        let outputs = WorkflowOutputs {
            values: output_values,
        };
        let _ = progress_tx
            .send(WorkflowProgress::WorkflowCompleted {
                outputs: outputs.clone(),
            })
            .await;

        Ok(outputs)
    }
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
