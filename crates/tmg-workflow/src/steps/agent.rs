//! Agent-step handler (SPEC §8.4 / §8.7).
//!
//! Each agent step starts a *fresh* subagent conversation: the prompt
//! template (with `inject_files` content appended as a context block)
//! is the only input to the spawned agent. No history is carried
//! across steps, satisfying the §8.7 context-isolation invariant.
//!
//! ## `model` override
//!
//! The current `tmg_agents::SubagentManager::spawn` API resolves the
//! `(endpoint, model)` pair from manager defaults plus per-agent
//! overrides on `CustomAgentDef` — there is no per-spawn knob. When
//! `StepDef::Agent::model` is set we record a `tracing::warn!` so
//! operators learn the override is currently advisory; future work
//! (likely tied to issue #41 or a follow-up to #36) will surface a
//! proper per-spawn override.

use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use tokio::sync::Mutex;

use tmg_agents::{AgentKind, AgentType, SubagentConfig, SubagentManager};

use crate::def::StepResult;
use crate::error::{Result, WorkflowError};
use crate::expr;

/// Inputs for [`execute`], grouped to satisfy clippy's
/// `too_many_arguments` lint and to make the engine's call site
/// self-documenting.
pub(crate) struct AgentStepArgs<'a> {
    pub subagent_manager: &'a Arc<Mutex<SubagentManager>>,
    pub workspace: &'a Path,
    pub step_id: &'a str,
    pub subagent_name: &'a str,
    pub prompt_template: &'a str,
    pub model: Option<&'a str>,
    pub inject_files: &'a [String],
    pub ctx: &'a expr::ExprContext<'a>,
}

/// Execute an agent step.
pub(crate) async fn execute(args: AgentStepArgs<'_>) -> Result<StepResult> {
    let AgentStepArgs {
        subagent_manager,
        workspace,
        step_id,
        subagent_name,
        prompt_template,
        model,
        inject_files,
        ctx,
    } = args;

    let mut prompt = expr::eval_string(prompt_template, ctx)?;

    // Inject file contents as a system-style context block. We append
    // rather than prepend so the user-supplied prompt remains the
    // primary instruction; the file block's heading makes the
    // distinction unambiguous to the model.
    if !inject_files.is_empty() {
        let mut block = String::from("\n\n# Injected context\n");
        for rel in inject_files {
            let path = workspace.join(rel);
            let content = tokio::fs::read_to_string(&path).await.map_err(|e| {
                WorkflowError::io(format!("reading inject_file {}", path.display()), e)
            })?;
            // `write!` on a `String` cannot fail; the explicit `_` keeps
            // clippy's `format_push_string` lint happy without an
            // allocator detour.
            let _ = write!(block, "\n## {rel}\n```\n{content}\n```\n");
        }
        prompt.push_str(&block);
    }

    if let Some(m) = model {
        if !m.is_empty() {
            tracing::warn!(
                step_id,
                model = m,
                "per-step `model` override is currently advisory; SubagentManager has no per-spawn model knob (see issue #36/#41 follow-ups)"
            );
        }
    }

    let agent_kind = resolve_agent_kind(subagent_name).ok_or_else(|| {
        WorkflowError::StepFailed {
            step_id: step_id.to_owned(),
            message: format!(
                "unknown subagent '{subagent_name}' (custom-agent loading is not yet wired into the workflow engine; only built-in agent types are accepted)"
            ),
        }
    })?;

    let config = SubagentConfig {
        agent_kind,
        task: prompt,
        background: false,
    };

    // Each spawn produces an independent runner with its own
    // conversation history (see `tmg_agents::SubagentRunner::new`).
    // We hold the manager lock just long enough to spawn-with-notify
    // and immediately release it so the manager's state-tracking
    // stays accessible while we await the result.
    let rx = {
        let mut mgr = subagent_manager.lock().await;
        let (_id, rx) = mgr.spawn_with_notify(config).await?;
        rx
    };

    let final_text = rx.await.map_err(|_| WorkflowError::StepFailed {
        step_id: step_id.to_owned(),
        message: "subagent task channel was dropped before producing a result".to_owned(),
    })??;

    // Try to parse the agent's final response as JSON; if that fails,
    // wrap it as `{"text": "..."}`.
    let output = serde_json::from_str::<serde_json::Value>(final_text.trim())
        .unwrap_or_else(|_| serde_json::json!({"text": final_text.clone()}));

    Ok(StepResult {
        output,
        exit_code: 0,
        stdout: final_text,
        stderr: String::new(),
        changed_files: Vec::new(),
    })
}

/// Resolve a subagent name to an [`AgentKind`].
///
/// At present we only accept built-in agent types. Custom agents
/// (`tmg_agents::discover_custom_agents`) require the engine to load
/// the discovery list at startup; that wiring lives in the CLI
/// integration (issue #41) so this crate stays focused on the engine
/// itself. Callers requesting a custom agent name see a clear
/// `StepFailed` error.
fn resolve_agent_kind(name: &str) -> Option<AgentKind> {
    AgentType::from_name(name).map(AgentKind::Builtin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_builtin_agents() {
        assert!(matches!(
            resolve_agent_kind("explore"),
            Some(AgentKind::Builtin(AgentType::Explore))
        ));
        assert!(matches!(
            resolve_agent_kind("worker"),
            Some(AgentKind::Builtin(AgentType::Worker))
        ));
        assert!(resolve_agent_kind("unknown_agent").is_none());
    }
}
