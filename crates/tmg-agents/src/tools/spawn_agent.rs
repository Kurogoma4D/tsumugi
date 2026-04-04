//! The `spawn_agent` tool: spawns a subagent from the main agent loop.
//!
//! Parameters:
//! - `agent_type` (string, required): one of `"explore"`, `"worker"`, `"plan"`
//! - `task` (string, required): the task description for the subagent
//! - `background` (boolean, optional, default: `false`): whether to run
//!   in the background

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::Mutex;

use tmg_tools::ToolError;
use tmg_tools::types::{Tool, ToolResult};

use crate::config::{AgentType, SubagentConfig};
use crate::manager::SubagentManager;

/// A tool that spawns subagents from the main agent loop.
///
/// When `background` is `false` (default), the tool acquires the
/// manager lock to spawn (obtaining the ID and a oneshot receiver),
/// then **drops the lock** and awaits the oneshot. This avoids holding
/// the lock for the entire subagent lifetime, which would freeze the
/// TUI's periodic `refresh_subagent_summaries` calls.
///
/// When `background` is `true`, the tool spawns the subagent, returns
/// its ID immediately, and the subagent runs concurrently.
pub struct SpawnAgentTool {
    /// Shared reference to the subagent manager.
    manager: Arc<Mutex<SubagentManager>>,
}

impl SpawnAgentTool {
    /// Create a new `spawn_agent` tool backed by the given manager.
    pub fn new(manager: Arc<Mutex<SubagentManager>>) -> Self {
        Self { manager }
    }
}

impl Tool for SpawnAgentTool {
    fn name(&self) -> &'static str {
        "spawn_agent"
    }

    fn description(&self) -> &'static str {
        "Spawn a subagent to perform a task. Available types: \
         'explore' (read-only codebase exploration), \
         'worker' (full tool access), \
         'plan' (read-only planning). \
         Set background=true to run asynchronously."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "agent_type": {
                    "type": "string",
                    "enum": ["explore", "worker", "plan"],
                    "description": "The type of subagent to spawn."
                },
                "task": {
                    "type": "string",
                    "description": "The task description for the subagent."
                },
                "background": {
                    "type": "boolean",
                    "description": "If true, the subagent runs in the background and the tool returns immediately with an ID. Default: false.",
                    "default": false
                }
            },
            "required": ["agent_type", "task"]
        })
    }

    fn execute(
        &self,
        params: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult, ToolError>> + Send + '_>> {
        Box::pin(async move {
            let agent_type_str = params
                .get("agent_type")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| ToolError::InvalidParams {
                    message: "missing required parameter: agent_type".to_owned(),
                })?;

            let agent_type =
                AgentType::from_name(agent_type_str).ok_or_else(|| ToolError::InvalidParams {
                    message: format!(
                        "unknown agent_type: {agent_type_str}. \
                         Valid types: explore, worker, plan"
                    ),
                })?;

            let task = params
                .get("task")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| ToolError::InvalidParams {
                    message: "missing required parameter: task".to_owned(),
                })?
                .to_owned();

            let background = params
                .get("background")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);

            let config = SubagentConfig {
                agent_type,
                task,
                background,
            };

            if background {
                // Background: acquire lock, spawn, release lock, return ID.
                let id = {
                    let mut manager = self.manager.lock().await;
                    manager.spawn(config).await
                };
                Ok(ToolResult::success(format!(
                    "Subagent spawned in background with ID {id}. \
                     It will run concurrently and results will be \
                     available when it completes."
                )))
            } else {
                // Foreground: acquire lock, spawn with oneshot notification,
                // release lock, then await the oneshot receiver. This
                // ensures the TUI can still acquire the lock for
                // `refresh_subagent_summaries` while the subagent runs.
                let (_id, rx) = {
                    let mut manager = self.manager.lock().await;
                    manager.spawn_with_notify(config).await
                };
                // Lock is now dropped -- the TUI can freely poll summaries.

                match rx.await {
                    Ok(Ok(output)) => Ok(ToolResult::success(output)),
                    Ok(Err(e)) => Ok(ToolResult::error(format!("Subagent failed: {e}"))),
                    Err(_) => Ok(ToolResult::error(
                        "Subagent task dropped without producing a result".to_owned(),
                    )),
                }
            }
        })
    }
}
