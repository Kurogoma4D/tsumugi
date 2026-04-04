//! The `spawn_agent` tool: spawns a subagent from the main agent loop.
//!
//! Parameters:
//! - `agent_type` (string, required): one of the built-in types
//!   (`"explore"`, `"worker"`, `"plan"`) or the name of a custom agent
//! - `task` (string, required): the task description for the subagent
//! - `background` (boolean, optional, default: `false`): whether to run
//!   in the background

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::Mutex;

use tmg_tools::ToolError;
use tmg_tools::types::{Tool, ToolResult};

use crate::config::{AgentKind, AgentType, SubagentConfig};
use crate::custom::CustomAgentDef;
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

    /// Custom agent definitions indexed by name.
    ///
    /// Uses `Arc<CustomAgentDef>` to avoid cloning the entire definition
    /// on every agent resolution.
    custom_agents: Arc<HashMap<String, Arc<CustomAgentDef>>>,
}

impl SpawnAgentTool {
    /// Create a new `spawn_agent` tool backed by the given manager.
    pub fn new(manager: Arc<Mutex<SubagentManager>>) -> Self {
        Self {
            manager,
            custom_agents: Arc::new(HashMap::new()),
        }
    }

    /// Create a new `spawn_agent` tool with custom agent definitions.
    pub fn with_custom_agents(
        manager: Arc<Mutex<SubagentManager>>,
        custom_agents: Vec<CustomAgentDef>,
    ) -> Self {
        let map: HashMap<String, Arc<CustomAgentDef>> = custom_agents
            .into_iter()
            .map(|def| (def.name().to_owned(), Arc::new(def)))
            .collect();
        Self {
            manager,
            custom_agents: Arc::new(map),
        }
    }

    /// Resolve an `agent_type` string to an `AgentKind`.
    ///
    /// First checks built-in types, then falls back to custom agents.
    /// Uses `Arc::clone` for custom agents to avoid copying the entire
    /// definition on each resolution.
    fn resolve_agent_kind(&self, name: &str) -> Option<AgentKind> {
        // Try built-in first.
        if let Some(builtin) = AgentType::from_name(name) {
            return Some(AgentKind::Builtin(builtin));
        }

        // Try custom agents.
        self.custom_agents
            .get(name)
            .map(|def| AgentKind::Custom(Arc::clone(def)))
    }

    /// Return all available agent type names for error messages.
    fn available_agent_names(&self) -> Vec<String> {
        let mut names: Vec<String> = AgentType::ALL.iter().map(|t| t.name().to_owned()).collect();
        let mut custom_names: Vec<String> = self.custom_agents.keys().cloned().collect();
        custom_names.sort();
        names.extend(custom_names);
        names
    }
}

impl Tool for SpawnAgentTool {
    fn name(&self) -> &'static str {
        "spawn_agent"
    }

    fn description(&self) -> &'static str {
        "Spawn a subagent to perform a task. Built-in types: \
         'explore' (read-only codebase exploration), \
         'worker' (full tool access), \
         'plan' (read-only planning). \
         Custom agents defined in .tsumugi/agents/ are also available. \
         Set background=true to run asynchronously."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        let valid_names: Vec<serde_json::Value> = self
            .available_agent_names()
            .into_iter()
            .map(serde_json::Value::String)
            .collect();

        serde_json::json!({
            "type": "object",
            "properties": {
                "agent_type": {
                    "type": "string",
                    "description": "The type of subagent to spawn. Can be a built-in type (explore, worker, plan) or the name of a custom agent.",
                    "enum": valid_names
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

            let agent_kind = self.resolve_agent_kind(agent_type_str).ok_or_else(|| {
                ToolError::InvalidParams {
                    message: format!(
                        "unknown agent_type: {agent_type_str}. \
                             Valid types: {}",
                        self.available_agent_names().join(", ")
                    ),
                }
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
                agent_kind,
                task,
                background,
            };

            if background {
                // Background: acquire lock, spawn, release lock, return ID.
                let id = {
                    let mut manager = self.manager.lock().await;
                    manager
                        .spawn(config)
                        .await
                        .map_err(|e| ToolError::InvalidParams {
                            message: format!("failed to spawn subagent: {e}"),
                        })?
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
                    manager.spawn_with_notify(config).await.map_err(|e| {
                        ToolError::InvalidParams {
                            message: format!("failed to spawn subagent: {e}"),
                        }
                    })?
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
