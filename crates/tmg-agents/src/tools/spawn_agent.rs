//! The `spawn_agent` tool: spawns a subagent from the main agent loop.
//!
//! Parameters:
//! - `agent_type` (string, required): one of the user-spawnable
//!   built-in types (`"explore"`, `"worker"`, `"plan"`,
//!   `"initializer"`, `"tester"`, `"qa"`) or the name of a custom
//!   agent. The `"escalator"` type is harness-orchestrated only and is
//!   intentionally excluded from this tool's schema (see
//!   [`AgentType::is_user_spawnable`]).
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
    /// First checks built-in types (excluding harness-orchestrated ones
    /// such as `escalator`; see [`AgentType::is_user_spawnable`]), then
    /// falls back to custom agents. Uses `Arc::clone` for custom agents
    /// to avoid copying the entire definition on each resolution.
    fn resolve_agent_kind(&self, name: &str) -> Option<AgentKind> {
        // Try built-in first, but only the user-spawnable subset --
        // harness-only agents (e.g. `escalator`) must not be reachable
        // through `spawn_agent`.
        if let Some(builtin) = AgentType::from_name(name)
            && builtin.is_user_spawnable()
        {
            return Some(AgentKind::Builtin(builtin));
        }

        // Try custom agents.
        self.custom_agents
            .get(name)
            .map(|def| AgentKind::Custom(Arc::clone(def)))
    }

    /// Return the agent type names this tool is allowed to spawn.
    ///
    /// Used both for the JSON-Schema `enum` and for the
    /// "valid types" hint inside `unknown agent_type` error messages.
    /// Filters [`AgentType::ALL`] through
    /// [`AgentType::is_user_spawnable`] so harness-only agents
    /// (currently `escalator`) never appear.
    fn available_agent_names(&self) -> Vec<String> {
        let mut names: Vec<String> = AgentType::ALL
            .iter()
            .filter(|t| t.is_user_spawnable())
            .map(|t| t.name().to_owned())
            .collect();
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
         'plan' (read-only planning), \
         'initializer' (project bootstrap: features.json/init.sh/progress.md + initial commit), \
         'tester' (smoke-test runner via shell_exec), \
         'qa' (read-only acceptance-criteria QA; only agent that can mark features passing). \
         The 'escalator' built-in is harness-orchestrated only and \
         cannot be spawned through this tool. \
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

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions")]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;
    use tokio_util::sync::CancellationToken;

    use crate::manager::SubagentManager;

    /// Build a `SpawnAgentTool` with no custom agents over a manager
    /// that points at a deliberately-unreachable endpoint. The schema
    /// and resolution paths exercised in these tests never actually
    /// dispatch an HTTP request, so the unreachable endpoint is fine.
    fn make_tool() -> Option<SpawnAgentTool> {
        let cfg = tmg_llm::LlmClientConfig::new("http://127.0.0.1:1", "test-model");
        let client = tmg_llm::LlmClient::new(cfg).ok()?;
        let manager = SubagentManager::new(
            client,
            CancellationToken::new(),
            "http://127.0.0.1:1",
            "test-model",
        );
        Some(SpawnAgentTool::new(Arc::new(Mutex::new(manager))))
    }

    #[test]
    fn schema_enum_excludes_escalator() {
        let Some(tool) = make_tool() else {
            // LlmClient construction failed for environmental reasons;
            // the schema does not depend on the client at runtime, but
            // the constructor does. Skip rather than fail spuriously.
            return;
        };
        let schema = tool.parameters_schema();
        let enum_values = schema
            .pointer("/properties/agent_type/enum")
            .and_then(serde_json::Value::as_array)
            .expect("agent_type enum must exist");
        let names: Vec<&str> = enum_values
            .iter()
            .filter_map(serde_json::Value::as_str)
            .collect();
        assert!(
            !names.contains(&"escalator"),
            "schema enum must not advertise 'escalator' as a spawnable agent: {names:?}"
        );
        // Sanity-check that the user-spawnable built-ins are still there.
        for expected in ["explore", "worker", "plan", "initializer", "tester", "qa"] {
            assert!(
                names.contains(&expected),
                "schema enum must advertise '{expected}': {names:?}"
            );
        }
    }

    #[tokio::test]
    async fn rejects_escalator_at_runtime() {
        let Some(tool) = make_tool() else {
            return;
        };
        let params = serde_json::json!({
            "agent_type": "escalator",
            "task": "should be rejected before any spawn",
        });
        let err = tool
            .execute(params)
            .await
            .expect_err("escalator must be rejected by spawn_agent");
        match err {
            ToolError::InvalidParams { message } => {
                assert!(
                    message.contains("unknown agent_type") && message.contains("escalator"),
                    "error must call out the rejected agent_type: {message}"
                );
                // Ensure the suggested-types list does not mention
                // 'escalator' either.
                let suggested = message.split("Valid types:").nth(1).map_or("", str::trim);
                assert!(
                    !suggested.contains("escalator"),
                    "error suggestions must not include 'escalator': {message}"
                );
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }
}
