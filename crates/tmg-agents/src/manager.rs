//! Subagent manager: lifecycle tracking and parallel execution via `JoinSet`.
//!
//! The [`SubagentManager`] tracks all subagent instances, spawns them
//! as tokio tasks in a [`JoinSet`], and provides methods to query
//! status and collect results.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use tokio::sync::{Mutex, oneshot};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use tmg_llm::LlmClient;

use crate::builtins::{
    RunToolProvider, registry_for_agent_kind, registry_for_agent_kind_with_run_provider,
};
use crate::config::{AgentKind, SubagentConfig};
use crate::error::AgentError;
use crate::runner::SubagentRunner;
use crate::status::SubagentStatus;

/// A unique identifier for a subagent instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SubagentId(u64);

impl SubagentId {
    /// Return the inner `u64` value.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SubagentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Summary information about a subagent for display purposes.
#[derive(Debug, Clone)]
pub struct SubagentSummary {
    /// The subagent's unique identifier.
    pub id: SubagentId,
    /// The display name of the agent kind (e.g. "explore", "reviewer").
    pub agent_name: String,
    /// The task description.
    pub task: String,
    /// The current status.
    pub status: SubagentStatus,
}

/// Internal state for a tracked subagent instance.
struct SubagentInstance {
    agent_name: String,
    task: String,
    status: SubagentStatus,
}

/// Manages the lifecycle of subagent instances.
///
/// Uses a [`JoinSet`] for structured concurrency: all spawned subagent
/// tasks are tracked and can be shut down together via
/// [`SubagentManager::shutdown`].
pub struct SubagentManager {
    /// The LLM client shared with all subagents.
    client: LlmClient,

    /// Default endpoint URL (used when a custom agent does not override it).
    default_endpoint: String,

    /// Default model name (used when a custom agent does not override it).
    default_model: String,

    /// The parent cancellation token.
    parent_cancel: CancellationToken,

    /// The `JoinSet` tracking all running subagent tasks.
    join_set: JoinSet<(SubagentId, Result<String, AgentError>)>,

    /// Shared state for all tracked subagent instances.
    ///
    /// Uses `BTreeMap` for sorted iteration (by ID) and better cache
    /// locality on the small collections expected here.
    instances: Arc<Mutex<BTreeMap<SubagentId, SubagentInstance>>>,

    /// Counter for generating unique subagent IDs.
    next_id: u64,

    /// Optional source of Run-aware tools (e.g. `progress_append`,
    /// `feature_list_*`).
    ///
    /// When set, harnessed-run subagents (`Initializer`, `Tester`,
    /// `Qa`) get a registry that includes the Run-aware tools they
    /// declare in [`AgentType::allowed_tools`](crate::config::AgentType::allowed_tools).
    /// When unset, those names are silently dropped — the subagent
    /// runs without those tools, which is the intended degraded
    /// behaviour for Run-less code paths.
    run_tool_provider: Option<Arc<dyn RunToolProvider>>,
}

impl SubagentManager {
    /// Create a new subagent manager.
    ///
    /// The `default_endpoint` and `default_model` are used when a custom
    /// agent does not specify its own endpoint/model overrides.
    pub fn new(
        client: LlmClient,
        parent_cancel: CancellationToken,
        default_endpoint: impl Into<String>,
        default_model: impl Into<String>,
    ) -> Self {
        Self {
            client,
            default_endpoint: default_endpoint.into(),
            default_model: default_model.into(),
            parent_cancel,
            join_set: JoinSet::new(),
            instances: Arc::new(Mutex::new(BTreeMap::new())),
            next_id: 1,
            run_tool_provider: None,
        }
    }

    /// Install (or replace) the [`RunToolProvider`] used for spawning
    /// subagents that need Run-aware tools.
    ///
    /// Pass `None` to clear a previously-installed provider (e.g. when
    /// the active run is finalised). Subsequent spawns will skip
    /// Run-aware tool names just as if no provider had ever been set.
    pub fn set_run_tool_provider(&mut self, provider: Option<Arc<dyn RunToolProvider>>) {
        self.run_tool_provider = provider;
    }

    /// Borrow the currently-installed [`RunToolProvider`], if any.
    #[must_use]
    pub fn run_tool_provider(&self) -> Option<&Arc<dyn RunToolProvider>> {
        self.run_tool_provider.as_ref()
    }

    /// Spawn a subagent and return its ID immediately.
    ///
    /// The subagent runs as a task in the internal `JoinSet`. Use
    /// [`collect_completed`] to drain finished results, or
    /// [`wait_for`] to await a specific subagent.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Llm`] if a custom agent requires a
    /// dedicated `LlmClient` and client creation fails.
    pub async fn spawn(&mut self, config: SubagentConfig) -> Result<SubagentId, AgentError> {
        self.spawn_inner(config, None).await
    }

    /// Spawn a subagent and return both its ID and a oneshot receiver
    /// that will deliver the result when the subagent completes.
    ///
    /// This is designed for foreground spawns where the caller needs to
    /// drop the manager lock before awaiting the result (e.g., so the
    /// TUI can still call `summaries()` while the subagent runs).
    /// # Errors
    ///
    /// Returns [`AgentError::Llm`] if a custom agent requires a
    /// dedicated `LlmClient` and client creation fails.
    pub async fn spawn_with_notify(
        &mut self,
        config: SubagentConfig,
    ) -> Result<(SubagentId, oneshot::Receiver<Result<String, AgentError>>), AgentError> {
        let (tx, rx) = oneshot::channel();
        let id = self.spawn_inner(config, Some(tx)).await?;
        Ok((id, rx))
    }

    /// Internal spawn implementation shared by `spawn` and `spawn_with_notify`.
    async fn spawn_inner(
        &mut self,
        config: SubagentConfig,
        notify: Option<oneshot::Sender<Result<String, AgentError>>>,
    ) -> Result<SubagentId, AgentError> {
        let id = SubagentId(self.next_id);
        self.next_id += 1;

        let agent_name = config.agent_kind.name().to_owned();

        // Insert directly as Running -- no need for the Pending ->
        // Running transition since we spawn the task immediately.
        let instance = SubagentInstance {
            agent_name: agent_name.clone(),
            task: config.task.clone(),
            status: SubagentStatus::Running,
        };

        {
            let mut instances = self.instances.lock().await;
            instances.insert(id, instance);
        }

        // For custom agents with a specific endpoint/model, create a
        // dedicated LlmClient. Otherwise, use the shared client.
        let client = if let AgentKind::Custom(ref def) = config.agent_kind {
            if def.endpoint().is_some() || def.model().is_some() {
                let endpoint = def.endpoint().unwrap_or(&self.default_endpoint);
                let model = def.model().unwrap_or(&self.default_model);
                let llm_config = tmg_llm::LlmClientConfig::new(endpoint, model);
                tmg_llm::LlmClient::new(llm_config).map_err(AgentError::Llm)?
            } else {
                self.client.clone()
            }
        } else {
            self.client.clone()
        };

        let agent_kind = config.agent_kind.clone();
        let task = config.task.clone();
        let cancel = self.parent_cancel.child_token();
        let instances = Arc::clone(&self.instances);
        let run_tool_provider = self.run_tool_provider.clone();

        self.join_set.spawn(async move {
            // Build the subagent's tool registry. With a
            // `RunToolProvider` installed, harnessed agents get
            // Run-aware tools (`progress_append`, `feature_list_*`)
            // registered as well; without one, those names are
            // silently dropped.
            let registry = match &run_tool_provider {
                Some(provider) => {
                    registry_for_agent_kind_with_run_provider(&agent_kind, &**provider)
                }
                None => registry_for_agent_kind(&agent_kind),
            };
            let mut runner = SubagentRunner::new(client, registry, &agent_kind, cancel);

            let result = runner.run(&task).await;

            // Update the instance status.
            {
                let mut insts = instances.lock().await;
                if let Some(inst) = insts.get_mut(&id) {
                    match &result {
                        Ok(output) => {
                            inst.status.complete(output.clone());
                        }
                        Err(AgentError::Cancelled) => {
                            inst.status.cancel();
                        }
                        Err(e) => {
                            inst.status.fail(e.to_string());
                        }
                    }
                }
            }

            // If a foreground caller is waiting via oneshot, deliver the
            // result. Ignore send errors (the receiver may have been
            // dropped if the caller was cancelled).
            if let Some(tx) = notify {
                let to_send = match &result {
                    Ok(output) => Ok(output.clone()),
                    Err(e) => Err(AgentError::JoinError {
                        message: e.to_string(),
                    }),
                };
                let _ = tx.send(to_send);
            }

            (id, result)
        });

        Ok(id)
    }

    /// Wait for a specific subagent to complete and return its result.
    ///
    /// Drains the `JoinSet` until the target subagent's result is found.
    /// Other completed results are recorded in their instance state.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] if the subagent fails, is cancelled, or
    /// the task panics.
    pub async fn wait_for(&mut self, target_id: SubagentId) -> Result<String, AgentError> {
        loop {
            let Some(join_result) = self.join_set.join_next().await else {
                return Err(AgentError::JoinError {
                    message: format!("subagent {target_id} not found in JoinSet"),
                });
            };

            let (id, result) = join_result.map_err(|e| AgentError::JoinError {
                message: e.to_string(),
            })?;

            if id == target_id {
                return result;
            }
            // Otherwise, the result was for a different subagent -- its
            // status was already updated in the spawn closure.
        }
    }

    /// Collect all completed subagent results without blocking.
    ///
    /// Returns a list of `(SubagentId, Result)` for subagents that have
    /// finished since the last call. Does not wait for running subagents.
    pub fn collect_completed(&mut self) -> Vec<(SubagentId, Result<String, AgentError>)> {
        let mut results = Vec::new();

        while let Some(Ok((id, result))) = self.join_set.try_join_next() {
            results.push((id, result));
        }

        results
    }

    /// Return summaries of all tracked subagent instances.
    ///
    /// Results are sorted by ID (guaranteed by `BTreeMap` iteration order).
    pub async fn summaries(&self) -> Vec<SubagentSummary> {
        let instances = self.instances.lock().await;
        instances
            .iter()
            .map(|(&id, inst)| SubagentSummary {
                id,
                agent_name: inst.agent_name.clone(),
                task: inst.task.clone(),
                status: inst.status.clone(),
            })
            .collect()
    }

    /// Return the number of currently running (non-terminal) subagents.
    pub async fn running_count(&self) -> usize {
        let instances = self.instances.lock().await;
        instances
            .values()
            .filter(|inst| !inst.status.is_terminal())
            .count()
    }

    /// Gracefully shut down all running subagents.
    ///
    /// Aborts all tasks in the `JoinSet` and waits for them to complete.
    pub async fn shutdown(&mut self) {
        self.join_set.shutdown().await;
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions")]
mod tests {
    use super::*;
    use crate::config::AgentType;

    #[test]
    fn subagent_summary_fields() {
        let summary = SubagentSummary {
            id: SubagentId(1),
            agent_name: "explore".to_owned(),
            task: "test task".to_owned(),
            status: SubagentStatus::Running,
        };

        assert_eq!(summary.id, SubagentId(1));
        assert_eq!(summary.agent_name, "explore");
        assert_eq!(summary.task, "test task");
        assert_eq!(summary.status, SubagentStatus::Running);
    }

    #[tokio::test]
    async fn manager_spawn_increments_id() {
        let config = tmg_llm::LlmClientConfig::new("http://localhost:9999", "test");
        // This will fail to connect, but we only test ID assignment.
        let client = tmg_llm::LlmClient::new(config);
        // If LlmClient::new can fail, we skip. Otherwise proceed.
        let Ok(client) = client else {
            return;
        };

        let cancel = CancellationToken::new();
        let mut manager =
            SubagentManager::new(client, cancel.clone(), "http://localhost:9999", "test");

        let config1 = SubagentConfig {
            agent_kind: AgentKind::Builtin(AgentType::Explore),
            task: "task 1".to_owned(),
            background: true,
        };
        let config2 = SubagentConfig {
            agent_kind: AgentKind::Builtin(AgentType::Plan),
            task: "task 2".to_owned(),
            background: true,
        };

        let id1 = manager
            .spawn(config1)
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        let id2 = manager
            .spawn(config2)
            .await
            .unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(id1, SubagentId(1));
        assert_eq!(id2, SubagentId(2));

        // Clean up: cancel and shutdown to avoid dangling tasks.
        cancel.cancel();
        manager.shutdown().await;
    }

    #[tokio::test]
    async fn manager_summaries_include_all() {
        let config = tmg_llm::LlmClientConfig::new("http://localhost:9999", "test");
        let Ok(client) = tmg_llm::LlmClient::new(config) else {
            return;
        };

        let cancel = CancellationToken::new();
        let mut manager =
            SubagentManager::new(client, cancel.clone(), "http://localhost:9999", "test");

        let config1 = SubagentConfig {
            agent_kind: AgentKind::Builtin(AgentType::Explore),
            task: "task 1".to_owned(),
            background: true,
        };
        manager
            .spawn(config1)
            .await
            .unwrap_or_else(|e| panic!("{e}"));

        let summaries = manager.summaries().await;
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].agent_name, "explore");

        cancel.cancel();
        manager.shutdown().await;
    }
}
