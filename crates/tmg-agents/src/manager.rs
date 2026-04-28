//! Subagent manager: lifecycle tracking and parallel execution via `JoinSet`.
//!
//! The [`SubagentManager`] tracks all subagent instances, spawns them
//! as tokio tasks in a [`JoinSet`], and provides methods to query
//! status and collect results.
//!
//! ## Endpoint / model resolution
//!
//! Each spawn picks an `(endpoint, model)` pair using the precedence
//! rules in [`SubagentManager::resolve_endpoint`]:
//!
//! - [`AgentKind::Custom`]: a non-empty `endpoint` / `model` on the
//!   custom-agent definition wins; missing fields fall back to the
//!   manager defaults.
//! - [`AgentKind::Builtin`] of [`AgentType::Escalator`]: non-empty
//!   values from [`EscalatorOverrides`] win; missing fields fall back
//!   to the manager defaults so the escalator can transparently share
//!   the main endpoint until a dedicated lightweight model is
//!   configured.
//! - All other built-in agents always run on the manager defaults.
//!
//! TODO(SPEC §10.1 `[llm.subagent_pool]`): when load-balanced subagent
//! endpoints land, the round-robin pool MUST NOT route the escalator;
//! the escalator owns its dedicated endpoint so its cost stays
//! predictable (SPEC §9.10). Keep that wiring out of
//! `resolve_endpoint` for the escalator branch even after the pool
//! exists.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use tokio::sync::{Mutex, oneshot};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use tmg_llm::LlmClient;
use tmg_sandbox::{SandboxContext, SandboxMode};

use crate::builtins::{
    RunToolProvider, registry_for_agent_kind, registry_for_agent_kind_with_run_provider,
};
use crate::config::{AgentKind, AgentType, SubagentConfig};
use crate::error::AgentError;
use crate::runner::SubagentRunner;
use crate::status::SubagentStatus;

/// Optional overrides for the escalator subagent's `(endpoint, model)`
/// pair plus an explicit disable switch (SPEC §9.10 / §10.1
/// `[harness.escalator]`).
///
/// Empty / `None` fields fall back to the manager's main endpoint and
/// model so a partially-configured TOML still produces a working
/// escalator. The `disabled` flag is enforced inside
/// [`SubagentManager::spawn_inner`]: a disabled escalator request
/// produces [`AgentError::EscalatorDisabled`] before any LLM client
/// is constructed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EscalatorOverrides {
    /// Optional override endpoint URL. `None` (or `Some("")` after the
    /// caller normalises empty strings) means "inherit the main
    /// endpoint".
    pub endpoint: Option<String>,

    /// Optional override model name. `None` means "inherit the main
    /// model".
    pub model: Option<String>,

    /// When `true`, requests for the escalator are rejected with
    /// [`AgentError::EscalatorDisabled`].
    pub disabled: bool,
}

impl EscalatorOverrides {
    /// Construct overrides from raw config values, treating empty
    /// strings as `None` so an unset `[harness.escalator]` field
    /// (`endpoint = ""`) inherits the main endpoint.
    #[must_use]
    pub fn from_strings(endpoint: String, model: String, disabled: bool) -> Self {
        let normalize = |s: String| if s.is_empty() { None } else { Some(s) };
        Self {
            endpoint: normalize(endpoint),
            model: normalize(model),
            disabled,
        }
    }
}

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

    /// Optional overrides for the escalator's endpoint / model and
    /// the operator's disable switch. Defaults to "no overrides, not
    /// disabled" so manager construction stays a single positional
    /// call until a `[harness.escalator]` section is present.
    escalator_overrides: EscalatorOverrides,

    /// Parent [`SandboxContext`] from which each spawn derives its
    /// per-subagent sandbox.
    ///
    /// The agent kind's [`AgentType::sandbox_mode`] picks the
    /// [`SandboxMode`] for the derivation, while the workspace,
    /// timeout, and OOM score are inherited from this parent. The
    /// parent sandbox is a required constructor argument for
    /// [`SubagentManager::new`], so production callers cannot
    /// silently fall back to an unrestricted default.
    parent_sandbox: Arc<SandboxContext>,
}

/// Derive a per-subagent [`SandboxContext`] from a parent context and
/// a sandbox mode.
///
/// The returned context inherits the parent's workspace, allowed
/// domains, timeout, and OOM-score adjustment, and applies the
/// supplied [`SandboxMode`] override. The result is **not** activated
/// at the OS level; subagents that need OS-level enforcement should
/// call [`SandboxContext::activate`] themselves.
///
/// Used by [`SubagentManager::spawn_inner`] to give each spawned
/// subagent a sandbox matching its
/// [`AgentType::sandbox_mode`](crate::config::AgentType::sandbox_mode):
/// `worker` / `initializer` / `tester` get [`SandboxMode::WorkspaceWrite`],
/// `explore` / `plan` / `qa` / `escalator` get
/// [`SandboxMode::ReadOnly`].
#[must_use]
pub fn derive_sandbox(parent: &SandboxContext, mode: SandboxMode) -> SandboxContext {
    parent.derive(mode)
}

impl SubagentManager {
    /// Create a new subagent manager.
    ///
    /// The `default_endpoint` and `default_model` are used when a custom
    /// agent does not specify its own endpoint/model overrides.
    /// The escalator overrides default to "no overrides, not disabled";
    /// install them with [`Self::with_escalator_overrides`] when a
    /// `[harness.escalator]` section is configured.
    ///
    /// The `parent_sandbox` argument is **required**: it is the
    /// [`SandboxContext`] from which every spawned subagent derives
    /// its own sandbox. Making it a constructor argument prevents
    /// callers from accidentally spawning subagents under an
    /// unrestricted default (issue #47 follow-up).
    pub fn new(
        client: LlmClient,
        parent_cancel: CancellationToken,
        default_endpoint: impl Into<String>,
        default_model: impl Into<String>,
        parent_sandbox: Arc<SandboxContext>,
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
            escalator_overrides: EscalatorOverrides::default(),
            parent_sandbox,
        }
    }

    /// Replace the parent [`SandboxContext`] in-place.
    ///
    /// Constructor-time installation via [`Self::new`] is the
    /// canonical path; this setter exists only for explicit
    /// reconfiguration (e.g. a future config-reload path that swaps
    /// the sandbox after the `[sandbox]` section has been re-merged
    /// with run-scope overrides).
    pub fn set_parent_sandbox(&mut self, sandbox: Arc<SandboxContext>) {
        self.parent_sandbox = sandbox;
    }

    /// Borrow the currently-installed parent [`SandboxContext`].
    #[must_use]
    pub fn parent_sandbox(&self) -> &Arc<SandboxContext> {
        &self.parent_sandbox
    }

    /// Install (or replace) the [`EscalatorOverrides`] used when
    /// spawning the escalator subagent.
    ///
    /// Builder-style consuming method so the call site reads:
    ///
    /// ```ignore
    /// let manager = SubagentManager::new(client, cancel, ep, model, parent_sandbox)
    ///     .with_escalator_overrides(EscalatorOverrides::from_strings(
    ///         cfg.endpoint.clone(),
    ///         cfg.model.clone(),
    ///         cfg.disable,
    ///     ));
    /// ```
    #[must_use]
    pub fn with_escalator_overrides(mut self, overrides: EscalatorOverrides) -> Self {
        self.escalator_overrides = overrides;
        self
    }

    /// Replace the [`EscalatorOverrides`] in-place.
    ///
    /// Useful when the manager already lives behind an `Arc<Mutex<_>>`
    /// and the escalator config changes after construction (e.g. a
    /// future config-reload path).
    pub fn set_escalator_overrides(&mut self, overrides: EscalatorOverrides) {
        self.escalator_overrides = overrides;
    }

    /// Borrow the currently-installed escalator overrides.
    #[must_use]
    pub fn escalator_overrides(&self) -> &EscalatorOverrides {
        &self.escalator_overrides
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

    /// Resolve the `(endpoint, model)` pair for a given [`AgentKind`].
    ///
    /// Precedence rules (see the module-level docs for context):
    ///
    /// - [`AgentKind::Custom`]: a non-empty `endpoint` / `model` on
    ///   the custom-agent definition wins; missing fields fall back
    ///   to the manager defaults.
    /// - [`AgentKind::Builtin`] of [`AgentType::Escalator`]: non-empty
    ///   values from [`EscalatorOverrides`] win; missing fields fall
    ///   back to the manager defaults.
    /// - All other built-in agents always run on the manager defaults.
    ///
    /// Exposed so the spawn path (and the dedicated test
    /// [`Self::resolved_endpoint_for_kind`]) share a single source of
    /// truth for the precedence rules.
    fn resolve_endpoint(&self, kind: &AgentKind) -> (String, String) {
        match kind {
            AgentKind::Custom(def) => {
                let endpoint = def.endpoint().unwrap_or(&self.default_endpoint).to_owned();
                let model = def.model().unwrap_or(&self.default_model).to_owned();
                (endpoint, model)
            }
            AgentKind::Builtin(AgentType::Escalator) => {
                let endpoint = self
                    .escalator_overrides
                    .endpoint
                    .clone()
                    .unwrap_or_else(|| self.default_endpoint.clone());
                let model = self
                    .escalator_overrides
                    .model
                    .clone()
                    .unwrap_or_else(|| self.default_model.clone());
                (endpoint, model)
            }
            AgentKind::Builtin(_) => (self.default_endpoint.clone(), self.default_model.clone()),
        }
    }

    /// Test/inspection helper that returns the `(endpoint, model)`
    /// pair a hypothetical spawn of `kind` would receive.
    ///
    /// Bypasses LLM-client construction so test code (and future
    /// diagnostics commands) can verify the precedence rules without
    /// a live llama-server.
    #[must_use]
    pub fn resolved_endpoint_for_kind(&self, kind: &AgentKind) -> (String, String) {
        self.resolve_endpoint(kind)
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
        // Reject disabled-escalator requests up front, before
        // allocating an ID or constructing an LLM client. SPEC §9.10
        // expects the auto-promotion path to treat this as "do not
        // escalate" rather than as a generic LLM/tool failure.
        if matches!(config.agent_kind, AgentKind::Builtin(AgentType::Escalator))
            && self.escalator_overrides.disabled
        {
            return Err(AgentError::EscalatorDisabled);
        }

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

        // Resolve the endpoint / model with the centralised precedence
        // rules (see `resolve_endpoint`). When the resolved pair
        // matches the manager defaults we keep the shared `self.client`
        // to avoid constructing a redundant `LlmClient`; otherwise we
        // spin up a dedicated client. This reuse keeps the common
        // built-in path allocation-free while still letting custom
        // agents and the escalator land on a different endpoint.
        let (resolved_endpoint, resolved_model) = self.resolve_endpoint(&config.agent_kind);
        let client =
            if resolved_endpoint == self.default_endpoint && resolved_model == self.default_model {
                self.client.clone()
            } else {
                let llm_config = tmg_llm::LlmClientConfig::new(&resolved_endpoint, &resolved_model);
                tmg_llm::LlmClient::new(llm_config).map_err(AgentError::Llm)?
            };

        let agent_kind = config.agent_kind.clone();
        let task = config.task.clone();
        let cancel = self.parent_cancel.child_token();
        let instances = Arc::clone(&self.instances);
        let run_tool_provider = self.run_tool_provider.clone();
        // Derive the per-subagent sandbox up-front so the spawn
        // closure does not need to know the precedence rules.
        // Built-in agents pick their mode from `AgentType::sandbox_mode`;
        // custom agents may override via `CustomAgentDef::sandbox_mode`.
        let subagent_mode = match &agent_kind {
            AgentKind::Builtin(t) => t.sandbox_mode(),
            AgentKind::Custom(def) => def
                .sandbox_mode()
                .unwrap_or_else(|| self.parent_sandbox.mode()),
        };
        let subagent_sandbox = Arc::new(derive_sandbox(&self.parent_sandbox, subagent_mode));

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
            let mut runner =
                SubagentRunner::new(client, registry, &agent_kind, cancel, subagent_sandbox);

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
#[expect(clippy::expect_used, reason = "test assertions")]
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
        let mut manager = SubagentManager::new(
            client,
            cancel.clone(),
            "http://localhost:9999",
            "test",
            Arc::new(SandboxContext::test_default()),
        );

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
        let mut manager = SubagentManager::new(
            client,
            cancel.clone(),
            "http://localhost:9999",
            "test",
            Arc::new(SandboxContext::test_default()),
        );

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

    /// Helper that builds a default-configured manager for endpoint
    /// resolution tests. Returns `None` when `LlmClient::new` fails so
    /// the test can early-exit gracefully (mirrors the pattern used by
    /// the spawn tests above).
    fn make_manager_for_resolution_tests() -> Option<SubagentManager> {
        let config = tmg_llm::LlmClientConfig::new("http://main:8080", "main-model");
        let client = tmg_llm::LlmClient::new(config).ok()?;
        Some(SubagentManager::new(
            client,
            CancellationToken::new(),
            "http://main:8080",
            "main-model",
            Arc::new(SandboxContext::test_default()),
        ))
    }

    #[test]
    fn escalator_overrides_from_strings_treats_empty_as_none() {
        let overrides = EscalatorOverrides::from_strings(String::new(), String::new(), false);
        assert_eq!(overrides.endpoint, None);
        assert_eq!(overrides.model, None);
        assert!(!overrides.disabled);

        let overrides = EscalatorOverrides::from_strings(
            "http://escalator.invalid".to_owned(),
            "tiny".to_owned(),
            true,
        );
        assert_eq!(
            overrides.endpoint.as_deref(),
            Some("http://escalator.invalid"),
        );
        assert_eq!(overrides.model.as_deref(), Some("tiny"));
        assert!(overrides.disabled);
    }

    #[test]
    fn resolved_endpoint_for_builtin_uses_main_defaults() {
        let Some(manager) = make_manager_for_resolution_tests() else {
            return;
        };
        for kind in [
            AgentKind::Builtin(AgentType::Explore),
            AgentKind::Builtin(AgentType::Worker),
            AgentKind::Builtin(AgentType::Plan),
            AgentKind::Builtin(AgentType::Initializer),
            AgentKind::Builtin(AgentType::Tester),
            AgentKind::Builtin(AgentType::Qa),
        ] {
            let (ep, model) = manager.resolved_endpoint_for_kind(&kind);
            assert_eq!(ep, "http://main:8080", "wrong endpoint for {}", kind.name());
            assert_eq!(model, "main-model", "wrong model for {}", kind.name());
        }
    }

    /// Acceptance criterion (issue #36): the escalator endpoint
    /// override must actually be used at spawn time, not silently
    /// dropped. We verify this via the public
    /// [`SubagentManager::resolved_endpoint_for_kind`] helper rather
    /// than a mock `LlmClient` so the test stays unit-scoped.
    #[test]
    fn resolved_endpoint_for_escalator_uses_overrides() {
        let Some(manager) = make_manager_for_resolution_tests() else {
            return;
        };
        let manager = manager.with_escalator_overrides(EscalatorOverrides::from_strings(
            "http://escalator.invalid".to_owned(),
            "lite".to_owned(),
            false,
        ));

        let (ep, model) =
            manager.resolved_endpoint_for_kind(&AgentKind::Builtin(AgentType::Escalator));
        assert_eq!(ep, "http://escalator.invalid");
        assert_eq!(model, "lite");

        // Other built-ins must keep using the main endpoint even when
        // escalator overrides are installed.
        let (ep, model) =
            manager.resolved_endpoint_for_kind(&AgentKind::Builtin(AgentType::Explore));
        assert_eq!(ep, "http://main:8080");
        assert_eq!(model, "main-model");
    }

    /// Regression: an empty `[harness.escalator] endpoint = ""` must
    /// fall back to the main endpoint instead of trying to talk to a
    /// blank URL.
    #[test]
    fn resolved_endpoint_for_escalator_falls_back_to_main_when_empty() {
        let Some(manager) = make_manager_for_resolution_tests() else {
            return;
        };
        let manager = manager.with_escalator_overrides(EscalatorOverrides::from_strings(
            String::new(),
            String::new(),
            false,
        ));

        let (ep, model) =
            manager.resolved_endpoint_for_kind(&AgentKind::Builtin(AgentType::Escalator));
        assert_eq!(ep, "http://main:8080");
        assert_eq!(model, "main-model");
    }

    /// Partial overrides: only `endpoint` set, `model` empty -> the
    /// model inherits the main config while the endpoint is honoured.
    #[test]
    fn resolved_endpoint_for_escalator_mixes_override_and_default() {
        let Some(manager) = make_manager_for_resolution_tests() else {
            return;
        };
        let manager = manager.with_escalator_overrides(EscalatorOverrides::from_strings(
            "http://escalator.invalid".to_owned(),
            String::new(),
            false,
        ));

        let (ep, model) =
            manager.resolved_endpoint_for_kind(&AgentKind::Builtin(AgentType::Escalator));
        assert_eq!(ep, "http://escalator.invalid");
        assert_eq!(model, "main-model");
    }

    #[test]
    fn resolved_endpoint_for_custom_uses_def_overrides() {
        let Some(manager) = make_manager_for_resolution_tests() else {
            return;
        };

        let toml = r#"
name = "reviewer"
description = "test"
instructions = "do things"
endpoint = "http://custom:7777"
model = "custom-model"

[tools]
allow = ["file_read"]
"#;
        let def = crate::custom::CustomAgentDef::from_toml(toml, "test.toml")
            .unwrap_or_else(|e| panic!("{e}"));
        let kind = AgentKind::Custom(Arc::new(def));
        let (ep, model) = manager.resolved_endpoint_for_kind(&kind);
        assert_eq!(ep, "http://custom:7777");
        assert_eq!(model, "custom-model");
    }

    #[tokio::test]
    async fn escalator_disabled_rejects_spawn() {
        let config = tmg_llm::LlmClientConfig::new("http://main:8080", "main-model");
        let Ok(client) = tmg_llm::LlmClient::new(config) else {
            return;
        };
        let cancel = CancellationToken::new();
        let mut manager = SubagentManager::new(
            client,
            cancel.clone(),
            "http://main:8080",
            "main-model",
            Arc::new(SandboxContext::test_default()),
        )
        .with_escalator_overrides(EscalatorOverrides::from_strings(
            String::new(),
            String::new(),
            true,
        ));

        let cfg = SubagentConfig {
            agent_kind: AgentKind::Builtin(AgentType::Escalator),
            task: "should be rejected".to_owned(),
            background: true,
        };
        let err = manager
            .spawn(cfg)
            .await
            .expect_err("disabled escalator must reject spawn");
        assert!(
            matches!(err, AgentError::EscalatorDisabled),
            "expected AgentError::EscalatorDisabled, got {err:?}"
        );

        // Other builtins must still be spawnable when only the
        // escalator is disabled.
        let cfg_explore = SubagentConfig {
            agent_kind: AgentKind::Builtin(AgentType::Explore),
            task: "explore should still spawn".to_owned(),
            background: true,
        };
        manager
            .spawn(cfg_explore)
            .await
            .unwrap_or_else(|e| panic!("non-escalator spawn must still succeed: {e}"));

        cancel.cancel();
        manager.shutdown().await;
    }

    #[tokio::test]
    async fn escalator_not_disabled_by_default() {
        let config = tmg_llm::LlmClientConfig::new("http://main:8080", "main-model");
        let Ok(client) = tmg_llm::LlmClient::new(config) else {
            return;
        };
        let cancel = CancellationToken::new();
        let mut manager = SubagentManager::new(
            client,
            cancel.clone(),
            "http://main:8080",
            "main-model",
            Arc::new(SandboxContext::test_default()),
        );

        let cfg = SubagentConfig {
            agent_kind: AgentKind::Builtin(AgentType::Escalator),
            task: "default escalator should spawn".to_owned(),
            background: true,
        };
        // The actual LLM call will fail because no server is running,
        // but the spawn-side gating must succeed.
        let result = manager.spawn(cfg).await;
        assert!(
            result.is_ok(),
            "default config should permit escalator spawn; got {result:?}",
        );

        cancel.cancel();
        manager.shutdown().await;
    }
}
