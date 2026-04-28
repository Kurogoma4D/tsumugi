//! Built-in subagent definitions and tool registry construction.
//!
//! Provides helpers to create a [`ToolRegistry`] filtered to only the
//! tools allowed for a given [`AgentType`] or [`AgentKind`].
//!
//! ## Stateless vs. Run-aware tools
//!
//! The built-in subagents declare two flavours of tool:
//!
//! - **Stateless tools** live in [`tmg_tools::default_registry`] and can
//!   be constructed without any external context (file I/O, grep,
//!   shell). [`registry_for_agent_type`] / [`registry_for_agent_kind`]
//!   handle these unconditionally.
//! - **Run-aware tools** (`progress_append`, `feature_list_read`,
//!   `feature_list_mark_passing`) require a `RunRunner` handle to
//!   operate. They are referenced by name in
//!   [`AgentType::allowed_tools`] but **cannot** be registered from
//!   inside `tmg-agents` because that would create a dependency cycle
//!   on `tmg-harness`.
//!
//! The cycle is broken by the [`RunToolProvider`] trait: anyone holding
//! a `RunRunner` (in practice `tmg-harness` / the CLI wiring layer) can
//! implement this trait and pass an `Arc<dyn RunToolProvider>` into the
//! [`SubagentManager`](crate::manager::SubagentManager). When a
//! subagent is spawned, the manager builds the stateless registry first
//! and then asks the provider to register any Run-aware tools the
//! agent's `allowed_tools` list mentions.
//!
//! In Run-less code paths (`--prompt` one-shots) the provider is
//! `None`; subagents whose allowed-tools list references Run-aware
//! tools simply receive a registry without those tools and the LLM is
//! told (via the system prompt) that the harnessed gates do not apply.

use tmg_tools::ToolRegistry;

use crate::config::{AgentKind, AgentType};

/// Source of Run-aware tools that cannot be constructed by `tmg-agents`
/// alone (they depend on a `RunRunner`).
///
/// Implementors decide which subset of `progress_append`,
/// `feature_list_read`, and `feature_list_mark_passing` is appropriate
/// for the active run scope (ad-hoc vs. harnessed) and must register
/// only those tools.
///
/// The trait is deliberately small — just one method that takes the
/// target registry and the tool name to register. Returning `bool`
/// makes the spawner's "skip unregistered Run-aware tool names"
/// behaviour explicit.
pub trait RunToolProvider: Send + Sync {
    /// Attempt to register a Run-aware tool with `name` into `registry`.
    ///
    /// Returns `true` if the tool was registered, `false` if `name` is
    /// not a Run-aware tool the provider knows about (or if the active
    /// run scope forbids it — e.g. a tester subagent asking for
    /// `feature_list_mark_passing` on an ad-hoc run).
    fn register_run_tool(&self, registry: &mut ToolRegistry, name: &str) -> bool;
}

/// Create a [`ToolRegistry`] containing only the stateless tools allowed
/// for the given agent type.
///
/// This filters the default registry, keeping only tools whose names
/// appear in [`AgentType::allowed_tools`]. Run-aware tools are silently
/// skipped — use [`registry_for_agent_kind_with_run_provider`] if you
/// have a [`RunToolProvider`] available.
#[must_use]
pub fn registry_for_agent_type(agent_type: AgentType) -> ToolRegistry {
    let allowed: Vec<&str> = agent_type.allowed_tools().to_vec();
    build_registry(&allowed, None)
}

/// Create a [`ToolRegistry`] containing only the stateless tools allowed
/// for the given agent kind (built-in or custom).
///
/// Run-aware tools are silently skipped — use
/// [`registry_for_agent_kind_with_run_provider`] if you have a
/// [`RunToolProvider`] available.
#[must_use]
pub fn registry_for_agent_kind(agent_kind: &AgentKind) -> ToolRegistry {
    let allowed = agent_kind.allowed_tool_names();
    build_registry(&allowed, None)
}

/// Create a [`ToolRegistry`] for the given agent kind, including
/// Run-aware tools registered through `provider`.
///
/// For each name in the agent's `allowed_tools` list:
///
/// 1. If the name belongs to the stateless default registry, the
///    corresponding tool is registered.
/// 2. Otherwise, [`RunToolProvider::register_run_tool`] is called with
///    the name; if the provider declines, the name is silently
///    skipped. This preserves the "unknown name = absent tool"
///    contract the subagent permission tests rely on.
#[must_use]
pub fn registry_for_agent_kind_with_run_provider(
    agent_kind: &AgentKind,
    provider: &dyn RunToolProvider,
) -> ToolRegistry {
    let allowed = agent_kind.allowed_tool_names();
    build_registry(&allowed, Some(provider))
}

/// Build a registry containing only the named tools, drawing first from
/// the stateless default registry and then from the optional
/// [`RunToolProvider`].
fn build_registry(allowed: &[&str], provider: Option<&dyn RunToolProvider>) -> ToolRegistry {
    let full_registry = tmg_tools::default_registry();
    let mut filtered = ToolRegistry::new();

    for name in allowed {
        if full_registry.get(name).is_some() {
            register_stateless_tool_by_name(&mut filtered, name);
        } else if let Some(provider) = provider {
            // Unknown to the stateless default registry; ask the Run
            // provider. A `false` return is the silent-skip path
            // documented above.
            let _ = provider.register_run_tool(&mut filtered, name);
        }
        // No provider AND not in default_registry: silently skip. The
        // subagent will simply not see the tool, which is the intended
        // behaviour for Run-less code paths.
    }

    filtered
}

/// Register a stateless built-in tool by name into the given registry.
fn register_stateless_tool_by_name(registry: &mut ToolRegistry, name: &str) {
    match name {
        "file_read" => registry.register(tmg_tools::tools::FileReadTool),
        "file_write" => registry.register(tmg_tools::tools::FileWriteTool),
        "file_patch" => registry.register(tmg_tools::tools::FilePatchTool),
        "grep_search" => registry.register(tmg_tools::tools::GrepSearchTool),
        "list_dir" => registry.register(tmg_tools::tools::ListDirTool),
        "shell_exec" => registry.register(tmg_tools::tools::ShellExecTool),
        _ => unreachable!("caller guarantees presence in default_registry"),
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions")]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn explore_registry_has_read_only_tools() {
        let registry = registry_for_agent_type(AgentType::Explore);
        assert!(registry.get("file_read").is_some());
        assert!(registry.get("list_dir").is_some());
        assert!(registry.get("grep_search").is_some());
        assert!(registry.get("file_write").is_none());
        assert!(registry.get("shell_exec").is_none());
        assert!(registry.get("spawn_agent").is_none());
    }

    #[test]
    fn worker_registry_has_all_standard_tools() {
        let registry = registry_for_agent_type(AgentType::Worker);
        assert!(registry.get("file_read").is_some());
        assert!(registry.get("file_write").is_some());
        assert!(registry.get("file_patch").is_some());
        assert!(registry.get("grep_search").is_some());
        assert!(registry.get("list_dir").is_some());
        assert!(registry.get("shell_exec").is_some());
        assert!(registry.get("spawn_agent").is_none());
    }

    #[test]
    fn plan_registry_matches_explore() {
        let plan_registry = registry_for_agent_type(AgentType::Plan);
        let explore_registry = registry_for_agent_type(AgentType::Explore);

        // Both should have the same tools.
        assert!(plan_registry.get("file_read").is_some());
        assert!(plan_registry.get("list_dir").is_some());
        assert!(plan_registry.get("grep_search").is_some());
        assert!(plan_registry.get("file_write").is_none());

        assert!(explore_registry.get("file_read").is_some());
        assert!(explore_registry.get("list_dir").is_some());
        assert!(explore_registry.get("grep_search").is_some());
        assert!(explore_registry.get("file_write").is_none());
    }

    #[test]
    fn custom_agent_kind_registry() {
        let toml = r#"
name = "test-custom"
description = "Test"
instructions = "Do things."

[tools]
allow = ["file_read", "grep_search"]
"#;
        let def = crate::custom::CustomAgentDef::from_toml(toml, "test.toml")
            .unwrap_or_else(|e| panic!("{e}"));
        let kind = AgentKind::Custom(std::sync::Arc::new(def));
        let registry = registry_for_agent_kind(&kind);

        assert!(registry.get("file_read").is_some());
        assert!(registry.get("grep_search").is_some());
        assert!(registry.get("file_write").is_none());
        assert!(registry.get("shell_exec").is_none());
    }

    /// Without a [`RunToolProvider`], the harnessed agents see only the
    /// subset of their allowed tools that lives in
    /// [`tmg_tools::default_registry`].
    #[test]
    fn harnessed_agents_without_provider_skip_run_aware_tools() {
        let initializer = registry_for_agent_type(AgentType::Initializer);
        // Stateless tools present:
        assert!(initializer.get("file_read").is_some());
        assert!(initializer.get("file_write").is_some());
        assert!(initializer.get("list_dir").is_some());
        assert!(initializer.get("shell_exec").is_some());
        // Run-aware tool absent because no provider was supplied:
        assert!(initializer.get("progress_append").is_none());

        let qa = registry_for_agent_type(AgentType::Qa);
        // Stateless tool present:
        assert!(qa.get("file_read").is_some());
        // Run-aware tools absent:
        assert!(qa.get("feature_list_read").is_none());
        assert!(qa.get("feature_list_mark_passing").is_none());
    }

    /// A test [`RunToolProvider`] that records which names are asked for
    /// and registers a sentinel tool for each "Run-aware" name.
    struct TestRunToolProvider {
        run_aware_names: Vec<&'static str>,
    }

    impl RunToolProvider for TestRunToolProvider {
        fn register_run_tool(&self, registry: &mut ToolRegistry, name: &str) -> bool {
            // Match against the static list so each registered stub
            // can carry a `&'static str` name (required by `Tool::name`).
            for static_name in &self.run_aware_names {
                if *static_name == name {
                    registry.register(StubTool { static_name });
                    return true;
                }
            }
            false
        }
    }

    /// Minimal stand-in tool used by `TestRunToolProvider` so registry
    /// presence can be asserted.
    struct StubTool {
        static_name: &'static str,
    }

    impl tmg_tools::Tool for StubTool {
        fn name(&self) -> &'static str {
            self.static_name
        }

        fn description(&self) -> &'static str {
            "stub"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        fn execute<'a>(
            &'a self,
            _params: serde_json::Value,
            _ctx: &'a tmg_sandbox::SandboxContext,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<tmg_tools::ToolResult, tmg_tools::ToolError>,
                    > + Send
                    + 'a,
            >,
        > {
            Box::pin(async { Ok(tmg_tools::ToolResult::success("ok")) })
        }
    }

    /// Acceptance: with a provider that registers all three Run-aware
    /// tools, the QA registry contains `feature_list_mark_passing`.
    #[test]
    fn qa_registry_with_provider_contains_mark_passing() {
        let provider: Arc<dyn RunToolProvider> = Arc::new(TestRunToolProvider {
            run_aware_names: vec![
                "progress_append",
                "feature_list_read",
                "feature_list_mark_passing",
            ],
        });
        let kind = AgentKind::Builtin(AgentType::Qa);
        let registry = registry_for_agent_kind_with_run_provider(&kind, &*provider);

        assert!(registry.get("feature_list_read").is_some());
        assert!(registry.get("feature_list_mark_passing").is_some());
        assert!(registry.get("file_read").is_some());
        // QA must not see other Run-aware tools it didn't ask for.
        assert!(registry.get("progress_append").is_none());
    }

    /// Acceptance: only the QA subagent gets `feature_list_mark_passing`
    /// in its registry. Initializer / Tester / Worker / Explore / Plan
    /// must NOT, even when a provider that knows about it is wired in.
    /// This is the core permission test for the issue.
    #[test]
    fn only_qa_can_call_feature_list_mark_passing() {
        let provider: Arc<dyn RunToolProvider> = Arc::new(TestRunToolProvider {
            run_aware_names: vec![
                "progress_append",
                "feature_list_read",
                "feature_list_mark_passing",
            ],
        });

        let positive = AgentKind::Builtin(AgentType::Qa);
        let qa_registry = registry_for_agent_kind_with_run_provider(&positive, &*provider);
        assert!(
            qa_registry.get("feature_list_mark_passing").is_some(),
            "qa registry MUST contain feature_list_mark_passing",
        );

        for agent_type in [
            AgentType::Initializer,
            AgentType::Tester,
            AgentType::Worker,
            AgentType::Explore,
            AgentType::Plan,
        ] {
            let kind = AgentKind::Builtin(agent_type);
            let registry = registry_for_agent_kind_with_run_provider(&kind, &*provider);
            assert!(
                registry.get("feature_list_mark_passing").is_none(),
                "{} registry MUST NOT contain feature_list_mark_passing",
                agent_type.name(),
            );
            // feature_list_read is also QA-only in the current spec.
            if !matches!(agent_type, AgentType::Qa) {
                assert!(
                    registry.get("feature_list_read").is_none(),
                    "{} registry MUST NOT contain feature_list_read",
                    agent_type.name(),
                );
            }
        }
    }

    /// The provider's `register_run_tool` is only called for names that
    /// are NOT in the stateless default registry. This keeps the
    /// stateless-vs-stateful separation enforced by construction.
    #[test]
    fn provider_only_consulted_for_run_aware_names() {
        use std::sync::Mutex;

        struct RecordingProvider {
            calls: Arc<Mutex<Vec<String>>>,
        }

        impl RunToolProvider for RecordingProvider {
            fn register_run_tool(&self, _registry: &mut ToolRegistry, name: &str) -> bool {
                if let Ok(mut calls) = self.calls.lock() {
                    calls.push(name.to_owned());
                }
                false
            }
        }

        let calls: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let provider: Arc<dyn RunToolProvider> = Arc::new(RecordingProvider {
            calls: Arc::clone(&calls),
        });

        // Worker only references stateless tools — provider must not be
        // consulted at all.
        let _ = registry_for_agent_kind_with_run_provider(
            &AgentKind::Builtin(AgentType::Worker),
            &*provider,
        );
        assert!(
            calls.lock().map(|c| c.is_empty()).unwrap_or(true),
            "provider unexpectedly invoked for Worker (stateless-only): {:?}",
            calls.lock().map(|c| c.clone()),
        );

        // Initializer references `progress_append`. The provider should
        // be consulted for that one name (and only that one).
        let _ = registry_for_agent_kind_with_run_provider(
            &AgentKind::Builtin(AgentType::Initializer),
            &*provider,
        );
        let recorded = calls.lock().map(|c| c.clone()).unwrap_or_default();
        assert_eq!(
            recorded,
            vec!["progress_append"],
            "expected provider to be consulted exactly for progress_append; got {recorded:?}",
        );
    }
}
