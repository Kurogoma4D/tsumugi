//! Built-in subagent definitions and tool registry construction.
//!
//! Provides helpers to create a [`ToolRegistry`] filtered to only the
//! tools allowed for a given [`AgentType`] or [`AgentKind`].

use tmg_tools::ToolRegistry;

use crate::config::{AgentKind, AgentType};

/// Create a [`ToolRegistry`] containing only the tools allowed for the
/// given agent type.
///
/// This filters the default registry, keeping only tools whose names
/// appear in [`AgentType::allowed_tools`].
pub fn registry_for_agent_type(agent_type: AgentType) -> ToolRegistry {
    let allowed: Vec<&str> = agent_type.allowed_tools().to_vec();
    registry_for_tool_names(&allowed)
}

/// Create a [`ToolRegistry`] containing only the tools allowed for the
/// given agent kind (built-in or custom).
pub fn registry_for_agent_kind(agent_kind: &AgentKind) -> ToolRegistry {
    let allowed = agent_kind.allowed_tool_names();
    registry_for_tool_names(&allowed)
}

/// Create a [`ToolRegistry`] filtered to only the named tools.
fn registry_for_tool_names(allowed: &[&str]) -> ToolRegistry {
    let full_registry = tmg_tools::default_registry();
    let mut filtered = ToolRegistry::new();

    for name in allowed {
        if full_registry.get(name).is_some() {
            register_tool_by_name(&mut filtered, name);
        }
    }

    filtered
}

/// Register a built-in tool by name into the given registry.
fn register_tool_by_name(registry: &mut ToolRegistry, name: &str) {
    match name {
        "file_read" => registry.register(tmg_tools::tools::FileReadTool),
        "file_write" => registry.register(tmg_tools::tools::FileWriteTool),
        "file_patch" => registry.register(tmg_tools::tools::FilePatchTool),
        "grep_search" => registry.register(tmg_tools::tools::GrepSearchTool),
        "list_dir" => registry.register(tmg_tools::tools::ListDirTool),
        "shell_exec" => registry.register(tmg_tools::tools::ShellExecTool),
        _ => {
            debug_assert!(false, "register_tool_by_name: unknown tool name: {name}");
        }
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions")]
mod tests {
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
}
