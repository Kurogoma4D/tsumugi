//! Subagent configuration types.
//!
//! Defines [`AgentType`] (the built-in subagent kinds) and
//! [`SubagentConfig`] (the parameters for spawning a subagent).

use serde::{Deserialize, Serialize};

/// The built-in subagent types.
///
/// Each type defines a different set of allowed tools and a tailored
/// system prompt for its purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentType {
    /// Read-only codebase exploration agent.
    ///
    /// Allowed tools: `file_read`, `list_dir`, `grep_search`.
    Explore,

    /// General-purpose worker agent with full tool access.
    ///
    /// Allowed tools: all tools except `spawn_agent` (no nesting).
    Worker,

    /// Read-only planning agent that produces structured plans.
    ///
    /// Allowed tools: `file_read`, `list_dir`, `grep_search`.
    Plan,
}

impl AgentType {
    /// All built-in agent types.
    pub const ALL: &'static [Self] = &[Self::Explore, Self::Worker, Self::Plan];

    /// Return the tool names allowed for this agent type.
    ///
    /// `spawn_agent` is never included (nesting is forbidden).
    pub fn allowed_tools(&self) -> &'static [&'static str] {
        match self {
            Self::Explore | Self::Plan => &["file_read", "list_dir", "grep_search"],
            Self::Worker => &[
                "file_read",
                "file_write",
                "file_patch",
                "grep_search",
                "list_dir",
                "shell_exec",
            ],
        }
    }

    /// Return the system prompt for this agent type.
    pub fn system_prompt(&self) -> &'static str {
        match self {
            Self::Explore => {
                "You are an explore subagent. Your job is to investigate the codebase \
                 and report findings. You have read-only access to files: you can read \
                 files, list directories, and search with grep. Provide a thorough and \
                 well-structured summary of what you find."
            }
            Self::Worker => {
                "You are a worker subagent. You have full tool access to read, write, \
                 patch files, and run shell commands. Execute the assigned task \
                 efficiently and report the result. You cannot spawn other subagents."
            }
            Self::Plan => {
                "You are a planning subagent. Your job is to analyze the codebase \
                 (read-only) and produce a structured plan. You can read files, list \
                 directories, and search with grep. Output a clear, actionable plan \
                 with numbered steps."
            }
        }
    }

    /// A human-readable description of this agent type.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Explore => "Read-only codebase exploration (file_read, list_dir, grep_search)",
            Self::Worker => "Full tool access worker (all tools except spawn_agent)",
            Self::Plan => "Read-only planning agent (file_read, list_dir, grep_search)",
        }
    }

    /// The display name of this agent type.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Explore => "explore",
            Self::Worker => "worker",
            Self::Plan => "plan",
        }
    }

    /// Parse an agent type from a string name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "explore" => Some(Self::Explore),
            "worker" => Some(Self::Worker),
            "plan" => Some(Self::Plan),
            _ => None,
        }
    }
}

impl std::fmt::Display for AgentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Configuration for spawning a subagent.
#[derive(Debug, Clone)]
pub struct SubagentConfig {
    /// The type of subagent to spawn.
    pub agent_type: AgentType,

    /// The task description for the subagent.
    pub task: String,

    /// Whether to run in the background (`true`) or await completion
    /// (`false`).
    pub background: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_type_from_name() {
        assert_eq!(AgentType::from_name("explore"), Some(AgentType::Explore));
        assert_eq!(AgentType::from_name("worker"), Some(AgentType::Worker));
        assert_eq!(AgentType::from_name("plan"), Some(AgentType::Plan));
        assert_eq!(AgentType::from_name("unknown"), None);
    }

    #[test]
    fn agent_type_display() {
        assert_eq!(AgentType::Explore.to_string(), "explore");
        assert_eq!(AgentType::Worker.to_string(), "worker");
        assert_eq!(AgentType::Plan.to_string(), "plan");
    }

    #[test]
    fn explore_tools_are_read_only() {
        let tools = AgentType::Explore.allowed_tools();
        assert!(tools.contains(&"file_read"));
        assert!(tools.contains(&"list_dir"));
        assert!(tools.contains(&"grep_search"));
        assert!(!tools.contains(&"file_write"));
        assert!(!tools.contains(&"shell_exec"));
        assert!(!tools.contains(&"spawn_agent"));
    }

    #[test]
    fn worker_tools_exclude_spawn_agent() {
        let tools = AgentType::Worker.allowed_tools();
        assert!(!tools.contains(&"spawn_agent"));
        assert!(tools.contains(&"file_read"));
        assert!(tools.contains(&"file_write"));
        assert!(tools.contains(&"shell_exec"));
    }

    #[test]
    fn plan_tools_are_read_only() {
        let tools = AgentType::Plan.allowed_tools();
        assert_eq!(tools, AgentType::Explore.allowed_tools());
    }

    #[test]
    fn all_types_covered() {
        assert_eq!(AgentType::ALL.len(), 3);
    }

    #[test]
    fn serde_roundtrip() {
        let json = serde_json::to_string(&AgentType::Explore).ok();
        assert_eq!(json.as_deref(), Some("\"explore\""));

        let parsed: AgentType = serde_json::from_str("\"worker\"")
            .ok()
            .unwrap_or(AgentType::Explore);
        assert_eq!(parsed, AgentType::Worker);
    }
}
