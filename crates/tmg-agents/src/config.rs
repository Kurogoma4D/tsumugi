//! Subagent configuration types.
//!
//! Defines [`AgentType`] (the built-in subagent kinds),
//! [`AgentKind`] (built-in or custom), and
//! [`SubagentConfig`] (the parameters for spawning a subagent).

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use tmg_sandbox::SandboxMode;

use crate::custom::CustomAgentDef;

/// The built-in subagent types.
///
/// Each type defines a different set of allowed tools and a tailored
/// system prompt for its purpose.
///
/// The harnessed-run subagents (`Initializer`, `Tester`, `Qa`) reference
/// Run-scoped tools (`progress_append`, `feature_list_read`,
/// `feature_list_mark_passing`) that are not part of
/// [`tmg_tools::default_registry`]. Registration of those tools is the
/// responsibility of the spawner via the
/// [`RunToolProvider`](crate::builtins::RunToolProvider) plumbing — the
/// subagent declares the names here, and they are filtered out of the
/// final registry when no provider is wired in (e.g. ad-hoc runs).
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

    /// Project bootstrap agent (SPEC §5.2).
    ///
    /// Generates `features.json`, `init.sh`, an initial `progress.md`,
    /// and the first git commit for a harnessed run.
    ///
    /// Allowed tools: `file_read`, `file_write`, `list_dir`,
    /// `shell_exec`, `progress_append`.
    Initializer,

    /// Smoke-test agent (SPEC §5.2).
    ///
    /// Runs the project's own quality gates (unit tests, lint, build,
    /// screenshot capture if applicable) and reports the outcome.
    ///
    /// Allowed tools: `shell_exec`, `file_read`.
    Tester,

    /// Acceptance-criteria QA agent (SPEC §5.2).
    ///
    /// Read-only inspector that walks `features.json`, verifies each
    /// feature against its acceptance criteria, and is the **only**
    /// subagent permitted to flip a feature's `passes` field via
    /// `feature_list_mark_passing`.
    ///
    /// Allowed tools: `feature_list_read`, `feature_list_mark_passing`,
    /// `file_read`.
    Qa,
}

impl AgentType {
    /// All built-in agent types.
    pub const ALL: &'static [Self] = &[
        Self::Explore,
        Self::Worker,
        Self::Plan,
        Self::Initializer,
        Self::Tester,
        Self::Qa,
    ];

    /// Return the tool names allowed for this agent type.
    ///
    /// `spawn_agent` is never included (nesting is forbidden).
    ///
    /// The harnessed agents (`Initializer`, `Tester`, `Qa`) include
    /// Run-scoped tool names (`progress_append`, `feature_list_read`,
    /// `feature_list_mark_passing`); the spawner is responsible for
    /// gating those tools on the presence of an active `RunRunner`
    /// (see [`RunToolProvider`](crate::builtins::RunToolProvider)).
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
            Self::Initializer => &[
                "file_read",
                "file_write",
                "list_dir",
                "shell_exec",
                "progress_append",
            ],
            Self::Tester => &["shell_exec", "file_read"],
            Self::Qa => &[
                "feature_list_read",
                "feature_list_mark_passing",
                "file_read",
            ],
        }
    }

    /// Return the sandbox mode this agent type expects to run under.
    ///
    /// SPEC §5.2 gates:
    ///
    /// - `Initializer`: [`SandboxMode::WorkspaceWrite`] — generates
    ///   `features.json` / `init.sh` / `progress.md` and commits.
    /// - `Tester`: [`SandboxMode::WorkspaceWrite`] — needs to persist
    ///   screenshots and other test artefacts.
    /// - `Qa`: [`SandboxMode::ReadOnly`] — verifies acceptance criteria
    ///   without mutating the workspace; the only state change is the
    ///   `passes` flag, which is mediated by
    ///   `feature_list_mark_passing`.
    /// - `Explore` / `Plan`: [`SandboxMode::ReadOnly`].
    /// - `Worker`: [`SandboxMode::WorkspaceWrite`].
    #[must_use]
    pub fn sandbox_mode(&self) -> SandboxMode {
        match self {
            Self::Explore | Self::Plan | Self::Qa => SandboxMode::ReadOnly,
            Self::Worker | Self::Initializer | Self::Tester => SandboxMode::WorkspaceWrite,
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
            Self::Initializer => {
                "You are the initializer subagent. Your single responsibility is to \
                 bootstrap a fresh harnessed project so subsequent sessions have a \
                 stable starting point. Produce four artefacts in this order: \
                 (1) `features.json` -- an inventory of acceptance-test features the \
                 project must satisfy. Cap the list at between 30 and 50 entries; \
                 fewer than 30 is too coarse and more than 50 makes per-session \
                 review intractable. Each entry needs `id`, `category`, \
                 `description`, `steps`, and `passes: false`. \
                 (2) `init.sh` -- an idempotent shell script that prepares the \
                 workspace for an agent (install deps, build, seed fixtures). \
                 Keep it deterministic and quiet on success. \
                 (3) `progress.md` -- the initial progress log. Use \
                 `progress_append` to record what bootstrap did so future \
                 sessions can re-discover the starting state. \
                 (4) An initial git commit on the current branch containing the \
                 generated files, with a clear message. \
                 You have read+write access to the workspace via `file_read`, \
                 `file_write`, `list_dir`, and `shell_exec`. Do not invent \
                 features the user did not request; mine the existing repo for \
                 features when possible."
            }
            Self::Tester => {
                "You are the tester subagent. Smoke-test the workspace per SPEC §5.2: \
                 run the project's quality gates (unit tests, integration tests, \
                 build, lint, format check) using `shell_exec`, and read failure \
                 output back via `file_read` when needed. Capture screenshots or \
                 other test artefacts to disk if the project supports it. Browser \
                 automation tools are a future addition (browser MCP); until then, \
                 focus on shell-driven test commands. Report results as a concise \
                 pass/fail summary with the exact commands run, exit codes, and \
                 short failure excerpts. Do not modify source files; use the \
                 workspace as a read-mostly target with writes limited to \
                 test-output directories."
            }
            Self::Qa => {
                "You are the QA subagent. Walk `features.json` via `feature_list_read`, \
                 verify each entry's acceptance criteria against the live workspace \
                 using `file_read`, and call `feature_list_mark_passing` for every \
                 feature you can confirm is now passing. You are the **only** \
                 subagent permitted to invoke `feature_list_mark_passing`; misuse \
                 corrupts the harnessed run's pass-fail accounting. Be conservative: \
                 if a feature's evidence is ambiguous, leave `passes` as `false` \
                 and explain why. Your environment is read-only -- you cannot \
                 modify any source file. Report the set of features marked \
                 passing and any features you explicitly chose not to mark."
            }
        }
    }

    /// A human-readable description of this agent type.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Explore => "Read-only codebase exploration (file_read, list_dir, grep_search)",
            Self::Worker => "Full tool access worker (all tools except spawn_agent)",
            Self::Plan => "Read-only planning agent (file_read, list_dir, grep_search)",
            Self::Initializer => {
                "Project bootstrap (features.json/init.sh/progress.md + initial commit)"
            }
            Self::Tester => "Smoke-test runner (shell-driven test commands, capture artefacts)",
            Self::Qa => {
                "Acceptance-criteria QA (read-only; only agent that can mark features passing)"
            }
        }
    }

    /// The display name of this agent type.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Explore => "explore",
            Self::Worker => "worker",
            Self::Plan => "plan",
            Self::Initializer => "initializer",
            Self::Tester => "tester",
            Self::Qa => "qa",
        }
    }

    /// Parse an agent type from a string name.
    ///
    /// Delegates to serde deserialization so the canonical name mapping
    /// is defined in a single place (`#[serde(rename_all = "snake_case")]`).
    pub fn from_name(name: &str) -> Option<Self> {
        let quoted = format!("\"{name}\"");
        serde_json::from_str(&quoted).ok()
    }
}

impl std::fmt::Display for AgentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Distinguishes between a built-in agent type and a custom (TOML-defined)
/// agent.
#[derive(Debug, Clone)]
pub enum AgentKind {
    /// A built-in agent type (explore, worker, plan, initializer, tester, qa).
    Builtin(AgentType),

    /// A custom agent loaded from a TOML definition file.
    ///
    /// Wrapped in `Arc` to avoid cloning the entire definition each time
    /// the agent kind is passed around.
    Custom(Arc<CustomAgentDef>),
}

impl AgentKind {
    /// Return the display name of this agent kind.
    pub fn name(&self) -> &str {
        match self {
            Self::Builtin(t) => t.name(),
            Self::Custom(def) => def.name(),
        }
    }

    /// Return a human-readable description.
    pub fn description(&self) -> &str {
        match self {
            Self::Builtin(t) => t.description(),
            Self::Custom(def) => def.description(),
        }
    }

    /// Return the system prompt / instructions for this agent.
    pub fn system_prompt(&self) -> &str {
        match self {
            Self::Builtin(t) => t.system_prompt(),
            Self::Custom(def) => def.instructions(),
        }
    }

    /// Return the allowed tool names for this agent.
    pub fn allowed_tool_names(&self) -> Vec<&str> {
        match self {
            Self::Builtin(t) => t.allowed_tools().to_vec(),
            Self::Custom(def) => def.allowed_tools().iter().map(String::as_str).collect(),
        }
    }
}

impl std::fmt::Display for AgentKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Configuration for spawning a subagent.
#[derive(Debug, Clone)]
pub struct SubagentConfig {
    /// The kind of subagent to spawn (built-in or custom).
    pub agent_kind: AgentKind,

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
        assert_eq!(
            AgentType::from_name("initializer"),
            Some(AgentType::Initializer)
        );
        assert_eq!(AgentType::from_name("tester"), Some(AgentType::Tester));
        assert_eq!(AgentType::from_name("qa"), Some(AgentType::Qa));
        assert_eq!(AgentType::from_name("unknown"), None);
    }

    #[test]
    fn agent_type_display() {
        assert_eq!(AgentType::Explore.to_string(), "explore");
        assert_eq!(AgentType::Worker.to_string(), "worker");
        assert_eq!(AgentType::Plan.to_string(), "plan");
        assert_eq!(AgentType::Initializer.to_string(), "initializer");
        assert_eq!(AgentType::Tester.to_string(), "tester");
        assert_eq!(AgentType::Qa.to_string(), "qa");
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
        assert_eq!(AgentType::ALL.len(), 6);
        assert!(AgentType::ALL.contains(&AgentType::Initializer));
        assert!(AgentType::ALL.contains(&AgentType::Tester));
        assert!(AgentType::ALL.contains(&AgentType::Qa));
    }

    #[test]
    fn initializer_spec() {
        let tools = AgentType::Initializer.allowed_tools();
        assert_eq!(
            tools,
            &[
                "file_read",
                "file_write",
                "list_dir",
                "shell_exec",
                "progress_append",
            ],
        );
        assert_eq!(
            AgentType::Initializer.sandbox_mode(),
            SandboxMode::WorkspaceWrite,
        );
        let prompt = AgentType::Initializer.system_prompt();
        assert!(!prompt.is_empty());
        assert!(
            prompt.contains("30") && prompt.contains("50"),
            "initializer prompt must cite the 30-50 feature ceiling: {prompt}"
        );
        assert!(prompt.contains("features.json"));
        assert!(prompt.contains("init.sh"));
        assert!(prompt.contains("git commit"));
    }

    #[test]
    fn tester_spec() {
        let tools = AgentType::Tester.allowed_tools();
        assert_eq!(tools, &["shell_exec", "file_read"]);
        assert_eq!(
            AgentType::Tester.sandbox_mode(),
            SandboxMode::WorkspaceWrite,
        );
        let prompt = AgentType::Tester.system_prompt();
        assert!(!prompt.is_empty());
        assert!(
            prompt.to_ascii_lowercase().contains("smoke"),
            "tester prompt must describe smoke testing: {prompt}"
        );
        assert!(
            prompt.to_ascii_lowercase().contains("browser"),
            "tester prompt must mention browser MCP as a future addition: {prompt}"
        );
    }

    #[test]
    fn qa_spec() {
        let tools = AgentType::Qa.allowed_tools();
        assert_eq!(
            tools,
            &[
                "feature_list_read",
                "feature_list_mark_passing",
                "file_read"
            ],
        );
        assert_eq!(AgentType::Qa.sandbox_mode(), SandboxMode::ReadOnly);
        let prompt = AgentType::Qa.system_prompt();
        assert!(!prompt.is_empty());
        assert!(
            prompt.contains("feature_list_mark_passing"),
            "qa prompt must explicitly mention feature_list_mark_passing: {prompt}"
        );
        assert!(
            prompt.to_ascii_lowercase().contains("only"),
            "qa prompt must emphasize that it is the ONLY subagent allowed to mark passing: {prompt}"
        );
    }

    #[test]
    fn descriptions_are_non_empty_for_all_types() {
        for agent_type in AgentType::ALL {
            assert!(
                !agent_type.description().is_empty(),
                "description for {agent_type} is empty",
            );
            assert!(
                !agent_type.system_prompt().is_empty(),
                "system_prompt for {agent_type} is empty",
            );
        }
    }

    #[test]
    fn names_round_trip_for_all_types() {
        for agent_type in AgentType::ALL {
            let name = agent_type.name();
            assert_eq!(
                AgentType::from_name(name),
                Some(*agent_type),
                "round-trip failed for {name}",
            );
        }
    }

    #[test]
    fn serde_roundtrip() {
        let json = serde_json::to_string(&AgentType::Explore).ok();
        assert_eq!(json.as_deref(), Some("\"explore\""));

        let parsed: AgentType = serde_json::from_str("\"worker\"")
            .ok()
            .unwrap_or(AgentType::Explore);
        assert_eq!(parsed, AgentType::Worker);

        let parsed: AgentType = serde_json::from_str("\"initializer\"")
            .ok()
            .unwrap_or(AgentType::Explore);
        assert_eq!(parsed, AgentType::Initializer);

        let parsed: AgentType = serde_json::from_str("\"qa\"")
            .ok()
            .unwrap_or(AgentType::Explore);
        assert_eq!(parsed, AgentType::Qa);
    }
}
