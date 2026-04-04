//! Custom subagent definition: TOML-based parsing and validation.
//!
//! A custom subagent is defined by a `.toml` file with the following
//! structure:
//!
//! ```toml
//! name = "reviewer"
//! description = "A code review subagent"
//! model = "codellama"                        # optional
//! endpoint = "http://localhost:8081"          # optional
//! sandbox_mode = "read_only"                 # optional
//! instructions = "Review the code carefully" # required
//!
//! [tools]
//! allow = ["file_read", "grep_search"]       # required
//! ```

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use tmg_sandbox::SandboxMode;

use crate::config::AgentType;
use crate::error::AgentError;

/// Known tool names in the registry.
///
/// Used for validation warnings. This list should be kept in sync with
/// [`tmg_tools::default_registry`].
const KNOWN_TOOL_NAMES: &[&str] = &[
    "file_read",
    "file_write",
    "file_patch",
    "grep_search",
    "list_dir",
    "shell_exec",
];

/// Validate that an agent name matches `[a-zA-Z][a-zA-Z0-9_-]*`.
fn is_valid_agent_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    first.is_ascii_alphabetic() && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Raw TOML representation of a custom subagent definition.
///
/// All fields are `Option` so that serde can parse any TOML file, and
/// validation is done explicitly via [`CustomAgentDef::from_toml`].
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawCustomAgentDef {
    name: Option<String>,
    description: Option<String>,
    model: Option<String>,
    endpoint: Option<String>,
    sandbox_mode: Option<SandboxMode>,
    instructions: Option<String>,
    tools: Option<RawToolsSection>,
}

/// Raw `[tools]` section from the TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawToolsSection {
    allow: Option<Vec<String>>,
}

/// A validated custom subagent definition.
///
/// Does not derive `Deserialize` to force all parsing through
/// [`CustomAgentDef::from_toml`], which performs validation that
/// serde cannot express (e.g. `allowed_tools` maps from `tools.allow`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CustomAgentDef {
    /// The unique name of this custom agent (e.g. `"reviewer"`).
    name: String,

    /// A human-readable description of this agent's purpose.
    description: String,

    /// Optional model override (uses the default model if `None`).
    model: Option<String>,

    /// Optional endpoint override (uses the default endpoint if `None`).
    endpoint: Option<String>,

    /// Optional sandbox mode override.
    sandbox_mode: Option<SandboxMode>,

    /// The system instructions for this agent.
    instructions: String,

    /// The list of tool names this agent is allowed to use.
    allowed_tools: Vec<String>,
}

impl CustomAgentDef {
    /// The unique name of this custom agent.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// A human-readable description of this agent's purpose.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Optional model override (uses the default model if `None`).
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// Optional endpoint override (uses the default endpoint if `None`).
    pub fn endpoint(&self) -> Option<&str> {
        self.endpoint.as_deref()
    }

    /// Optional sandbox mode override.
    pub fn sandbox_mode(&self) -> Option<SandboxMode> {
        self.sandbox_mode
    }

    /// The system instructions for this agent.
    pub fn instructions(&self) -> &str {
        &self.instructions
    }

    /// The list of tool names this agent is allowed to use.
    pub fn allowed_tools(&self) -> &[String] {
        &self.allowed_tools
    }
}

/// The source priority of a discovered custom agent definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AgentSource {
    /// `.tsumugi/agents/` in the project root (highest priority).
    ProjectLocal = 0,
    /// `~/.config/tsumugi/agents/` in the global config directory.
    UserGlobal = 1,
}

impl std::fmt::Display for AgentSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProjectLocal => write!(f, ".tsumugi/agents"),
            Self::UserGlobal => write!(f, "~/.config/tsumugi/agents"),
        }
    }
}

/// Metadata about a discovered custom agent (definition + source info).
#[derive(Debug, Clone, PartialEq)]
pub struct CustomAgentMeta {
    /// The validated agent definition.
    pub def: CustomAgentDef,

    /// Where this agent was discovered from.
    pub source: AgentSource,

    /// The absolute path to the TOML definition file.
    pub path: PathBuf,
}

impl CustomAgentDef {
    /// Parse and validate a custom agent definition from TOML content.
    ///
    /// Uses `let-else` for required field validation, returning
    /// structured errors for each missing field.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::TomlParse`] if the TOML is syntactically
    /// invalid, or [`AgentError::InvalidCustomAgent`] if a required field
    /// is missing.
    pub fn from_toml(content: &str, file_path: &str) -> Result<Self, AgentError> {
        let raw: RawCustomAgentDef =
            toml::from_str(content).map_err(|e| AgentError::TomlParse {
                path: file_path.to_owned(),
                source: e,
            })?;

        let Some(name) = raw.name else {
            return Err(AgentError::InvalidCustomAgent {
                path: file_path.to_owned(),
                reason: "missing required field: name".to_owned(),
            });
        };

        // Validate name format: must start with a letter, followed by
        // letters, digits, underscores, or hyphens.
        if !is_valid_agent_name(&name) {
            return Err(AgentError::InvalidCustomAgent {
                path: file_path.to_owned(),
                reason: format!("invalid agent name '{name}': must match [a-zA-Z][a-zA-Z0-9_-]*"),
            });
        }

        // Reject names that collide with built-in agent types.
        if AgentType::from_name(&name).is_some() {
            return Err(AgentError::InvalidCustomAgent {
                path: file_path.to_owned(),
                reason: format!("agent name '{name}' conflicts with built-in agent type"),
            });
        }

        let Some(description) = raw.description else {
            return Err(AgentError::InvalidCustomAgent {
                path: file_path.to_owned(),
                reason: "missing required field: description".to_owned(),
            });
        };

        let Some(instructions) = raw.instructions else {
            return Err(AgentError::InvalidCustomAgent {
                path: file_path.to_owned(),
                reason: "missing required field: instructions".to_owned(),
            });
        };

        let Some(tools_section) = raw.tools else {
            return Err(AgentError::InvalidCustomAgent {
                path: file_path.to_owned(),
                reason: "missing required section: [tools]".to_owned(),
            });
        };

        let Some(allowed_tools) = tools_section.allow else {
            return Err(AgentError::InvalidCustomAgent {
                path: file_path.to_owned(),
                reason: "missing required field: tools.allow".to_owned(),
            });
        };

        if allowed_tools.is_empty() {
            return Err(AgentError::InvalidCustomAgent {
                path: file_path.to_owned(),
                reason: "tools.allow must contain at least one tool name".to_owned(),
            });
        }

        // Reject spawn_agent in allowed tools to prevent nesting.
        if allowed_tools.iter().any(|t| t == "spawn_agent") {
            return Err(AgentError::InvalidCustomAgent {
                path: file_path.to_owned(),
                reason: "tools.allow must not include 'spawn_agent' (nesting is forbidden)"
                    .to_owned(),
            });
        }

        // Validate that all tool names are recognized.
        let unknown: Vec<&String> = allowed_tools
            .iter()
            .filter(|t| !KNOWN_TOOL_NAMES.contains(&t.as_str()))
            .collect();
        if !unknown.is_empty() {
            let names: Vec<&str> = unknown.iter().map(|s| s.as_str()).collect();
            return Err(AgentError::InvalidCustomAgent {
                path: file_path.to_owned(),
                reason: format!(
                    "unrecognized tool name(s): {}. Known tools: {}",
                    names.join(", "),
                    KNOWN_TOOL_NAMES.join(", "),
                ),
            });
        }

        Ok(Self {
            name,
            description,
            model: raw.model,
            endpoint: raw.endpoint,
            sandbox_mode: raw.sandbox_mode,
            instructions,
            allowed_tools,
        })
    }

    /// Serialize this definition back to TOML.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::TomlSerialize`] if serialization fails.
    pub fn to_toml(&self) -> Result<String, AgentError> {
        // Build a raw struct for serialization to match the expected format.
        let raw = RawCustomAgentDef {
            name: Some(self.name.clone()),
            description: Some(self.description.clone()),
            model: self.model.clone(),
            endpoint: self.endpoint.clone(),
            sandbox_mode: self.sandbox_mode,
            instructions: Some(self.instructions.clone()),
            tools: Some(RawToolsSection {
                allow: Some(self.allowed_tools.clone()),
            }),
        };
        toml::to_string_pretty(&raw).map_err(|e| AgentError::TomlSerialize {
            reason: e.to_string(),
        })
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions")]
#[expect(clippy::unwrap_used, reason = "test assertions")]
mod tests {
    use super::*;

    #[test]
    fn parse_full_custom_agent() {
        let toml = r#"
name = "reviewer"
description = "A code review subagent"
model = "codellama"
endpoint = "http://localhost:8081"
sandbox_mode = "read_only"
instructions = "Review the code carefully and provide feedback."

[tools]
allow = ["file_read", "grep_search"]
"#;

        let def = CustomAgentDef::from_toml(toml, "test.toml").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(def.name, "reviewer");
        assert_eq!(def.description, "A code review subagent");
        assert_eq!(def.model.as_deref(), Some("codellama"));
        assert_eq!(def.endpoint.as_deref(), Some("http://localhost:8081"));
        assert_eq!(def.sandbox_mode, Some(SandboxMode::ReadOnly));
        assert_eq!(
            def.instructions,
            "Review the code carefully and provide feedback."
        );
        assert_eq!(def.allowed_tools, vec!["file_read", "grep_search"]);
    }

    #[test]
    fn parse_minimal_custom_agent() {
        let toml = r#"
name = "minimal"
description = "A minimal agent"
instructions = "Do the thing."

[tools]
allow = ["file_read"]
"#;

        let def = CustomAgentDef::from_toml(toml, "test.toml").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(def.name, "minimal");
        assert!(def.model.is_none());
        assert!(def.endpoint.is_none());
        assert!(def.sandbox_mode.is_none());
    }

    #[test]
    fn missing_name_returns_error() {
        let toml = r#"
description = "No name"
instructions = "Do something."

[tools]
allow = ["file_read"]
"#;

        let err = CustomAgentDef::from_toml(toml, "bad.toml").unwrap_err();
        assert!(err.to_string().contains("missing required field: name"));
    }

    #[test]
    fn missing_description_returns_error() {
        let toml = r#"
name = "test"
instructions = "Do something."

[tools]
allow = ["file_read"]
"#;

        let err = CustomAgentDef::from_toml(toml, "bad.toml").unwrap_err();
        assert!(
            err.to_string()
                .contains("missing required field: description")
        );
    }

    #[test]
    fn missing_instructions_returns_error() {
        let toml = r#"
name = "test"
description = "Test"

[tools]
allow = ["file_read"]
"#;

        let err = CustomAgentDef::from_toml(toml, "bad.toml").unwrap_err();
        assert!(
            err.to_string()
                .contains("missing required field: instructions")
        );
    }

    #[test]
    fn missing_tools_section_returns_error() {
        let toml = r#"
name = "test"
description = "Test"
instructions = "Do something."
"#;

        let err = CustomAgentDef::from_toml(toml, "bad.toml").unwrap_err();
        assert!(
            err.to_string()
                .contains("missing required section: [tools]")
        );
    }

    #[test]
    fn missing_tools_allow_returns_error() {
        let toml = r#"
name = "test"
description = "Test"
instructions = "Do something."

[tools]
"#;

        let err = CustomAgentDef::from_toml(toml, "bad.toml").unwrap_err();
        assert!(
            err.to_string()
                .contains("missing required field: tools.allow")
        );
    }

    #[test]
    fn empty_tools_allow_returns_error() {
        let toml = r#"
name = "test"
description = "Test"
instructions = "Do something."

[tools]
allow = []
"#;

        let err = CustomAgentDef::from_toml(toml, "bad.toml").unwrap_err();
        assert!(
            err.to_string()
                .contains("tools.allow must contain at least one tool name")
        );
    }

    #[test]
    fn spawn_agent_in_tools_rejected() {
        let toml = r#"
name = "test"
description = "Test"
instructions = "Do something."

[tools]
allow = ["file_read", "spawn_agent"]
"#;

        let err = CustomAgentDef::from_toml(toml, "bad.toml").unwrap_err();
        assert!(err.to_string().contains("must not include 'spawn_agent'"));
    }

    #[test]
    fn builtin_name_collision_rejected() {
        let toml = r#"
name = "explore"
description = "Test"
instructions = "Do something."

[tools]
allow = ["file_read"]
"#;

        let err = CustomAgentDef::from_toml(toml, "bad.toml").unwrap_err();
        assert!(
            err.to_string()
                .contains("conflicts with built-in agent type")
        );
    }

    #[test]
    fn invalid_name_format_rejected() {
        let toml = r#"
name = "123bad"
description = "Test"
instructions = "Do something."

[tools]
allow = ["file_read"]
"#;

        let err = CustomAgentDef::from_toml(toml, "bad.toml").unwrap_err();
        assert!(err.to_string().contains("invalid agent name"));
    }

    #[test]
    fn unrecognized_tool_name_rejected() {
        let toml = r#"
name = "test"
description = "Test"
instructions = "Do something."

[tools]
allow = ["file_read", "nonexistent_tool"]
"#;

        let err = CustomAgentDef::from_toml(toml, "bad.toml").unwrap_err();
        assert!(err.to_string().contains("unrecognized tool name"));
    }

    #[test]
    fn valid_name_formats_accepted() {
        for name in &["myAgent", "a", "my-agent", "my_agent", "Agent123"] {
            let toml = format!(
                r#"
name = "{name}"
description = "Test"
instructions = "Do something."

[tools]
allow = ["file_read"]
"#
            );
            assert!(
                CustomAgentDef::from_toml(&toml, "test.toml").is_ok(),
                "expected name '{name}' to be valid"
            );
        }
    }

    #[test]
    fn invalid_toml_syntax_returns_error() {
        let err = CustomAgentDef::from_toml("not valid toml [[[", "bad.toml").unwrap_err();
        assert!(err.to_string().contains("toml parse error"));
    }

    #[test]
    fn roundtrip_serialize_deserialize() {
        let original = CustomAgentDef {
            name: "roundtrip".to_owned(),
            description: "Roundtrip test".to_owned(),
            model: Some("test-model".to_owned()),
            endpoint: Some("http://localhost:9999".to_owned()),
            sandbox_mode: Some(SandboxMode::ReadOnly),
            instructions: "Test instructions.".to_owned(),
            allowed_tools: vec!["file_read".to_owned(), "grep_search".to_owned()],
        };

        let toml_str = original.to_toml().unwrap_or_else(|e| panic!("{e}"));
        let parsed = CustomAgentDef::from_toml(&toml_str, "roundtrip.toml")
            .unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(original, parsed);
    }

    #[test]
    fn roundtrip_minimal() {
        let original = CustomAgentDef {
            name: "minimal".to_owned(),
            description: "Minimal".to_owned(),
            model: None,
            endpoint: None,
            sandbox_mode: None,
            instructions: "Do it.".to_owned(),
            allowed_tools: vec!["file_read".to_owned()],
        };

        let toml_str = original.to_toml().unwrap_or_else(|e| panic!("{e}"));
        let parsed =
            CustomAgentDef::from_toml(&toml_str, "minimal.toml").unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(original, parsed);
    }

    #[expect(clippy::expect_used, reason = "proptest assertions")]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_sandbox_mode() -> impl Strategy<Value = Option<SandboxMode>> {
            prop_oneof![
                Just(None),
                Just(Some(SandboxMode::ReadOnly)),
                Just(Some(SandboxMode::WorkspaceWrite)),
                Just(Some(SandboxMode::Full)),
            ]
        }

        /// Generate a non-empty string suitable for TOML values.
        /// Avoids characters that would break TOML parsing.
        fn arb_toml_safe_string() -> impl Strategy<Value = String> {
            "[a-zA-Z][a-zA-Z0-9 _-]{0,30}[a-zA-Z0-9]"
        }

        /// Generate a valid agent name that won't collide with built-in names.
        fn arb_agent_name() -> impl Strategy<Value = String> {
            "[a-zA-Z][a-zA-Z0-9_-]{2,20}"
                .prop_filter("must not collide with built-in names", |name| {
                    crate::config::AgentType::from_name(name).is_none()
                })
        }

        fn arb_tool_name() -> impl Strategy<Value = String> {
            // Use known valid tool names to avoid the "spawn_agent" rejection.
            prop_oneof![
                Just("file_read".to_owned()),
                Just("file_write".to_owned()),
                Just("file_patch".to_owned()),
                Just("grep_search".to_owned()),
                Just("list_dir".to_owned()),
                Just("shell_exec".to_owned()),
            ]
        }

        fn arb_optional_string() -> impl Strategy<Value = Option<String>> {
            prop_oneof![Just(None), arb_toml_safe_string().prop_map(Some),]
        }

        fn arb_optional_endpoint() -> impl Strategy<Value = Option<String>> {
            prop_oneof![
                Just(None),
                Just(Some("http://localhost:8080".to_owned())),
                Just(Some("http://localhost:8081".to_owned())),
                Just(Some("https://example.com".to_owned())),
            ]
        }

        fn arb_custom_agent_def() -> impl Strategy<Value = CustomAgentDef> {
            (
                arb_agent_name(),
                arb_toml_safe_string(),
                arb_optional_string(),
                arb_optional_endpoint(),
                arb_sandbox_mode(),
                arb_toml_safe_string(),
                proptest::collection::vec(arb_tool_name(), 1..4),
            )
                .prop_map(
                    |(name, description, model, endpoint, sandbox_mode, instructions, tools)| {
                        // Deduplicate tools.
                        let mut unique_tools: Vec<String> = Vec::new();
                        for t in tools {
                            if !unique_tools.contains(&t) {
                                unique_tools.push(t);
                            }
                        }
                        CustomAgentDef {
                            name,
                            description,
                            model,
                            endpoint,
                            sandbox_mode,
                            instructions,
                            allowed_tools: unique_tools,
                        }
                    },
                )
        }

        proptest! {
            #[test]
            fn toml_roundtrip(def in arb_custom_agent_def()) {
                let toml_str = def.to_toml().expect("serialize");
                let parsed = CustomAgentDef::from_toml(&toml_str, "prop.toml")
                    .expect("parse");
                prop_assert_eq!(&def, &parsed);
            }
        }
    }
}
