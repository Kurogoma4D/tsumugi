//! tmg-agents: Subagent system for independent, parallel agent execution.
//!
//! Provides built-in subagent types (`explore`, `worker`, `plan`) and
//! custom TOML-defined subagents that run with independent conversation
//! contexts, restricted tool sets, and structured lifecycle management
//! via [`tokio::task::JoinSet`].

pub mod builtins;
pub mod config;
pub mod custom;
pub mod discovery;
pub mod error;
pub mod manager;
pub mod runner;
pub mod status;
pub mod tools;

pub use config::{AgentKind, AgentType, SubagentConfig};
pub use custom::{AgentSource, CustomAgentDef, CustomAgentMeta};
pub use discovery::discover_custom_agents;
pub use error::AgentError;
pub use manager::{SubagentId, SubagentManager, SubagentSummary};
pub use runner::SubagentRunner;
pub use status::{SubagentStatus, truncate_str};
pub use tools::SpawnAgentTool;
