//! tmg-agents: Subagent system for independent, parallel agent execution.
//!
//! Provides built-in subagent types (`explore`, `worker`, `plan`,
//! `initializer`, `tester`, `qa`, `escalator`) and custom TOML-defined
//! subagents that run with independent conversation contexts,
//! restricted tool sets, and structured lifecycle management via
//! [`tokio::task::JoinSet`].

pub mod builtins;
pub mod config;
pub mod custom;
pub mod discovery;
pub mod error;
pub mod escalator;
pub mod manager;
pub mod runner;
pub mod status;
pub mod tools;

pub use builtins::{
    RunToolProvider, registry_for_agent_kind, registry_for_agent_kind_with_run_provider,
    registry_for_agent_type,
};
pub use config::{AgentKind, AgentType, SubagentConfig};
pub use custom::{AgentSource, CustomAgentDef, CustomAgentMeta};
pub use discovery::discover_custom_agents;
pub use error::AgentError;
pub use escalator::{EscalatorVerdict, ParseError as EscalatorParseError, parse_verdict};
pub use manager::{
    EscalatorOverrides, SubagentId, SubagentManager, SubagentSummary, derive_sandbox,
};
pub use runner::SubagentRunner;
pub use status::{SubagentStatus, truncate_str};
pub use tools::SpawnAgentTool;
