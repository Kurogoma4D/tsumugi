//! tmg-agents: Subagent system for independent, parallel agent execution.
//!
//! Provides built-in subagent types (`explore`, `worker`, `plan`) that run
//! with independent conversation contexts, restricted tool sets, and
//! structured lifecycle management via [`tokio::task::JoinSet`].

pub mod builtins;
pub mod config;
pub mod error;
pub mod manager;
pub mod runner;
pub mod status;
pub mod tools;

pub use config::{AgentType, SubagentConfig};
pub use error::AgentError;
pub use manager::{SubagentId, SubagentManager, SubagentSummary};
pub use runner::SubagentRunner;
pub use status::{SubagentStatus, truncate_str};
pub use tools::SpawnAgentTool;
