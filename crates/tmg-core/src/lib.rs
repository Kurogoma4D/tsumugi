//! tmg-core: Core domain types, traits, and orchestration logic for tsumugi.

pub mod agent_loop;
pub mod error;
pub mod message;
pub mod prompt;

pub use agent_loop::{AgentLoop, StreamSink};
pub use error::CoreError;
pub use message::Message;
