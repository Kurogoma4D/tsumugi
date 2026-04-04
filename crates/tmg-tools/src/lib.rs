//! tmg-tools: Tool definitions and implementations for the coding agent.
//!
//! This crate provides the [`Tool`] trait, a [`ToolRegistry`] for looking up
//! tools by name, and built-in tool implementations for file operations,
//! directory listing, grep search, and shell command execution.

pub mod error;
mod path_util;
pub mod signatures;
pub mod tools;
pub mod types;

pub use error::ToolError;
pub use signatures::{Signature, extract_signatures, format_signatures};
pub use tools::default_registry;
pub use types::{Tool, ToolRegistry, ToolResult};
