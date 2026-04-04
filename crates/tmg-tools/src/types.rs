//! Core types: `Tool` trait, `ToolResult`, and `ToolRegistry`.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use crate::error::{MAX_OUTPUT_LENGTH, ToolError};

/// The result of a tool execution.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolResult {
    /// The textual output of the tool.
    pub output: String,
    /// Whether this result represents an error condition reported by the tool
    /// itself (as opposed to a `ToolError` which prevents execution entirely).
    pub is_error: bool,
}

impl ToolResult {
    /// Create a successful result.
    pub fn success(output: impl Into<String>) -> Self {
        let output = Self::maybe_truncate(output.into());
        Self {
            output,
            is_error: false,
        }
    }

    /// Create an error result (tool executed but produced an error output).
    pub fn error(output: impl Into<String>) -> Self {
        let output = Self::maybe_truncate(output.into());
        Self {
            output,
            is_error: true,
        }
    }

    /// Truncate the output if it exceeds the maximum length.
    fn maybe_truncate(output: String) -> String {
        if output.len() <= MAX_OUTPUT_LENGTH {
            return output;
        }
        let half = MAX_OUTPUT_LENGTH / 2;
        let truncated_bytes = output.len() - MAX_OUTPUT_LENGTH;
        // Find char boundaries near the cut points.
        let start_end = floor_char_boundary(&output, half);
        let tail_start = ceil_char_boundary(&output, output.len() - half);
        format!(
            "{}\n\n... ({truncated_bytes} bytes truncated) ...\n\n{}",
            &output[..start_end],
            &output[tail_start..],
        )
    }
}

/// Find the largest byte index <= `index` that is a char boundary.
///
/// This is a polyfill for [`str::floor_char_boundary`] which is stable since
/// Rust 1.91.0. Replace with the std method once MSRV is raised.
fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Find the smallest byte index >= `index` that is a char boundary.
///
/// This is a polyfill for [`str::ceil_char_boundary`] which is stable since
/// Rust 1.91.0. Replace with the std method once MSRV is raised.
fn ceil_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// A tool that the coding agent can invoke.
///
/// Implementations must provide a name, description, JSON Schema for
/// parameters, and an async `execute` method.
pub trait Tool: Send + Sync {
    /// The unique name of this tool (e.g. `"file_read"`).
    fn name(&self) -> &'static str;

    /// A human-readable description of what this tool does.
    fn description(&self) -> &'static str;

    /// JSON Schema describing the parameters this tool accepts.
    fn parameters_schema(&self) -> serde_json::Value;

    /// Execute the tool with the given JSON parameters.
    ///
    /// Returns a boxed future to allow dyn-compatible trait objects in the
    /// registry. This is _not_ using the `#[async_trait]` macro; implementors
    /// can write `async fn` bodies and wrap them with [`Box::pin`].
    fn execute(
        &self,
        params: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult, ToolError>> + Send + '_>>;
}

/// A registry of available tools, supporting lookup by name and batch schema
/// retrieval.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool. If a tool with the same name already exists, it is
    /// replaced.
    pub fn register(&mut self, tool: impl Tool + 'static) {
        self.tools.insert(tool.name().to_owned(), Box::new(tool));
    }

    /// Look up a tool by name.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(AsRef::as_ref)
    }

    /// Execute a tool by name with the given parameters.
    pub async fn execute(
        &self,
        name: &str,
        params: serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        let Some(tool) = self.tools.get(name) else {
            return Err(ToolError::NotFound {
                name: name.to_owned(),
            });
        };
        tool.execute(params).await
    }

    /// Return the JSON Schema definitions for all registered tools.
    ///
    /// The returned value is a JSON array of objects, each containing
    /// `name`, `description`, and `parameters`.
    pub fn all_schemas(&self) -> serde_json::Value {
        let mut schemas: Vec<serde_json::Value> = self
            .tools
            .values()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": tool.parameters_schema(),
                })
            })
            .collect();
        // Sort by name for deterministic output.
        schemas.sort_by(|a, b| {
            let a_name = a.get("name").and_then(serde_json::Value::as_str);
            let b_name = b.get("name").and_then(serde_json::Value::as_str);
            a_name.cmp(&b_name)
        });
        serde_json::Value::Array(schemas)
    }

    /// Return an iterator over all tool names.
    pub fn tool_names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(String::as_str)
    }

    /// Return tool definitions in the format expected by the LLM chat
    /// completions API `tools` field.
    ///
    /// Each entry wraps a tool's name, description, and parameter schema
    /// in a [`tmg_llm::ToolDefinition`].
    pub fn tool_definitions(&self) -> Vec<tmg_llm::ToolDefinition> {
        let mut defs: Vec<tmg_llm::ToolDefinition> = self
            .tools
            .values()
            .map(|tool| tmg_llm::ToolDefinition {
                kind: tmg_llm::ToolKind::Function,
                function: tmg_llm::FunctionDefinition {
                    name: tool.name().to_owned(),
                    description: Some(tool.description().to_owned()),
                    parameters: Some(tool.parameters_schema()),
                },
            })
            .collect();
        // Sort by name for deterministic ordering.
        defs.sort_by(|a, b| a.function.name.cmp(&b.function.name));
        defs
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_result_truncation() {
        // Small output should not be truncated.
        let result = ToolResult::success("hello");
        assert_eq!(result.output, "hello");
        assert!(!result.is_error);

        // Large output should be truncated.
        let large = "a".repeat(MAX_OUTPUT_LENGTH + 1000);
        let result = ToolResult::success(large);
        assert!(result.output.len() < MAX_OUTPUT_LENGTH + 100);
        assert!(result.output.contains("bytes truncated"));
    }

    #[test]
    fn tool_result_error() {
        let result = ToolResult::error("something went wrong");
        assert!(result.is_error);
        assert_eq!(result.output, "something went wrong");
    }
}
