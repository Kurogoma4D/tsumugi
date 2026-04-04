//! Prompt-based tool calling: parse `<tool_call>` tags from LLM output
//! and inject calling conventions into system prompts.
//!
//! This module provides a fallback for models that do not support native
//! OpenAI-compatible function calling. Instead, tools are described in the
//! system prompt and the model is expected to emit `<tool_call>` XML tags
//! which are then parsed into the same [`ToolCall`] type used by native mode.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::types::{FunctionCall, ToolCall, ToolDefinition, ToolKind};

// ---------------------------------------------------------------------------
// ToolCallingMode
// ---------------------------------------------------------------------------

/// The tool calling strategy to use when communicating with the LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallingMode {
    /// Use the OpenAI-compatible native function calling API.
    Native,

    /// Inject tool descriptions into the system prompt and parse
    /// `<tool_call>` tags from the model's text output.
    PromptBased,

    /// Try native first; if no `tool_calls` are returned, re-check
    /// the text content for `<tool_call>` tags (prompt-based fallback).
    #[default]
    Auto,
}

impl std::fmt::Display for ToolCallingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Native => f.write_str("native"),
            Self::PromptBased => f.write_str("prompt_based"),
            Self::Auto => f.write_str("auto"),
        }
    }
}

impl std::str::FromStr for ToolCallingMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "native" => Ok(Self::Native),
            "prompt_based" => Ok(Self::PromptBased),
            "auto" => Ok(Self::Auto),
            other => Err(format!(
                "invalid tool calling mode: {other:?} (expected \"native\", \"prompt_based\", or \"auto\")"
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// ParseError
// ---------------------------------------------------------------------------

/// Errors encountered while parsing `<tool_call>` tags from model output.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    /// A `<tool_call>` tag was found but the JSON content is invalid.
    #[error("invalid JSON in <tool_call> tag: {reason}")]
    InvalidJson {
        /// Description of the JSON parse failure.
        reason: String,
    },

    /// The parsed JSON is missing the required `name` field.
    #[error("<tool_call> JSON is missing the \"name\" field")]
    MissingName,

    /// The parsed JSON `name` field is not a string.
    #[error("<tool_call> JSON \"name\" field is not a string")]
    InvalidName,

    /// The parsed JSON `arguments` field is not an object.
    #[error("<tool_call> JSON \"arguments\" field is not an object")]
    InvalidArguments,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parse `<tool_call>` tags from raw LLM text output.
///
/// Extracts all `<tool_call>...</tool_call>` blocks, parses each as JSON,
/// and converts them into [`ToolCall`] values compatible with native mode.
///
/// Each `<tool_call>` block is expected to contain JSON with the structure:
///
/// ```json
/// {"name": "tool_name", "arguments": {"param": "value"}}
/// ```
///
/// The `arguments` field is optional and defaults to `{}`.
///
/// Returns a tuple of `(cleaned_text, tool_calls, errors)` where:
/// - `cleaned_text` has all `<tool_call>...</tool_call>` blocks removed
/// - `tool_calls` contains successfully parsed calls
/// - `errors` contains parse errors for malformed blocks
pub fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>, Vec<ParseError>) {
    let mut cleaned = String::with_capacity(text.len());
    let mut tool_calls = Vec::new();
    let mut errors = Vec::new();
    let mut call_counter: u32 = 0;

    let mut remaining = text;

    loop {
        let Some(start_pos) = remaining.find("<tool_call>") else {
            cleaned.push_str(remaining);
            break;
        };

        // Add text before the tag to cleaned output.
        cleaned.push_str(&remaining[..start_pos]);

        let after_open = &remaining[start_pos + "<tool_call>".len()..];

        let Some(end_pos) = after_open.find("</tool_call>") else {
            // No closing tag: treat as plain text from this point on.
            cleaned.push_str(&remaining[start_pos..]);
            break;
        };

        let json_content = after_open[..end_pos].trim();

        match parse_single_tool_call(json_content, call_counter) {
            Ok(tc) => {
                tool_calls.push(tc);
                call_counter += 1;
            }
            Err(e) => {
                errors.push(e);
            }
        }

        remaining = &after_open[end_pos + "</tool_call>".len()..];
    }

    // Trim trailing whitespace that may result from tag removal.
    let cleaned = cleaned.trim().to_owned();

    (cleaned, tool_calls, errors)
}

/// Parse a single `<tool_call>` JSON body into a [`ToolCall`].
fn parse_single_tool_call(json_str: &str, index: u32) -> Result<ToolCall, ParseError> {
    let value: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| ParseError::InvalidJson {
            reason: e.to_string(),
        })?;

    let serde_json::Value::Object(ref obj) = value else {
        return Err(ParseError::InvalidJson {
            reason: "expected a JSON object".to_owned(),
        });
    };

    let Some(name_val) = obj.get("name") else {
        return Err(ParseError::MissingName);
    };

    let serde_json::Value::String(ref name) = *name_val else {
        return Err(ParseError::InvalidName);
    };

    let arguments = match obj.get("arguments") {
        Some(serde_json::Value::Object(args)) => {
            // Safety: `serde_json::to_string` on a `Map<String, Value>` cannot
            // fail because the value was already successfully deserialized from
            // valid JSON — it contains no non-string map keys or other
            // constructs that would cause serialization to error.
            serde_json::to_string(args).unwrap_or_else(|_| "{}".to_owned())
        }
        Some(serde_json::Value::Null) | None => "{}".to_owned(),
        Some(_) => return Err(ParseError::InvalidArguments),
    };

    Ok(ToolCall {
        id: format!("prompt_call_{index}"),
        kind: ToolKind::Function,
        function: FunctionCall {
            name: name.clone(),
            arguments,
        },
    })
}

// ---------------------------------------------------------------------------
// Compressed tool definitions for prompt injection
// ---------------------------------------------------------------------------

/// Format tool definitions in a compact form suitable for prompt injection.
///
/// Produces a lightweight representation with just the tool name, a one-line
/// description, and parameter names (no full JSON Schema). This is
/// automatically selected for `prompt_based` mode to minimize token usage.
///
/// # Example output
///
/// ```text
/// - file_read(path): Read the contents of a file at the given path.
/// - grep_search(pattern, path, include): Search for a regex pattern in files.
/// ```
#[must_use]
pub fn format_compressed_tool_defs(tools: &[ToolDefinition]) -> String {
    let mut buf = String::new();
    for tool in tools {
        let name = &tool.function.name;
        let desc = tool
            .function
            .description
            .as_deref()
            .unwrap_or("(no description)");

        let param_names = extract_parameter_names(tool.function.parameters.as_ref());
        let params_str = param_names.join(", ");

        let _ = writeln!(buf, "- {name}({params_str}): {desc}");
    }
    buf
}

/// Extract parameter names from a JSON Schema `parameters` object.
fn extract_parameter_names(schema: Option<&serde_json::Value>) -> Vec<String> {
    let Some(schema) = schema else {
        return Vec::new();
    };

    let Some(properties) = schema
        .get("properties")
        .and_then(serde_json::Value::as_object)
    else {
        return Vec::new();
    };

    let mut names: Vec<String> = properties.keys().cloned().collect();
    names.sort();
    names
}

/// Build the system prompt suffix that teaches the model the `<tool_call>`
/// format.
///
/// This is appended to the existing system prompt when `prompt_based` or
/// `auto` mode is active.
#[must_use]
pub fn build_tool_calling_prompt(tools: &[ToolDefinition]) -> String {
    let compressed = format_compressed_tool_defs(tools);

    format!(
        "\n\n## Available Tools\n\n\
         {compressed}\n\
         ## How to Call Tools\n\n\
         When you need to use a tool, output a `<tool_call>` XML tag containing \
         a JSON object with `\"name\"` and `\"arguments\"` fields. For example:\n\n\
         <tool_call>\n\
         {{\"name\": \"file_read\", \"arguments\": {{\"path\": \"/src/main.rs\"}}}}\n\
         </tool_call>\n\n\
         You may output multiple `<tool_call>` tags in a single response. \
         After each tool call is executed, you will receive the result and can \
         continue your response. Only use tools when necessary.\n"
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FunctionDefinition;

    #[test]
    fn parse_single_valid_tool_call() {
        let text = r#"Let me read that file.
<tool_call>
{"name": "file_read", "arguments": {"path": "/tmp/test.rs"}}
</tool_call>
Done."#;

        let (cleaned, calls, errors) = parse_tool_calls(text);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "file_read");
        assert_eq!(calls[0].function.arguments, r#"{"path":"/tmp/test.rs"}"#);
        assert_eq!(calls[0].id, "prompt_call_0");
        assert!(cleaned.contains("Let me read that file."));
        assert!(cleaned.contains("Done."));
        assert!(!cleaned.contains("<tool_call>"));
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let text = r#"<tool_call>
{"name": "file_read", "arguments": {"path": "/a"}}
</tool_call>
<tool_call>
{"name": "file_write", "arguments": {"path": "/b", "content": "hello"}}
</tool_call>"#;

        let (_, calls, errors) = parse_tool_calls(text);
        assert!(errors.is_empty());
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "file_read");
        assert_eq!(calls[0].id, "prompt_call_0");
        assert_eq!(calls[1].function.name, "file_write");
        assert_eq!(calls[1].id, "prompt_call_1");
    }

    #[test]
    fn parse_tool_call_no_arguments() {
        let text = r#"<tool_call>
{"name": "list_dir"}
</tool_call>"#;

        let (_, calls, errors) = parse_tool_calls(text);
        assert!(errors.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "list_dir");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn parse_tool_call_null_arguments() {
        let text = r#"<tool_call>
{"name": "list_dir", "arguments": null}
</tool_call>"#;

        let (_, calls, errors) = parse_tool_calls(text);
        assert!(errors.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn parse_invalid_json_produces_error() {
        let text = r"<tool_call>
{not valid json}
</tool_call>";

        let (_, calls, errors) = parse_tool_calls(text);
        assert!(calls.is_empty());
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0], ParseError::InvalidJson { .. }));
    }

    #[test]
    fn parse_missing_name_produces_error() {
        let text = r#"<tool_call>
{"arguments": {"path": "/tmp"}}
</tool_call>"#;

        let (_, calls, errors) = parse_tool_calls(text);
        assert!(calls.is_empty());
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0], ParseError::MissingName);
    }

    #[test]
    fn parse_invalid_name_type_produces_error() {
        let text = r#"<tool_call>
{"name": 42, "arguments": {}}
</tool_call>"#;

        let (_, calls, errors) = parse_tool_calls(text);
        assert!(calls.is_empty());
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0], ParseError::InvalidName);
    }

    #[test]
    fn parse_invalid_arguments_type_produces_error() {
        let text = r#"<tool_call>
{"name": "test", "arguments": "not an object"}
</tool_call>"#;

        let (_, calls, errors) = parse_tool_calls(text);
        assert!(calls.is_empty());
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0], ParseError::InvalidArguments);
    }

    #[test]
    fn parse_unclosed_tag_treated_as_text() {
        let text = "Hello <tool_call> some stuff without closing";
        let (cleaned, calls, errors) = parse_tool_calls(text);
        assert!(calls.is_empty());
        assert!(errors.is_empty());
        assert!(cleaned.contains("<tool_call>"));
    }

    #[test]
    fn parse_no_tool_calls_returns_original() {
        let text = "Just a regular message with no tools.";
        let (cleaned, calls, errors) = parse_tool_calls(text);
        assert!(calls.is_empty());
        assert!(errors.is_empty());
        assert_eq!(cleaned, text);
    }

    #[test]
    fn parse_mixed_valid_and_invalid() {
        let text = r#"<tool_call>
{"name": "good_tool", "arguments": {}}
</tool_call>
<tool_call>
{bad json}
</tool_call>
<tool_call>
{"name": "another_good", "arguments": {"x": 1}}
</tool_call>"#;

        let (_, calls, errors) = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(errors.len(), 1);
        assert_eq!(calls[0].function.name, "good_tool");
        assert_eq!(calls[1].function.name, "another_good");
    }

    #[test]
    fn compressed_tool_defs_format() {
        let tools = vec![
            ToolDefinition {
                kind: ToolKind::Function,
                function: FunctionDefinition {
                    name: "file_read".to_owned(),
                    description: Some("Read the contents of a file.".to_owned()),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        }
                    })),
                },
            },
            ToolDefinition {
                kind: ToolKind::Function,
                function: FunctionDefinition {
                    name: "grep_search".to_owned(),
                    description: Some("Search for a pattern in files.".to_owned()),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string"},
                            "include": {"type": "string"}
                        }
                    })),
                },
            },
        ];

        let result = format_compressed_tool_defs(&tools);
        assert!(result.contains("file_read(path)"));
        assert!(result.contains("grep_search(include, path, pattern)"));
        assert!(result.contains("Read the contents of a file."));
    }

    #[test]
    fn build_tool_calling_prompt_contains_example() {
        let tools = vec![ToolDefinition {
            kind: ToolKind::Function,
            function: FunctionDefinition {
                name: "test_tool".to_owned(),
                description: Some("A test tool.".to_owned()),
                parameters: None,
            },
        }];

        let prompt = build_tool_calling_prompt(&tools);
        assert!(prompt.contains("<tool_call>"));
        assert!(prompt.contains("</tool_call>"));
        assert!(prompt.contains("test_tool"));
        assert!(prompt.contains("Available Tools"));
    }

    #[test]
    fn tool_calling_mode_display() {
        assert_eq!(ToolCallingMode::Native.to_string(), "native");
        assert_eq!(ToolCallingMode::PromptBased.to_string(), "prompt_based");
        assert_eq!(ToolCallingMode::Auto.to_string(), "auto");
    }

    #[test]
    fn tool_calling_mode_serde_roundtrip() {
        let modes = [
            ToolCallingMode::Native,
            ToolCallingMode::PromptBased,
            ToolCallingMode::Auto,
        ];
        for mode in modes {
            let json = serde_json::to_string(&mode).unwrap_or_default();
            let parsed: ToolCallingMode =
                serde_json::from_str(&json).unwrap_or(ToolCallingMode::Auto);
            assert_eq!(parsed, mode);
        }
    }

    #[test]
    fn tool_calling_mode_toml_roundtrip() {
        // Simulate TOML deserialization like tsumugi.toml would use.
        #[derive(Deserialize)]
        struct Config {
            tool_calling: ToolCallingMode,
        }

        let toml_str = r#"tool_calling = "prompt_based""#;
        let config: Config = toml::from_str(toml_str).unwrap_or(Config {
            tool_calling: ToolCallingMode::Auto,
        });
        assert_eq!(config.tool_calling, ToolCallingMode::PromptBased);

        let toml_str = r#"tool_calling = "native""#;
        let config: Config = toml::from_str(toml_str).unwrap_or(Config {
            tool_calling: ToolCallingMode::Auto,
        });
        assert_eq!(config.tool_calling, ToolCallingMode::Native);

        let toml_str = r#"tool_calling = "auto""#;
        let config: Config = toml::from_str(toml_str).unwrap_or(Config {
            tool_calling: ToolCallingMode::Auto,
        });
        assert_eq!(config.tool_calling, ToolCallingMode::Auto);
    }

    #[test]
    fn parse_whitespace_around_json() {
        let text = "<tool_call>\n  \n  {\"name\": \"test\", \"arguments\": {}}  \n  \n</tool_call>";
        let (_, calls, errors) = parse_tool_calls(text);
        assert!(errors.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "test");
    }

    #[test]
    fn parse_inline_tool_call() {
        // Some models may emit tags inline without newlines.
        let text =
            "Let me check: <tool_call>{\"name\": \"list_dir\", \"arguments\": {}}</tool_call> ok.";
        let (cleaned, calls, errors) = parse_tool_calls(text);
        assert!(errors.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "list_dir");
        assert_eq!(cleaned, "Let me check:  ok.");
    }

    #[test]
    fn extract_param_names_empty_schema() {
        assert!(extract_parameter_names(None).is_empty());
    }

    #[test]
    fn extract_param_names_no_properties() {
        let schema = serde_json::json!({"type": "object"});
        assert!(extract_parameter_names(Some(&schema)).is_empty());
    }
}
