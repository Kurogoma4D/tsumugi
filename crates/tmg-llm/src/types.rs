//! Request and response types for the OpenAI-compatible chat completions API.

use serde::{Deserialize, Serialize};

/// Maximum number of concurrent tool calls allowed in a single response.
///
/// This prevents unbounded memory allocation from malformed server responses
/// with absurdly large tool call indices.
const MAX_TOOL_CALL_INDEX: usize = 128;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// A chat completion request sent to the LLM server.
#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    /// Model identifier (e.g. `"llama3"`).
    pub model: String,

    /// Conversation messages.
    pub messages: Vec<ChatMessage>,

    /// Whether to enable SSE streaming.
    pub stream: bool,

    /// Tool definitions available to the model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,

    /// Optional temperature override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

/// A single message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    /// The role of the message author.
    pub role: Role,

    /// Text content (may be `None` for assistant messages with only tool calls).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Tool calls requested by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// For role `tool`: the id of the tool call this message responds to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Message author role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System prompt.
    System,
    /// User message.
    User,
    /// Assistant (model) response.
    Assistant,
    /// Tool result.
    Tool,
}

/// The kind of a tool or tool call. Currently only `function` is supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolKind {
    /// A callable function.
    #[default]
    Function,
}

/// A tool definition exposed to the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    /// The tool kind (always [`ToolKind::Function`]).
    #[serde(rename = "type", default)]
    pub kind: ToolKind,

    /// Function metadata.
    pub function: FunctionDefinition,
}

/// Metadata for a callable function.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionDefinition {
    /// Function name.
    pub name: String,

    /// Human-readable description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema describing the parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Response types (non-streaming)
// ---------------------------------------------------------------------------

/// A complete (non-streaming) chat completion response.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
    /// Unique response id.
    pub id: String,

    /// List of completion choices.
    pub choices: Vec<Choice>,
}

/// A single completion choice.
#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    /// Choice index.
    pub index: u32,

    /// The generated message.
    pub message: ChatMessage,

    /// Reason the generation stopped.
    pub finish_reason: Option<String>,
}

// ---------------------------------------------------------------------------
// Streaming (SSE) response types
// ---------------------------------------------------------------------------

/// A single SSE chunk from a streaming chat completion.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionChunk {
    /// Unique response id.
    pub id: String,

    /// Chunk choices.
    pub choices: Vec<ChunkChoice>,
}

/// A single choice within a streaming chunk.
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkChoice {
    /// Choice index.
    pub index: u32,

    /// Incremental delta for this choice.
    pub delta: Delta,

    /// Reason the generation stopped (present in the final chunk).
    pub finish_reason: Option<String>,
}

/// Incremental content delivered in a streaming chunk.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct Delta {
    /// The role, if present in this delta.
    pub role: Option<Role>,

    /// Incremental text content.
    pub content: Option<String>,

    /// Incremental reasoning/thinking content (e.g. from DeepSeek-style models).
    pub reasoning_content: Option<String>,

    /// Incremental tool call data.
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// Incremental tool call data within a streaming delta.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ToolCallDelta {
    /// Index of the tool call being built.
    pub index: u32,

    /// Tool call id (present in the first delta for this call).
    pub id: Option<String>,

    /// Always [`ToolKind::Function`] when present.
    #[serde(rename = "type")]
    pub kind: Option<ToolKind>,

    /// Incremental function data.
    pub function: Option<FunctionCallDelta>,
}

/// Incremental function call data within a tool call delta.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct FunctionCallDelta {
    /// Function name (present in the first delta).
    pub name: Option<String>,

    /// Incremental arguments JSON string.
    pub arguments: Option<String>,
}

// ---------------------------------------------------------------------------
// Assembled tool call
// ---------------------------------------------------------------------------

/// A fully assembled tool call reconstructed from streaming deltas.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,

    /// The tool kind (always [`ToolKind::Function`]).
    #[serde(rename = "type", default)]
    pub kind: ToolKind,

    /// The function to call.
    pub function: FunctionCall,
}

/// A fully assembled function call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    /// Function name.
    pub name: String,

    /// Arguments as a JSON string.
    pub arguments: String,
}

// ---------------------------------------------------------------------------
// Stream event
// ---------------------------------------------------------------------------

/// High-level events emitted by the streaming client.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// An incremental reasoning/thinking token.
    ThinkingDelta(String),

    /// An incremental text token.
    ContentDelta(String),

    /// A complete tool call has been assembled from deltas.
    ToolCallComplete(ToolCall),

    /// The stream has finished. Contains the finish reason.
    Done(Option<String>),
}

// ---------------------------------------------------------------------------
// Tokenize response
// ---------------------------------------------------------------------------

/// Response from the llama-server `POST /tokenize` endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenizeResponse {
    /// The token ids produced by the tokenizer.
    pub tokens: Vec<i64>,
}

// ---------------------------------------------------------------------------
// Tool call accumulator
// ---------------------------------------------------------------------------

/// Accumulates incremental tool call deltas into complete [`ToolCall`]s.
#[derive(Debug, Default)]
pub struct ToolCallAccumulator {
    /// In-progress tool calls, keyed by index.
    calls: Vec<ToolCallBuilder>,
}

/// Error returned when a tool call delta has an index exceeding the allowed maximum.
#[derive(Debug, thiserror::Error)]
#[error("tool call index {index} exceeds maximum allowed ({MAX_TOOL_CALL_INDEX})")]
pub struct ToolCallIndexError {
    /// The offending index value.
    pub index: u32,
}

#[derive(Debug, Default)]
struct ToolCallBuilder {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallAccumulator {
    /// Create a new empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a batch of tool call deltas. Returns any newly completed tool calls.
    ///
    /// A tool call is considered complete when a subsequent delta starts a new
    /// call at a higher index, or when the stream signals completion via
    /// [`Self::finish`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if any delta has an index exceeding [`MAX_TOOL_CALL_INDEX`],
    /// which guards against unbounded memory allocation from malformed responses.
    pub fn feed(&mut self, deltas: &[ToolCallDelta]) -> Result<Vec<ToolCall>, ToolCallIndexError> {
        let mut completed = Vec::new();

        for delta in deltas {
            let idx = delta.index as usize;

            // Reject absurdly large indices to prevent OOM.
            if idx > MAX_TOOL_CALL_INDEX {
                return Err(ToolCallIndexError { index: delta.index });
            }

            // Ensure we have enough builder slots.
            while self.calls.len() <= idx {
                self.calls.push(ToolCallBuilder::default());
            }

            let builder = &mut self.calls[idx];

            if let Some(id) = &delta.id {
                builder.id.clone_from(id);
            }
            if let Some(func) = &delta.function {
                if let Some(name) = &func.name {
                    builder.name.clone_from(name);
                }
                if let Some(args) = &func.arguments {
                    builder.arguments.push_str(args);
                }
            }
        }

        // Check if any lower-indexed calls have been superseded.
        // This heuristic: if we received a delta for index N, all calls < N
        // with non-empty ids are considered complete.
        if let Some(max_idx) = deltas.iter().map(|d| d.index as usize).max() {
            for idx in 0..max_idx {
                if !self.calls[idx].id.is_empty() {
                    let builder = std::mem::take(&mut self.calls[idx]);
                    completed.push(ToolCall {
                        id: builder.id,
                        kind: ToolKind::Function,
                        function: FunctionCall {
                            name: builder.name,
                            arguments: builder.arguments,
                        },
                    });
                }
            }
        }

        Ok(completed)
    }

    /// Drain all remaining in-progress tool calls as complete.
    pub fn finish(&mut self) -> Vec<ToolCall> {
        self.calls
            .drain(..)
            .filter(|b| !b.id.is_empty())
            .map(|b| ToolCall {
                id: b.id,
                kind: ToolKind::Function,
                function: FunctionCall {
                    name: b.name,
                    arguments: b.arguments,
                },
            })
            .collect()
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "expect is appropriate in test assertions"
)]
mod tests {
    use super::*;

    #[test]
    fn parse_chunk_with_content_delta() {
        let json = r#"{
            "id": "chatcmpl-1",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": null
            }]
        }"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse chunk");
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hello"));
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn parse_chunk_with_tool_call_delta() {
        let json = r#"{
            "id": "chatcmpl-2",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\":"
                        }
                    }]
                },
                "finish_reason": null
            }]
        }"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse chunk");
        let tc = &chunk.choices[0]
            .delta
            .tool_calls
            .as_ref()
            .expect("tool_calls")[0];
        assert_eq!(tc.id.as_deref(), Some("call_abc"));
        assert_eq!(
            tc.function.as_ref().expect("function").name.as_deref(),
            Some("read_file")
        );
    }

    #[test]
    fn parse_non_streaming_response_with_tool_calls() {
        let json = r#"{
            "id": "chatcmpl-3",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_xyz",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": "{\"path\": \"/tmp/test.txt\", \"content\": \"hello\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }"#;

        let resp: ChatResponse = serde_json::from_str(json).expect("parse response");
        let tool_calls = resp.choices[0]
            .message
            .tool_calls
            .as_ref()
            .expect("tool_calls");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "write_file");
    }

    #[test]
    fn accumulator_assembles_single_tool_call() {
        let mut acc = ToolCallAccumulator::new();

        // First delta: id + name + partial args
        let d1 = vec![ToolCallDelta {
            index: 0,
            id: Some("call_1".into()),
            kind: Some(ToolKind::Function),
            function: Some(FunctionCallDelta {
                name: Some("read_file".into()),
                arguments: Some("{\"path\":".into()),
            }),
        }];
        let completed = acc.feed(&d1).expect("feed d1");
        assert!(completed.is_empty());

        // Second delta: more args
        let d2 = vec![ToolCallDelta {
            index: 0,
            id: None,
            kind: None,
            function: Some(FunctionCallDelta {
                name: None,
                arguments: Some("\"/tmp/f\"}".into()),
            }),
        }];
        let completed = acc.feed(&d2).expect("feed d2");
        assert!(completed.is_empty());

        // Finish
        let completed = acc.finish();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].id, "call_1");
        assert_eq!(completed[0].kind, ToolKind::Function);
        assert_eq!(completed[0].function.name, "read_file");
        assert_eq!(completed[0].function.arguments, "{\"path\":\"/tmp/f\"}");
    }

    #[test]
    fn accumulator_assembles_multiple_tool_calls() {
        let mut acc = ToolCallAccumulator::new();

        // First tool call
        acc.feed(&[ToolCallDelta {
            index: 0,
            id: Some("call_a".into()),
            kind: Some(ToolKind::Function),
            function: Some(FunctionCallDelta {
                name: Some("foo".into()),
                arguments: Some("{}".into()),
            }),
        }])
        .expect("feed first");

        // Second tool call starts - first should be completed
        let completed = acc
            .feed(&[ToolCallDelta {
                index: 1,
                id: Some("call_b".into()),
                kind: Some(ToolKind::Function),
                function: Some(FunctionCallDelta {
                    name: Some("bar".into()),
                    arguments: Some("{\"x\":1}".into()),
                }),
            }])
            .expect("feed second");

        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].id, "call_a");
        assert_eq!(completed[0].function.name, "foo");

        // Finish to get the second
        let completed = acc.finish();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].id, "call_b");
    }

    #[test]
    fn accumulator_rejects_excessive_index() {
        let mut acc = ToolCallAccumulator::new();

        let result = acc.feed(&[ToolCallDelta {
            index: u32::MAX,
            id: Some("call_bad".into()),
            kind: Some(ToolKind::Function),
            function: Some(FunctionCallDelta {
                name: Some("exploit".into()),
                arguments: Some("{}".into()),
            }),
        }]);

        assert!(result.is_err());
    }

    #[test]
    fn serialize_chat_request() {
        let req = ChatRequest {
            model: "llama3".to_owned(),
            messages: vec![ChatMessage {
                role: Role::User,
                content: Some("Hello".to_owned()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: true,
            tools: vec![],
            temperature: None,
        };

        let json = serde_json::to_value(&req).expect("serialize");
        assert_eq!(json["model"], "llama3");
        assert_eq!(json["stream"], true);
        // tools should be omitted when empty
        assert!(json.get("tools").is_none());
        // temperature should be omitted when None
        assert!(json.get("temperature").is_none());
    }

    #[test]
    fn parse_finish_reason_stop() {
        let json = r#"{
            "id": "chatcmpl-done",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
    }
}
