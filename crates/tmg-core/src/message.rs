//! Conversation message types for the agent loop.

use tmg_llm::{ChatMessage, Role, ToolCall};

/// A conversation message with a role and content.
///
/// This is a simplified view used by the agent loop. It converts to/from
/// the LLM layer's [`ChatMessage`] for wire transmission.
///
/// Supports text-only messages (system, user, plain assistant), assistant
/// messages with tool calls, and tool-result messages.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    /// The author role of this message.
    pub(crate) role: Role,

    /// The text content of this message (may be empty for tool-call-only
    /// assistant messages).
    pub(crate) content: String,

    /// Tool calls requested by the assistant (non-empty only for
    /// `Role::Assistant` messages that triggered tool use).
    pub(crate) tool_calls: Option<Vec<ToolCall>>,

    /// For `Role::Tool` messages: the id of the tool call this message
    /// responds to.
    pub(crate) tool_call_id: Option<String>,
}

impl Message {
    /// Create a new system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a new user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a new assistant message (text only).
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create an assistant message that contains tool calls (and
    /// optionally some text content).
    pub fn assistant_with_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    /// Create a tool-result message responding to a specific tool call.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    /// Return the author role of this message.
    pub fn role(&self) -> Role {
        self.role
    }

    /// Return the text content of this message.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Return the tool calls, if any.
    pub fn tool_calls(&self) -> Option<&[ToolCall]> {
        self.tool_calls.as_deref()
    }

    /// Create a [`ChatMessage`] by borrowing self, cloning strings.
    pub fn to_chat_message(&self) -> ChatMessage {
        let content = if self.content.is_empty() {
            None
        } else {
            Some(self.content.clone())
        };
        ChatMessage {
            role: self.role,
            content,
            tool_calls: self.tool_calls.clone(),
            tool_call_id: self.tool_call_id.clone(),
        }
    }

    /// Convert this message into a [`ChatMessage`] for the LLM API.
    pub fn into_chat_message(self) -> ChatMessage {
        let content = if self.content.is_empty() {
            None
        } else {
            Some(self.content)
        };
        ChatMessage {
            role: self.role,
            content,
            tool_calls: self.tool_calls,
            tool_call_id: self.tool_call_id,
        }
    }
}

impl From<Message> for ChatMessage {
    fn from(msg: Message) -> Self {
        msg.into_chat_message()
    }
}
