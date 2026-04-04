//! Conversation message types for the agent loop.

use tmg_llm::{ChatMessage, Role};

/// A conversation message with a role and text content.
///
/// This is a simplified view used by the agent loop. It converts to/from
/// the LLM layer's [`ChatMessage`] for wire transmission.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    /// The author role of this message.
    pub role: Role,

    /// The text content of this message.
    pub content: String,
}

impl Message {
    /// Create a new system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    /// Create a new user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    /// Create a new assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }

    /// Convert this message into a [`ChatMessage`] for the LLM API.
    pub fn into_chat_message(self) -> ChatMessage {
        ChatMessage {
            role: self.role,
            content: Some(self.content),
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

impl From<Message> for ChatMessage {
    fn from(msg: Message) -> Self {
        msg.into_chat_message()
    }
}
