//! Minimal agent loop: text-only turn-based conversation with an LLM.

use std::path::Path;

use tokio_stream::StreamExt as _;
use tokio_util::sync::CancellationToken;

use tmg_llm::{LlmClient, StreamEvent};

use crate::error::CoreError;
use crate::message::Message;
use crate::prompt;

/// The default system prompt injected at the start of every conversation.
const DEFAULT_SYSTEM_PROMPT: &str = "\
You are tsumugi, a helpful coding assistant running locally. \
Answer concisely and accurately.";

/// Callback invoked for each streamed content token.
///
/// Implementations should write the token to the appropriate output
/// (stdout, TUI widget, etc.). Returning an error aborts the current turn.
pub trait StreamSink {
    /// Called for each incremental text token from the LLM.
    fn on_token(&mut self, token: &str) -> Result<(), CoreError>;

    /// Called when the LLM response stream has completed for the current turn.
    fn on_done(&mut self) -> Result<(), CoreError> {
        Ok(())
    }
}

/// A minimal text-only agent loop.
///
/// Manages conversation history and sends it to the LLM on each turn.
/// Tool calls are not handled in this initial implementation.
pub struct AgentLoop {
    /// The LLM client used to send requests.
    client: LlmClient,

    /// Conversation history (system prompt + user/assistant messages).
    history: Vec<Message>,

    /// Cancellation token for graceful shutdown.
    cancel: CancellationToken,
}

impl AgentLoop {
    /// Create a new agent loop.
    ///
    /// Loads TSUMUGI.md / AGENTS.md prompt files from the global config
    /// directory and from `project_root` down to `cwd`, injecting them
    /// as initial user-role messages after the system prompt.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Io`] if a prompt file exists but cannot be read.
    pub fn new(
        client: LlmClient,
        cancel: CancellationToken,
        project_root: &Path,
        cwd: &Path,
    ) -> Result<Self, CoreError> {
        let mut history = vec![Message::system(DEFAULT_SYSTEM_PROMPT)];

        // Load prompt files and inject as initial messages.
        let prompt_messages = prompt::load_prompt_files(project_root, cwd)?;
        history.extend(prompt_messages);

        Ok(Self {
            client,
            history,
            cancel,
        })
    }

    /// Return a read-only view of the conversation history.
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Execute a single conversation turn.
    ///
    /// Adds the user message to history, streams the LLM response through
    /// the provided [`StreamSink`], assembles the full response, and appends
    /// it to history as an assistant message.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] on LLM communication failure, cancellation,
    /// or if the sink returns an error.
    ///
    /// # Cancel safety
    ///
    /// If the `CancellationToken` is triggered during streaming, the partial
    /// response is discarded and [`CoreError::Cancelled`] is returned.
    pub async fn turn(
        &mut self,
        user_input: &str,
        sink: &mut impl StreamSink,
    ) -> Result<(), CoreError> {
        // Add user message to history.
        self.history.push(Message::user(user_input));

        // Build the messages for the LLM request.
        let chat_messages: Vec<_> = self
            .history
            .iter()
            .cloned()
            .map(Message::into_chat_message)
            .collect();

        // Send streaming request.
        let mut stream = self
            .client
            .chat_streaming(chat_messages, vec![], self.cancel.clone())
            .await?;

        let mut response_text = String::new();

        while let Some(event) = stream.next().await {
            match event {
                Ok(StreamEvent::ContentDelta(token)) => {
                    response_text.push_str(&token);
                    sink.on_token(&token)?;
                }
                Ok(StreamEvent::Done(_)) => {
                    sink.on_done()?;
                    break;
                }
                Ok(StreamEvent::ToolCallComplete(_)) => {
                    // Tool calls are not handled in this minimal loop.
                }
                Err(tmg_llm::LlmError::Cancelled) => {
                    return Err(CoreError::Cancelled);
                }
                Err(e) => {
                    return Err(CoreError::Llm(e));
                }
            }
        }

        // Append assistant response to history.
        if !response_text.is_empty() {
            self.history.push(Message::assistant(response_text));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_constructors() {
        let sys = Message::system("hello");
        assert_eq!(sys.role, tmg_llm::Role::System);
        assert_eq!(sys.content, "hello");

        let usr = Message::user("world");
        assert_eq!(usr.role, tmg_llm::Role::User);

        let asst = Message::assistant("response");
        assert_eq!(asst.role, tmg_llm::Role::Assistant);
    }

    #[test]
    fn message_to_chat_message() {
        let msg = Message::user("test");
        let chat = msg.into_chat_message();
        assert_eq!(chat.role, tmg_llm::Role::User);
        assert_eq!(chat.content.as_deref(), Some("test"));
        assert!(chat.tool_calls.is_none());
        assert!(chat.tool_call_id.is_none());
    }
}
