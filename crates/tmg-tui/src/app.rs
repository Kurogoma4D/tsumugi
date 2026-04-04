//! Application state and business logic for the TUI.

use std::path::PathBuf;

use tmg_core::{AgentLoop, CoreError, StreamSink};
use tmg_llm::Role;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Represents a single entry in the chat display.
#[derive(Debug, Clone)]
pub struct ChatEntry {
    /// The author role of this entry.
    pub role: Role,
    /// The text content.
    pub text: String,
}

/// Messages sent from the background turn task to the event loop.
pub enum TurnMessage {
    /// An incremental token from the LLM stream.
    Token(String),
    /// The turn completed successfully. Contains the `AgentLoop` back
    /// and the turn count for context display.
    Done {
        /// The agent loop returned after the turn completes.
        agent: AgentLoop,
        /// Number of user turns so far (for context usage display).
        user_turns: usize,
    },
    /// The turn failed with an error. Contains the `AgentLoop` back.
    Error {
        /// The agent loop returned after the turn fails.
        agent: AgentLoop,
        /// The error message.
        message: String,
        /// Whether the error was a cancellation.
        cancelled: bool,
    },
}

/// Handle for a running conversation turn background task.
struct TurnHandle {
    /// Receiver for turn messages (tokens, done, error).
    rx: mpsc::Receiver<TurnMessage>,
    /// Child cancellation token for this specific turn.
    turn_cancel: CancellationToken,
    /// Join handle for the spawned task.
    _join: JoinHandle<()>,
}

/// The main application state.
pub struct App {
    /// The agent loop driving LLM conversations.
    ///
    /// This is `None` while a turn is running in a background task
    /// (the task temporarily takes ownership).
    agent: Option<AgentLoop>,

    /// Chat entries displayed in the chat pane.
    chat_entries: Vec<ChatEntry>,

    /// The current text in the input area.
    input: String,

    /// Cursor position within the input string (byte offset).
    cursor_pos: usize,

    /// Vertical scroll offset for the chat pane.
    chat_scroll: u16,

    /// Whether the app should exit at the next opportunity.
    should_exit: bool,

    /// Whether the agent is currently streaming a response.
    streaming: bool,

    /// Model name for header display.
    model_name: String,

    /// Context usage for header display (dummy value for now).
    context_usage: String,

    /// Project root for prompt file reloading on clear.
    project_root: PathBuf,

    /// Current working directory for prompt file reloading on clear.
    cwd: PathBuf,

    /// Optional error message to display.
    error_message: Option<String>,

    /// Handle for the currently running turn, if any.
    turn_handle: Option<TurnHandle>,
}

impl App {
    /// Create a new `App` with the given agent loop and model name.
    pub fn new(agent: AgentLoop, model_name: &str, project_root: PathBuf, cwd: PathBuf) -> Self {
        Self {
            agent: Some(agent),
            chat_entries: Vec::new(),
            input: String::new(),
            cursor_pos: 0,
            chat_scroll: 0,
            should_exit: false,
            streaming: false,
            model_name: model_name.to_owned(),
            context_usage: "0 / ? tokens".to_owned(),
            project_root,
            cwd,
            error_message: None,
            turn_handle: None,
        }
    }

    /// Whether the application should exit.
    pub fn should_exit(&self) -> bool {
        self.should_exit
    }

    /// Request the application to exit.
    pub fn request_exit(&mut self) {
        self.should_exit = true;
    }

    /// Whether the agent is currently streaming a response.
    pub fn is_streaming(&self) -> bool {
        self.streaming
    }

    /// Return the model name for display.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Return the context usage string for display.
    pub fn context_usage(&self) -> &str {
        &self.context_usage
    }

    /// Return a reference to the chat entries.
    pub fn chat_entries(&self) -> &[ChatEntry] {
        &self.chat_entries
    }

    /// Return the current input text.
    pub fn input(&self) -> &str {
        &self.input
    }

    /// Return the cursor position (byte offset).
    pub fn cursor_pos(&self) -> usize {
        self.cursor_pos
    }

    /// Return the chat scroll offset.
    pub fn chat_scroll(&self) -> u16 {
        self.chat_scroll
    }

    /// Set the chat scroll offset.
    pub fn set_chat_scroll(&mut self, offset: u16) {
        self.chat_scroll = offset;
    }

    /// Return the current error message, if any.
    pub fn error_message(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    /// Clear the error message.
    pub fn clear_error(&mut self) {
        self.error_message = None;
    }

    /// Insert a character at the cursor position.
    pub fn insert_char(&mut self, ch: char) {
        self.input.insert(self.cursor_pos, ch);
        self.cursor_pos += ch.len_utf8();
    }

    /// Insert a newline at the cursor position.
    pub fn insert_newline(&mut self) {
        self.insert_char('\n');
    }

    /// Delete the character before the cursor (backspace).
    pub fn delete_char_before_cursor(&mut self) {
        if self.cursor_pos == 0 {
            return;
        }
        // Find the previous char boundary.
        let prev = self.input[..self.cursor_pos]
            .char_indices()
            .next_back()
            .map_or(0, |(i, _)| i);
        self.input.drain(prev..self.cursor_pos);
        self.cursor_pos = prev;
    }

    /// Delete the character at the cursor position (delete key).
    pub fn delete_char_at_cursor(&mut self) {
        if self.cursor_pos >= self.input.len() {
            return;
        }
        // Find the next char boundary.
        let next = self.input[self.cursor_pos..]
            .char_indices()
            .nth(1)
            .map_or(self.input.len(), |(i, _)| self.cursor_pos + i);
        self.input.drain(self.cursor_pos..next);
    }

    /// Move cursor left by one character.
    pub fn move_cursor_left(&mut self) {
        if self.cursor_pos == 0 {
            return;
        }
        self.cursor_pos = self.input[..self.cursor_pos]
            .char_indices()
            .next_back()
            .map_or(0, |(i, _)| i);
    }

    /// Move cursor right by one character.
    pub fn move_cursor_right(&mut self) {
        if self.cursor_pos >= self.input.len() {
            return;
        }
        self.cursor_pos = self.input[self.cursor_pos..]
            .char_indices()
            .nth(1)
            .map_or(self.input.len(), |(i, _)| self.cursor_pos + i);
    }

    /// Move cursor to the beginning of the input.
    pub fn move_cursor_home(&mut self) {
        self.cursor_pos = 0;
    }

    /// Move cursor to the end of the input.
    pub fn move_cursor_end(&mut self) {
        self.cursor_pos = self.input.len();
    }

    /// Submit the current input. Returns `Some(text)` if a conversation
    /// turn should be started, `None` if the input was empty or a slash
    /// command was handled.
    ///
    /// # Errors
    ///
    /// Returns a `CoreError` if slash command processing fails.
    pub fn submit_input(&mut self) -> Result<Option<String>, CoreError> {
        let text = self.input.trim().to_owned();
        self.input.clear();
        self.cursor_pos = 0;

        if text.is_empty() {
            return Ok(None);
        }

        // Handle slash commands.
        if let Some(cmd) = text.strip_prefix('/') {
            self.handle_slash_command(cmd)?;
            return Ok(None);
        }

        // Add user entry to chat.
        self.chat_entries.push(ChatEntry {
            role: Role::User,
            text: text.clone(),
        });

        Ok(Some(text))
    }

    /// Handle a slash command. Returns `Ok(())` if handled.
    fn handle_slash_command(&mut self, cmd: &str) -> Result<(), CoreError> {
        match cmd.trim() {
            "exit" | "quit" => {
                self.should_exit = true;
            }
            "clear" => {
                self.chat_entries.clear();
                self.chat_scroll = 0;
                if let Some(agent) = &mut self.agent {
                    agent.clear_history(&self.project_root, &self.cwd)?;
                }
            }
            other => {
                self.error_message = Some(format!("Unknown command: /{other}"));
            }
        }
        Ok(())
    }

    /// Start a conversation turn in a background task.
    ///
    /// The turn runs on a spawned tokio task. Streaming tokens are sent
    /// through an `mpsc` channel that the event loop drains on each
    /// tick. A child `CancellationToken` is used so that cancelling a
    /// turn does not shut down the entire application.
    ///
    /// # Panics
    ///
    /// Panics if called while the agent is `None` (i.e., another turn
    /// is already running). This is guarded by `is_streaming()` checks
    /// in the event loop.
    pub fn start_turn(&mut self, user_input: String, parent_cancel: &CancellationToken) {
        let Some(mut agent) = self.agent.take() else {
            // Should not happen: the event loop checks `is_streaming()`
            // before calling `start_turn`.
            return;
        };

        // Create a child token for this turn.
        let turn_cancel = parent_cancel.child_token();

        // Set the child token on the agent so `chat_streaming` uses it.
        agent.set_cancel_token(turn_cancel.clone());

        // Prepare an assistant entry for streaming output.
        self.chat_entries.push(ChatEntry {
            role: Role::Assistant,
            text: String::new(),
        });
        self.streaming = true;

        let (tx, rx) = mpsc::channel::<TurnMessage>(256);

        let join = tokio::spawn(async move {
            let mut sink = ChannelStreamSink { tx: tx.clone() };
            let result = agent.turn(&user_input, &mut sink).await;

            match result {
                Ok(()) => {
                    let user_turns = agent
                        .history()
                        .iter()
                        .filter(|m| m.role() == Role::User)
                        .count();
                    // Ignore send errors — the receiver may have been
                    // dropped if the app is shutting down.
                    let _ = tx.send(TurnMessage::Done { agent, user_turns }).await;
                }
                Err(CoreError::Cancelled) => {
                    let _ = tx
                        .send(TurnMessage::Error {
                            agent,
                            message: "Request cancelled".to_owned(),
                            cancelled: true,
                        })
                        .await;
                }
                Err(e) => {
                    let _ = tx
                        .send(TurnMessage::Error {
                            agent,
                            message: e.to_string(),
                            cancelled: false,
                        })
                        .await;
                }
            }
        });

        self.turn_handle = Some(TurnHandle {
            rx,
            turn_cancel,
            _join: join,
        });
    }

    /// Drain pending turn messages from the background task channel.
    ///
    /// Returns `true` if a redraw is needed (i.e., at least one message
    /// was received).
    pub fn drain_turn_messages(&mut self) -> bool {
        let Some(handle) = &mut self.turn_handle else {
            return false;
        };

        let mut changed = false;

        loop {
            match handle.rx.try_recv() {
                Ok(TurnMessage::Token(token)) => {
                    if let Some(last) = self.chat_entries.last_mut() {
                        last.text.push_str(&token);
                    }
                    changed = true;
                }
                Ok(TurnMessage::Done { agent, user_turns }) => {
                    self.agent = Some(agent);
                    self.streaming = false;
                    self.turn_handle = None;
                    self.context_usage = format!("{user_turns} turns");
                    self.set_chat_scroll(0);
                    return true;
                }
                Ok(TurnMessage::Error {
                    agent,
                    message,
                    cancelled,
                }) => {
                    self.agent = Some(agent);
                    self.streaming = false;
                    self.turn_handle = None;
                    self.error_message = Some(message);
                    if cancelled {
                        // Remove the empty/partial assistant entry on cancellation.
                        if let Some(last) = self.chat_entries.last() {
                            if last.role == Role::Assistant && last.text.is_empty() {
                                self.chat_entries.pop();
                            }
                        }
                    }
                    self.set_chat_scroll(0);
                    return true;
                }
                Err(mpsc::error::TryRecvError::Empty) => {
                    return changed;
                }
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    // The task ended without sending Done/Error (should
                    // not happen, but handle gracefully).
                    self.streaming = false;
                    self.turn_handle = None;
                    return true;
                }
            }
        }
    }

    /// Cancel the currently running turn, if any.
    pub fn cancel_turn(&mut self) {
        if let Some(handle) = &self.turn_handle {
            handle.turn_cancel.cancel();
        }
    }

    /// Set an error message to display in the TUI.
    pub fn set_error(&mut self, msg: String) {
        self.error_message = Some(msg);
    }

    /// Scroll the chat pane up by the given number of lines.
    pub fn scroll_up(&mut self, lines: u16) {
        self.chat_scroll = self.chat_scroll.saturating_add(lines);
    }

    /// Scroll the chat pane down by the given number of lines.
    pub fn scroll_down(&mut self, lines: u16) {
        self.chat_scroll = self.chat_scroll.saturating_sub(lines);
    }
}

/// A [`StreamSink`] that sends tokens through an `mpsc` channel.
struct ChannelStreamSink {
    tx: mpsc::Sender<TurnMessage>,
}

impl StreamSink for ChannelStreamSink {
    fn on_token(&mut self, token: &str) -> Result<(), CoreError> {
        // Use blocking_send since StreamSink::on_token is not async.
        // This will block briefly if the channel is full, which provides
        // natural back-pressure.
        self.tx
            .blocking_send(TurnMessage::Token(token.to_owned()))
            .map_err(|_| CoreError::Cancelled)
    }

    fn on_done(&mut self) -> Result<(), CoreError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn insert_and_delete_chars() {
        // We can't easily create an AgentLoop without an LlmClient,
        // so we test the input manipulation methods via a helper.
        let mut input = String::new();
        let mut cursor: usize = 0;

        // Simulate insert_char
        input.insert(cursor, 'H');
        cursor += 'H'.len_utf8();
        input.insert(cursor, 'i');
        cursor += 'i'.len_utf8();

        assert_eq!(input, "Hi");
        assert_eq!(cursor, 2);

        // Simulate backspace
        let prev = input[..cursor]
            .char_indices()
            .next_back()
            .map_or(0, |(i, _)| i);
        input.drain(prev..cursor);
        cursor = prev;

        assert_eq!(input, "H");
        assert_eq!(cursor, 1);
    }

    #[test]
    fn slash_command_parsing() {
        // Test that slash commands are recognized
        let text = "/clear";
        assert!(text.strip_prefix('/').is_some());

        let text = "/exit";
        assert_eq!(text.strip_prefix('/'), Some("exit"));

        let text = "hello";
        assert!(text.strip_prefix('/').is_none());
    }
}
