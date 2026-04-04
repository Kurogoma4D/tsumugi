//! Agent loop with tool dispatch: streams LLM responses, detects tool
//! calls, executes them in parallel, feeds results back, and loops until
//! the model produces a final text response.

use std::path::Path;
use std::sync::Arc;

use tokio::task::JoinSet;
use tokio_stream::StreamExt as _;
use tokio_util::sync::CancellationToken;

use tmg_llm::{LlmClient, StreamEvent, ToolCall, ToolDefinition};
use tmg_tools::ToolRegistry;

use crate::context::{ContextCompressor, ContextConfig, TokenCounter, truncate_tool_result};
use crate::error::CoreError;
use crate::message::Message;
use crate::prompt;

/// The default system prompt injected at the start of every conversation.
const DEFAULT_SYSTEM_PROMPT: &str = "\
You are tsumugi, a helpful coding assistant running locally. \
Answer concisely and accurately. You have access to tools for \
reading, writing, and patching files, running shell commands, \
listing directories, and searching with grep. Use them when \
appropriate to answer the user's requests.";

/// Maximum number of consecutive tool-call rounds before the loop is
/// forcibly terminated. This prevents infinite loops when the model
/// keeps requesting tools without producing a final text response.
const MAX_TOOL_ROUNDS: usize = 20;

// ---------------------------------------------------------------------------
// StreamSink trait
// ---------------------------------------------------------------------------

/// Callback invoked for streamed events during a conversation turn.
///
/// Implementations should write tokens and tool activity to the
/// appropriate output (stdout, TUI widget, etc.). Returning an error
/// aborts the current turn.
pub trait StreamSink {
    /// Called for each incremental text token from the LLM.
    fn on_token(&mut self, token: &str) -> Result<(), CoreError>;

    /// Called when the LLM response stream has completed for the
    /// current round (there may be more rounds if tool calls follow).
    fn on_done(&mut self) -> Result<(), CoreError> {
        Ok(())
    }

    /// Called when a tool call is about to be dispatched.
    fn on_tool_call(&mut self, _name: &str, _arguments: &str) -> Result<(), CoreError> {
        Ok(())
    }

    /// Called when a tool call has completed with a result.
    fn on_tool_result(
        &mut self,
        _name: &str,
        _output: &str,
        _is_error: bool,
    ) -> Result<(), CoreError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AgentLoop
// ---------------------------------------------------------------------------

/// An agent loop that manages conversation history, streams LLM
/// responses, dispatches tool calls, and loops until the model is done.
///
/// Tool calls within a single LLM response are executed in parallel
/// using a [`JoinSet`]. The loop continues sending tool results back
/// to the LLM until it produces a final text-only response (or the
/// maximum round count is reached).
/// Number of recent messages to preserve during compression.
///
/// These messages are never summarized to ensure the model has
/// immediate context for the current task.
const PRESERVE_RECENT_MESSAGES: usize = 6;

pub struct AgentLoop {
    /// The LLM client used to send requests.
    client: LlmClient,

    /// The tool registry for looking up and executing tools.
    registry: Arc<ToolRegistry>,

    /// Pre-computed tool definitions for the `tools` field in requests.
    tool_defs: Vec<ToolDefinition>,

    /// Conversation history (system prompt + user/assistant/tool messages).
    history: Vec<Message>,

    /// Cancellation token for graceful shutdown.
    cancel: CancellationToken,

    /// Context window configuration.
    context_config: ContextConfig,

    /// Async token counter for tracking context usage.
    token_counter: TokenCounter,

    /// Context compressor for automatic and manual compression.
    compressor: ContextCompressor,
}

impl AgentLoop {
    /// Create a new agent loop with tool support.
    ///
    /// Loads TSUMUGI.md / AGENTS.MD prompt files from the global config
    /// directory and from `project_root` down to `cwd`, injecting them
    /// as initial user-role messages after the system prompt.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Io`] if a prompt file exists but cannot be read.
    pub fn new(
        client: LlmClient,
        registry: ToolRegistry,
        cancel: CancellationToken,
        project_root: &Path,
        cwd: &Path,
    ) -> Result<Self, CoreError> {
        Self::with_context_config(
            client,
            registry,
            cancel,
            project_root,
            cwd,
            ContextConfig::default(),
        )
    }

    /// Create a new agent loop with tool support and custom context config.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Io`] if a prompt file exists but cannot be read.
    pub fn with_context_config(
        client: LlmClient,
        registry: ToolRegistry,
        cancel: CancellationToken,
        project_root: &Path,
        cwd: &Path,
        context_config: ContextConfig,
    ) -> Result<Self, CoreError> {
        let mut history = vec![Message::system(DEFAULT_SYSTEM_PROMPT)];

        // Load prompt files and inject as initial messages.
        let prompt_messages = prompt::load_prompt_files(project_root, cwd)?;
        history.extend(prompt_messages);

        let tool_defs = registry.tool_definitions();
        let registry = Arc::new(registry);
        let token_counter = TokenCounter::new(client.clone());
        let compressor = ContextCompressor::new(client.clone());

        Ok(Self {
            client,
            registry,
            tool_defs,
            history,
            cancel,
            context_config,
            token_counter,
            compressor,
        })
    }

    /// Replace the cancellation token used for future turns.
    ///
    /// This is useful for using a child token per conversation turn so
    /// that cancelling one turn does not shut down the entire app.
    pub fn set_cancel_token(&mut self, token: CancellationToken) {
        self.cancel = token;
    }

    /// Return a read-only view of the conversation history.
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Clear conversation history, retaining only the system prompt and
    /// any injected prompt-file messages.
    ///
    /// This reloads prompt files from `project_root`/`cwd` so the
    /// conversation restarts with a fresh context.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Io`] if a prompt file exists but cannot be read.
    pub fn clear_history(&mut self, project_root: &Path, cwd: &Path) -> Result<(), CoreError> {
        let mut history = vec![Message::system(DEFAULT_SYSTEM_PROMPT)];
        let prompt_messages = prompt::load_prompt_files(project_root, cwd)?;
        history.extend(prompt_messages);
        self.history = history;
        Ok(())
    }

    /// Return the context configuration.
    pub fn context_config(&self) -> &ContextConfig {
        &self.context_config
    }

    /// Return the current cached token count.
    pub fn token_count(&self) -> usize {
        self.token_counter.cached_count()
    }

    /// Request an asynchronous token count update for the current history.
    pub async fn update_token_count(&self) {
        self.token_counter.request_update(&self.history).await;
    }

    /// Manually compress the context (used by `/compact` command).
    ///
    /// # Cancel safety
    ///
    /// If the [`CancellationToken`] is cancelled during compression,
    /// the history remains unchanged and [`CoreError::Cancelled`] is
    /// returned.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] on LLM communication failure or cancellation.
    pub async fn compact(&mut self) -> Result<(), CoreError> {
        let new_history = self
            .compressor
            .compress(&self.history, self.cancel.clone(), PRESERVE_RECENT_MESSAGES)
            .await?;
        self.history = new_history;
        // Update token count after compression.
        self.token_counter.request_update(&self.history).await;
        Ok(())
    }

    /// Check if context compression should be auto-triggered and
    /// compress if needed. Called after each turn completes.
    async fn maybe_auto_compress(&mut self) -> Result<(), CoreError> {
        let current_tokens = self.token_counter.cached_count();
        let trigger = self.context_config.compression_trigger();

        if current_tokens > trigger && trigger > 0 {
            self.compact().await?;
        }
        Ok(())
    }

    /// Execute a single conversation turn with tool dispatch.
    ///
    /// Adds the user message to history, then enters a loop:
    /// 1. Stream the LLM response, collecting text and tool calls.
    /// 2. If tool calls are present, execute them in parallel via
    ///    `JoinSet`, add results to history, and go to step 1.
    /// 3. If no tool calls, append the text response and return.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] on LLM communication failure, tool
    /// execution failure, cancellation, or if the sink returns an error.
    ///
    /// # Cancel safety
    ///
    /// If the `CancellationToken` is triggered during streaming or tool
    /// execution, the partial state is discarded and
    /// [`CoreError::Cancelled`] is returned.
    pub async fn turn(
        &mut self,
        user_input: &str,
        sink: &mut impl StreamSink,
    ) -> Result<(), CoreError> {
        // Add user message to history.
        self.history.push(Message::user(user_input));

        for _round in 0..MAX_TOOL_ROUNDS {
            if self.cancel.is_cancelled() {
                return Err(CoreError::Cancelled);
            }

            let (response_text, tool_calls) = self.stream_one_round(sink).await?;

            if tool_calls.is_empty() {
                // No tool calls: final text response.
                if !response_text.is_empty() {
                    self.history.push(Message::assistant(response_text));
                }

                // Update token count and check for auto-compression.
                self.token_counter.request_update(&self.history).await;
                // Ignore compression errors -- a failed compression should
                // not prevent the turn from completing successfully.
                let _ = self.maybe_auto_compress().await;

                return Ok(());
            }

            // We have tool calls. Record the assistant message with tool
            // calls in history, then execute them.
            self.history.push(Message::assistant_with_tool_calls(
                response_text,
                tool_calls.clone(),
            ));

            self.dispatch_tool_calls(&tool_calls, sink).await?;

            // Loop: send tool results back to the LLM.
        }

        // Safety valve: too many tool rounds.
        self.history.push(Message::assistant(
            "[Agent loop terminated: maximum tool-call rounds reached]".to_owned(),
        ));
        sink.on_done()?;

        // Update token count even on max-rounds termination.
        self.token_counter.request_update(&self.history).await;

        Ok(())
    }

    /// Stream one round of LLM output, returning the accumulated text
    /// and any completed tool calls.
    async fn stream_one_round(
        &self,
        sink: &mut impl StreamSink,
    ) -> Result<(String, Vec<ToolCall>), CoreError> {
        let chat_messages: Vec<_> = self.history.iter().map(Message::to_chat_message).collect();

        let mut stream = self
            .client
            .chat_streaming(chat_messages, self.tool_defs.clone(), self.cancel.clone())
            .await?;

        let mut response_text = String::new();
        let mut tool_calls = Vec::new();

        while let Some(event) = stream.next().await {
            match event {
                Ok(StreamEvent::ContentDelta(token)) => {
                    response_text.push_str(&token);
                    sink.on_token(&token)?;
                }
                Ok(StreamEvent::ToolCallComplete(tc)) => {
                    tool_calls.push(tc);
                }
                Ok(StreamEvent::Done(_)) => {
                    break;
                }
                Err(tmg_llm::LlmError::Cancelled) => {
                    return Err(CoreError::Cancelled);
                }
                Err(e) => {
                    return Err(CoreError::Llm(e));
                }
            }
        }

        sink.on_done()?;

        Ok((response_text, tool_calls))
    }

    /// Execute tool calls in parallel via `JoinSet`, notifying the sink,
    /// and appending tool-result messages to history.
    ///
    /// On error (including cancellation), remaining tasks are dropped
    /// (the `JoinSet` is dropped, aborting outstanding tasks).
    async fn dispatch_tool_calls(
        &mut self,
        tool_calls: &[ToolCall],
        sink: &mut impl StreamSink,
    ) -> Result<(), CoreError> {
        // Notify sink of each tool call before dispatching.
        for tc in tool_calls {
            sink.on_tool_call(&tc.function.name, &tc.function.arguments)?;
        }

        // Spawn each tool call in a JoinSet for parallel execution.
        let mut join_set = JoinSet::new();
        for tc in tool_calls {
            let registry = Arc::clone(&self.registry);
            let name = tc.function.name.clone();
            let arguments_str = tc.function.arguments.clone();
            let call_id = tc.id.clone();
            let cancel = self.cancel.clone();

            join_set.spawn(async move {
                // Check cancellation before executing.
                if cancel.is_cancelled() {
                    return (call_id, name, Err(CoreError::Cancelled));
                }

                // Parse the arguments JSON.
                let params: serde_json::Value = match serde_json::from_str(&arguments_str) {
                    Ok(v) => v,
                    Err(e) => {
                        let result =
                            tmg_tools::ToolResult::error(format!("invalid JSON arguments: {e}"));
                        return (call_id, name, Ok(result));
                    }
                };

                let result = registry.execute(&name, params).await;
                match result {
                    Ok(tool_result) => (call_id, name, Ok(tool_result)),
                    Err(e) => {
                        // Tool errors are reported as error results to the
                        // LLM rather than aborting the entire turn, so the
                        // model can attempt recovery.
                        let error_result = tmg_tools::ToolResult::error(e.to_string());
                        (call_id, name, Ok(error_result))
                    }
                }
            });
        }

        // Collect results in completion order.
        // We store (call_id, tool_name, ToolResult) and sort by original
        // order at the end to maintain deterministic history.
        let mut results: Vec<(String, String, tmg_tools::ToolResult)> =
            Vec::with_capacity(tool_calls.len());

        while let Some(join_result) = join_set.join_next().await {
            let (call_id, name, tool_result) = join_result.map_err(|e| {
                CoreError::Io(std::io::Error::other(format!("task join error: {e}")))
            })?;

            let tool_result = tool_result?;

            sink.on_tool_result(&name, &tool_result.output, tool_result.is_error)?;

            results.push((call_id, name, tool_result));
        }

        // Sort results to match the original tool_calls order so history
        // is deterministic regardless of execution order.
        let call_order: Vec<&str> = tool_calls.iter().map(|tc| tc.id.as_str()).collect();
        results.sort_by_key(|(id, _, _)| {
            call_order
                .iter()
                .position(|&cid| cid == id)
                .unwrap_or(usize::MAX)
        });

        // Append tool-result messages to history, truncating if needed.
        let max_tool_tokens = self.context_config.max_tool_result_tokens;
        for (call_id, _name, tool_result) in results {
            let output = truncate_tool_result(&tool_result.output, max_tool_tokens);
            self.history.push(Message::tool_result(call_id, output));
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
        assert_eq!(sys.role(), tmg_llm::Role::System);
        assert_eq!(sys.content(), "hello");

        let usr = Message::user("world");
        assert_eq!(usr.role(), tmg_llm::Role::User);

        let asst = Message::assistant("response");
        assert_eq!(asst.role(), tmg_llm::Role::Assistant);
    }

    #[test]
    fn message_to_chat_message() {
        let msg = Message::user("test");
        let chat = msg.to_chat_message();
        assert_eq!(chat.role, tmg_llm::Role::User);
        assert_eq!(chat.content.as_deref(), Some("test"));
        assert!(chat.tool_calls.is_none());
        assert!(chat.tool_call_id.is_none());
    }

    #[test]
    fn message_with_tool_calls() {
        let tool_call = ToolCall {
            id: "call_1".to_owned(),
            kind: tmg_llm::ToolKind::Function,
            function: tmg_llm::FunctionCall {
                name: "file_read".to_owned(),
                arguments: r#"{"path": "/tmp/test"}"#.to_owned(),
            },
        };

        let msg = Message::assistant_with_tool_calls("thinking...", vec![tool_call.clone()]);
        assert_eq!(msg.role(), tmg_llm::Role::Assistant);
        assert_eq!(msg.content(), "thinking...");
        assert_eq!(msg.tool_calls().map(<[tmg_llm::ToolCall]>::len), Some(1));

        let chat = msg.to_chat_message();
        assert_eq!(chat.content.as_deref(), Some("thinking..."));
        assert!(chat.tool_calls.is_some());
    }

    #[test]
    fn tool_result_message() {
        let msg = Message::tool_result("call_1", "file contents here");
        assert_eq!(msg.role(), tmg_llm::Role::Tool);
        assert_eq!(msg.content(), "file contents here");

        let chat = msg.to_chat_message();
        assert_eq!(chat.role, tmg_llm::Role::Tool);
        assert_eq!(chat.tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn empty_content_message_serializes_as_none() {
        let msg = Message::assistant_with_tool_calls(
            "",
            vec![ToolCall {
                id: "call_x".to_owned(),
                kind: tmg_llm::ToolKind::Function,
                function: tmg_llm::FunctionCall {
                    name: "test".to_owned(),
                    arguments: "{}".to_owned(),
                },
            }],
        );

        let chat = msg.to_chat_message();
        assert!(chat.content.is_none());
        assert!(chat.tool_calls.is_some());
    }
}
