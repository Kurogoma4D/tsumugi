//! Subagent runner: independent agent loop for a single subagent.
//!
//! Each subagent gets its own conversation history, filtered tool
//! registry, and system prompt. The runner streams LLM responses,
//! dispatches tool calls (only allowed tools), and loops until the
//! model produces a final text response.

use std::sync::Arc;

use tokio::task::JoinSet;
use tokio_stream::StreamExt as _;
use tokio_util::sync::CancellationToken;

use tmg_llm::{LlmClient, StreamEvent, ToolCall, ToolDefinition};
use tmg_sandbox::SandboxContext;
use tmg_tools::ToolRegistry;

use crate::config::AgentKind;
use crate::error::AgentError;

/// Maximum number of consecutive tool-call rounds for a subagent.
const MAX_SUBAGENT_TOOL_ROUNDS: usize = 15;

/// A single message in the subagent's conversation history.
#[derive(Debug, Clone)]
struct SubagentMessage {
    role: tmg_llm::Role,
    content: String,
    tool_calls: Option<Vec<ToolCall>>,
    tool_call_id: Option<String>,
}

impl SubagentMessage {
    fn system(content: &str) -> Self {
        Self {
            role: tmg_llm::Role::System,
            content: content.to_owned(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn user(content: &str) -> Self {
        Self {
            role: tmg_llm::Role::User,
            content: content.to_owned(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn assistant(content: String) -> Self {
        Self {
            role: tmg_llm::Role::Assistant,
            content,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn assistant_with_tool_calls(content: String, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: tmg_llm::Role::Assistant,
            content,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    fn tool_result(call_id: String, output: String) -> Self {
        Self {
            role: tmg_llm::Role::Tool,
            content: output,
            tool_calls: None,
            tool_call_id: Some(call_id),
        }
    }

    fn to_chat_message(&self) -> tmg_llm::ChatMessage {
        let content = if self.content.is_empty() {
            None
        } else {
            Some(self.content.clone())
        };
        tmg_llm::ChatMessage {
            role: self.role,
            content,
            tool_calls: self.tool_calls.clone(),
            tool_call_id: self.tool_call_id.clone(),
        }
    }
}

/// Runs an independent agent loop for a subagent.
///
/// The runner owns its own conversation history and tool registry,
/// completely independent of the main agent loop.
pub struct SubagentRunner {
    /// The LLM client (shared with the main agent).
    client: LlmClient,

    /// The filtered tool registry for this subagent.
    registry: Arc<ToolRegistry>,

    /// Pre-computed tool definitions for the LLM API.
    tool_defs: Vec<ToolDefinition>,

    /// The subagent's conversation history.
    history: Vec<SubagentMessage>,

    /// Cancellation token for graceful shutdown.
    cancel: CancellationToken,

    /// Sandbox context this subagent's tool dispatch operates under.
    ///
    /// Derived from the parent agent's sandbox via
    /// [`SandboxContext::derive`] so the subagent inherits the
    /// workspace root and process budgets but applies the
    /// per-agent-kind [`tmg_sandbox::SandboxMode`] (e.g. `ReadOnly`
    /// for `explore`, `WorkspaceWrite` for `worker`).
    sandbox: Arc<SandboxContext>,
}

impl SubagentRunner {
    /// Create a new subagent runner.
    ///
    /// The runner initializes with a system prompt appropriate for
    /// the agent kind (built-in or custom) and a filtered tool registry.
    /// The supplied `sandbox` is the per-subagent context every tool
    /// dispatch will receive; callers (typically
    /// [`SubagentManager`](crate::manager::SubagentManager)) are
    /// responsible for deriving it from the parent context with the
    /// appropriate [`tmg_sandbox::SandboxMode`] for the agent kind.
    pub fn new(
        client: LlmClient,
        registry: ToolRegistry,
        agent_kind: &AgentKind,
        cancel: CancellationToken,
        sandbox: Arc<SandboxContext>,
    ) -> Self {
        let tool_defs = registry.tool_definitions();
        let registry = Arc::new(registry);

        let history = vec![SubagentMessage::system(agent_kind.system_prompt())];

        Self {
            client,
            registry,
            tool_defs,
            history,
            cancel,
            sandbox,
        }
    }

    /// Execute the subagent's task and return the final text response.
    ///
    /// The subagent runs its own agent loop: streaming LLM responses,
    /// dispatching tool calls, and looping until a final text response
    /// is produced.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] on LLM failure, tool errors, or
    /// cancellation.
    ///
    /// # Cancel safety
    ///
    /// If the `CancellationToken` fires during execution, in-flight
    /// tool tasks are dropped via `JoinSet` drop and
    /// [`AgentError::Cancelled`] is returned.
    pub async fn run(&mut self, task: &str) -> Result<String, AgentError> {
        self.history.push(SubagentMessage::user(task));

        for _round in 0..MAX_SUBAGENT_TOOL_ROUNDS {
            if self.cancel.is_cancelled() {
                return Err(AgentError::Cancelled);
            }

            let (response_text, tool_calls) = self.stream_one_round().await?;

            if tool_calls.is_empty() {
                if !response_text.is_empty() {
                    self.history
                        .push(SubagentMessage::assistant(response_text.clone()));
                }
                return Ok(response_text);
            }

            // Record the assistant message with tool calls.
            self.history
                .push(SubagentMessage::assistant_with_tool_calls(
                    response_text,
                    tool_calls.clone(),
                ));

            self.dispatch_tool_calls(&tool_calls).await?;
        }

        Ok("[Subagent terminated: maximum tool-call rounds reached]".to_owned())
    }

    /// Stream one round of LLM output.
    ///
    /// # Cancel safety
    ///
    /// Uses `tokio::select!` to respond to cancellation promptly even
    /// during slow LLM responses. The `stream.next()` future is
    /// cancel-safe (dropping it just means we stop reading from the
    /// stream).
    async fn stream_one_round(&self) -> Result<(String, Vec<ToolCall>), AgentError> {
        let chat_messages: Vec<_> = self
            .history
            .iter()
            .map(SubagentMessage::to_chat_message)
            .collect();

        let mut stream = self
            .client
            .chat_streaming(chat_messages, self.tool_defs.clone(), self.cancel.clone())
            .await
            .map_err(AgentError::Llm)?;

        let mut response_text = String::new();
        let mut tool_calls = Vec::new();

        loop {
            tokio::select! {
                () = self.cancel.cancelled() => {
                    return Err(AgentError::Cancelled);
                }
                maybe_event = stream.next() => {
                    let Some(event) = maybe_event else {
                        break;
                    };
                    match event {
                        Ok(StreamEvent::ThinkingDelta(_)) => {
                            // Subagent thinking tokens are discarded.
                        }
                        Ok(StreamEvent::ContentDelta(token)) => {
                            response_text.push_str(&token);
                        }
                        Ok(StreamEvent::ToolCallComplete(tc)) => {
                            // Check: subagents must not call spawn_agent.
                            if tc.function.name == "spawn_agent" {
                                return Err(AgentError::NestingForbidden);
                            }
                            tool_calls.push(tc);
                        }
                        Ok(StreamEvent::Done(_)) => {
                            break;
                        }
                        Err(tmg_llm::LlmError::Cancelled) => {
                            return Err(AgentError::Cancelled);
                        }
                        Err(e) => {
                            return Err(AgentError::Llm(e));
                        }
                    }
                }
            }
        }

        Ok((response_text, tool_calls))
    }

    /// Execute tool calls in parallel via `JoinSet`.
    ///
    /// # Cancellation behaviour
    ///
    /// If any tool task returns `AgentError::Cancelled`, the `?`
    /// propagation causes this method to return early and discard
    /// results from other (potentially completed) tool calls. This is
    /// intentional: once cancellation is signaled the entire subagent
    /// loop is shutting down, so preserving partial results is not
    /// useful. The `JoinSet` is dropped, which aborts remaining tasks.
    async fn dispatch_tool_calls(&mut self, tool_calls: &[ToolCall]) -> Result<(), AgentError> {
        let mut join_set = JoinSet::new();

        for tc in tool_calls {
            let registry = Arc::clone(&self.registry);
            let sandbox = Arc::clone(&self.sandbox);
            let name = tc.function.name.clone();
            let arguments_str = tc.function.arguments.clone();
            let call_id = tc.id.clone();
            let cancel = self.cancel.clone();

            join_set.spawn(async move {
                if cancel.is_cancelled() {
                    return (call_id, Err(AgentError::Cancelled));
                }

                let params: serde_json::Value = match serde_json::from_str(&arguments_str) {
                    Ok(v) => v,
                    Err(e) => {
                        let result =
                            tmg_tools::ToolResult::error(format!("invalid JSON arguments: {e}"));
                        return (call_id, Ok(result));
                    }
                };

                match registry.execute(&name, params, &sandbox).await {
                    Ok(tool_result) => (call_id, Ok(tool_result)),
                    Err(e) => {
                        let error_result = tmg_tools::ToolResult::error(e.to_string());
                        (call_id, Ok(error_result))
                    }
                }
            });
        }

        let mut results: Vec<(String, tmg_tools::ToolResult)> =
            Vec::with_capacity(tool_calls.len());

        while let Some(join_result) = join_set.join_next().await {
            let (call_id, tool_result) = join_result.map_err(|e| AgentError::JoinError {
                message: e.to_string(),
            })?;

            let tool_result = tool_result?;
            results.push((call_id, tool_result));
        }

        // Sort results to match original order for deterministic history.
        let call_order: Vec<&str> = tool_calls.iter().map(|tc| tc.id.as_str()).collect();
        results.sort_by_key(|(id, _)| {
            call_order
                .iter()
                .position(|&cid| cid == id)
                .unwrap_or(usize::MAX)
        });

        for (call_id, tool_result) in results {
            self.history
                .push(SubagentMessage::tool_result(call_id, tool_result.output));
        }

        Ok(())
    }
}
