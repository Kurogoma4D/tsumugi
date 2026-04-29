//! Agent loop with tool dispatch: streams LLM responses, detects tool
//! calls, executes them in parallel, feeds results back, and loops until
//! the model produces a final text response.

use std::path::Path;
use std::sync::Arc;

use tokio::task::JoinSet;
use tokio_stream::StreamExt as _;
use tokio_util::sync::CancellationToken;

use tmg_llm::{
    LlmClient, StreamEvent, ToolCall, ToolCallingMode, ToolDefinition, build_tool_calling_prompt,
    parse_tool_calls,
};
use tmg_sandbox::SandboxContext;
use tmg_tools::ToolRegistry;

use crate::context::{ContextCompressor, ContextConfig, TokenCounter};
use crate::error::CoreError;
use crate::message::Message;
use crate::prompt;

/// The default system prompt injected at the start of every conversation.
const DEFAULT_SYSTEM_PROMPT: &str = "\
You are tsumugi, a helpful coding assistant running locally. \
Answer concisely and accurately. You have access to tools for \
reading, writing, and patching files, running shell commands, \
listing directories, and searching with grep. Use them when \
appropriate to answer the user's requests. Before starting on a \
task, scan the [memory] index for topics related to the request \
and call the `memory` tool's `read` action on any that look \
relevant; record durable lessons via `add` / `update`.";

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
    /// Called for each incremental thinking/reasoning token from the LLM.
    fn on_thinking(&mut self, _token: &str) -> Result<(), CoreError> {
        Ok(())
    }

    /// Called for each incremental text token from the LLM.
    fn on_token(&mut self, token: &str) -> Result<(), CoreError>;

    /// Called when the LLM response stream has completed for the
    /// current round (there may be more rounds if tool calls follow).
    fn on_done(&mut self) -> Result<(), CoreError> {
        Ok(())
    }

    /// Called when a tool call is about to be dispatched.
    ///
    /// `call_id` is the unique tool-call identifier from the LLM; sinks
    /// that need to correlate this call with its later
    /// [`Self::on_tool_result`] / [`Self::on_tool_result_compressed`]
    /// callback should stamp it on the entry they create here.
    fn on_tool_call(
        &mut self,
        _call_id: &str,
        _name: &str,
        _arguments: &str,
    ) -> Result<(), CoreError> {
        Ok(())
    }

    /// Called when a tool call has completed with a result.
    ///
    /// `call_id` is the same identifier passed to the matching
    /// [`Self::on_tool_call`] so concurrent tool calls of the same
    /// `name` can be paired without ambiguity.
    fn on_tool_result(
        &mut self,
        _call_id: &str,
        _name: &str,
        _output: &str,
        _is_error: bool,
    ) -> Result<(), CoreError> {
        Ok(())
    }

    /// Called after [`Self::on_tool_result`] when the recorded
    /// (history-bound) version of the output was rewritten by
    /// [`crate::context::ContextCompressor::compress_tool_result`].
    ///
    /// `call_id` matches the preceding [`Self::on_tool_result`] so a
    /// "compressed" marker can be stamped on the right entry even
    /// when concurrent `file_read` calls of the same `name` interleave.
    ///
    /// `symbol_count` is the number of structural symbols extracted by
    /// tree-sitter. Sinks that do not care about compression metadata
    /// can leave the default no-op implementation.
    ///
    /// Note: this hook is **only** fired when the result was rewritten
    /// via signature extraction (i.e. issue #49's tree-sitter path).
    /// Tail-truncated results — the pre-#49 behaviour, still the
    /// fallback for non-`file_read` tools — do **not** fire it.
    fn on_tool_result_compressed(
        &mut self,
        _call_id: &str,
        _name: &str,
        _symbol_count: usize,
    ) -> Result<(), CoreError> {
        Ok(())
    }

    /// Called when a non-fatal warning occurs (e.g., auto-compression failure).
    fn on_warning(&mut self, _message: &str) -> Result<(), CoreError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AgentLoop
// ---------------------------------------------------------------------------

/// Number of recent messages to preserve during compression.
///
/// These messages are never summarized to ensure the model has
/// immediate context for the current task.
const PRESERVE_RECENT_MESSAGES: usize = 6;

/// Lightweight summary of one [`AgentLoop::turn`] handed to the
/// optional [`TurnObserver`] callback.
///
/// Carries enough metric inputs for the harness's auto-promotion gate
/// (SPEC §9.10) without forcing the loop to reach into the harness.
/// Producing it is allocation-cheap: the underlying counters live on
/// `AgentLoop` already, and the user-message text is owned anyway.
#[derive(Debug, Clone, Default)]
pub struct TurnSummary {
    /// Token count after the turn, taken from the loop's cached counter.
    pub tokens_used: usize,

    /// Number of tool calls observed across all rounds in this turn.
    pub tool_calls: u32,

    /// User input that opened this turn. Carried verbatim so harness
    /// keyword detectors can see the original text.
    pub user_message: String,
}

/// Boxed callback invoked once per [`AgentLoop::turn`] completion.
///
/// The harness installs one of these via
/// [`AgentLoop::set_turn_observer`] to feed the auto-promotion gate
/// (SPEC §9.10). Callbacks must not panic; any error is intentionally
/// swallowed by the loop because the metric channel is best-effort.
pub type TurnObserver = Box<dyn Fn(&TurnSummary) + Send + Sync>;

/// An agent loop that manages conversation history, streams LLM
/// responses, dispatches tool calls, and loops until the model is done.
///
/// Tool calls within a single LLM response are executed in parallel
/// using a [`JoinSet`]. The loop continues sending tool results back
/// to the LLM until it produces a final text-only response (or the
/// maximum round count is reached).
pub struct AgentLoop {
    /// The LLM client used to send requests.
    client: LlmClient,

    /// The tool registry for looking up and executing tools.
    registry: Arc<ToolRegistry>,

    /// Pre-computed tool definitions for the `tools` field in requests.
    /// Wrapped in `Arc` to avoid deep-cloning every round when passed to
    /// the streaming request in `Native` / `Auto` modes.
    tool_defs: Arc<Vec<ToolDefinition>>,

    /// Conversation history (system prompt + user/assistant/tool messages).
    history: Vec<Message>,

    /// Cancellation token for graceful shutdown.
    cancel: CancellationToken,

    /// Context window configuration.
    context_config: ContextConfig,

    /// Async token counter for tracking context usage.
    token_counter: TokenCounter,

    /// Context compressor for automatic and manual compression.
    ///
    /// Stored behind an [`Arc`] so each spawned tool dispatch task in
    /// [`Self::dispatch_tool_calls`] can call into the compressor
    /// without a deep clone (the synchronous `compress_tool_result`
    /// path runs inside the worker task to avoid the per-call
    /// `serde_json::Value::clone` overhead).
    compressor: Arc<ContextCompressor>,

    /// Tool calling strategy (native, `prompt_based`, or auto).
    tool_calling_mode: ToolCallingMode,

    /// Sandbox context every tool dispatch runs under.
    ///
    /// Constructed by the CLI from the operator's `[sandbox]` config
    /// and threaded into [`Self::dispatch_tool_calls`]. The sandbox is
    /// a required constructor argument (see [`Self::new`] /
    /// [`Self::with_context_config`]) so production callers cannot
    /// silently fall back to an unrestricted [`SandboxContext`] by
    /// forgetting to install one.
    sandbox: Arc<SandboxContext>,

    /// Optional callback invoked at the end of every [`Self::turn`].
    ///
    /// Installed by the harness wire-up to feed the auto-promotion
    /// gate (SPEC §9.10). The default is `None`, so non-harness
    /// callers (e.g. one-shot CLI mode) pay nothing.
    turn_observer: Option<TurnObserver>,

    /// Optional memory index injected into the prompt (issue #52).
    ///
    /// Stored so that [`Self::clear_history`] can re-inject the same
    /// payload without forcing the caller to thread it back in. Set
    /// via [`Self::set_memory_index`]; `None` means the memory layer
    /// is disabled or the index is empty.
    memory_index: Option<String>,

    /// Shared pending-swap slot consulted at the top of every
    /// [`Self::turn`]. When the `memory` tool succeeds it pushes a
    /// freshly-rendered index here; the next turn picks it up,
    /// replaces [`Self::memory_index`], and pushes a refreshed
    /// memory user message into history so the LLM sees the change.
    /// Issue #4 of PR #76 review.
    ///
    /// `None` means no provider has been wired (e.g. one-shot mode
    /// without a memory tool). Cloned via `Arc` so the producer side
    /// (the tool callback) and the consumer (`Self::turn`) share the
    /// same slot.
    pending_memory_index: Option<Arc<std::sync::Mutex<Option<String>>>>,

    /// Tracks whether [`Self::inject_memory_index_now`] has already
    /// emitted a user-role message for the current `memory_index`.
    /// Issue #16 of PR #76 review: previously a second call would
    /// silently duplicate the memory block.
    memory_index_injected: bool,
}

impl AgentLoop {
    /// Create a new agent loop with tool support.
    ///
    /// Loads TSUMUGI.md / AGENTS.MD prompt files from the global config
    /// directory and from `project_root` down to `cwd`, injecting them
    /// as initial user-role messages after the system prompt.
    ///
    /// The `sandbox` argument is **required**: it is the
    /// [`SandboxContext`] every dispatched tool runs under. Making it
    /// a constructor argument prevents callers from accidentally
    /// running tools under an unrestricted default
    /// (issue #47 follow-up).
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
        sandbox: Arc<SandboxContext>,
    ) -> Result<Self, CoreError> {
        Self::with_context_config(
            client,
            registry,
            cancel,
            project_root,
            cwd,
            ContextConfig::default(),
            ToolCallingMode::Auto,
            sandbox,
        )
    }

    /// Create a new agent loop with tool support and custom context config.
    ///
    /// The `sandbox` argument is **required**: see [`Self::new`].
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Io`] if a prompt file exists but cannot be read.
    #[expect(
        clippy::too_many_arguments,
        reason = "constructor wires together independent capabilities (LLM, registry, paths, context budget, mode, sandbox); grouping them into a config struct adds boilerplate without clarifying intent"
    )]
    pub fn with_context_config(
        client: LlmClient,
        registry: ToolRegistry,
        cancel: CancellationToken,
        project_root: &Path,
        cwd: &Path,
        context_config: ContextConfig,
        tool_calling_mode: ToolCallingMode,
        sandbox: Arc<SandboxContext>,
    ) -> Result<Self, CoreError> {
        let tool_defs = Arc::new(registry.tool_definitions());

        let system_prompt = Self::build_system_prompt(tool_calling_mode, &tool_defs);

        let mut history = vec![Message::system(system_prompt)];

        // Load prompt files (and the memory index, if available)
        // and inject as initial messages.
        let prompt_messages = prompt::load_prompt_files(project_root, cwd)?;
        history.extend(prompt_messages);
        // The memory index is wired in lazily via `set_memory_index`;
        // the initial constructor cannot read it without forcing every
        // call site to provide a `MemoryStore`. The CLI startup path
        // calls `set_memory_index` immediately after construction.

        let registry = Arc::new(registry);
        let token_counter = TokenCounter::new(client.clone());
        let compressor = Arc::new(ContextCompressor::with_config(
            client.clone(),
            context_config.clone(),
        ));

        Ok(Self {
            client,
            registry,
            tool_defs,
            history,
            cancel,
            context_config,
            token_counter,
            compressor,
            tool_calling_mode,
            sandbox,
            turn_observer: None,
            memory_index: None,
            pending_memory_index: None,
            memory_index_injected: false,
        })
    }

    /// Install (or replace) the memory index injected into the system
    /// prompt on every [`Self::clear_history`].
    ///
    /// Pass `None` to disable the injection. The caller is responsible
    /// for refreshing this whenever the index changes (e.g. after a
    /// `memory` tool write). To make the new value visible to the LLM
    /// without re-reading prompt files, follow up with
    /// [`Self::clear_history`] (which re-injects from
    /// [`prompt::load_prompt_files_with_memory`]).
    pub fn set_memory_index(&mut self, index: Option<String>) {
        self.memory_index = index;
        // Clear the injection flag so the new value can be (re-)pushed
        // into history at the next opportunity.
        self.memory_index_injected = false;
    }

    /// Install (or replace) the shared pending-swap slot consulted at
    /// the top of every [`Self::turn`]. Producers — typically the
    /// `memory` tool's refresh callback — push freshly-rendered index
    /// snapshots here; the agent loop picks them up before sending the
    /// next request to the LLM, so the model always sees the
    /// post-mutation state of the project memory. Issue #4 of PR #76
    /// review.
    ///
    /// Pass `None` to detach the slot. The slot is consulted with
    /// [`std::sync::Mutex::lock`]; poisoning is treated as no-op.
    pub fn set_pending_memory_index(
        &mut self,
        slot: Option<Arc<std::sync::Mutex<Option<String>>>>,
    ) {
        self.pending_memory_index = slot;
    }

    /// Borrow the shared pending-swap slot, if any.
    #[must_use]
    pub fn pending_memory_index(&self) -> Option<&Arc<std::sync::Mutex<Option<String>>>> {
        self.pending_memory_index.as_ref()
    }

    /// Drain the shared pending-swap slot (if any) and apply the new
    /// index. When a swap happens we replace [`Self::memory_index`]
    /// AND push a fresh user-role memory block into history so the
    /// LLM observes the post-mutation state on its next request.
    ///
    /// Returns `Some(new_index_payload)` when a swap occurred,
    /// `None` otherwise. The returned payload is the same one pushed
    /// into history, suitable for stream-event emission.
    fn apply_pending_memory_index(&mut self) -> Option<String> {
        let slot = self.pending_memory_index.as_ref()?;
        let mut guard = slot.lock().ok()?;
        let new_index = guard.take()?;
        drop(guard);

        let trimmed = new_index.trim();
        if trimmed.is_empty() {
            self.memory_index = None;
            self.memory_index_injected = false;
            return None;
        }

        self.memory_index = Some(new_index.clone());
        let payload = format!("{}\n\n{trimmed}", prompt::MEMORY_PROMPT_HEADER);
        self.history.push(Message::user(payload.clone()));
        // Mark as injected so a subsequent `inject_memory_index_now()`
        // call is a no-op until the next swap.
        self.memory_index_injected = true;
        Some(payload)
    }

    /// Inject the carried memory index as a user-role message at the
    /// end of the current history. Idempotent: a second call without
    /// an intervening [`Self::set_memory_index`] / pending-swap apply
    /// is a no-op. Issue #16 of PR #76 review.
    pub fn inject_memory_index_now(&mut self) {
        if self.memory_index_injected {
            return;
        }
        let Some(index) = self.memory_index.as_deref() else {
            return;
        };
        if index.trim().is_empty() {
            return;
        }
        let payload = format!("{}\n\n{index}", prompt::MEMORY_PROMPT_HEADER);
        self.history.push(Message::user(payload));
        self.memory_index_injected = true;
    }

    /// Borrow the active memory index (if any).
    #[must_use]
    pub fn memory_index(&self) -> Option<&str> {
        self.memory_index.as_deref()
    }

    /// Replace the active [`SandboxContext`].
    ///
    /// Constructor-time installation via [`Self::new`] /
    /// [`Self::with_context_config`] is the canonical path; this
    /// setter exists only for explicit reconfiguration (e.g. a future
    /// `tmg run upgrade` flow that re-derives the sandbox after
    /// promotion).
    pub fn set_sandbox(&mut self, sandbox: Arc<SandboxContext>) {
        self.sandbox = sandbox;
    }

    /// Borrow the active [`SandboxContext`].
    #[must_use]
    pub fn sandbox(&self) -> &Arc<SandboxContext> {
        &self.sandbox
    }

    /// Install (or replace) the per-turn observer callback.
    ///
    /// Pass `None` to clear an existing observer. The callback is
    /// invoked at most once per [`Self::turn`] call; if the agent
    /// loop is cancelled mid-turn, the observer is **not** invoked.
    pub fn set_turn_observer(&mut self, observer: Option<TurnObserver>) {
        self.turn_observer = observer;
    }

    /// Build the system prompt string based on the tool calling mode.
    ///
    /// In `PromptBased` or `Auto` mode, appends the tool calling convention
    /// so the model knows how to emit `<tool_call>` tags.
    fn build_system_prompt(
        tool_calling_mode: ToolCallingMode,
        tool_defs: &[ToolDefinition],
    ) -> String {
        match tool_calling_mode {
            ToolCallingMode::Native => DEFAULT_SYSTEM_PROMPT.to_owned(),
            ToolCallingMode::PromptBased | ToolCallingMode::Auto => {
                let suffix = build_tool_calling_prompt(tool_defs);
                format!("{DEFAULT_SYSTEM_PROMPT}{suffix}")
            }
        }
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

    /// Insert a system message immediately after the initial system
    /// prompt at `history[0]`.
    ///
    /// The harness startup path uses this to inject the
    /// `session_bootstrap` payload before the first user turn. Inserting
    /// at index `1` (rather than appending after any prompt-file
    /// user-role messages) keeps the conversation compatible with
    /// chat templates that reject `system` messages following a `user`
    /// turn (Mistral, Qwen, Gemma, etc.).
    ///
    /// Subsequent calls insert at the same position, so the most-recent
    /// bootstrap appears closest to the initial system prompt; existing
    /// bootstrap messages are pushed one slot deeper. Bootstraps are
    /// preserved across the loop until the next
    /// [`clear_history`](Self::clear_history).
    pub fn insert_bootstrap_system_message(&mut self, content: impl Into<String>) {
        // Defensive: if `history` is empty (impossible in normal use —
        // `with_context_config` always seeds index 0 with the system
        // prompt), fall back to push so we don't panic.
        let position = usize::from(!self.history.is_empty());
        self.history.insert(position, Message::system(content));
    }

    /// Clear conversation history, retaining only the system prompt and
    /// any injected prompt-file messages.
    ///
    /// This reloads prompt files from `project_root`/`cwd` so the
    /// conversation restarts with a fresh context.
    ///
    /// **Bootstrap messages are dropped.** Any system messages added
    /// via [`insert_bootstrap_system_message`](Self::insert_bootstrap_system_message)
    /// (e.g. the `session_bootstrap` payload injected at startup) are
    /// removed by this call. Callers that want the bootstrap preserved
    /// across `/clear` must re-inject it after `clear_history` returns.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Io`] if a prompt file exists but cannot be read.
    pub fn clear_history(&mut self, project_root: &Path, cwd: &Path) -> Result<(), CoreError> {
        let system_prompt = Self::build_system_prompt(self.tool_calling_mode, &self.tool_defs);
        let mut history = vec![Message::system(system_prompt)];
        let prompt_messages =
            prompt::load_prompt_files_with_memory(project_root, cwd, self.memory_index.as_deref())?;
        history.extend(prompt_messages);
        self.history = history;
        // The fresh history already contains the memory block (via
        // `load_prompt_files_with_memory`), so mark injected so that a
        // follow-up `inject_memory_index_now()` is a no-op (issue #16).
        self.memory_index_injected = self.memory_index.is_some();
        Ok(())
    }

    /// Return the context configuration.
    pub fn context_config(&self) -> &ContextConfig {
        &self.context_config
    }

    /// Return the active tool calling mode.
    pub fn tool_calling_mode(&self) -> ToolCallingMode {
        self.tool_calling_mode
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
        // Drain any pending memory-index swap before the user message.
        // The `memory` tool's refresh callback may have pushed a fresh
        // snapshot into the shared slot since the last turn ended; we
        // pick it up here so the LLM sees the post-mutation index on
        // its next request. Issue #4 of PR #76 review. The sink hears
        // about the swap via `on_warning` so the TUI Activity Pane
        // can render a "memory updated" hint.
        if let Some(payload) = self.apply_pending_memory_index() {
            // Best-effort: a sink that errors here would abort the
            // turn before it begins. Surfacing the message as a
            // warning is enough to drive the TUI hint without making
            // memory churn fatal.
            let preview: String = payload.chars().take(80).collect();
            let _ = sink.on_warning(&format!("memory updated: {preview}"));
        }

        // Add user message to history.
        self.history.push(Message::user(user_input));

        // Track per-turn metrics for the optional `turn_observer`. We
        // only need a tool-call counter; tokens come from the cached
        // counter at turn end and the user message is already owned.
        let mut tool_calls_total: u32 = 0;

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
                // Surface compression errors as a warning so users know
                // compression failed, but do not abort the turn.
                if let Err(e) = self.maybe_auto_compress().await {
                    sink.on_warning(&format!("auto-compression failed: {e}"))?;
                }

                self.notify_turn_observer(user_input, tool_calls_total);
                return Ok(());
            }

            // We have tool calls. Record the assistant message with tool
            // calls in history, then execute them.
            tool_calls_total = tool_calls_total
                .saturating_add(u32::try_from(tool_calls.len()).unwrap_or(u32::MAX));
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

        self.notify_turn_observer(user_input, tool_calls_total);
        Ok(())
    }

    /// Invoke the installed [`TurnObserver`] (if any) with a fresh
    /// [`TurnSummary`].
    ///
    /// Errors from the observer are intentionally swallowed: the
    /// metric channel is best-effort and must not abort an already-
    /// completed turn.
    fn notify_turn_observer(&self, user_input: &str, tool_calls_total: u32) {
        if let Some(observer) = self.turn_observer.as_ref() {
            let summary = TurnSummary {
                tokens_used: self.token_counter.cached_count(),
                tool_calls: tool_calls_total,
                user_message: user_input.to_owned(),
            };
            observer(&summary);
        }
    }

    /// Stream one round of LLM output, returning the accumulated text
    /// and any completed tool calls.
    ///
    /// Tool call extraction depends on [`ToolCallingMode`]:
    /// - **Native**: tool calls come via the streaming API's `tool_call` deltas.
    /// - **Prompt-based**: tool calls are extracted by parsing `<tool_call>` tags
    ///   from the text content. Native `tools` are not sent in the request.
    /// - **Auto**: native tool calls are used if present; otherwise the text
    ///   content is scanned for `<tool_call>` tags as a fallback.
    async fn stream_one_round(
        &self,
        sink: &mut impl StreamSink,
    ) -> Result<(String, Vec<ToolCall>), CoreError> {
        let chat_messages: Vec<_> = self.history.iter().map(Message::to_chat_message).collect();

        // In prompt_based mode, do not send native tool definitions;
        // the model learns tools from the system prompt instead.
        let request_tools = match self.tool_calling_mode {
            ToolCallingMode::PromptBased => vec![],
            ToolCallingMode::Native | ToolCallingMode::Auto => self.tool_defs.to_vec(),
        };

        let mut stream = self
            .client
            .chat_streaming(chat_messages, request_tools, self.cancel.clone())
            .await?;

        let mut response_text = String::new();
        let mut native_tool_calls = Vec::new();

        while let Some(event) = stream.next().await {
            match event {
                Ok(StreamEvent::ThinkingDelta(token)) => {
                    sink.on_thinking(&token)?;
                }
                Ok(StreamEvent::ContentDelta(token)) => {
                    response_text.push_str(&token);
                    sink.on_token(&token)?;
                }
                Ok(StreamEvent::ToolCallComplete(tc)) => {
                    native_tool_calls.push(tc);
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

        // Determine final tool calls based on mode.
        let (final_text, tool_calls) = match self.tool_calling_mode {
            ToolCallingMode::Native => (response_text, native_tool_calls),
            ToolCallingMode::PromptBased => {
                // NOTE: During streaming, raw `<tool_call>` tags are delivered
                // to the sink before we can parse and strip them here. This is
                // expected behavior — buffering the entire response to strip
                // tags before streaming would defeat the purpose of incremental
                // token delivery. The stored history uses the cleaned text.
                let (cleaned, parsed, errors) = parse_tool_calls(&response_text);
                for err in &errors {
                    sink.on_warning(&format!("prompt-based tool call parse error: {err}"))?;
                }
                (cleaned, parsed)
            }
            ToolCallingMode::Auto => {
                if native_tool_calls.is_empty() {
                    // Fallback: try parsing <tool_call> tags from text.
                    let (cleaned, parsed, errors) = parse_tool_calls(&response_text);
                    for err in &errors {
                        sink.on_warning(&format!(
                            "auto-mode fallback tool call parse error: {err}"
                        ))?;
                    }
                    if parsed.is_empty() {
                        // No tool calls found via either method.
                        (response_text, Vec::new())
                    } else {
                        (cleaned, parsed)
                    }
                } else {
                    // Native tool calls were returned; use them.
                    // Strip any residual <tool_call> tags that the model may
                    // have emitted alongside native tool calls.
                    let (cleaned, _, _) = parse_tool_calls(&response_text);
                    (cleaned, native_tool_calls)
                }
            }
        };

        Ok((final_text, tool_calls))
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
        // Notify sink of each tool call before dispatching. `call_id`
        // is threaded through so concurrent calls of the same `name`
        // can be paired with their result/compression hooks downstream.
        for tc in tool_calls {
            sink.on_tool_call(&tc.id, &tc.function.name, &tc.function.arguments)?;
        }

        // Spawn each tool call in a JoinSet for parallel execution.
        // The compressor is `Arc`-cloned (cheap pointer bump) so each
        // task can run `compress_tool_result` on its own owned `params`,
        // avoiding the per-call `serde_json::Value::clone` of issue #5.
        let mut join_set = JoinSet::new();
        for tc in tool_calls {
            let registry = Arc::clone(&self.registry);
            let sandbox = Arc::clone(&self.sandbox);
            let compressor = Arc::clone(&self.compressor);
            let name = tc.function.name.clone();
            let arguments_str = tc.function.arguments.clone();
            let call_id = tc.id.clone();
            let cancel = self.cancel.clone();

            join_set.spawn(async move {
                // Check cancellation before executing. We surface
                // cancellation as a sentinel `cancelled` flag in the
                // payload rather than as a `CoreError` so the result
                // type stays `Clone`-free at the channel boundary.
                if cancel.is_cancelled() {
                    return DispatchOutcome {
                        call_id,
                        name,
                        cancelled: true,
                        payload: None,
                    };
                }

                // Parse the arguments JSON.
                let params: serde_json::Value = match serde_json::from_str(&arguments_str) {
                    Ok(v) => v,
                    Err(e) => {
                        let result =
                            tmg_tools::ToolResult::error(format!("invalid JSON arguments: {e}"));
                        // Compress using a null-params view; non-
                        // `file_read` tools never consult `params`.
                        let compressed = compressor.compress_tool_result(
                            &name,
                            &serde_json::Value::Null,
                            &result.output,
                        );
                        return DispatchOutcome {
                            call_id,
                            name,
                            cancelled: false,
                            payload: Some((result, compressed)),
                        };
                    }
                };

                // Execute the tool. We need `params` again for the
                // compressor call afterwards, but only the `file_read`
                // path actually reads it. Keep `params` owned by the
                // task so the registry call can take it by value
                // without a redundant clone at the dispatcher level.
                let exec_result = registry.execute(&name, params.clone(), &sandbox).await;
                let tool_result = match exec_result {
                    Ok(r) => r,
                    Err(e) => {
                        // Tool errors are reported as error results to
                        // the LLM rather than aborting the turn so the
                        // model can attempt recovery.
                        tmg_tools::ToolResult::error(e.to_string())
                    }
                };

                // Compress *inside* the spawned task. This moves the
                // `serde_json::Value` we already own here into the
                // synchronous compressor call without copying the JSON
                // tree, fixing the per-call `params.clone()` overhead
                // that the previous on-collect approach incurred.
                let compressed =
                    compressor.compress_tool_result(&name, &params, &tool_result.output);
                DispatchOutcome {
                    call_id,
                    name,
                    cancelled: false,
                    payload: Some((tool_result, compressed)),
                }
            });
        }

        // Collect results as they finish. We *do not* fire sink hooks
        // here — sinks are invoked in the deterministic call order
        // after the sort below so concurrent calls of the same `name`
        // never cross-stamp activity entries (issue #6 / #7).
        let mut results: Vec<DispatchOutcome> = Vec::with_capacity(tool_calls.len());

        while let Some(join_result) = join_set.join_next().await {
            let outcome = join_result.map_err(|e| {
                CoreError::Io(std::io::Error::other(format!("task join error: {e}")))
            })?;
            if outcome.cancelled {
                return Err(CoreError::Cancelled);
            }
            results.push(outcome);
        }

        // Sort results to match the original tool_calls order so
        // history (and the sink fire order) is deterministic
        // regardless of task completion order.
        let call_order: Vec<&str> = tool_calls.iter().map(|tc| tc.id.as_str()).collect();
        results.sort_by_key(|outcome| {
            call_order
                .iter()
                .position(|&cid| cid == outcome.call_id)
                .unwrap_or(usize::MAX)
        });

        // Now fire sink hooks and append history in deterministic
        // order, threading `call_id` so the TUI / harness sinks can
        // stamp the right activity entry even with concurrent
        // `file_read` calls in flight (issues #6 / #7).
        for outcome in results {
            let DispatchOutcome {
                call_id,
                name,
                payload,
                ..
            } = outcome;
            let Some((tool_result, compressed)) = payload else {
                continue;
            };
            sink.on_tool_result(&call_id, &name, &tool_result.output, tool_result.is_error)?;
            if compressed.compressed_via_signatures {
                sink.on_tool_result_compressed(&call_id, &name, compressed.symbol_count)?;
            }
            self.history
                .push(Message::tool_result(call_id, compressed.text));
        }

        Ok(())
    }
}

/// Outcome of one parallel tool dispatch task.
///
/// Returned to [`AgentLoop::dispatch_tool_calls`] from each spawned
/// `JoinSet` task and then sorted into call-order before sink hooks
/// fire so concurrent calls never cross-stamp activity entries.
struct DispatchOutcome {
    /// The unique LLM-issued call identifier; threaded through every
    /// sink hook so concurrent calls of the same `name` can be paired.
    call_id: String,
    /// The tool name (e.g. `"file_read"`).
    name: String,
    /// `true` when the task observed cancellation before producing a
    /// result. The dispatcher converts this to [`CoreError::Cancelled`]
    /// at collection time.
    cancelled: bool,
    /// `(tool_result, compressed)` when the task produced one;
    /// `None` only on the cancelled branch.
    payload: Option<(tmg_tools::ToolResult, crate::context::CompressedToolResult)>,
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
    fn insert_bootstrap_system_message_role_and_content() {
        // Build a Message directly (not via AgentLoop::new which requires
        // an LLM client) and verify the role/content roundtrip used by
        // `insert_bootstrap_system_message`.
        let injected = Message::system("BOOTSTRAP: hello");
        assert_eq!(injected.role(), tmg_llm::Role::System);
        assert_eq!(injected.content(), "BOOTSTRAP: hello");

        let chat = injected.to_chat_message();
        assert_eq!(chat.role, tmg_llm::Role::System);
        assert_eq!(chat.content.as_deref(), Some("BOOTSTRAP: hello"));
    }

    /// `insert_bootstrap_system_message` must place the bootstrap
    /// directly after the initial system prompt (index 0), even when
    /// prompt-file user-role messages have already been loaded into
    /// history. This is required by chat templates (Mistral / Qwen /
    /// Gemma) that reject `system` messages following a `user` turn.
    ///
    /// We exercise the position logic directly on a `Vec<Message>`
    /// instead of constructing a full `AgentLoop` (which needs an LLM
    /// client); the implementation under test only manipulates this
    /// vector.
    #[test]
    fn insert_bootstrap_system_message_inserted_after_history_zero() {
        let mut history = vec![
            Message::system("BASE"),
            Message::user("prompt-file content from TSUMUGI.md"),
        ];

        // Mirror the production logic.
        let position = usize::from(!history.is_empty());
        history.insert(position, Message::system("BOOTSTRAP-1"));

        assert_eq!(history.len(), 3);
        assert_eq!(history[0].content(), "BASE");
        assert_eq!(history[1].role(), tmg_llm::Role::System);
        assert_eq!(history[1].content(), "BOOTSTRAP-1");
        assert_eq!(history[2].role(), tmg_llm::Role::User);

        // A second bootstrap injection lands at the same slot, pushing
        // the previous bootstrap one slot deeper but staying entirely
        // ahead of the first user-role message.
        let position = usize::from(!history.is_empty());
        history.insert(position, Message::system("BOOTSTRAP-2"));

        assert_eq!(history.len(), 4);
        assert_eq!(history[0].content(), "BASE");
        assert_eq!(history[1].content(), "BOOTSTRAP-2");
        assert_eq!(history[2].content(), "BOOTSTRAP-1");
        assert_eq!(history[3].role(), tmg_llm::Role::User);
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
