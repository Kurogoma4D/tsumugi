//! Application state and business logic for the TUI.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use tmg_agents::{AgentType, CustomAgentDef, SubagentManager, truncate_str};
use tmg_core::{AgentLoop, CoreError, StreamSink, format_context_usage};
use tmg_harness::{HarnessStreamSink, RunProgressEvent, RunRunner, RunSummary};
use tmg_llm::Role;
use tmg_memory::MemoryStore;
use tmg_search::SearchIndex;
use tmg_skills::{SkillMeta, SlashCommand};
use tmg_workflow::{WorkflowMeta, WorkflowProgress};
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::activity::{
    ActivityPane, RunHeader, RunProgressSection, ToolActivityEntry as ActivityEntry,
    WorkflowProgressSection,
};
use crate::banner::TransientBanner;
use crate::diff::DiffPreview;
use crate::human_prompt::HumanPrompt;

/// Outcome of [`App::submit_input`].
#[derive(Debug)]
pub enum SubmitDecision {
    /// The input was empty (or `/`); nothing to do.
    Nothing,
    /// Start a regular conversation turn with the carried text.
    StartTurn(String),
    /// Dispatch the carried slash command via [`App::dispatch_slash`]
    /// or the event loop's async dispatcher.
    SlashCommand(SlashCommand),
    /// The slash parser surfaced a structured error; surface it to the
    /// user via the standard error overlay.
    ParseError(String),
}

/// Represents a single entry in the chat display.
#[derive(Debug, Clone)]
pub struct ChatEntry {
    /// The author role of this entry.
    pub role: Role,
    /// The text content.
    pub text: String,
}

/// A single log entry in the tool activity pane.
///
/// This is a re-export of [`crate::activity::ToolActivityEntry`] kept
/// here so existing callers (and the public `App::tool_activity`
/// accessor) keep their import path stable across the #45 refactor.
pub type ToolActivityEntry = ActivityEntry;

/// Messages sent from the background turn task to the event loop.
pub enum TurnMessage {
    /// An incremental thinking/reasoning token from the LLM stream.
    Thinking(String),
    /// An incremental token from the LLM stream.
    Token(String),
    /// A tool call is being dispatched.
    ToolCall {
        /// The unique LLM-issued tool-call identifier. Pair with the
        /// matching `ToolResult` / `ToolResultCompressed` to find the
        /// right activity entry even with concurrent same-name calls.
        call_id: String,
        /// The tool name.
        name: String,
        /// The tool call arguments (JSON string).
        arguments: String,
    },
    /// A tool call completed with a result.
    ToolResult {
        /// The unique LLM-issued tool-call identifier (matches the
        /// preceding `ToolCall`).
        call_id: String,
        /// The tool name.
        name: String,
        /// The tool output (may be truncated for display).
        output: String,
        /// Whether the tool reported an error.
        is_error: bool,
    },
    /// A tool result was rewritten via tree-sitter signature extraction
    /// before being recorded in the conversation history (issue #49).
    ///
    /// This message is sent **in addition to** [`Self::ToolResult`] —
    /// the activity pane receives the raw output via `ToolResult` for
    /// display, then this message stamps a "compressed" marker on the
    /// most recent matching entry so the TUI can show the
    /// `[compressed via tree-sitter: N symbols]` hint.
    ToolResultCompressed {
        /// The unique LLM-issued tool-call identifier (matches the
        /// `ToolCall` / `ToolResult` for the same call).
        call_id: String,
        /// The tool name (matches the preceding `ToolResult`).
        name: String,
        /// Number of structural symbols tree-sitter extracted.
        symbol_count: usize,
    },
    /// The turn completed successfully. Contains the `AgentLoop` back
    /// and context usage information.
    Done {
        /// The agent loop returned after the turn completes.
        agent: AgentLoop,
        /// Current token count in the context.
        token_count: usize,
        /// Maximum context tokens configured.
        max_tokens: usize,
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
    /// The `/compact` background task completed successfully.
    CompactDone {
        /// The agent loop returned after compaction.
        agent: AgentLoop,
        /// Current token count after compression.
        token_count: usize,
        /// Maximum context tokens configured.
        max_tokens: usize,
    },
    /// The `/compact` background task failed.
    CompactError {
        /// The agent loop returned after compaction failure.
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
#[expect(
    clippy::struct_excessive_bools,
    reason = "TUI state flags are independent and rarely change together"
)]
pub struct App {
    /// The agent loop driving LLM conversations.
    ///
    /// This is `None` while a turn is running in a background task
    /// (the task temporarily takes ownership).
    agent: Option<AgentLoop>,

    /// Chat entries displayed in the chat pane.
    chat_entries: Vec<ChatEntry>,

    /// Structured Activity Pane state (issue #45). Replaces the
    /// previous flat `tool_activity` field.
    activity: ActivityPane,

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

    /// Shared reference to the subagent manager for status display.
    subagent_manager: Option<Arc<Mutex<SubagentManager>>>,

    /// Custom agent definitions for display in `/agents` list.
    custom_agents: Vec<CustomAgentDef>,

    /// Whether a `/compact` command is pending execution.
    pending_compact: bool,

    /// Whether the model is currently in the thinking/reasoning phase.
    thinking: bool,

    /// Optional path for structured event logging (JSON Lines).
    event_log_path: Option<PathBuf>,

    /// The currently active run summary (kept for backward compatible
    /// accessor `current_run()`). The header now also derives from
    /// [`Self::run_header`].
    current_run: Option<RunSummary>,

    /// Pre-formatted run header for the top bar (issue #45 SPEC §7.1).
    /// `None` when no run is attached.
    run_header: Option<RunHeader>,

    /// Optional [`RunRunner`] handle used to wrap each turn's sink with
    /// [`HarnessStreamSink`], so the active session's `tool_calls_count`
    /// and `files_modified` reflect actual tool activity.
    runner: Option<Arc<Mutex<RunRunner>>>,

    /// Optional receiver of [`RunProgressEvent`]s from
    /// [`RunRunner::progress_channel`]. Set by the CLI startup path
    /// after subscribing to the runner.
    run_progress_rx: Option<mpsc::Receiver<RunProgressEvent>>,

    /// Optional receiver of [`WorkflowProgress`] events from a running
    /// workflow. Wired by the foreground `run_workflow` path so the
    /// engine's events reach the activity pane.
    workflow_progress_rx: Option<mpsc::Receiver<WorkflowProgress>>,

    /// Whether the most recent `shell_exec` tool *call* was a `git diff`
    /// invocation. Set on `ToolCall { name == "shell_exec" }` and read
    /// in the matching `ToolResult` so we only run unified-diff
    /// extraction on outputs that are actually likely to be diffs —
    /// stray `git log -p` / `cat foo.diff` blobs no longer overwrite a
    /// legitimate `file_write` / `file_patch` preview.
    last_shell_exec_was_git_diff: bool,

    /// Optional transient banner overlaid on the chat pane (SPEC §7.1
    /// scope-upgrade notification). Cleared by the event loop once
    /// `is_expired()` returns `true`.
    transient_banner: Option<TransientBanner>,

    /// Optional in-flight `human` step prompt (SPEC §8.4). When `Some`,
    /// the chat pane is overlaid with the modal and key events are
    /// consumed by [`crate::event::handle_human_prompt_key`] until the
    /// prompt is dismissed.
    human_prompt: Option<HumanPrompt>,

    /// Discovered workflows for `/workflows` listing. Loaded once at
    /// startup; the TUI does not refresh this list at runtime.
    workflows: Vec<WorkflowMeta>,

    /// Discovered skills for `/skills` listing. Loaded once at startup
    /// from the same discovery pass that registered `use_skill` tool
    /// metadata; the TUI does not refresh this list at runtime.
    skills: Vec<SkillMeta>,

    /// Pending slash command awaiting async dispatch. Populated by
    /// [`Self::dispatch_slash`] (`SlashDispatch::Async`) and
    /// drained by the event loop on the next tick.
    pending_slash: Option<SlashCommand>,

    /// Optional memory store for `/memory` slash commands. Constructed
    /// once in `tmg-cli::run_tui` and shared with the `MemoryTool` on
    /// the registry so the TUI display path and the LLM's tool calls
    /// see exactly the same on-disk state.
    memory_store: Option<Arc<MemoryStore>>,

    /// Optional cross-session search index for the `/search` slash
    /// command (issue #53). Shared with [`tmg_search::SessionSearchTool`]
    /// on the registry so the TUI display path and the LLM tool see the
    /// same on-disk state.
    search_index: Option<Arc<SearchIndex>>,

    /// Whether `/skills disable-auto` was issued during this session
    /// (issue #54). The harness inspects this flag through
    /// [`Self::skill_auto_disabled`] before invoking `skill_critic`.
    skill_auto_disabled: bool,

    /// Set when `/skill capture` is requested. The harness flips it
    /// back to `false` once the manual signal is consumed. Tracked
    /// here so the request survives a brief gap between dispatch and
    /// the next agent-loop turn.
    skill_capture_requested: bool,

    /// Names of skills awaiting rejection via `/skills reject <name>`
    /// (issue #54). Drained by the harness, which deletes each skill
    /// directory and writes a feedback memory entry.
    skill_rejection_requests: Vec<String>,
}

impl App {
    /// Create a new `App` with the given agent loop and model name.
    pub fn new(
        agent: AgentLoop,
        model_name: &str,
        project_root: PathBuf,
        cwd: PathBuf,
        event_log: Option<PathBuf>,
    ) -> Self {
        let max_tokens = agent.context_config().max_context_tokens;
        let context_usage = format_context_usage(0, max_tokens);
        Self {
            agent: Some(agent),
            chat_entries: Vec::new(),
            activity: ActivityPane::new(),
            input: String::new(),
            cursor_pos: 0,
            chat_scroll: 0,
            should_exit: false,
            streaming: false,
            model_name: model_name.to_owned(),
            context_usage,
            project_root,
            cwd,
            error_message: None,
            turn_handle: None,
            subagent_manager: None,
            custom_agents: Vec::new(),
            pending_compact: false,
            thinking: false,
            event_log_path: event_log,
            current_run: None,
            run_header: None,
            runner: None,
            run_progress_rx: None,
            workflow_progress_rx: None,
            last_shell_exec_was_git_diff: false,
            transient_banner: None,
            human_prompt: None,
            workflows: Vec::new(),
            skills: Vec::new(),
            pending_slash: None,
            memory_store: None,
            search_index: None,
            skill_auto_disabled: false,
            skill_capture_requested: false,
            skill_rejection_requests: Vec::new(),
        }
    }

    /// Whether `/skills disable-auto` was issued during this session.
    ///
    /// The harness consults this flag before spawning `skill_critic`;
    /// when `true`, autonomous skill creation is suppressed for the
    /// remainder of the session.
    #[must_use]
    pub fn skill_auto_disabled(&self) -> bool {
        self.skill_auto_disabled
    }

    /// Take the pending `/skill capture` request, if any. Returns
    /// `true` exactly once per dispatch.
    #[must_use = "the consumed capture request must be acted on or it is lost"]
    pub fn take_skill_capture_request(&mut self) -> bool {
        std::mem::replace(&mut self.skill_capture_requested, false)
    }

    /// Drain pending `/skills reject <name>` requests. The harness is
    /// expected to remove each skill directory and write a feedback
    /// memory entry.
    #[must_use = "the drained rejection list must be processed or the user request is lost"]
    pub fn drain_skill_rejection_requests(&mut self) -> Vec<String> {
        std::mem::take(&mut self.skill_rejection_requests)
    }

    /// Install the shared memory store so the `/memory` slash commands
    /// can render index / entries without round-tripping through the
    /// LLM. When unset (or `None`), `/memory` reports memory disabled.
    pub fn set_memory_store(&mut self, store: Arc<MemoryStore>) {
        self.memory_store = Some(store);
    }

    /// Install the shared search index so `/search <query>` can render
    /// results inline. When unset, `/search` reports search disabled.
    pub fn set_search_index(&mut self, index: Arc<SearchIndex>) {
        self.search_index = Some(index);
    }

    /// Dequeue any pending async slash command so the event loop can
    /// dispatch it. Returns `None` when nothing is queued.
    pub fn take_pending_slash(&mut self) -> Option<SlashCommand> {
        self.pending_slash.take()
    }

    /// Enqueue a slash command for async dispatch. Internal helper so
    /// [`Self::dispatch_slash`] can defer commands without
    /// owning the runtime.
    fn enqueue_async_slash(&mut self, cmd: SlashCommand) {
        self.pending_slash = Some(cmd);
    }

    /// Replace the discovered workflows list (used by `/workflows`).
    pub fn set_workflows(&mut self, workflows: Vec<WorkflowMeta>) {
        self.workflows = workflows;
    }

    /// Borrow the discovered workflows.
    #[must_use]
    pub fn workflows(&self) -> &[WorkflowMeta] {
        &self.workflows
    }

    /// Replace the discovered skills list (used by `/skills`).
    pub fn set_skills(&mut self, skills: Vec<SkillMeta>) {
        self.skills = skills;
    }

    /// Borrow the discovered skills.
    #[must_use]
    pub fn skills(&self) -> &[SkillMeta] {
        &self.skills
    }

    /// Borrow the active transient banner, if any.
    #[must_use]
    pub fn transient_banner(&self) -> Option<&TransientBanner> {
        self.transient_banner.as_ref()
    }

    /// Set or replace the transient banner.
    pub fn set_transient_banner(&mut self, banner: TransientBanner) {
        self.transient_banner = Some(banner);
    }

    /// Clear the transient banner if it has expired. Returns `true`
    /// when a banner was cleared (so the event loop can request a
    /// redraw).
    pub fn expire_banner_if_due(&mut self) -> bool {
        if self
            .transient_banner
            .as_ref()
            .is_some_and(TransientBanner::is_expired)
        {
            self.transient_banner = None;
            return true;
        }
        false
    }

    /// Borrow the active human prompt, if any.
    #[must_use]
    pub fn human_prompt(&self) -> Option<&HumanPrompt> {
        self.human_prompt.as_ref()
    }

    /// Borrow the active human prompt mutably.
    pub fn human_prompt_mut(&mut self) -> Option<&mut HumanPrompt> {
        self.human_prompt.as_mut()
    }

    /// Whether a human prompt modal is currently active.
    #[must_use]
    pub fn has_human_prompt(&self) -> bool {
        self.human_prompt.is_some()
    }

    /// Take the active human prompt, leaving `None` in its place.
    pub fn take_human_prompt(&mut self) -> Option<HumanPrompt> {
        self.human_prompt.take()
    }

    /// Re-stash a human prompt into [`Self`].
    ///
    /// Used by the event handler when it needs to surface a soft error
    /// (e.g. a `Revise` selection without a `revise_target`) without
    /// dismissing the modal — the user may still pick `Approve` or
    /// `Reject`. Any existing prompt is overwritten; callers should
    /// only invoke this immediately after [`Self::take_human_prompt`]
    /// to avoid clobbering a fresh prompt that arrived in the meantime.
    pub fn set_human_prompt(&mut self, prompt: HumanPrompt) {
        self.human_prompt = Some(prompt);
    }

    /// Attach a workflow progress receiver.
    ///
    /// Intended use: the CLI / `run_workflow` foreground path forks
    /// the engine's progress channel and hands the receiver here so
    /// the activity pane can drain `WorkflowProgress` events.
    pub fn set_workflow_progress_rx(&mut self, rx: mpsc::Receiver<WorkflowProgress>) {
        self.workflow_progress_rx = Some(rx);
        self.activity.workflow_progress = None;
    }

    /// Attach a run progress receiver.
    ///
    /// The CLI calls this once with the receiver from
    /// [`tmg_harness::RunRunner::progress_channel`] so scope upgrades
    /// and session-end events flow into [`Self::run_header`] and the
    /// activity pane.
    pub fn set_run_progress_rx(&mut self, rx: mpsc::Receiver<RunProgressEvent>) {
        self.run_progress_rx = Some(rx);
    }

    /// Set the active run for header display.
    pub fn set_current_run(&mut self, run: RunSummary) {
        self.run_header = Some(RunHeader::from_summary(&run));
        self.activity.run_progress.apply_summary(&run);
        self.current_run = Some(run);
    }

    /// Return a snapshot of the current run header (`None` if no run
    /// is attached).
    #[must_use]
    pub fn run_header(&self) -> Option<&RunHeader> {
        self.run_header.as_ref()
    }

    /// Return a reference to the structured Activity Pane state.
    #[must_use]
    pub fn activity(&self) -> &ActivityPane {
        &self.activity
    }

    /// Borrow the run-progress section.
    #[must_use]
    pub fn run_progress(&self) -> &RunProgressSection {
        &self.activity.run_progress
    }

    /// Borrow the workflow-progress section, if any.
    #[must_use]
    pub fn workflow_progress(&self) -> Option<&WorkflowProgressSection> {
        self.activity.workflow_progress.as_ref()
    }

    /// Borrow the most recent diff preview, if any.
    #[must_use]
    pub fn diff_preview(&self) -> Option<&DiffPreview> {
        self.activity.diff_preview.as_ref()
    }

    /// Set the shared [`RunRunner`] handle used to wrap turn sinks with
    /// [`HarnessStreamSink`].
    pub fn set_runner(&mut self, runner: Arc<Mutex<RunRunner>>) {
        self.runner = Some(runner);
    }

    /// Borrow the shared [`RunRunner`] handle, if any. The async slash
    /// dispatcher uses this to drive `tmg_harness::commands` calls.
    #[must_use]
    pub fn runner(&self) -> Option<&Arc<Mutex<RunRunner>>> {
        self.runner.as_ref()
    }

    /// Return the active run, if any.
    pub fn current_run(&self) -> Option<&RunSummary> {
        self.current_run.as_ref()
    }

    /// Set the subagent manager for status display.
    pub fn set_subagent_manager(&mut self, manager: Arc<Mutex<SubagentManager>>) {
        self.subagent_manager = Some(manager);
    }

    /// Set the custom agent definitions for display in `/agents` list.
    pub fn set_custom_agents(&mut self, agents: Vec<CustomAgentDef>) {
        self.custom_agents = agents;
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

    /// Whether the model is currently in the thinking/reasoning phase.
    pub fn is_thinking(&self) -> bool {
        self.thinking
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

    /// Return a reference to the tool activity log.
    pub fn tool_activity(&self) -> &[ToolActivityEntry] {
        &self.activity.tool_log
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

    /// Return the cached subagent summaries for display.
    pub fn subagent_summaries(&self) -> &[tmg_agents::SubagentSummary] {
        &self.activity.subagents
    }

    /// Return the count of running subagents from cached summaries.
    pub fn running_subagent_count(&self) -> usize {
        self.activity
            .subagents
            .iter()
            .filter(|s| !s.status.is_terminal())
            .count()
    }

    /// Refresh the cached subagent summaries from the manager.
    ///
    /// This is called periodically from the event loop to update
    /// the TUI display.
    pub async fn refresh_subagent_summaries(&mut self) {
        if let Some(manager) = &self.subagent_manager {
            let manager = manager.lock().await;
            self.activity.subagents = manager.summaries().await;
        }
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

    /// Submit the current input. Returns one of:
    /// * [`SubmitDecision::StartTurn(text)`] — a regular chat message
    ///   that the event loop should drive through `start_turn`.
    /// * [`SubmitDecision::SlashCommand(cmd)`] — a parsed slash command
    ///   that the event loop should dispatch via the async dispatcher
    ///   (the event loop's `dispatch_slash_async`). The TUI keeps slash
    ///   dispatch out of `App` so the async harness/runner calls can
    ///   live alongside the rest of the event-loop async wiring.
    /// * [`SubmitDecision::Nothing`] — the input was empty.
    /// * [`SubmitDecision::ParseError(msg)`] — the slash parser
    ///   rejected the input. The event loop surfaces the message via
    ///   the standard error overlay.
    pub fn submit_input(&mut self) -> SubmitDecision {
        let text = self.input.trim().to_owned();
        self.input.clear();
        self.cursor_pos = 0;

        if text.is_empty() {
            return SubmitDecision::Nothing;
        }

        // Slash commands: parse here; dispatch in the event loop.
        if text.starts_with('/') {
            match tmg_skills::parse_slash_command(&text) {
                Ok(Some(cmd)) => return SubmitDecision::SlashCommand(cmd),
                Ok(None) => {
                    // A bare `/` falls through as ordinary chat input.
                }
                Err(e) => {
                    return SubmitDecision::ParseError(e.to_string());
                }
            }
        }

        // Add user entry to chat.
        self.chat_entries.push(ChatEntry {
            role: Role::User,
            text: text.clone(),
        });

        SubmitDecision::StartTurn(text)
    }

    /// Whether the `/compact` command needs to run asynchronously.
    ///
    /// Set to `true` when the user types `/compact` and cleared after
    /// the background task picks it up.
    pub fn needs_compact(&self) -> bool {
        self.pending_compact
    }

    /// Clear the pending compact flag.
    pub fn clear_pending_compact(&mut self) {
        self.pending_compact = false;
    }

    /// Start context compression in a background task.
    ///
    /// Takes ownership of the agent for the duration, similar to
    /// [`start_turn`]. Results are communicated via `TurnMessage::CompactDone`
    /// or `TurnMessage::CompactError`.
    pub fn start_compact(&mut self, parent_cancel: &CancellationToken) {
        let Some(mut agent) = self.agent.take() else {
            return;
        };

        let turn_cancel = parent_cancel.child_token();
        agent.set_cancel_token(turn_cancel.clone());

        self.streaming = true;

        let (tx, rx) = mpsc::channel::<TurnMessage>(16);

        let join = tokio::spawn(async move {
            let result = agent.compact().await;

            match result {
                Ok(()) => {
                    let token_count = agent.token_count();
                    let max_tokens = agent.context_config().max_context_tokens;
                    let _ = tx
                        .send(TurnMessage::CompactDone {
                            agent,
                            token_count,
                            max_tokens,
                        })
                        .await;
                }
                Err(CoreError::Cancelled) => {
                    let _ = tx
                        .send(TurnMessage::CompactError {
                            agent,
                            message: "Context compression cancelled.".to_owned(),
                            cancelled: true,
                        })
                        .await;
                }
                Err(e) => {
                    let _ = tx
                        .send(TurnMessage::CompactError {
                            agent,
                            message: format!("Context compression failed: {e}"),
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

    /// Dispatch a parsed slash command, handling synchronous variants
    /// in-place and queuing async ones via [`Self::take_pending_slash`].
    ///
    /// # Errors
    ///
    /// Returns a [`CoreError`] when `/clear` fails to reload the prompt
    /// files. Other failure paths surface via `error_message`.
    pub fn dispatch_slash(&mut self, cmd: SlashCommand) -> Result<(), CoreError> {
        match cmd {
            SlashCommand::Exit => {
                self.should_exit = true;
            }
            SlashCommand::Clear => {
                self.chat_entries.clear();
                self.activity.tool_log.clear();
                self.activity.diff_preview = None;
                self.chat_scroll = 0;
                if let Some(agent) = &mut self.agent {
                    agent.clear_history(&self.project_root, &self.cwd)?;
                }
            }
            SlashCommand::Compact => {
                if self.agent.is_some() {
                    self.pending_compact = true;
                    self.chat_entries.push(ChatEntry {
                        role: Role::System,
                        text: "Compressing context...".to_owned(),
                    });
                } else {
                    self.error_message = Some("Cannot compact while a turn is running".to_owned());
                }
            }
            SlashCommand::ListAgents => {
                self.show_agents_list();
            }
            SlashCommand::ListWorkflows => {
                self.show_workflows_list();
            }
            SlashCommand::MemoryIndex => {
                self.show_memory_index();
            }
            SlashCommand::MemoryShow { name } => {
                self.show_memory_entry(&name);
            }
            SlashCommand::Search { query } => {
                self.show_search_results(&query);
            }
            // Skill capture / reject / disable-auto (issue #54). The
            // actual signal-collector and skill_critic spawn happen on
            // the harness side; the TUI just records the user intent
            // here as a system chat entry so the next agent turn can
            // observe it. When the harness wiring lands the body of
            // these branches will dispatch to the SignalCollector.
            SlashCommand::SkillCapture => {
                self.skill_capture_requested = true;
                self.chat_entries.push(ChatEntry {
                    role: Role::System,
                    text: "Skill capture requested for the current turn range. The next \
                           agent loop will run the skill_critic over the recent activity."
                        .to_owned(),
                });
            }
            SlashCommand::SkillsDisableAuto => {
                self.skill_auto_disabled = true;
                self.chat_entries.push(ChatEntry {
                    role: Role::System,
                    text: "Autonomous skill creation is disabled for this session. Use \
                           /skill capture to opt back in for a single turn."
                        .to_owned(),
                });
            }
            SlashCommand::SkillReject { name } => {
                self.skill_rejection_requests.push(name.as_str().to_owned());
                self.chat_entries.push(ChatEntry {
                    role: Role::System,
                    text: format!(
                        "Queued rejection for skill {:?}; the harness will delete the \
                         skill directory and record the reason in feedback memory.",
                        name.as_str()
                    ),
                });
            }
            // The /run family and /<skill> need async work (harness
            // mutex, workflow tools). Defer to the event loop.
            other @ (SlashCommand::RunStart { .. }
            | SlashCommand::RunResume { .. }
            | SlashCommand::RunList
            | SlashCommand::RunStatus { .. }
            | SlashCommand::RunUpgrade
            | SlashCommand::RunDowngrade
            | SlashCommand::RunNewSession
            | SlashCommand::RunPause
            | SlashCommand::RunAbort
            | SlashCommand::ListSkills
            | SlashCommand::InvokeSkill { .. }) => {
                self.enqueue_async_slash(other);
            }
            // `SlashCommand` is `#[non_exhaustive]`. Future variants
            // surface as a friendly error overlay so the user knows
            // the command was parsed but not (yet) supported here.
            other => {
                self.error_message = Some(format!("Unsupported slash command: {other:?}"));
            }
        }
        Ok(())
    }

    /// Display the list of available subagent types and running subagents.
    fn show_agents_list(&mut self) {
        let mut text = String::from("Built-in subagent types:\n");
        for agent_type in AgentType::ALL {
            let _ = writeln!(
                text,
                "  - {}: {}",
                agent_type.name(),
                agent_type.description()
            );
        }

        if !self.custom_agents.is_empty() {
            text.push_str("\nCustom subagents:\n");
            for agent in &self.custom_agents {
                let tools_summary = agent.allowed_tools().join(", ");
                let _ = writeln!(
                    text,
                    "  - {}: {} [tools: {}]",
                    agent.name(),
                    agent.description(),
                    tools_summary
                );
            }
        }

        if !self.activity.subagents.is_empty() {
            text.push_str("\nActive subagents:\n");
            for summary in &self.activity.subagents {
                let task_preview = if summary.task.chars().count() > 60 {
                    format!("{}...", truncate_str(&summary.task, 57))
                } else {
                    summary.task.clone()
                };
                let _ = writeln!(
                    text,
                    "  [{}] {} ({}): {}",
                    summary.id,
                    summary.agent_name,
                    summary.status.label(),
                    task_preview,
                );
            }
        }

        self.chat_entries.push(ChatEntry {
            role: Role::System,
            text,
        });
    }

    /// Display the list of discovered workflows (`/workflows`).
    fn show_workflows_list(&mut self) {
        let mut text = String::from("Available workflows:\n");
        if self.workflows.is_empty() {
            text.push_str("  (none discovered)\n");
            text.push_str(
                "\nDrop a *.yaml file under `.tsumugi/workflows/` and restart the TUI.\n",
            );
        } else {
            for meta in &self.workflows {
                match &meta.description {
                    Some(desc) if !desc.is_empty() => {
                        let _ = writeln!(text, "  - {} — {}", meta.id, desc);
                    }
                    _ => {
                        let _ = writeln!(text, "  - {}", meta.id);
                    }
                }
            }
            text.push_str("\nStart with `/run start <workflow>`.\n");
        }
        self.chat_entries.push(ChatEntry {
            role: Role::System,
            text,
        });
    }

    /// Render the merged memory index in the chat pane.
    fn show_memory_index(&mut self) {
        let Some(store) = self.memory_store.as_ref() else {
            self.error_message =
                Some("memory is disabled; enable [memory] in tsumugi.toml".to_owned());
            return;
        };
        match store.read_merged_index() {
            Ok(index) if !index.trim().is_empty() => {
                let mut text = String::from("Memory index:\n\n");
                text.push_str(&index);
                self.chat_entries.push(ChatEntry {
                    role: Role::System,
                    text,
                });
            }
            Ok(_) => {
                self.chat_entries.push(ChatEntry {
                    role: Role::System,
                    text: "(memory is empty — use the `memory` tool's `add` action or \
                          `tmg memory edit <name>`)"
                        .to_owned(),
                });
            }
            Err(e) => {
                self.error_message = Some(format!("failed to read memory index: {e}"));
            }
        }
    }

    /// Render one memory entry in the chat pane.
    fn show_memory_entry(&mut self, name: &str) {
        let Some(store) = self.memory_store.as_ref() else {
            self.error_message =
                Some("memory is disabled; enable [memory] in tsumugi.toml".to_owned());
            return;
        };
        match store.read(name) {
            Ok((entry, scope)) => {
                let mut text = format!(
                    "[{}] {} ({}) — {}\n\n",
                    scope.as_str(),
                    entry.frontmatter.name,
                    entry.frontmatter.kind,
                    entry.frontmatter.description,
                );
                text.push_str(&entry.body);
                self.chat_entries.push(ChatEntry {
                    role: Role::System,
                    text,
                });
            }
            Err(e) => {
                self.error_message = Some(format!("memory: {e}"));
            }
        }
    }

    /// Render search results from the cross-session index in the chat
    /// pane. Issue #53 — `/search <query>` slash command.
    fn show_search_results(&mut self, query: &str) {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            self.chat_entries.push(ChatEntry {
                role: Role::System,
                text: "Usage: /search <FTS5 query>".to_owned(),
            });
            return;
        }
        let Some(index) = self.search_index.as_ref() else {
            self.error_message =
                Some("search is disabled; enable [search] in tsumugi.toml".to_owned());
            return;
        };
        match index.query(trimmed, 10, tmg_search::SearchScope::All, None) {
            Ok(hits) if hits.is_empty() => {
                self.chat_entries.push(ChatEntry {
                    role: Role::System,
                    text: format!("(no matches for {trimmed:?})"),
                });
            }
            Ok(hits) => {
                let mut text = format!("Search results for {trimmed:?}:\n\n");
                for hit in &hits {
                    let _ = writeln!(
                        text,
                        "[{:.3}] {}#{}  {}\n  summary: {}\n  snippet: {}\n",
                        hit.score,
                        hit.run_id,
                        hit.session_num,
                        hit.started_at,
                        hit.summary,
                        hit.snippet,
                    );
                }
                self.chat_entries.push(ChatEntry {
                    role: Role::System,
                    text,
                });
            }
            Err(e) => {
                self.error_message = Some(format!("search: {e}"));
            }
        }
    }

    /// Append a system message describing slash-command output (e.g. a
    /// run summary). Public so the event loop's async dispatcher can
    /// surface harness-command results without re-implementing the
    /// chat-entry shape.
    pub fn push_slash_output(&mut self, text: impl Into<String>) {
        self.chat_entries.push(ChatEntry {
            role: Role::System,
            text: text.into(),
        });
    }

    /// Start a conversation turn in a background task.
    ///
    /// The turn runs on a spawned tokio task. Streaming tokens and tool
    /// activity are sent through an `mpsc` channel that the event loop
    /// drains on each tick. A child `CancellationToken` is used so that
    /// cancelling a turn does not shut down the entire application.
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

        // Open the event log writer if configured (append mode so all
        // turns accumulate in a single file).
        let event_log_writer = self
            .event_log_path
            .as_deref()
            .and_then(|path| tmg_core::EventLogWriter::open_append(path).ok());

        let runner = self.runner.clone();

        let join = tokio::spawn(async move {
            let channel_sink = ChannelStreamSink { tx: tx.clone() };
            // The chain of sinks below is built bottom-up: the channel
            // sink (and optional event log) sit at the bottom, and
            // `HarnessStreamSink` wraps them so harness session-stat
            // updates fire on every event without changing the in-flight
            // forwarding to the TUI.
            let result = match (event_log_writer, runner) {
                (Some(log), Some(runner)) => {
                    let tee = tmg_core::TeeStreamSink::new(channel_sink, log);
                    let mut wrapped = HarnessStreamSink::new(tee, runner);
                    agent.turn(&user_input, &mut wrapped).await
                }
                (Some(log), None) => {
                    let mut tee = tmg_core::TeeStreamSink::new(channel_sink, log);
                    agent.turn(&user_input, &mut tee).await
                }
                (None, Some(runner)) => {
                    let mut wrapped = HarnessStreamSink::new(channel_sink, runner);
                    agent.turn(&user_input, &mut wrapped).await
                }
                (None, None) => {
                    let mut sink = channel_sink;
                    agent.turn(&user_input, &mut sink).await
                }
            };

            match result {
                Ok(()) => {
                    let token_count = agent.token_count();
                    let max_tokens = agent.context_config().max_context_tokens;
                    // Ignore send errors -- the receiver may have been
                    // dropped if the app is shutting down.
                    let _ = tx
                        .send(TurnMessage::Done {
                            agent,
                            token_count,
                            max_tokens,
                        })
                        .await;
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
    #[expect(
        clippy::too_many_lines,
        reason = "message dispatch loop handles all TurnMessage variants; splitting would obscure control flow"
    )]
    pub fn drain_turn_messages(&mut self) -> bool {
        let Some(handle) = &mut self.turn_handle else {
            return false;
        };

        let mut changed = false;

        loop {
            match handle.rx.try_recv() {
                Ok(TurnMessage::Thinking(_)) => {
                    self.thinking = true;
                    changed = true;
                }
                Ok(TurnMessage::Token(token)) => {
                    if self.thinking {
                        self.thinking = false;
                    }
                    if let Some(last) = self.chat_entries.last_mut() {
                        last.text.push_str(&token);
                    }
                    changed = true;
                }
                Ok(TurnMessage::ToolCall {
                    call_id,
                    name,
                    arguments,
                }) => {
                    // Capture diff preview source from `file_write` /
                    // `file_patch` arguments. We parse the JSON
                    // best-effort: a malformed tool call still gets a
                    // log entry, just no preview update.
                    capture_diff_from_call(&mut self.activity.diff_preview, &name, &arguments);

                    // Latch whether this `shell_exec` is a `git diff`
                    // invocation. The matching `ToolResult` consults
                    // the flag so we only extract a unified diff out
                    // of outputs that are actually likely to contain
                    // one — `git log -p`, `cat diff.txt`, and other
                    // stray diff-looking blobs no longer overwrite a
                    // legitimate `file_write` / `file_patch` preview.
                    if name == "shell_exec" {
                        self.last_shell_exec_was_git_diff = shell_exec_is_git_diff(&arguments);
                    }

                    let summary = truncate_for_display(&arguments, 120);
                    self.activity.tool_log.push(ToolActivityEntry {
                        call_id,
                        tool_name: name,
                        summary: format!("calling: {summary}"),
                        is_error: false,
                        compressed: false,
                        compressed_symbol_count: 0,
                    });
                    changed = true;
                }
                Ok(TurnMessage::ToolResult {
                    call_id,
                    name,
                    output,
                    is_error,
                }) => {
                    // For `shell_exec` results whose original command
                    // was `git diff` (latched on the matching
                    // `ToolCall`), try to extract the unified diff.
                    // Failure is silent — the tool log entry still
                    // lands.
                    if !is_error && name == "shell_exec" && self.last_shell_exec_was_git_diff {
                        if let Some(preview) = crate::diff::try_extract_from_shell_output(&output) {
                            self.activity.diff_preview = Some(preview);
                        }
                    }
                    if name == "shell_exec" {
                        // Reset the latch — each `git diff` call must
                        // re-arm it on a fresh `ToolCall`.
                        self.last_shell_exec_was_git_diff = false;
                    }

                    let summary = truncate_for_display(&output, 200);
                    self.activity.tool_log.push(ToolActivityEntry {
                        call_id,
                        tool_name: name,
                        summary,
                        is_error,
                        compressed: false,
                        compressed_symbol_count: 0,
                    });
                    changed = true;
                }
                Ok(TurnMessage::ToolResultCompressed {
                    call_id,
                    name,
                    symbol_count,
                }) => {
                    // Stamp the compression marker on the entry with
                    // the matching `(call_id, name)` pair. Matching by
                    // `call_id` (in addition to name) prevents
                    // concurrent same-name calls from cross-stamping
                    // each other's results (issue #49 review #6).
                    if let Some(entry) = self
                        .activity
                        .tool_log
                        .iter_mut()
                        .rev()
                        .find(|e| e.call_id == call_id && e.tool_name == name && !e.compressed)
                    {
                        entry.compressed = true;
                        entry.compressed_symbol_count = symbol_count;
                        changed = true;
                    }
                }
                Ok(TurnMessage::Done {
                    agent,
                    token_count,
                    max_tokens,
                }) => {
                    self.agent = Some(agent);
                    self.streaming = false;
                    self.thinking = false;
                    self.turn_handle = None;
                    self.context_usage = format_context_usage(token_count, max_tokens);
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
                    self.thinking = false;
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
                Ok(TurnMessage::CompactDone {
                    agent,
                    token_count,
                    max_tokens,
                }) => {
                    self.agent = Some(agent);
                    self.streaming = false;
                    self.turn_handle = None;
                    self.context_usage = format_context_usage(token_count, max_tokens);
                    self.chat_entries.push(ChatEntry {
                        role: Role::System,
                        text: "Context compressed successfully.".to_owned(),
                    });
                    self.set_chat_scroll(0);
                    return true;
                }
                Ok(TurnMessage::CompactError {
                    agent,
                    message,
                    cancelled: _,
                }) => {
                    self.agent = Some(agent);
                    self.streaming = false;
                    self.turn_handle = None;
                    self.error_message = Some(message);
                    self.set_chat_scroll(0);
                    return true;
                }
                Err(mpsc::error::TryRecvError::Empty) => {
                    return changed;
                }
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    // The task ended without sending Done/Error -- this
                    // means it panicked or was dropped unexpectedly.
                    // Reset streaming state and surface an error so the
                    // user knows recovery requires a restart.
                    self.streaming = false;
                    self.turn_handle = None;
                    self.error_message =
                        Some("Background task ended unexpectedly; please restart.".to_owned());
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

    /// Drain pending [`RunProgressEvent`]s from the runner's progress
    /// channel and update [`Self::run_header`] / the activity pane's
    /// `RunProgressSection`. Returns `true` if anything changed.
    pub fn drain_run_progress(&mut self) -> bool {
        let Some(rx) = self.run_progress_rx.as_mut() else {
            return false;
        };
        let mut changed = false;
        loop {
            match rx.try_recv() {
                Ok(RunProgressEvent::ScopeUpgraded { features_count }) => {
                    if let Some(header) = self.run_header.as_mut() {
                        // We don't have the canonical workflow id at
                        // this layer; the scope label still flips and
                        // the feature counter updates. The CLI's
                        // higher-level wiring is responsible for
                        // re-loading the run summary on the next idle
                        // tick to pick up the workflow id.
                        header.scope_label = "harnessed";
                        if let Some(total) = features_count {
                            header.features = Some((0, total));
                        }
                    }
                    let progress = &mut self.activity.run_progress;
                    if !progress.is_harnessed() {
                        progress.scope = tmg_harness::RunScope::harnessed("(unknown)", None);
                    }
                    if let Some(total) = features_count {
                        progress.features_total = total;
                    }
                    // SPEC §7.1: surface a transient overlay banner so
                    // the user sees the scope flip even if they were
                    // not watching the activity pane.
                    self.transient_banner = Some(TransientBanner::scope_upgrade(features_count));
                    changed = true;
                }
                Ok(RunProgressEvent::SessionEnded { trigger }) => {
                    // The runner moves to a fresh successor session
                    // for everything except `UserExit`; bumping the
                    // header counter is best-effort because the CLI
                    // re-attaches the new run summary on the next
                    // idle pass. On `UserExit` no successor session
                    // exists, so the counter must stay put.
                    if !matches!(trigger, tmg_harness::SessionEndTrigger::UserExit) {
                        if let Some(header) = self.run_header.as_mut() {
                            header.session_num = header.session_num.saturating_add(1);
                        }
                        self.activity.run_progress.session_num =
                            self.activity.run_progress.session_num.saturating_add(1);
                        self.activity.run_progress.turns = 0;
                    }
                    changed = true;
                }
                Err(mpsc::error::TryRecvError::Empty) => return changed,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    self.run_progress_rx = None;
                    return changed;
                }
            }
        }
    }

    /// Drain pending [`WorkflowProgress`] events. Returns `true` if a
    /// redraw is needed.
    ///
    /// Delegates the state transition to
    /// [`ActivityPane::apply_workflow_event`] so the activity-pane
    /// state machine is testable in isolation.
    pub fn drain_workflow_progress(&mut self) -> bool {
        let Some(rx) = self.workflow_progress_rx.as_mut() else {
            return false;
        };
        let mut changed = false;
        loop {
            match rx.try_recv() {
                Ok(ev) => {
                    // Snapshot the activity-pane state machine first
                    // (regardless of variant) so the workflow progress
                    // section reflects the latest step.
                    self.activity.apply_workflow_event(&ev);
                    // For human-input events, capture the responder so
                    // the modal UI can drive the reply.
                    if let WorkflowProgress::HumanInputRequired {
                        step_id,
                        message,
                        options,
                        show,
                        revise_target,
                        response_tx,
                    } = ev
                    {
                        // If a previous prompt is still alive (engines
                        // emit one prompt at a time, but a flaky
                        // observer might double-fire), the new prompt
                        // wins. The prior responder is dropped, which
                        // the engine treats as a missing reply.
                        self.human_prompt = Some(
                            HumanPrompt::new(step_id, message, show, options, response_tx)
                                .with_revise_target(revise_target),
                        );
                    }
                    changed = true;
                }
                Err(mpsc::error::TryRecvError::Empty) => return changed,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    self.workflow_progress_rx = None;
                    return changed;
                }
            }
        }
    }

    /// Drain all three event channels (turn / run / workflow) in one
    /// sweep. The event loop calls this every tick; returns `true` if
    /// any channel produced an update.
    ///
    /// Order: turn messages first (they may take ownership of the
    /// agent and gate the next compact step), then run progress
    /// (header refresh), then workflow progress (activity pane
    /// section). The order matches the user-visible priority — turn
    /// updates always win.
    pub fn drain_app_events(&mut self) -> bool {
        let mut changed = false;
        if self.drain_turn_messages() {
            changed = true;
        }
        if self.drain_run_progress() {
            changed = true;
        }
        if self.drain_workflow_progress() {
            changed = true;
        }
        changed
    }

    /// Return a mutable reference to the agent loop, if available.
    pub fn agent_mut(&mut self) -> Option<&mut AgentLoop> {
        self.agent.as_mut()
    }

    /// Update the context usage display string.
    pub fn update_context_usage(&mut self, usage: String) {
        self.context_usage = usage;
    }

    /// Push a system-role message to the chat display.
    pub fn push_system_message(&mut self, text: String) {
        self.chat_entries.push(ChatEntry {
            role: Role::System,
            text,
        });
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

/// Inspect a tool-call argument blob for `file_write` / `file_patch`
/// and update `current` with a synthesised diff preview when the
/// arguments parse successfully.
///
/// We accept both the production `path` / `content` shape used by
/// `tmg-tools::FileWriteTool` and a fallback that picks up `content`
/// from a `file_patch` payload (which carries `path`, `search`,
/// `replace`). The fallback synthesises a tiny "search → replace"
/// diff so the activity pane has something to render until #46
/// teaches `file_patch` to emit a real unified diff.
fn capture_diff_from_call(current: &mut Option<DiffPreview>, name: &str, arguments: &str) {
    match name {
        "file_write" => {
            let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) else {
                return;
            };
            let path = value.get("path").and_then(|v| v.as_str());
            let content = value.get("content").and_then(|v| v.as_str());
            if let (Some(path), Some(content)) = (path, content) {
                *current = Some(DiffPreview::from_file_write(path, content));
            }
        }
        "file_patch" => {
            let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) else {
                return;
            };
            let path = value.get("path").and_then(|v| v.as_str());
            let search = value.get("search").and_then(|v| v.as_str());
            let replace = value.get("replace").and_then(|v| v.as_str());
            if let (Some(path), Some(search), Some(replace)) = (path, search, replace) {
                *current = Some(synthesise_patch_preview(path, search, replace));
            }
        }
        _ => {}
    }
}

/// Inspect a `shell_exec` tool-call argument blob and report whether
/// the requested command is a `git diff` invocation.
///
/// The `shell_exec` tool's input shape is `{"command": "..."}`. We
/// JSON-parse best-effort, trim leading whitespace from the command,
/// and check (case-insensitively) for a `git diff` prefix. Anything
/// non-conforming is treated as "not a git diff" so the TUI falls
/// back to leaving the existing diff preview alone.
fn shell_exec_is_git_diff(arguments: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) else {
        return false;
    };
    let Some(command) = value.get("command").and_then(|v| v.as_str()) else {
        return false;
    };
    let trimmed = command.trim_start();
    // Match `git diff` and `git diff <args>` while letting `git
    // difftool` or `git diffstat` slip through (the next char must be
    // whitespace or end-of-string). Case-insensitive on the leading
    // command so `GIT DIFF` (rare, but legal in some shells via
    // aliases) still matches.
    let lower = trimmed.to_ascii_lowercase();
    let prefix = "git diff";
    if !lower.starts_with(prefix) {
        return false;
    }
    matches!(
        lower.as_bytes().get(prefix.len()),
        None | Some(b' ' | b'\t')
    )
}

/// Build a tiny synthesised hunk for a `file_patch` call. This is
/// best-effort — we do not have access to the surrounding file
/// content at the TUI layer — so the hunk markers are zero, and the
/// hunk body is `search` lines as `-` followed by `replace` lines as
/// `+`.
fn synthesise_patch_preview(path: &str, search: &str, replace: &str) -> DiffPreview {
    use crate::diff::{DiffHunk, DiffLine, DiffLineKind};

    let mut lines: Vec<DiffLine> = Vec::new();
    for line in search.lines() {
        lines.push(DiffLine {
            kind: DiffLineKind::Removed,
            content: line.to_owned(),
        });
    }
    for line in replace.lines() {
        lines.push(DiffLine {
            kind: DiffLineKind::Added,
            content: line.to_owned(),
        });
    }
    let removed = u32::try_from(search.lines().count()).unwrap_or(u32::MAX);
    let added = u32::try_from(replace.lines().count()).unwrap_or(u32::MAX);
    DiffPreview::with_hunks(
        PathBuf::from(path),
        vec![DiffHunk {
            old_start: 0,
            old_lines: removed,
            new_start: 0,
            new_lines: added,
            header_context: "file_patch".to_owned(),
            lines,
        }],
    )
}

/// Truncate a string for display, replacing middle with ellipsis if
/// it exceeds `max_len` characters (measured in `char` count, not bytes).
fn truncate_for_display(s: &str, max_len: usize) -> String {
    let first_line = s.lines().next().unwrap_or(s);
    if first_line.chars().count() <= max_len {
        return first_line.to_owned();
    }
    let half = max_len / 2;
    let start: String = first_line.chars().take(half).collect();
    let end: String = first_line
        .chars()
        .rev()
        .take(half)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{start}...{end}")
}

/// A [`StreamSink`] that sends tokens and tool activity through an
/// `mpsc` channel.
struct ChannelStreamSink {
    tx: mpsc::Sender<TurnMessage>,
}

impl StreamSink for ChannelStreamSink {
    fn on_thinking(&mut self, token: &str) -> Result<(), CoreError> {
        self.tx
            .try_send(TurnMessage::Thinking(token.to_owned()))
            .map_err(|_| CoreError::Cancelled)
    }

    fn on_token(&mut self, token: &str) -> Result<(), CoreError> {
        self.tx
            .try_send(TurnMessage::Token(token.to_owned()))
            .map_err(|_| CoreError::Cancelled)
    }

    fn on_done(&mut self) -> Result<(), CoreError> {
        Ok(())
    }

    fn on_tool_call(
        &mut self,
        call_id: &str,
        name: &str,
        arguments: &str,
    ) -> Result<(), CoreError> {
        self.tx
            .try_send(TurnMessage::ToolCall {
                call_id: call_id.to_owned(),
                name: name.to_owned(),
                arguments: arguments.to_owned(),
            })
            .map_err(|_| CoreError::Cancelled)
    }

    fn on_tool_result(
        &mut self,
        call_id: &str,
        name: &str,
        output: &str,
        is_error: bool,
    ) -> Result<(), CoreError> {
        self.tx
            .try_send(TurnMessage::ToolResult {
                call_id: call_id.to_owned(),
                name: name.to_owned(),
                output: output.to_owned(),
                is_error,
            })
            .map_err(|_| CoreError::Cancelled)
    }

    fn on_tool_result_compressed(
        &mut self,
        call_id: &str,
        name: &str,
        symbol_count: usize,
    ) -> Result<(), CoreError> {
        self.tx
            .try_send(TurnMessage::ToolResultCompressed {
                call_id: call_id.to_owned(),
                name: name.to_owned(),
                symbol_count,
            })
            .map_err(|_| CoreError::Cancelled)
    }

    fn on_warning(&mut self, message: &str) -> Result<(), CoreError> {
        self.tx
            .try_send(TurnMessage::ToolResult {
                // System warnings are synthetic — they don't have a real
                // tool-call id. We use an empty string so the activity
                // pane treats them as unmatched and never tries to pair
                // them with a `compressed` marker.
                call_id: String::new(),
                name: "system".to_owned(),
                output: format!("warning: {message}"),
                is_error: true,
            })
            .map_err(|_| CoreError::Cancelled)
    }
}

#[expect(
    clippy::expect_used,
    reason = "tests use expect for clarity; the workspace policy denies them in production code"
)]
#[cfg(test)]
mod tests {
    use super::*;

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
    fn slash_command_parsing_routes_through_shared_parser() {
        // Sanity-check that the shared parser produces the canonical
        // SlashCommand for a few well-known inputs. The full parser
        // contract lives in `tmg-skills::slash`; this test exists so a
        // future rewire to a TUI-local parser is detected here.
        use tmg_skills::SlashCommand as Cmd;
        assert_eq!(
            tmg_skills::parse_slash_command("/clear").expect("ok"),
            Some(Cmd::Clear),
        );
        assert_eq!(
            tmg_skills::parse_slash_command("/exit").expect("ok"),
            Some(Cmd::Exit),
        );
        assert_eq!(
            tmg_skills::parse_slash_command("/run upgrade").expect("ok"),
            Some(Cmd::RunUpgrade),
        );
        assert_eq!(tmg_skills::parse_slash_command("hello").expect("ok"), None,);
    }

    #[test]
    fn truncate_for_display_short() {
        let result = truncate_for_display("hello", 10);
        assert_eq!(result, "hello");
    }

    #[test]
    fn truncate_for_display_long() {
        let long = "a".repeat(300);
        let result = truncate_for_display(&long, 20);
        assert!(result.len() < 30);
        assert!(result.contains("..."));
    }

    #[test]
    fn truncate_for_display_multiline() {
        let text = "first line\nsecond line\nthird line";
        let result = truncate_for_display(text, 100);
        assert_eq!(result, "first line");
    }

    #[test]
    fn capture_diff_from_file_write_call() {
        let mut current: Option<DiffPreview> = None;
        let args = serde_json::json!({
            "path": "src/foo.rs",
            "content": "fn main() {}\n",
        })
        .to_string();
        capture_diff_from_call(&mut current, "file_write", &args);
        let preview = current.expect("preview should be set");
        assert_eq!(preview.file, PathBuf::from("src/foo.rs"));
        assert_eq!(preview.hunks.len(), 1);
    }

    #[test]
    fn capture_diff_from_file_patch_call() {
        let mut current: Option<DiffPreview> = None;
        let args = serde_json::json!({
            "path": "src/bar.rs",
            "search": "fn old() {}",
            "replace": "fn new() {}",
        })
        .to_string();
        capture_diff_from_call(&mut current, "file_patch", &args);
        let preview = current.expect("preview should be set");
        assert_eq!(preview.file, PathBuf::from("src/bar.rs"));
        assert_eq!(preview.hunks[0].lines.len(), 2);
    }

    #[test]
    fn capture_diff_ignores_other_tools() {
        let mut current: Option<DiffPreview> = None;
        let args = serde_json::json!({"command": "ls"}).to_string();
        capture_diff_from_call(&mut current, "shell_exec", &args);
        assert!(current.is_none());
    }

    #[test]
    fn capture_diff_handles_malformed_json() {
        let mut current: Option<DiffPreview> = None;
        capture_diff_from_call(&mut current, "file_write", "not json {");
        assert!(current.is_none());
    }

    #[test]
    fn shell_exec_is_git_diff_matches_plain_command() {
        let args = serde_json::json!({"command": "git diff"}).to_string();
        assert!(shell_exec_is_git_diff(&args));
    }

    #[test]
    fn shell_exec_is_git_diff_matches_with_args_and_whitespace() {
        let args = serde_json::json!({"command": "  git diff --stat HEAD~1"}).to_string();
        assert!(shell_exec_is_git_diff(&args));
    }

    #[test]
    fn shell_exec_is_git_diff_is_case_insensitive() {
        let args = serde_json::json!({"command": "GIT DIFF"}).to_string();
        assert!(shell_exec_is_git_diff(&args));
    }

    #[test]
    fn shell_exec_is_git_diff_rejects_other_commands() {
        for cmd in [
            "ls",
            "git log -p",
            "git status",
            "cat foo.diff",
            "git difftool",
            "git diffstat",
        ] {
            let args = serde_json::json!({ "command": cmd }).to_string();
            assert!(
                !shell_exec_is_git_diff(&args),
                "expected `{cmd}` to not match git diff",
            );
        }
    }

    #[test]
    fn shell_exec_is_git_diff_rejects_malformed() {
        assert!(!shell_exec_is_git_diff("not json {"));
        assert!(!shell_exec_is_git_diff("{}"));
        assert!(!shell_exec_is_git_diff(
            &serde_json::json!({"command": 42}).to_string(),
        ));
    }
}
