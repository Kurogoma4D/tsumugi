//! tmg-tui: Terminal UI built with ratatui and crossterm.
//!
//! Provides a chat interface for interactive conversations with a
//! local LLM via the tsumugi agent loop.

pub mod activity;
pub mod app;
pub mod banner;
pub mod diff;
pub mod error;
pub mod event;
pub mod human_prompt;
pub mod ui;

pub use activity::{
    ActivityPane, RunHeader, RunProgressSection, ToolActivityEntry, WorkflowProgressSection,
};
pub use app::App;
pub use banner::TransientBanner;
pub use diff::DiffPreview;
pub use error::TuiError;
pub use human_prompt::HumanPrompt;

use std::path::PathBuf;
use std::sync::Arc;

use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use tmg_agents::{CustomAgentDef, SubagentManager};
use tmg_core::AgentLoop;
use tmg_harness::{RunProgressReceiver, RunRunner, RunSummary};
use tmg_memory::MemoryStore;
use tmg_search::SearchIndex;
use tmg_skills::{SkillMeta, SkillsRuntime, TurnOutcomeRecorder};
use tmg_workflow::{WorkflowMeta, WorkflowProgress};
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

/// Run the TUI application.
///
/// Sets up the terminal, creates the application state, and runs the
/// event loop. The terminal is restored on exit (including on error
/// or panic).
///
/// # Arguments
///
/// * `agent` - The agent loop to drive conversations.
/// * `model_name` - Model name to display in the header.
/// * `cancel` - Cancellation token for graceful shutdown.
/// * `project_root` - Project root directory for prompt file loading.
/// * `cwd` - Current working directory for prompt file loading.
/// * `subagent_manager` - Optional shared subagent manager for status display.
/// * `custom_agents` - Custom agent definitions for `/agents` list display.
/// * `event_log` - Optional path to write structured event log (JSON Lines).
/// * `current_run` - Optional active run summary for header display.
/// * `runner` - Optional shared [`RunRunner`] used to wrap each turn's
///   sink with [`HarnessStreamSink`](tmg_harness::HarnessStreamSink) so
///   the active session's `tool_calls_count` and `files_modified` are
///   updated as the LLM drives the conversation.
/// * `run_progress_rx` - Optional [`RunProgressReceiver`] subscribed
///   to the runner's progress channel; the activity pane drains it on
///   every tick to refresh the run header / progress section.
/// * `workflow_progress_rx` - Optional receiver paired with a
///   foreground `run_workflow` observer (see
///   [`tmg_workflow::register_workflow_tools_with_observer`]); the
///   activity pane drains it every tick to surface live workflow
///   progress events.
/// * `workflows` - Discovered workflows for the `/workflows` slash
///   command listing. Empty vec disables that listing.
/// * `skills` - Discovered skills for the `/skills` slash command
///   listing. Empty vec disables that listing.
///
/// # Errors
///
/// Returns `TuiError` on terminal I/O failure or agent errors.
#[expect(
    clippy::too_many_arguments,
    reason = "entry point collecting all setup parameters; a config struct would add indirection without clarity"
)]
pub async fn run(
    agent: AgentLoop,
    model_name: &str,
    cancel: CancellationToken,
    project_root: PathBuf,
    cwd: PathBuf,
    subagent_manager: Option<Arc<Mutex<SubagentManager>>>,
    custom_agents: Vec<CustomAgentDef>,
    event_log: Option<PathBuf>,
    current_run: Option<RunSummary>,
    runner: Option<Arc<Mutex<RunRunner>>>,
    run_progress_rx: Option<RunProgressReceiver>,
    workflow_progress_rx: Option<mpsc::Receiver<WorkflowProgress>>,
    workflows: Vec<WorkflowMeta>,
    skills: Vec<SkillMeta>,
    memory_store: Option<Arc<MemoryStore>>,
    search_index: Option<Arc<SearchIndex>>,
    startup_banner: Option<String>,
    skills_runtime: Option<Arc<Mutex<SkillsRuntime>>>,
    skill_outcome_recorder: Option<Arc<TurnOutcomeRecorder>>,
    trajectory_recorder: Option<Arc<tmg_trajectory::Recorder>>,
) -> Result<(), TuiError> {
    // Pre-warm the syntect bundle off the rendering thread. Loading
    // `SyntaxSet::load_defaults_newlines` + `ThemeSet::load_defaults`
    // is ~30 ms locally; running it on a `spawn_blocking` task keeps
    // the first diff preview frame from stalling. The task is fire-
    // and-forget — the `OnceLock` inside `SyntaxBundle::get` is the
    // synchronisation surface — so any panic during load is logged
    // but does not block startup.
    tokio::task::spawn_blocking(|| {
        diff::prewarm_syntax_bundle();
    });

    let mut terminal = ratatui::init();

    // Enable mouse capture for scroll wheel support.
    let _ = execute!(std::io::stdout(), EnableMouseCapture);

    // Enable the Kitty keyboard protocol so that modifier keys
    // (e.g. Shift+Enter) are reported distinctly from plain keys.
    // Not all terminals support this; if the push fails we proceed
    // without it -- the fallback keybinding will apply.
    let kbd_enhanced = execute!(
        std::io::stdout(),
        PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::REPORT_EVENT_TYPES)
    )
    .is_ok();

    // Install a panic hook that restores the terminal before printing
    // the panic message.  Without this, a panic during TUI operation
    // leaves the terminal in raw mode, making the shell unusable.
    // Set the hook after enabling mouse/keyboard enhancements so we
    // can capture `kbd_enhanced` and conditionally clean up.
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = execute!(std::io::stdout(), DisableMouseCapture);
        if kbd_enhanced {
            let _ = execute!(std::io::stdout(), PopKeyboardEnhancementFlags);
        }
        ratatui::restore();
        original_hook(info);
    }));

    let mut app = App::new(agent, model_name, project_root, cwd, event_log);

    if let Some(manager) = subagent_manager {
        app.set_subagent_manager(manager);
    }

    if !custom_agents.is_empty() {
        app.set_custom_agents(custom_agents);
    }

    if let Some(run) = current_run {
        app.set_current_run(run);
    }

    if let Some(r) = runner {
        app.set_runner(r);
    }

    if let Some(rx) = run_progress_rx {
        app.set_run_progress_rx(rx);
    }

    if let Some(rx) = workflow_progress_rx {
        app.set_workflow_progress_rx(rx);
    }

    if !workflows.is_empty() {
        app.set_workflows(workflows);
    }

    if !skills.is_empty() {
        app.set_skills(skills);
    }

    if let Some(store) = memory_store {
        app.set_memory_store(store);
    }

    if let Some(idx) = search_index {
        app.set_search_index(idx);
    }

    if let Some(runtime) = skills_runtime {
        app.set_skills_runtime(runtime);
    }

    if let Some(recorder) = skill_outcome_recorder {
        app.set_skill_outcome_recorder(recorder);
    }

    if let Some(recorder) = trajectory_recorder {
        app.set_trajectory_recorder(recorder);
    }

    // Issue #54: surface the auto-generated-skill banner at startup
    // when the previous session left unacknowledged `provenance: agent`
    // skills behind. The CLI computes the banner text via
    // `tmg_skills::pending_banner_names` + `format_banner` and threads
    // it in here.
    if let Some(text) = startup_banner {
        app.set_transient_banner(TransientBanner::with_ttl(
            text,
            std::time::Duration::from_secs(8),
        ));
    }

    let result = event::run_event_loop(&mut terminal, &mut app, cancel).await;

    let _ = execute!(std::io::stdout(), DisableMouseCapture);
    if kbd_enhanced {
        let _ = execute!(std::io::stdout(), PopKeyboardEnhancementFlags);
    }

    ratatui::restore();

    result
}
