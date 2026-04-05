//! tmg-tui: Terminal UI built with ratatui and crossterm.
//!
//! Provides a chat interface for interactive conversations with a
//! local LLM via the tsumugi agent loop.

pub mod app;
pub mod error;
pub mod event;
pub mod ui;

pub use app::App;
pub use error::TuiError;

use std::path::PathBuf;
use std::sync::Arc;

use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture,
    KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use tmg_agents::{CustomAgentDef, SubagentManager};
use tmg_core::AgentLoop;
use tokio::sync::Mutex;
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
///
/// # Errors
///
/// Returns `TuiError` on terminal I/O failure or agent errors.
pub async fn run(
    agent: AgentLoop,
    model_name: &str,
    cancel: CancellationToken,
    project_root: PathBuf,
    cwd: PathBuf,
    subagent_manager: Option<Arc<Mutex<SubagentManager>>>,
    custom_agents: Vec<CustomAgentDef>,
) -> Result<(), TuiError> {
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

    let mut app = App::new(agent, model_name, project_root, cwd);

    if let Some(manager) = subagent_manager {
        app.set_subagent_manager(manager);
    }

    if !custom_agents.is_empty() {
        app.set_custom_agents(custom_agents);
    }

    let result = event::run_event_loop(&mut terminal, &mut app, cancel).await;

    let _ = execute!(std::io::stdout(), DisableMouseCapture);
    if kbd_enhanced {
        let _ = execute!(std::io::stdout(), PopKeyboardEnhancementFlags);
    }

    ratatui::restore();

    result
}
