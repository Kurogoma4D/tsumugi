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

use tmg_agents::SubagentManager;
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
) -> Result<(), TuiError> {
    // Install a panic hook that restores the terminal before printing
    // the panic message.  Without this, a panic during TUI operation
    // leaves the terminal in raw mode, making the shell unusable.
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        ratatui::restore();
        original_hook(info);
    }));

    let mut terminal = ratatui::init();
    let mut app = App::new(agent, model_name, project_root, cwd);

    if let Some(manager) = subagent_manager {
        app.set_subagent_manager(manager);
    }

    let result = event::run_event_loop(&mut terminal, &mut app, cancel).await;

    ratatui::restore();

    result
}
