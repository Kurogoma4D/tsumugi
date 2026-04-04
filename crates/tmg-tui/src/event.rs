//! Event loop: polls terminal events and drives the application.

use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::DefaultTerminal;
use tokio_util::sync::CancellationToken;

use crate::app::App;
use crate::error::TuiError;
use crate::ui;

/// The main event loop tick interval.
///
/// This controls how often the TUI checks for new terminal events
/// and redraws the screen during streaming.
const TICK_RATE: Duration = Duration::from_millis(33); // ~30 fps

/// Run the TUI event loop.
///
/// This function takes ownership of the terminal and drives the
/// application until exit is requested or the cancellation token
/// fires.
///
/// # Cancel safety
///
/// The `tokio::select!` in this loop uses only cancel-safe futures:
/// - `cancel.cancelled()` is cancel-safe (it is a future that
///   completes when the token is cancelled).
/// - `tokio::task::spawn_blocking` for event polling returns a
///   `JoinHandle` which is cancel-safe.
///
/// # Errors
///
/// Returns `TuiError` on I/O failure or agent errors.
pub async fn run_event_loop(
    terminal: &mut DefaultTerminal,
    app: &mut App,
    cancel: CancellationToken,
) -> Result<(), TuiError> {
    loop {
        // Drain any pending turn messages before drawing.
        app.drain_turn_messages();

        // Draw the current state.
        terminal.draw(|frame| ui::draw(frame, app))?;

        if app.should_exit() {
            return Ok(());
        }

        // Poll for events with cancellation support.
        tokio::select! {
            () = cancel.cancelled() => {
                app.request_exit();
                return Ok(());
            }
            maybe_event = poll_event() => {
                match maybe_event {
                    Ok(Some(event)) => {
                        handle_event(app, &event, &cancel);
                    }
                    Ok(None) => {
                        // No event within tick rate, continue
                        // (allows redraw with new streaming tokens).
                    }
                    Err(e) => {
                        return Err(TuiError::Io(e));
                    }
                }
            }
        }
    }
}

/// Poll for a terminal event with a timeout.
///
/// This uses `spawn_blocking` to avoid blocking the async runtime.
///
/// # Cancel safety
///
/// Returns a `JoinHandle` future which is cancel-safe.
async fn poll_event() -> Result<Option<Event>, std::io::Error> {
    tokio::task::spawn_blocking(move || {
        if event::poll(TICK_RATE)? {
            Ok(Some(event::read()?))
        } else {
            Ok(None)
        }
    })
    .await
    .map_err(std::io::Error::other)?
}

/// Handle a single terminal event.
fn handle_event(app: &mut App, event: &Event, cancel: &CancellationToken) {
    // Terminal resize and other events are handled automatically by
    // ratatui on the next draw call.
    if let Event::Key(key_event) = event {
        handle_key(app, *key_event, cancel);
    }
}

/// Handle a key event.
fn handle_key(app: &mut App, key: KeyEvent, cancel: &CancellationToken) {
    // Clear error on any keypress.
    app.clear_error();

    // While streaming, only Ctrl+C cancels the current turn.
    if app.is_streaming() {
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            app.cancel_turn();
        }
        return;
    }

    match (key.code, key.modifiers) {
        // Ctrl+C: exit (only when not streaming).
        (KeyCode::Char('c'), m) if m.contains(KeyModifiers::CONTROL) => {
            cancel.cancel();
            app.request_exit();
        }
        // Ctrl+Enter: submit input and start turn.
        (KeyCode::Enter, m) if m.contains(KeyModifiers::CONTROL) => {
            submit_and_start_turn(app, cancel);
        }
        // Enter alone: insert newline.
        (KeyCode::Enter, _) => {
            app.insert_newline();
        }
        // Backspace: delete char before cursor.
        (KeyCode::Backspace, _) => {
            app.delete_char_before_cursor();
        }
        // Delete: delete char at cursor.
        (KeyCode::Delete, _) => {
            app.delete_char_at_cursor();
        }
        // Arrow keys: cursor movement.
        (KeyCode::Left, _) => {
            app.move_cursor_left();
        }
        (KeyCode::Right, _) => {
            app.move_cursor_right();
        }
        // Home / End.
        (KeyCode::Home, _) => {
            app.move_cursor_home();
        }
        (KeyCode::End, _) => {
            app.move_cursor_end();
        }
        // Page Up / Page Down for scrolling.
        (KeyCode::PageUp, _) => {
            app.scroll_up(10);
        }
        (KeyCode::PageDown, _) => {
            app.scroll_down(10);
        }
        // Regular character input.
        (KeyCode::Char(ch), m) if !m.contains(KeyModifiers::CONTROL) => {
            app.insert_char(ch);
        }
        _ => {}
    }
}

/// Submit input and start a background conversation turn.
fn submit_and_start_turn(app: &mut App, cancel: &CancellationToken) {
    let user_input = match app.submit_input() {
        Ok(Some(text)) => text,
        Ok(None) => return,
        Err(e) => {
            app.set_error(e.to_string());
            return;
        }
    };

    app.start_turn(user_input, cancel);
}
