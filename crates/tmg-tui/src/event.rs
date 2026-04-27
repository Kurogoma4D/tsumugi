//! Event loop: polls terminal events and drives the application.

use std::time::{Duration, Instant};

use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind,
};
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

/// Minimum interval between subagent summary refreshes.
///
/// Refreshing every tick (~33ms) is excessive since it acquires a
/// `tokio::sync::Mutex` lock. Once per second is sufficient for
/// human-readable status updates.
const SUBAGENT_REFRESH_INTERVAL: Duration = Duration::from_secs(1);

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
    let mut last_subagent_refresh = Instant::now()
        .checked_sub(SUBAGENT_REFRESH_INTERVAL)
        .unwrap_or_else(Instant::now);

    loop {
        // Drain pending events from all three channels (turn / run /
        // workflow) before drawing. `drain_app_events` is the
        // canonical entry-point introduced in #45; it preserves the
        // previous turn-message behaviour and additionally pumps the
        // run-progress and workflow-progress streams.
        app.drain_app_events();

        // Handle pending /compact command as a background task.
        if app.needs_compact() && !app.is_streaming() {
            app.clear_pending_compact();
            app.start_compact(&cancel);
        }

        // Refresh subagent summaries at a throttled rate (once per second)
        // to avoid excessive lock contention with the manager mutex.
        if last_subagent_refresh.elapsed() >= SUBAGENT_REFRESH_INTERVAL {
            app.refresh_subagent_summaries().await;
            last_subagent_refresh = Instant::now();
        }

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
    match event {
        Event::Key(key_event) => handle_key(app, *key_event, cancel),
        Event::Mouse(mouse_event) => handle_mouse(app, *mouse_event),
        // Terminal resize and other events are handled automatically by
        // ratatui on the next draw call.
        _ => {}
    }
}

/// Number of lines to scroll per mouse wheel tick.
const MOUSE_SCROLL_LINES: u16 = 3;

/// Handle a mouse event.
fn handle_mouse(app: &mut App, mouse: MouseEvent) {
    match mouse.kind {
        MouseEventKind::ScrollUp => app.scroll_up(MOUSE_SCROLL_LINES),
        MouseEventKind::ScrollDown => app.scroll_down(MOUSE_SCROLL_LINES),
        _ => {}
    }
}

/// Handle a key event.
fn handle_key(app: &mut App, key: KeyEvent, cancel: &CancellationToken) {
    // Only process key-press events. With the Kitty keyboard protocol
    // enabled, Release and Repeat events are also reported; ignoring
    // them prevents double-firing and interference with IME composition.
    if key.kind != KeyEventKind::Press {
        return;
    }

    // Clear error on any keypress.
    app.clear_error();

    // While streaming, allow Ctrl+C to cancel and PageUp/PageDown to scroll.
    if app.is_streaming() {
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.cancel_turn();
            }
            KeyCode::PageUp => {
                app.scroll_up(10);
            }
            KeyCode::PageDown => {
                app.scroll_down(10);
            }
            _ => {}
        }
        return;
    }

    match (key.code, key.modifiers) {
        // Ctrl+C: exit (only when not streaming).
        (KeyCode::Char('c'), m) if m.contains(KeyModifiers::CONTROL) => {
            cancel.cancel();
            app.request_exit();
        }
        // Shift+Enter or Alt+Enter: insert newline.
        // (Alt+Enter is the fallback for terminals without Kitty keyboard protocol.)
        (KeyCode::Enter, m) if m.contains(KeyModifiers::SHIFT) || m.contains(KeyModifiers::ALT) => {
            app.insert_newline();
        }
        // Enter alone: submit input and start turn.
        (KeyCode::Enter, _) => {
            submit_and_start_turn(app, cancel);
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
