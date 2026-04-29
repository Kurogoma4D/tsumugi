//! Event loop: polls terminal events and drives the application.

use std::time::{Duration, Instant};

use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind,
};
use ratatui::DefaultTerminal;
use tmg_agents::{AgentKind, AgentType, SubagentConfig};
use tmg_memory::MemoryType;
use tmg_skills::{
    SkillCandidacySignal, SkillsRuntime, SlashCommand, TriggerKind,
    TurnSummary as SkillTurnSummary, parse_verdict,
};
use tmg_workflow::{HumanResponse, HumanResponseKind};
use tokio_util::sync::CancellationToken;

use crate::app::{App, SubmitDecision, TurnFinishedSignal};
use crate::error::TuiError;
use crate::human_prompt::{HumanPrompt, option_kind};

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

        // Issue #54: forward `/skill capture` / `/skills disable-auto`
        // flags into the SkillsRuntime BEFORE the after-turn pipeline
        // fires so the runtime sees them on the same evaluation pass.
        app.forward_skill_flags_to_runtime().await;

        // Issue #54: drive the autonomous-skill-pipeline once per
        // completed turn. The signal is set by `drain_turn_messages`
        // when a `Done`/`Error` (non-cancellation) message arrives.
        if let Some(signal) = app.take_turn_finished() {
            run_skill_after_turn(app, &signal).await;
        }

        // Issue #54: drain `/skills reject` requests outside the
        // after-turn hook so the user can reject a skill at any tick
        // (the rejection path is independent of signal evaluation).
        process_skill_rejections(app).await;

        // Expire the scope-upgrade banner once its TTL has lapsed.
        let _ = app.expire_banner_if_due();

        // Drain any queued async slash command produced by the most
        // recent `App::dispatch_slash` call. This must run BEFORE the
        // compact gate so a `/compact` queued from a slash dispatch
        // (none currently, but the path is symmetric) wouldn't be
        // shadowed by an in-flight compact.
        //
        // The dispatcher contains `runner.lock().await` calls that can
        // contend with a long-running tool turn; race it against the
        // cancellation token so Ctrl+C still breaks the TUI out of a
        // stuck dispatch (the partial side-effects are tolerable —
        // we're shutting down anyway).
        if let Some(cmd) = app.take_pending_slash() {
            tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    app.request_exit();
                    return Ok(());
                }
                () = dispatch_slash_async(app, cmd) => {}
            }
        }

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
        terminal.draw(|frame| crate::ui::draw(frame, app))?;

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
                        handle_event(app, &event, &cancel).await;
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
async fn handle_event(app: &mut App, event: &Event, cancel: &CancellationToken) {
    match event {
        Event::Key(key_event) => handle_key(app, *key_event, cancel).await,
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
async fn handle_key(app: &mut App, key: KeyEvent, cancel: &CancellationToken) {
    // Only process key-press events. With the Kitty keyboard protocol
    // enabled, Release and Repeat events are also reported; ignoring
    // them prevents double-firing and interference with IME composition.
    if key.kind != KeyEventKind::Press {
        return;
    }

    // The human-prompt modal takes priority over every other input
    // path so the user cannot accidentally cancel a pending workflow
    // step by typing into the chat box.
    if app.has_human_prompt() {
        handle_human_prompt_key(app, key).await;
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
    match app.submit_input() {
        SubmitDecision::Nothing => {}
        SubmitDecision::StartTurn(text) => {
            app.start_turn(text, cancel);
        }
        SubmitDecision::SlashCommand(cmd) => {
            if let Err(e) = app.dispatch_slash(cmd) {
                app.set_error(e.to_string());
            }
        }
        SubmitDecision::ParseError(msg) => {
            app.set_error(msg);
        }
    }
}

/// Handle a key event while the human-prompt modal is active.
///
/// Routes:
/// * `Tab` / `Right` / `Down`           → focus next option
/// * `Shift+Tab` / `Left` / `Up`        → focus previous option
/// * Letter keys                        → focus by initial
/// * `Enter`                            → confirm focused option
/// * `Esc`                              → cancel (drops the responder)
/// * `Ctrl+C`                           → forward to the standard
///   cancel path so the user can still abort the TUI
async fn handle_human_prompt_key(app: &mut App, key: KeyEvent) {
    match (key.code, key.modifiers) {
        (KeyCode::Char('c'), m) if m.contains(KeyModifiers::CONTROL) => {
            // Drop the responder before requesting exit so the engine
            // observes the missing reply on its next poll.
            if let Some(prompt) = app.take_human_prompt() {
                prompt.take_responder().await;
            }
            app.request_exit();
        }
        (KeyCode::Esc, _) => {
            if let Some(prompt) = app.take_human_prompt() {
                prompt.take_responder().await;
                app.set_error(format!(
                    "Cancelled human prompt for step '{}'",
                    prompt.step_id,
                ));
            }
        }
        (KeyCode::Tab, m) if !m.contains(KeyModifiers::SHIFT) => {
            if let Some(p) = app.human_prompt_mut() {
                p.focus_next();
            }
        }
        (KeyCode::BackTab | KeyCode::Tab, _) if key.modifiers.contains(KeyModifiers::SHIFT) => {
            if let Some(p) = app.human_prompt_mut() {
                p.focus_prev();
            }
        }
        (KeyCode::Right | KeyCode::Down, _) => {
            if let Some(p) = app.human_prompt_mut() {
                p.focus_next();
            }
        }
        (KeyCode::Left | KeyCode::Up, _) => {
            if let Some(p) = app.human_prompt_mut() {
                p.focus_prev();
            }
        }
        (KeyCode::Enter, _) => {
            if let Some(prompt) = app.take_human_prompt() {
                deliver_human_response(app, prompt).await;
            }
        }
        (KeyCode::Char(ch), m) if !m.contains(KeyModifiers::CONTROL) => {
            // Letter keys jump-select an option by initial. Don't fall
            // through to "type into chat" — the chat input is hidden
            // while the modal is up.
            if let Some(p) = app.human_prompt_mut() {
                p.focus_by_initial(ch);
            }
        }
        _ => {}
    }
}

/// Send the focused option of a human prompt back to the engine.
async fn deliver_human_response(app: &mut App, prompt: HumanPrompt) {
    let focused = prompt.focused_option().to_owned();
    let kind = option_kind(&focused);
    // For revise responses we require a `revise_target` declared by the
    // workflow; without one we cannot pick a meaningful target and
    // silently falling back to the human-step id would mismatch the
    // engine's `step_results` map (since the human step has not been
    // recorded yet) and break the rewind. Surface an error overlay
    // instead and keep the prompt alive so the user can pick a
    // different option.
    let target = if matches!(kind, HumanResponseKind::Revise) {
        if let Some(t) = prompt.revise_target.clone() {
            Some(t)
        } else {
            // Without a `revise_target` declared by the workflow we
            // cannot pick a meaningful rewind target — silently
            // falling back to the human-step id would mismatch the
            // engine's `step_results` map (the human step has not been
            // recorded yet) and break the rewind. Surface an error
            // overlay and re-stash the prompt so the user can pick
            // `Approve` or `Reject` instead. The responder is left
            // intact (we do not call `take_responder`) so the modal
            // is still actionable.
            let step_id = prompt.step_id.clone();
            app.set_error(format!(
                "[human:{step_id}] cannot revise: workflow did not declare `revise_target`",
            ));
            app.set_human_prompt(prompt);
            return;
        }
    } else {
        None
    };
    let response = HumanResponse { kind, target };
    let label = focused.clone();
    let step_id = prompt.step_id.clone();
    let send_result = prompt.respond(response).await;
    match send_result {
        Ok(()) => {
            app.push_slash_output(format!("[human:{step_id}] responded with `{label}`",));
        }
        Err(crate::human_prompt::RespondError::AlreadyResponded) => {
            app.set_error(format!(
                "[human:{step_id}] failed to deliver `{label}` (already responded)",
            ));
        }
        Err(crate::human_prompt::RespondError::ChannelClosed) => {
            app.set_error(format!(
                "[human:{step_id}] failed to deliver `{label}` (channel closed)",
            ));
        }
    }
}

/// Drive an async slash command. Routes /run subcommands through
/// [`tmg_harness::commands`] and surfaces results / errors as system
/// chat entries or error overlays.
async fn dispatch_slash_async(app: &mut App, cmd: SlashCommand) {
    match cmd {
        SlashCommand::ListSkills => {
            // Render the discovered skills list. The same metadata is
            // used by the agent's `use_skill` tool, so the listing
            // mirrors the LLM's view of available skills.
            let text = tmg_skills::format_skills_list(app.skills());
            app.push_slash_output(text);
        }
        SlashCommand::InvokeSkill { .. } => {
            // Direct skill invocation from the TUI is not wired up:
            // skills run via the LLM's `use_skill` tool inside a turn.
            // Surface a friendly hint so the user reaches the canonical
            // path.
            app.set_error(
                "Skill invocation is dispatched via the agent's tool loop; \
                 type your request normally and the agent will pick the right skill"
                    .to_owned(),
            );
        }
        SlashCommand::RunList => run_list(app).await,
        SlashCommand::RunStatus { run_id } => run_status(app, run_id.as_deref()).await,
        SlashCommand::RunUpgrade => run_upgrade(app).await,
        SlashCommand::RunDowngrade => run_downgrade(app).await,
        SlashCommand::RunNewSession => run_new_session(app).await,
        SlashCommand::RunPause => run_pause(app).await,
        SlashCommand::RunAbort => run_abort(app).await,
        SlashCommand::RunStart { workflow, args } => {
            // In-TUI workflow start is tracked in #71. For now, the TUI
            // cannot drive a workflow synchronously: workflows execute
            // via the LLM's `run_workflow` tool call. Surface a hint
            // that drives the user to the canonical path.
            let arg_hint = args.as_deref().unwrap_or("");
            let suffix = if arg_hint.is_empty() {
                String::new()
            } else {
                format!(" with arguments `{arg_hint}`")
            };
            app.push_slash_output(format!(
                "/run start is not yet wired into the TUI (see #71). \
                 Ask the agent: \"run the {workflow} workflow{suffix}\". \
                 The agent dispatches workflows via the run_workflow tool.",
            ));
        }
        SlashCommand::RunResume { run_id } => {
            // In-TUI run rotation is tracked in #71. The TUI currently
            // attaches to a single run for its whole lifetime; show a
            // hint pointing at the CLI command.
            let target = run_id.unwrap_or_else(|| "<id>".to_owned());
            app.push_slash_output(format!(
                "/run resume is not yet wired into the TUI (see #71). \
                 Exit and run `tmg run resume {target}` to attach to that run.",
            ));
        }
        // The synchronous variants are handled in `App::dispatch_slash`;
        // they should never reach the async path. Treat as a no-op so
        // a future extension that re-routes them does not panic.
        SlashCommand::Clear
        | SlashCommand::Compact
        | SlashCommand::Exit
        | SlashCommand::ListAgents
        | SlashCommand::ListWorkflows => {}
        // `SlashCommand` is `#[non_exhaustive]`; future variants land
        // here as a friendly error overlay until the dispatcher learns
        // about them.
        other => {
            app.set_error(format!("Unsupported async slash command: {other:?}",));
        }
    }
}

async fn run_list(app: &mut App) {
    let Some(runner) = app.runner().cloned() else {
        app.set_error("/run list requires an active run store".to_owned());
        return;
    };
    let store = {
        let guard = runner.lock().await;
        guard.store().clone()
    };
    match tmg_harness::commands::list(&store) {
        Ok(summaries) => {
            if summaries.is_empty() {
                app.push_slash_output("(no runs found)".to_owned());
                return;
            }
            let mut text = String::from("Runs:\n");
            for s in summaries {
                use std::fmt::Write as _;
                // The summary carries the full RunStatus; render it
                // via Debug for now (the CLI uses the same shape).
                let _ = writeln!(
                    text,
                    "  {} {} {:?} (sessions: {})",
                    s.short_id(),
                    s.scope_label,
                    s.status,
                    s.session_count,
                );
            }
            app.push_slash_output(text);
        }
        Err(e) => {
            app.set_error(format!("/run list failed: {e}"));
        }
    }
}

async fn run_status(app: &mut App, run_id: Option<&str>) {
    let Some(runner) = app.runner().cloned() else {
        app.set_error("/run status requires an active run".to_owned());
        return;
    };
    let (store, default_id) = {
        let guard = runner.lock().await;
        (guard.store().clone(), guard.run_id().clone())
    };
    let resolved = match run_id {
        Some(raw) => match tmg_harness::RunId::parse(raw.to_owned()) {
            Ok(id) => id,
            Err(e) => {
                app.set_error(format!("/run status: invalid run id {raw:?}: {e}"));
                return;
            }
        },
        None => default_id,
    };
    match tmg_harness::commands::status(&store, &resolved, 5) {
        Ok(report) => {
            use std::fmt::Write as _;
            let mut text = String::new();
            let _ = writeln!(
                text,
                "Run {} ({})",
                report.run.id.short(),
                report.run.scope.label(),
            );
            let _ = writeln!(text, "  Status:    {:?}", report.run.status);
            let _ = writeln!(text, "  Sessions:  {}", report.run.session_count);
            if let Some(hist) = report.feature_histogram {
                let _ = writeln!(text, "  Features:  {}/{} passing", hist.passing, hist.total,);
            }
            if !report.progress_tail.is_empty() {
                let _ = writeln!(
                    text,
                    "\nprogress.md (last {} lines):\n{}",
                    report.progress_tail_lines, report.progress_tail,
                );
            }
            app.push_slash_output(text);
        }
        Err(e) => {
            app.set_error(format!("/run status failed: {e}"));
        }
    }
}

async fn run_upgrade(app: &mut App) {
    let Some(runner) = app.runner().cloned() else {
        app.set_error("/run upgrade requires an active run".to_owned());
        return;
    };
    let mut guard = runner.lock().await;
    match tmg_harness::commands::upgrade(&mut guard) {
        Ok(()) => {
            let label = guard.run_id().short();
            app.push_slash_output(format!("Upgraded run {label} -> harnessed"));
        }
        Err(e) => {
            app.set_error(format!("/run upgrade failed: {e}"));
        }
    }
}

async fn run_downgrade(app: &mut App) {
    let Some(runner) = app.runner().cloned() else {
        app.set_error("/run downgrade requires an active run".to_owned());
        return;
    };
    let mut guard = runner.lock().await;
    match tmg_harness::commands::downgrade(&mut guard) {
        Ok(()) => {
            let label = guard.run_id().short();
            app.push_slash_output(format!("Downgraded run {label} -> ad_hoc"));
        }
        Err(e) => {
            app.set_error(format!("/run downgrade failed: {e}"));
        }
    }
}

async fn run_new_session(app: &mut App) {
    let Some(runner) = app.runner().cloned() else {
        app.set_error("/run new-session requires an active run".to_owned());
        return;
    };
    let mut guard = runner.lock().await;
    match tmg_harness::commands::new_session(&mut guard) {
        Ok(()) => {
            let label = guard.run_id().short();
            let session = guard.run().session_count;
            app.push_slash_output(format!("Rotated run {label} to session #{session}",));
        }
        Err(e) => {
            app.set_error(format!("/run new-session failed: {e}"));
        }
    }
}

async fn run_pause(app: &mut App) {
    let Some(runner) = app.runner().cloned() else {
        app.set_error("/run pause requires an active run".to_owned());
        return;
    };
    // The shared `commands::pause` helper refuses to mutate when a
    // live TUI sentinel is detected. That guard exists to keep an
    // external `tmg run pause` CLI invocation from racing the live
    // runner — but we ARE the live runner here, so call the
    // guard-free `set_status` path directly. The TUI then exits via
    // request_exit so the user observes a paused run on disk.
    let mut guard = runner.lock().await;
    match guard.set_status(tmg_harness::RunStatus::Paused) {
        Ok(()) => {
            let label = guard.run_id().short();
            app.push_slash_output(format!("Paused run {label}; exit the TUI to release it",));
        }
        Err(e) => {
            app.set_error(format!("/run pause failed: {e}"));
        }
    }
}

async fn run_abort(app: &mut App) {
    let Some(runner) = app.runner().cloned() else {
        app.set_error("/run abort requires an active run".to_owned());
        return;
    };
    // Same rationale as `/run pause`: bypass the TUI-attached guard
    // because we *are* the attached TUI.
    let mut guard = runner.lock().await;
    match guard.set_status(tmg_harness::RunStatus::Failed {
        reason: "user aborted (TUI)".to_owned(),
    }) {
        Ok(()) => {
            let label = guard.run_id().short();
            app.push_slash_output(format!("Aborted run {label} (status=failed)"));
        }
        Err(e) => {
            app.set_error(format!("/run abort failed: {e}"));
        }
    }
}

// =====================================================================
// Issue #54: autonomous-skill-creation pipeline driver.
//
// The flow:
//
// 1. `drain_turn_messages` produces a [`TurnFinishedSignal`] every time
//    the agent loop's `turn` finishes (Done or Error, but not
//    cancellation).
// 2. The event loop's main tick calls `run_skill_after_turn`, which:
//    a. Drains the per-turn outcome buffer ([`TurnOutcomeRecorder`])
//       and converts it to a [`SkillTurnSummary`].
//    b. Calls `SkillsRuntime::record_turn` to evaluate the
//       [`tmg_skills::SignalCollector`] triggers.
//    c. If a signal fires AND the runtime is allowed to auto-create,
//       spawns `AgentType::SkillCritic` via the [`SubagentManager`],
//       awaits the JSON verdict, parses it, and applies it via
//       `SkillsRuntime::apply_verdict` — which itself invokes
//       `SkillManageTool::execute_inner` to write the SKILL.md.
//    d. Logs every step via `tracing::info!` / `tracing::warn!`.
// =====================================================================

/// Cap on how long the `skill_critic` subagent gets to produce a verdict
/// before we abandon the round. Local LLMs producing JSON for a ~1k
/// token prompt should resolve in well under 30 s; anything longer is
/// almost certainly a stuck/failed model.
const SKILL_CRITIC_TIMEOUT: Duration = Duration::from_secs(60);

/// Drive the autonomous-skill-creation pipeline for one just-completed
/// turn. Best-effort: every failure path logs and returns without
/// affecting the live TUI.
#[expect(
    clippy::too_many_lines,
    reason = "linear after-turn pipeline; splitting into helpers obscures the step ordering"
)]
async fn run_skill_after_turn(app: &mut App, signal: &TurnFinishedSignal) {
    let Some(runtime_arc) = app.skills_runtime().cloned() else {
        return;
    };
    let Some(recorder) = app.skill_outcome_recorder().cloned() else {
        return;
    };

    // Drain the per-turn `use_skill` invocations and record each one's
    // outcome. We use the turn-level errored flag as a coarse
    // success/failure proxy: every `use_skill` invocation in a turn
    // that ended cleanly is counted as a success, while invocations
    // in an errored turn are counted as failures (with the error
    // message as the improvement hint). This is the simplest correct
    // approximation — per-call success would require pairing tool
    // calls with subsequent assistant responses, which is out of
    // scope for the per-turn observer.
    let invoked_skills = recorder.take_use_skill_invocations();
    if !invoked_skills.is_empty() {
        let outcome = if signal.errored {
            tmg_skills::UseSkillOutcome::Failure("follow-up turn ended in error".to_owned())
        } else {
            tmg_skills::UseSkillOutcome::Success
        };
        for skill_name in invoked_skills {
            let runtime_guard = runtime_arc.lock().await;
            match runtime_guard
                .record_use_skill_outcome(&skill_name, outcome.clone())
                .await
            {
                Ok(metrics) => {
                    tracing::info!(
                        skill = %skill_name,
                        success_count = metrics.success_count,
                        failure_count = metrics.failure_count,
                        "use_skill outcome recorded",
                    );
                }
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        skill = %skill_name,
                        "record_use_skill_outcome failed (skill may not yet exist on disk)",
                    );
                }
            }
        }
    }

    // Build the skills `TurnSummary` from the outcome buffer. If the
    // buffer is empty AND the turn did not error, there's nothing
    // meaningful to record — skip the round-trip.
    let mut summary = recorder
        .take_for_turn(signal.turn_index)
        .unwrap_or_else(|| SkillTurnSummary {
            turn_index: signal.turn_index,
            tool_calls: Vec::new(),
            feedback_memory_written: false,
            turn_errored: signal.errored,
        });
    if signal.errored {
        summary.turn_errored = true;
    }

    // Feed the summary into the SignalCollector and pick the highest-
    // priority signal. The runtime carries the manual-capture flag
    // forwarded from `forward_skill_flags_to_runtime`, so a manual
    // signal also surfaces here.
    let signals = {
        let mut guard = runtime_arc.lock().await;
        guard.record_turn(&summary)
    };
    if signals.is_empty() {
        return;
    }
    let Some(chosen) = SkillsRuntime::highest_priority(&signals).cloned() else {
        return;
    };
    tracing::info!(
        kind = ?chosen.kind,
        turn_range = ?chosen.turn_range,
        tool_call_count = chosen.tool_call_count,
        "skill emergence trigger fired",
    );

    // Manual signals bypass the per-session budget; auto signals
    // require both `auto_creation_allowed` AND that the user did not
    // issue `/skills disable-auto` for the current session.
    let allow_auto = {
        let guard = runtime_arc.lock().await;
        guard.auto_creation_allowed()
    };
    let is_manual = chosen.kind == TriggerKind::Manual;
    if !is_manual && !allow_auto {
        tracing::info!(
            "skill_critic skipped: auto-creation not allowed (budget consumed or disabled)",
        );
        return;
    }

    // Spawn skill_critic and await its verdict.
    let Some(manager) = app.subagent_manager().cloned() else {
        tracing::warn!("skill_critic skipped: SubagentManager not installed in TUI");
        return;
    };

    let critic_input = format_critic_input(&signal.user_message, &summary, &chosen);
    let raw = match spawn_skill_critic(&manager, critic_input).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "skill_critic spawn or wait failed");
            return;
        }
    };

    let verdict = match parse_verdict(&raw) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, raw = %raw, "skill_critic verdict failed to parse");
            return;
        }
    };

    if !verdict.create {
        tracing::info!(reason = %verdict.reason, "skill_critic declined to create");
        app.push_system_message(format!(
            "skill_critic skipped: {} (reason: {})",
            verdict.name.as_deref().unwrap_or("unnamed"),
            verdict.reason,
        ));
        return;
    }

    // Apply the verdict. Borrow the SubagentManager's parent sandbox
    // so the SkillManageTool runs under the same restrictions as the
    // top-level agent.
    let sandbox_arc = {
        let guard = manager.lock().await;
        guard.parent_sandbox().clone()
    };
    let result = {
        let mut guard = runtime_arc.lock().await;
        guard.apply_verdict(&verdict, &sandbox_arc).await
    };
    match result {
        Ok(Some(path)) => {
            tracing::info!(
                path = %path.display(),
                name = %verdict.name.as_deref().unwrap_or("unnamed"),
                "skill_critic verdict applied: skill created",
            );
            app.push_system_message(format!(
                "skill_critic created skill {:?} at {}",
                verdict.name.as_deref().unwrap_or("unnamed"),
                path.display(),
            ));
        }
        Ok(None) => {
            // Should not happen: we guarded on `verdict.create` above.
            tracing::debug!("apply_verdict returned None despite create=true");
        }
        Err(e) => {
            tracing::warn!(error = %e, "skill_critic apply_verdict failed");
            app.set_error(format!("skill_critic apply failed: {e}"));
        }
    }
}

/// Process pending `/skills reject <name>` requests:
///   1. delete the skill directory via [`SkillsRuntime::apply_rejection`]
///   2. write a `feedback`-kind memory note describing the rejection.
async fn process_skill_rejections(app: &mut App) {
    let names = app.drain_skill_rejection_requests();
    if names.is_empty() {
        return;
    }
    let Some(runtime_arc) = app.skills_runtime().cloned() else {
        // No runtime — nothing to do but warn.
        for name in &names {
            app.set_error(format!(
                "cannot reject skill {name:?}: SkillsRuntime not installed",
            ));
        }
        return;
    };
    // Sandbox: borrow from the SubagentManager when available.
    let sandbox_arc = if let Some(manager) = app.subagent_manager().cloned() {
        let guard = manager.lock().await;
        Some(guard.parent_sandbox().clone())
    } else {
        None
    };

    for name in names {
        // Step 1: delete the skill directory.
        if let Some(sandbox) = sandbox_arc.as_ref() {
            let runtime_guard = runtime_arc.lock().await;
            match runtime_guard.apply_rejection(&name, sandbox).await {
                Ok(()) => {
                    tracing::info!(skill = %name, "rejected skill: directory removed");
                }
                Err(e) => {
                    tracing::warn!(error = %e, skill = %name, "skill rejection failed");
                }
            }
        }

        // Step 2: write a feedback memory note. Best-effort; failures
        // are logged but do not block the rejection.
        if let Some(memory) = app.memory_store().cloned() {
            let entry_name = format!("skill-reject-{name}");
            let description = format!("user rejected auto-generated skill {name:?}");
            let body = format!(
                "The autonomous skill_critic created a skill named `{name}` \
                 and the user rejected it via `/skills reject`. Avoid \
                 re-creating skills with this name without explicit user \
                 consent.\n",
            );
            match tokio::task::spawn_blocking(move || {
                memory.add(&entry_name, MemoryType::Feedback, &description, &body)
            })
            .await
            {
                Ok(Ok(_)) => {
                    tracing::info!(skill = %name, "wrote feedback memory note for rejection");
                }
                Ok(Err(e)) => {
                    tracing::warn!(
                        error = %e,
                        skill = %name,
                        "failed to write feedback memory note for rejection",
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        skill = %name,
                        "feedback memory spawn_blocking task panicked",
                    );
                }
            }
        }

        app.push_system_message(format!("Skill {name:?} rejected and removed."));
    }
}

/// Build the input string the `skill_critic` subagent sees.
///
/// The shape follows the `DEFAULT_SYSTEM_PROMPT` schema: a free-form
/// recap of what just happened. The critic's job is to decide
/// `create` / `skip`; it doesn't need a JSON envelope on the input
/// side.
fn format_critic_input(
    user_message: &str,
    summary: &SkillTurnSummary,
    signal: &SkillCandidacySignal,
) -> String {
    let tool_recap: Vec<String> = summary
        .tool_calls
        .iter()
        .map(|c| {
            format!(
                "  - {} ({})",
                c.tool,
                if c.success { "success" } else { "failed" }
            )
        })
        .collect();
    let trigger_label = match signal.kind {
        TriggerKind::SuccessfulComplexTask => "successful_complex_task",
        TriggerKind::ErrorRecovery => "error_recovery",
        TriggerKind::UserCorrection => "user_correction",
        TriggerKind::Manual => "manual_capture",
    };
    format!(
        "Trigger: {trigger_label}\nUser request: {user_message}\nTool calls observed:\n{tool_log}\n\
         Decide whether this procedure should be saved as a Skill.\n\
         Respond with the JSON verdict described in the system prompt.",
        user_message = user_message,
        tool_log = if tool_recap.is_empty() {
            "  (none)".to_owned()
        } else {
            tool_recap.join("\n")
        },
    )
}

/// Spawn `AgentType::SkillCritic` via the [`SubagentManager`] and wait
/// for its verdict (with a timeout). Returns the raw output string.
async fn spawn_skill_critic(
    manager: &std::sync::Arc<tokio::sync::Mutex<tmg_agents::SubagentManager>>,
    task: String,
) -> Result<String, String> {
    let config = SubagentConfig {
        agent_kind: AgentKind::Builtin(AgentType::SkillCritic),
        task,
        background: false,
    };
    let (id, rx) = {
        let mut guard = manager.lock().await;
        guard
            .spawn_with_notify(config)
            .await
            .map_err(|e| format!("spawn skill_critic: {e}"))?
    };
    tracing::info!(?id, "spawned skill_critic");
    // We hold no locks while awaiting — the manager's JoinSet drives
    // the subagent in the background.
    let recv = tokio::time::timeout(SKILL_CRITIC_TIMEOUT, rx)
        .await
        .map_err(|_| format!("skill_critic timed out after {SKILL_CRITIC_TIMEOUT:?}"))?
        .map_err(|e| format!("skill_critic channel: {e}"))?;
    recv.map_err(|e| format!("skill_critic returned error: {e}"))
}
