//! TUI rendering: layout and widget drawing.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, Borders, Gauge, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
};
use unicode_width::UnicodeWidthStr;

use tmg_agents::truncate_str;

use crate::activity::{RunProgressSection, WorkflowProgressSection};
use crate::app::App;
use crate::diff::DiffPreview;

/// Render the full application UI into the given frame.
pub fn draw(frame: &mut Frame, app: &App) {
    let size = frame.area();

    // Top-level layout: header, main area, input area.
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(5),    // Main area (chat + tool activity)
            Constraint::Length(5), // Input area
        ])
        .split(size);

    draw_header(frame, app, vertical[0]);
    draw_main_area(frame, app, vertical[1]);
    draw_input_area(frame, app, vertical[2]);

    // Draw the transient scope-upgrade banner overlay (SPEC §7.1).
    // Uses the chat-pane region so it sits above the conversation.
    if let Some(banner) = app.transient_banner() {
        draw_transient_banner(frame, banner, vertical[1]);
    }

    // Draw the human-prompt modal (SPEC §8.4) on top of everything
    // so it dominates the screen until dismissed.
    if let Some(prompt) = app.human_prompt() {
        draw_human_prompt(frame, prompt, size);
    }

    // Draw error overlay if present.
    if let Some(err) = app.error_message() {
        draw_error_overlay(frame, err, size);
    }
}

/// Draw the transient scope-upgrade banner at the top of the chat
/// pane.
fn draw_transient_banner(frame: &mut Frame, banner: &crate::TransientBanner, area: Rect) {
    let banner_height: u16 = 3;
    if area.height < banner_height {
        return;
    }
    // Position: top edge of the supplied region (typically the main
    // area); width spans the full pane minus a small inset so it
    // doesn't cover the activity pane border.
    let banner_area = Rect {
        x: area.x + 1,
        y: area.y,
        width: area.width.saturating_sub(2),
        height: banner_height,
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Notice ")
        .style(Style::default().fg(Color::Yellow));
    let text = Paragraph::new(Span::styled(
        format!("⚡ {}", banner.text),
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD),
    ))
    .block(block)
    .wrap(Wrap { trim: true });
    frame.render_widget(
        Block::default().style(Style::default().bg(Color::Black)),
        banner_area,
    );
    frame.render_widget(text, banner_area);
}

/// Draw the human-prompt modal centred on the screen.
fn draw_human_prompt(frame: &mut Frame, prompt: &crate::HumanPrompt, area: Rect) {
    // Centre a box that's 60% of the screen wide and at most 16 lines
    // tall (smaller for short messages).
    let modal_width =
        u16::try_from((u32::from(area.width) * 60 / 100).clamp(40, 100)).unwrap_or(u16::MAX);
    let body_lines: u16 = u16::try_from(prompt.message.lines().count()).unwrap_or(u16::MAX);
    let show_lines: u16 = prompt
        .show
        .as_ref()
        .map_or(0, |s| u16::try_from(s.lines().count()).unwrap_or(u16::MAX));
    // `options_lines` is an upper bound — the renderer collapses every
    // option into a single ratatui `Line` (see `lines.push(Line::from(opt_spans))`
    // below), so the actual height contribution is 1 row regardless of
    // option count. We keep the per-option count here as a generous
    // upper bound for narrow terminals where wrap can cause overflow,
    // and clamp the final value to `area.height` regardless.
    let options_lines: u16 = u16::try_from(prompt.options.len().max(1)).unwrap_or(u16::MAX);
    // Rendered rows (upper bound, clamped to `area.height`):
    //   borders (2) + body + spacer-before-options (1) + options (1)
    //   + spacer-before-footer (1) + footer (1)                = body + 6
    //   plus, when `show` is present:
    //       spacer-before-show (1) + show_lines                = show + 1
    // The formula below sums those plus `options_lines` (an upper
    // bound — see comment above) so it can over-estimate by
    // `options_lines - 1` rows; `min(area.height)` papers over both
    // the over- and under-estimate in practice.
    let show_block = if show_lines > 0 { show_lines + 1 } else { 0 };
    let modal_height = (6 + body_lines + show_block + options_lines).min(area.height);

    if area.width < modal_width || area.height < modal_height {
        return;
    }

    let modal_area = Rect {
        x: area.x + (area.width.saturating_sub(modal_width)) / 2,
        y: area.y + (area.height.saturating_sub(modal_height)) / 2,
        width: modal_width,
        height: modal_height,
    };

    // Clear behind the modal so the underlying chat doesn't bleed through.
    frame.render_widget(
        Block::default().style(Style::default().bg(Color::Black)),
        modal_area,
    );

    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" Human input — {} ", prompt.step_id))
        .style(Style::default().fg(Color::Cyan));
    let inner = block.inner(modal_area);
    frame.render_widget(block, modal_area);

    let mut lines: Vec<Line<'_>> = Vec::new();
    for ml in prompt.message.lines() {
        lines.push(Line::from(Span::styled(
            ml.to_owned(),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )));
    }
    if let Some(show) = &prompt.show {
        lines.push(Line::from(""));
        for s in show.lines() {
            lines.push(Line::from(Span::styled(
                s.to_owned(),
                Style::default().fg(Color::Gray),
            )));
        }
    }
    lines.push(Line::from(""));
    let mut opt_spans: Vec<Span<'_>> = Vec::new();
    for (idx, opt) in prompt.options.iter().enumerate() {
        let focused = idx == prompt.focused;
        let style = if focused {
            Style::default()
                .fg(Color::Black)
                .bg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Yellow)
        };
        let label = if focused {
            format!(" › {opt} ")
        } else {
            format!("   {opt} ")
        };
        opt_spans.push(Span::styled(label, style));
    }
    lines.push(Line::from(opt_spans));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Tab/←→: focus | Enter: confirm | a/r/v: jump | Esc: cancel",
        Style::default().fg(Color::DarkGray),
    )));

    let p = Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false });
    frame.render_widget(p, inner);
}

/// Draw the header bar: model name, context usage, run header
/// (`[run: <id> <scope> [N/M]] [session: #N]`, SPEC §7.1), and
/// subagent count.
fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let running_count = app.running_subagent_count();
    let has_run = app.run_header().is_some();
    let has_agents = running_count > 0;

    // Build constraints dynamically. The run pane is allocated more
    // space than the others when present because the SPEC §7.1
    // header string can be long (id + scope + features + session).
    let mut constraints: Vec<Constraint> = Vec::with_capacity(4);
    constraints.push(Constraint::Min(20)); // Model
    constraints.push(Constraint::Min(18)); // Context
    if has_run {
        constraints.push(Constraint::Min(36)); // Run header
    }
    if has_agents {
        constraints.push(Constraint::Min(14)); // Agents
    }

    let header_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(constraints)
        .split(area);

    let mut idx = 0usize;

    let model_block = Block::default().borders(Borders::ALL).title(" Model ");
    let model_text = Paragraph::new(Line::from(vec![Span::styled(
        app.model_name(),
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    )]))
    .block(model_block);
    frame.render_widget(model_text, header_layout[idx]);
    idx += 1;

    let context_block = Block::default().borders(Borders::ALL).title(" Context ");
    let context_text = Paragraph::new(Line::from(vec![Span::styled(
        app.context_usage(),
        Style::default().fg(Color::Yellow),
    )]))
    .block(context_block);
    frame.render_widget(context_text, header_layout[idx]);
    idx += 1;

    if let Some(header) = app.run_header() {
        // Refresh the header's feature counters from the activity
        // pane's `RunProgressSection` so harnessed runs always
        // reflect the latest counters even when they were not part
        // of a `RunProgressEvent` (e.g. a feature_list_mark_passing
        // tool call).
        let mut header_str = header.clone();
        let progress = app.run_progress();
        if progress.is_harnessed() {
            header_str.features = Some((progress.features_done, progress.features_total));
        }
        let run_block = Block::default().borders(Borders::ALL).title(" Run ");
        let run_text = Paragraph::new(Line::from(vec![Span::styled(
            header_str.format(),
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        )]))
        .block(run_block);
        frame.render_widget(run_text, header_layout[idx]);
        idx += 1;
    }

    if has_agents {
        let agents_block = Block::default().borders(Borders::ALL).title(" Agents ");
        let agents_text = Paragraph::new(Line::from(vec![Span::styled(
            format!("{running_count} running"),
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )]))
        .block(agents_block);
        frame.render_widget(agents_text, header_layout[idx]);
    }
}

/// Draw the main area: chat pane (left) and tool activity pane (right).
fn draw_main_area(frame: &mut Frame, app: &App, area: Rect) {
    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(area);

    draw_chat_pane(frame, app, horizontal[0]);
    draw_tool_pane(frame, app, horizontal[1]);
}

/// Draw the chat pane showing conversation entries.
fn draw_chat_pane(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title(" Chat ");

    let inner = block.inner(area);

    let mut lines: Vec<Line<'_>> = Vec::new();

    for entry in app.chat_entries() {
        let (prefix, style) = match entry.role {
            tmg_llm::Role::User => (
                "You: ",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            tmg_llm::Role::Assistant => ("Agent: ", Style::default().fg(Color::Blue)),
            tmg_llm::Role::System | tmg_llm::Role::Tool => {
                ("", Style::default().fg(Color::DarkGray))
            }
        };

        // First line with prefix; remaining lines indented to align.
        let mut content_lines = entry.text.split('\n');
        if let Some(first) = content_lines.next() {
            lines.push(Line::from(vec![
                Span::styled(prefix, style),
                Span::styled(first, style),
            ]));
            let indent = " ".repeat(prefix.len());
            for line in content_lines {
                lines.push(Line::from(vec![
                    Span::raw(indent.clone()),
                    Span::styled(line, style),
                ]));
            }
        }

        // Add a blank line between entries.
        lines.push(Line::from(""));
    }

    // Show streaming/thinking indicator.
    if app.is_thinking() {
        lines.push(Line::from(Span::styled(
            "thinking...",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::DIM | Modifier::ITALIC),
        )));
    } else if app.is_streaming() {
        lines.push(Line::from(Span::styled(
            "...",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM),
        )));
    }

    let visible_height = inner.height;
    let wrap_width = inner.width;

    // Calculate the total number of display lines accounting for wrapping.
    // Each logical line wraps to ceil(display_width / wrap_width) lines,
    // with empty lines counting as 1.
    // NOTE: This is an approximation. ratatui wraps on word boundaries,
    // not at exact character-width boundaries, so the actual rendered
    // line count may differ slightly for wordy content.
    let total_lines = {
        let mut count: u16 = 0;
        for line in &lines {
            let w: u16 = u16::try_from(line.width()).unwrap_or(u16::MAX);
            if w == 0 || wrap_width == 0 {
                count = count.saturating_add(1);
            } else {
                count = count.saturating_add((w.saturating_add(wrap_width - 1)) / wrap_width);
            }
        }
        count
    };

    let chat = Paragraph::new(Text::from(lines))
        .block(block)
        .wrap(Wrap { trim: false });

    // Auto-scroll to bottom when new content arrives.
    // Clamp chat_scroll to max_scroll so the offset cannot drift
    // unboundedly when the user scrolls up many times.
    let max_scroll = total_lines.saturating_sub(visible_height);
    let scroll = max_scroll.saturating_sub(app.chat_scroll().min(max_scroll));

    let chat = chat.scroll((scroll, 0));

    frame.render_widget(chat, area);

    // Scrollbar.
    if total_lines > visible_height {
        let mut scrollbar_state = ScrollbarState::new(usize::from(total_lines))
            .position(usize::from(scroll))
            .viewport_content_length(usize::from(visible_height));
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight);
        frame.render_stateful_widget(scrollbar, area, &mut scrollbar_state);
    }
}

/// Draw the structured Activity Pane (SPEC §7.1 / §7.2).
///
/// Sections, top-to-bottom:
///   1. Run progress  (always present; thicker for harnessed runs)
///   2. Workflow progress  (present only while a workflow is running)
///   3. Subagents  (present only when subagents are active)
///   4. Tool log
///   5. Diff preview  (present after a `file_write`/`file_patch` /
///      `git diff` shell call)
fn draw_tool_pane(frame: &mut Frame, app: &App, area: Rect) {
    let subagent_summaries = app.subagent_summaries();
    let has_subagents = !subagent_summaries.is_empty();
    let has_workflow = app.workflow_progress().is_some();
    let has_diff = app.diff_preview().is_some();
    let progress = app.run_progress();
    let harnessed = progress.is_harnessed();

    // Build adaptive constraints. The numbers below are deliberately
    // empirical: the activity pane sits in the right column of the
    // main area, so each section needs to be thick enough to look
    // settled but not so thick that it pushes the tool log off
    // screen.
    let mut constraints: Vec<Constraint> = Vec::new();
    let mut sections: Vec<Section> = Vec::new();

    // Run progress.
    let run_height: u16 = if harnessed {
        // Header + bar + session line + current + 2 upcoming + borders.
        // Cap at 9 so very small terminals don't lose the tool log.
        9
    } else {
        // Header + 2 lines + borders.
        5
    };
    constraints.push(Constraint::Length(run_height));
    sections.push(Section::RunProgress);

    if has_workflow {
        constraints.push(Constraint::Length(5));
        sections.push(Section::Workflow);
    }
    if has_subagents {
        // Cap subagents to ~30 % of remaining space; 6 lines is enough
        // for ~2 entries.
        constraints.push(Constraint::Length(8));
        sections.push(Section::Subagents);
    }
    // Tool log always gets the leftover space.
    constraints.push(Constraint::Min(5));
    sections.push(Section::ToolLog);

    if has_diff {
        // Diff preview gets a fixed slice at the bottom.
        constraints.push(Constraint::Length(12));
        sections.push(Section::DiffPreview);
    }

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    for (section, chunk) in sections.into_iter().zip(chunks.iter()) {
        match section {
            Section::RunProgress => draw_run_progress(frame, progress, *chunk),
            Section::Workflow => {
                if let Some(wf) = app.workflow_progress() {
                    draw_workflow_progress(frame, wf, *chunk);
                }
            }
            Section::Subagents => draw_subagent_pane(frame, subagent_summaries, *chunk),
            Section::ToolLog => draw_tool_activity(frame, app, *chunk),
            Section::DiffPreview => {
                if let Some(diff) = app.diff_preview() {
                    draw_diff_preview(frame, diff, *chunk);
                }
            }
        }
    }
}

/// Activity-pane section discriminator. Used to pair `Layout`
/// constraints with their renderer in `draw_tool_pane`.
enum Section {
    RunProgress,
    Workflow,
    Subagents,
    ToolLog,
    DiffPreview,
}

/// Draw the run-progress section. Two layouts:
///   * Harnessed: scope label + features `Gauge` + session counter +
///     current/next features.
///   * Ad-hoc: scope label + session/turn line + commit count.
#[expect(
    clippy::too_many_lines,
    reason = "two scope variants share the function; splitting harnessed/ad-hoc into separate fns would obscure the shared `block` setup"
)]
fn draw_run_progress(frame: &mut Frame, progress: &RunProgressSection, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Run progress ");

    if progress.is_harnessed() {
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let gauge_height: u16 = 1;
        let body_height = inner.height.saturating_sub(gauge_height + 2);
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),            // scope label
                Constraint::Length(gauge_height), // gauge
                Constraint::Length(1),            // session line
                Constraint::Min(body_height.max(1)),
            ])
            .split(inner);

        // 1. Scope label.
        let scope_line = Paragraph::new(Line::from(vec![
            Span::styled("scope: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "harnessed",
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        frame.render_widget(scope_line, split[0]);

        // 2. Gauge.
        let total = u16::try_from(progress.features_total).unwrap_or(u16::MAX);
        let done = u16::try_from(progress.features_done).unwrap_or(u16::MAX);
        let ratio = if total == 0 {
            0.0
        } else {
            f64::from(done) / f64::from(total)
        };
        let gauge_label = format!("{done}/{total}");
        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(Color::Green).bg(Color::Black))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(gauge_label);
        frame.render_widget(gauge, split[1]);

        // 3. Session line.
        let session_line = match progress.session_max {
            Some(max) => format!("session {}/{}", progress.session_num, max),
            None => format!("session {}", progress.session_num),
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                session_line,
                Style::default().fg(Color::Yellow),
            ))),
            split[2],
        );

        // 4. Current / upcoming features.
        let mut body: Vec<Line<'_>> = Vec::new();
        if let Some(current) = &progress.current_feature {
            body.push(Line::from(vec![
                Span::styled("current: ", Style::default().fg(Color::DarkGray)),
                Span::styled(current.clone(), Style::default().fg(Color::Cyan)),
            ]));
        }
        if !progress.upcoming_features.is_empty() {
            let upcoming = progress
                .upcoming_features
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            body.push(Line::from(vec![
                Span::styled("next: ", Style::default().fg(Color::DarkGray)),
                Span::styled(upcoming, Style::default().fg(Color::DarkGray)),
            ]));
        }
        if body.is_empty() {
            body.push(Line::from(Span::styled(
                "(features not loaded yet)",
                Style::default().fg(Color::DarkGray),
            )));
        }
        frame.render_widget(
            Paragraph::new(Text::from(body)).wrap(Wrap { trim: true }),
            split[3],
        );
    } else {
        let lines = vec![
            Line::from(vec![
                Span::styled("scope: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "ad-hoc",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    format!("session {}", progress.session_num),
                    Style::default().fg(Color::Yellow),
                ),
                Span::raw(", "),
                Span::styled(
                    format!("{} turns", progress.turns),
                    Style::default().fg(Color::DarkGray),
                ),
            ]),
            Line::from(vec![
                Span::styled("commits: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    progress.commits.to_string(),
                    Style::default().fg(Color::Green),
                ),
            ]),
        ];
        let p = Paragraph::new(Text::from(lines))
            .block(block)
            .wrap(Wrap { trim: true });
        frame.render_widget(p, area);
    }
}

/// Draw the workflow-progress section. A single block with the
/// current step id, optional iteration counter, and elapsed time.
fn draw_workflow_progress(frame: &mut Frame, wf: &WorkflowProgressSection, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Workflow progress ");
    let elapsed = wf.elapsed();
    let elapsed_text = format!(
        "{}.{:01}s",
        elapsed.as_secs(),
        elapsed.subsec_millis() / 100
    );
    let mut lines: Vec<Line<'_>> = Vec::new();
    if !wf.workflow_id.is_empty() {
        lines.push(Line::from(vec![
            Span::styled("workflow: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                wf.workflow_id.clone(),
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
    }
    let step_label = if wf.current_step.is_empty() {
        "(starting)".to_owned()
    } else {
        wf.current_step.clone()
    };
    lines.push(Line::from(vec![
        Span::styled("step: ", Style::default().fg(Color::DarkGray)),
        Span::styled(step_label, Style::default().fg(Color::Cyan)),
    ]));
    if let Some((iter, max)) = wf.iteration {
        lines.push(Line::from(vec![
            Span::styled("loop: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{iter}/{max}"), Style::default().fg(Color::Yellow)),
        ]));
    }
    lines.push(Line::from(vec![
        Span::styled("elapsed: ", Style::default().fg(Color::DarkGray)),
        Span::styled(elapsed_text, Style::default().fg(Color::Yellow)),
    ]));
    let p = Paragraph::new(Text::from(lines))
        .block(block)
        .wrap(Wrap { trim: true });
    frame.render_widget(p, area);
}

/// Draw the diff preview at the bottom of the activity pane.
fn draw_diff_preview(frame: &mut Frame, diff: &DiffPreview, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Diff preview ");
    let inner = block.inner(area);
    let max_lines = inner.height.saturating_sub(1) as usize;
    let lines = diff.render_lines(max_lines);
    let p = Paragraph::new(Text::from(lines))
        .block(block)
        .wrap(Wrap { trim: false });
    frame.render_widget(p, area);
}

/// Draw the tool activity log.
fn draw_tool_activity(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title(" Tools ");

    let activity = app.tool_activity();

    if activity.is_empty() {
        let placeholder_block = block.style(Style::default().fg(Color::DarkGray));
        let placeholder = Paragraph::new(Text::from(vec![
            Line::from(""),
            Line::from(Span::styled(
                "  (no tool activity)",
                Style::default().fg(Color::DarkGray),
            )),
        ]))
        .block(placeholder_block);
        frame.render_widget(placeholder, area);
        return;
    }

    let inner = block.inner(area);
    let mut lines: Vec<Line<'_>> = Vec::new();

    for entry in activity {
        let name_style = if entry.is_error {
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
        } else {
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD)
        };

        let summary_style = if entry.is_error {
            Style::default().fg(Color::Red)
        } else {
            Style::default().fg(Color::DarkGray)
        };

        lines.push(Line::from(vec![Span::styled(
            format!("[{}]", entry.tool_name),
            name_style,
        )]));

        // Wrap the summary across multiple lines if needed.
        for line in entry.summary.lines() {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(line, summary_style),
            ]));
        }

        lines.push(Line::from(""));
    }

    // Auto-scroll to bottom to show latest activity.
    let total_lines = u16::try_from(lines.len()).unwrap_or(u16::MAX);
    let visible_height = inner.height;
    let scroll = total_lines.saturating_sub(visible_height);

    let tool_log = Paragraph::new(Text::from(lines))
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));

    frame.render_widget(tool_log, area);
}

/// Draw the subagent status pane.
fn draw_subagent_pane(frame: &mut Frame, summaries: &[tmg_agents::SubagentSummary], area: Rect) {
    let block = Block::default().borders(Borders::ALL).title(" Subagents ");

    let mut lines: Vec<Line<'_>> = Vec::new();

    for summary in summaries {
        let status_style = match summary.status.label() {
            "running" => Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
            "completed" => Style::default().fg(Color::Cyan),
            "failed" => Style::default().fg(Color::Red),
            "cancelled" => Style::default().fg(Color::Yellow),
            _ => Style::default().fg(Color::DarkGray),
        };

        let task_preview = if summary.task.chars().count() > 40 {
            format!("{}...", truncate_str(&summary.task, 37))
        } else {
            summary.task.clone()
        };

        lines.push(Line::from(vec![
            Span::styled(
                format!("[{}:{}] ", summary.id, summary.agent_name),
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(summary.status.label(), status_style),
        ]));

        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(task_preview, Style::default().fg(Color::DarkGray)),
        ]));

        lines.push(Line::from(""));
    }

    let subagent_log = Paragraph::new(Text::from(lines))
        .block(block)
        .wrap(Wrap { trim: false });

    frame.render_widget(subagent_log, area);
}

/// Draw the input area at the bottom.
fn draw_input_area(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Input (Enter: send, Shift/Alt+Enter: newline) ");

    let input_text = if app.input().is_empty() && !app.is_streaming() {
        Text::from(Span::styled(
            "Type a message...",
            Style::default().fg(Color::DarkGray),
        ))
    } else {
        Text::from(app.input())
    };

    let input_widget = Paragraph::new(input_text)
        .block(block)
        .wrap(Wrap { trim: false });

    frame.render_widget(input_widget, area);

    // Set cursor position if not streaming.
    if !app.is_streaming() {
        let inner = Block::default().borders(Borders::ALL).inner(area);

        // Calculate cursor position within the input text.
        let text_before_cursor = &app.input()[..app.cursor_pos()];
        let cursor_line =
            u16::try_from(text_before_cursor.matches('\n').count()).unwrap_or(u16::MAX);
        let last_newline_pos = text_before_cursor.rfind('\n').map_or(0, |p| p + 1);
        let cursor_col =
            u16::try_from(text_before_cursor[last_newline_pos..].width()).unwrap_or(u16::MAX);

        frame.set_cursor_position((inner.x + cursor_col, inner.y + cursor_line));
    }
}

/// Draw an error overlay at the bottom of the screen.
fn draw_error_overlay(frame: &mut Frame, message: &str, area: Rect) {
    let error_height = 3;
    if area.height < error_height {
        return;
    }

    let error_area = Rect {
        x: area.x + 2,
        y: area.height.saturating_sub(error_height + 1),
        width: area.width.saturating_sub(4),
        height: error_height,
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Error ")
        .style(Style::default().fg(Color::Red));

    let error_text = Paragraph::new(Span::styled(message, Style::default().fg(Color::Red)))
        .block(block)
        .wrap(Wrap { trim: true });

    // Clear the background area first.
    frame.render_widget(
        Block::default().style(Style::default().bg(Color::Black)),
        error_area,
    );
    frame.render_widget(error_text, error_area);
}
