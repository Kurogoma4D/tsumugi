//! TUI rendering: layout and widget drawing.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
};
use unicode_width::UnicodeWidthStr;

use tmg_agents::truncate_str;

use crate::app::App;

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

    // Draw error overlay if present.
    if let Some(err) = app.error_message() {
        draw_error_overlay(frame, err, size);
    }
}

/// Draw the header bar: model name, context usage, optional run id,
/// and subagent count.
fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let running_count = app.running_subagent_count();
    let has_run = app.current_run().is_some();
    let has_agents = running_count > 0;

    // Build constraints dynamically: Model, Context are always present;
    // Run is added when a run is active; Agents when subagents are running.
    let mut constraints: Vec<Constraint> = Vec::with_capacity(4);
    let total_panes: u16 = 2 + u16::from(has_run) + u16::from(has_agents);
    let pane_pct: u16 = 100 / total_panes;
    for _ in 0..total_panes {
        constraints.push(Constraint::Percentage(pane_pct));
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

    if let Some(run) = app.current_run() {
        let run_block = Block::default().borders(Borders::ALL).title(" Run ");
        let run_text = Paragraph::new(Line::from(vec![
            Span::styled(
                run.short_id(),
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(run.scope_label, Style::default().fg(Color::DarkGray)),
        ]))
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

/// Draw the tool activity pane showing tool call logs and subagent status.
fn draw_tool_pane(frame: &mut Frame, app: &App, area: Rect) {
    let subagent_summaries = app.subagent_summaries();
    let has_subagents = !subagent_summaries.is_empty();

    if has_subagents {
        // Split the tool pane vertically: tool activity on top, subagent
        // status on bottom.
        let vertical = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(area);

        draw_tool_activity(frame, app, vertical[0]);
        draw_subagent_pane(frame, subagent_summaries, vertical[1]);
    } else {
        draw_tool_activity(frame, app, area);
    }
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
