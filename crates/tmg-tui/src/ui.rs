//! TUI rendering: layout and widget drawing.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
};

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

/// Draw the header bar: model name and context usage.
fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let header_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let model_block = Block::default().borders(Borders::ALL).title(" Model ");
    let model_text = Paragraph::new(Line::from(vec![Span::styled(
        app.model_name(),
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    )]))
    .block(model_block);
    frame.render_widget(model_text, header_layout[0]);

    let context_block = Block::default().borders(Borders::ALL).title(" Context ");
    let context_text = Paragraph::new(Line::from(vec![Span::styled(
        app.context_usage(),
        Style::default().fg(Color::Yellow),
    )]))
    .block(context_block);
    frame.render_widget(context_text, header_layout[1]);
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

    // Show streaming indicator.
    if app.is_streaming() {
        lines.push(Line::from(Span::styled(
            "...",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM),
        )));
    }

    // Saturate to u16::MAX if line count exceeds display capacity.
    let total_lines = u16::try_from(lines.len()).unwrap_or(u16::MAX);
    let visible_height = inner.height;

    // Auto-scroll to bottom when new content arrives.
    let scroll = if total_lines > visible_height {
        total_lines
            .saturating_sub(visible_height)
            .saturating_sub(app.chat_scroll())
    } else {
        0
    };

    let chat = Paragraph::new(Text::from(lines))
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));

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

/// Draw the tool activity pane showing tool call logs.
fn draw_tool_pane(frame: &mut Frame, app: &App, area: Rect) {
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

/// Draw the input area at the bottom.
fn draw_input_area(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Input (Ctrl+Enter to send) ");

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
        let cursor_col = u16::try_from(text_before_cursor[last_newline_pos..].chars().count())
            .unwrap_or(u16::MAX);

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
