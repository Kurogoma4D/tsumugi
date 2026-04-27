//! Unified-diff parsing and syntax-highlighted preview rendering.
//!
//! This module is consumed by the structured Activity Pane (issue #45)
//! to display the most recent diff produced by a `file_write` /
//! `file_patch` tool call, or extracted from a `git diff` shell run.
//!
//! ## Trade-offs
//!
//! - **Parsing**: a hand-rolled minimal unified-diff parser. We do
//!   *not* attempt to reproduce a full RFC-quality diff parser
//!   (perfectly handling `\ No newline at end of file`, binary diffs,
//!   rename headers, mode changes, etc.). The intent is a best-effort
//!   preview — anything that fails to parse is shown as plain text
//!   without hunk grouping.
//! - **Highlighting**: we load `syntect`'s default-fancy syntax /
//!   theme set lazily via `OnceLock`. Loading is one-shot per process
//!   so the first preview pays a small cost (~30 ms locally) and
//!   subsequent renders are fast. We tolerate any highlight failure
//!   by falling back to the raw line.
//! - **Display**: the renderer produces `ratatui::text::Line`s that
//!   the Activity Pane stitches into a `Paragraph`. Highlighting is
//!   applied per *line* (not per token of a multi-line construct) so
//!   long blocks (raw strings, multiline comments) may lose context;
//!   this matches `bat`-style line-by-line presentation and is fine
//!   for diff previews where each line stands on its own.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use syntect::easy::HighlightLines;
use syntect::highlighting::{Style as SyntectStyle, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

/// A single diff hunk — a contiguous range of `@@ -a,b +c,d @@` lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffHunk {
    /// Original-side starting line (1-indexed).
    pub old_start: u32,
    /// Original-side line count.
    pub old_lines: u32,
    /// New-side starting line (1-indexed).
    pub new_start: u32,
    /// New-side line count.
    pub new_lines: u32,
    /// Hunk header trailing context (text after the closing `@@`).
    pub header_context: String,
    /// Lines belonging to this hunk, including their leading `+`/`-`/` ` marker.
    pub lines: Vec<DiffLine>,
}

/// One line inside a hunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffLine {
    /// `+`, `-`, or ` ` (context). Other markers (`\\`, `\`, etc.)
    /// are normalised to context.
    pub kind: DiffLineKind,
    /// The content *without* the leading marker character.
    pub content: String,
}

/// Marker for [`DiffLine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffLineKind {
    /// Added (`+`) line.
    Added,
    /// Removed (`-`) line.
    Removed,
    /// Context (` `) line.
    Context,
}

/// Parsed unified diff.
///
/// We retain only the *first* file in the diff — the activity pane
/// renders one file at a time; multi-file diffs are not interesting in
/// the preview surface and would push more material than the pane has
/// vertical room for.
///
/// ## Render cache
///
/// [`Self::render_lines`] memoises its output keyed on `max_lines`. A
/// `DiffPreview` is replaced wholesale whenever a new diff arrives, so
/// the cache lifetime is bounded by the struct's own lifetime — the
/// data fields (`file` / `hunks`) are never mutated in place.
/// `max_lines` can change across renders (when the activity pane
/// resizes), so the cache key tracks it and recomputes on miss.
///
/// `Clone`, `PartialEq`, and `Eq` are intentionally *not* derived: the
/// `Mutex<Option<...>>` cache cell is neither cloneable in a meaningful
/// way nor comparable. Callers that previously needed those traits
/// should compare on `(file, hunks)` directly.
#[derive(Debug)]
pub struct DiffPreview {
    /// File path inferred from the `+++ b/<path>` header (or `--- a/<path>` if absent).
    pub file: PathBuf,
    /// Hunks belonging to this file.
    pub hunks: Vec<DiffHunk>,
    /// Cached output of [`Self::render_lines`] keyed by `max_lines`.
    /// `Mutex` rather than `OnceLock` because the `max_lines` cap can
    /// change between renders (terminal resize), so the cache must
    /// support invalidation; a single `OnceLock` would either pin the
    /// first observed cap forever or require a wholesale struct
    /// replacement on every resize. Contention is irrelevant — only
    /// the render path acquires this lock and only briefly.
    cached_lines: Mutex<Option<(usize, Vec<Line<'static>>)>>,
}

impl Default for DiffPreview {
    fn default() -> Self {
        Self {
            file: PathBuf::new(),
            hunks: Vec::new(),
            cached_lines: Mutex::new(None),
        }
    }
}

impl DiffPreview {
    /// Parse a unified diff. Returns `None` if no recognisable hunk is
    /// found.
    ///
    /// The parser is intentionally permissive: anything before the
    /// first `--- ` line is ignored, and a stray line inside a hunk
    /// that is not `+`/`-`/` ` ends the hunk.
    #[must_use]
    pub fn parse(diff_text: &str) -> Option<Self> {
        parse_unified_diff(diff_text)
    }

    /// Build a `DiffPreview` from already-parsed hunks. The cache cell
    /// is initialised empty; callers that previously used the struct
    /// literal directly should switch to this constructor so the
    /// private cache field stays encapsulated.
    #[must_use]
    pub fn with_hunks(file: impl Into<PathBuf>, hunks: Vec<DiffHunk>) -> Self {
        Self {
            file: file.into(),
            hunks,
            cached_lines: Mutex::new(None),
        }
    }

    /// Synthesise a single-hunk preview from a `file_write` tool call.
    ///
    /// `file_write` does not produce a real unified diff — we frame
    /// the entire new content as a single "+" hunk so the preview
    /// surface still shows something meaningful for green-field file
    /// creation.
    #[must_use]
    pub fn from_file_write(path: impl Into<PathBuf>, content: &str) -> Self {
        let lines: Vec<DiffLine> = content
            .lines()
            .map(|l| DiffLine {
                kind: DiffLineKind::Added,
                content: l.to_owned(),
            })
            .collect();
        let new_lines = u32::try_from(lines.len()).unwrap_or(u32::MAX);
        Self {
            file: path.into(),
            hunks: vec![DiffHunk {
                old_start: 0,
                old_lines: 0,
                new_start: 1,
                new_lines,
                header_context: String::new(),
                lines,
            }],
            cached_lines: Mutex::new(None),
        }
    }

    /// Total number of `+`/`-` lines across all hunks. Useful for the
    /// Activity Pane to decide whether to truncate.
    #[must_use]
    pub fn change_count(&self) -> usize {
        self.hunks
            .iter()
            .flat_map(|h| h.lines.iter())
            .filter(|l| !matches!(l.kind, DiffLineKind::Context))
            .count()
    }

    /// Render the diff preview to `ratatui::text::Line`s with
    /// `syntect`-based syntax highlighting applied per line.
    ///
    /// `max_lines` caps the number of body lines (not counting the
    /// file header / hunk headers); excess lines are truncated and
    /// replaced with a single "... (N more lines)" indicator.
    ///
    /// The result is memoised on `(self, max_lines)`; the activity
    /// pane re-renders every tick, so caching avoids paying the
    /// per-line `syntect` highlight cost on every frame. The cache
    /// invalidates whenever `max_lines` changes (e.g. terminal
    /// resize); the underlying diff data is never mutated in place.
    #[must_use]
    pub fn render_lines(&self, max_lines: usize) -> Vec<Line<'static>> {
        // Fast path: return a clone of the cached lines if the cap
        // matches. We deliberately clone (rather than handing out a
        // shared `Arc<Vec<Line>>`) so callers can keep using the
        // existing `Vec<Line<'static>>` API — the per-frame clone is
        // O(N) on a vector that has already been built without any
        // syntect work.
        if let Ok(guard) = self.cached_lines.lock() {
            if let Some((cached_max, lines)) = guard.as_ref() {
                if *cached_max == max_lines {
                    return lines.clone();
                }
            }
        }

        let lines = self.render_lines_uncached(max_lines);

        // Refresh the cache. A poisoned mutex is non-fatal — it just
        // means a previous panic happened during cache update; we
        // skip caching this round and the caller still gets the
        // freshly-rendered lines.
        if let Ok(mut guard) = self.cached_lines.lock() {
            *guard = Some((max_lines, lines.clone()));
        }
        lines
    }

    fn render_lines_uncached(&self, max_lines: usize) -> Vec<Line<'static>> {
        let mut out: Vec<Line<'static>> = Vec::new();

        // File header.
        let display_path = self.file.display().to_string();
        out.push(Line::from(Span::styled(
            format!("--- {display_path}"),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));

        let highlighter = SyntaxBundle::get();
        let extension = self
            .file
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or("");

        let mut body_count: usize = 0;
        let mut truncated_extra: usize = 0;

        for hunk in &self.hunks {
            // Hunk header.
            let header = format!(
                "@@ -{},{} +{},{} @@{}",
                hunk.old_start,
                hunk.old_lines,
                hunk.new_start,
                hunk.new_lines,
                if hunk.header_context.is_empty() {
                    String::new()
                } else {
                    format!(" {}", hunk.header_context)
                },
            );
            out.push(Line::from(Span::styled(
                header,
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            )));

            for line in &hunk.lines {
                if body_count >= max_lines {
                    truncated_extra += 1;
                    continue;
                }
                body_count += 1;

                let (marker, base_style) = match line.kind {
                    DiffLineKind::Added => ("+", Style::default().fg(Color::Green)),
                    DiffLineKind::Removed => ("-", Style::default().fg(Color::Red)),
                    DiffLineKind::Context => (" ", Style::default().fg(Color::DarkGray)),
                };

                let mut spans: Vec<Span<'static>> =
                    vec![Span::styled(marker.to_string(), base_style)];

                if let Some(highlighted) = highlighter.highlight_line(extension, &line.content) {
                    for (style, text) in highlighted {
                        spans.push(blend_span(text, style, line.kind));
                    }
                } else {
                    spans.push(Span::styled(line.content.clone(), base_style));
                }

                out.push(Line::from(spans));
            }
        }

        if truncated_extra > 0 {
            out.push(Line::from(Span::styled(
                format!("... ({truncated_extra} more lines)"),
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            )));
        }

        out
    }
}

/// Wrapper around `syntect`'s `SyntaxSet` and `ThemeSet`. Lazily
/// initialised on first use.
pub(crate) struct SyntaxBundle {
    syntaxes: SyntaxSet,
    theme: syntect::highlighting::Theme,
}

impl SyntaxBundle {
    /// Return the process-wide bundle, initialising it on first call.
    pub(crate) fn get() -> &'static Self {
        static BUNDLE: OnceLock<SyntaxBundle> = OnceLock::new();
        BUNDLE.get_or_init(|| {
            let syntaxes = SyntaxSet::load_defaults_newlines();
            let themes = ThemeSet::load_defaults();
            let theme = themes
                .themes
                .get("base16-ocean.dark")
                .cloned()
                .unwrap_or_else(|| {
                    // base16-ocean.dark ships with default-fancy; the
                    // fallback exists so a future syntect upgrade that
                    // renames the default theme does not regress to a
                    // panic. Picking *any* available theme is preferable
                    // to crashing the TUI on first render.
                    themes
                        .themes
                        .values()
                        .next()
                        .cloned()
                        .unwrap_or_else(syntect::highlighting::Theme::default)
                });
            Self { syntaxes, theme }
        })
    }

    /// Highlight a single line of text. Returns `None` when no syntax
    /// matches the extension or the highlighter errors out.
    fn highlight_line(&self, extension: &str, line: &str) -> Option<Vec<(SyntectStyle, String)>> {
        let syntax = if extension.is_empty() {
            None
        } else {
            self.syntaxes.find_syntax_by_extension(extension)
        }?;
        let mut highlighter = HighlightLines::new(syntax, &self.theme);
        let mut out: Vec<(SyntectStyle, String)> = Vec::new();
        // `highlight_line` expects a trailing newline to keep state
        // sane between calls; we feed the whole "line" through
        // `LinesWithEndings` so single-line inputs get the right
        // handling.
        for sub in LinesWithEndings::from(line) {
            match highlighter.highlight_line(sub, &self.syntaxes) {
                Ok(ranges) => {
                    for (st, text) in ranges {
                        out.push((st, text.trim_end_matches('\n').to_owned()));
                    }
                }
                Err(_) => return None,
            }
        }
        Some(out)
    }
}

/// Convert a syntect range into a ratatui `Span`, blending the diff
/// kind's base background hint with the syntax-highlight foreground.
fn blend_span(text: String, st: SyntectStyle, kind: DiffLineKind) -> Span<'static> {
    let fg = Color::Rgb(st.foreground.r, st.foreground.g, st.foreground.b);
    let style = match kind {
        DiffLineKind::Added => Style::default().fg(fg).add_modifier(Modifier::DIM).fg(fg),
        DiffLineKind::Removed => Style::default().fg(fg),
        DiffLineKind::Context => Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::DIM),
    };
    Span::styled(text, style)
}

/// Parse a unified diff into a `DiffPreview`.
fn parse_unified_diff(text: &str) -> Option<DiffPreview> {
    let mut current_path: Option<PathBuf> = None;
    let mut hunks: Vec<DiffHunk> = Vec::new();
    let mut current_hunk: Option<DiffHunk> = None;

    for line in text.lines() {
        if let Some(path) = parse_minus_header(line) {
            // First file header. If a `+++` follows we'll prefer it,
            // but record this as a fallback.
            if current_path.is_none() {
                current_path = Some(path);
            }
            continue;
        }
        if let Some(path) = parse_plus_header(line) {
            current_path = Some(path);
            continue;
        }
        if let Some(parsed_hunk) = parse_hunk_header(line) {
            if let Some(h) = current_hunk.take() {
                hunks.push(h);
            }
            current_hunk = Some(parsed_hunk);
            continue;
        }
        if let Some(h) = current_hunk.as_mut() {
            // Inside a hunk; classify the line.
            if let Some(rest) = line.strip_prefix('+') {
                h.lines.push(DiffLine {
                    kind: DiffLineKind::Added,
                    content: rest.to_owned(),
                });
            } else if let Some(rest) = line.strip_prefix('-') {
                h.lines.push(DiffLine {
                    kind: DiffLineKind::Removed,
                    content: rest.to_owned(),
                });
            } else if let Some(rest) = line.strip_prefix(' ') {
                h.lines.push(DiffLine {
                    kind: DiffLineKind::Context,
                    content: rest.to_owned(),
                });
            } else if line.starts_with('\\') {
                // "\ No newline at end of file" — ignore.
            } else {
                // End of hunk on an unexpected marker; commit and
                // restart the search for a new `@@`.
                if let Some(h) = current_hunk.take() {
                    hunks.push(h);
                }
            }
        }
    }

    if let Some(h) = current_hunk.take() {
        hunks.push(h);
    }

    if hunks.is_empty() {
        return None;
    }

    let file = current_path.unwrap_or_else(|| PathBuf::from("(unknown)"));
    Some(DiffPreview {
        file,
        hunks,
        cached_lines: Mutex::new(None),
    })
}

fn parse_minus_header(line: &str) -> Option<PathBuf> {
    let rest = line.strip_prefix("--- ")?;
    let cleaned = rest.strip_prefix("a/").unwrap_or(rest);
    if cleaned == "/dev/null" {
        return None;
    }
    Some(PathBuf::from(cleaned.trim()))
}

fn parse_plus_header(line: &str) -> Option<PathBuf> {
    let rest = line.strip_prefix("+++ ")?;
    let cleaned = rest.strip_prefix("b/").unwrap_or(rest);
    if cleaned == "/dev/null" {
        return None;
    }
    Some(PathBuf::from(cleaned.trim()))
}

fn parse_hunk_header(line: &str) -> Option<DiffHunk> {
    let body = line.strip_prefix("@@ ")?;
    let close_idx = body.find(" @@")?;
    let head = &body[..close_idx];
    let tail = &body[close_idx + 3..];

    let mut parts = head.split_whitespace();
    let old_part = parts.next()?.strip_prefix('-')?;
    let new_part = parts.next()?.strip_prefix('+')?;

    let (old_start, old_lines) = parse_range(old_part)?;
    let (new_start, new_lines) = parse_range(new_part)?;

    Some(DiffHunk {
        old_start,
        old_lines,
        new_start,
        new_lines,
        header_context: tail.trim().to_owned(),
        lines: Vec::new(),
    })
}

fn parse_range(s: &str) -> Option<(u32, u32)> {
    let mut iter = s.split(',');
    let start = iter.next()?.parse::<u32>().ok()?;
    let count = iter.next().map_or(Some(1), |v| v.parse::<u32>().ok())?;
    Some((start, count))
}

/// Pre-warm the process-wide [`SyntaxBundle`].
///
/// The bundle's first-use load is ~30 ms; deferring it to a background
/// blocking task at TUI startup keeps that cost off the rendering
/// thread so the first diff preview frame doesn't stall. Subsequent
/// calls to [`SyntaxBundle::get`] are O(1) — `OnceLock` resolves once.
pub fn prewarm_syntax_bundle() {
    // Deliberately no `&'static`-binding so the compiler can drop the
    // reference; the `OnceLock` inside `get` keeps the bundle alive.
    let _ = SyntaxBundle::get();
}

/// Try to extract a unified diff out of a shell-tool result whose
/// command was `git diff` (or similar). The result body is fed
/// directly to [`DiffPreview::parse`]; this thin wrapper exists so
/// callers can document intent.
#[must_use]
pub fn try_extract_from_shell_output(output: &str) -> Option<DiffPreview> {
    DiffPreview::parse(output)
}

#[expect(
    clippy::expect_used,
    clippy::format_push_string,
    reason = "tests use expect/format!+push_str for clarity; the workspace policy denies them in production code"
)]
#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,3 +1,4 @@ fn main()
 use std::io;
-fn old() {}
+fn new() {}
+fn extra() {}
";

    #[test]
    fn parses_simple_unified_diff() {
        let preview = DiffPreview::parse(SAMPLE).expect("parse");
        assert_eq!(preview.file, PathBuf::from("src/lib.rs"));
        assert_eq!(preview.hunks.len(), 1);
        let hunk = &preview.hunks[0];
        assert_eq!(hunk.old_start, 1);
        assert_eq!(hunk.old_lines, 3);
        assert_eq!(hunk.new_start, 1);
        assert_eq!(hunk.new_lines, 4);
        assert_eq!(hunk.header_context, "fn main()");
        assert_eq!(hunk.lines.len(), 4);
        assert_eq!(hunk.lines[0].kind, DiffLineKind::Context);
        assert_eq!(hunk.lines[1].kind, DiffLineKind::Removed);
        assert_eq!(hunk.lines[2].kind, DiffLineKind::Added);
        assert_eq!(hunk.lines[3].kind, DiffLineKind::Added);
    }

    #[test]
    fn render_produces_lines_with_header() {
        let preview = DiffPreview::parse(SAMPLE).expect("parse");
        let rendered = preview.render_lines(100);
        assert!(!rendered.is_empty());
        let first = format!("{:?}", rendered[0]);
        assert!(first.contains("src/lib.rs"));
    }

    #[test]
    fn change_count_excludes_context() {
        let preview = DiffPreview::parse(SAMPLE).expect("parse");
        assert_eq!(preview.change_count(), 3); // 1 removed + 2 added
    }

    #[test]
    fn parse_returns_none_on_garbage() {
        assert!(DiffPreview::parse("hello world").is_none());
        assert!(DiffPreview::parse("").is_none());
    }

    #[test]
    fn from_file_write_synthesises_added_hunk() {
        let preview = DiffPreview::from_file_write("foo.rs", "fn x() {}\nfn y() {}");
        assert_eq!(preview.hunks.len(), 1);
        assert_eq!(preview.hunks[0].lines.len(), 2);
        assert!(
            preview.hunks[0]
                .lines
                .iter()
                .all(|l| matches!(l.kind, DiffLineKind::Added)),
        );
    }

    #[test]
    fn render_truncates_when_over_max() {
        let mut content = String::new();
        for i in 0..50 {
            content.push_str(&format!("line {i}\n"));
        }
        let preview = DiffPreview::from_file_write("big.txt", &content);
        let rendered = preview.render_lines(10);
        // 1 file header + 1 hunk header + 10 body lines + 1 truncation indicator.
        assert_eq!(rendered.len(), 13);
    }

    #[test]
    fn highlight_unknown_extension_falls_back() {
        let preview = DiffPreview::from_file_write("foo.unknownext", "blah\n");
        let rendered = preview.render_lines(100);
        // Should render without panic.
        assert!(!rendered.is_empty());
    }

    #[test]
    fn parse_handles_dev_null_minus_header() {
        let diff = "\
--- /dev/null
+++ b/new.rs
@@ -0,0 +1,1 @@
+hello
";
        let preview = DiffPreview::parse(diff).expect("parse");
        assert_eq!(preview.file, PathBuf::from("new.rs"));
    }

    #[test]
    fn parse_range_no_count_defaults_to_one() {
        assert_eq!(parse_range("5"), Some((5, 1)));
        assert_eq!(parse_range("5,3"), Some((5, 3)));
    }

    #[test]
    fn render_lines_caches_when_max_lines_unchanged() {
        let preview = DiffPreview::parse(SAMPLE).expect("parse");
        // Cache miss on the first call.
        let first = preview.render_lines(50);
        // Verify the cache is populated.
        {
            let guard = preview.cached_lines.lock().expect("lock");
            let (cached_max, cached_lines) = guard.as_ref().expect("cache populated");
            assert_eq!(*cached_max, 50);
            assert_eq!(cached_lines.len(), first.len());
        }
        // Second call must return an equivalent vector.
        let second = preview.render_lines(50);
        assert_eq!(format!("{first:?}"), format!("{second:?}"));
    }

    #[test]
    fn render_lines_invalidates_when_max_lines_changes() {
        let preview = DiffPreview::parse(SAMPLE).expect("parse");
        let first = preview.render_lines(2);
        let second = preview.render_lines(50);
        // Different caps must produce different cached vectors.
        assert_ne!(first.len(), second.len());
        let guard = preview.cached_lines.lock().expect("lock");
        let (cached_max, _) = guard.as_ref().expect("cache populated");
        assert_eq!(*cached_max, 50);
    }
}
