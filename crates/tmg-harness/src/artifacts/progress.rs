//! `progress.md` writer/reader.
//!
//! [`ProgressLog`] manages the append-only `progress.md` file inside a
//! run directory (SPEC §9.6). The file is initialised with a single
//! `# Progress Log` header at run creation time, then accumulates one
//! `## Session #N (timestamp) [scope]` block per session, with
//! `progress_append`-emitted `- {entry}` lines beneath each header.
//!
//! ## Append-only guarantee
//!
//! The public API intentionally exposes no overwrite or rewrite
//! operation; all write methods open the file with
//! [`OpenOptions::append`](std::fs::OpenOptions::append) so concurrent
//! appends from a single process are interleaved at the byte level
//! rather than truncating each other. The `init` constructor is the
//! single exception: it creates the file when it does not yet exist
//! and writes the initial header. Existing content is preserved.

use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};

use crate::error::HarnessError;
use crate::run::RunScope;

/// Append-only writer/reader for a run's `progress.md`.
///
/// Cloning a [`ProgressLog`] is cheap (path-only) and produces another
/// handle to the same file. There is no in-memory buffering: every
/// append immediately writes through to disk so a crash mid-session
/// does not lose the entries written before the crash.
#[derive(Debug, Clone)]
pub struct ProgressLog {
    path: PathBuf,
}

impl ProgressLog {
    /// Create a handle for the file at `path` without touching disk.
    ///
    /// Use [`init`](Self::init) to also write the initial header when
    /// the file does not yet exist.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Initialise `progress.md` at `path` if missing.
    ///
    /// Writes the SPEC §9.6 boilerplate `# Progress Log\n` only when
    /// the file does not already exist; existing content is left
    /// untouched so resuming a run preserves prior history.
    ///
    /// Creates parent directories as needed.
    pub fn init(path: impl Into<PathBuf>) -> Result<Self, HarnessError> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| HarnessError::io(parent, e))?;
        }
        if !path.exists() {
            // `create_new` so a concurrent process cannot have us
            // truncate a non-empty file.
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut f) => {
                    f.write_all(b"# Progress Log\n")
                        .map_err(|e| HarnessError::io(&path, e))?;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    // Another process beat us to it; leave its content alone.
                }
                Err(e) => return Err(HarnessError::io(&path, e)),
            }
        }
        Ok(Self { path })
    }

    /// Borrow the on-disk path of this progress log.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append a `## Session #N (timestamp) [scope]` header.
    ///
    /// `ts` is rendered in RFC 3339 / ISO 8601 form with second
    /// precision so the output stays human-readable.
    pub fn append_session_header(
        &self,
        session_num: u32,
        scope: &RunScope,
        ts: DateTime<Utc>,
    ) -> Result<(), HarnessError> {
        let ts_str = ts.format("%Y-%m-%dT%H:%M:%SZ");
        let scope_label = scope.label();
        // Leading newline keeps a blank line between sessions when the
        // file is non-empty; the very first session inherits a blank
        // line after the `# Progress Log` initial header.
        let line = format!("\n## Session #{session_num} ({ts_str}) [{scope_label}]\n");
        self.append_raw(line.as_bytes())
    }

    /// Append a single bullet `- {line}\n` under the current session.
    ///
    /// `line` is taken verbatim except for stripping trailing newlines.
    /// Multi-line entries are collapsed into one bullet by escaping
    /// embedded newlines as ` / ` so the markdown stays well-formed.
    pub fn append_entry(&self, line: &str) -> Result<(), HarnessError> {
        let trimmed = line.trim_end_matches(['\n', '\r']);
        let one_line = trimmed.replace('\n', " / ").replace('\r', " ");
        let formatted = format!("- {one_line}\n");
        self.append_raw(formatted.as_bytes())
    }

    /// Append a multi-line block verbatim, preserving its internal
    /// newlines.
    ///
    /// Used by the auto-promotion path
    /// ([`crate::runner::RunRunner::escalate_to_harnessed`]) to write a
    /// `## Session #N (...) [SCOPE UPGRADE]` block consisting of a
    /// header line plus a few `- key: value` bullets. The shape of the
    /// block is owned by the caller; this helper only guarantees the
    /// append-only / through-disk semantics shared by every other
    /// mutator on this type.
    ///
    /// Visibility is intentionally `pub(crate)`: the auto-promotion
    /// runner is the only consumer today, and the API contract is
    /// loose enough that exposing it more broadly would invite
    /// free-form writers that defeat the structured shape of
    /// `progress.md`.
    pub(crate) fn append_raw_block(&self, block: &str) -> Result<(), HarnessError> {
        self.append_raw(block.as_bytes())
    }

    /// Read the entire file as a string, returning the empty string
    /// when the file does not yet exist.
    ///
    /// Used by the auto-promotion path to test for an existing
    /// `[SCOPE UPGRADE]` marker before re-appending the block — so a
    /// retry after a partial first attempt does not double-write.
    pub(crate) fn read_all(&self) -> Result<String, HarnessError> {
        match std::fs::read_to_string(&self.path) {
            Ok(c) => Ok(c),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
            Err(e) => Err(HarnessError::io(&self.path, e)),
        }
    }

    /// Read the last `n` session blocks (header + entries) and return
    /// them concatenated, newest-first.
    ///
    /// Returns an empty string when the file does not exist or contains
    /// no sessions yet.
    pub fn read_recent_sessions(&self, n: usize) -> Result<String, HarnessError> {
        if n == 0 {
            return Ok(String::new());
        }
        let content = match std::fs::read_to_string(&self.path) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(String::new()),
            Err(e) => return Err(HarnessError::io(&self.path, e)),
        };

        // Find every `## Session ` header offset.
        let starts: Vec<usize> = content
            .match_indices("\n## Session ")
            .map(|(i, _)| i + 1)
            .chain(if content.starts_with("## Session ") {
                Some(0)
            } else {
                None
            })
            .collect();

        if starts.is_empty() {
            return Ok(String::new());
        }

        let mut starts = starts;
        starts.sort_unstable();
        starts.dedup();

        let take_from = starts.len().saturating_sub(n);
        let start_offset = starts[take_from];
        Ok(content[start_offset..].to_owned())
    }

    /// Helper: append raw bytes to the file in append mode.
    ///
    /// Always opens with `append(true)`; we intentionally do **not**
    /// expose any overwrite path. Callers that need to inspect the
    /// file should use [`read_recent_sessions`](Self::read_recent_sessions)
    /// or read it directly through the [`path`](Self::path) accessor.
    fn append_raw(&self, bytes: &[u8]) -> Result<(), HarnessError> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| HarnessError::io(parent, e))?;
        }
        let mut f = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.path)
            .map_err(|e| HarnessError::io(&self.path, e))?;
        f.write_all(bytes)
            .map_err(|e| HarnessError::io(&self.path, e))?;
        f.flush().map_err(|e| HarnessError::io(&self.path, e))?;
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;

    fn tmp_log() -> (tempfile::TempDir, ProgressLog) {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let path = dir.path().join("progress.md");
        let log = ProgressLog::init(&path).unwrap_or_else(|e| panic!("{e}"));
        (dir, log)
    }

    #[test]
    fn init_writes_header_when_missing() {
        let (_dir, log) = tmp_log();
        let content = std::fs::read_to_string(log.path()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(content, "# Progress Log\n");
    }

    #[test]
    fn init_preserves_existing_content() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let path = dir.path().join("progress.md");
        std::fs::write(
            &path,
            "# Progress Log\n\n## Session #1 (...) [ad-hoc]\n- entry\n",
        )
        .unwrap_or_else(|e| panic!("{e}"));

        let _log = ProgressLog::init(&path).unwrap_or_else(|e| panic!("{e}"));
        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
        assert!(content.contains("- entry"));
    }

    #[test]
    fn append_session_header_includes_index_scope() {
        let (_dir, log) = tmp_log();
        let ts = chrono::DateTime::parse_from_rfc3339("2025-01-02T03:04:05Z")
            .unwrap_or_else(|e| panic!("{e}"))
            .with_timezone(&Utc);
        log.append_session_header(7, &RunScope::AdHoc, ts)
            .unwrap_or_else(|e| panic!("{e}"));
        let content = std::fs::read_to_string(log.path()).unwrap_or_else(|e| panic!("{e}"));
        assert!(content.contains("## Session #7 (2025-01-02T03:04:05Z) [ad-hoc]"));
    }

    #[test]
    fn append_entry_writes_bullet() {
        let (_dir, log) = tmp_log();
        log.append_entry("did the thing")
            .unwrap_or_else(|e| panic!("{e}"));
        let content = std::fs::read_to_string(log.path()).unwrap_or_else(|e| panic!("{e}"));
        assert!(content.ends_with("- did the thing\n"), "{content}");
    }

    #[test]
    fn append_entry_collapses_newlines() {
        let (_dir, log) = tmp_log();
        log.append_entry("line one\nline two")
            .unwrap_or_else(|e| panic!("{e}"));
        let content = std::fs::read_to_string(log.path()).unwrap_or_else(|e| panic!("{e}"));
        assert!(content.contains("- line one / line two\n"), "{content}");
    }

    #[test]
    fn read_recent_sessions_returns_last_n() {
        let (_dir, log) = tmp_log();
        let ts = Utc::now();
        for i in 1..=4 {
            log.append_session_header(i, &RunScope::AdHoc, ts)
                .unwrap_or_else(|e| panic!("{e}"));
            log.append_entry(&format!("entry {i}"))
                .unwrap_or_else(|e| panic!("{e}"));
        }
        let recent = log
            .read_recent_sessions(2)
            .unwrap_or_else(|e| panic!("{e}"));
        // Last two: #3 and #4.
        assert!(recent.contains("Session #3"), "{recent}");
        assert!(recent.contains("Session #4"), "{recent}");
        assert!(!recent.contains("Session #1"), "{recent}");
        assert!(!recent.contains("Session #2"), "{recent}");
    }

    #[test]
    fn read_recent_sessions_handles_missing_file() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let log = ProgressLog::new(dir.path().join("missing.md"));
        let out = log
            .read_recent_sessions(3)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(out.is_empty());
    }

    /// Disk-level overwrite verification: every public mutator opens
    /// the file in append mode, so the initial header byte is never
    /// rewritten. We check this by appending a bunch of entries and
    /// confirming the file's leading bytes still match `# Progress
    /// Log\n`.
    #[test]
    fn append_only_at_disk_level_preserves_header() {
        let (_dir, log) = tmp_log();
        for i in 0..50 {
            log.append_entry(&format!("e{i}"))
                .unwrap_or_else(|e| panic!("{e}"));
        }
        let content = std::fs::read_to_string(log.path()).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            content.starts_with("# Progress Log\n"),
            "header was rewritten: {content}"
        );
        // And the count of entries is exactly what we wrote.
        let entry_count = content.matches("\n- e").count();
        assert_eq!(entry_count, 50);
    }
}
