//! `session_log/session_NNN.json` writer/reader.
//!
//! [`SessionLog`] manages the directory of per-session JSON records
//! described in SPEC §9.4. One file per session is created with a
//! zero-padded three-digit index (`session_001.json`, `session_002.json`,
//! ...). The on-disk layout is:
//!
//! ```text
//! .tsumugi/runs/<run-id>/session_log/
//!     session_001.json
//!     session_002.json
//!     ...
//! ```
//!
//! Lookups (`list`, `last_hint`) iterate the directory and sort by
//! sequence number; sessions that fail to parse are logged via
//! `tracing::warn!` and skipped, mirroring the policy in
//! [`RunStore::list`](crate::store::RunStore::list).

use std::fs;
use std::path::{Path, PathBuf};

use crate::error::HarnessError;
use crate::session::Session;

/// Persistent store for `session_NNN.json` records under one run.
///
/// Cloning is cheap (path-only) and produces another handle to the
/// same directory.
#[derive(Debug, Clone)]
pub struct SessionLog {
    dir: PathBuf,
}

/// Lightweight summary entry returned by [`SessionLog::list`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionLogEntry {
    /// Session sequence number (matches `session.index`).
    pub index: u32,
    /// Path to the JSON file on disk.
    pub path: PathBuf,
}

impl SessionLog {
    /// Create a handle for the directory at `dir` without touching disk.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Borrow the on-disk directory path.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Compute the path for a session by sequence number.
    #[must_use]
    pub fn session_path(&self, index: u32) -> PathBuf {
        self.dir.join(format!("session_{index:03}.json"))
    }

    /// Persist `session` to `session_<index>.json`.
    ///
    /// Writes via a `*.tmp` + rename for atomicity, matching
    /// [`RunStore::save`](crate::store::RunStore::save). Creates the
    /// session log directory on first call.
    pub fn save(&self, session: &Session) -> Result<(), HarnessError> {
        fs::create_dir_all(&self.dir).map_err(|e| HarnessError::io(&self.dir, e))?;

        let path = self.session_path(session.index);
        let tmp_path = path.with_extension("json.tmp");
        let serialized = serde_json::to_string_pretty(session)
            .map_err(|e| HarnessError::session_serialize(&path, e))?;

        if let Err(e) = fs::write(&tmp_path, serialized) {
            let _ = fs::remove_file(&tmp_path);
            return Err(HarnessError::io(&tmp_path, e));
        }
        if let Err(e) = fs::rename(&tmp_path, &path) {
            let _ = fs::remove_file(&tmp_path);
            return Err(HarnessError::io(&path, e));
        }
        Ok(())
    }

    /// List every recorded session, sorted by `index` ascending.
    ///
    /// Files that fail to parse, or whose name does not match the
    /// `session_NNN.json` shape, are skipped with a `tracing::warn!`.
    pub fn list(&self) -> Result<Vec<SessionLogEntry>, HarnessError> {
        if !self.dir.exists() {
            return Ok(Vec::new());
        }

        let read_dir = fs::read_dir(&self.dir).map_err(|e| HarnessError::io(&self.dir, e))?;
        let mut entries = Vec::new();
        for dirent in read_dir {
            let dirent = dirent.map_err(|e| HarnessError::io(&self.dir, e))?;
            let path = dirent.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(index) = parse_session_filename(name) else {
                tracing::warn!(
                    file = %path.display(),
                    "skipping unrecognised session log file"
                );
                continue;
            };
            entries.push(SessionLogEntry { index, path });
        }
        entries.sort_by_key(|e| e.index);
        Ok(entries)
    }

    /// Load the [`Session`] record for the given `index`, if present.
    pub fn load(&self, index: u32) -> Result<Option<Session>, HarnessError> {
        let path = self.session_path(index);
        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(HarnessError::io(&path, e)),
        };
        let session: Session = serde_json::from_str(&content)
            .map_err(|e| HarnessError::session_deserialize(&path, e))?;
        Ok(Some(session))
    }

    /// Return the most recent `next_session_hint`, scanning sessions
    /// from the highest index downwards.
    ///
    /// Returns `Ok(None)` when no session has set a hint yet.
    pub fn last_hint(&self) -> Result<Option<String>, HarnessError> {
        let mut entries = self.list()?;
        entries.sort_by_key(|e| std::cmp::Reverse(e.index));
        for entry in entries {
            if let Some(session) = self.load(entry.index)?
                && let Some(hint) = session.next_session_hint
                && !hint.trim().is_empty()
            {
                return Ok(Some(hint));
            }
        }
        Ok(None)
    }
}

/// Parse `session_NNN.json` -> `Some(NNN)`, otherwise `None`.
fn parse_session_filename(name: &str) -> Option<u32> {
    let stem = name.strip_suffix(".json")?;
    let num = stem.strip_prefix("session_")?;
    num.parse::<u32>().ok()
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use crate::session::SessionEndTrigger;

    fn tmp_log() -> (tempfile::TempDir, SessionLog) {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let dir = tmp.path().join("session_log");
        (tmp, SessionLog::new(dir))
    }

    #[test]
    fn save_writes_session_file_with_padded_index() {
        let (_tmp, log) = tmp_log();
        let mut session = Session::begin(7);
        session.summary = "did stuff".to_owned();
        session.tool_calls_count = 3;
        session.end(SessionEndTrigger::Completed);
        log.save(&session).unwrap_or_else(|e| panic!("{e}"));

        let path = log.dir().join("session_007.json");
        assert!(path.exists());
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
        // Schema sanity: SPEC §9.4 keys.
        assert!(raw.contains("\"session_number\": 7"), "{raw}");
        assert!(raw.contains("\"trigger\""), "{raw}");
        assert!(raw.contains("\"summary\": \"did stuff\""), "{raw}");
    }

    #[test]
    fn save_round_trips_through_serde() {
        let (_tmp, log) = tmp_log();
        let mut original = Session::begin(2);
        original.summary = "round trip".to_owned();
        original.tool_calls_count = 9;
        original.files_modified.push("a.rs".to_owned());
        original.context_usage_peak = 0.51;
        original.next_session_hint = Some("look at b.rs".to_owned());
        original.end(SessionEndTrigger::Completed);
        log.save(&original).unwrap_or_else(|e| panic!("{e}"));

        let loaded = log
            .load(2)
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_else(|| panic!("session 2 should exist"));
        assert_eq!(loaded, original);
    }

    #[test]
    fn list_returns_sessions_sorted_by_index() {
        let (_tmp, log) = tmp_log();
        for i in [3, 1, 2_u32] {
            let mut s = Session::begin(i);
            s.end(SessionEndTrigger::Completed);
            log.save(&s).unwrap_or_else(|e| panic!("{e}"));
        }
        let listed = log.list().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            listed.iter().map(|e| e.index).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn list_returns_empty_when_dir_missing() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let log = SessionLog::new(dir.path().join("nope"));
        let listed = log.list().unwrap_or_else(|e| panic!("{e}"));
        assert!(listed.is_empty());
    }

    #[test]
    fn last_hint_returns_most_recent_non_empty() {
        let (_tmp, log) = tmp_log();

        let mut s1 = Session::begin(1);
        s1.next_session_hint = Some("hint one".to_owned());
        s1.end(SessionEndTrigger::Completed);
        log.save(&s1).unwrap_or_else(|e| panic!("{e}"));

        let mut s2 = Session::begin(2);
        s2.next_session_hint = Some("hint two".to_owned());
        s2.end(SessionEndTrigger::Completed);
        log.save(&s2).unwrap_or_else(|e| panic!("{e}"));

        // s3 has no hint: last_hint should still walk back to s2.
        let mut s3 = Session::begin(3);
        s3.end(SessionEndTrigger::Completed);
        log.save(&s3).unwrap_or_else(|e| panic!("{e}"));

        let hint = log.last_hint().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(hint.as_deref(), Some("hint two"));
    }

    #[test]
    fn last_hint_returns_none_when_no_sessions() {
        let (_tmp, log) = tmp_log();
        let hint = log.last_hint().unwrap_or_else(|e| panic!("{e}"));
        assert!(hint.is_none());
    }
}
