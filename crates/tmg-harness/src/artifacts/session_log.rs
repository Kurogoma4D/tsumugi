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

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::HarnessError;
use crate::session::{Session, SessionEndTrigger};

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
    ///
    /// The scan also consults the compacted summary aggregate written
    /// by [`compress_old_sessions`](Self::compress_old_sessions): when
    /// the live `session_NNN.json` files do not carry a hint (e.g. all
    /// recent sessions ended without explicitly setting one), the
    /// aggregate is parsed and its newest non-empty hint is returned.
    ///
    /// **Cost:** `O(N)` worst case in the number of live
    /// `session_NNN.json` files: the implementation calls
    /// [`Self::list`] once and then `load`s the individual session
    /// files newest-first until a non-empty hint is found. When every
    /// recent session lacks a hint, the entire live set is read off
    /// disk and the compacted aggregate is parsed once on top.
    /// Optimisation (e.g. caching the hint pointer in `run.toml` or
    /// a small index file) is tracked as a follow-up.
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

        // Fall back to the compacted aggregate: walk its entries
        // newest-first so the most recent persisted hint wins.
        let aggregate_path = self.summary_aggregate_path();
        if !aggregate_path.exists() {
            return Ok(None);
        }
        let raw = fs::read_to_string(&aggregate_path)
            .map_err(|e| HarnessError::io(&aggregate_path, e))?;
        let aggregate: SessionSummaryAggregate = serde_json::from_str(&raw)
            .map_err(|e| HarnessError::session_deserialize(&aggregate_path, e))?;
        for entry in aggregate.entries.iter().rev() {
            if let Some(hint) = entry.next_session_hint.as_ref()
                && !hint.trim().is_empty()
            {
                return Ok(Some(hint.clone()));
            }
        }
        Ok(None)
    }

    /// Compute the path of the compacted summary aggregate file.
    ///
    /// The filename is the constant
    /// [`SUMMARY_AGGREGATE_FILENAME`](self::SUMMARY_AGGREGATE_FILENAME)
    /// (`session_summaries.json`); the choice of a single fixed name
    /// keeps merging across multiple compaction passes simple — every
    /// new pass merges into the same file rather than producing one
    /// `session_001-019.summary.json` per compaction.
    #[must_use]
    pub fn summary_aggregate_path(&self) -> PathBuf {
        self.dir.join(SUMMARY_AGGREGATE_FILENAME)
    }

    /// Compact every session older than the most recent `keep_recent`
    /// into a single [`SessionSummaryAggregate`] JSON file at
    /// [`Self::summary_aggregate_path`], then delete the originals.
    ///
    /// **Mechanical-only:** this is purely text manipulation; no LLM
    /// is consulted. Each compacted entry inherits the source
    /// session's `summary`, `session_number`, `started_at`,
    /// `ended_at`, `trigger`, and `next_session_hint` verbatim. SPEC
    /// §9.11 leaves richer aggregations to a future issue.
    ///
    /// **Atomicity:** the aggregate is written via a `*.tmp` + rename
    /// before any of the source `session_NNN.json` files are removed.
    /// If the rename succeeds but a subsequent unlink fails, the
    /// aggregate is consistent and the leftover live files merely
    /// duplicate data; a re-run of `compress_old_sessions` cleans
    /// them up.
    ///
    /// **Idempotency:** repeated calls re-merge the live files into
    /// the existing aggregate; entries already in the aggregate are
    /// preserved (we union by `session_number`, with the live file
    /// winning on a duplicate so a hand-edited
    /// `session_NNN.json` overrides a stale aggregate entry).
    ///
    /// `keep_recent == 0` compacts every available session.
    /// `keep_recent` greater than the available session count is a
    /// no-op.
    ///
    /// # Errors
    ///
    /// - [`HarnessError::Io`] for read/write failures of the source or
    ///   aggregate files.
    /// - [`HarnessError::SessionDeserialize`] when a live
    ///   `session_NNN.json` cannot be parsed.
    /// - [`HarnessError::SessionSerialize`] when the aggregate cannot
    ///   be serialized.
    pub fn compress_old_sessions(&self, keep_recent: usize) -> Result<(), HarnessError> {
        let entries = self.list()?;
        if entries.len() <= keep_recent {
            return Ok(());
        }
        let cutoff = entries.len() - keep_recent;
        let to_compact: Vec<&SessionLogEntry> = entries.iter().take(cutoff).collect();
        if to_compact.is_empty() {
            return Ok(());
        }

        // Load existing aggregate if present so a subsequent compaction
        // pass merges into it rather than overwriting earlier entries.
        let aggregate_path = self.summary_aggregate_path();
        let mut aggregate = if aggregate_path.exists() {
            let raw = fs::read_to_string(&aggregate_path)
                .map_err(|e| HarnessError::io(&aggregate_path, e))?;
            serde_json::from_str::<SessionSummaryAggregate>(&raw)
                .map_err(|e| HarnessError::session_deserialize(&aggregate_path, e))?
        } else {
            SessionSummaryAggregate::default()
        };

        // Build the new entries first so an early failure leaves both
        // the live files and the aggregate untouched.
        let mut new_entries = Vec::with_capacity(to_compact.len());
        for entry in &to_compact {
            let session = self.load(entry.index)?.ok_or_else(|| HarnessError::Io {
                path: entry.path.clone(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "session listed by SessionLog::list disappeared before load",
                ),
            })?;
            new_entries.push(SessionSummaryEntry::from_session(&session));
        }

        // Merge: live entries overwrite any pre-existing aggregate
        // entry for the same `session_number`, then we sort.
        for incoming in new_entries {
            if let Some(slot) = aggregate
                .entries
                .iter_mut()
                .find(|e| e.session_number == incoming.session_number)
            {
                *slot = incoming;
            } else {
                aggregate.entries.push(incoming);
            }
        }
        aggregate.entries.sort_by_key(|e| e.session_number);

        // Atomic write of the aggregate BEFORE deleting any source.
        fs::create_dir_all(&self.dir).map_err(|e| HarnessError::io(&self.dir, e))?;
        let tmp_path = aggregate_path.with_extension("json.tmp");
        let serialized = serde_json::to_string_pretty(&aggregate)
            .map_err(|e| HarnessError::session_serialize(&aggregate_path, e))?;
        if let Err(e) = fs::write(&tmp_path, serialized) {
            let _ = fs::remove_file(&tmp_path);
            return Err(HarnessError::io(&tmp_path, e));
        }
        if let Err(e) = fs::rename(&tmp_path, &aggregate_path) {
            let _ = fs::remove_file(&tmp_path);
            return Err(HarnessError::io(&aggregate_path, e));
        }

        // Now safe to delete originals.
        for entry in &to_compact {
            if let Err(e) = fs::remove_file(&entry.path)
                && e.kind() != std::io::ErrorKind::NotFound
            {
                tracing::warn!(
                    file = %entry.path.display(),
                    %e,
                    "failed to remove compacted session file; aggregate is durable",
                );
            }
        }

        Ok(())
    }
}

/// On-disk filename for the compacted summary aggregate produced by
/// [`SessionLog::compress_old_sessions`].
pub const SUMMARY_AGGREGATE_FILENAME: &str = "session_summaries.json";

/// Aggregate file written by
/// [`SessionLog::compress_old_sessions`].
///
/// Each entry preserves the source session's identity
/// (`session_number`, `started_at`, `ended_at`, `trigger`) plus the
/// LLM-or-fallback `summary` text and the optional
/// `next_session_hint`. The aggregate is a flat array sorted by
/// `session_number` ascending.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionSummaryAggregate {
    /// Compacted session entries, sorted by `session_number`
    /// ascending.
    #[serde(default)]
    pub entries: Vec<SessionSummaryEntry>,
}

/// One compacted session entry inside a
/// [`SessionSummaryAggregate`].
///
/// **Field-loss notice:** [`Self::from_session`] preserves only the
/// SPEC §9.11 minimum (identity + summary + hint). The following
/// [`Session`] fields are intentionally **not** carried into the
/// aggregate:
///
/// - [`Session::tool_calls_count`]
/// - [`Session::files_modified`]
/// - [`Session::features_marked_passing`]
/// - [`Session::context_usage_peak`]
///
/// Once the live `session_NNN.json` is removed by
/// [`SessionLog::compress_old_sessions`] these values are no longer
/// recoverable. **TODO:** include in a future SPEC §9.11 enhancement
/// when consumers (e.g. analytics dashboards or post-mortem tooling)
/// need them. Until then, callers that require these fields must
/// snapshot them before compaction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionSummaryEntry {
    /// Session sequence number within the parent run.
    pub session_number: u32,

    /// When the session started.
    pub started_at: DateTime<Utc>,

    /// When the session ended, if it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<DateTime<Utc>>,

    /// Why the session ended, if known.
    #[serde(default, rename = "trigger", skip_serializing_if = "Option::is_none")]
    pub end_trigger: Option<SessionEndTrigger>,

    /// Free-form summary (`Session::summary`).
    #[serde(default)]
    pub summary: String,

    /// Optional hint that the next session's bootstrap can consume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_session_hint: Option<String>,
}

impl SessionSummaryEntry {
    /// Build an entry from a [`Session`], copying the SPEC §9.11
    /// fields verbatim.
    #[must_use]
    pub fn from_session(session: &Session) -> Self {
        Self {
            session_number: session.index,
            started_at: session.started_at,
            ended_at: session.ended_at,
            end_trigger: session.end_trigger.clone(),
            summary: session.summary.clone(),
            next_session_hint: session.next_session_hint.clone(),
        }
    }
}

/// Parse `session_NNN.json` -> `Some(NNN)`, otherwise `None`.
///
/// The compacted `session_summaries.json` aggregate produced by
/// [`SessionLog::compress_old_sessions`] does not match this shape and
/// is filtered out implicitly: its stem
/// (`session_summaries`) does not start with `session_<digits>`.
fn parse_session_filename(name: &str) -> Option<u32> {
    if name == SUMMARY_AGGREGATE_FILENAME {
        return None;
    }
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

    /// Acceptance: 25 sessions with `keep_recent = 20` aggregates
    /// sessions 1..=5 into the summary file and leaves 6..=25 as
    /// individual JSONs.
    #[test]
    fn compress_old_sessions_aggregates_oldest_keeps_recent() {
        let (_tmp, log) = tmp_log();
        for i in 1..=25_u32 {
            let mut s = Session::begin(i);
            s.summary = format!("summary {i}");
            s.next_session_hint = Some(format!("hint {i}"));
            s.end(SessionEndTrigger::Completed);
            log.save(&s).unwrap_or_else(|e| panic!("{e}"));
        }

        log.compress_old_sessions(20)
            .unwrap_or_else(|e| panic!("{e}"));

        // Live files: 6..=25 remain.
        let listed = log.list().unwrap_or_else(|e| panic!("{e}"));
        let live_indices: Vec<u32> = listed.iter().map(|e| e.index).collect();
        assert_eq!(live_indices, (6..=25).collect::<Vec<_>>());

        // Sessions 1..=5 must be gone from disk.
        for i in 1..=5_u32 {
            assert!(
                !log.session_path(i).exists(),
                "session {i} should be compacted: {}",
                log.session_path(i).display(),
            );
        }

        // Aggregate file exists and carries the expected entries.
        let aggregate_path = log.summary_aggregate_path();
        assert!(aggregate_path.exists());
        let raw = fs::read_to_string(&aggregate_path).unwrap_or_else(|e| panic!("{e}"));
        let aggregate: SessionSummaryAggregate =
            serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{e}"));
        let aggregate_indices: Vec<u32> =
            aggregate.entries.iter().map(|e| e.session_number).collect();
        assert_eq!(aggregate_indices, vec![1, 2, 3, 4, 5]);
        assert_eq!(aggregate.entries[0].summary, "summary 1");
        assert_eq!(aggregate.entries[4].summary, "summary 5");
        assert_eq!(
            aggregate.entries[2].next_session_hint.as_deref(),
            Some("hint 3"),
        );
    }

    /// Compacting fewer-than-`keep_recent` sessions is a no-op.
    #[test]
    fn compress_old_sessions_no_op_when_below_keep_recent() {
        let (_tmp, log) = tmp_log();
        for i in 1..=5_u32 {
            let mut s = Session::begin(i);
            s.end(SessionEndTrigger::Completed);
            log.save(&s).unwrap_or_else(|e| panic!("{e}"));
        }
        log.compress_old_sessions(20)
            .unwrap_or_else(|e| panic!("{e}"));
        let listed = log.list().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(listed.len(), 5);
        assert!(!log.summary_aggregate_path().exists());
    }

    /// Repeated compaction passes merge new entries into the existing
    /// aggregate rather than overwriting it.
    #[test]
    fn compress_old_sessions_merges_into_existing_aggregate() {
        let (_tmp, log) = tmp_log();
        for i in 1..=10_u32 {
            let mut s = Session::begin(i);
            s.summary = format!("first {i}");
            s.end(SessionEndTrigger::Completed);
            log.save(&s).unwrap_or_else(|e| panic!("{e}"));
        }
        // First pass: keep 5 recent -> 1..=5 compacted.
        log.compress_old_sessions(5)
            .unwrap_or_else(|e| panic!("{e}"));

        // Add more sessions and run another pass. Now we keep_recent=3
        // out of the 5 surviving, so 6, 7 also get compacted.
        for i in 11..=15_u32 {
            let mut s = Session::begin(i);
            s.summary = format!("second {i}");
            s.end(SessionEndTrigger::Completed);
            log.save(&s).unwrap_or_else(|e| panic!("{e}"));
        }
        // Now 10 live files (6..=15). keep_recent=3 -> 7 oldest get
        // compacted, leaving 13..=15.
        log.compress_old_sessions(3)
            .unwrap_or_else(|e| panic!("{e}"));

        let listed = log.list().unwrap_or_else(|e| panic!("{e}"));
        let live_indices: Vec<u32> = listed.iter().map(|e| e.index).collect();
        assert_eq!(live_indices, vec![13, 14, 15]);

        let raw =
            fs::read_to_string(log.summary_aggregate_path()).unwrap_or_else(|e| panic!("{e}"));
        let aggregate: SessionSummaryAggregate =
            serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{e}"));
        let agg_indices: Vec<u32> = aggregate.entries.iter().map(|e| e.session_number).collect();
        assert_eq!(agg_indices, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }

    /// `last_hint` falls back to the compacted aggregate when no live
    /// session carries a hint.
    #[test]
    fn last_hint_falls_back_to_aggregate() {
        let (_tmp, log) = tmp_log();

        // Three sessions, only the oldest carries a hint.
        let mut s1 = Session::begin(1);
        s1.next_session_hint = Some("aggregate hint".to_owned());
        s1.end(SessionEndTrigger::Completed);
        log.save(&s1).unwrap_or_else(|e| panic!("{e}"));
        for i in 2..=3_u32 {
            let mut s = Session::begin(i);
            s.end(SessionEndTrigger::Completed);
            log.save(&s).unwrap_or_else(|e| panic!("{e}"));
        }

        // Compact the oldest into the aggregate.
        log.compress_old_sessions(2)
            .unwrap_or_else(|e| panic!("{e}"));

        // No live session has a hint; the aggregate still does.
        let hint = log.last_hint().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(hint.as_deref(), Some("aggregate hint"));
    }

    /// `compress_old_sessions(0)` compacts every session.
    #[test]
    fn compress_old_sessions_zero_keep_recent_compacts_all() {
        let (_tmp, log) = tmp_log();
        for i in 1..=4_u32 {
            let mut s = Session::begin(i);
            s.end(SessionEndTrigger::Completed);
            log.save(&s).unwrap_or_else(|e| panic!("{e}"));
        }
        log.compress_old_sessions(0)
            .unwrap_or_else(|e| panic!("{e}"));
        let listed = log.list().unwrap_or_else(|e| panic!("{e}"));
        assert!(listed.is_empty());
        assert!(log.summary_aggregate_path().exists());
    }
}
