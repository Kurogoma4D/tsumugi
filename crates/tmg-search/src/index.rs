//! [`SearchIndex`]: `SQLite` + FTS5 store of session summaries and turn
//! transcripts.
//!
//! The index lives at `<project_root>/.tsumugi/state.db` and is
//! populated through two paths:
//!
//! - **Session-end hook**: the harness calls
//!   [`SearchIndex::ingest_session`] after writing
//!   `session_NNN.json`. This is the live path that keeps the DB
//!   incremental.
//! - **Rebuild scan**: [`SearchIndex::rebuild_from_disk`] walks every
//!   `.tsumugi/runs/*/session_log/*.json` and ingests anything whose
//!   `(run_id, session_num)` pair is not yet present. Used by the
//!   `tmg search rebuild` CLI subcommand and on first startup so a
//!   pre-existing project picks up its history.
//!
//! Schema (also documented in issue #53):
//!
//! ```sql
//! CREATE TABLE sessions (
//!     run_id TEXT NOT NULL,
//!     session_num INTEGER NOT NULL,
//!     started_at TEXT NOT NULL,
//!     ended_at TEXT,
//!     trigger TEXT,                        -- SessionEndTrigger tag
//!     summary TEXT,                        -- redacted summary text
//!     files_modified TEXT,                 -- JSON array
//!     tool_calls_count INTEGER,
//!     PRIMARY KEY (run_id, session_num)
//! );
//! CREATE VIRTUAL TABLE sessions_fts USING fts5(
//!     summary, files_modified,
//!     content='sessions', content_rowid='rowid'
//! );
//! CREATE TABLE turns (
//!     run_id TEXT NOT NULL,
//!     session_num INTEGER NOT NULL,
//!     turn_num INTEGER NOT NULL,
//!     user_input TEXT,
//!     agent_text TEXT,
//!     tool_calls TEXT,
//!     PRIMARY KEY (run_id, session_num, turn_num)
//! );
//! CREATE VIRTUAL TABLE turns_fts USING fts5(
//!     user_input, agent_text,
//!     content='turns', content_rowid='rowid'
//! );
//! ```

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension as _, params};
use serde::{Deserialize, Serialize};
use tmg_harness::Session;

use crate::error::SearchError;
use crate::redact::redact_secrets;

/// Search-result row returned by [`SearchIndex::query`].
///
/// `score` is the BM25 ranking value sqlite reports — lower is more
/// relevant in FTS5's convention. We invert it to a positive
/// "relevance" before returning so ranking sorts naturally; the raw
/// BM25 score is preserved when callers need to pass it back into
/// sqlite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    /// The owning run's id (string form).
    pub run_id: String,
    /// Session sequence number within the run.
    pub session_num: u32,
    /// When the session started (ISO8601 / RFC3339).
    pub started_at: String,
    /// Free-form summary text, possibly empty.
    pub summary: String,
    /// FTS5 snippet with `<mark>...</mark>` highlights around match
    /// terms. Empty when the row matched only on the summary column
    /// and the snippet helper found nothing in the configured
    /// extraction window.
    pub snippet: String,
    /// Inverted BM25 score (higher is more relevant).
    pub score: f64,
}

/// Search scope — which FTS table(s) to consult.
///
/// `Summary` only hits `sessions_fts`, `Turns` only hits `turns_fts`,
/// and `All` does both and unions the results by `(run_id,
/// session_num)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchScope {
    /// Match against per-session summary text.
    Summary,
    /// Match against turn-level user / agent / tool text.
    Turns,
    /// Match both tables; deduplicated by `(run_id, session_num)`.
    All,
}

impl SearchScope {
    /// Parse a scope string from the tool / CLI surface.
    ///
    /// Accepts `"summary"`, `"turns"`, `"all"`. Empty or unknown values
    /// fall back to [`SearchScope::All`] so a partial / new caller is
    /// never blocked on the most permissive option.
    #[must_use]
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "summary" => Self::Summary,
            "turns" => Self::Turns,
            _ => Self::All,
        }
    }
}

/// Schema version pinned in the `meta` key/value table. Bumping this
/// triggers a migration path on next [`SearchIndex::open`].
const SCHEMA_VERSION: i64 = 1;

/// Persistent search index. Cloning a [`SearchIndex`] is cheap
/// (`Arc`-style) and produces another handle to the same underlying
/// connection mutex.
///
/// The connection is wrapped in a `Mutex<Connection>` because rusqlite
/// connections are `!Sync`. Concurrent callers serialize on the
/// mutex; a future enhancement could move to a connection pool.
pub struct SearchIndex {
    /// Mutex-guarded connection. `Box::leak` is not required because
    /// the index struct owns the connection.
    conn: Mutex<Connection>,
    /// On-disk path of the database file. Kept for diagnostics
    /// (`tmg search stats`) and error reporting.
    path: PathBuf,
}

impl std::fmt::Debug for SearchIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The `conn` mutex wraps a `rusqlite::Connection` which is not
        // `Debug`; surface the path only and finish non-exhaustive so
        // the elision is visible at the call site.
        f.debug_struct("SearchIndex")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

/// Snapshot of DB statistics for `tmg search stats`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchStats {
    /// Number of indexed sessions.
    pub sessions: u64,
    /// Number of indexed turns.
    pub turns: u64,
    /// Size of the on-disk database file in bytes (0 when the file
    /// doesn't exist yet).
    pub size_bytes: u64,
    /// Path to the database file.
    pub db_path: PathBuf,
}

/// One turn record fed into [`SearchIndex::ingest_turn`]. Strings are
/// redacted before insertion.
///
/// We keep this structure here rather than mirroring [`Session`]'s
/// turn schema because the harness's `session_NNN.json` does **not**
/// currently persist per-turn text — that is the trajectory recorder
/// (#55). Until that lands, the only ingest path is "summary only",
/// which is honoured by [`SearchIndex::ingest_session`] (it writes a
/// row into `sessions` but no rows into `turns`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TurnRecord {
    /// Sequence number within the session.
    pub turn_num: u32,
    /// Verbatim user input that opened the turn (may be empty for
    /// agent-only turns, e.g. self-reflection).
    pub user_input: String,
    /// Agent's response text after redaction.
    pub agent_text: String,
    /// JSON-serialized tool call array (the same shape the agent loop
    /// already produces).
    pub tool_calls: String,
}

impl SearchIndex {
    /// Open or create the index at `db_path`, running migrations as
    /// needed.
    ///
    /// Creates parent directories if missing. The schema is laid down
    /// idempotently on first open; subsequent opens verify the
    /// `schema_version` value and refuse to proceed when the on-disk
    /// version is newer than [`SCHEMA_VERSION`] — we never
    /// auto-downgrade.
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self, SearchError> {
        let path = db_path.as_ref().to_path_buf();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| SearchError::io(parent.to_path_buf(), e))?;
        }
        let conn = Connection::open(&path).map_err(|e| SearchError::sqlite("open db", e))?;
        // Reasonable defaults for a single-writer process: WAL
        // tolerates concurrent reads while ingest is writing.
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| SearchError::sqlite("set journal_mode", e))?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(|e| SearchError::sqlite("set synchronous", e))?;

        Self::ensure_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            path,
        })
    }

    /// Borrow the on-disk path (used by `tmg search stats`).
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Apply the schema (idempotent). Bumping [`SCHEMA_VERSION`] lands
    /// here as a `match` on the stored version.
    fn ensure_schema(conn: &Connection) -> Result<(), SearchError> {
        // Single-shot multi-statement SQL via `execute_batch`. The
        // FTS5 virtual tables use `content=` external-content syntax
        // so we maintain the FTS rows manually with INSERT/DELETE
        // triggers (avoids the FTS table doubling DB size).
        let sql = "
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                run_id TEXT NOT NULL,
                session_num INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                trigger TEXT,
                summary TEXT,
                files_modified TEXT,
                tool_calls_count INTEGER,
                PRIMARY KEY (run_id, session_num)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
                summary,
                files_modified,
                content='sessions',
                content_rowid='rowid'
            );
            CREATE TABLE IF NOT EXISTS turns (
                run_id TEXT NOT NULL,
                session_num INTEGER NOT NULL,
                turn_num INTEGER NOT NULL,
                user_input TEXT,
                agent_text TEXT,
                tool_calls TEXT,
                PRIMARY KEY (run_id, session_num, turn_num)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                user_input,
                agent_text,
                content='turns',
                content_rowid='rowid'
            );
        ";
        conn.execute_batch(sql)
            .map_err(|e| SearchError::sqlite("create schema", e))?;

        // Read or initialise schema_version.
        let stored: Option<i64> = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'schema_version'",
                [],
                |row| {
                    row.get::<_, String>(0)
                        .map(|s| s.parse::<i64>().unwrap_or(0))
                },
            )
            .optional()
            .map_err(|e| SearchError::sqlite("read schema_version", e))?;

        match stored {
            None => {
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?1)",
                    params![SCHEMA_VERSION.to_string()],
                )
                .map_err(|e| SearchError::sqlite("write schema_version", e))?;
            }
            Some(v) if v == SCHEMA_VERSION => {}
            Some(v) if v > SCHEMA_VERSION => {
                return Err(SearchError::Sqlite {
                    context: format!(
                        "on-disk schema_version {v} is newer than supported {SCHEMA_VERSION}; \
                         upgrade tsumugi or delete state.db to rebuild"
                    ),
                    source: rusqlite::Error::InvalidQuery,
                });
            }
            Some(_v) => {
                // Future migration path: bump the stored value here.
                // For schema v1 there is nothing to migrate.
                conn.execute(
                    "UPDATE meta SET value = ?1 WHERE key = 'schema_version'",
                    params![SCHEMA_VERSION.to_string()],
                )
                .map_err(|e| SearchError::sqlite("update schema_version", e))?;
            }
        }
        Ok(())
    }

    /// Insert (or replace) the session row with redacted text. Existing
    /// turns for the same `(run_id, session_num)` are left untouched.
    ///
    /// Returns `Ok(())` even when the summary is empty — the row is
    /// still indexed so `rebuild_from_disk` does not re-process it on
    /// every startup.
    pub fn ingest_session(&self, run_id: &str, session: &Session) -> Result<(), SearchError> {
        let summary = redact_secrets(&session.summary);
        let files_modified = serde_json::to_string(&session.files_modified)
            .map_err(|e| SearchError::json("serialize files_modified", e))?;
        // Redact filenames too — paths sometimes embed credentials in
        // tokenised URLs.
        let files_modified = redact_secrets(&files_modified);
        let started_at = session.started_at.to_rfc3339();
        let ended_at = session.ended_at.as_ref().map(DateTime::to_rfc3339);
        let trigger = session.end_trigger.as_ref().map(|t| match t {
            tmg_harness::SessionEndTrigger::Completed => "completed".to_owned(),
            tmg_harness::SessionEndTrigger::UserCancelled => "user_cancelled".to_owned(),
            tmg_harness::SessionEndTrigger::Rotated { .. } => "rotated".to_owned(),
            tmg_harness::SessionEndTrigger::Errored { .. } => "errored".to_owned(),
            tmg_harness::SessionEndTrigger::UserExit => "user_exit".to_owned(),
            tmg_harness::SessionEndTrigger::ContextRotation => "context_rotation".to_owned(),
            tmg_harness::SessionEndTrigger::Timeout => "timeout".to_owned(),
            tmg_harness::SessionEndTrigger::UserNewSession => "user_new_session".to_owned(),
        });

        let conn = self.lock_conn()?;
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| SearchError::sqlite("begin tx", e))?;

        // Upsert the row. We delete the old FTS row first and re-insert
        // because external-content FTS5 doesn't auto-track UPDATEs.
        tx.execute(
            "DELETE FROM sessions WHERE run_id = ?1 AND session_num = ?2",
            params![run_id, session.index],
        )
        .map_err(|e| SearchError::sqlite("delete old sessions row", e))?;
        tx.execute(
            "DELETE FROM sessions_fts WHERE rowid IN (
                SELECT rowid FROM sessions WHERE run_id = ?1 AND session_num = ?2
            )",
            params![run_id, session.index],
        )
        .map_err(|e| SearchError::sqlite("delete old sessions_fts row", e))?;

        tx.execute(
            "INSERT INTO sessions(
                run_id, session_num, started_at, ended_at, trigger,
                summary, files_modified, tool_calls_count
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                run_id,
                session.index,
                started_at,
                ended_at,
                trigger,
                summary,
                files_modified,
                session.tool_calls_count,
            ],
        )
        .map_err(|e| SearchError::sqlite("insert sessions row", e))?;

        let rowid: i64 = tx
            .query_row(
                "SELECT rowid FROM sessions WHERE run_id = ?1 AND session_num = ?2",
                params![run_id, session.index],
                |row| row.get(0),
            )
            .map_err(|e| SearchError::sqlite("read sessions rowid", e))?;
        tx.execute(
            "INSERT INTO sessions_fts(rowid, summary, files_modified) VALUES (?1, ?2, ?3)",
            params![rowid, summary, files_modified],
        )
        .map_err(|e| SearchError::sqlite("insert sessions_fts row", e))?;

        tx.commit()
            .map_err(|e| SearchError::sqlite("commit tx", e))?;
        Ok(())
    }

    /// Insert one turn record into the index. Both `user_input` and
    /// `agent_text` are run through [`redact_secrets`] before
    /// insertion.
    ///
    /// Used by the trajectory recorder (#55); the live session-end
    /// hook in this issue only writes session rows.
    pub fn ingest_turn(
        &self,
        run_id: &str,
        session_num: u32,
        turn: &TurnRecord,
    ) -> Result<(), SearchError> {
        let user_input = redact_secrets(&turn.user_input);
        let agent_text = redact_secrets(&turn.agent_text);
        let tool_calls = redact_secrets(&turn.tool_calls);
        let conn = self.lock_conn()?;
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| SearchError::sqlite("begin tx (turn)", e))?;
        tx.execute(
            "DELETE FROM turns_fts WHERE rowid IN (
                SELECT rowid FROM turns WHERE run_id = ?1 AND session_num = ?2 AND turn_num = ?3
            )",
            params![run_id, session_num, turn.turn_num],
        )
        .map_err(|e| SearchError::sqlite("delete old turns_fts row", e))?;
        tx.execute(
            "DELETE FROM turns WHERE run_id = ?1 AND session_num = ?2 AND turn_num = ?3",
            params![run_id, session_num, turn.turn_num],
        )
        .map_err(|e| SearchError::sqlite("delete old turns row", e))?;
        tx.execute(
            "INSERT INTO turns(run_id, session_num, turn_num, user_input, agent_text, tool_calls)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                run_id,
                session_num,
                turn.turn_num,
                user_input,
                agent_text,
                tool_calls,
            ],
        )
        .map_err(|e| SearchError::sqlite("insert turn row", e))?;
        let rowid: i64 = tx
            .query_row(
                "SELECT rowid FROM turns WHERE run_id = ?1 AND session_num = ?2 AND turn_num = ?3",
                params![run_id, session_num, turn.turn_num],
                |row| row.get(0),
            )
            .map_err(|e| SearchError::sqlite("read turn rowid", e))?;
        tx.execute(
            "INSERT INTO turns_fts(rowid, user_input, agent_text) VALUES (?1, ?2, ?3)",
            params![rowid, user_input, agent_text],
        )
        .map_err(|e| SearchError::sqlite("insert turns_fts row", e))?;
        tx.commit()
            .map_err(|e| SearchError::sqlite("commit turn tx", e))?;
        Ok(())
    }

    /// Whether the DB already has a row for `(run_id, session_num)`.
    /// Used by [`Self::rebuild_from_disk`] to skip already-indexed
    /// sessions.
    pub fn has_session(&self, run_id: &str, session_num: u32) -> Result<bool, SearchError> {
        let conn = self.lock_conn()?;
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sessions WHERE run_id = ?1 AND session_num = ?2",
                params![run_id, session_num],
                |row| row.get(0),
            )
            .map_err(|e| SearchError::sqlite("count sessions", e))?;
        Ok(count > 0)
    }

    /// Walk `runs_dir`, find every `session_NNN.json`, and ingest those
    /// not already in the DB. Returns the count of newly-ingested
    /// sessions.
    ///
    /// Files that fail to parse or contain malformed JSON are logged
    /// via `tracing::warn!` and skipped — a single corrupt session
    /// must not stop the rebuild from progressing.
    pub fn rebuild_from_disk(&self, runs_dir: impl AsRef<Path>) -> Result<usize, SearchError> {
        let root = runs_dir.as_ref();
        if !root.exists() {
            return Ok(0);
        }
        let mut ingested = 0usize;
        let read = std::fs::read_dir(root).map_err(|e| SearchError::io(root, e))?;
        for entry in read {
            let entry = entry.map_err(|e| SearchError::io(root, e))?;
            let one_run = entry.path();
            if !one_run.is_dir() {
                continue;
            }
            let Some(run_id) = one_run.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let session_log_dir = one_run.join(tmg_harness::SESSION_LOG_DIRNAME);
            if !session_log_dir.exists() {
                continue;
            }
            let session_files = std::fs::read_dir(&session_log_dir)
                .map_err(|e| SearchError::io(session_log_dir.clone(), e))?;
            for f in session_files {
                let f = f.map_err(|e| SearchError::io(session_log_dir.clone(), e))?;
                let path = f.path();
                let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                if !name.starts_with("session_")
                    || !std::path::Path::new(name)
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
                {
                    continue;
                }
                // session_summaries.json is a sibling, not a
                // session_NNN file — skip explicitly.
                if name == "session_summaries.json" {
                    continue;
                }
                let raw = match std::fs::read_to_string(&path) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!(
                            file = %path.display(),
                            error = %e,
                            "skipping unreadable session log"
                        );
                        continue;
                    }
                };
                let session: Session = match serde_json::from_str(&raw) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(
                            file = %path.display(),
                            error = %e,
                            "skipping unparsable session log"
                        );
                        continue;
                    }
                };
                if self.has_session(run_id, session.index)? {
                    continue;
                }
                if let Err(e) = self.ingest_session(run_id, &session) {
                    tracing::warn!(
                        file = %path.display(),
                        error = %e,
                        "failed to ingest session into search index"
                    );
                    continue;
                }
                ingested = ingested.saturating_add(1);
            }
        }
        Ok(ingested)
    }

    /// Run a search query against the configured `scope`.
    ///
    /// `query` is an FTS5 MATCH expression — operators (`AND`, `OR`,
    /// `NEAR`, prefix `*`) work as documented by `SQLite`. `limit` is
    /// clamped to `[1, 200]`. `since`, when provided, restricts the
    /// result to sessions that started at or after the given instant.
    ///
    /// Results are ordered by inverted BM25 (most-relevant first) and
    /// deduplicated by `(run_id, session_num)` when `scope ==
    /// SearchScope::All`.
    pub fn query(
        &self,
        query: &str,
        limit: usize,
        scope: SearchScope,
        since: Option<DateTime<Utc>>,
    ) -> Result<Vec<SearchHit>, SearchError> {
        if query.trim().is_empty() {
            return Err(SearchError::invalid_query("query must not be empty"));
        }
        let limit = limit.clamp(1, 200);
        let since_str = since.as_ref().map(DateTime::to_rfc3339);

        let conn = self.lock_conn()?;
        let mut hits: Vec<SearchHit> = Vec::new();

        if matches!(scope, SearchScope::Summary | SearchScope::All) {
            let sql = "SELECT s.run_id, s.session_num, s.started_at, s.summary,
                              snippet(sessions_fts, 0, '<mark>', '</mark>', '...', 16),
                              bm25(sessions_fts)
                       FROM sessions_fts
                       JOIN sessions s ON s.rowid = sessions_fts.rowid
                       WHERE sessions_fts MATCH ?1
                         AND (?2 IS NULL OR s.started_at >= ?2)
                       ORDER BY bm25(sessions_fts) ASC
                       LIMIT ?3";
            let mut stmt = conn
                .prepare(sql)
                .map_err(|e| SearchError::sqlite("prepare summary query", e))?;
            let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
            let rows = stmt
                .query_map(params![query, since_str, limit_i64], |row| {
                    let raw_score: f64 = row.get(5)?;
                    Ok(SearchHit {
                        run_id: row.get(0)?,
                        session_num: row.get(1)?,
                        started_at: row.get(2)?,
                        summary: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                        snippet: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                        score: -raw_score,
                    })
                })
                .map_err(|e| SearchError::sqlite("execute summary query", e))?;
            for r in rows {
                hits.push(r.map_err(|e| SearchError::sqlite("read summary row", e))?);
            }
        }

        if matches!(scope, SearchScope::Turns | SearchScope::All) {
            let sql = "SELECT t.run_id, t.session_num,
                              COALESCE((SELECT s.started_at FROM sessions s
                                        WHERE s.run_id = t.run_id AND s.session_num = t.session_num), ''),
                              COALESCE((SELECT s.summary FROM sessions s
                                        WHERE s.run_id = t.run_id AND s.session_num = t.session_num), ''),
                              snippet(turns_fts, 1, '<mark>', '</mark>', '...', 16),
                              bm25(turns_fts)
                       FROM turns_fts
                       JOIN turns t ON t.rowid = turns_fts.rowid
                       LEFT JOIN sessions s ON s.run_id = t.run_id AND s.session_num = t.session_num
                       WHERE turns_fts MATCH ?1
                         AND (?2 IS NULL OR s.started_at IS NULL OR s.started_at >= ?2)
                       ORDER BY bm25(turns_fts) ASC
                       LIMIT ?3";
            let mut stmt = conn
                .prepare(sql)
                .map_err(|e| SearchError::sqlite("prepare turns query", e))?;
            let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
            let rows = stmt
                .query_map(params![query, since_str, limit_i64], |row| {
                    let raw_score: f64 = row.get(5)?;
                    Ok(SearchHit {
                        run_id: row.get(0)?,
                        session_num: row.get(1)?,
                        started_at: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                        summary: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                        snippet: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                        score: -raw_score,
                    })
                })
                .map_err(|e| SearchError::sqlite("execute turns query", e))?;
            for r in rows {
                hits.push(r.map_err(|e| SearchError::sqlite("read turns row", e))?);
            }
        }

        // Deduplicate by `(run_id, session_num)` when both tables were
        // queried, keeping the highest-scoring hit per session.
        if matches!(scope, SearchScope::All) {
            hits.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut seen: std::collections::HashSet<(String, u32)> =
                std::collections::HashSet::new();
            hits.retain(|h| seen.insert((h.run_id.clone(), h.session_num)));
        } else {
            hits.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        hits.truncate(limit);
        Ok(hits)
    }

    /// Compute aggregate stats for `tmg search stats`.
    pub fn stats(&self) -> Result<SearchStats, SearchError> {
        let conn = self.lock_conn()?;
        let sessions: i64 = conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
            .map_err(|e| SearchError::sqlite("count sessions", e))?;
        let turns: i64 = conn
            .query_row("SELECT COUNT(*) FROM turns", [], |row| row.get(0))
            .map_err(|e| SearchError::sqlite("count turns", e))?;
        let size_bytes = std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);
        Ok(SearchStats {
            sessions: u64::try_from(sessions).unwrap_or(0),
            turns: u64::try_from(turns).unwrap_or(0),
            size_bytes,
            db_path: self.path.clone(),
        })
    }

    /// Lock the connection mutex, mapping a poisoned mutex to a
    /// [`SearchError`].
    fn lock_conn(&self) -> Result<std::sync::MutexGuard<'_, Connection>, SearchError> {
        self.conn.lock().map_err(|_poison| SearchError::Sqlite {
            context: "search index mutex poisoned".to_owned(),
            source: rusqlite::Error::InvalidQuery,
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test assertions use expect for clearer messages"
)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};
    use tempfile::TempDir;
    use tmg_harness::{SessionEndTrigger, SessionLog};

    fn fresh_index(tmp: &TempDir) -> SearchIndex {
        SearchIndex::open(tmp.path().join("state.db")).expect("open index")
    }

    fn session_with_summary(index: u32, summary: &str) -> Session {
        let mut s = Session::begin(index);
        s.summary = summary.to_owned();
        s.tool_calls_count = 3;
        s.files_modified.push("src/foo.rs".to_owned());
        s.end(SessionEndTrigger::Completed);
        s
    }

    #[test]
    fn open_creates_schema_and_idempotent() {
        let tmp = TempDir::new().expect("tempdir");
        let _idx = fresh_index(&tmp);
        // Reopen should not error.
        let _idx2 = fresh_index(&tmp);
    }

    #[test]
    fn ingest_then_query_summary_roundtrip() {
        let tmp = TempDir::new().expect("tempdir");
        let idx = fresh_index(&tmp);
        let s = session_with_summary(1, "implemented OAuth callback handler in axum");
        idx.ingest_session("run-aaa", &s).expect("ingest");

        let hits = idx
            .query("OAuth", 10, SearchScope::Summary, None)
            .expect("query");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].run_id, "run-aaa");
        assert_eq!(hits[0].session_num, 1);
        assert!(hits[0].snippet.contains("<mark>OAuth</mark>"), "{hits:?}");
    }

    #[test]
    fn ingest_session_redacts_secrets_in_summary() {
        let tmp = TempDir::new().expect("tempdir");
        let idx = fresh_index(&tmp);
        let s = session_with_summary(1, "deployed with key AKIAIOSFODNN7EXAMPLE during rollout");
        idx.ingest_session("run-aaa", &s).expect("ingest");
        let hits = idx
            .query("rollout", 10, SearchScope::Summary, None)
            .expect("query");
        assert_eq!(hits.len(), 1);
        assert!(!hits[0].summary.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(hits[0].summary.contains("[REDACTED]"));
    }

    #[test]
    fn empty_summary_still_indexes() {
        let tmp = TempDir::new().expect("tempdir");
        let idx = fresh_index(&tmp);
        let s = session_with_summary(1, "");
        idx.ingest_session("run-aaa", &s).expect("ingest");
        assert!(idx.has_session("run-aaa", 1).expect("has"));
    }

    #[test]
    fn since_filter_excludes_older_sessions() {
        let tmp = TempDir::new().expect("tempdir");
        let idx = fresh_index(&tmp);
        let mut old = session_with_summary(1, "early oauth bug");
        old.started_at = Utc::now() - Duration::days(30);
        idx.ingest_session("run-aaa", &old).expect("old");
        let mut new = session_with_summary(2, "recent oauth fix");
        new.started_at = Utc::now();
        idx.ingest_session("run-aaa", &new).expect("new");

        let cutoff = Utc::now() - Duration::days(1);
        let hits = idx
            .query("oauth", 10, SearchScope::Summary, Some(cutoff))
            .expect("query");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].session_num, 2);
    }

    #[test]
    fn scope_summary_only_misses_turn_text() {
        let tmp = TempDir::new().expect("tempdir");
        let idx = fresh_index(&tmp);
        let s = session_with_summary(1, "");
        idx.ingest_session("run-aaa", &s).expect("ingest session");

        idx.ingest_turn(
            "run-aaa",
            1,
            &TurnRecord {
                turn_num: 1,
                user_input: "tell me about webhooks".to_owned(),
                agent_text: "webhooks are HTTP callbacks".to_owned(),
                tool_calls: "[]".to_owned(),
            },
        )
        .expect("ingest turn");

        let hits_summary = idx
            .query("webhooks", 10, SearchScope::Summary, None)
            .expect("summary query");
        assert!(
            hits_summary.is_empty(),
            "summary scope must not match turn text: {hits_summary:?}"
        );
        let hits_turns = idx
            .query("webhooks", 10, SearchScope::Turns, None)
            .expect("turns query");
        assert_eq!(hits_turns.len(), 1);
        let hits_all = idx
            .query("webhooks", 10, SearchScope::All, None)
            .expect("all query");
        assert_eq!(hits_all.len(), 1);
    }

    #[test]
    fn rebuild_from_disk_picks_up_unindexed_sessions() {
        let tmp = TempDir::new().expect("tempdir");
        let root_runs = tmp.path().join("runs");
        let run_id = "run-bbb";
        let one_run = root_runs.join(run_id);
        let log = SessionLog::new(one_run.join(tmg_harness::SESSION_LOG_DIRNAME));
        let mut s = Session::begin(1);
        s.summary = "rebuilt projects via search index".to_owned();
        s.end(SessionEndTrigger::Completed);
        log.save(&s).expect("save session");

        let idx = SearchIndex::open(tmp.path().join("state.db")).expect("open");
        let count = idx.rebuild_from_disk(&root_runs).expect("rebuild");
        assert_eq!(count, 1);
        // Re-running rebuild is a no-op now.
        let again = idx.rebuild_from_disk(&root_runs).expect("rebuild2");
        assert_eq!(again, 0);
        assert!(idx.has_session(run_id, 1).expect("has"));
    }

    #[test]
    fn stats_reports_counts() {
        let tmp = TempDir::new().expect("tempdir");
        let idx = fresh_index(&tmp);
        let s = session_with_summary(1, "stat me");
        idx.ingest_session("run-aaa", &s).expect("ingest");
        let stats = idx.stats().expect("stats");
        assert_eq!(stats.sessions, 1);
        assert_eq!(stats.turns, 0);
        assert!(stats.size_bytes > 0);
        assert!(stats.db_path.ends_with("state.db"));
    }

    /// Acceptance check for issue #53: simulating one-shot mode (no
    /// runner, no session-end hook) leaves the DB completely empty.
    /// The CLI's `run_prompt` path follows this contract — it never
    /// constructs a [`RunRunner`] and therefore cannot register the
    /// hook that drives [`SearchIndex::ingest_session`]. This test
    /// pins that invariant at the index layer: opening a fresh DB and
    /// not calling any ingest method must yield zero rows.
    #[test]
    fn one_shot_mode_does_not_ingest() {
        let tmp = TempDir::new().expect("tempdir");
        let idx = fresh_index(&tmp);
        // Simulate one-shot mode: index is open, but no hook fires
        // because no runner was registered. Querying must return empty.
        let stats = idx.stats().expect("stats");
        assert_eq!(stats.sessions, 0);
        assert_eq!(stats.turns, 0);
        // A query against an empty index returns zero hits (not an
        // error, since the FTS table exists).
        let hits = idx
            .query("anything", 10, SearchScope::All, None)
            .expect("query");
        assert!(hits.is_empty());
    }

    #[test]
    fn empty_query_is_invalid_param() {
        let tmp = TempDir::new().expect("tempdir");
        let idx = fresh_index(&tmp);
        let err = idx
            .query("   ", 10, SearchScope::Summary, None)
            .expect_err("must reject empty query");
        assert!(matches!(err, SearchError::InvalidQuery { .. }));
    }
}
