//! Run data model.
//!
//! A [`Run`] is the top-level unit of work in tsumugi. Each run has a
//! unique [`RunId`], a [`RunScope`] indicating whether it is ad-hoc or
//! managed by a workflow harness, and a [`RunStatus`] tracking its
//! lifecycle. Multiple [`Session`](crate::session::Session)s may be
//! associated with a single run; the run's `session_count` and
//! `last_session_at` are updated each time a session begins.
//!
//! The on-disk format is `run.toml` under `.tsumugi/runs/<run-id>/`,
//! using the structure described in SPEC §9.5.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::HarnessError;

/// Short hex identifier for a run, e.g. `"a3f1e2b9"`.
///
/// Generated from a UUID v4 by taking the first 8 hex characters of the
/// simple representation. Short ids are user-friendly for display and
/// directory naming; full uniqueness is not required because the
/// directory namespace under `.tsumugi/runs/` is small.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RunId(String);

impl RunId {
    /// Create a fresh random run id.
    #[must_use]
    pub fn new() -> Self {
        let uuid = uuid::Uuid::new_v4();
        let simple = uuid.simple().to_string();
        // Take the first 8 hex characters for a short, human-friendly id.
        Self(simple[..8].to_owned())
    }

    /// Parse a user-supplied run id, validating that it matches the
    /// canonical 8-character hexadecimal shape produced by [`Self::new`].
    ///
    /// Returns [`HarnessError::InvalidRunId`] when `s` does not match
    /// `^[0-9a-f]{8}$`. This is the only constructor that should be used
    /// at trust boundaries (CLI arguments, network input, etc.) so we
    /// never construct paths from caller-controlled values that escape
    /// the runs-dir.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::InvalidRunId`] when `s` is not exactly
    /// eight ASCII hex digits (lowercase a-f).
    pub fn parse(s: impl Into<String>) -> Result<Self, HarnessError> {
        let raw = s.into();
        if Self::is_valid_shape(&raw) {
            Ok(Self(raw))
        } else {
            Err(HarnessError::InvalidRunId { run_id: raw })
        }
    }

    /// Construct a run id from a raw string without validating its
    /// shape.
    ///
    /// Internal-only escape hatch for callers that have already
    /// validated the input by other means — primarily
    /// [`crate::store::RunStore::list`] (the directory name on disk
    /// stands in for validation; a hand-edited directory with a bogus
    /// name will simply fail to load `run.toml`) and the JSON-pointer
    /// fallback inside [`crate::store::RunStore::current`] (the
    /// pointer's payload is also self-validated when the run is loaded).
    /// **Do not** use this for CLI arguments or any other untrusted
    /// input — call [`Self::parse`] instead.
    #[must_use]
    pub(crate) fn from_string_unchecked(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Return `true` when `s` matches the canonical 8-char hex shape
    /// produced by [`Self::new`].
    #[must_use]
    pub fn is_valid_shape(s: &str) -> bool {
        s.len() == 8
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }

    /// Return the run id as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Return a short (8-char) version of the run id for display.
    ///
    /// If the underlying string is shorter than 8 characters this returns
    /// the full string.
    #[must_use]
    pub fn short(&self) -> &str {
        let take = self
            .0
            .char_indices()
            .nth(8)
            .map_or(self.0.len(), |(i, _)| i);
        &self.0[..take]
    }
}

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Scope of a run. SPEC §9.2 defines a 2-stage model: a run starts as
/// [`AdHoc`](RunScope::AdHoc) and can later be promoted to
/// [`Harnessed`](RunScope::Harnessed) when a workflow attaches.
///
/// The `Harnessed` variant carries optional auto-promotion metadata
/// (`upgraded_at`, `upgraded_from_session`, `upgrade_reason`,
/// `features_path`, `init_script_path`) populated by
/// [`RunStore::upgrade_to_harnessed`](crate::store::RunStore::upgrade_to_harnessed)
/// when an ad-hoc run is promoted via the SPEC §9.3 flow. Runs created
/// directly as harnessed (e.g. via the future `tmg run start` command)
/// leave these fields `None`, so existing on-disk records remain
/// readable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunScope {
    /// Free-form interactive run with no workflow attached.
    AdHoc,
    /// Run managed by a workflow harness (workflow id, max sessions, etc.).
    Harnessed {
        /// Identifier of the workflow definition driving this run.
        workflow_id: String,
        /// Optional cap on the number of sessions allowed in this run.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_sessions: Option<u32>,
        /// Path (relative to the run directory) of `features.json`.
        ///
        /// Populated by [`RunStore::upgrade_to_harnessed`](crate::store::RunStore::upgrade_to_harnessed)
        /// on auto-promotion; runs created directly as harnessed
        /// leave this `None`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        features_path: Option<String>,
        /// Path (relative to the run directory) of `init.sh`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        init_script_path: Option<String>,
        /// Timestamp of the auto-promotion that produced this scope.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        upgraded_at: Option<DateTime<Utc>>,
        /// 1-indexed session number that was active when the
        /// auto-promotion fired.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        upgraded_from_session: Option<u32>,
        /// Concise reason text emitted by the escalator subagent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        upgrade_reason: Option<String>,
    },
}

impl RunScope {
    /// Short label for display.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::AdHoc => "ad-hoc",
            Self::Harnessed { .. } => "harnessed",
        }
    }

    /// Whether this scope is the [`AdHoc`](Self::AdHoc) variant.
    ///
    /// Convenience for the auto-promotion gate; equivalent to
    /// `matches!(scope, RunScope::AdHoc)`.
    #[must_use]
    pub fn is_ad_hoc(&self) -> bool {
        matches!(self, Self::AdHoc)
    }

    /// Construct a fresh harnessed scope without any auto-promotion
    /// metadata.
    ///
    /// Equivalent to writing
    /// `RunScope::Harnessed { workflow_id, max_sessions, features_path: None, init_script_path: None, upgraded_at: None, upgraded_from_session: None, upgrade_reason: None }`
    /// — provided as a constructor so that test fixtures and the
    /// future `tmg run start` path do not have to enumerate every
    /// auto-promotion field.
    #[must_use]
    pub fn harnessed(workflow_id: impl Into<String>, max_sessions: Option<u32>) -> Self {
        Self::Harnessed {
            workflow_id: workflow_id.into(),
            max_sessions,
            features_path: None,
            init_script_path: None,
            upgraded_at: None,
            upgraded_from_session: None,
            upgrade_reason: None,
        }
    }
}

impl Default for RunScope {
    /// Default scope is [`Self::AdHoc`]; matches the freshly-created
    /// run from [`Run::new_ad_hoc`] and the
    /// [`SessionState`](crate::state::SessionState) seed.
    fn default() -> Self {
        Self::AdHoc
    }
}

/// Lifecycle state of a run. SPEC §9.4.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum RunStatus {
    /// The run is currently active or being resumed.
    Running,
    /// The run was paused (e.g. by user action) and may be resumed.
    Paused,
    /// The run completed successfully.
    Completed,
    /// The run failed; `reason` carries a human-readable description.
    Failed {
        /// Reason for failure.
        reason: String,
    },
    /// The run hit `max_sessions` and stopped without completing.
    Exhausted,
}

impl RunStatus {
    /// Return true if the run is in a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed { .. })
    }

    /// Return true if the run is unfinished and may be resumed on startup.
    ///
    /// SPEC §9.4: `Running` / `Paused` / `Exhausted` are eligible for
    /// auto-resume because they are not terminal in the same way as
    /// `Completed` or `Failed`.
    #[must_use]
    pub fn is_resumable(&self) -> bool {
        matches!(self, Self::Running | Self::Paused | Self::Exhausted)
    }
}

/// Top-level run record. Persisted as `run.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Run {
    /// Unique identifier for this run.
    pub id: RunId,
    /// Run scope (ad-hoc or harnessed by a workflow).
    pub scope: RunScope,
    /// Current lifecycle status.
    pub status: RunStatus,
    /// Optional workflow id (mirrored at top level for convenience).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workflow_id: Option<String>,
    /// When the run was created.
    pub created_at: DateTime<Utc>,
    /// When the most recent session began, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_session_at: Option<DateTime<Utc>>,
    /// Number of sessions that have been started under this run.
    #[serde(default)]
    pub session_count: u32,
    /// Optional cap on the number of sessions (mirrors `RunScope::Harnessed`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_sessions: Option<u32>,
    /// Workspace path for this run (target codebase).
    pub workspace_path: PathBuf,
    /// Free-form structured inputs supplied at run creation.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub inputs: BTreeMap<String, serde_json::Value>,
}

impl Run {
    /// Construct a new ad-hoc run with default lifecycle state.
    ///
    /// `workspace_path` should be the canonicalised path to the user's
    /// project root; this becomes the symlink target inside the run
    /// directory.
    #[must_use]
    pub fn new_ad_hoc(workspace_path: PathBuf) -> Self {
        Self {
            id: RunId::new(),
            scope: RunScope::AdHoc,
            status: RunStatus::Running,
            workflow_id: None,
            created_at: Utc::now(),
            last_session_at: None,
            session_count: 0,
            max_sessions: None,
            workspace_path,
            inputs: BTreeMap::new(),
        }
    }
}

/// Lightweight summary used for listing and TUI header display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunSummary {
    /// Run id.
    pub id: RunId,
    /// Scope label (`"ad-hoc"` or `"harnessed"`).
    pub scope_label: &'static str,
    /// Workflow id (only set for harnessed scope).
    pub workflow_id: Option<String>,
    /// Current lifecycle status (cloned for cheap display).
    pub status: RunStatus,
    /// When the run was created.
    pub created_at: DateTime<Utc>,
    /// Last session timestamp, if any.
    pub last_session_at: Option<DateTime<Utc>>,
    /// Number of sessions so far.
    pub session_count: u32,
}

impl RunSummary {
    /// Construct a summary from a [`Run`] reference.
    #[must_use]
    pub fn from_run(run: &Run) -> Self {
        let workflow_id = match &run.scope {
            RunScope::AdHoc => None,
            RunScope::Harnessed { workflow_id, .. } => Some(workflow_id.clone()),
        };
        Self {
            id: run.id.clone(),
            scope_label: run.scope.label(),
            workflow_id,
            status: run.status.clone(),
            created_at: run.created_at,
            last_session_at: run.last_session_at,
            session_count: run.session_count,
        }
    }

    /// Short (8-char) run id for display.
    #[must_use]
    pub fn short_id(&self) -> &str {
        self.id.short()
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;

    #[test]
    fn run_id_is_short_hex() {
        let id = RunId::new();
        assert_eq!(id.as_str().len(), 8);
        assert!(id.as_str().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn run_id_short_handles_long_strings() {
        let id = RunId::from_string_unchecked("abcdef0123456789");
        assert_eq!(id.short(), "abcdef01");
    }

    #[test]
    fn run_id_short_handles_short_strings() {
        let id = RunId::from_string_unchecked("abc");
        assert_eq!(id.short(), "abc");
    }

    #[test]
    fn run_id_parse_accepts_canonical_shape() {
        let id = RunId::parse("abc12345").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(id.as_str(), "abc12345");
    }

    #[test]
    fn run_id_parse_rejects_uppercase() {
        match RunId::parse("ABC12345") {
            Err(HarnessError::InvalidRunId { run_id }) => assert_eq!(run_id, "ABC12345"),
            Ok(id) => panic!("expected InvalidRunId, got Ok({})", id.as_str()),
            Err(other) => panic!("expected InvalidRunId, got {other:?}"),
        }
    }

    #[test]
    fn run_id_parse_rejects_wrong_length() {
        assert!(matches!(
            RunId::parse("abc"),
            Err(HarnessError::InvalidRunId { .. })
        ));
        assert!(matches!(
            RunId::parse("abc1234567"),
            Err(HarnessError::InvalidRunId { .. })
        ));
    }

    #[test]
    fn run_id_parse_rejects_non_hex() {
        assert!(matches!(
            RunId::parse("../foo12"),
            Err(HarnessError::InvalidRunId { .. })
        ));
        assert!(matches!(
            RunId::parse("zzzzzzzz"),
            Err(HarnessError::InvalidRunId { .. })
        ));
    }

    #[test]
    fn run_id_new_passes_parse_validation() {
        let generated = RunId::new();
        let raw = generated.as_str().to_owned();
        let reparsed = RunId::parse(raw.clone()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(reparsed.as_str(), raw);
    }

    #[test]
    fn run_status_is_resumable() {
        assert!(RunStatus::Running.is_resumable());
        assert!(RunStatus::Paused.is_resumable());
        assert!(RunStatus::Exhausted.is_resumable());
        assert!(!RunStatus::Completed.is_resumable());
        assert!(
            !RunStatus::Failed {
                reason: "boom".to_owned(),
            }
            .is_resumable()
        );
    }

    #[test]
    fn ad_hoc_run_serializes_round_trip() {
        let run = Run::new_ad_hoc(PathBuf::from("/tmp/project"));
        let serialized = toml::to_string(&run).unwrap_or_else(|e| panic!("{e}"));
        let parsed: Run = toml::from_str(&serialized).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(run, parsed);
    }

    #[test]
    fn harnessed_run_round_trip() {
        let mut run = Run::new_ad_hoc(PathBuf::from("/tmp/project"));
        run.scope = RunScope::Harnessed {
            workflow_id: "fix-bug".to_owned(),
            max_sessions: Some(8),
            features_path: None,
            init_script_path: None,
            upgraded_at: None,
            upgraded_from_session: None,
            upgrade_reason: None,
        };
        run.workflow_id = Some("fix-bug".to_owned());
        run.max_sessions = Some(8);
        let serialized = toml::to_string(&run).unwrap_or_else(|e| panic!("{e}"));
        let parsed: Run = toml::from_str(&serialized).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(run, parsed);
    }

    #[test]
    fn harnessed_run_with_upgrade_metadata_round_trip() {
        let mut run = Run::new_ad_hoc(PathBuf::from("/tmp/project"));
        run.scope = RunScope::Harnessed {
            workflow_id: "auto-promoted".to_owned(),
            max_sessions: None,
            features_path: Some("features.json".to_owned()),
            init_script_path: Some("init.sh".to_owned()),
            upgraded_at: Some(chrono::Utc::now()),
            upgraded_from_session: Some(2),
            upgrade_reason: Some("multi-feature scope".to_owned()),
        };
        let serialized = toml::to_string(&run).unwrap_or_else(|e| panic!("{e}"));
        let parsed: Run = toml::from_str(&serialized).unwrap_or_else(|e| panic!("{e}"));
        // chrono's Utc::now() loses sub-second precision through TOML
        // serialization in some chrono versions; compare structurally
        // rather than asserting full equality on the timestamp.
        match (&run.scope, &parsed.scope) {
            (
                RunScope::Harnessed {
                    workflow_id: a,
                    upgrade_reason: ra,
                    ..
                },
                RunScope::Harnessed {
                    workflow_id: b,
                    upgrade_reason: rb,
                    ..
                },
            ) => {
                assert_eq!(a, b);
                assert_eq!(ra, rb);
            }
            other => panic!("expected harnessed/harnessed, got {other:?}"),
        }
    }

    #[test]
    fn run_summary_label() {
        let run = Run::new_ad_hoc(PathBuf::from("/tmp/project"));
        let summary = RunSummary::from_run(&run);
        assert_eq!(summary.scope_label, "ad-hoc");
        assert_eq!(summary.short_id().len(), 8);
    }
}
