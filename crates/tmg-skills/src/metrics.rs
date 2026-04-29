//! Per-skill success/failure metrics.
//!
//! After every `use_skill` invocation the harness records whether the
//! follow-up turns succeeded or failed. The counters live in
//! `<skill-dir>/.metrics.json` (a hidden sibling of `SKILL.md`) so the
//! human-authored body never gets churned by automated counters. This
//! module owns reading and updating that file.
//!
//! ## File shape
//!
//! ```json
//! {
//!   "success_count": 3,
//!   "failure_count": 1,
//!   "improvement_hints": [
//!     "step 4 failed: expected 'cargo' on PATH, got 'command not found'"
//!   ]
//! }
//! ```
//!
//! ## Outcome rules (issue #54)
//!
//! - **Success**: the next N turns produce no error and no user
//!   negation. Recorded as a single `record_success`.
//! - **Failure**: the immediately following turn errors *or* the user
//!   issues a negation. Recorded as a single `record_failure` and the
//!   error/negation snippet appended to `improvement_hints`.
//!
//! Auto-patch of the SKILL.md body based on `improvement_hints` is
//! **explicitly out of scope** for this issue. The hints are produced
//! and persisted; consuming them is a follow-up.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as AsyncMutex;

use crate::error::SkillError;

/// Per-skill-dir async mutex registry.
///
/// Two parallel `update_metrics` calls in the same process targeting the
/// same skill could otherwise race the load → mutate → save sequence
/// and clobber each other. We serialise per skill-dir using one
/// [`AsyncMutex`] per resolved path. Cross-process synchronisation is
/// best-effort (we still write atomically via tempfile + rename so
/// readers never see a half-written file), but two `tmg` instances
/// updating the same skill simultaneously can lose the read-modify-write
/// race on the file's contents — operators running concurrent agents
/// against the same skill should serialise out-of-band.
fn metrics_locks() -> &'static std::sync::Mutex<HashMap<PathBuf, Arc<AsyncMutex<()>>>> {
    static LOCKS: OnceLock<std::sync::Mutex<HashMap<PathBuf, Arc<AsyncMutex<()>>>>> =
        OnceLock::new();
    LOCKS.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// Acquire (or create) the per-skill-dir guard.
fn lock_for(skill_dir: &Path) -> Arc<AsyncMutex<()>> {
    let key = skill_dir.to_path_buf();
    let table = metrics_locks();
    let mut guard = match table.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    Arc::clone(
        guard
            .entry(key)
            .or_insert_with(|| Arc::new(AsyncMutex::new(()))),
    )
}

/// The hidden filename used for per-skill metrics.
pub const METRICS_FILENAME: &str = ".metrics.json";

/// Maximum number of `improvement_hints` entries kept on disk.
///
/// We cap the list so a noisy local model can't churn the file
/// unbounded. The newest entry wins on overflow.
pub const MAX_HINTS: usize = 16;

/// Persistent counters and improvement hints for one skill.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillMetrics {
    /// Number of recorded successes.
    #[serde(default)]
    pub success_count: u32,

    /// Number of recorded failures.
    #[serde(default)]
    pub failure_count: u32,

    /// Most recent error / user-negation strings tied to failures, in
    /// chronological order. Capped at [`MAX_HINTS`] entries; new entries
    /// push older ones off the front.
    #[serde(default)]
    pub improvement_hints: Vec<String>,
}

impl SkillMetrics {
    /// Construct an empty metrics record.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment the success counter.
    pub fn record_success(&mut self) {
        self.success_count = self.success_count.saturating_add(1);
    }

    /// Increment the failure counter and append `hint` to
    /// `improvement_hints`. If the buffer exceeds [`MAX_HINTS`] the
    /// oldest entry is dropped.
    pub fn record_failure(&mut self, hint: impl Into<String>) {
        self.failure_count = self.failure_count.saturating_add(1);
        self.improvement_hints.push(hint.into());
        while self.improvement_hints.len() > MAX_HINTS {
            self.improvement_hints.remove(0);
        }
    }
}

/// Compute the path of the `.metrics.json` file inside `skill_dir`.
#[must_use]
pub fn metrics_path(skill_dir: impl AsRef<Path>) -> PathBuf {
    skill_dir.as_ref().join(METRICS_FILENAME)
}

/// Load metrics for a skill. Returns [`SkillMetrics::default`] when the
/// file does not exist yet.
///
/// # Errors
///
/// Returns [`SkillError::Io`] if the file exists but cannot be read,
/// or if the JSON is malformed.
pub async fn load_metrics(skill_dir: impl AsRef<Path>) -> Result<SkillMetrics, SkillError> {
    let path = metrics_path(skill_dir);
    match tokio::fs::read_to_string(&path).await {
        Ok(text) => serde_json::from_str(&text).map_err(|e| {
            SkillError::io(
                format!("parsing {}", path.display()),
                std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
            )
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(SkillMetrics::new()),
        Err(e) => Err(SkillError::io(format!("reading {}", path.display()), e)),
    }
}

/// Persist metrics for a skill.
///
/// Writes the JSON document with a trailing newline so the file is
/// well-behaved under POSIX text-file conventions. The write is
/// atomic-on-rename: the document is staged to `<path>.tmp` first, then
/// renamed over the destination, so concurrent readers never observe a
/// half-written file.
///
/// # Errors
///
/// Returns [`SkillError::Io`] when the parent directory does not exist
/// or the write/rename fails.
pub async fn save_metrics(
    skill_dir: impl AsRef<Path>,
    metrics: &SkillMetrics,
) -> Result<(), SkillError> {
    let path = metrics_path(skill_dir);
    let mut serialized = serde_json::to_string_pretty(metrics).map_err(|e| {
        SkillError::io(
            format!("serializing metrics for {}", path.display()),
            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
        )
    })?;
    serialized.push('\n');

    // Stage to <path>.tmp, then rename. `rename` is atomic on POSIX
    // when source and destination live on the same filesystem; siblings
    // of the target file always do.
    let tmp_path = path.with_extension("json.tmp");
    tokio::fs::write(&tmp_path, &serialized)
        .await
        .map_err(|e| SkillError::io(format!("writing {}", tmp_path.display()), e))?;
    tokio::fs::rename(&tmp_path, &path).await.map_err(|e| {
        SkillError::io(
            format!("renaming {} -> {}", tmp_path.display(), path.display()),
            e,
        )
    })
}

/// Convenience: load, run `f`, and save back atomically (per process).
///
/// This is the entry point the harness uses after each post-skill turn:
///
/// ```ignore
/// update_metrics(&skill_dir, |m| m.record_success()).await?;
/// ```
///
/// ## Concurrency semantics
///
/// Within the current process, two concurrent calls targeting the same
/// `skill_dir` are serialised through a per-path async mutex so the
/// load → mutate → save sequence is atomic. The save itself uses
/// tempfile + rename so readers (and crashed writers) never observe a
/// half-written `.metrics.json`. **Cross-process** synchronisation is
/// best-effort: two `tmg` instances updating the same skill at the same
/// instant can lose the read-modify-write race; operators running
/// concurrent agents against the same skill should serialise out-of-band.
///
/// # Errors
///
/// Propagates [`SkillError`] from either the load or save phase.
pub async fn update_metrics<F>(
    skill_dir: impl AsRef<Path>,
    f: F,
) -> Result<SkillMetrics, SkillError>
where
    F: FnOnce(&mut SkillMetrics),
{
    let dir = skill_dir.as_ref();
    let lock = lock_for(dir);
    let _guard = lock.lock().await;
    let mut metrics = load_metrics(dir).await?;
    f(&mut metrics);
    save_metrics(dir, &metrics).await?;
    Ok(metrics)
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests assert with unwrap for clarity; the workspace policy denies them in production code"
)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn load_returns_default_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let metrics = load_metrics(tmp.path()).await.unwrap();
        assert_eq!(metrics, SkillMetrics::default());
    }

    #[tokio::test]
    async fn record_success_increments_counter() {
        let tmp = tempfile::tempdir().unwrap();
        let metrics = update_metrics(tmp.path(), SkillMetrics::record_success)
            .await
            .unwrap();
        assert_eq!(metrics.success_count, 1);
        assert_eq!(metrics.failure_count, 0);
        assert!(metrics.improvement_hints.is_empty());

        // Persisted to disk.
        let on_disk = load_metrics(tmp.path()).await.unwrap();
        assert_eq!(on_disk, metrics);
    }

    #[tokio::test]
    async fn record_failure_appends_hint() {
        let tmp = tempfile::tempdir().unwrap();
        let metrics = update_metrics(tmp.path(), |m| {
            m.record_failure("bad command");
        })
        .await
        .unwrap();
        assert_eq!(metrics.failure_count, 1);
        assert_eq!(metrics.improvement_hints, vec!["bad command".to_owned()]);
    }

    #[tokio::test]
    async fn improvement_hints_are_capped() {
        let tmp = tempfile::tempdir().unwrap();
        for i in 0..(MAX_HINTS + 5) {
            update_metrics(tmp.path(), |m| {
                m.record_failure(format!("hint {i}"));
            })
            .await
            .unwrap();
        }
        let metrics = load_metrics(tmp.path()).await.unwrap();
        assert_eq!(metrics.improvement_hints.len(), MAX_HINTS);
        // Oldest entries dropped; newest preserved.
        assert!(metrics.improvement_hints[0].starts_with("hint 5"));
        assert!(
            metrics
                .improvement_hints
                .last()
                .unwrap()
                .starts_with("hint 20")
        );
    }
}
