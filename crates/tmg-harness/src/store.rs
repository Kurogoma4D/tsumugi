//! Persistence layer for runs.
//!
//! [`RunStore`] manages the on-disk layout under
//! `<runs_dir>/<run-id>/` and provides CRUD-style operations for
//! [`Run`] records. SPEC §9.2 describes the directory structure.
//!
//! The minimal version implemented here covers the operations required
//! by the startup sequence in `tmg-cli`:
//!
//! - [`RunStore::create_ad_hoc`] – allocate a new run directory with a
//!   `workspace` symlink and persisted `run.toml`.
//! - [`RunStore::load`] – read `run.toml` for an existing run.
//! - [`RunStore::list`] – enumerate runs as lightweight summaries.
//! - [`RunStore::save`] – write `run.toml` for an existing run.
//!
//! Symlink creation is best-effort: on platforms (or filesystems) where
//! it is not supported, the run is still usable; only the
//! `<run-dir>/workspace` shortcut is missing.

use std::fs;
use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::ser::Error as _;

use crate::artifacts::ProgressLog;
use crate::error::HarnessError;
use crate::run::{Run, RunId, RunScope, RunSummary};

/// Filename used for the persistent run record under each run directory.
pub const RUN_FILENAME: &str = "run.toml";

/// Filename of the workspace symlink under each run directory.
pub const WORKSPACE_LINK: &str = "workspace";

/// Filename of the "current run" pointer at the runs-dir root.
///
/// The pointer is written as a symlink on Unix-like platforms (the
/// runtime falls back to a JSON pointer file if symlink creation
/// fails). [`RunStore::current`] resolves it back to a [`RunId`] for
/// CLI commands that default to the most-recent active run.
pub const CURRENT_FILENAME: &str = "current";

/// Fallback pointer-file name used when symlink creation fails.
///
/// Stores `{ "run_id": "<id>" }` in JSON so the CLI can still resolve
/// the current run on platforms where symlinks are restricted.
pub const CURRENT_POINTER_FILENAME: &str = "current.json";

/// Filename of the human-readable progress log under each run directory.
pub const PROGRESS_FILENAME: &str = "progress.md";

/// Sub-directory under each run directory holding `session_NNN.json` files.
pub const SESSION_LOG_DIRNAME: &str = "session_log";

/// Filename of the harnessed-only features list under each run directory.
pub const FEATURES_FILENAME: &str = "features.json";

/// Filename of the harnessed-only init script under each run directory.
pub const INIT_SCRIPT_FILENAME: &str = "init.sh";

/// Persistent store for runs rooted at a directory (typically
/// `.tsumugi/runs/`).
///
/// **Concurrency:** this store assumes a single `tmg` process is
/// operating against `runs_dir` at a time. There is no inter-process
/// locking; concurrent writers can race on `run.toml`. A file-lock /
/// advisory-lock layer is tracked as a follow-up.
#[derive(Debug, Clone)]
pub struct RunStore {
    runs_dir: PathBuf,
}

impl RunStore {
    /// Construct a store rooted at `runs_dir`.
    ///
    /// The directory does not need to exist; it will be created lazily
    /// by [`create_ad_hoc`](Self::create_ad_hoc) and other write paths.
    #[must_use]
    pub fn new(runs_dir: impl Into<PathBuf>) -> Self {
        Self {
            runs_dir: runs_dir.into(),
        }
    }

    /// Return the configured runs directory.
    #[must_use]
    pub fn runs_dir(&self) -> &Path {
        &self.runs_dir
    }

    /// Path for a run's directory.
    #[must_use]
    pub fn run_dir(&self, run_id: &RunId) -> PathBuf {
        self.runs_dir.join(run_id.as_str())
    }

    /// Path for a run's `run.toml`.
    #[must_use]
    pub fn run_file(&self, run_id: &RunId) -> PathBuf {
        self.run_dir(run_id).join(RUN_FILENAME)
    }

    /// Path for a run's `progress.md`.
    #[must_use]
    pub fn progress_file(&self, run_id: &RunId) -> PathBuf {
        self.run_dir(run_id).join(PROGRESS_FILENAME)
    }

    /// Directory holding a run's `session_NNN.json` files.
    #[must_use]
    pub fn session_log_dir(&self, run_id: &RunId) -> PathBuf {
        self.run_dir(run_id).join(SESSION_LOG_DIRNAME)
    }

    /// Path for a harnessed run's `features.json`.
    #[must_use]
    pub fn features_file(&self, run_id: &RunId) -> PathBuf {
        self.run_dir(run_id).join(FEATURES_FILENAME)
    }

    /// Path for a harnessed run's `init.sh`.
    #[must_use]
    pub fn init_script_file(&self, run_id: &RunId) -> PathBuf {
        self.run_dir(run_id).join(INIT_SCRIPT_FILENAME)
    }

    /// Create a fresh ad-hoc run rooted at `workspace_path`.
    ///
    /// `initial_request` is currently unused at the persistence level
    /// but is part of the public API so future revisions can record the
    /// kicked-off prompt without breaking callers.
    pub fn create_ad_hoc(
        &self,
        workspace_path: PathBuf,
        initial_request: Option<&str>,
    ) -> Result<Run, HarnessError> {
        let mut run = Run::new_ad_hoc(workspace_path);
        if let Some(req) = initial_request {
            run.inputs.insert(
                "initial_request".to_owned(),
                serde_json::Value::String(req.to_owned()),
            );
        }

        let run_dir = self.run_dir(&run.id);
        fs::create_dir_all(&run_dir).map_err(|e| HarnessError::io(&run_dir, e))?;

        // Best-effort workspace symlink. Failure is non-fatal: the run
        // is still usable without the convenience link.
        let link_path = run_dir.join(WORKSPACE_LINK);
        let _ = create_workspace_symlink(&run.workspace_path, &link_path);

        // Initialise progress.md with the SPEC §9.6 boilerplate. Done
        // here (and not in RunRunner) so that even resuming an
        // externally-deleted progress.md re-creates a well-formed file
        // on the next run-creation; existing content is preserved.
        ProgressLog::init(self.progress_file(&run.id))?;

        self.save(&run)?;
        // Best-effort: point `current` at the freshly-created run so
        // CLI commands without a `run_id` argument default to it.
        if let Err(e) = self.set_current(&run.id) {
            tracing::warn!(error = %e, "failed to update current run pointer");
        }
        Ok(run)
    }

    /// Load a run by id.
    ///
    /// The loaded `Run.id` is validated against the requested `run_id`
    /// (which is also the directory name). A mismatch surfaces
    /// [`HarnessError::IdMismatch`] rather than silently returning a
    /// run record with a different id than the directory it lives in.
    pub fn load(&self, run_id: &RunId) -> Result<Run, HarnessError> {
        let path = self.run_file(run_id);
        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(HarnessError::RunNotFound {
                    run_id: run_id.as_str().to_owned(),
                });
            }
            Err(e) => return Err(HarnessError::io(&path, e)),
        };
        let loaded: Run =
            toml::from_str(&content).map_err(|e| HarnessError::Deserialize { path, source: e })?;
        if loaded.id != *run_id {
            return Err(HarnessError::IdMismatch {
                expected: run_id.as_str().to_owned(),
                found: loaded.id.as_str().to_owned(),
            });
        }
        Ok(loaded)
    }

    /// Persist a run to disk atomically.
    ///
    /// Writes to `<run-dir>/run.toml.tmp` first, then renames over
    /// `run.toml`. On most platforms `rename` within the same
    /// directory is atomic, so concurrent readers will never observe a
    /// half-written file. If the temp write or rename fails we
    /// best-effort remove the temp file before returning.
    pub fn save(&self, run: &Run) -> Result<(), HarnessError> {
        let run_dir = self.run_dir(&run.id);
        fs::create_dir_all(&run_dir).map_err(|e| HarnessError::io(&run_dir, e))?;

        let path = self.run_file(&run.id);
        let tmp_path = run_dir.join(format!("{RUN_FILENAME}.tmp"));
        let serialized = toml::to_string_pretty(run).map_err(|e| HarnessError::Serialize {
            path: path.clone(),
            source: e,
        })?;

        if let Err(e) = fs::write(&tmp_path, serialized) {
            // Best-effort cleanup; ignore secondary errors.
            let _ = fs::remove_file(&tmp_path);
            return Err(HarnessError::io(&tmp_path, e));
        }
        if let Err(e) = fs::rename(&tmp_path, &path) {
            let _ = fs::remove_file(&tmp_path);
            return Err(HarnessError::io(&path, e));
        }
        Ok(())
    }

    /// Promote an ad-hoc run to harnessed scope, persisting the new
    /// `[scope.harnessed]` section in `run.toml`.
    ///
    /// SPEC §9.3 step 2: writes the auto-promotion metadata
    /// (`upgraded_at`, `upgraded_from_session`, `upgrade_reason`) plus
    /// the relative paths of `features.json` / `init.sh` so a later
    /// process restart can locate the artifacts. The
    /// `workflow_id` is set to the synthetic `"auto-promoted"` label
    /// so listings and TUI surfaces can distinguish a run created
    /// directly as harnessed from one promoted by the auto-gate.
    ///
    /// **Atomicity:** the write reuses [`Self::save`]'s tmp-and-rename
    /// pattern, so concurrent readers never observe a torn `run.toml`.
    /// The on-disk `features.json` / `init.sh` files must already
    /// exist (the `Initializer` subagent writes them); we record only
    /// their relative paths here.
    ///
    /// `current_session` is the 1-indexed session number that was
    /// active when the auto-promotion fired (typically
    /// `runner.run().session_count`). It is stored verbatim in
    /// `upgraded_from_session` so `progress.md` and TUI surfaces can
    /// link the upgrade to the specific session.
    ///
    /// # Errors
    ///
    /// - [`HarnessError::RunNotFound`] when the run id has no
    ///   corresponding `run.toml`.
    /// - [`HarnessError::IdMismatch`] when the loaded record's id
    ///   disagrees with the directory name.
    /// - [`HarnessError::Serialize`] / [`HarnessError::Io`] for write
    ///   failures.
    pub fn upgrade_to_harnessed(
        &self,
        run_id: &RunId,
        current_session: u32,
        upgrade_reason: &str,
    ) -> Result<Run, HarnessError> {
        let mut run = self.load(run_id)?;
        let workflow_id = match &run.scope {
            RunScope::AdHoc => "auto-promoted".to_owned(),
            RunScope::Harnessed { workflow_id, .. } => workflow_id.clone(),
        };
        // Use the existing run's max_sessions if the scope already
        // carried one; otherwise leave it open (None). The auto-gate
        // intentionally does not impose a default cap so the user can
        // configure `[harness] default_max_sessions` separately.
        let existing_cap = match &run.scope {
            RunScope::AdHoc => None,
            RunScope::Harnessed { max_sessions, .. } => *max_sessions,
        };

        run.scope = RunScope::Harnessed {
            workflow_id: workflow_id.clone(),
            max_sessions: existing_cap,
            features_path: Some(FEATURES_FILENAME.to_owned()),
            init_script_path: Some(INIT_SCRIPT_FILENAME.to_owned()),
            upgraded_at: Some(Utc::now()),
            upgraded_from_session: Some(current_session),
            upgrade_reason: Some(upgrade_reason.to_owned()),
        };
        run.workflow_id = Some(workflow_id);
        run.max_sessions = existing_cap;

        self.save(&run)?;
        Ok(run)
    }

    /// Demote a harnessed run back to ad-hoc scope.
    ///
    /// SPEC §9.8 `tmg run downgrade`: flips `Run::scope` back to
    /// [`RunScope::AdHoc`] and clears the top-level `workflow_id` /
    /// `max_sessions` mirrors so the LLM stops seeing harnessed-only
    /// tools on the next registry refresh.
    ///
    /// **Important:** the on-disk `features.json` and `init.sh` files
    /// are intentionally **not** deleted. They are preserved so a
    /// subsequent `tmg run upgrade` can re-promote without losing the
    /// initializer's work, and so post-mortem inspection of an
    /// ad-hoc-now run still has access to the harness artifacts. The
    /// `RunRunnerToolProvider` re-installed by the caller decides
    /// whether the LLM sees those tools — see
    /// [`crate::tools::RunRunnerToolProvider`] for the policy.
    ///
    /// Returns the updated [`Run`] record. A no-op when the run is
    /// already ad-hoc (still re-saves to keep `run.toml` durable).
    ///
    /// # Errors
    ///
    /// - [`HarnessError::RunNotFound`] when the run id has no
    ///   corresponding `run.toml`.
    /// - [`HarnessError::IdMismatch`] when the loaded record's id
    ///   disagrees with the directory name.
    /// - [`HarnessError::Serialize`] / [`HarnessError::Io`] for write
    ///   failures.
    pub fn downgrade_to_ad_hoc(&self, run_id: &RunId) -> Result<Run, HarnessError> {
        let mut run = self.load(run_id)?;
        run.scope = RunScope::AdHoc;
        run.workflow_id = None;
        run.max_sessions = None;
        self.save(&run)?;
        Ok(run)
    }

    /// List all runs as lightweight summaries.
    ///
    /// Entries that fail to load (e.g. missing or malformed `run.toml`)
    /// are skipped to avoid breaking startup on a single bad run
    /// directory; a `tracing::warn!` is emitted for each skipped entry
    /// so operators can investigate.
    pub fn list(&self) -> Result<Vec<RunSummary>, HarnessError> {
        if !self.runs_dir.exists() {
            return Ok(Vec::new());
        }

        let mut summaries = Vec::new();
        let read_dir =
            fs::read_dir(&self.runs_dir).map_err(|e| HarnessError::io(&self.runs_dir, e))?;
        for entry in read_dir {
            let entry = entry.map_err(|e| HarnessError::io(&self.runs_dir, e))?;
            let file_type = entry
                .file_type()
                .map_err(|e| HarnessError::io(entry.path(), e))?;
            if !file_type.is_dir() {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                tracing::warn!(
                    path = %entry.path().display(),
                    "skipping run directory with non-utf8 name"
                );
                continue;
            };
            let run_id = RunId::from_string(name.clone());
            match self.load(&run_id) {
                Ok(run) => summaries.push(RunSummary::from_run(&run)),
                Err(err) => {
                    tracing::warn!(
                        run_id = %name,
                        path = %entry.path().display(),
                        error = %err,
                        "skipping unreadable run directory in list()"
                    );
                }
            }
        }

        // Sort newest-first for predictable resume behaviour.
        summaries.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(summaries)
    }

    /// Return the most recent run that is still resumable, or `None`.
    ///
    /// "Resumable" follows
    /// [`RunStatus::is_resumable`](crate::run::RunStatus::is_resumable):
    /// `Running`, `Paused`, or `Exhausted`. When `workspace_path` is
    /// `Some`, only runs whose stored `workspace_path` matches are
    /// eligible; this prevents resuming a run from a different project
    /// when several projects share the same `runs_dir` configuration.
    ///
    /// As a side-effect, the [`current`](Self::current) pointer is
    /// updated to the returned run so subsequent argument-less CLI
    /// commands target it. Failures to update the pointer are logged
    /// at `warn` and do not propagate.
    pub fn latest_resumable(
        &self,
        workspace_path: Option<&Path>,
    ) -> Result<Option<RunSummary>, HarnessError> {
        let summaries = self.list()?;
        for summary in summaries {
            if !summary.status.is_resumable() {
                continue;
            }
            if let Some(expected) = workspace_path {
                // The summary doesn't carry workspace_path; load the
                // full run to check. This is acceptable: the list is
                // bounded by the number of run directories and the
                // first match wins.
                let run = self.load(&summary.id)?;
                if run.workspace_path != expected {
                    continue;
                }
            }
            // Best-effort: update `current` to the resolved run.
            if let Err(e) = self.set_current(&summary.id) {
                tracing::warn!(error = %e, "failed to update current run pointer");
            }
            return Ok(Some(summary));
        }
        Ok(None)
    }

    /// Path of the `current` pointer at the runs-dir root.
    ///
    /// Symlink form is preferred but a `current.json` pointer file is
    /// used as a fallback when symlink creation fails.
    #[must_use]
    pub fn current_link_path(&self) -> PathBuf {
        self.runs_dir.join(CURRENT_FILENAME)
    }

    /// Path of the JSON-pointer fallback used when symlink creation
    /// fails.
    #[must_use]
    pub fn current_pointer_path(&self) -> PathBuf {
        self.runs_dir.join(CURRENT_POINTER_FILENAME)
    }

    /// Set the `current` pointer to `run_id`.
    ///
    /// Tries to create a symlink at `<runs_dir>/current` pointing at
    /// `<run_id>`. If symlink creation fails (e.g. on platforms where
    /// symlinks require elevated permissions), a JSON pointer file
    /// (`<runs_dir>/current.json`) is written as a fallback so the
    /// CLI's `tmg run resume` / `status` / etc. can still resolve the
    /// most recent active run.
    pub fn set_current(&self, run_id: &RunId) -> Result<(), HarnessError> {
        fs::create_dir_all(&self.runs_dir).map_err(|e| HarnessError::io(&self.runs_dir, e))?;

        let link = self.current_link_path();
        let pointer = self.current_pointer_path();

        // Remove any existing symlink/pointer-file before re-creating
        // so we never observe a stale value mid-update.
        let _ = fs::remove_file(&link);
        let _ = fs::remove_file(&pointer);

        match create_current_symlink(run_id.as_str(), &link) {
            Ok(()) => Ok(()),
            Err(e) => {
                tracing::debug!(
                    error = %e,
                    "current symlink creation failed; falling back to JSON pointer file",
                );
                let payload = serde_json::json!({ "run_id": run_id.as_str() });
                let serialized = serde_json::to_string(&payload).map_err(|src| {
                    HarnessError::Serialize {
                        path: pointer.clone(),
                        // Re-package serde_json error as toml::ser::Error
                        // is awkward; fall back to an `Io` error so the
                        // caller still sees the failure path. We embed
                        // the JSON error message into the chained error.
                        source: toml::ser::Error::custom(src.to_string()),
                    }
                })?;
                fs::write(&pointer, serialized).map_err(|e| HarnessError::io(&pointer, e))?;
                Ok(())
            }
        }
    }

    /// Resolve the `current` pointer back to a [`RunId`], or `None` if
    /// no current pointer is set.
    ///
    /// Tries the symlink first, then the JSON-pointer fallback. A
    /// dangling symlink (target run was deleted) is treated as `None`.
    pub fn current(&self) -> Result<Option<RunId>, HarnessError> {
        let link = self.current_link_path();
        match fs::read_link(&link) {
            Ok(target) => {
                // The symlink target is the run-id (relative path).
                let id_str = target
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(str::to_owned);
                if let Some(id) = id_str {
                    let run_id = RunId::from_string(id);
                    // Verify the target run still exists; treat
                    // dangling links as "no current".
                    if self.run_file(&run_id).exists() {
                        return Ok(Some(run_id));
                    }
                }
                // Fall through to JSON pointer if the symlink is
                // dangling or otherwise unparseable.
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Try the JSON-pointer fallback below.
            }
            Err(_) => {
                // Some other read error (e.g. EINVAL because the entry
                // is not a symlink). Fall through to JSON pointer.
            }
        }

        let pointer = self.current_pointer_path();
        let content = match fs::read_to_string(&pointer) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(HarnessError::io(&pointer, e)),
        };
        let parsed: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    path = %pointer.display(),
                    error = %e,
                    "current.json is not valid JSON; treating as no current",
                );
                return Ok(None);
            }
        };
        let Some(id_str) = parsed.get("run_id").and_then(serde_json::Value::as_str) else {
            return Ok(None);
        };
        let run_id = RunId::from_string(id_str.to_owned());
        if !self.run_file(&run_id).exists() {
            return Ok(None);
        }
        Ok(Some(run_id))
    }
}

/// Decide what to do when a path already exists at `link`.
///
/// If a symlink is already in place we leave it alone. If a regular
/// file or directory is sitting where the workspace symlink should
/// be, we surface a `tracing::warn!` and leave the existing entry
/// alone (returning `Ok(())`) rather than silently overwriting
/// possibly-important user data. Removing the wrong file is a
/// destructive action we don't want to take here; the operator can
/// resolve the conflict manually.
///
/// Returns `Some(())` when no further action is needed (entry is a
/// symlink, or we deferred to a warning), and `None` when the link
/// path is empty and the caller should create the symlink.
fn check_existing_link(link: &Path) -> Option<()> {
    match link.symlink_metadata() {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(_) => {
            // Some other metadata error (e.g. permission). Treat as
            // "needs creation" so the symlink syscall returns the
            // canonical error to the caller.
            None
        }
        Ok(meta) => {
            if meta.file_type().is_symlink() {
                Some(())
            } else {
                tracing::warn!(
                    path = %link.display(),
                    "workspace path exists but is not a symlink; leaving in place"
                );
                Some(())
            }
        }
    }
}

/// Create a `workspace` symlink pointing at `target`. On platforms
/// without symlink support (or if creation fails), this returns the
/// underlying I/O error and the caller is expected to ignore it.
#[cfg(unix)]
fn create_workspace_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    if check_existing_link(link).is_some() {
        return Ok(());
    }
    std::os::unix::fs::symlink(target, link)
}

/// Windows fallback: try a directory symlink, otherwise return an error
/// that the caller will silently ignore.
#[cfg(windows)]
fn create_workspace_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    if check_existing_link(link).is_some() {
        return Ok(());
    }
    std::os::windows::fs::symlink_dir(target, link)
}

#[cfg(not(any(unix, windows)))]
fn create_workspace_symlink(_target: &Path, _link: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "symlinks not supported on this platform",
    ))
}

/// Create a symlink at `link` pointing at `<runs_dir>/<target>`. The
/// link target is a *relative* path (just the run id) so the symlink
/// works irrespective of how the runs-dir is reached.
#[cfg(unix)]
fn create_current_symlink(target: &str, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

#[cfg(windows)]
fn create_current_symlink(target: &str, link: &Path) -> std::io::Result<()> {
    // Windows directory symlinks need an absolute or canonical-ish
    // path; we resolve it relative to the link's parent directory.
    let target_path = match link.parent() {
        Some(parent) => parent.join(target),
        None => std::path::PathBuf::from(target),
    };
    std::os::windows::fs::symlink_dir(&target_path, link)
}

#[cfg(not(any(unix, windows)))]
fn create_current_symlink(_target: &str, _link: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "symlinks not supported on this platform",
    ))
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use crate::run::{RunScope, RunStatus};

    fn make_store() -> (tempfile::TempDir, RunStore) {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let store = RunStore::new(dir.path().join("runs"));
        (dir, store)
    }

    #[test]
    fn create_ad_hoc_initialises_progress_md() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));

        let progress_path = store.progress_file(&run.id);
        assert!(
            progress_path.exists(),
            "progress.md should be created at run init"
        );
        let content = std::fs::read_to_string(&progress_path).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(content, "# Progress Log\n");
    }

    #[test]
    fn create_ad_hoc_writes_run_toml() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace.clone(), Some("hello"))
            .unwrap_or_else(|e| panic!("{e}"));
        let path = store.run_file(&run.id);
        assert!(path.exists(), "run.toml should exist at {path:?}");

        let loaded = store.load(&run.id).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(loaded, run);
        assert_eq!(loaded.scope, RunScope::AdHoc);
        assert_eq!(loaded.status, RunStatus::Running);
        assert_eq!(
            loaded.inputs.get("initial_request"),
            Some(&serde_json::Value::String("hello".to_owned()))
        );
    }

    #[cfg(unix)]
    #[test]
    fn workspace_symlink_is_created() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace.clone(), None)
            .unwrap_or_else(|e| panic!("{e}"));
        let link = store.run_dir(&run.id).join(WORKSPACE_LINK);
        let meta = std::fs::symlink_metadata(&link).unwrap_or_else(|e| panic!("{e}"));
        assert!(meta.file_type().is_symlink(), "workspace link missing");
    }

    #[test]
    fn save_round_trips_run() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        let mut run = Run::new_ad_hoc(workspace);
        run.session_count = 3;
        store.save(&run).unwrap_or_else(|e| panic!("{e}"));
        let loaded = store.load(&run.id).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(loaded.session_count, 3);
    }

    #[test]
    fn list_returns_runs_newest_first() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");

        let run1 = store
            .create_ad_hoc(workspace.clone(), None)
            .unwrap_or_else(|e| panic!("{e}"));
        // Force a slightly later timestamp.
        std::thread::sleep(std::time::Duration::from_millis(10));
        let run2 = store
            .create_ad_hoc(workspace.clone(), None)
            .unwrap_or_else(|e| panic!("{e}"));

        let listed = store.list().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(listed.len(), 2);
        // Newest first.
        assert_eq!(listed[0].id, run2.id);
        assert_eq!(listed[1].id, run1.id);
    }

    #[test]
    fn list_returns_empty_when_dir_missing() {
        let (_tmp, store) = make_store();
        let listed = store.list().unwrap_or_else(|e| panic!("{e}"));
        assert!(listed.is_empty());
    }

    #[test]
    fn latest_resumable_skips_terminal_runs() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");

        let mut completed = store
            .create_ad_hoc(workspace.clone(), None)
            .unwrap_or_else(|e| panic!("{e}"));
        completed.status = RunStatus::Completed;
        store.save(&completed).unwrap_or_else(|e| panic!("{e}"));

        std::thread::sleep(std::time::Duration::from_millis(10));
        let running = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));

        let resumable = store
            .latest_resumable(None)
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_else(|| panic!("expected a resumable run"));
        assert_eq!(resumable.id, running.id);
    }

    #[test]
    fn latest_resumable_filters_by_workspace() {
        let (tmp, store) = make_store();
        let workspace_a = tmp.path().join("workspace-a");
        let workspace_b = tmp.path().join("workspace-b");

        // A resumable run exists, but for workspace_a.
        let run_a = store
            .create_ad_hoc(workspace_a.clone(), None)
            .unwrap_or_else(|e| panic!("{e}"));

        // Asking for workspace_b's resumable run should return None
        // (we should fall back to creating a new run).
        let result = store
            .latest_resumable(Some(&workspace_b))
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(
            result.is_none(),
            "should not resume across different workspace"
        );

        // Asking for workspace_a should still find it.
        let result = store
            .latest_resumable(Some(&workspace_a))
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_else(|| panic!("expected a resumable run for workspace_a"));
        assert_eq!(result.id, run_a.id);
    }

    #[test]
    fn load_missing_returns_run_not_found() {
        let (_tmp, store) = make_store();
        let result = store.load(&RunId::from_string("deadbeef"));
        assert!(matches!(
            result,
            Err(HarnessError::RunNotFound { ref run_id }) if run_id == "deadbeef"
        ));
    }

    #[test]
    fn load_detects_id_mismatch() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));

        // Hand-edit run.toml so the id inside no longer matches the directory.
        let path = store.run_file(&run.id);
        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
        let tampered = content.replacen(run.id.as_str(), "ffffffff", 1);
        std::fs::write(&path, tampered).unwrap_or_else(|e| panic!("{e}"));

        let result = store.load(&run.id);
        match result {
            Err(HarnessError::IdMismatch { expected, found }) => {
                assert_eq!(expected, run.id.as_str());
                assert_eq!(found, "ffffffff");
            }
            other => panic!("expected IdMismatch, got {other:?}"),
        }
    }

    #[test]
    fn upgrade_to_harnessed_flips_scope_and_records_metadata() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(run.scope, RunScope::AdHoc);

        let upgraded = store
            .upgrade_to_harnessed(&run.id, 2, "spans crates")
            .unwrap_or_else(|e| panic!("{e}"));

        match upgraded.scope {
            RunScope::Harnessed {
                workflow_id,
                features_path,
                init_script_path,
                upgraded_at,
                upgraded_from_session,
                upgrade_reason,
                ..
            } => {
                assert_eq!(workflow_id, "auto-promoted");
                assert_eq!(features_path.as_deref(), Some("features.json"));
                assert_eq!(init_script_path.as_deref(), Some("init.sh"));
                assert!(upgraded_at.is_some());
                assert_eq!(upgraded_from_session, Some(2));
                assert_eq!(upgrade_reason.as_deref(), Some("spans crates"));
            }
            RunScope::AdHoc => panic!("expected harnessed, got ad-hoc"),
        }

        // The on-disk record reflects the upgrade.
        let reloaded = store.load(&run.id).unwrap_or_else(|e| panic!("{e}"));
        assert!(matches!(reloaded.scope, RunScope::Harnessed { .. }));
        assert_eq!(reloaded.workflow_id.as_deref(), Some("auto-promoted"));
    }

    #[test]
    fn create_ad_hoc_sets_current_pointer() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));

        let resolved = store
            .current()
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_else(|| panic!("expected current pointer to be set"));
        assert_eq!(resolved, run.id);
    }

    #[test]
    fn set_current_overwrites_previous_pointer() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run_a = store
            .create_ad_hoc(workspace.clone(), None)
            .unwrap_or_else(|e| panic!("{e}"));
        std::thread::sleep(std::time::Duration::from_millis(5));
        let run_b = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));

        // After two `create_ad_hoc` calls the pointer should track the
        // most recent one.
        let resolved = store
            .current()
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_else(|| panic!("expected current pointer to be set"));
        assert_eq!(resolved, run_b.id);
        assert_ne!(resolved, run_a.id);
    }

    #[test]
    fn current_returns_none_when_unset() {
        let (_tmp, store) = make_store();
        // No runs created yet; no pointer should exist.
        let resolved = store.current().unwrap_or_else(|e| panic!("{e}"));
        assert!(resolved.is_none());
    }

    #[test]
    fn current_returns_none_for_dangling_pointer() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));

        // Delete the run directory underneath the pointer.
        std::fs::remove_dir_all(store.run_dir(&run.id)).unwrap_or_else(|e| panic!("{e}"));

        let resolved = store.current().unwrap_or_else(|e| panic!("{e}"));
        assert!(
            resolved.is_none(),
            "dangling current pointer should resolve to None"
        );
    }

    #[test]
    fn save_writes_no_lingering_tmp_file() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        let run = Run::new_ad_hoc(workspace);
        store.save(&run).unwrap_or_else(|e| panic!("{e}"));

        let tmp_path = store.run_dir(&run.id).join(format!("{RUN_FILENAME}.tmp"));
        assert!(
            !tmp_path.exists(),
            "atomic save should not leave run.toml.tmp behind"
        );
    }
}
