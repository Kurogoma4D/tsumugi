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

use crate::error::HarnessError;
use crate::run::{Run, RunId, RunSummary};

/// Filename used for the persistent run record under each run directory.
pub const RUN_FILENAME: &str = "run.toml";

/// Filename of the workspace symlink under each run directory.
pub const WORKSPACE_LINK: &str = "workspace";

/// Persistent store for runs rooted at a directory (typically
/// `.tsumugi/runs/`).
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

        self.save(&run)?;
        Ok(run)
    }

    /// Load a run by id.
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
        toml::from_str(&content).map_err(|e| HarnessError::Deserialize { path, source: e })
    }

    /// Persist a run to disk.
    pub fn save(&self, run: &Run) -> Result<(), HarnessError> {
        let run_dir = self.run_dir(&run.id);
        fs::create_dir_all(&run_dir).map_err(|e| HarnessError::io(&run_dir, e))?;

        let path = self.run_file(&run.id);
        let serialized = toml::to_string_pretty(run).map_err(|e| HarnessError::Serialize {
            path: path.clone(),
            source: e,
        })?;
        fs::write(&path, serialized).map_err(|e| HarnessError::io(&path, e))?;
        Ok(())
    }

    /// List all runs as lightweight summaries.
    ///
    /// Entries that fail to load (e.g. missing or malformed `run.toml`)
    /// are skipped silently to avoid breaking startup on a single bad
    /// run directory.
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
                continue;
            };
            let run_id = RunId::from_string(name);
            if let Ok(run) = self.load(&run_id) {
                summaries.push(RunSummary::from_run(&run));
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
    /// `Running`, `Paused`, or `Exhausted`.
    pub fn latest_resumable(&self) -> Result<Option<RunSummary>, HarnessError> {
        let summaries = self.list()?;
        Ok(summaries.into_iter().find(|s| s.status.is_resumable()))
    }
}

/// Create a `workspace` symlink pointing at `target`. On platforms
/// without symlink support (or if creation fails), this returns the
/// underlying I/O error and the caller is expected to ignore it.
#[cfg(unix)]
fn create_workspace_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    if link.exists() || link.symlink_metadata().is_ok() {
        return Ok(());
    }
    std::os::unix::fs::symlink(target, link)
}

/// Windows fallback: try a directory symlink, otherwise return an error
/// that the caller will silently ignore.
#[cfg(windows)]
fn create_workspace_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    if link.exists() || link.symlink_metadata().is_ok() {
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
            .latest_resumable()
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_else(|| panic!("expected a resumable run"));
        assert_eq!(resumable.id, running.id);
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
}
