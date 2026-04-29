//! [`MemoryStore`]: filesystem-backed CRUD for project / global memory.
//!
//! The store owns a project memory directory (e.g. `.tsumugi/memory/`)
//! and optionally a low-priority global directory (e.g.
//! `~/.config/tsumugi/memory/`). Reads merge both layers (project wins
//! on name collision); writes always target the project layer.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use chrono::Utc;
use regex::Regex;
use tracing::debug;

use crate::budget::{BudgetReport, MemoryBudget};
use crate::entry::{Frontmatter, MemoryEntry, MemoryType, parse_entry};
use crate::error::MemoryError;

/// File name of the index inside each memory directory.
pub const INDEX_FILE_NAME: &str = "MEMORY.md";

/// Origin layer of a [`MemoryEntry`] discovered by the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryScope {
    /// Project-local memory at `<project_root>/.tsumugi/memory/`.
    Project,
    /// Global memory at `~/.config/tsumugi/memory/`.
    Global,
}

impl MemoryScope {
    /// Lowercase string used in event log payloads / display.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Project => "project",
            Self::Global => "global",
        }
    }
}

/// Filesystem-backed memory store.
///
/// Construction is cheap (no I/O); the directory is created on first
/// write. The store is `Send + Sync` and intended to be wrapped in
/// `Arc` for shared access from the agent loop and CLI tools.
///
/// Concurrent writes within the same process are serialised via an
/// internal [`AsyncMutex`] held only for the duration of each
/// mutation (so the index file cannot be corrupted by interleaved
/// read-modify-write sequences when the agent loop dispatches multiple
/// `memory` tool calls in parallel via `JoinSet`). Cross-process
/// safety is **not** provided: multiple `tmg` processes editing the
/// same store concurrently can still race; this is acceptable because
/// the memory store is project-scoped and there is no production use
/// case for multiple `tmg` agents writing the same `.tsumugi/memory/`
/// at once.
#[derive(Debug, Clone)]
pub struct MemoryStore {
    project_dir: PathBuf,
    global_dir: Option<PathBuf>,
    budget: MemoryBudget,
    /// In-process mutation lock. Cloned via `Arc` so every clone of the
    /// store sees the same lock state (the `Clone` derive on `Arc` is a
    /// pointer bump, not a deep copy). Held only for the brief duration
    /// of each synchronous read-modify-write of `MEMORY.md`; tools
    /// dispatched via `JoinSet` therefore see consistent index state
    /// even when their `memory add` calls overlap.
    write_lock: Arc<Mutex<()>>,
}

impl MemoryStore {
    /// Open a store rooted at `project_root`.
    ///
    /// The project directory is `<project_root>/.tsumugi/memory/`.
    /// The global directory defaults to `~/.config/tsumugi/memory/`
    /// (via the [`dirs`] crate) when available.
    ///
    /// The returned store has run [`Self::reconcile`] best-effort: any
    /// drift between the on-disk markdown files and the index rows is
    /// repaired so `MEMORY.md` always reflects reality. Reconciliation
    /// failure is logged via `tracing::debug!` and otherwise ignored
    /// (the store still functions, just without the self-heal).
    #[must_use]
    pub fn open(project_root: impl AsRef<Path>) -> Self {
        let project_dir = project_root.as_ref().join(".tsumugi").join("memory");
        let global_dir = dirs::config_dir().map(|d| d.join("tsumugi").join("memory"));
        let store = Self {
            project_dir,
            global_dir,
            budget: MemoryBudget::default(),
            write_lock: Arc::new(Mutex::new(())),
        };
        if let Err(e) = store.reconcile() {
            debug!(error = %e, "memory store reconcile on open failed (ignoring)");
        }
        store
    }

    /// Construct a store with explicit project / global directories
    /// and a custom budget. Used by the CLI when honouring the
    /// `[memory]` section of `tsumugi.toml`.
    ///
    /// Like [`Self::open`], runs [`Self::reconcile`] best-effort so
    /// callers do not need to remember to invoke it themselves.
    #[must_use]
    pub fn with_dirs(
        project_dir: impl Into<PathBuf>,
        global_dir: Option<PathBuf>,
        budget: MemoryBudget,
    ) -> Self {
        let store = Self {
            project_dir: project_dir.into(),
            global_dir,
            budget,
            write_lock: Arc::new(Mutex::new(())),
        };
        if let Err(e) = store.reconcile() {
            debug!(error = %e, "memory store reconcile on open failed (ignoring)");
        }
        store
    }

    /// Override the budget after construction.
    #[must_use]
    pub fn with_budget(mut self, budget: MemoryBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Path to the project memory directory.
    #[must_use]
    pub fn project_dir(&self) -> &Path {
        &self.project_dir
    }

    /// Path to the global memory directory (if any).
    #[must_use]
    pub fn global_dir(&self) -> Option<&Path> {
        self.global_dir.as_deref()
    }

    /// Path to the project `MEMORY.md` file.
    #[must_use]
    pub fn project_index_path(&self) -> PathBuf {
        self.project_dir.join(INDEX_FILE_NAME)
    }

    /// Active capacity budget.
    #[must_use]
    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    /// Read the merged `MEMORY.md` index (global followed by project,
    /// with project entries overriding any global rows whose link
    /// target collides on file name).
    ///
    /// Returns an empty string when neither index exists.
    ///
    /// # Errors
    ///
    /// Surfaces I/O errors other than `NotFound`.
    pub fn read_merged_index(&self) -> Result<String, MemoryError> {
        let project = match read_index_at(&self.project_dir) {
            Ok(text) => Some(text),
            Err(MemoryError::Io { source, .. })
                if source.kind() == std::io::ErrorKind::NotFound =>
            {
                None
            }
            Err(e) => return Err(e),
        };
        let global = match self.global_dir.as_deref() {
            Some(dir) => match read_index_at(dir) {
                Ok(text) => Some(text),
                Err(MemoryError::Io { source, .. })
                    if source.kind() == std::io::ErrorKind::NotFound =>
                {
                    None
                }
                Err(e) => return Err(e),
            },
            None => None,
        };

        let project_names = project
            .as_deref()
            .map(extract_index_names)
            .unwrap_or_default();

        let mut out = String::new();
        if let Some(g) = global.as_deref() {
            // Drop global lines whose `(file.md)` token collides with a
            // project entry. Lines that do not match the index format
            // are passed through unchanged (e.g. user-added headers
            // like `# My Memory` should not be relabelled `[global] # My Memory`).
            for line in g.lines() {
                if let Some(file) = extract_link_target(line)
                    && project_names.iter().any(|p| p == &file)
                {
                    continue;
                }
                let is_link_row = extract_link_target(line).is_some();
                if is_link_row && !line.trim().is_empty() {
                    out.push_str("[global] ");
                }
                out.push_str(line);
                out.push('\n');
            }
        }
        if let Some(p) = project.as_deref() {
            if !out.is_empty() && !out.ends_with('\n') {
                out.push('\n');
            }
            for line in p.lines() {
                out.push_str(line);
                out.push('\n');
            }
        }
        Ok(out)
    }

    /// Take a [`BudgetReport`] snapshot of the project store. Index
    /// line count and file count are measured against [`Self::budget`].
    ///
    /// # Errors
    ///
    /// Surfaces I/O errors other than `NotFound`.
    pub fn budget_report(&self) -> Result<BudgetReport, MemoryError> {
        let index_lines = match read_index_at(&self.project_dir) {
            Ok(s) => s.lines().filter(|l| !l.trim().is_empty()).count(),
            Err(MemoryError::Io { source, .. })
                if source.kind() == std::io::ErrorKind::NotFound =>
            {
                0
            }
            Err(e) => return Err(e),
        };
        let file_count = count_entry_files(&self.project_dir)?;
        Ok(BudgetReport::from_measurements(
            index_lines,
            file_count,
            &self.budget,
        ))
    }

    /// Read one memory entry by name.
    ///
    /// Project layer wins on name collision; if the entry is missing
    /// from project, the global layer is consulted.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NotFound`] when no entry matches in
    /// either layer; other I/O / parse failures surface as their
    /// respective variants.
    pub fn read(&self, name: &str) -> Result<(MemoryEntry, MemoryScope), MemoryError> {
        validate_name(name)?;
        let file = entry_file_name(name);
        let project_path = self.project_dir.join(&file);
        match read_entry_file(&project_path) {
            Ok(entry) => return Ok((entry, MemoryScope::Project)),
            Err(MemoryError::Io { source, .. })
                if source.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        if let Some(dir) = self.global_dir.as_deref() {
            let global_path = dir.join(&file);
            match read_entry_file(&global_path) {
                Ok(entry) => return Ok((entry, MemoryScope::Global)),
                Err(MemoryError::Io { source, .. })
                    if source.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        }
        Err(MemoryError::NotFound {
            name: name.to_owned(),
        })
    }

    /// Add a new memory entry. Returns an error if an entry with the
    /// same `name` already exists in the project layer.
    ///
    /// Writes the entry file and appends a row to `MEMORY.md`.
    ///
    /// # Errors
    ///
    /// - [`MemoryError::AlreadyExists`] on name collision.
    /// - [`MemoryError::InvalidName`] on bad input.
    /// - [`MemoryError::Io`] / [`MemoryError::Yaml`] on persist failure.
    pub fn add(
        &self,
        name: &str,
        kind: MemoryType,
        description: &str,
        content: &str,
    ) -> Result<BudgetReport, MemoryError> {
        validate_name(name)?;
        validate_description(description)?;

        // Hold the in-process write lock for the duration of the
        // read-modify-write so concurrent `memory add` calls
        // (dispatched through the agent loop's JoinSet) cannot drop
        // index rows. `lock()` returns a poisoned guard if a previous
        // holder panicked; we ignore the poison and proceed because the
        // protected data lives entirely on disk and is reconciled from
        // there.
        let _guard = self
            .write_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let path = self.entry_path_in_project(name);
        if path.exists() {
            return Err(MemoryError::AlreadyExists {
                name: name.to_owned(),
            });
        }

        ensure_dir(&self.project_dir)?;

        let now = Utc::now();
        let entry = MemoryEntry {
            frontmatter: Frontmatter {
                name: name.to_owned(),
                description: description.to_owned(),
                kind,
                created_at: Some(now),
                updated_at: Some(now),
            },
            body: content.to_owned(),
        };
        let markdown = entry.to_markdown()?;
        std::fs::write(&path, markdown)
            .map_err(|e| MemoryError::io(format!("writing {}", path.display()), e))?;

        let new_line = format_index_line(name, description);
        // Use `replace_index_line` rather than `append_index_line` so a
        // stale row left behind by a manual file deletion + re-add is
        // updated in place rather than duplicated. `replace_index_line`
        // appends when no existing row matches, so the first-add path
        // still works.
        replace_index_line(&self.project_index_path(), name, Some(&new_line))?;
        debug!(name = %name, "memory add: wrote entry and updated index");

        self.budget_report()
    }

    /// Update an existing memory entry. `description` and `content`
    /// are partial updates: pass `None` to leave that field unchanged.
    /// `kind` is updated when `Some`.
    ///
    /// `updated_at` is bumped on every successful update.
    ///
    /// # Errors
    ///
    /// - [`MemoryError::NotFound`] when the entry is absent from the
    ///   project layer (global entries are read-only).
    pub fn update(
        &self,
        name: &str,
        kind: Option<MemoryType>,
        description: Option<&str>,
        content: Option<&str>,
    ) -> Result<BudgetReport, MemoryError> {
        validate_name(name)?;
        if let Some(d) = description {
            validate_description(d)?;
        }

        let _guard = self
            .write_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let path = self.entry_path_in_project(name);
        if !path.exists() {
            return Err(MemoryError::NotFound {
                name: name.to_owned(),
            });
        }

        let existing_text = std::fs::read_to_string(&path)
            .map_err(|e| MemoryError::io(format!("reading {}", path.display()), e))?;
        let mut entry = parse_entry(&existing_text, &path.display().to_string())?;

        if let Some(k) = kind {
            entry.frontmatter.kind = k;
        }
        if let Some(d) = description {
            d.clone_into(&mut entry.frontmatter.description);
        }
        if let Some(c) = content {
            c.clone_into(&mut entry.body);
        }
        let now = Utc::now();
        if entry.frontmatter.created_at.is_none() {
            entry.frontmatter.created_at = Some(now);
        }
        entry.frontmatter.updated_at = Some(now);

        let markdown = entry.to_markdown()?;
        std::fs::write(&path, markdown)
            .map_err(|e| MemoryError::io(format!("writing {}", path.display()), e))?;

        // Refresh index row (description may have changed).
        let new_line = format_index_line(name, &entry.frontmatter.description);
        replace_index_line(&self.project_index_path(), name, Some(&new_line))?;
        debug!(name = %name, "memory update: rewrote entry and refreshed index");

        self.budget_report()
    }

    /// Remove a memory entry. Idempotent: missing entries are an error.
    ///
    /// # Errors
    ///
    /// - [`MemoryError::NotFound`] when the entry is absent.
    pub fn remove(&self, name: &str) -> Result<BudgetReport, MemoryError> {
        validate_name(name)?;

        let _guard = self
            .write_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let path = self.entry_path_in_project(name);
        if !path.exists() {
            return Err(MemoryError::NotFound {
                name: name.to_owned(),
            });
        }
        std::fs::remove_file(&path)
            .map_err(|e| MemoryError::io(format!("removing {}", path.display()), e))?;
        replace_index_line(&self.project_index_path(), name, None)?;
        debug!(name = %name, "memory remove: deleted entry and dropped index row");
        self.budget_report()
    }

    /// List the names of every entry in the project layer (no global merge).
    ///
    /// # Errors
    ///
    /// Surfaces I/O errors other than `NotFound`.
    pub fn list_project(&self) -> Result<Vec<String>, MemoryError> {
        let read_dir = match std::fs::read_dir(&self.project_dir) {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => {
                return Err(MemoryError::io(
                    format!("reading {}", self.project_dir.display()),
                    e,
                ));
            }
        };
        let mut names = Vec::new();
        for ent in read_dir {
            let ent = ent.map_err(|e| {
                MemoryError::io(format!("iterating {}", self.project_dir.display()), e)
            })?;
            let name_os = ent.file_name();
            let Some(name_str) = name_os.to_str() else {
                continue;
            };
            if name_str == INDEX_FILE_NAME {
                continue;
            }
            if let Some(stem) = name_str.strip_suffix(".md") {
                names.push(stem.to_owned());
            }
        }
        names.sort();
        Ok(names)
    }

    fn entry_path_in_project(&self, name: &str) -> PathBuf {
        self.project_dir.join(entry_file_name(name))
    }

    /// Best-effort self-heal between the on-disk markdown files and
    /// the `MEMORY.md` index. Called automatically by [`Self::open`]
    /// and [`Self::with_dirs`] so callers do not need to invoke it
    /// themselves; exposed as `pub` so tests and recovery tooling can
    /// trigger it on demand.
    ///
    /// Concretely:
    ///
    /// - Index rows that point at a missing `<name>.md` are dropped.
    /// - Entry files without a matching index row get an appended row
    ///   constructed from their frontmatter `description`.
    ///
    /// The reconciliation does **not** mutate entry-file contents — it
    /// only edits the index. Files with malformed frontmatter are
    /// silently skipped (logged at `tracing::debug!`); a future repair
    /// command can be added if more aggressive recovery is needed.
    ///
    /// Holds the in-process write lock so concurrent CRUD calls cannot
    /// observe a partial reconciliation state.
    ///
    /// # Errors
    ///
    /// Returns I/O errors from reading the project directory or
    /// writing `MEMORY.md`. Errors that would require interactive
    /// recovery (malformed frontmatter on a single file) are skipped
    /// silently; the caller can run `tmg memory show <name>` to
    /// surface them.
    pub fn reconcile(&self) -> Result<(), MemoryError> {
        let _guard = self
            .write_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // Project dir absent? Nothing to reconcile.
        if !self.project_dir.exists() {
            return Ok(());
        }

        // 1. Enumerate on-disk entry files and read their descriptions.
        let mut on_disk: Vec<(String, String)> = Vec::new();
        let read_dir = match std::fs::read_dir(&self.project_dir) {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => {
                return Err(MemoryError::io(
                    format!("reading {}", self.project_dir.display()),
                    e,
                ));
            }
        };
        for ent in read_dir {
            let Ok(ent) = ent else { continue };
            let name_os = ent.file_name();
            let Some(name_str) = name_os.to_str() else {
                continue;
            };
            if name_str == INDEX_FILE_NAME {
                continue;
            }
            let Some(stem) = name_str.strip_suffix(".md") else {
                continue;
            };
            let path = ent.path();
            let Ok(text) = std::fs::read_to_string(&path) else {
                debug!(path = %path.display(), "reconcile: failed to read entry file");
                continue;
            };
            match parse_entry(&text, &path.display().to_string()) {
                Ok(entry) => {
                    on_disk.push((stem.to_owned(), entry.frontmatter.description));
                }
                Err(e) => {
                    debug!(path = %path.display(), error = %e, "reconcile: skipping malformed entry");
                }
            }
        }

        // 2. Read the existing index (may not exist).
        let index_path = self.project_index_path();
        let existing = match std::fs::read_to_string(&index_path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => {
                return Err(MemoryError::io(
                    format!("reading {}", index_path.display()),
                    e,
                ));
            }
        };

        let on_disk_names: Vec<String> = on_disk.iter().map(|(n, _)| n.clone()).collect();

        // 3. Rebuild the index: keep non-link rows verbatim, drop link
        //    rows whose target is missing, and append rows for files
        //    that lack one.
        let mut out = String::with_capacity(existing.len());
        let mut covered: Vec<String> = Vec::new();
        for line in existing.lines() {
            if let Some(file) = extract_link_target(line) {
                let stem_owned = file
                    .strip_suffix(".md")
                    .map_or_else(|| file.clone(), str::to_owned);
                if on_disk_names.iter().any(|n| n == &stem_owned) {
                    out.push_str(line);
                    out.push('\n');
                    covered.push(stem_owned);
                }
                // Else: drop the orphan row.
            } else {
                // Pass-through (header, blank line, comment, etc.).
                out.push_str(line);
                out.push('\n');
            }
        }
        for (name, desc) in &on_disk {
            if !covered.iter().any(|c| c == name) {
                if !out.is_empty() && !out.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str(&format_index_line(name, desc));
                out.push('\n');
            }
        }

        // 4. Only write if something changed and we have entries (or
        //    a previously-existing index file). Avoid creating an empty
        //    `MEMORY.md` on a fresh, empty project directory.
        if out != existing && (!out.is_empty() || index_path.exists()) {
            ensure_dir(&self.project_dir)?;
            std::fs::write(&index_path, &out)
                .map_err(|e| MemoryError::io(format!("writing {}", index_path.display()), e))?;
            debug!(path = %index_path.display(), "memory reconcile: rewrote index");
        }

        Ok(())
    }
}

fn entry_file_name(name: &str) -> String {
    format!("{name}.md")
}

fn read_index_at(dir: &Path) -> Result<String, MemoryError> {
    let path = dir.join(INDEX_FILE_NAME);
    std::fs::read_to_string(&path)
        .map_err(|e| MemoryError::io(format!("reading {}", path.display()), e))
}

fn read_entry_file(path: &Path) -> Result<MemoryEntry, MemoryError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| MemoryError::io(format!("reading {}", path.display()), e))?;
    parse_entry(&text, &path.display().to_string())
}

fn count_entry_files(dir: &Path) -> Result<usize, MemoryError> {
    let read_dir = match std::fs::read_dir(dir) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(MemoryError::io(format!("reading {}", dir.display()), e)),
    };
    let mut count = 0;
    for ent in read_dir {
        let ent = ent.map_err(|e| MemoryError::io(format!("iterating {}", dir.display()), e))?;
        let name = ent.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        if name_str == INDEX_FILE_NAME {
            continue;
        }
        if std::path::Path::new(name_str)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("md"))
        {
            count += 1;
        }
    }
    Ok(count)
}

fn ensure_dir(dir: &Path) -> Result<(), MemoryError> {
    std::fs::create_dir_all(dir)
        .map_err(|e| MemoryError::io(format!("creating {}", dir.display()), e))
}

fn format_index_line(name: &str, description: &str) -> String {
    let file = entry_file_name(name);
    format!("- [{name}]({file}) — {description}")
}

// `append_index_line` was used by the previous `add` path but issue
// #13 of PR #76 review moved `add` to `replace_index_line` (which
// dedupes stale rows on the way through), so this helper is unused.
// Kept only as a comment marker; the function has been removed to
// silence dead-code warnings.

fn replace_index_line(
    index_path: &Path,
    name: &str,
    replacement: Option<&str>,
) -> Result<(), MemoryError> {
    let current = match std::fs::read_to_string(index_path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Index file doesn't exist yet. If we have a replacement
            // line, we need to create the file so the row is
            // persisted. Without a replacement (i.e. we're removing a
            // row that was never indexed) the missing file is
            // already the correct state.
            if replacement.is_none() {
                return Ok(());
            }
            String::new()
        }
        Err(e) => {
            return Err(MemoryError::io(
                format!("reading {}", index_path.display()),
                e,
            ));
        }
    };
    let target_file = entry_file_name(name);
    let mut out = String::with_capacity(current.len());
    let mut replaced = false;
    for line in current.lines() {
        if extract_link_target(line).as_deref() == Some(target_file.as_str()) {
            if let Some(new_line) = replacement
                && !replaced
            {
                out.push_str(new_line);
                out.push('\n');
                replaced = true;
            }
            // remove==None: drop the line; replace==Some & already replaced: dedupe.
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    if let Some(new_line) = replacement
        && !replaced
    {
        // Entry was missing from the index; append so callers always
        // see the row reflect reality.
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(new_line);
        out.push('\n');
    }
    std::fs::write(index_path, out)
        .map_err(|e| MemoryError::io(format!("writing {}", index_path.display()), e))
}

/// Cached compiled regex for index link rows. Pattern matches a
/// strict markdown link-list item where the `(...)` target ends in
/// `.md`. This deliberately rejects user-added comments like
/// `<!-- index of memory (auto-generated) -->` so they round-trip
/// through `read_merged_index` unchanged.
fn link_row_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // `^\s*-\s*\[name\]\(target.md\)` with the `target.md`
        // captured. Trailing description / em-dash text is allowed.
        // The pattern is anchored so non-link lines (headers, blank
        // lines, freeform comments) do not match.
        #[expect(
            clippy::expect_used,
            reason = "static regex literal cannot fail at runtime"
        )]
        Regex::new(r"^\s*-\s*\[[^\]]+\]\(([^)]+\.md)\)").expect("static regex compiles")
    })
}

/// Extract the file name inside the `(...)` of an index line.
///
/// Returns `None` for lines that do not look like a markdown link list
/// item with a `.md` target; user-added headers, comments, and blank
/// lines are passed through unchanged by the merge logic.
fn extract_link_target(line: &str) -> Option<String> {
    let captures = link_row_regex().captures(line)?;
    Some(captures.get(1)?.as_str().to_owned())
}

fn extract_index_names(index: &str) -> Vec<String> {
    index.lines().filter_map(extract_link_target).collect()
}

fn validate_name(name: &str) -> Result<(), MemoryError> {
    if name.is_empty() {
        return Err(MemoryError::invalid_name(name, "name must not be empty"));
    }
    if name.contains('/') || name.contains('\\') {
        return Err(MemoryError::invalid_name(
            name,
            "name must not contain path separators",
        ));
    }
    if name.contains("..") {
        return Err(MemoryError::invalid_name(
            name,
            "name must not contain '..'",
        ));
    }
    if name.starts_with('.') {
        return Err(MemoryError::invalid_name(
            name,
            "name must not start with '.'",
        ));
    }
    // The literal stem of the index file must not be reused as an entry
    // name; otherwise `entry_file_name(name)` would collide with
    // `INDEX_FILE_NAME` (`MEMORY.md`) and the entry write would either
    // overwrite the index or be overwritten by the next index append.
    // The check is case-insensitive because macOS / Windows have
    // case-insensitive filesystems by default.
    let index_stem = INDEX_FILE_NAME.trim_end_matches(".md");
    if name.eq_ignore_ascii_case(index_stem) {
        return Err(MemoryError::invalid_name(
            name,
            "name must not collide with the reserved index file stem (MEMORY)",
        ));
    }
    let valid = name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if !valid {
        return Err(MemoryError::invalid_name(
            name,
            "name must be ASCII alphanumeric with underscores or hyphens",
        ));
    }
    Ok(())
}

fn validate_description(description: &str) -> Result<(), MemoryError> {
    if description.trim().is_empty() {
        return Err(MemoryError::invalid_description(
            "description must not be empty",
        ));
    }
    if description.contains('\n') {
        return Err(MemoryError::invalid_description(
            "description must be a single line",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;

    fn store_at(root: &Path) -> MemoryStore {
        MemoryStore::with_dirs(
            root.join(".tsumugi").join("memory"),
            None,
            MemoryBudget::default(),
        )
    }

    #[test]
    fn validate_name_rejects_path_traversal() {
        assert!(validate_name("../escape").is_err());
        assert!(validate_name("foo/bar").is_err());
        assert!(validate_name("").is_err());
        assert!(validate_name(".hidden").is_err());
        assert!(validate_name("ok_name").is_ok());
        assert!(validate_name("ok-name-1").is_ok());
    }

    /// Regression test for issue #1 in PR #76 review: an entry named
    /// `MEMORY` (or any case variant) collides with the index file
    /// stem. Without this guard, `add("MEMORY", ...)` would write the
    /// entry as `MEMORY.md` and the subsequent index append would
    /// overwrite the brand-new entry.
    #[test]
    fn validate_name_rejects_reserved_index_stem() {
        assert!(validate_name("MEMORY").is_err());
        assert!(validate_name("memory").is_err());
        assert!(validate_name("Memory").is_err());
        assert!(validate_name("MeMoRy").is_err());
        // Substrings that contain the stem are still allowed.
        assert!(validate_name("MEMORY_2").is_ok());
        assert!(validate_name("project_memory").is_ok());
    }

    /// End-to-end regression test: even if the validator gate was
    /// bypassed, calling `add("MEMORY", ...)` must not corrupt the
    /// index. `add` returns `MemoryError::InvalidName` and
    /// `MEMORY.md` is left untouched (still empty / absent).
    #[test]
    fn add_with_reserved_name_does_not_corrupt_index() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = store_at(dir.path());
        // Seed a real entry so the index has known content to detect
        // corruption against.
        store
            .add("real_topic", MemoryType::Project, "real desc", "body")
            .expect("seed");
        let before = std::fs::read_to_string(store.project_index_path()).expect("read index");

        // Both the upper-case and lower-case forms must reject.
        let err = store
            .add("MEMORY", MemoryType::Project, "d", "c")
            .expect_err("MEMORY");
        assert!(
            matches!(err, MemoryError::InvalidName { .. }),
            "expected InvalidName, got {err:?}",
        );
        let err = store
            .add("memory", MemoryType::Project, "d", "c")
            .expect_err("memory");
        assert!(matches!(err, MemoryError::InvalidName { .. }));

        let after = std::fs::read_to_string(store.project_index_path()).expect("read index 2");
        assert_eq!(
            before, after,
            "index file changed after a rejected add call",
        );
    }

    #[test]
    fn add_then_read_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = store_at(dir.path());
        store
            .add("topic", MemoryType::Project, "desc", "body content")
            .expect("add");
        let (entry, scope) = store.read("topic").expect("read");
        assert_eq!(scope, MemoryScope::Project);
        assert_eq!(entry.frontmatter.name, "topic");
        assert!(entry.body.starts_with("body content"));

        let index = store.read_merged_index().expect("index");
        assert!(index.contains("- [topic](topic.md) — desc"));
    }

    #[test]
    fn add_duplicate_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = store_at(dir.path());
        store.add("t", MemoryType::User, "d", "body").expect("add");
        let err = store
            .add("t", MemoryType::User, "d", "body")
            .expect_err("dup");
        assert!(matches!(err, MemoryError::AlreadyExists { .. }));
    }

    #[test]
    fn update_modifies_description_and_body() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = store_at(dir.path());
        store
            .add("t", MemoryType::User, "old", "old body")
            .expect("add");
        store
            .update("t", None, Some("new desc"), Some("new body"))
            .expect("update");
        let (entry, _) = store.read("t").expect("read");
        assert_eq!(entry.frontmatter.description, "new desc");
        assert!(entry.body.contains("new body"));

        let index = store.read_merged_index().expect("index");
        assert!(index.contains("new desc"));
        assert!(!index.contains("— old"));
    }

    #[test]
    fn remove_drops_file_and_index_row() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = store_at(dir.path());
        store
            .add("t", MemoryType::Project, "desc", "body")
            .expect("add");
        store.remove("t").expect("remove");
        let err = store.read("t").expect_err("missing");
        assert!(matches!(err, MemoryError::NotFound { .. }));
        let index = store.read_merged_index().expect("index");
        assert!(!index.contains("topic.md"));
    }

    #[test]
    fn global_layer_is_merged_and_overridden_by_project() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project_dir = dir.path().join("proj").join(".tsumugi").join("memory");
        let global_dir = dir.path().join("global");
        std::fs::create_dir_all(&global_dir).expect("global dir");
        // Build a global entry by hand.
        let entry_path = global_dir.join("shared.md");
        let entry = MemoryEntry {
            frontmatter: Frontmatter {
                name: "shared".to_owned(),
                description: "global desc".to_owned(),
                kind: MemoryType::Reference,
                created_at: None,
                updated_at: None,
            },
            body: "global body".to_owned(),
        };
        std::fs::write(&entry_path, entry.to_markdown().expect("md")).expect("write");
        let global_index = format_index_line("shared", "global desc");
        std::fs::write(global_dir.join(INDEX_FILE_NAME), global_index + "\n").expect("idx");

        let store = MemoryStore::with_dirs(
            project_dir.clone(),
            Some(global_dir.clone()),
            MemoryBudget::default(),
        );

        // Read falls back to global.
        let (got, scope) = store.read("shared").expect("read");
        assert_eq!(scope, MemoryScope::Global);
        assert_eq!(got.frontmatter.description, "global desc");

        // Project add wins on collision.
        store
            .add(
                "shared",
                MemoryType::Project,
                "project desc",
                "project body",
            )
            .expect("add");
        let (got, scope) = store.read("shared").expect("read");
        assert_eq!(scope, MemoryScope::Project);
        assert_eq!(got.frontmatter.description, "project desc");

        let merged = store.read_merged_index().expect("merge");
        assert!(merged.contains("project desc"));
        // Global row is suppressed because the file name collides.
        let global_count = merged.matches("global desc").count();
        assert_eq!(global_count, 0);
    }

    #[test]
    fn list_project_excludes_index_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = store_at(dir.path());
        store.add("a", MemoryType::User, "d", "b").expect("a");
        store.add("b", MemoryType::User, "d", "b").expect("b");
        let mut names = store.list_project().expect("list");
        names.sort();
        assert_eq!(names, vec!["a".to_owned(), "b".to_owned()]);
    }

    /// Regression test for issue #7: when an entry file is deleted out
    /// of band but the index still references it, `reconcile` repairs
    /// the index. Symmetrically, when a file exists with no row,
    /// `reconcile` appends the row.
    #[test]
    fn reconcile_drops_orphan_rows_and_appends_missing_rows() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = store_at(dir.path());
        store
            .add("alpha", MemoryType::Project, "alpha desc", "alpha body")
            .expect("alpha");
        store
            .add("beta", MemoryType::Project, "beta desc", "beta body")
            .expect("beta");

        // 1. Delete `beta.md` out-of-band: the index still references it.
        let beta_path = store.project_dir().join("beta.md");
        std::fs::remove_file(&beta_path).expect("remove beta");

        // 2. Hand-write a `gamma.md` file with no matching row.
        let gamma_path = store.project_dir().join("gamma.md");
        let gamma_entry = MemoryEntry {
            frontmatter: Frontmatter {
                name: "gamma".to_owned(),
                description: "gamma desc".to_owned(),
                kind: MemoryType::User,
                created_at: None,
                updated_at: None,
            },
            body: "gamma body".to_owned(),
        };
        std::fs::write(&gamma_path, gamma_entry.to_markdown().expect("md")).expect("write gamma");

        store.reconcile().expect("reconcile");

        let index = std::fs::read_to_string(store.project_index_path()).expect("read index");
        assert!(index.contains("alpha.md"), "alpha row preserved");
        assert!(!index.contains("beta.md"), "beta row dropped");
        assert!(index.contains("gamma.md"), "gamma row appended");
    }

    /// Regression test for issue #11: a stray `(...)` in user content
    /// must not be misclassified as a link target. The merge logic
    /// should pass header/comment lines through unchanged.
    #[test]
    fn read_merged_index_preserves_non_link_lines() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project_dir = dir.path().join("proj").join(".tsumugi").join("memory");
        let global_dir = dir.path().join("global");
        std::fs::create_dir_all(&global_dir).expect("global dir");
        // Global index with a comment + a link row.
        let global_index = "# My global memory (the good one)\n\
            <!-- index of memory (auto-generated) -->\n\
            - [thing](thing.md) — global desc\n";
        std::fs::write(global_dir.join(INDEX_FILE_NAME), global_index).expect("idx");
        // Need a matching entry file so the row is sound; reconcile
        // will not delete it because the file is on disk.
        let thing = MemoryEntry {
            frontmatter: Frontmatter {
                name: "thing".to_owned(),
                description: "global desc".to_owned(),
                kind: MemoryType::Reference,
                created_at: None,
                updated_at: None,
            },
            body: "body".to_owned(),
        };
        std::fs::write(
            global_dir.join("thing.md"),
            thing.to_markdown().expect("md"),
        )
        .expect("write thing");

        let store = MemoryStore::with_dirs(
            project_dir.clone(),
            Some(global_dir.clone()),
            MemoryBudget::default(),
        );
        let merged = store.read_merged_index().expect("merged");
        // The header and comment must NOT get the `[global] ` prefix.
        assert!(merged.contains("# My global memory (the good one)"));
        assert!(merged.contains("<!-- index of memory (auto-generated) -->"));
        assert!(!merged.contains("[global] # My"));
        assert!(!merged.contains("[global] <!--"));
        // Link row gets the prefix.
        assert!(merged.contains("[global] - [thing]"));
    }

    #[test]
    fn budget_report_tracks_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        // total_files_limit = 5 so the 80% threshold (= 4) leaves
        // headroom for the under-budget assertion. With the previous
        // limit of 2 the 80% threshold rounded down to 1, which fired
        // immediately on the first add and made the assertion flaky.
        let small = MemoryBudget {
            index_max_lines: 200,
            entry_max_chars: 600,
            total_files_limit: 5,
        };
        let store = MemoryStore::with_dirs(dir.path().join(".tsumugi").join("memory"), None, small);
        store.add("a", MemoryType::User, "d", "b").expect("a");
        let r1 = store.budget_report().expect("r1");
        assert!(!r1.near_capacity(), "1/5 files: under 80% threshold");
        for n in ["b", "c", "d"] {
            store.add(n, MemoryType::User, "d", "b").expect("seed");
        }
        let r2 = store.budget_report().expect("r2");
        assert!(r2.near_capacity(), "4/5 files: at 80% threshold");
    }
}
