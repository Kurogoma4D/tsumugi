//! `tmg memory ...` subcommand implementations.
//!
//! These commands operate on the project memory store directly (no
//! agent loop, no LLM round-trip). They mirror the `MemoryTool` JSON
//! API but skip the sandbox check because the operator is invoking
//! them by hand.

use std::io::Write as _;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use anyhow::{Context as _, anyhow};
use tmg_memory::MemoryStore;

use crate::config::{MemoryConfig, TsumugiConfig};

/// Build a [`MemoryStore`] from the merged [`MemoryConfig`].
///
/// The project root is derived from the canonicalised cwd because the
/// memory store is project-scoped: invoking `tmg memory list` in a
/// subdirectory still resolves the project memory dir relative to the
/// directory the user is running tmg from.
fn make_store(config: &MemoryConfig) -> anyhow::Result<Arc<MemoryStore>> {
    let cwd = std::env::current_dir().context("reading current working directory")?;
    let canonical = cwd.canonicalize().unwrap_or_else(|_| cwd.clone());
    let project_dir = config.resolve_project_dir(&canonical);
    let global_dir = config.resolve_global_dir();
    let store = MemoryStore::with_dirs(project_dir, global_dir, config.to_budget());
    Ok(Arc::new(store))
}

/// `tmg memory list` — print one entry name per line.
#[expect(
    clippy::print_stdout,
    reason = "CLI subcommand: this is the documented stdout surface"
)]
pub fn cmd_list(config: &TsumugiConfig) -> anyhow::Result<()> {
    if !config.memory.enabled {
        return Err(anyhow!("memory is disabled in [memory].enabled"));
    }
    let store = make_store(&config.memory)?;
    let names = store.list_project().context("listing project memory")?;
    if names.is_empty() {
        println!("(no memory entries yet — use `tmg memory edit <name>` or the `memory` tool)");
        return Ok(());
    }
    for name in names {
        println!("{name}");
    }
    Ok(())
}

/// `tmg memory show <name>` — print the body of one entry.
#[expect(
    clippy::print_stdout,
    reason = "CLI subcommand: this is the documented stdout surface"
)]
pub fn cmd_show(config: &TsumugiConfig, name: &str) -> anyhow::Result<()> {
    if !config.memory.enabled {
        return Err(anyhow!("memory is disabled in [memory].enabled"));
    }
    let store = make_store(&config.memory)?;
    let (entry, scope) = store
        .read(name)
        .with_context(|| format!("reading memory entry {name:?}"))?;
    println!(
        "[{}] {} ({}) — {}\n",
        scope.as_str(),
        entry.frontmatter.name,
        entry.frontmatter.kind,
        entry.frontmatter.description,
    );
    println!("{}", entry.body);
    Ok(())
}

/// `tmg memory edit <name>` — open the entry in `$EDITOR`.
///
/// Missing entries are scaffolded with a frontmatter template so the
/// user can complete the file in the editor; no entry is written
/// before the editor exits successfully.
pub fn cmd_edit(config: &TsumugiConfig, name: &str) -> anyhow::Result<()> {
    if !config.memory.enabled {
        return Err(anyhow!("memory is disabled in [memory].enabled"));
    }
    let store = make_store(&config.memory)?;
    let project_dir = store.project_dir().to_owned();
    if !project_dir.exists() {
        std::fs::create_dir_all(&project_dir)
            .with_context(|| format!("creating project memory dir {}", project_dir.display()))?;
    }
    let path = project_dir.join(format!("{name}.md"));
    if !path.exists() {
        scaffold_entry(&path, name)?;
    }
    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_owned());
    let status = Command::new(&editor)
        .arg(&path)
        .status()
        .with_context(|| format!("launching editor {editor:?}"))?;
    if !status.success() {
        return Err(anyhow!("editor exited with non-zero status"));
    }
    // Re-parse the file to validate; if the user broke the
    // frontmatter, surface the error so they can re-open and fix.
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {} after edit", path.display()))?;
    tmg_memory::parse_entry(&text, &path.display().to_string())
        .with_context(|| format!("validating memory entry at {}", path.display()))?;
    // Re-sync the index row in case the description was edited.
    sync_index_row(&store, name)?;
    Ok(())
}

/// Write a placeholder entry so the editor opens to a sensible
/// frontmatter skeleton instead of an empty file.
fn scaffold_entry(path: &Path, name: &str) -> anyhow::Result<()> {
    let body = format!(
        "---\nname: {name}\ndescription: TODO one-line summary\ntype: project\n---\n\nTODO body.\n",
    );
    let mut file = std::fs::File::create(path)
        .with_context(|| format!("creating scaffold at {}", path.display()))?;
    file.write_all(body.as_bytes())
        .with_context(|| format!("writing scaffold at {}", path.display()))?;
    Ok(())
}

/// After an external edit, rewrite the matching `MEMORY.md` row so
/// the description in the index always reflects the current entry.
fn sync_index_row(store: &MemoryStore, name: &str) -> anyhow::Result<()> {
    let (entry, _) = store
        .read(name)
        .with_context(|| format!("re-reading memory entry {name:?} after edit"))?;
    // Fast path: rewrite description via update with no body change.
    // `update` bumps `updated_at`; that's the desired side-effect of a
    // manual edit too.
    store
        .update(
            name,
            Some(entry.frontmatter.kind),
            Some(&entry.frontmatter.description),
            None,
        )
        .with_context(|| format!("syncing index row for {name:?}"))?;
    Ok(())
}

/// Resolve a memory subcommand and dispatch.
pub fn dispatch(op: crate::MemoryCommand, config: &TsumugiConfig) -> anyhow::Result<()> {
    match op {
        crate::MemoryCommand::List => cmd_list(config),
        crate::MemoryCommand::Show { name } => cmd_show(config, &name),
        crate::MemoryCommand::Edit { name } => cmd_edit(config, &name),
    }
}
