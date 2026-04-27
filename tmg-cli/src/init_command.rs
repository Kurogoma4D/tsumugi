//! Implementation of the `tmg init` subcommand (issue #44).
//!
//! Materialises a `.tsumugi/` directory at the project root from the
//! built-in template catalogue exposed by [`tmg_workflow::templates`].
//! The command is non-interactive: every selection is supplied via
//! flags so CI can call it with confidence. Every existing file is
//! preserved by default; `--force` is the explicit "I understand this
//! overwrites" opt-in.
//!
//! ## Trade-off: refuse-on-conflict vs. interactive merge
//!
//! When `--force` is *not* set and at least one selected template
//! would clobber an existing file, the command aborts before writing
//! anything. We deliberately do **not** offer an interactive
//! per-file prompt: it would defeat the CI-friendly contract and
//! complicate the binary's surface for what is really a one-time
//! scaffold operation. Operators who want a partial overwrite should
//! delete the conflicting files first.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use tmg_workflow::templates::{self, TemplateCategory, TemplateMeta};

/// Plan computed from user flags before any I/O happens.
///
/// Building the plan up front means we can detect conflicts and
/// missing template names *before* writing a single byte. A failed
/// `tmg init` should leave the workspace untouched.
#[derive(Debug)]
pub(crate) struct InitPlan {
    /// Each selected template plus the absolute target path.
    pub entries: Vec<InitPlanEntry>,
    /// Directories that need to exist before any write happens.
    pub directories: BTreeSet<PathBuf>,
}

#[derive(Debug)]
pub(crate) struct InitPlanEntry {
    pub meta: TemplateMeta,
    pub target: PathBuf,
}

/// Build an [`InitPlan`] for the requested templates.
///
/// `tsumugi_dir` is the absolute `.tsumugi/` directory the templates
/// will be installed under. Caller is responsible for picking the
/// directory (project root vs. a custom path).
///
/// # Errors
///
/// Returns an error when `--workflows` / `--harness` references a name
/// that is not in the built-in catalogue. The user-facing message
/// lists the unknown names so a typo is easy to spot.
pub(crate) fn build_plan(
    tsumugi_dir: &Path,
    workflows: &[String],
    harness: &[String],
    all: bool,
) -> anyhow::Result<InitPlan> {
    let mut requested: Vec<TemplateMeta> = Vec::new();
    let mut unknown: Vec<String> = Vec::new();

    if all {
        requested.extend(templates::list_available());
    } else {
        for name in workflows {
            match templates::find(name) {
                Some(meta) if matches!(meta.category, TemplateCategory::Workflow) => {
                    requested.push(meta);
                }
                Some(meta) => {
                    anyhow::bail!(
                        "template '{name}' is a {category}, not a workflow; pass it via --harness or --all instead",
                        category = meta.category.label(),
                    );
                }
                None => unknown.push(name.clone()),
            }
        }
        for name in harness {
            match templates::find(name) {
                Some(meta) if matches!(meta.category, TemplateCategory::Harness) => {
                    requested.push(meta);
                }
                Some(meta) => {
                    anyhow::bail!(
                        "template '{name}' is a {category}, not a harness template; pass it via --workflows or --all instead",
                        category = meta.category.label(),
                    );
                }
                None => unknown.push(name.clone()),
            }
        }
    }

    if !unknown.is_empty() {
        anyhow::bail!(
            "unknown template(s): {}. Run `tmg init --help` to see available names.",
            unknown.join(", "),
        );
    }

    if requested.is_empty() {
        anyhow::bail!(
            "no templates selected. Pass --workflows, --harness, or --all (see `tmg init --help`).",
        );
    }

    // De-duplicate (--all may overlap; or a user might list the same
    // name twice). Stable order preserves the first occurrence so the
    // installer's confirmation lines match the user's flag order.
    let mut seen: BTreeSet<&'static str> = BTreeSet::new();
    requested.retain(|meta| seen.insert(meta.name));

    let mut directories: BTreeSet<PathBuf> = BTreeSet::new();
    let mut entries: Vec<InitPlanEntry> = Vec::with_capacity(requested.len());
    directories.insert(tsumugi_dir.to_path_buf());

    for meta in requested {
        let target = tsumugi_dir.join(meta.install_path);
        if let Some(parent) = target.parent() {
            directories.insert(parent.to_path_buf());
        }
        entries.push(InitPlanEntry { meta, target });
    }

    Ok(InitPlan {
        entries,
        directories,
    })
}

/// Detect every entry whose target already exists.
pub(crate) fn detect_conflicts(plan: &InitPlan) -> Vec<PathBuf> {
    plan.entries
        .iter()
        .filter(|e| e.target.exists())
        .map(|e| e.target.clone())
        .collect()
}

/// Run `tmg init` with the supplied flags.
///
/// The function is synchronous (no async work needed for filesystem
/// writes) and is invoked directly from `main.rs`'s subcommand
/// dispatch. Returns the number of files written so the caller can
/// print a concise confirmation.
#[expect(
    clippy::print_stdout,
    reason = "tmg init prints install confirmations to stdout"
)]
pub(crate) fn cmd_init(
    workflows: &[String],
    harness: &[String],
    all: bool,
    force: bool,
) -> anyhow::Result<()> {
    let cwd = std::env::current_dir().context("reading current working directory")?;
    let canonical = cwd.canonicalize().unwrap_or(cwd);
    let tsumugi_dir = canonical.join(".tsumugi");

    let plan = build_plan(&tsumugi_dir, workflows, harness, all)?;

    let conflicts = detect_conflicts(&plan);
    if !conflicts.is_empty() && !force {
        let mut msg = String::from("the following files already exist:\n");
        for c in &conflicts {
            // `write!` on a String never fails per std docs.
            let _ = writeln!(msg, "  {}", c.display());
        }
        msg.push_str("re-run with --force to overwrite.");
        anyhow::bail!(msg);
    }

    // Materialise directories first so a partial template list still
    // produces a usable layout.
    for dir in &plan.directories {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating directory {}", dir.display()))?;
    }

    // Always create the `skills/develop/` parent even when the
    // `develop` skill template is not selected — the SPEC §8.5 layout
    // expects the directory to be present so users can drop their own
    // SKILL.md in.  We only create the *parent* `skills/` directory
    // here (not `skills/develop/`); a deeper sentinel would over-claim
    // intent.
    let skills_dir = tsumugi_dir.join("skills");
    std::fs::create_dir_all(&skills_dir)
        .with_context(|| format!("creating directory {}", skills_dir.display()))?;
    let workflows_dir = tsumugi_dir.join("workflows");
    std::fs::create_dir_all(&workflows_dir)
        .with_context(|| format!("creating directory {}", workflows_dir.display()))?;
    let harness_dir = tsumugi_dir.join("harness");
    std::fs::create_dir_all(&harness_dir)
        .with_context(|| format!("creating directory {}", harness_dir.display()))?;

    let mut written = 0usize;
    for entry in &plan.entries {
        let content = templates::read_template(entry.meta.name).ok_or_else(|| {
            anyhow::anyhow!(
                "template registry desync: '{}' has metadata but no content",
                entry.meta.name,
            )
        })?;
        if let Some(parent) = entry.target.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating parent {}", parent.display()))?;
        }
        std::fs::write(&entry.target, content)
            .with_context(|| format!("writing {}", entry.target.display()))?;
        written += 1;
        let action = if conflicts.contains(&entry.target) {
            "overwrote"
        } else {
            "wrote"
        };
        println!(
            "{action} {} ({} '{}')",
            entry.target.display(),
            entry.meta.category.label(),
            entry.meta.name,
        );
    }

    println!(
        "\ninstalled {written} template{s} into {}",
        tsumugi_dir.display(),
        s = if written == 1 { "" } else { "s" },
    );
    Ok(())
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions")]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn names(plan: &InitPlan) -> Vec<&'static str> {
        plan.entries.iter().map(|e| e.meta.name).collect()
    }

    #[test]
    fn build_plan_all_includes_every_template() {
        let tmp = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let plan = build_plan(&tmp.path().join(".tsumugi"), &[], &[], true)
            .unwrap_or_else(|e| panic!("{e}"));
        let n = templates::list_available().len();
        assert_eq!(plan.entries.len(), n);
    }

    #[test]
    fn build_plan_specific_workflows() {
        let tmp = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let workflows = vec!["plan".to_owned(), "implement".to_owned()];
        let plan = build_plan(&tmp.path().join(".tsumugi"), &workflows, &[], false)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(names(&plan), vec!["plan", "implement"]);
    }

    #[test]
    fn build_plan_specific_harness() {
        let tmp = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let harness = vec!["build-app".to_owned()];
        let plan = build_plan(&tmp.path().join(".tsumugi"), &[], &harness, false)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(names(&plan), vec!["build-app"]);
    }

    #[test]
    fn build_plan_rejects_unknown_name() {
        let tmp = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let workflows = vec!["does-not-exist".to_owned()];
        let result = build_plan(&tmp.path().join(".tsumugi"), &workflows, &[], false);
        assert!(result.is_err());
    }

    #[test]
    fn build_plan_rejects_category_mismatch() {
        let tmp = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        // 'build-app' is a harness template; passing it via --workflows
        // should fail with a hint to use --harness.
        let workflows = vec!["build-app".to_owned()];
        let result = build_plan(&tmp.path().join(".tsumugi"), &workflows, &[], false);
        match result {
            Err(err) => {
                let msg = err.to_string();
                assert!(msg.contains("--harness"), "{msg}");
            }
            Ok(_) => panic!("expected category-mismatch error"),
        }
    }

    #[test]
    fn build_plan_rejects_empty_selection() {
        let tmp = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        assert!(build_plan(&tmp.path().join(".tsumugi"), &[], &[], false).is_err());
    }

    #[test]
    fn build_plan_dedups_repeats() {
        let tmp = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let workflows = vec!["plan".to_owned(), "plan".to_owned()];
        let plan = build_plan(&tmp.path().join(".tsumugi"), &workflows, &[], false)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(plan.entries.len(), 1);
    }

    #[test]
    fn detect_conflicts_finds_existing_files() {
        let tmp = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let plan = build_plan(
            &tmp.path().join(".tsumugi"),
            &["plan".to_owned()],
            &[],
            false,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        // Pre-create the target file to simulate a conflict.
        let parent = plan.entries[0]
            .target
            .parent()
            .unwrap_or_else(|| panic!("template target has no parent"));
        std::fs::create_dir_all(parent).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(&plan.entries[0].target, "existing").unwrap_or_else(|e| panic!("{e}"));
        let conflicts = detect_conflicts(&plan);
        assert_eq!(conflicts.len(), 1);
    }
}
