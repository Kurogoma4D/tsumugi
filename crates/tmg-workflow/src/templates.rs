//! Built-in workflow / harness / skill templates (issue #44).
//!
//! The crate ships a small library of opinionated workflow YAML and a
//! sample skill so `tmg init` can scaffold a `.tsumugi/` directory
//! without external network access. Templates are embedded at compile
//! time via [`include_str!`]; no runtime asset lookup is required.
//!
//! ## Naming convention
//!
//! Each template has:
//!
//! - A short kebab-case **name** (e.g. `"plan"`, `"build-app"`) used by
//!   `tmg init --workflows ...` and `tmg init --harness ...`.
//! - A relative **install path** under `.tsumugi/` (e.g.
//!   `workflows/plan.yaml`).
//! - A category ([`TemplateCategory`]) so the installer knows which
//!   subdirectory to create.
//!
//! ## Trade-off: `include_str!` vs `include_dir`
//!
//! We use plain `include_str!` to avoid pulling another dependency
//! (`include_dir`) into the workspace just for ~10 small files. The
//! cost is that adding a new template requires a one-line entry in
//! [`ALL`] alongside the file. The build is still completely static —
//! no `build.rs` needed — and the embed is verified at compile time
//! by `include_str!` itself.
//!
//! ## Validation invariant
//!
//! `crates/tmg-workflow/src/templates.rs::tests::all_workflow_templates_parse`
//! asserts that every YAML template parses through
//! [`crate::parse::parse_workflow_str`]. This catches drift between
//! the parser and the embedded templates whenever either side changes.

/// Category of a built-in template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TemplateCategory {
    /// A plain workflow YAML to be installed under
    /// `.tsumugi/workflows/`.
    Workflow,
    /// A long-running harness workflow YAML to be installed under
    /// `.tsumugi/workflows/` (harness templates are functionally
    /// long-running workflows; the separate name is purely UX).
    Harness,
    /// A skill bundle entry to be installed under
    /// `.tsumugi/skills/<skill_id>/`.
    Skill,
}

impl TemplateCategory {
    /// Human-readable label used by `tmg init --help` text and the
    /// install confirmation messages.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Workflow => "workflow",
            Self::Harness => "harness",
            Self::Skill => "skill",
        }
    }
}

/// Metadata for a single built-in template.
#[derive(Debug, Clone, Copy)]
pub struct TemplateMeta {
    /// Short, kebab-case name. Used as the value passed to
    /// `tmg init --workflows ...` / `--harness ...`.
    pub name: &'static str,
    /// Category (workflow / harness / skill).
    pub category: TemplateCategory,
    /// Path the file should be installed at, relative to the
    /// `.tsumugi/` root.
    pub install_path: &'static str,
    /// One-line description shown in help output.
    pub description: &'static str,
}

// -----------------------------------------------------------------------------
// Embedded contents
// -----------------------------------------------------------------------------

const PLAN_YAML: &str = include_str!("../templates/workflows/plan.yaml");
const IMPLEMENT_YAML: &str = include_str!("../templates/workflows/implement.yaml");
const REVIEW_YAML: &str = include_str!("../templates/workflows/review.yaml");
const REFACTOR_YAML: &str = include_str!("../templates/workflows/refactor.yaml");
const BUILD_APP_YAML: &str = include_str!("../templates/harness/build-app.yaml");
const MIGRATE_CODEBASE_YAML: &str = include_str!("../templates/harness/migrate-codebase.yaml");
const ADD_FEATURE_YAML: &str = include_str!("../templates/harness/add-feature.yaml");
const BUG_FIX_BATCH_YAML: &str = include_str!("../templates/harness/bug-fix-batch.yaml");
const SKILL_DEVELOP_MD: &str = include_str!("../templates/skills/develop/SKILL.md");

/// Master table of every embedded template.
///
/// Order matters for stable output ordering in `tmg init` confirmation
/// messages: workflows first (alphabetical), then harness templates,
/// then skills. Adding a new template means adding one line here and
/// one entry in [`read_template`].
pub const ALL: &[TemplateMeta] = &[
    TemplateMeta {
        name: "plan",
        category: TemplateCategory::Workflow,
        install_path: "workflows/plan.yaml",
        description: "Explore the codebase, draft a plan, and audit it",
    },
    TemplateMeta {
        name: "implement",
        category: TemplateCategory::Workflow,
        install_path: "workflows/implement.yaml",
        description: "Implement a feature with agent + verify (SPEC §8.3)",
    },
    TemplateMeta {
        name: "review",
        category: TemplateCategory::Workflow,
        install_path: "workflows/review.yaml",
        description: "Review-then-fix loop (SPEC §8.4)",
    },
    TemplateMeta {
        name: "refactor",
        category: TemplateCategory::Workflow,
        install_path: "workflows/refactor.yaml",
        description: "Implement a refactor and review it",
    },
    TemplateMeta {
        name: "build-app",
        category: TemplateCategory::Harness,
        install_path: "workflows/build-app.yaml",
        description: "Build a new app from scratch (SPEC §8.12 / §9.12)",
    },
    TemplateMeta {
        name: "migrate-codebase",
        category: TemplateCategory::Harness,
        install_path: "workflows/migrate-codebase.yaml",
        description: "Migrate a codebase file-by-file",
    },
    TemplateMeta {
        name: "add-feature",
        category: TemplateCategory::Harness,
        install_path: "workflows/add-feature.yaml",
        description: "Add a feature to an existing project iteratively",
    },
    TemplateMeta {
        name: "bug-fix-batch",
        category: TemplateCategory::Harness,
        install_path: "workflows/bug-fix-batch.yaml",
        description: "Fix a batch of bugs, one per feature",
    },
    TemplateMeta {
        name: "develop",
        category: TemplateCategory::Skill,
        install_path: "skills/develop/SKILL.md",
        description: "Sample method-1 skill (SPEC §8.5)",
    },
];

/// Return the full template catalogue.
#[must_use]
pub fn list_available() -> Vec<TemplateMeta> {
    ALL.to_vec()
}

/// Look up a template by its kebab-case name.
///
/// Returns `None` when no template matches.
#[must_use]
pub fn read_template(name: &str) -> Option<&'static str> {
    match name {
        "plan" => Some(PLAN_YAML),
        "implement" => Some(IMPLEMENT_YAML),
        "review" => Some(REVIEW_YAML),
        "refactor" => Some(REFACTOR_YAML),
        "build-app" => Some(BUILD_APP_YAML),
        "migrate-codebase" => Some(MIGRATE_CODEBASE_YAML),
        "add-feature" => Some(ADD_FEATURE_YAML),
        "bug-fix-batch" => Some(BUG_FIX_BATCH_YAML),
        "develop" => Some(SKILL_DEVELOP_MD),
        _ => None,
    }
}

/// Look up the [`TemplateMeta`] for a kebab-case name.
#[must_use]
pub fn find(name: &str) -> Option<TemplateMeta> {
    ALL.iter().copied().find(|m| m.name == name)
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions")]
mod tests {
    use super::*;
    use crate::parse::parse_workflow_str;

    /// CI guard: every embedded YAML template must parse through
    /// [`parse_workflow_str`]. This is the single test that catches
    /// drift between the parser surface and the shipped templates.
    #[test]
    fn all_workflow_templates_parse() {
        for meta in ALL {
            if matches!(meta.category, TemplateCategory::Skill) {
                continue;
            }
            let content = read_template(meta.name).unwrap_or_else(|| {
                panic!(
                    "read_template returned None for known template '{}'",
                    meta.name
                )
            });
            parse_workflow_str(content, format!("<embedded:{}>", meta.name)).unwrap_or_else(|e| {
                panic!("template '{}' failed to parse: {e}", meta.name);
            });
        }
    }

    /// Each entry in [`ALL`] must have a working [`read_template`]
    /// arm. A missing arm would compile but silently lose the template
    /// at runtime.
    #[test]
    fn every_meta_has_a_read_arm() {
        for meta in ALL {
            assert!(
                read_template(meta.name).is_some(),
                "no read_template arm for '{}'",
                meta.name,
            );
        }
    }

    /// `find` agrees with the `ALL` table.
    #[test]
    fn find_round_trips() {
        for meta in ALL {
            let found = find(meta.name).unwrap_or_else(|| panic!("find('{}')", meta.name));
            assert_eq!(found.name, meta.name);
            assert_eq!(found.install_path, meta.install_path);
        }
        assert!(find("does-not-exist").is_none());
    }

    /// Names are unique. (Defensive: a duplicate would shadow.)
    #[test]
    fn names_are_unique() {
        let mut seen = std::collections::BTreeSet::new();
        for meta in ALL {
            assert!(
                seen.insert(meta.name),
                "duplicate template name '{}'",
                meta.name,
            );
        }
    }
}
