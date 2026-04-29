//! Auto-generated-skill banner formatter.
//!
//! Issue #54 acceptance criterion: "auto-generated skill が次回起動時に
//! TUI バナーで通知される". The banner is a single line shown by the
//! TUI when one or more `provenance: agent` skills landed during the
//! previous session.
//!
//! Detection is purely on-disk: the TUI scans the discovered skills
//! list for `provenance: agent` entries that have not yet been
//! acknowledged in `.tsumugi/skills/.banner_ack` (a tiny tracker file
//! storing the names that have already been shown). This module owns
//! the formatter and the tracker; wiring is on the TUI side.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::error::SkillError;
use crate::types::{Provenance, SkillMeta};

/// Hidden tracker file recording which auto-generated skills the user
/// has already seen the banner for.
pub const BANNER_ACK_FILENAME: &str = ".banner_ack";

/// Compute the path of the banner-acknowledgement file.
#[must_use]
pub fn banner_ack_path(skills_root: impl AsRef<Path>) -> PathBuf {
    skills_root.as_ref().join(BANNER_ACK_FILENAME)
}

/// Load the set of skill names already acknowledged.
///
/// # Errors
///
/// Returns [`SkillError::Io`] if the file exists but cannot be read.
pub async fn load_acknowledged(
    skills_root: impl AsRef<Path>,
) -> Result<BTreeSet<String>, SkillError> {
    let path = banner_ack_path(skills_root);
    match tokio::fs::read_to_string(&path).await {
        Ok(text) => Ok(text
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(BTreeSet::new()),
        Err(e) => Err(SkillError::io(format!("reading {}", path.display()), e)),
    }
}

/// Persist the acknowledgement set.
///
/// # Errors
///
/// Returns [`SkillError::Io`] when the file cannot be written.
pub async fn save_acknowledged(
    skills_root: impl AsRef<Path>,
    names: &BTreeSet<String>,
) -> Result<(), SkillError> {
    let path = banner_ack_path(skills_root);
    let mut content = String::new();
    for name in names {
        content.push_str(name);
        content.push('\n');
    }
    tokio::fs::write(&path, content)
        .await
        .map_err(|e| SkillError::io(format!("writing {}", path.display()), e))
}

/// Compute the list of skill names that need a banner shown.
///
/// Returns the slice of `skills` whose [`Provenance`] is `agent` and
/// whose name is not already in `acknowledged`.
#[must_use]
pub fn pending_banner_names(skills: &[SkillMeta], acknowledged: &BTreeSet<String>) -> Vec<String> {
    skills
        .iter()
        .filter(|s| s.frontmatter.provenance == Some(Provenance::Agent))
        .map(|s| s.name.as_str().to_owned())
        .filter(|n| !acknowledged.contains(n))
        .collect()
}

/// Format the banner string. Returns `None` when the slice is empty.
///
/// Format matches the issue exemplar:
///
/// ```text
/// ⚡ 1 new skill auto-generated this session: "deploy-rust"
/// ```
///
/// For multiple names, the names are quoted and comma-separated.
#[must_use]
pub fn format_banner(names: &[String]) -> Option<String> {
    if names.is_empty() {
        return None;
    }
    let count = names.len();
    let plural = if count == 1 { "skill" } else { "skills" };
    let quoted: Vec<String> = names.iter().map(|n| format!("\"{n}\"")).collect();
    Some(format!(
        "\u{26a1} {count} new {plural} auto-generated this session: {}",
        quoted.join(", ")
    ))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions")]
mod tests {
    use super::*;
    use crate::types::{InvocationPolicy, SkillFrontmatter, SkillName, SkillPath, SkillSource};

    fn fake(name: &str, prov: Option<Provenance>) -> SkillMeta {
        SkillMeta {
            name: SkillName::new(name),
            frontmatter: SkillFrontmatter {
                name: name.to_owned(),
                description: format!("desc {name}"),
                invocation: InvocationPolicy::Auto,
                allowed_tools: None,
                version: None,
                created_at: None,
                updated_at: None,
                provenance: prov,
            },
            source: SkillSource::ProjectTsumugi,
            path: SkillPath::new(format!("/fake/{name}/SKILL.md")),
        }
    }

    #[test]
    fn pending_filters_to_unack_agent_skills() {
        let skills = vec![
            fake("user-1", Some(Provenance::User)),
            fake("agent-a", Some(Provenance::Agent)),
            fake("agent-b", Some(Provenance::Agent)),
            fake("legacy", None),
        ];
        let mut ack = BTreeSet::new();
        ack.insert("agent-a".to_owned());
        let pending = pending_banner_names(&skills, &ack);
        assert_eq!(pending, vec!["agent-b".to_owned()]);
    }

    #[test]
    fn format_banner_singular() {
        let s = format_banner(&["deploy-rust".to_owned()]).unwrap();
        assert!(s.starts_with('\u{26a1}'), "missing prefix in {s:?}");
        assert!(s.contains("1 new skill"));
        assert!(s.contains("\"deploy-rust\""));
    }

    #[test]
    fn format_banner_plural() {
        let s = format_banner(&["a".to_owned(), "b".to_owned()]).unwrap();
        assert!(s.starts_with('\u{26a1}'), "missing prefix in {s:?}");
        assert!(s.contains("2 new skills"));
        assert!(s.contains("\"a\""));
        assert!(s.contains("\"b\""));
    }

    #[test]
    fn format_banner_empty_returns_none() {
        assert!(format_banner(&[]).is_none());
    }

    #[tokio::test]
    async fn ack_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let mut set = BTreeSet::new();
        set.insert("alpha".to_owned());
        set.insert("beta".to_owned());
        save_acknowledged(tmp.path(), &set).await.unwrap();
        let loaded = load_acknowledged(tmp.path()).await.unwrap();
        assert_eq!(loaded, set);
    }

    #[tokio::test]
    async fn load_returns_empty_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let loaded = load_acknowledged(tmp.path()).await.unwrap();
        assert!(loaded.is_empty());
    }
}
