//! Skill discovery: scans configured directories for SKILL.md files.
//!
//! Priority order (highest first):
//! 1. `.tsumugi/skills/`
//! 2. `~/.config/tsumugi/skills/`
//! 3. Additional paths from `[skills].discovery_paths` in `tsumugi.toml`
//! 4. `.claude/skills/` (if `compat_claude` is enabled)
//! 5. `.agents/skills/` (if `compat_agent_skills` is enabled)
//!
//! Same-named skills from higher-priority sources shadow lower ones.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::config::SkillsConfig;
use crate::error::SkillError;
use crate::parse::parse_skill_md;
use crate::types::{SkillMeta, SkillName, SkillPath, SkillSource};

/// The expected skill definition file name.
const SKILL_FILENAME: &str = "SKILL.md";

/// Discover all skills from the configured directories.
///
/// Scans each source directory in priority order. When two skills
/// share the same name, the one from the higher-priority source wins.
///
/// # Arguments
///
/// * `project_root` - The project root directory (used for
///   `.tsumugi/skills/`, `.claude/skills/`, `.agents/skills/`).
///
/// # Errors
///
/// Returns [`SkillError`] if a SKILL.md file exists but cannot be read
/// or contains invalid frontmatter.
pub async fn discover_skills(project_root: impl AsRef<Path>) -> Result<Vec<SkillMeta>, SkillError> {
    discover_skills_with_config(project_root, &SkillsConfig::default()).await
}

/// Discover all skills using a custom [`SkillsConfig`].
///
/// This allows controlling which compatibility paths are enabled
/// and adding extra discovery directories.
///
/// # Errors
///
/// Returns [`SkillError`] if a SKILL.md file exists but cannot be read
/// or contains invalid frontmatter.
pub async fn discover_skills_with_config(
    project_root: impl AsRef<Path>,
    config: &SkillsConfig,
) -> Result<Vec<SkillMeta>, SkillError> {
    let project_root = project_root.as_ref();
    let mut skills_by_name: HashMap<SkillName, SkillMeta> = HashMap::new();

    // Build the ordered list of directories to scan.
    let scan_dirs = build_scan_dirs(project_root, config);

    for (source, dir) in &scan_dirs {
        let entries = match tokio::fs::read_dir(dir).await {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => continue,
            Err(e) => {
                return Err(SkillError::io(
                    format!("reading skill directory {}", dir.display()),
                    e,
                ));
            }
        };

        scan_directory(entries, *source, &mut skills_by_name).await?;
    }

    // Collect into a sorted Vec for deterministic output.
    let mut skills: Vec<SkillMeta> = skills_by_name.into_values().collect();
    skills.sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));

    Ok(skills)
}

/// Build the ordered list of directories to scan based on config.
fn build_scan_dirs(project_root: &Path, config: &SkillsConfig) -> Vec<(SkillSource, PathBuf)> {
    let mut dirs = Vec::new();

    // 1. Project-local tsumugi skills (always).
    dirs.push((
        SkillSource::ProjectTsumugi,
        project_root.join(".tsumugi").join("skills"),
    ));

    // 2. Global config skills (always).
    if let Some(config_dir) = dirs::config_dir() {
        dirs.push((
            SkillSource::GlobalConfig,
            config_dir.join("tsumugi").join("skills"),
        ));
    }

    // 3. Additional discovery paths from config.
    // These use GlobalConfig source since they are user-configured paths.
    for extra_path in &config.discovery_paths {
        dirs.push((SkillSource::GlobalConfig, extra_path.clone()));
    }

    // 4. .claude/skills/ compatibility.
    if config.compat_claude {
        dirs.push((
            SkillSource::ProjectClaude,
            project_root.join(".claude").join("skills"),
        ));
    }

    // 5. .agents/skills/ compatibility.
    if config.compat_agent_skills {
        dirs.push((
            SkillSource::ProjectAgents,
            project_root.join(".agents").join("skills"),
        ));
    }

    dirs
}

/// Scan a single directory for skill subdirectories containing SKILL.md.
async fn scan_directory(
    mut entries: tokio::fs::ReadDir,
    source: SkillSource,
    skills_by_name: &mut HashMap<SkillName, SkillMeta>,
) -> Result<(), SkillError> {
    loop {
        let entry = match entries.next_entry().await {
            Ok(Some(entry)) => entry,
            Ok(None) => break,
            Err(e) => {
                return Err(SkillError::io("reading directory entry", e));
            }
        };

        // Each skill lives in a subdirectory containing a SKILL.md.
        let entry_path = entry.path();

        let is_dir = match entry.file_type().await {
            Ok(ft) => ft.is_dir(),
            Err(_) => continue,
        };

        if !is_dir {
            continue;
        }

        let skill_file = entry_path.join(SKILL_FILENAME);
        let content = match tokio::fs::read_to_string(&skill_file).await {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                return Err(SkillError::io(
                    format!("reading {}", skill_file.display()),
                    e,
                ));
            }
        };

        let file_path_str = skill_file.display().to_string();
        let (frontmatter, _body) = parse_skill_md(&content, &file_path_str)?;

        let name = SkillName::new(&frontmatter.name);

        // Only insert if no higher-priority skill with the same name exists.
        // Since we iterate in priority order, we skip if already present.
        if !skills_by_name.contains_key(&name) {
            skills_by_name.insert(
                name.clone(),
                SkillMeta {
                    name,
                    frontmatter,
                    source,
                    path: SkillPath::new(skill_file),
                },
            );
        }
    }

    Ok(())
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions")]
mod tests {
    use super::*;

    #[tokio::test]
    async fn discover_empty_project() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let skills = discover_skills(tmp.path())
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(skills.is_empty());
    }

    #[tokio::test]
    async fn discover_single_skill() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let skill_dir = tmp.path().join(".tsumugi").join("skills").join("my-skill");
        std::fs::create_dir_all(&skill_dir).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: my-skill\ndescription: A test skill\n---\n\nBody here.\n",
        )
        .unwrap_or_else(|e| panic!("{e}"));

        let skills = discover_skills(tmp.path())
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name.as_str(), "my-skill");
        assert_eq!(skills[0].source, SkillSource::ProjectTsumugi);
    }

    #[tokio::test]
    async fn higher_priority_shadows_lower() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));

        // Create same-named skill in two different sources.
        let tsumugi_dir = tmp.path().join(".tsumugi").join("skills").join("dup-skill");
        std::fs::create_dir_all(&tsumugi_dir).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(
            tsumugi_dir.join("SKILL.md"),
            "---\nname: dup-skill\ndescription: From tsumugi\n---\n\nTsumugi body.\n",
        )
        .unwrap_or_else(|e| panic!("{e}"));

        let claude_dir = tmp.path().join(".claude").join("skills").join("dup-skill");
        std::fs::create_dir_all(&claude_dir).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(
            claude_dir.join("SKILL.md"),
            "---\nname: dup-skill\ndescription: From claude\n---\n\nClaude body.\n",
        )
        .unwrap_or_else(|e| panic!("{e}"));

        let skills = discover_skills(tmp.path())
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].frontmatter.description, "From tsumugi");
        assert_eq!(skills[0].source, SkillSource::ProjectTsumugi);
    }

    #[tokio::test]
    async fn discover_multiple_skills_sorted() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let base = tmp.path().join(".tsumugi").join("skills");

        for name in &["zebra-skill", "alpha-skill", "middle-skill"] {
            let dir = base.join(name);
            std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
            std::fs::write(
                dir.join("SKILL.md"),
                format!("---\nname: {name}\ndescription: Skill {name}\n---\n\nBody.\n"),
            )
            .unwrap_or_else(|e| panic!("{e}"));
        }

        let skills = discover_skills(tmp.path())
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(skills.len(), 3);
        assert_eq!(skills[0].name.as_str(), "alpha-skill");
        assert_eq!(skills[1].name.as_str(), "middle-skill");
        assert_eq!(skills[2].name.as_str(), "zebra-skill");
    }

    #[tokio::test]
    async fn compat_claude_disabled() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));

        // Create a skill in .claude/skills/.
        let claude_dir = tmp
            .path()
            .join(".claude")
            .join("skills")
            .join("claude-skill");
        std::fs::create_dir_all(&claude_dir).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(
            claude_dir.join("SKILL.md"),
            "---\nname: claude-skill\ndescription: From claude\n---\n\nBody.\n",
        )
        .unwrap_or_else(|e| panic!("{e}"));

        // Disable compat_claude.
        let config = SkillsConfig {
            discovery_paths: Vec::new(),
            compat_claude: false,
            compat_agent_skills: true,
        };

        let skills = discover_skills_with_config(tmp.path(), &config)
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(skills.is_empty());
    }

    #[tokio::test]
    async fn compat_agent_skills_disabled() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));

        // Create a skill in .agents/skills/.
        let agents_dir = tmp
            .path()
            .join(".agents")
            .join("skills")
            .join("agent-skill");
        std::fs::create_dir_all(&agents_dir).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(
            agents_dir.join("SKILL.md"),
            "---\nname: agent-skill\ndescription: From agents\n---\n\nBody.\n",
        )
        .unwrap_or_else(|e| panic!("{e}"));

        // Disable compat_agent_skills.
        let config = SkillsConfig {
            discovery_paths: Vec::new(),
            compat_claude: true,
            compat_agent_skills: false,
        };

        let skills = discover_skills_with_config(tmp.path(), &config)
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(skills.is_empty());
    }

    #[tokio::test]
    async fn additional_discovery_paths() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));

        // Create a skill in an additional discovery path.
        let extra_dir = tmp.path().join("extra-skills").join("bonus-skill");
        std::fs::create_dir_all(&extra_dir).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(
            extra_dir.join("SKILL.md"),
            "---\nname: bonus-skill\ndescription: From extra\n---\n\nBody.\n",
        )
        .unwrap_or_else(|e| panic!("{e}"));

        let config = SkillsConfig {
            discovery_paths: vec![tmp.path().join("extra-skills")],
            compat_claude: false,
            compat_agent_skills: false,
        };

        let skills = discover_skills_with_config(tmp.path(), &config)
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name.as_str(), "bonus-skill");
    }
}
