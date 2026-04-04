//! Skill loading: reads the full SKILL.md body and associated resource files.

use std::fmt::Write as _;
use std::path::Path;

use crate::error::SkillError;
use crate::parse::parse_skill_md;
use crate::types::{InvocationPolicy, SkillContent, SkillMeta};

/// Load the full content of a skill, including its instruction body
/// and lists of associated scripts/ and references/ files.
///
/// # Errors
///
/// Returns [`SkillError::Io`] if the SKILL.md or resource directories
/// cannot be read.
pub async fn load_skill(meta: &SkillMeta) -> Result<SkillContent, SkillError> {
    let skill_file = meta.path.as_path();
    let content = tokio::fs::read_to_string(skill_file)
        .await
        .map_err(|e| SkillError::io(format!("reading {}", skill_file.display()), e))?;

    let file_path_str = skill_file.display().to_string();
    let (_frontmatter, body) = parse_skill_md(&content, &file_path_str)?;

    // The skill directory is the parent of SKILL.md.
    let skill_dir = skill_file.parent().unwrap_or(Path::new("."));

    let scripts = list_files_in_subdir(skill_dir, "scripts").await?;
    let references = list_files_in_subdir(skill_dir, "references").await?;

    Ok(SkillContent {
        meta: meta.clone(),
        body,
        scripts,
        references,
    })
}

/// List files in a subdirectory (non-recursive). Returns an empty Vec
/// if the subdirectory does not exist.
async fn list_files_in_subdir(
    base: &Path,
    subdir: &str,
) -> Result<Vec<std::path::PathBuf>, SkillError> {
    let dir = base.join(subdir);
    let mut entries = match tokio::fs::read_dir(&dir).await {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(SkillError::io(format!("reading {}", dir.display()), e));
        }
    };

    let mut files = Vec::new();
    loop {
        match entries.next_entry().await {
            Ok(Some(entry)) => {
                let ft = entry
                    .file_type()
                    .await
                    .map_err(|e| SkillError::io("reading file type", e))?;
                if ft.is_file() {
                    files.push(entry.path());
                }
            }
            Ok(None) => break,
            Err(e) => {
                return Err(SkillError::io("reading directory entry", e));
            }
        }
    }

    files.sort();
    Ok(files)
}

/// Format skill metadata for injection into the system prompt.
///
/// Each skill is represented as two lines:
/// ```text
/// - <name>: <description>
///   invocation: <auto|explicit_only>
/// ```
///
/// Only skills with `invocation: auto` are included, since
/// `explicit_only` skills should not appear in auto-discovery metadata.
pub fn format_skill_metadata(skills: &[SkillMeta]) -> String {
    let mut output = String::new();

    let auto_skills: Vec<&SkillMeta> = skills
        .iter()
        .filter(|s| s.frontmatter.invocation == InvocationPolicy::Auto)
        .collect();

    if auto_skills.is_empty() {
        return output;
    }

    output.push_str("Available skills (use `use_skill` tool to invoke):\n");
    for skill in auto_skills {
        let _ = writeln!(
            output,
            "- {}: {}",
            skill.name, skill.frontmatter.description
        );
    }

    output
}

/// Format a loaded skill for inclusion in a tool result.
///
/// Includes the instruction body and lists of scripts/ and references/ files.
pub fn format_skill_for_tool_result(content: &SkillContent) -> String {
    let mut result = String::new();

    let _ = writeln!(result, "# Skill: {}\n", content.meta.name);
    result.push_str(&content.body);

    if !content.scripts.is_empty() {
        result.push_str("\n\n## Scripts\n");
        for script in &content.scripts {
            let _ = writeln!(result, "- {}", script.display());
        }
    }

    if !content.references.is_empty() {
        result.push_str("\n\n## References\n");
        for reference in &content.references {
            let _ = writeln!(result, "- {}", reference.display());
        }
    }

    if let Some(tools) = &content.meta.frontmatter.allowed_tools {
        result.push_str("\n\n## Allowed Tools\n");
        for tool in tools {
            let _ = writeln!(result, "- {tool}");
        }
    }

    result
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions")]
mod tests {
    use super::*;
    use crate::types::{InvocationPolicy, SkillFrontmatter, SkillName, SkillPath, SkillSource};

    fn sample_meta(name: &str, invocation: InvocationPolicy) -> SkillMeta {
        SkillMeta {
            name: SkillName::new(name),
            frontmatter: SkillFrontmatter {
                name: name.to_owned(),
                description: format!("Description of {name}"),
                invocation,
                allowed_tools: None,
            },
            source: SkillSource::ProjectTsumugi,
            path: SkillPath::new(format!("/fake/{name}/SKILL.md")),
        }
    }

    #[test]
    fn format_metadata_auto_only() {
        let skills = vec![
            sample_meta("auto-skill", InvocationPolicy::Auto),
            sample_meta("explicit-skill", InvocationPolicy::ExplicitOnly),
            sample_meta("another-auto", InvocationPolicy::Auto),
        ];

        let output = format_skill_metadata(&skills);
        assert!(output.contains("auto-skill"));
        assert!(output.contains("another-auto"));
        assert!(!output.contains("explicit-skill"));
    }

    #[test]
    fn format_metadata_empty() {
        let output = format_skill_metadata(&[]);
        assert!(output.is_empty());
    }

    #[test]
    fn format_metadata_all_explicit() {
        let skills = vec![sample_meta("explicit", InvocationPolicy::ExplicitOnly)];
        let output = format_skill_metadata(&skills);
        assert!(output.is_empty());
    }

    #[tokio::test]
    async fn load_skill_with_resources() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let skill_dir = tmp.path().join("my-skill");
        std::fs::create_dir_all(skill_dir.join("scripts")).unwrap_or_else(|e| panic!("{e}"));
        std::fs::create_dir_all(skill_dir.join("references")).unwrap_or_else(|e| panic!("{e}"));

        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: my-skill\ndescription: Test\n---\n\nDo things.\n",
        )
        .unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(skill_dir.join("scripts").join("run.sh"), "#!/bin/bash\n")
            .unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(skill_dir.join("references").join("guide.md"), "# Guide\n")
            .unwrap_or_else(|e| panic!("{e}"));

        let meta = SkillMeta {
            name: SkillName::new("my-skill"),
            frontmatter: SkillFrontmatter {
                name: "my-skill".to_owned(),
                description: "Test".to_owned(),
                invocation: InvocationPolicy::Auto,
                allowed_tools: None,
            },
            source: SkillSource::ProjectTsumugi,
            path: SkillPath::new(skill_dir.join("SKILL.md")),
        };

        let content = load_skill(&meta).await.unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(content.body, "Do things.\n");
        assert_eq!(content.scripts.len(), 1);
        assert_eq!(content.references.len(), 1);
    }

    #[test]
    fn format_tool_result_with_resources() {
        let content = SkillContent {
            meta: SkillMeta {
                name: SkillName::new("test-skill"),
                frontmatter: SkillFrontmatter {
                    name: "test-skill".to_owned(),
                    description: "A test".to_owned(),
                    invocation: InvocationPolicy::Auto,
                    allowed_tools: Some(vec!["file_read".to_owned()]),
                },
                source: SkillSource::ProjectTsumugi,
                path: SkillPath::new("/fake/SKILL.md"),
            },
            body: "Instructions here.\n".to_owned(),
            scripts: vec!["/fake/scripts/run.sh".into()],
            references: vec!["/fake/references/guide.md".into()],
        };

        let result = format_skill_for_tool_result(&content);
        assert!(result.contains("# Skill: test-skill"));
        assert!(result.contains("Instructions here."));
        assert!(result.contains("## Scripts"));
        assert!(result.contains("run.sh"));
        assert!(result.contains("## References"));
        assert!(result.contains("guide.md"));
        assert!(result.contains("## Allowed Tools"));
        assert!(result.contains("file_read"));
    }
}
