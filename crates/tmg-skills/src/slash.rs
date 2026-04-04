//! Slash command parsing for skill invocation.
//!
//! Supports:
//! - `/skills` — list all installed skills
//! - `/<skill-name>` — invoke a skill explicitly (with optional arguments)

use std::fmt::Write as _;

use crate::types::{SkillName, SlashCommand};

/// Try to parse user input as a slash command.
///
/// Returns `None` if the input does not start with `/` or is not a
/// recognized command pattern.
pub fn parse_slash_command(input: &str) -> Option<SlashCommand> {
    let trimmed = input.trim();

    let rest = trimmed.strip_prefix('/')?;

    if rest.is_empty() {
        return None;
    }

    // Split into command name and optional arguments.
    let (command, args) = match rest.split_once(char::is_whitespace) {
        Some((cmd, remainder)) => {
            let remainder = remainder.trim();
            let args = if remainder.is_empty() {
                None
            } else {
                Some(remainder.to_owned())
            };
            (cmd, args)
        }
        None => (rest, None),
    };

    if command == "skills" {
        return Some(SlashCommand::ListSkills);
    }

    // Any other `/foo` is treated as a skill invocation.
    Some(SlashCommand::InvokeSkill {
        name: SkillName::new(command),
        args,
    })
}

/// Format a list of skills for display (e.g. in TUI).
pub fn format_skills_list(skills: &[crate::types::SkillMeta]) -> String {
    if skills.is_empty() {
        return "No skills installed.".to_owned();
    }

    let mut output = String::from("Installed skills:\n\n");
    for skill in skills {
        let _ = writeln!(
            output,
            "  /{} — {} [{}] (from {})",
            skill.name, skill.frontmatter.description, skill.frontmatter.invocation, skill.source,
        );
    }
    output
}

impl std::fmt::Display for crate::types::InvocationPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::ExplicitOnly => write!(f, "explicit_only"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_skills_command() {
        let cmd = parse_slash_command("/skills");
        assert_eq!(cmd, Some(SlashCommand::ListSkills));
    }

    #[test]
    fn parse_skills_command_with_whitespace() {
        let cmd = parse_slash_command("  /skills  ");
        assert_eq!(cmd, Some(SlashCommand::ListSkills));
    }

    #[test]
    fn parse_skill_invocation_no_args() {
        let cmd = parse_slash_command("/code-review");
        assert_eq!(
            cmd,
            Some(SlashCommand::InvokeSkill {
                name: SkillName::new("code-review"),
                args: None,
            })
        );
    }

    #[test]
    fn parse_skill_invocation_with_args() {
        let cmd = parse_slash_command("/test-runner src/main.rs");
        assert_eq!(
            cmd,
            Some(SlashCommand::InvokeSkill {
                name: SkillName::new("test-runner"),
                args: Some("src/main.rs".to_owned()),
            })
        );
    }

    #[test]
    fn parse_empty_slash() {
        let cmd = parse_slash_command("/");
        assert_eq!(cmd, None);
    }

    #[test]
    fn parse_no_slash() {
        let cmd = parse_slash_command("hello world");
        assert_eq!(cmd, None);
    }

    #[test]
    fn format_empty_list() {
        let output = format_skills_list(&[]);
        assert_eq!(output, "No skills installed.");
    }

    #[test]
    fn format_nonempty_list() {
        use crate::types::*;

        let skills = vec![SkillMeta {
            name: SkillName::new("test-skill"),
            frontmatter: SkillFrontmatter {
                name: "test-skill".to_owned(),
                description: "A test skill".to_owned(),
                invocation: InvocationPolicy::Auto,
                allowed_tools: None,
            },
            source: SkillSource::ProjectTsumugi,
            path: SkillPath::new("/fake/SKILL.md"),
        }];

        let output = format_skills_list(&skills);
        assert!(output.contains("/test-skill"));
        assert!(output.contains("A test skill"));
        assert!(output.contains("auto"));
        assert!(output.contains(".tsumugi/skills"));
    }
}
