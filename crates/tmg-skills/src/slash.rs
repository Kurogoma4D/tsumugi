//! Slash command parsing for skill invocation and TUI-level commands.
//!
//! Supports:
//! - `/skills` — list all installed skills
//! - `/<skill-name>` — invoke a skill explicitly (with optional arguments)
//! - `/clear`, `/compact`, `/exit`, `/quit`, `/agents`, `/workflows`
//! - `/run start|resume|list|status|upgrade|downgrade|new-session|pause|abort ...`
//!
//! Anything beginning with `/run ` falls through a dedicated parser
//! that returns either a typed [`SlashCommand`] or a structured
//! [`SlashParseError`] when the subcommand is unknown / mistyped.
//! Other slashes (`/skill-name`, `/agents`, ...) parse straight into
//! their typed variants. Returning an error rather than `None` for
//! malformed `/run` input lets the TUI surface a precise message.

use std::fmt::Write as _;

use crate::types::{SkillName, SlashCommand};

/// Errors produced by [`parse_slash_command`] for input that *looked*
/// like a slash command but failed to match any known shape.
///
/// The TUI uses the `Display` impl directly to feed `App::set_error`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashParseError {
    /// `/run` was supplied without a subcommand (or with an unknown
    /// one). The carried string is the raw remainder after `/run`.
    UnknownRunSubcommand(String),
    /// `/run start` requires a workflow id.
    MissingWorkflowId,
}

impl std::fmt::Display for SlashParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownRunSubcommand(rest) => {
                if rest.is_empty() {
                    write!(
                        f,
                        "/run requires a subcommand: start, resume, list, status, \
                         upgrade, downgrade, new-session, pause, or abort"
                    )
                } else {
                    write!(
                        f,
                        "unknown /run subcommand: {rest:?}; expected one of \
                         start, resume, list, status, upgrade, downgrade, \
                         new-session, pause, abort"
                    )
                }
            }
            Self::MissingWorkflowId => {
                write!(
                    f,
                    "/run start requires a workflow id (e.g. /run start build)"
                )
            }
        }
    }
}

impl std::error::Error for SlashParseError {}

/// Outcome of [`parse_slash_command`]:
///
/// * `Ok(Some(cmd))` — the input was a recognised slash command.
/// * `Ok(None)` — the input was not a slash command at all (no leading
///   `/`, or just `/`). Callers should treat this as ordinary chat
///   text.
/// * `Err(SlashParseError)` — the input was a slash command but had a
///   malformed subcommand. Surface to the user.
pub type SlashParseResult = Result<Option<SlashCommand>, SlashParseError>;

/// Try to parse user input as a slash command.
///
/// Returns:
/// - `Ok(Some(cmd))` when the input matches a known command.
/// - `Ok(None)` when the input is not a slash command at all.
/// - `Err(SlashParseError)` when the input *looked* like a command but
///   could not be parsed (currently only the `/run` family produces
///   structured errors).
pub fn parse_slash_command(input: &str) -> SlashParseResult {
    let trimmed = input.trim();

    let Some(rest) = trimmed.strip_prefix('/') else {
        return Ok(None);
    };

    if rest.is_empty() {
        return Ok(None);
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

    // A leading-whitespace input like "/  foo" splits into
    // (command = "", args = Some("foo")), which would otherwise be
    // dispatched as `InvokeSkill { name: "" }`. An empty skill name is
    // never valid, so fall through to chat (the same path taken by a
    // bare `/`).
    if command.is_empty() {
        return Ok(None);
    }

    // Built-in commands first; the broad-net "any other /foo is a
    // skill invocation" path lives at the bottom so it never shadows
    // a typed command.
    match command {
        "skills" => return Ok(Some(SlashCommand::ListSkills)),
        "clear" => return Ok(Some(SlashCommand::Clear)),
        "compact" => return Ok(Some(SlashCommand::Compact)),
        "exit" | "quit" => return Ok(Some(SlashCommand::Exit)),
        "agents" => return Ok(Some(SlashCommand::ListAgents)),
        "workflows" => return Ok(Some(SlashCommand::ListWorkflows)),
        "run" => return parse_run_subcommand(args.as_deref()).map(Some),
        "memory" => return Ok(Some(parse_memory_subcommand(args.as_deref()))),
        "search" => {
            return Ok(Some(SlashCommand::Search {
                query: args.unwrap_or_default(),
            }));
        }
        _ => {}
    }

    // Any other `/foo` is treated as a skill invocation.
    Ok(Some(SlashCommand::InvokeSkill {
        name: SkillName::new(command),
        args,
    }))
}

/// Parse the body of a `/run ...` invocation.
///
/// `args` is the trimmed remainder after `/run`. The empty / `None`
/// case maps to [`SlashParseError::UnknownRunSubcommand`] with an empty
/// remainder so the user sees the full subcommand list.
fn parse_run_subcommand(args: Option<&str>) -> Result<SlashCommand, SlashParseError> {
    let body = args.unwrap_or("").trim();
    if body.is_empty() {
        return Err(SlashParseError::UnknownRunSubcommand(String::new()));
    }
    let (sub, rest) = match body.split_once(char::is_whitespace) {
        Some((s, r)) => (s, r.trim()),
        None => (body, ""),
    };
    let rest_opt = if rest.is_empty() {
        None
    } else {
        Some(rest.to_owned())
    };

    match sub {
        "start" => {
            // First whitespace-delimited token after `start` is the
            // workflow id; anything past that becomes the trailing
            // `args` string for the TUI to pass to the tool.
            let body_after_start = rest;
            if body_after_start.is_empty() {
                return Err(SlashParseError::MissingWorkflowId);
            }
            let (workflow, args_after) = match body_after_start.split_once(char::is_whitespace) {
                Some((w, a)) => (w.to_owned(), {
                    let trimmed = a.trim();
                    if trimmed.is_empty() {
                        None
                    } else {
                        Some(trimmed.to_owned())
                    }
                }),
                None => (body_after_start.to_owned(), None),
            };
            Ok(SlashCommand::RunStart {
                workflow,
                args: args_after,
            })
        }
        "resume" => Ok(SlashCommand::RunResume { run_id: rest_opt }),
        "list" => Ok(SlashCommand::RunList),
        "status" => Ok(SlashCommand::RunStatus { run_id: rest_opt }),
        "upgrade" => Ok(SlashCommand::RunUpgrade),
        "downgrade" => Ok(SlashCommand::RunDowngrade),
        "new-session" | "new_session" => Ok(SlashCommand::RunNewSession),
        "pause" => Ok(SlashCommand::RunPause),
        "abort" => Ok(SlashCommand::RunAbort),
        other => Err(SlashParseError::UnknownRunSubcommand(other.to_owned())),
    }
}

/// Parse the body of a `/memory ...` invocation.
///
/// `/memory` (no body) → [`SlashCommand::MemoryIndex`].
/// `/memory show <name>` → [`SlashCommand::MemoryShow`].
/// Anything else falls back to `MemoryIndex` so the user always gets
/// a useful response (the malformed shape is documented in the body
/// the TUI prints).
fn parse_memory_subcommand(args: Option<&str>) -> SlashCommand {
    let body = args.unwrap_or("").trim();
    if body.is_empty() {
        return SlashCommand::MemoryIndex;
    }
    let (sub, rest) = match body.split_once(char::is_whitespace) {
        Some((s, r)) => (s, r.trim()),
        None => (body, ""),
    };
    if sub == "show" && !rest.is_empty() {
        return SlashCommand::MemoryShow {
            name: rest.to_owned(),
        };
    }
    SlashCommand::MemoryIndex
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
#[expect(
    clippy::expect_used,
    clippy::panic,
    clippy::match_wildcard_for_single_variants,
    reason = "tests assert with expect/panic for clarity; the workspace policy denies them in production code"
)]
mod tests {
    use super::*;

    #[test]
    fn parse_skills_command() {
        let cmd = parse_slash_command("/skills").expect("ok");
        assert_eq!(cmd, Some(SlashCommand::ListSkills));
    }

    #[test]
    fn parse_skills_command_with_whitespace() {
        let cmd = parse_slash_command("  /skills  ").expect("ok");
        assert_eq!(cmd, Some(SlashCommand::ListSkills));
    }

    #[test]
    fn parse_skill_invocation_no_args() {
        let cmd = parse_slash_command("/code-review").expect("ok");
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
        let cmd = parse_slash_command("/test-runner src/main.rs").expect("ok");
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
        let cmd = parse_slash_command("/").expect("ok");
        assert_eq!(cmd, None);
    }

    #[test]
    fn parse_slash_with_leading_whitespace_is_not_skill() {
        // Inputs like "/  foo" and "/ " must not produce a skill
        // invocation with an empty name; they should fall through to
        // chat (Ok(None)).
        assert_eq!(parse_slash_command("/  foo").expect("ok"), None);
        assert_eq!(parse_slash_command("/ ").expect("ok"), None);
        assert_eq!(parse_slash_command("/\t").expect("ok"), None);
    }

    #[test]
    fn parse_no_slash() {
        let cmd = parse_slash_command("hello world").expect("ok");
        assert_eq!(cmd, None);
    }

    #[test]
    fn parse_clear_compact_exit_quit() {
        assert_eq!(
            parse_slash_command("/clear").expect("ok"),
            Some(SlashCommand::Clear)
        );
        assert_eq!(
            parse_slash_command("/compact").expect("ok"),
            Some(SlashCommand::Compact)
        );
        assert_eq!(
            parse_slash_command("/exit").expect("ok"),
            Some(SlashCommand::Exit)
        );
        assert_eq!(
            parse_slash_command("/quit").expect("ok"),
            Some(SlashCommand::Exit)
        );
    }

    #[test]
    fn parse_agents_workflows() {
        assert_eq!(
            parse_slash_command("/agents").expect("ok"),
            Some(SlashCommand::ListAgents)
        );
        assert_eq!(
            parse_slash_command("/workflows").expect("ok"),
            Some(SlashCommand::ListWorkflows)
        );
    }

    #[test]
    fn parse_run_start() {
        assert_eq!(
            parse_slash_command("/run start build").expect("ok"),
            Some(SlashCommand::RunStart {
                workflow: "build".to_owned(),
                args: None,
            })
        );
        assert_eq!(
            parse_slash_command("/run start build target=foo").expect("ok"),
            Some(SlashCommand::RunStart {
                workflow: "build".to_owned(),
                args: Some("target=foo".to_owned()),
            })
        );
    }

    #[test]
    fn parse_run_start_without_workflow_errors() {
        let err = parse_slash_command("/run start").expect_err("missing workflow id");
        assert_eq!(err, SlashParseError::MissingWorkflowId);
    }

    #[test]
    fn parse_run_subcommands() {
        assert_eq!(
            parse_slash_command("/run resume").expect("ok"),
            Some(SlashCommand::RunResume { run_id: None })
        );
        assert_eq!(
            parse_slash_command("/run resume 1234abcd").expect("ok"),
            Some(SlashCommand::RunResume {
                run_id: Some("1234abcd".to_owned()),
            })
        );
        assert_eq!(
            parse_slash_command("/run list").expect("ok"),
            Some(SlashCommand::RunList)
        );
        assert_eq!(
            parse_slash_command("/run status").expect("ok"),
            Some(SlashCommand::RunStatus { run_id: None })
        );
        assert_eq!(
            parse_slash_command("/run status 1234abcd").expect("ok"),
            Some(SlashCommand::RunStatus {
                run_id: Some("1234abcd".to_owned()),
            })
        );
        assert_eq!(
            parse_slash_command("/run upgrade").expect("ok"),
            Some(SlashCommand::RunUpgrade)
        );
        assert_eq!(
            parse_slash_command("/run downgrade").expect("ok"),
            Some(SlashCommand::RunDowngrade)
        );
        assert_eq!(
            parse_slash_command("/run new-session").expect("ok"),
            Some(SlashCommand::RunNewSession)
        );
        assert_eq!(
            parse_slash_command("/run new_session").expect("ok"),
            Some(SlashCommand::RunNewSession)
        );
        assert_eq!(
            parse_slash_command("/run pause").expect("ok"),
            Some(SlashCommand::RunPause)
        );
        assert_eq!(
            parse_slash_command("/run abort").expect("ok"),
            Some(SlashCommand::RunAbort)
        );
    }

    #[test]
    fn parse_run_unknown_subcommand_errors() {
        let err = parse_slash_command("/run bogus").expect_err("unknown subcommand");
        match err {
            SlashParseError::UnknownRunSubcommand(rest) => assert_eq!(rest, "bogus"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_memory_bare() {
        assert_eq!(
            parse_slash_command("/memory").expect("ok"),
            Some(SlashCommand::MemoryIndex)
        );
    }

    #[test]
    fn parse_memory_show() {
        assert_eq!(
            parse_slash_command("/memory show topic_one").expect("ok"),
            Some(SlashCommand::MemoryShow {
                name: "topic_one".to_owned(),
            })
        );
    }

    #[test]
    fn parse_run_without_subcommand_errors() {
        let err = parse_slash_command("/run").expect_err("missing subcommand");
        match err {
            SlashParseError::UnknownRunSubcommand(rest) => assert!(rest.is_empty()),
            other => panic!("unexpected: {other:?}"),
        }
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
