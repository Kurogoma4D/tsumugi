//! Domain types for the skills system.
//!
//! Uses newtypes for domain concepts to avoid primitive obsession.

use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Newtypes
// ---------------------------------------------------------------------------

/// A unique skill name (e.g. `"code-review"`, `"test-runner"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SkillName(String);

impl SkillName {
    /// Create a new `SkillName`.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Return the name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume self and return the inner `String`.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for SkillName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// An absolute path to a SKILL.md file.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SkillPath(PathBuf);

impl SkillPath {
    /// Create a new `SkillPath`.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self(path.into())
    }

    /// Return the inner path reference.
    pub fn as_path(&self) -> &std::path::Path {
        &self.0
    }

    /// Consume self and return the inner `PathBuf`.
    pub fn into_inner(self) -> PathBuf {
        self.0
    }
}

impl fmt::Display for SkillPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.display())
    }
}

// ---------------------------------------------------------------------------
// Invocation policy
// ---------------------------------------------------------------------------

/// Controls when a skill can be invoked by the LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum InvocationPolicy {
    /// The LLM may invoke the skill automatically when it determines
    /// the skill is relevant.
    #[default]
    Auto,

    /// The skill can only be invoked explicitly via a `/skill-name`
    /// slash command. It will not appear in auto-discovery metadata
    /// sent to the LLM.
    ExplicitOnly,
}

// ---------------------------------------------------------------------------
// Skill source (priority ordering)
// ---------------------------------------------------------------------------

/// Where a skill was discovered from, in descending priority order.
///
/// Skills from higher-priority sources shadow skills with the same name
/// from lower-priority sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SkillSource {
    /// `.tsumugi/skills/` in the project root (highest priority).
    ProjectTsumugi = 0,
    /// `~/.config/tsumugi/skills/` in the global config directory.
    GlobalConfig = 1,
    /// User-specified additional discovery path (from `[skills].discovery_paths`).
    Custom = 2,
    /// `.claude/skills/` in the project root.
    ProjectClaude = 3,
    /// `.agents/skills/` in the project root.
    ProjectAgents = 4,
}

impl fmt::Display for SkillSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProjectTsumugi => write!(f, ".tsumugi/skills"),
            Self::GlobalConfig => write!(f, "~/.config/tsumugi/skills"),
            Self::Custom => write!(f, "custom discovery path"),
            Self::ProjectClaude => write!(f, ".claude/skills"),
            Self::ProjectAgents => write!(f, ".agents/skills"),
        }
    }
}

// ---------------------------------------------------------------------------
// YAML frontmatter
// ---------------------------------------------------------------------------

/// The parsed YAML frontmatter from a SKILL.md file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillFrontmatter {
    /// The skill's human-readable name.
    pub name: String,

    /// A short description of what the skill does.
    pub description: String,

    /// How and when the skill can be invoked.
    #[serde(default)]
    pub invocation: InvocationPolicy,

    /// Optional list of tool names this skill is allowed to use.
    /// When `None`, all tools are available.
    #[serde(default)]
    pub allowed_tools: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Skill metadata (frontmatter + source info)
// ---------------------------------------------------------------------------

/// Metadata about a discovered skill (frontmatter + source info).
///
/// This is the lightweight representation used during discovery and
/// for system prompt injection. The full skill body is not loaded
/// until `load_skill()` is called.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillMeta {
    /// The skill's unique name (derived from frontmatter or directory name).
    pub name: SkillName,

    /// The parsed frontmatter.
    pub frontmatter: SkillFrontmatter,

    /// Where this skill was discovered from.
    pub source: SkillSource,

    /// The absolute path to the SKILL.md file.
    pub path: SkillPath,
}

// ---------------------------------------------------------------------------
// Full skill content
// ---------------------------------------------------------------------------

/// The full content of a loaded skill, including the instruction body
/// and any associated resource files.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillContent {
    /// The skill metadata.
    pub meta: SkillMeta,

    /// The instruction body (everything after the frontmatter).
    pub body: String,

    /// Paths to files in the `scripts/` subdirectory (if any).
    pub scripts: Vec<PathBuf>,

    /// Paths to files in the `references/` subdirectory (if any).
    pub references: Vec<PathBuf>,
}

// ---------------------------------------------------------------------------
// Slash command
// ---------------------------------------------------------------------------

/// A parsed slash command from user input.
///
/// The variants below cover both skill-related commands (the original
/// scope of this enum) and TUI-level operations (clear, exit, run-
/// management, workflow listing) consolidated here in issue #46 so a
/// single `dispatch_slash` site in `tmg-tui` can route every slash
/// command through one match.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SlashCommand {
    /// `/skills` — list all installed skills.
    ListSkills,

    /// `/<skill-name>` with optional arguments — invoke a skill explicitly.
    InvokeSkill {
        /// The skill name (without the leading `/`).
        name: SkillName,
        /// Optional arguments passed after the skill name.
        args: Option<String>,
    },

    // ---- TUI-level commands (issue #46) ----------------------------------
    /// `/clear` — wipe chat & tool log and reload the prompt files.
    Clear,
    /// `/compact` — trigger a context compression turn.
    Compact,
    /// `/exit` (or `/quit`) — request the application to exit.
    Exit,
    /// `/agents` — show the list of available subagent types and live
    /// subagent statuses.
    ListAgents,
    /// `/workflows` — show the list of discovered workflows.
    ListWorkflows,

    // ---- /run subcommands (SPEC §9.8) ------------------------------------
    /// `/run start <workflow> [args]` — start a workflow run.
    RunStart {
        /// Workflow id (matches `WorkflowDef::id`).
        workflow: String,
        /// Optional remaining arguments string (typically `key=value`
        /// pairs). The TUI passes this through to the workflow tool.
        args: Option<String>,
    },
    /// `/run resume [<id>]` — resume a paused or backgrounded run.
    RunResume {
        /// Optional explicit run id; when `None` the TUI defers to the
        /// auto-resume policy (most-recent resumable run).
        run_id: Option<String>,
    },
    /// `/run list` — list all runs in the run store.
    RunList,
    /// `/run status [<id>]` — show status of a run.
    RunStatus {
        /// Optional run id; when `None` the active run is used.
        run_id: Option<String>,
    },
    /// `/run upgrade` — promote the active run to harnessed scope.
    RunUpgrade,
    /// `/run downgrade` — demote the active run back to ad-hoc.
    RunDowngrade,
    /// `/run new-session` — manually rotate to a fresh session.
    RunNewSession,
    /// `/run pause` — flip the active run to `Paused`.
    RunPause,
    /// `/run abort` — flip the active run to `Failed { reason: "user
    /// aborted" }`.
    RunAbort,

    // ---- /memory subcommands (issue #52) --------------------------------
    /// `/memory` — show the merged memory index.
    MemoryIndex,
    /// `/memory show <name>` — show one entry's body.
    MemoryShow {
        /// Topic name.
        name: String,
    },

    // ---- /search (issue #53) --------------------------------------------
    /// `/search <query>` — full-text search the cross-session index.
    /// Result is rendered into the Activity Pane as a system message.
    Search {
        /// FTS5 MATCH expression. Empty when the user typed `/search`
        /// with no body — surfaced as a usage hint by the dispatcher.
        query: String,
    },
}
