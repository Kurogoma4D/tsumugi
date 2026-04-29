//! tmg-skills: Skill discovery, loading, and invocation for tsumugi.
//!
//! This crate implements the Agent Skills system: parsing SKILL.md files
//! with YAML frontmatter, discovering skills from multiple directories
//! with priority ordering, loading skill content on demand, and providing
//! a `use_skill` tool for LLM invocation.
//!
//! ## Skill directory structure
//!
//! ```text
//! .tsumugi/skills/
//!   my-skill/
//!     SKILL.md          # Frontmatter + instruction body
//!     scripts/          # Optional helper scripts
//!     references/       # Optional reference files
//! ```
//!
//! ## Discovery priority
//!
//! 1. `.tsumugi/skills/` (project-local, highest priority)
//! 2. `~/.config/tsumugi/skills/` (global config)
//! 3. Additional paths from `[skills].discovery_paths` in `tsumugi.toml`
//! 4. `.claude/skills/` (compatibility, controlled by `compat_claude`)
//! 5. `.agents/skills/` (compatibility, controlled by `compat_agent_skills`)

pub mod banner;
pub mod config;
pub mod critic;
pub mod discovery;
pub mod emergence;
pub mod error;
pub mod loader;
pub mod manage;
pub mod metrics;
pub mod parse;
pub mod slash;
pub mod tool;
pub mod types;

pub use banner::{format_banner, load_acknowledged, pending_banner_names, save_acknowledged};
pub use config::SkillsConfig;
pub use critic::{
    CriticParseError, DEFAULT_SYSTEM_PROMPT, SkillCriticConfig, SkillCriticVerdict, parse_verdict,
};
pub use discovery::{discover_skills, discover_skills_with_config};
pub use emergence::{
    SignalCollector, SkillCandidacySignal, ToolCallOutcome, TriggerKind, TurnSummary,
};
pub use error::SkillError;
pub use loader::{format_skill_for_tool_result, format_skill_metadata, load_skill};
pub use manage::{SkillManageTool, regenerate_index};
pub use metrics::{METRICS_FILENAME, SkillMetrics, load_metrics, save_metrics, update_metrics};
pub use parse::parse_skill_md;
pub use slash::{SlashParseError, SlashParseResult, format_skills_list, parse_slash_command};
pub use tool::UseSkillTool;
pub use types::{
    InvocationPolicy, Provenance, SkillContent, SkillFrontmatter, SkillMeta, SkillName, SkillPath,
    SkillSource, SlashCommand,
};
