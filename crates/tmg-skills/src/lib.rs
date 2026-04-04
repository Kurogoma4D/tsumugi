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
//! 3. `.claude/skills/` (compatibility)
//! 4. `.agents/skills/` (compatibility)

pub mod discovery;
pub mod error;
pub mod loader;
pub mod parse;
pub mod slash;
pub mod tool;
pub mod types;

pub use discovery::discover_skills;
pub use error::SkillError;
pub use loader::{format_skill_for_tool_result, format_skill_metadata, load_skill};
pub use parse::parse_skill_md;
pub use slash::{format_skills_list, parse_slash_command};
pub use tool::UseSkillTool;
pub use types::{
    InvocationPolicy, SkillContent, SkillFrontmatter, SkillMeta, SkillName, SkillPath, SkillSource,
    SlashCommand,
};
