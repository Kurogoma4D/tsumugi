//! Skill configuration from `tsumugi.toml`.
//!
//! The `[skills]` section of `tsumugi.toml` allows customizing skill
//! discovery behavior:
//!
//! ```toml
//! [skills]
//! discovery_paths = ["~/my-skills", "/shared/skills"]
//! compat_claude = true
//! compat_agent_skills = false
//! ```

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Configuration for the skills system from `tsumugi.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillsConfig {
    /// Additional directories to search for skills.
    ///
    /// These are scanned after the built-in discovery paths but before
    /// compatibility paths. Each path should point to a directory
    /// containing skill subdirectories.
    #[serde(default)]
    pub discovery_paths: Vec<PathBuf>,

    /// Whether to search `.claude/skills/` for compatibility.
    ///
    /// Defaults to `true` to maintain backward compatibility.
    #[serde(default = "default_true")]
    pub compat_claude: bool,

    /// Whether to search `.agents/skills/` for compatibility.
    ///
    /// Defaults to `true` to maintain backward compatibility.
    #[serde(default = "default_true")]
    pub compat_agent_skills: bool,
}

fn default_true() -> bool {
    true
}

impl Default for SkillsConfig {
    fn default() -> Self {
        Self {
            discovery_paths: Vec::new(),
            compat_claude: true,
            compat_agent_skills: true,
        }
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions")]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = SkillsConfig::default();
        assert!(config.discovery_paths.is_empty());
        assert!(config.compat_claude);
        assert!(config.compat_agent_skills);
    }

    #[test]
    fn deserialize_empty_section() {
        let toml_str = "";
        let config: SkillsConfig = toml::from_str(toml_str).unwrap_or_else(|e| panic!("{e}"));
        assert!(config.discovery_paths.is_empty());
        assert!(config.compat_claude);
        assert!(config.compat_agent_skills);
    }

    #[test]
    fn deserialize_with_paths() {
        let toml_str = r#"
discovery_paths = ["/extra/skills", "~/my-skills"]
compat_claude = false
compat_agent_skills = true
"#;
        let config: SkillsConfig = toml::from_str(toml_str).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(config.discovery_paths.len(), 2);
        assert!(!config.compat_claude);
        assert!(config.compat_agent_skills);
    }

    #[test]
    fn deserialize_partial() {
        let toml_str = "compat_claude = false\n";
        let config: SkillsConfig = toml::from_str(toml_str).unwrap_or_else(|e| panic!("{e}"));
        assert!(config.discovery_paths.is_empty());
        assert!(!config.compat_claude);
        assert!(config.compat_agent_skills);
    }

    #[test]
    fn roundtrip_serde() {
        let original = SkillsConfig {
            discovery_paths: vec![PathBuf::from("/extra/skills")],
            compat_claude: false,
            compat_agent_skills: true,
        };
        let toml_str = toml::to_string_pretty(&original).unwrap_or_else(|e| panic!("{e}"));
        let parsed: SkillsConfig = toml::from_str(&toml_str).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(original, parsed);
    }
}
