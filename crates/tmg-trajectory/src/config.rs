//! `[trajectory]` configuration for `tsumugi.toml`.
//!
//! The crate carries its own config struct so the CLI's
//! `tsumugi.toml` parser can wire it in directly via
//! `#[serde(default)]` without depending on a third helper.
//!
//! # Defaults
//!
//! Every knob defaults to the privacy-preserving option:
//!
//! - `enabled = false` — opt-in only.
//! - `output_dir = "trajectories"` — under each run dir.
//! - `include_thinking = false` — keep size manageable.
//! - `include_tool_results = "redacted"` — never ship raw tool output.
//! - `redact_extra_patterns = []` — empty list.
//!
//! Operators flip individual knobs in `tsumugi.toml`; the recorder
//! validates extra patterns at construction so a bad regex surfaces
//! before the first session.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Rendering policy for tool-result records.
///
/// The variants live behind a small enum (rather than a free-form
/// string) so the CLI surfaces clap-side validation errors with the
/// list of accepted values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolResultMode {
    /// Pass the tool output through verbatim. Use only when the
    /// operator has accepted that the trajectory will contain raw
    /// payloads (handy for offline analysis on a trusted machine).
    Full,
    /// Redact via the shared regex set + `redact_extra_patterns`.
    /// This is the default and the recommended setting for shipping
    /// trajectories outside the recording machine.
    #[default]
    Redacted,
    /// Replace the payload with a one-line placeholder of the form
    /// `"<N chars, ok|err>"`. Useful when the conversation flow
    /// matters for training but the actual tool output is sensitive.
    SummaryOnly,
}

/// `[trajectory]` configuration block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryConfig {
    /// Master switch — `false` keeps the recorder unconstructed and
    /// no files are written.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Output directory **relative to each run dir**. The default
    /// `"trajectories"` produces files at
    /// `.tsumugi/runs/<run-id>/trajectories/session_NNN.jsonl`.
    #[serde(default = "default_output_dir")]
    pub output_dir: String,

    /// Include the assistant's `thinking` / reasoning block in the
    /// trajectory. `false` (default) keeps file size manageable;
    /// `true` is desirable for SFT training datasets.
    #[serde(default = "default_include_thinking")]
    pub include_thinking: bool,

    /// How to render tool-result payloads. See [`ToolResultMode`].
    #[serde(default)]
    pub include_tool_results: ToolResultMode,

    /// Operator-supplied additional redaction patterns. Each entry
    /// must be a valid Rust [`regex`] pattern. Compilation is
    /// validated at recorder construction.
    #[serde(default)]
    pub redact_extra_patterns: Vec<String>,
}

const fn default_enabled() -> bool {
    false
}

fn default_output_dir() -> String {
    "trajectories".to_owned()
}

const fn default_include_thinking() -> bool {
    false
}

impl Default for TrajectoryConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            output_dir: default_output_dir(),
            include_thinking: default_include_thinking(),
            include_tool_results: ToolResultMode::Redacted,
            redact_extra_patterns: Vec::new(),
        }
    }
}

impl TrajectoryConfig {
    /// Resolve the output directory for one run dir, returning the
    /// absolute path the recorder writes into. Relative `output_dir`
    /// values are joined onto `run_dir`; absolute values are honoured
    /// verbatim (rare but lets operators redirect to a network mount).
    #[must_use]
    pub fn resolve_output_dir(&self, run_dir: &std::path::Path) -> PathBuf {
        let p = PathBuf::from(&self.output_dir);
        if p.is_absolute() { p } else { run_dir.join(p) }
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_privacy_preserving() {
        let c = TrajectoryConfig::default();
        assert!(!c.enabled);
        assert_eq!(c.output_dir, "trajectories");
        assert!(!c.include_thinking);
        assert_eq!(c.include_tool_results, ToolResultMode::Redacted);
        assert!(c.redact_extra_patterns.is_empty());
    }

    #[test]
    fn parses_full_toml_block() {
        let toml_src = r#"
            enabled = true
            output_dir = "/var/log/tsumugi/traj"
            include_thinking = true
            include_tool_results = "summary_only"
            redact_extra_patterns = ["A-[0-9]+"]
        "#;
        let parsed: TrajectoryConfig =
            toml::from_str(toml_src).unwrap_or_else(|e| panic!("parse: {e}"));
        assert!(parsed.enabled);
        assert_eq!(parsed.output_dir, "/var/log/tsumugi/traj");
        assert!(parsed.include_thinking);
        assert_eq!(parsed.include_tool_results, ToolResultMode::SummaryOnly);
        assert_eq!(parsed.redact_extra_patterns, vec!["A-[0-9]+"]);
    }

    #[test]
    fn resolve_output_dir_handles_relative_and_absolute() {
        let mut c = TrajectoryConfig::default();
        let run_dir = std::path::Path::new("/runs/abc");
        assert_eq!(
            c.resolve_output_dir(run_dir),
            PathBuf::from("/runs/abc/trajectories")
        );
        c.output_dir = "/elsewhere".into();
        assert_eq!(c.resolve_output_dir(run_dir), PathBuf::from("/elsewhere"));
    }
}
