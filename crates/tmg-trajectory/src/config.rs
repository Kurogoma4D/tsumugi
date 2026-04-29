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

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::TrajectoryError;

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
    /// absolute path the recorder writes into.
    ///
    /// Relative `output_dir` values are joined onto `run_dir`;
    /// absolute values are honoured verbatim (rare but lets operators
    /// redirect to a network mount).
    ///
    /// **No sandbox check is performed.** Callers that need to enforce
    /// "trajectories live under the project root" must use
    /// [`Self::resolve_output_dir_within`] — see its rustdoc for the
    /// contract.
    #[must_use]
    pub fn resolve_output_dir(&self, run_dir: &Path) -> PathBuf {
        let p = PathBuf::from(&self.output_dir);
        if p.is_absolute() { p } else { run_dir.join(p) }
    }

    /// Resolve the output directory like [`Self::resolve_output_dir`]
    /// but reject paths that escape the supplied `project_root`.
    ///
    /// # Contract
    ///
    /// On success the returned path is guaranteed to be a descendant
    /// (or equal to) the canonicalised `project_root`. Symlinks in the
    /// configured `output_dir` are resolved before comparison so a
    /// user cannot escape the project root via `..` or via a symlink
    /// pointing outside the tree.
    ///
    /// This is the recommended entry point for the live recorder —
    /// the live agent loop already operates inside a sandbox bound to
    /// the project root, so an absolute `output_dir` pointing
    /// elsewhere should be rejected at config-load time rather than
    /// silently falling outside Landlock's reach on macOS or other
    /// non-Linux platforms.
    ///
    /// # Errors
    ///
    /// Returns [`TrajectoryError::InvalidConfig`] when:
    /// - `project_root` does not exist or cannot be canonicalised, or
    /// - the resolved output dir, after best-effort canonicalisation,
    ///   does not fall under the canonical `project_root`.
    pub fn resolve_output_dir_within(
        &self,
        project_root: &Path,
        run_dir: &Path,
    ) -> Result<PathBuf, TrajectoryError> {
        let resolved = self.resolve_output_dir(run_dir);
        let canonical_root = project_root.canonicalize().map_err(|e| {
            TrajectoryError::InvalidConfig(format!(
                "canonicalising project root {}: {e}",
                project_root.display(),
            ))
        })?;
        // The resolved dir does not necessarily exist yet; canonicalise
        // the deepest existing ancestor and rejoin the tail. This
        // matches what `std::fs::canonicalize` would have produced if
        // the directory existed.
        let canonical_resolved = canonicalise_or_existing_ancestor(&resolved);
        if canonical_resolved.starts_with(&canonical_root) {
            Ok(resolved)
        } else {
            Err(TrajectoryError::InvalidConfig(format!(
                "trajectory output_dir {} resolves outside project root {}",
                resolved.display(),
                canonical_root.display(),
            )))
        }
    }
}

/// Canonicalise `path`, falling back to canonicalising the deepest
/// ancestor that actually exists and re-joining the unresolved tail.
/// Used by [`TrajectoryConfig::resolve_output_dir_within`] so the
/// check works even before the trajectory directory has been created.
fn canonicalise_or_existing_ancestor(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    let mut existing: Option<&Path> = None;
    let mut ancestor = path.parent();
    while let Some(p) = ancestor {
        if p.exists() {
            existing = Some(p);
            break;
        }
        ancestor = p.parent();
    }
    let Some(existing) = existing else {
        return path.to_path_buf();
    };
    let Ok(canonical_existing) = existing.canonicalize() else {
        return path.to_path_buf();
    };
    let Ok(tail) = path.strip_prefix(existing) else {
        return path.to_path_buf();
    };
    canonical_existing.join(tail)
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

    /// Sandbox check — a relative `output_dir` under the project root
    /// resolves successfully.
    #[test]
    fn resolve_output_dir_within_accepts_in_root() {
        let tmp = tempfile::TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let project_root = tmp.path();
        let run_dir = project_root.join(".tsumugi/runs/abc");
        std::fs::create_dir_all(&run_dir).unwrap_or_else(|e| panic!("{e}"));
        let c = TrajectoryConfig::default();
        let out = c
            .resolve_output_dir_within(project_root, &run_dir)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(out.ends_with("trajectories"), "{}", out.display());
    }

    /// Sandbox check — an absolute `output_dir` pointing outside the
    /// project root is rejected with [`TrajectoryError::InvalidConfig`].
    #[test]
    fn resolve_output_dir_within_rejects_escape() {
        let tmp = tempfile::TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let project_root = tmp.path();
        let run_dir = project_root.join(".tsumugi/runs/abc");
        std::fs::create_dir_all(&run_dir).unwrap_or_else(|e| panic!("{e}"));
        // A separate tempdir that is *not* under the project root.
        let escape = tempfile::TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let c = TrajectoryConfig {
            output_dir: escape.path().to_string_lossy().into_owned(),
            ..TrajectoryConfig::default()
        };
        let result = c.resolve_output_dir_within(project_root, &run_dir);
        match result {
            Err(TrajectoryError::InvalidConfig(msg)) => {
                assert!(
                    msg.contains("outside project root"),
                    "unexpected message: {msg}"
                );
            }
            Ok(p) => panic!("expected error, got Ok({})", p.display()),
            Err(other) => panic!("expected InvalidConfig, got {other:?}"),
        }
    }
}
