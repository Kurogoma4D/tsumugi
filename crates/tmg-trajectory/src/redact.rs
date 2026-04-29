//! Trajectory-side redaction.
//!
//! The credential patterns themselves live in [`tmg_search::redact`]
//! (issue #53); this module re-exports them so we never duplicate the
//! regex set. The trajectory layer adds two pieces of behaviour on
//! top:
//!
//! 1. A [`Redactor`] type that combines the shared base patterns with
//!    *extra* patterns supplied by the operator via
//!    `[trajectory] redact_extra_patterns`.
//! 2. A short module re-export so call sites can write
//!    `tmg_trajectory::redact::redact_secrets` and stay symmetric with
//!    `tmg_search`.
//!
//! ## Why not move the regex set into a third crate?
//!
//! The base patterns are tightly coupled to the search-index ingestion
//! tests in `tmg-search` (the false-positive suite for git SHAs, etc.).
//! Carving them into a fourth crate just for sharing would force a
//! leaf crate that owns nothing but `regex`-compile cost; re-exporting
//! the existing helper from `tmg-search` is the minimal change. If a
//! third consumer ever appears we can promote the patterns then.
//!
//! Both crates compile the regex set lazily inside their respective
//! `OnceLock` cells, so loading both crates only pays the compile cost
//! once for the base set (the `tmg-search` static is shared) and once
//! for any extra patterns the trajectory operator added.

use regex::Regex;

pub use tmg_search::redact::{REDACTION_TOKEN, redact_secrets};

use crate::error::TrajectoryError;

/// A composed redactor that applies the shared `tmg-search` regex set
/// plus any operator-supplied extras.
///
/// Construction validates every extra pattern; an invalid regex is
/// surfaced as [`TrajectoryError::InvalidRedactionPattern`] so the
/// recorder never has to silently drop a misconfigured rule.
#[derive(Debug, Clone)]
pub struct Redactor {
    extras: Vec<Regex>,
}

impl Redactor {
    /// Build a redactor with no operator-supplied extras. Equivalent to
    /// calling [`redact_secrets`] directly, but cheap to clone and
    /// pass around.
    #[must_use]
    pub fn new() -> Self {
        Self { extras: Vec::new() }
    }

    /// Build a redactor with operator-supplied extra patterns.
    ///
    /// # Errors
    ///
    /// Returns [`TrajectoryError::InvalidRedactionPattern`] when any
    /// pattern fails to compile.
    pub fn with_extras<I, S>(extras: I) -> Result<Self, TrajectoryError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut compiled = Vec::new();
        for pat in extras {
            let raw = pat.as_ref();
            let re =
                Regex::new(raw).map_err(|source| TrajectoryError::InvalidRedactionPattern {
                    pattern: raw.to_owned(),
                    source,
                })?;
            compiled.push(re);
        }
        Ok(Self { extras: compiled })
    }

    /// Redact `text` by applying the shared base patterns first, then
    /// each extra pattern in order.
    #[must_use]
    pub fn apply(&self, text: &str) -> String {
        let mut current = redact_secrets(text);
        for re in &self.extras {
            current = re.replace_all(&current, REDACTION_TOKEN).into_owned();
        }
        current
    }
}

impl Default for Redactor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;

    #[test]
    fn default_redactor_runs_shared_patterns() {
        let r = Redactor::new();
        let out = r.apply("token=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijxxxx tail");
        assert!(out.contains(REDACTION_TOKEN), "{out}");
        assert!(out.contains("tail"), "{out}");
    }

    #[test]
    fn extra_patterns_apply() {
        let r = Redactor::with_extras(["INTERNAL-[A-Z0-9]{8}"]).unwrap_or_else(|e| panic!("{e}"));
        let out = r.apply("see INTERNAL-ABCD1234 for details");
        assert!(!out.contains("INTERNAL-ABCD1234"), "{out}");
        assert!(out.contains(REDACTION_TOKEN), "{out}");
        assert!(out.contains("for details"), "{out}");
    }

    #[test]
    fn invalid_pattern_surfaced() {
        let r = Redactor::with_extras(["[unterminated"]);
        match r {
            Err(TrajectoryError::InvalidRedactionPattern { pattern, .. }) => {
                assert_eq!(pattern, "[unterminated");
            }
            other => panic!("expected InvalidRedactionPattern, got {other:?}"),
        }
    }

    /// Five-pattern smoke test mirroring `tmg-search`'s acceptance
    /// suite: every shared redactor must fire on the trajectory side.
    #[test]
    fn five_pattern_acceptance_via_trajectory() {
        let r = Redactor::new();
        let input = "AKIAIOSFODNN7EXAMPLE \
                     sk-abcdefghijklmnopqrstuvwxyz1234567890ABCD \
                     ghp_abcdefghijklmnopqrstuvwxyz0123456789AB \
                     Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 \
                     token=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijxxxx";
        let out = r.apply(input);
        for needle in [
            "AKIAIOSFODNN7EXAMPLE",
            "sk-abcdefghijkl",
            "ghp_abcdefghijkl",
            "eyJhbGciOiJ",
            "token=ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        ] {
            assert!(!out.contains(needle), "leaked {needle}: {out}");
        }
    }
}
