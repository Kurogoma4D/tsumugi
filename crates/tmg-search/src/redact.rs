//! Secret-redaction helpers shared by the search index ingestion path
//! and the trajectory recorder (issue #55).
//!
//! [`redact_secrets`] runs the input through a series of regex patterns
//! that match well-known credential shapes (AWS access keys, `OpenAI`
//! `sk-` keys, `GitHub` PATs) plus a generic high-entropy fallback for
//! tokens that don't fit any specific format. Authorization headers
//! get a dedicated pattern so the bearer value is masked even when the
//! payload is otherwise innocuous.
//!
//! All matches are replaced with the literal string [`REDACTION_TOKEN`];
//! the surrounding text is preserved so search snippets still convey
//! the topic of a leak even after redaction.
//!
//! # Determinism
//!
//! The compiled regex set is cached behind [`std::sync::OnceLock`] so
//! repeated calls share the same compiled state. [`redact_secrets`] is
//! pure: same input → same output.

use std::sync::OnceLock;

use regex::Regex;

/// Replacement token that takes the place of every redacted match.
///
/// Chosen to be eye-catching but parseable as ASCII so search hits
/// containing redacted text don't break on unicode segmentation.
pub const REDACTION_TOKEN: &str = "[REDACTED]";

/// Run the secret-shaped patterns from [`patterns()`] over `text` in
/// order and return a fresh `String` with every match replaced by
/// [`REDACTION_TOKEN`].
///
/// The function is `O(N * P)` in the size of `text` and the number of
/// patterns; the patterns themselves are cheap (no backtracking) so
/// the cost is dominated by the per-byte scan. Pre-allocating the
/// output buffer would be a micro-optimisation we leave to a follow-up.
#[must_use]
pub fn redact_secrets(text: &str) -> String {
    let mut current = text.to_owned();
    for pattern in patterns() {
        current = pattern.replace_all(&current, REDACTION_TOKEN).into_owned();
    }
    current
}

/// Compile the regex set on first call and cache it for subsequent
/// calls. Order matters: more specific patterns run first so a generic
/// high-entropy token detector cannot eat a structured AWS key before
/// the dedicated AWS pattern fires.
fn patterns() -> &'static [Regex] {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        // Each pattern is compiled once and stored in a `Vec<Regex>`.
        // `Regex::new` only fails on a malformed pattern string; the
        // strings here are constants under our control, so a panic
        // here would mean the source itself is broken — surfacing a
        // panic at first-use is acceptable for a static initialiser.
        // Use `expect` rather than `unwrap_or_else(panic!)` so the
        // failure includes a distinguishing message.
        #[expect(
            clippy::expect_used,
            reason = "static-initialiser regex compilation; failure is a compile-time-equivalent bug"
        )]
        let v = vec![
            // Authorization header values. Capture the prefix and
            // replace the entire header → value range so callers see
            // `Authorization: [REDACTED]` rather than a partial mask.
            Regex::new(r"(?i)Authorization:\s*[A-Za-z]+\s+[A-Za-z0-9._~+/=\-]+")
                .expect("compile Authorization header pattern"),
            // AWS access key id. Always 20 chars total: `AKIA` + 16.
            // Anchored on `\b` to avoid matching inside random hex
            // dumps that happen to contain `AKIA`.
            Regex::new(r"\bAKIA[A-Z0-9]{16,}\b").expect("compile AWS access key id pattern"),
            // OpenAI-style secret key (`sk-` prefix, ≥32 alnum chars).
            Regex::new(r"\bsk-[A-Za-z0-9]{32,}\b").expect("compile OpenAI sk- pattern"),
            // GitHub personal access token (`ghp_` prefix, 36 chars).
            Regex::new(r"\bghp_[A-Za-z0-9]{36,}\b").expect("compile GitHub PAT pattern"),
            // Generic high-entropy token. Matches a run of ≥40
            // alphanumerics / `_` / `-`. Runs LAST so the structured
            // patterns above win when their prefix is present.
            Regex::new(r"\b[A-Za-z0-9_\-]{40,}\b").expect("compile generic high-entropy pattern"),
        ];
        v
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_aws_access_key() {
        let input = "deploy uses AKIAIOSFODNN7EXAMPLE for prod";
        let out = redact_secrets(input);
        assert!(
            !out.contains("AKIAIOSFODNN7EXAMPLE"),
            "AWS key not redacted: {out}"
        );
        assert!(out.contains(REDACTION_TOKEN), "{out}");
        assert!(out.contains("deploy uses"));
    }

    #[test]
    fn redacts_openai_sk_key() {
        let input = "OPENAI_KEY=sk-abcdefghijklmnopqrstuvwxyz1234567890ABCD";
        let out = redact_secrets(input);
        assert!(!out.contains("sk-abcdefghijkl"), "{out}");
        assert!(out.contains(REDACTION_TOKEN), "{out}");
    }

    #[test]
    fn redacts_github_pat() {
        let input = "token: ghp_abcdefghijklmnopqrstuvwxyz0123456789AB extra";
        let out = redact_secrets(input);
        assert!(!out.contains("ghp_abcdefghijkl"), "{out}");
        assert!(out.contains(REDACTION_TOKEN), "{out}");
        assert!(out.contains("extra"), "tail preserved: {out}");
    }

    #[test]
    fn redacts_authorization_header() {
        let input = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9";
        let out = redact_secrets(input);
        assert!(!out.contains("eyJhbGciOiJ"), "{out}");
        assert!(out.contains(REDACTION_TOKEN), "{out}");
    }

    #[test]
    fn redacts_generic_high_entropy_token() {
        // 50-char alphanumeric blob — neither AWS nor sk- nor ghp_,
        // but still high-entropy enough that we treat it as secret.
        let input = "session=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijxxxx other=ok";
        let out = redact_secrets(input);
        assert!(out.contains(REDACTION_TOKEN), "{out}");
        assert!(out.contains("other=ok"), "tail preserved: {out}");
    }

    /// Five-pattern smoke test: one input, one output, every redactor
    /// fires at least once. Acceptance criterion in issue #53 calls for
    /// "5 unit test patterns"; this consolidates them so adding a sixth
    /// pattern only needs one line.
    #[test]
    fn five_pattern_acceptance_check() {
        let input = "AKIAIOSFODNN7EXAMPLE \
                     sk-abcdefghijklmnopqrstuvwxyz1234567890ABCD \
                     ghp_abcdefghijklmnopqrstuvwxyz0123456789AB \
                     Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 \
                     session=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijxxxx";
        let out = redact_secrets(input);
        // Every original secret must be gone.
        for needle in [
            "AKIAIOSFODNN7EXAMPLE",
            "sk-abcdefghijkl",
            "ghp_abcdefghijkl",
            "eyJhbGciOiJ",
            "session=ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        ] {
            assert!(!out.contains(needle), "leaked {needle}: {out}");
        }
        // At least 4 redaction tokens (Authorization eats its full
        // span, the generic catches both leftovers — count is at
        // least the number of distinct patterns).
        let tokens = out.matches(REDACTION_TOKEN).count();
        assert!(tokens >= 4, "expected ≥4 redactions, got {tokens}: {out}");
    }

    #[test]
    fn leaves_innocuous_text_untouched() {
        let input = "lorem ipsum dolor sit amet";
        assert_eq!(redact_secrets(input), input);
    }

    #[test]
    fn idempotent_on_already_redacted() {
        let once = redact_secrets("ghp_abcdefghijklmnopqrstuvwxyz0123456789AB");
        let twice = redact_secrets(&once);
        assert_eq!(once, twice);
    }
}
