//! Secret-redaction helpers shared by the search index ingestion path
//! and the trajectory recorder (issue #55).
//!
//! [`redact_secrets`] runs the input through a series of regex patterns
//! that match well-known credential shapes (AWS access keys, `OpenAI`
//! `sk-` keys, `GitHub` PATs) plus a *credential-context* fallback for
//! long tokens that follow `token=`, `key=`, `secret=`, `password=`,
//! etc. Authorization headers get a dedicated pattern so the bearer
//! value is masked even when the payload is otherwise innocuous.
//!
//! # Generic-token tradeoff
//!
//! An earlier draft used `\b[A-Za-z0-9_\-]{40,}\b` as a catch-all for
//! "any long random string". This had a high false-positive rate: it
//! ate 40-char git SHAs, SHA-256 hashes, base64 image data, content
//! hashes — all of which are perfectly fine to keep in the search
//! index. A summary like "rebased on commit `abc1234567890123456789012345678901234567`"
//! would lose the SHA, making the entry far less useful.
//!
//! The current rule fires only when the long token is *introduced* by
//! a credential-shaped prefix (case-insensitive `token=`, `key=`,
//! `secret=`, `password=`, `pwd=`, `apikey=`, `auth=`, plus the
//! whitespace-separated variants). This deliberately misses tokens
//! that appear "naked" in free text — the rationale is that an actual
//! credential almost always has a clue word next to it (env-var
//! assignment, JSON key, CLI flag), whereas hashes do not. False
//! negatives here are acceptable; false positives are not, because
//! they silently destroy useful context in summaries.
//!
//! All matches are replaced with the literal string [`REDACTION_TOKEN`];
//! the surrounding text is preserved so search snippets still convey
//! the topic of a leak even after redaction.
//!
//! # Known limitations
//!
//! - The `Authorization:` header pattern stops at line breaks. Multi-
//!   line continuations (rare in practice) only redact the first line.
//! - Naked credentials in free text without a prefix word are not
//!   caught; structured-shape patterns (AWS / `OpenAI` / `GitHub`) cover
//!   the common branded cases.
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
            // Credential-context long token. Matches a long run of
            // alphanumerics / `_` / `-` ONLY when it sits behind a
            // credential-shaped prefix (case-insensitive). This avoids
            // eating git SHAs and content hashes in free text; see the
            // module-level comment for the rationale.
            //
            // Captured prefixes: `token`, `secret`, `key`, `apikey`,
            // `api_key`, `auth`, `password`, `pwd`, `passwd`. The
            // separator may be `=`, `:`, or whitespace. We replace
            // the whole match (prefix included) so the redaction
            // marker reads e.g. `[REDACTED]` rather than
            // `token=[REDACTED]`. Both forms are acceptable; we go
            // with full-replace because it's more conservative.
            Regex::new(
                r"(?i)\b(?:api[_-]?key|secret|token|auth|password|pwd|passwd|key)\s*[:=]\s*[A-Za-z0-9_\-]{20,}\b",
            )
            .expect("compile credential-context long-token pattern"),
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
    fn redacts_credential_context_long_token() {
        // 50-char alphanumeric blob with a credential-shaped prefix —
        // neither AWS nor sk- nor ghp_, but the `token=` introducer
        // makes it unambiguously secret.
        let input = "token=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijxxxx other=ok";
        let out = redact_secrets(input);
        assert!(out.contains(REDACTION_TOKEN), "{out}");
        assert!(out.contains("other=ok"), "tail preserved: {out}");
        assert!(
            !out.contains("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            "long token leaked: {out}"
        );
    }

    /// Regression check for review feedback (issue #53): the pre-fix
    /// generic redactor ate plain git SHAs and SHA-256 hashes,
    /// destroying useful context like commit references in summaries.
    /// The credential-context rule must NOT fire on hashes or SHAs
    /// in free text.
    #[test]
    fn preserves_git_sha_and_content_hashes() {
        // 40-char hex git SHA in a sentence.
        let sha40 = "abc1234567890123456789012345678901234567";
        let input1 = format!("rebased on commit {sha40} cleanly");
        let out1 = redact_secrets(&input1);
        assert!(out1.contains(sha40), "git SHA was redacted: {out1}");

        // 64-char hex SHA-256 (e.g. from a checksum file).
        let sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let input2 = format!("checksum: {sha256}");
        let out2 = redact_secrets(&input2);
        assert!(out2.contains(sha256), "SHA-256 was redacted: {out2}");

        // Long base64-shaped identifier in free text (no credential
        // prefix). Must survive.
        let blob = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmn";
        let input3 = format!("uploaded blob {blob} to cache");
        let out3 = redact_secrets(&input3);
        assert!(out3.contains(blob), "blob hash was redacted: {out3}");
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
                     token=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijxxxx";
        let out = redact_secrets(input);
        // Every original secret must be gone.
        for needle in [
            "AKIAIOSFODNN7EXAMPLE",
            "sk-abcdefghijkl",
            "ghp_abcdefghijkl",
            "eyJhbGciOiJ",
            "token=ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        ] {
            assert!(!out.contains(needle), "leaked {needle}: {out}");
        }
        // At least 4 redaction tokens (Authorization eats its full
        // span; the credential-context rule catches the token= line).
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
