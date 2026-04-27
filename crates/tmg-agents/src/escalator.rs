//! Escalator subagent verdict types and JSON parsing.
//!
//! The escalator subagent (SPEC §5.2 / §9.3) produces a strict JSON
//! verdict that decides whether the current task should be promoted
//! from an ad-hoc run to a fully harnessed run. Its output schema is
//!
//! ```json
//! {
//!   "escalate": true,
//!   "reason": "spans tmg-llm and tmg-tui; needs features.json",
//!   "estimated_features": 36
//! }
//! ```
//!
//! `estimated_features` is OPTIONAL and is only meaningful when
//! `escalate` is `true`; emitting it as `null`, `0`, or a string is
//! rejected so that downstream automation does not have to guess what
//! the model meant.
//!
//! [`parse_verdict`] is the only sanctioned entry point for accepting
//! escalator output. It uses `#[serde(deny_unknown_fields)]` and an
//! explicit "no trailing input" check so a malformed model response is
//! surfaced loudly rather than silently routed to the wrong control
//! flow.

use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Deserialize `estimated_features` as a strict non-negative integer
/// when present, rejecting explicit `null` (the system prompt forbids
/// it -- see [`crate::config::AgentType::Escalator::system_prompt`]).
///
/// Used in combination with `#[serde(default)]` on the field so that
/// "field omitted" still maps to `None`. Without this helper, serde's
/// default `Option<u32>` adapter would silently accept `null` and
/// `None` interchangeably, masking model bugs that emit `null`
/// instead of omitting the key entirely.
fn deserialize_estimated_features<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    // Deserialize as a plain `u32`. If serde encounters `null` it will
    // raise an `invalid type: null` error, which is exactly what we
    // want -- it gets surfaced as `ParseError::Json` to the caller.
    let value = u32::deserialize(deserializer)?;
    Ok(Some(value))
}

/// Strict verdict emitted by the escalator subagent.
///
/// Constructed exclusively via [`parse_verdict`]; the struct itself is
/// `pub` so downstream automation can inspect the fields, but the
/// public constructor surface goes through the parser to keep the
/// validation rules in one place.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EscalatorVerdict {
    /// `true` when the escalator recommends promoting the request to a
    /// harnessed run; `false` to keep the ad-hoc scope.
    pub escalate: bool,

    /// Concise human-readable justification (one or two sentences,
    /// no newlines) explaining the decision.
    pub reason: String,

    /// Optional integer estimate of how many entries `features.json`
    /// will eventually need (target 30-50 per the initializer spec).
    ///
    /// Omitted entirely when the escalator cannot estimate or when
    /// `escalate` is `false`. Encoded as `Option<u32>` and skipped on
    /// serialization so the round-trip preserves the "field omitted"
    /// vs. "field present with zero" distinction the schema requires.
    ///
    /// `deserialize_with` rejects explicit `null`; only "field
    /// omitted" maps to `None`. Strings, floats, and negative numbers
    /// are also rejected.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_estimated_features"
    )]
    pub estimated_features: Option<u32>,
}

/// Errors produced when parsing an escalator response.
///
/// Distinct variants make it easier for callers (and tests) to assert
/// on the failure mode without string-matching.
#[derive(Debug, Error)]
pub enum ParseError {
    /// The input was not valid JSON, was missing a required field, or
    /// contained an unknown / mistyped field. Wraps the underlying
    /// `serde_json` error for diagnostics.
    #[error("invalid escalator verdict JSON: {0}")]
    Json(#[from] serde_json::Error),

    /// The JSON parsed cleanly but extra non-whitespace input followed
    /// the verdict object. The escalator system prompt forbids this
    /// (no prose, no code fences, no trailing text), and silently
    /// dropping the trailing bytes risks masking a malformed response.
    #[error("trailing input after escalator verdict: {trailing:?}")]
    TrailingInput {
        /// The bytes that followed the parsed JSON object, lossily
        /// converted to a UTF-8 string for error display.
        trailing: String,
    },
}

/// Parse a JSON string into an [`EscalatorVerdict`].
///
/// Strict rules:
///
/// - Required fields: `escalate` (bool) and `reason` (string).
/// - Optional field: `estimated_features` (non-negative integer, fits
///   in `u32`).
/// - Unknown fields are rejected by `#[serde(deny_unknown_fields)]`.
/// - Trailing non-whitespace input is rejected via [`ParseError::TrailingInput`].
///
/// # Errors
///
/// Returns [`ParseError::Json`] for malformed input or schema
/// violations, and [`ParseError::TrailingInput`] when valid JSON is
/// followed by extra content.
pub fn parse_verdict(json: &str) -> Result<EscalatorVerdict, ParseError> {
    // Use a streaming deserializer so we can detect trailing
    // non-whitespace input without re-scanning the source. Calling
    // `serde_json::from_str` would silently accept `{"escalate":true,
    // "reason":"x"} garbage` because the default parser stops at the
    // closing brace.
    let mut stream = serde_json::Deserializer::from_str(json).into_iter::<EscalatorVerdict>();
    let verdict = stream.next().ok_or_else(|| {
        // Synthesize a Json error so the caller does not need to
        // distinguish "empty input" from "malformed JSON" — both
        // are genuine parse failures.
        let synthetic = serde_json::from_str::<EscalatorVerdict>("")
            .err()
            .unwrap_or_else(|| {
                // serde_json should always fail on empty input;
                // keep a defensive fallback so the helper stays
                // total.
                serde_json::Error::io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "empty escalator verdict",
                ))
            });
        ParseError::Json(synthetic)
    })??;

    let consumed = stream.byte_offset();
    let remainder = json[consumed..].trim_start();
    if !remainder.is_empty() {
        return Err(ParseError::TrailingInput {
            trailing: remainder.to_owned(),
        });
    }

    Ok(verdict)
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions")]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;

    #[test]
    fn parses_escalate_true_with_features() {
        let json = r#"{
            "escalate": true,
            "reason": "spans tmg-llm and tmg-tui; needs features.json",
            "estimated_features": 36
        }"#;
        let v = parse_verdict(json).unwrap_or_else(|e| panic!("{e}"));
        assert!(v.escalate);
        assert_eq!(v.reason, "spans tmg-llm and tmg-tui; needs features.json");
        assert_eq!(v.estimated_features, Some(36));
    }

    #[test]
    fn parses_escalate_false_without_features() {
        let json = r#"{
            "escalate": false,
            "reason": "single-file fix; no features needed"
        }"#;
        let v = parse_verdict(json).unwrap_or_else(|e| panic!("{e}"));
        assert!(!v.escalate);
        assert_eq!(v.reason, "single-file fix; no features needed");
        assert_eq!(v.estimated_features, None);
    }

    #[test]
    fn rejects_malformed_json() {
        let json = r#"{ "escalate": true, "reason": "missing brace" "#;
        let err = parse_verdict(json).expect_err("malformed JSON must fail");
        assert!(
            matches!(err, ParseError::Json(_)),
            "expected ParseError::Json, got {err:?}"
        );
    }

    #[test]
    fn rejects_unknown_field() {
        // SPEC §9.3 -- the schema is closed.
        let json = r#"{
            "escalate": true,
            "reason": "ok",
            "estimated_features": 5,
            "confidence": 0.9
        }"#;
        let err = parse_verdict(json).expect_err("unknown field must fail");
        assert!(
            matches!(err, ParseError::Json(_)),
            "expected ParseError::Json, got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("confidence") || msg.contains("unknown field"),
            "error must identify the unknown field; got: {msg}"
        );
    }

    #[test]
    fn rejects_missing_required_field_escalate() {
        let json = r#"{ "reason": "no escalate key" }"#;
        let err = parse_verdict(json).expect_err("missing escalate must fail");
        assert!(matches!(err, ParseError::Json(_)));
    }

    #[test]
    fn rejects_missing_required_field_reason() {
        let json = r#"{ "escalate": true }"#;
        let err = parse_verdict(json).expect_err("missing reason must fail");
        assert!(matches!(err, ParseError::Json(_)));
    }

    #[test]
    fn rejects_estimated_features_as_null() {
        // The system prompt forbids `null`; we want a parse failure
        // rather than silently treating null as "omitted", since that
        // would let a sloppy model send `{"estimated_features": null}`
        // in place of an actual estimate.
        let json = r#"{
            "escalate": true,
            "reason": "ok",
            "estimated_features": null
        }"#;
        let err = parse_verdict(json).expect_err("null estimate must fail");
        assert!(matches!(err, ParseError::Json(_)));
    }

    #[test]
    fn rejects_estimated_features_as_string() {
        let json = r#"{
            "escalate": true,
            "reason": "ok",
            "estimated_features": "many"
        }"#;
        let err = parse_verdict(json).expect_err("string estimate must fail");
        assert!(matches!(err, ParseError::Json(_)));
    }

    #[test]
    fn rejects_trailing_input() {
        let json = r#"{ "escalate": false, "reason": "x" } trailing prose"#;
        let err = parse_verdict(json).expect_err("trailing input must fail");
        assert!(
            matches!(err, ParseError::TrailingInput { .. }),
            "expected ParseError::TrailingInput, got {err:?}"
        );
    }

    #[test]
    fn rejects_empty_input() {
        let err = parse_verdict("").expect_err("empty input must fail");
        assert!(matches!(err, ParseError::Json(_)));
    }

    #[test]
    fn round_trip_omits_estimated_features_when_none() {
        let v = EscalatorVerdict {
            escalate: false,
            reason: "no need".to_owned(),
            estimated_features: None,
        };
        let s = serde_json::to_string(&v).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            !s.contains("estimated_features"),
            "field must be skipped when None, got: {s}"
        );
        let parsed = parse_verdict(&s).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(parsed, v);
    }

    #[test]
    fn round_trip_includes_estimated_features_when_some() {
        let v = EscalatorVerdict {
            escalate: true,
            reason: "spans crates".to_owned(),
            estimated_features: Some(42),
        };
        let s = serde_json::to_string(&v).unwrap_or_else(|e| panic!("{e}"));
        assert!(s.contains("\"estimated_features\":42"));
        let parsed = parse_verdict(&s).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(parsed, v);
    }
}
