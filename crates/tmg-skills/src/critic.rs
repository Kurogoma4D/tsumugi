//! `skill_critic` subagent plumbing.
//!
//! This module owns:
//!
//! - The strict JSON verdict schema (`{ "create": bool, "name": ...,
//!   "description": ..., "draft": ..., "reason": ... }`) the critic is
//!   required to emit.
//! - The system prompt the harness feeds into the critic agent's first
//!   message.
//! - A small parser ([`parse_verdict`]) that tolerates a single
//!   surrounding code fence and rejects anything else.
//!
//! The actual subagent type lives in `tmg-agents` as
//! `AgentType::SkillCritic`. The runtime spawn loop calls
//! [`parse_verdict`] on whatever the critic returns and acts on the
//! verdict.

use serde::{Deserialize, Serialize};

/// The system prompt baked into [`SkillCriticConfig::system_prompt`].
///
/// Lifted from the issue body almost verbatim. The closing JSON-only
/// directive is critical: local LLMs love to wrap responses in prose,
/// and the parser will refuse anything but a single JSON object.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You evaluate whether a recently completed sequence of actions in this project should be \
     saved as a reusable Skill.\n\n\
     CREATE if:\n\
     - The procedure is non-trivial (involves judgment, error handling, or multi-tool \
     orchestration)\n\
     - It is likely to be repeated in this codebase (build / deploy / migration / review \
     patterns)\n\
     - The procedure isn't already covered by an existing skill\n\n\
     DO NOT CREATE if:\n\
     - It was a one-off bug fix\n\
     - It's already in MEMORY.md or an existing skill\n\
     - The user expressed no preference for repeating this approach\n\n\
     Output JSON only. No prose, no code fences, no comments. Schema:\n\
     { \"create\": bool, \"name\": string?, \"description\": string?, \"draft\": string?, \
     \"reason\": string }\n\n\
     - `create` is REQUIRED.\n\
     - `reason` is REQUIRED, one or two sentences, no newlines inside.\n\
     - When `create` is true, `name`, `description`, and `draft` are REQUIRED. `name` must \
     be a kebab-case identifier safe for use as a directory name (no slashes, no dots). \
     `description` is one line. `draft` is the full SKILL.md body that would be saved.\n\
     - When `create` is false, OMIT `name`, `description`, and `draft` entirely (do not send \
     null or empty strings).";

/// Configuration for the `skill_critic` subagent, mirroring
/// `[harness.escalator]` in shape (issue #54). Loaded from
/// `[skills.critic]` in `tsumugi.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillCriticConfig {
    /// Whether the critic is enabled at all. Defaults to `true`.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Endpoint URL override (e.g. `http://localhost:8081`). Empty
    /// string means "use the main / escalator endpoint".
    #[serde(default)]
    pub endpoint: String,

    /// Model name override. Empty string means "use the main /
    /// escalator model".
    #[serde(default)]
    pub model: String,

    /// Maximum number of auto-generated skills per session. Defaults
    /// to 1 per the issue's hard-limit guidance.
    #[serde(default = "default_max_per_session")]
    pub max_per_session: u32,
}

const fn default_true() -> bool {
    true
}

const fn default_max_per_session() -> u32 {
    1
}

impl Default for SkillCriticConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: String::new(),
            model: String::new(),
            max_per_session: default_max_per_session(),
        }
    }
}

impl SkillCriticConfig {
    /// Return the system prompt the harness should feed into the
    /// critic. Currently a const; configurable overrides are reserved
    /// for a follow-up.
    #[must_use]
    pub fn system_prompt(&self) -> &'static str {
        DEFAULT_SYSTEM_PROMPT
    }
}

/// Strictly parsed verdict returned by the critic.
///
/// `name`, `description`, and `draft` are tied: when `create` is true
/// all three MUST be `Some`; when `create` is false they MUST all be
/// `None`. [`parse_verdict`] enforces this invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillCriticVerdict {
    /// Whether to actually create the skill.
    pub create: bool,
    /// Proposed skill name (only when `create == true`).
    pub name: Option<String>,
    /// Proposed one-line description.
    pub description: Option<String>,
    /// Proposed SKILL.md body.
    pub draft: Option<String>,
    /// Justification for the create/skip decision.
    pub reason: String,
}

/// Errors surfaced by [`parse_verdict`].
#[derive(Debug, thiserror::Error)]
pub enum CriticParseError {
    /// The input wasn't JSON at all (or was wrapped in something we
    /// couldn't strip).
    #[error("critic output is not valid JSON: {0}")]
    NotJson(String),

    /// The JSON parsed but a required field was missing or had the
    /// wrong type.
    #[error("invalid critic verdict: {0}")]
    Invalid(String),
}

/// Parse a critic verdict from raw text.
///
/// Tolerates exactly **one** surrounding fenced block (` ```json ... ```
/// `or `` ``` ... ``` ``) because local models occasionally insist on
/// wrapping JSON despite the prompt. Anything else (prose before/after,
/// multiple objects, missing fields) is a hard error.
///
/// # Errors
///
/// Returns [`CriticParseError`] when parsing or validation fails.
pub fn parse_verdict(text: &str) -> Result<SkillCriticVerdict, CriticParseError> {
    let stripped = strip_fence(text.trim());

    let value: serde_json::Value =
        serde_json::from_str(stripped).map_err(|e| CriticParseError::NotJson(e.to_string()))?;
    let obj = value.as_object().ok_or_else(|| {
        CriticParseError::Invalid("top-level value must be a JSON object".to_owned())
    })?;

    let create = obj
        .get("create")
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| {
            CriticParseError::Invalid("missing required boolean field: create".to_owned())
        })?;
    let reason = obj
        .get("reason")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            CriticParseError::Invalid("missing required string field: reason".to_owned())
        })?
        .to_owned();

    let name = obj
        .get("name")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let description = obj
        .get("description")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let draft = obj
        .get("draft")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);

    if create {
        if name.is_none() || description.is_none() || draft.is_none() {
            return Err(CriticParseError::Invalid(
                "create=true requires non-null `name`, `description`, and `draft`".to_owned(),
            ));
        }
    } else if name.is_some() || description.is_some() || draft.is_some() {
        return Err(CriticParseError::Invalid(
            "create=false must omit `name`, `description`, and `draft`".to_owned(),
        ));
    }

    Ok(SkillCriticVerdict {
        create,
        name,
        description,
        draft,
        reason,
    })
}

/// Strip a single optional ```` ``` ```` fence (with optional `json`
/// hint) around the input.
fn strip_fence(input: &str) -> &str {
    let candidate = input.trim();
    let Some(rest) = candidate.strip_prefix("```") else {
        return candidate;
    };
    // Drop the optional language hint (e.g. `json\n...`).
    let after_hint = rest.split_once('\n').map_or(rest, |(_, body)| body);
    if let Some(end) = after_hint.rfind("```") {
        after_hint[..end].trim()
    } else {
        candidate
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests assert with unwrap for clarity; the workspace policy denies them in production code"
)]
mod tests {
    use super::*;

    #[test]
    fn parse_create_true_full_verdict() {
        let json = r#"{
            "create": true,
            "name": "deploy-rust",
            "description": "Publish a Rust crate",
            "draft": "---\nname: deploy-rust\n...\n",
            "reason": "Multi-step deploy procedure with retries."
        }"#;
        let v = parse_verdict(json).unwrap();
        assert!(v.create);
        assert_eq!(v.name.as_deref(), Some("deploy-rust"));
        assert_eq!(v.description.as_deref(), Some("Publish a Rust crate"));
        assert!(v.draft.as_ref().unwrap().contains("name: deploy-rust"));
        assert!(v.reason.starts_with("Multi-step"));
    }

    #[test]
    fn parse_create_false_skips_creation() {
        let json = r#"{ "create": false, "reason": "One-off fix." }"#;
        let v = parse_verdict(json).unwrap();
        assert!(!v.create);
        assert!(v.name.is_none());
        assert!(v.description.is_none());
        assert!(v.draft.is_none());
    }

    #[test]
    fn parse_create_false_with_payload_rejected() {
        let json = r#"{
            "create": false,
            "reason": "no",
            "name": "extra"
        }"#;
        let err = parse_verdict(json).unwrap_err();
        assert!(matches!(err, CriticParseError::Invalid(_)));
    }

    #[test]
    fn parse_create_true_missing_fields_rejected() {
        let json = r#"{ "create": true, "reason": "yes" }"#;
        let err = parse_verdict(json).unwrap_err();
        assert!(matches!(err, CriticParseError::Invalid(_)));
    }

    #[test]
    fn parse_with_fence() {
        let fenced = "```json\n{\n  \"create\": false,\n  \"reason\": \"meh\"\n}\n```";
        let v = parse_verdict(fenced).unwrap();
        assert!(!v.create);
    }

    #[test]
    fn parse_garbage_rejected() {
        let err = parse_verdict("not json at all").unwrap_err();
        assert!(matches!(err, CriticParseError::NotJson(_)));
    }

    #[test]
    fn config_defaults() {
        let c = SkillCriticConfig::default();
        assert!(c.enabled);
        assert_eq!(c.max_per_session, 1);
        assert!(c.endpoint.is_empty());
        assert!(c.model.is_empty());
    }

    #[test]
    fn config_deserializes_full() {
        let toml = r#"
enabled = true
endpoint = "http://localhost:8081"
model = "qwen-coder-7b"
max_per_session = 2
"#;
        let c: SkillCriticConfig = toml::from_str(toml).unwrap();
        assert_eq!(c.endpoint, "http://localhost:8081");
        assert_eq!(c.model, "qwen-coder-7b");
        assert_eq!(c.max_per_session, 2);
    }

    #[test]
    fn config_defaults_when_keys_missing() {
        let c: SkillCriticConfig = toml::from_str("").unwrap();
        assert!(c.enabled);
        assert_eq!(c.max_per_session, 1);
    }
}
