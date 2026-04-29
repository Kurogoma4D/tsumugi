//! Serialisable trajectory record types.
//!
//! The on-disk format is JSONL (one record per line), and records are
//! tagged by a `type` field so external converters can stream them with
//! a single decode pass. The shape is deliberately a *superset* of the
//! `OpenAI` chat-completion conversation format: every record's
//! `type` field corresponds to a `role` (or to the auxiliary
//! `tool_result` shape), so a thin transform can produce
//! [Atropos](https://github.com/NousResearch/atropos) /
//! [TRL](https://github.com/huggingface/trl) compatible training
//! datasets without re-reading the source-of-truth log.
//!
//! # Type / role mapping
//!
//! | trajectory `type` | `OpenAI` chat `role` | Notes                                  |
//! |-------------------|----------------------|----------------------------------------|
//! | `system`          | `system`             | 1:1                                    |
//! | `user`            | `user`               | 1:1                                    |
//! | `assistant`       | `assistant`          | 1:1; carries optional `tool_calls`     |
//! | `tool_result`     | `tool`               | `OpenAI` uses the `tool` role here     |
//! | `meta`            | n/a                  | dataset metadata; outside the chat log |
//! | `feedback`        | n/a                  | RL signal; outside the chat log        |
//! | `verdict`         | n/a                  | session outcome; outside the chat log  |
//!
//! Converters that target `OpenAI` exactly should drop `meta`,
//! `feedback`, and `verdict` (or fold `feedback` into a separate
//! reward channel).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// One record line in a trajectory JSONL file.
///
/// `#[serde(tag = "type")]` writes a `type` field on the wire so the
/// JSONL is self-describing. `#[serde(rename_all = "snake_case")]`
/// keeps the discriminator values lower-case (`"meta"`, `"system"`,
/// ...) per the issue spec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TrajectoryRecord {
    /// Dataset metadata. Always the first record in a session file.
    Meta(MetaRecord),
    /// System prompt at the start of the conversation.
    System(SystemRecord),
    /// User-authored turn input.
    User(UserRecord),
    /// Assistant response (free text + optional reasoning + optional
    /// `tool_calls`).
    Assistant(AssistantRecord),
    /// Result of a single tool invocation, paired with the
    /// `tool_call_id` from the matching [`AssistantRecord`].
    ToolResult(ToolResultRecord),
    /// User intervention or correction signal (RL / SFT).
    Feedback(FeedbackRecord),
    /// Final session outcome label.
    Verdict(VerdictRecord),
}

/// Dataset metadata record (one per file, written first).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaRecord {
    /// Run id (8-char short form).
    pub run_id: String,
    /// 1-indexed session number within the run.
    pub session_num: u32,
    /// LLM model name as recorded in `tsumugi.toml`.
    pub model: String,
    /// Run scope label (`"ad-hoc"` or `"harnessed"`).
    pub scope: String,
    /// Session start timestamp (UTC).
    pub started_at: DateTime<Utc>,
    /// Session end timestamp; `None` while the session is still live.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<DateTime<Utc>>,
    /// Outcome label written when the session ends. Free-form short
    /// string (`"completed"`, `"errored"`, `"cancelled"`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
}

/// System-prompt record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemRecord {
    /// System prompt text. Redaction is applied before write.
    pub content: String,
}

/// User-turn record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserRecord {
    /// User message text. Redaction is applied before write.
    pub content: String,
}

/// Assistant-turn record.
///
/// `thinking` is gated by `[trajectory] include_thinking` (off by
/// default to keep file size manageable). `tool_calls` is the same
/// shape as the `OpenAI` tool-calling response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantRecord {
    /// Final visible assistant text (post tool-call extraction).
    pub content: String,
    /// Optional reasoning / chain-of-thought block, when
    /// `include_thinking = true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// Tool calls the assistant emitted in this turn (empty when none
    /// were issued).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallRecord>,
}

/// One tool call inside an [`AssistantRecord`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallRecord {
    /// Unique LLM-issued call identifier; pairs with the
    /// [`ToolResultRecord::tool_call_id`].
    pub id: String,
    /// Function/tool name.
    pub name: String,
    /// Arguments JSON object (already parsed when possible, otherwise
    /// the raw string verbatim — preserves out-of-spec emissions for
    /// later inspection).
    pub arguments: serde_json::Value,
}

/// Tool result record. Pairs with the matching
/// [`ToolCallRecord::id`] via `tool_call_id`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultRecord {
    /// LLM-issued call id; matches the corresponding
    /// [`ToolCallRecord::id`].
    pub tool_call_id: String,
    /// Tool name (duplicated for ergonomics; some external converters
    /// look here rather than chasing the `tool_call_id` back to the
    /// assistant record).
    pub tool_name: String,
    /// Result body. Truncated, redacted, or summarised based on
    /// `[trajectory] include_tool_results`.
    pub output: String,
    /// Whether the tool reported an error.
    pub is_error: bool,
}

/// User-intervention record (`/run abort`, "違う", "yes 続けて", ...).
///
/// These records are the analogue of the RL reward signal Hermes
/// carries to Atropos; tsumugi only tags them, the actual reward
/// shaping lives in the consumer pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedbackRecord {
    /// Free-form short label identifying the source of the feedback.
    /// Issue #55 recommends `"user_correction"`, `"abort"`,
    /// `"approval"`. The string is stored verbatim so callers can
    /// extend the vocabulary without a schema change.
    pub source: String,
    /// Free-form payload. Redaction is applied before write.
    pub content: String,
}

/// Final session outcome record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerdictRecord {
    /// Session-end outcome label. Mirrors the
    /// [`tmg_harness::SessionEndTrigger`] discriminator (e.g.
    /// `"completed"`, `"user_cancelled"`, `"timeout"`, `"errored"`).
    pub outcome: String,
    /// Feature ids the session marked as passing, when applicable.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub feature_marked_passing: Vec<String>,
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;

    /// Round-trip every record variant through `serde_json` to guard
    /// the on-disk schema. A failure here is a breaking change to the
    /// trajectory format.
    #[test]
    fn round_trip_meta() {
        let rec = TrajectoryRecord::Meta(MetaRecord {
            run_id: "abc12345".into(),
            session_num: 7,
            model: "qwen3.5-4b".into(),
            scope: "harnessed".into(),
            started_at: DateTime::parse_from_rfc3339("2026-04-28T00:00:00Z")
                .unwrap_or_else(|e| panic!("{e}"))
                .with_timezone(&Utc),
            ended_at: None,
            outcome: None,
        });
        let s = serde_json::to_string(&rec).unwrap_or_else(|e| panic!("{e}"));
        assert!(s.contains(r#""type":"meta""#), "tag missing: {s}");
        let back: TrajectoryRecord = serde_json::from_str(&s).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(rec, back);
    }

    #[test]
    fn round_trip_assistant_with_tool_calls() {
        let rec = TrajectoryRecord::Assistant(AssistantRecord {
            content: "calling file_read".into(),
            thinking: Some("I should read the file".into()),
            tool_calls: vec![ToolCallRecord {
                id: "call_1".into(),
                name: "file_read".into(),
                arguments: serde_json::json!({"path": "Cargo.toml"}),
            }],
        });
        let s = serde_json::to_string(&rec).unwrap_or_else(|e| panic!("{e}"));
        assert!(s.contains(r#""type":"assistant""#), "{s}");
        assert!(s.contains(r#""thinking""#), "{s}");
        let back: TrajectoryRecord = serde_json::from_str(&s).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(rec, back);
    }

    #[test]
    fn round_trip_tool_result() {
        let rec = TrajectoryRecord::ToolResult(ToolResultRecord {
            tool_call_id: "call_1".into(),
            tool_name: "file_read".into(),
            output: "[workspace]\nresolver = \"3\"".into(),
            is_error: false,
        });
        let s = serde_json::to_string(&rec).unwrap_or_else(|e| panic!("{e}"));
        assert!(s.contains(r#""type":"tool_result""#), "{s}");
        let back: TrajectoryRecord = serde_json::from_str(&s).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(rec, back);
    }

    #[test]
    fn round_trip_feedback() {
        let rec = TrajectoryRecord::Feedback(FeedbackRecord {
            source: "user_correction".into(),
            content: "use approach X instead".into(),
        });
        let s = serde_json::to_string(&rec).unwrap_or_else(|e| panic!("{e}"));
        let back: TrajectoryRecord = serde_json::from_str(&s).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(rec, back);
    }

    #[test]
    fn round_trip_verdict() {
        let rec = TrajectoryRecord::Verdict(VerdictRecord {
            outcome: "completed".into(),
            feature_marked_passing: vec!["feat-014".into(), "feat-015".into()],
        });
        let s = serde_json::to_string(&rec).unwrap_or_else(|e| panic!("{e}"));
        let back: TrajectoryRecord = serde_json::from_str(&s).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(rec, back);
    }

    /// `thinking = None` and empty `tool_calls` must serialize to a
    /// compact form (no `null` / no `[]`).
    #[test]
    fn assistant_compact_when_optional_fields_absent() {
        let rec = TrajectoryRecord::Assistant(AssistantRecord {
            content: "hi".into(),
            thinking: None,
            tool_calls: Vec::new(),
        });
        let s = serde_json::to_string(&rec).unwrap_or_else(|e| panic!("{e}"));
        assert!(!s.contains("thinking"), "{s}");
        assert!(!s.contains("tool_calls"), "{s}");
    }
}
