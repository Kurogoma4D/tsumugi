//! [`SessionSearchTool`]: agent-facing FTS5 search over historical
//! sessions.
//!
//! Wraps a shared [`SearchIndex`] and exposes the JSON shape promised
//! by issue #53:
//!
//! ```text
//! Tool: session_search
//! Parameters: {
//!   "query": "<FTS5 query>",
//!   "limit": 10,
//!   "scope": "summary" | "turns" | "all",
//!   "since": "<ISO8601>"
//! }
//! ```
//!
//! Result: `[{run_id, session_num, started_at, summary, snippet, score}]`.
//!
//! Full turn payloads are intentionally **not** returned here — the
//! agent should follow up with `file_read` against
//! `.tsumugi/runs/<id>/session_log/session_NNN.json` when it wants the
//! verbatim transcript. This keeps the search hit small enough to fit
//! in the context window even for noisy queries.

use std::pin::Pin;
use std::sync::Arc;

use chrono::DateTime;
use serde_json::Value as JsonValue;
use tmg_sandbox::SandboxContext;
use tmg_tools::error::ToolError;
use tmg_tools::types::{Tool, ToolResult};

use crate::index::{SearchIndex, SearchScope};

/// Tool name as registered in [`tmg_tools::ToolRegistry`].
pub const SESSION_SEARCH_TOOL_NAME: &str = "session_search";

/// Wrapper that adapts [`SearchIndex`] to the [`Tool`] trait.
///
/// The index is held behind `Option<Arc<SearchIndex>>` so the tool can
/// be registered unconditionally; when the index is `None` (e.g. the
/// search feature is disabled, or the DB couldn't be opened) the tool
/// returns a soft `ToolResult::error` rather than failing hard. This
/// matches the issue's "DB unavailable → tool returns soft error"
/// requirement.
pub struct SessionSearchTool {
    index: Option<Arc<SearchIndex>>,
}

impl SessionSearchTool {
    /// Construct a tool around a live index.
    #[must_use]
    pub fn new(index: Arc<SearchIndex>) -> Self {
        Self { index: Some(index) }
    }

    /// Construct a tool that always returns a soft error. Used when
    /// the search feature is disabled at config time so the agent
    /// still sees the tool surface but every call short-circuits.
    #[must_use]
    pub fn disabled() -> Self {
        Self { index: None }
    }
}

impl Tool for SessionSearchTool {
    fn name(&self) -> &'static str {
        SESSION_SEARCH_TOOL_NAME
    }

    fn description(&self) -> &'static str {
        "Search past session summaries and turn transcripts (full-text via SQLite FTS5). \
         Use this before tackling a problem to see if you've already solved something similar. \
         Returns run_id, session_num, summary, and a highlighted snippet — read the full \
         session via `file_read` on `.tsumugi/runs/<run_id>/session_log/session_NNN.json` \
         for verbatim turn text."
    }

    fn parameters_schema(&self) -> JsonValue {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "FTS5 MATCH query. Operators: AND, OR, NEAR, prefix `*`."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "default": 10,
                    "description": "Maximum number of hits to return (clamped to 200)."
                },
                "scope": {
                    "type": "string",
                    "enum": ["summary", "turns", "all"],
                    "default": "all",
                    "description": "Which FTS table(s) to consult."
                },
                "since": {
                    "type": "string",
                    "description": "Optional ISO8601 / RFC3339 timestamp; only sessions started \
                                    at or after this time are returned."
                }
            },
            "required": ["query"],
            "additionalProperties": false
        })
    }

    fn execute<'a>(
        &'a self,
        params: JsonValue,
        _ctx: &'a SandboxContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult, ToolError>> + Send + 'a>> {
        let result = self.execute_inner(&params);
        Box::pin(std::future::ready(result))
    }
}

impl SessionSearchTool {
    fn execute_inner(&self, params: &JsonValue) -> Result<ToolResult, ToolError> {
        let Some(index) = self.index.as_ref() else {
            return Ok(ToolResult::error(
                "session_search is disabled: enable [search] in tsumugi.toml",
            ));
        };
        let Some(query) = params.get("query").and_then(JsonValue::as_str) else {
            return Err(ToolError::invalid_params(
                "missing required parameter: query",
            ));
        };
        let limit = params
            .get("limit")
            .and_then(JsonValue::as_u64)
            .map_or(10_usize, |n| usize::try_from(n).unwrap_or(10));
        let scope = params
            .get("scope")
            .and_then(JsonValue::as_str)
            .map_or(SearchScope::All, SearchScope::parse);
        let since = match params.get("since").and_then(JsonValue::as_str) {
            Some(s) => match DateTime::parse_from_rfc3339(s) {
                Ok(dt) => Some(dt.with_timezone(&chrono::Utc)),
                Err(e) => {
                    return Err(ToolError::invalid_params(format!(
                        "invalid since timestamp {s:?}: {e}"
                    )));
                }
            },
            None => None,
        };

        match index.query(query, limit, scope, since) {
            Ok(hits) => {
                // Serialize the structured hit list as the tool's
                // output. The agent loop already JSON-pretty-prints
                // tool results in the activity pane; emitting JSON
                // here preserves machine-parseability for scripts.
                let payload = serde_json::to_string_pretty(&hits)
                    .map_err(|e| ToolError::json("serialize search hits", e))?;
                if hits.is_empty() {
                    Ok(ToolResult::success(format!(
                        "no matches for query {query:?}\n\n{payload}"
                    )))
                } else {
                    Ok(ToolResult::success(payload))
                }
            }
            Err(crate::error::SearchError::InvalidQuery { message }) => {
                Err(ToolError::invalid_params(message))
            }
            Err(other) => {
                // Treat infrastructure failures (DB locked, schema
                // mismatch, ...) as soft errors so the agent can
                // recover or move on.
                Ok(ToolResult::error(format!("session_search failed: {other}")))
            }
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test assertions use expect for clearer messages"
)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tmg_harness::{Session, SessionEndTrigger};

    fn make_index(tmp: &TempDir) -> Arc<SearchIndex> {
        Arc::new(SearchIndex::open(tmp.path().join("state.db")).expect("open"))
    }

    #[tokio::test]
    async fn search_returns_hits_as_json() {
        let tmp = TempDir::new().expect("tempdir");
        let idx = make_index(&tmp);
        let mut s = Session::begin(1);
        s.summary = "fixed OAuth callback redirect loop".to_owned();
        s.end(SessionEndTrigger::Completed);
        idx.ingest_session("run-aaa", &s).expect("ingest");

        let tool = SessionSearchTool::new(Arc::clone(&idx));
        let ctx = SandboxContext::test_default();
        let res = tool
            .execute(serde_json::json!({"query": "OAuth"}), &ctx)
            .await
            .expect("execute");
        assert!(!res.is_error, "{}", res.output);
        // JSON-parseable.
        let parsed: serde_json::Value = serde_json::from_str(&res.output).expect("output is JSON");
        let arr = parsed.as_array().expect("array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["run_id"], "run-aaa");
        assert_eq!(arr[0]["session_num"], 1);
    }

    #[tokio::test]
    async fn disabled_tool_returns_soft_error() {
        let tool = SessionSearchTool::disabled();
        let ctx = SandboxContext::test_default();
        let res = tool
            .execute(serde_json::json!({"query": "anything"}), &ctx)
            .await
            .expect("execute");
        assert!(res.is_error);
        assert!(res.output.contains("disabled"), "{}", res.output);
    }

    #[tokio::test]
    async fn missing_query_is_invalid_params() {
        let tmp = TempDir::new().expect("tempdir");
        let idx = make_index(&tmp);
        let tool = SessionSearchTool::new(idx);
        let ctx = SandboxContext::test_default();
        let err = tool
            .execute(serde_json::json!({}), &ctx)
            .await
            .expect_err("missing query");
        assert!(matches!(err, ToolError::InvalidParams { .. }));
    }

    #[tokio::test]
    async fn invalid_since_is_invalid_params() {
        let tmp = TempDir::new().expect("tempdir");
        let idx = make_index(&tmp);
        let tool = SessionSearchTool::new(idx);
        let ctx = SandboxContext::test_default();
        let err = tool
            .execute(
                serde_json::json!({"query": "x", "since": "not-a-timestamp"}),
                &ctx,
            )
            .await
            .expect_err("bad since");
        assert!(matches!(err, ToolError::InvalidParams { .. }));
    }

    #[tokio::test]
    async fn empty_results_still_succeed() {
        let tmp = TempDir::new().expect("tempdir");
        let idx = make_index(&tmp);
        let tool = SessionSearchTool::new(idx);
        let ctx = SandboxContext::test_default();
        let res = tool
            .execute(serde_json::json!({"query": "nothing matches"}), &ctx)
            .await
            .expect("execute");
        assert!(!res.is_error, "{}", res.output);
        assert!(res.output.contains("no matches"));
    }
}
