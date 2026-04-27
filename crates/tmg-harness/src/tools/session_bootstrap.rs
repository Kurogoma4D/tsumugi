//! `session_bootstrap` tool: returns the four SPEC §9.7 fields used to
//! warm up a session at startup.
//!
//! This tool is the ad-hoc-scope variant. The harnessed branch
//! (features.json + init.sh) lands in #34; for now we always emit the
//! four fields:
//!
//! - `working_directory`
//! - `recent_git_log` (output of `git log --oneline -N`)
//! - `progress_summary` (last K sessions of `progress.md`)
//! - `last_session_hint` (most recent `next_session_hint`)
//!
//! When the combined output exceeds `bootstrap_max_tokens`, we shed
//! older `progress.md` sessions and trim the git log until we fit.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use tmg_core::TokenCounter;
use tmg_tools::{Tool, ToolError, ToolResult};
use tokio::sync::Mutex;

use crate::runner::RunRunner;
// `TokenCounter` is referenced for its `estimate_tokens` heuristic only.
// The previously-supported "attach an LLM-backed counter" path was
// dropped in PR #57 review (see commit history): the `/tokenize`
// round-trip is undesirable on the synchronous bootstrap path, so we
// always fall back to the heuristic.

/// Default number of git log entries to include.
const DEFAULT_GIT_LOG_LIMIT: usize = 30;

/// Default number of progress.md sessions to include.
const DEFAULT_PROGRESS_SESSIONS: usize = 5;

/// Maximum time to wait for `git log` before giving up.
const GIT_LOG_TIMEOUT: Duration = Duration::from_secs(5);

/// Tool that returns the SPEC §9.7 bootstrap payload for the active run.
pub struct SessionBootstrapTool {
    runner: Arc<Mutex<RunRunner>>,
}

impl SessionBootstrapTool {
    /// Construct the tool over the given runner.
    pub fn new(runner: Arc<Mutex<RunRunner>>) -> Self {
        Self { runner }
    }

    /// Execute the bootstrap directly, without the [`Tool`] indirection.
    ///
    /// Used by the CLI startup path so the result can be injected as a
    /// system message into the [`AgentLoop`](tmg_core::AgentLoop)
    /// history before the user sends their first turn.
    pub async fn run_once(&self) -> Result<BootstrapPayload, ToolError> {
        run_bootstrap(&self.runner).await
    }
}

impl Tool for SessionBootstrapTool {
    fn name(&self) -> &'static str {
        "session_bootstrap"
    }

    fn description(&self) -> &'static str {
        "Return a compact context bundle for the current run: working \
         directory, recent git log, the last few progress.md sessions, \
         and the hint left by the previous session. Call once at the \
         start of a session if a bootstrap was not auto-injected."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        })
    }

    fn execute(
        &self,
        _params: serde_json::Value,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + '_>,
    > {
        let runner = Arc::clone(&self.runner);
        Box::pin(async move {
            let payload = run_bootstrap(&runner).await?;
            let json = serde_json::to_string_pretty(&payload)
                .map_err(|e| ToolError::json("serializing bootstrap payload", e))?;
            Ok(ToolResult::success(json))
        })
    }
}

/// Bootstrap payload returned by `session_bootstrap`.
///
/// Field order follows SPEC §9.7.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct BootstrapPayload {
    /// Absolute path to the workspace root.
    pub working_directory: String,
    /// Recent git log output (one commit per line, oldest last). Empty
    /// string when not in a git repository or git is unavailable.
    pub recent_git_log: String,
    /// Concatenated last-K sessions from `progress.md`. Empty string
    /// when no sessions have been recorded yet.
    pub progress_summary: String,
    /// The most recent `next_session_hint`, if any.
    pub last_session_hint: Option<String>,
    /// Whether the payload was truncated to fit `bootstrap_max_tokens`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub truncated: bool,
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if requires `&T -> bool`"
)]
const fn is_false(b: &bool) -> bool {
    !*b
}

/// Snapshot of state read out of the runner under one short lock.
struct BootstrapInputs {
    workspace_path: PathBuf,
    progress_log_path: PathBuf,
    bootstrap_max_tokens: usize,
    last_hint: Option<String>,
}

async fn run_bootstrap(runner: &Arc<Mutex<RunRunner>>) -> Result<BootstrapPayload, ToolError> {
    // Read everything we need under a single short lock so we never
    // hold the runner while shelling out to `git log`.
    let inputs: BootstrapInputs = {
        let guard = runner.lock().await;
        let last_hint = guard.session_log().last_hint().map_err(|e| {
            ToolError::io(
                "reading last_session_hint",
                std::io::Error::other(e.to_string()),
            )
        })?;
        BootstrapInputs {
            workspace_path: guard.workspace_path_owned(),
            progress_log_path: guard.progress_log().path().to_path_buf(),
            bootstrap_max_tokens: guard.bootstrap_max_tokens(),
            last_hint,
        }
    };

    let working_directory = inputs.workspace_path.display().to_string();
    let recent_git_log = collect_git_log(&inputs.workspace_path, DEFAULT_GIT_LOG_LIMIT).await;
    let progress_summary =
        read_recent_progress(&inputs.progress_log_path, DEFAULT_PROGRESS_SESSIONS)?;

    let mut payload = BootstrapPayload {
        working_directory,
        recent_git_log,
        progress_summary,
        last_session_hint: inputs.last_hint,
        truncated: false,
    };

    let max_tokens = inputs.bootstrap_max_tokens;
    if max_tokens > 0 {
        truncate_to_budget(&mut payload, max_tokens);
    }

    Ok(payload)
}

/// Read the last `n` sessions from `progress.md`.
fn read_recent_progress(path: &std::path::Path, n: usize) -> Result<String, ToolError> {
    use crate::artifacts::ProgressLog;
    let log = ProgressLog::new(path);
    log.read_recent_sessions(n)
        .map_err(|e| ToolError::io("reading progress.md", std::io::Error::other(e.to_string())))
}

/// Run `git log --oneline -N` inside `workspace`, returning stdout on
/// success and the empty string when the command fails (e.g. not a git
/// repo, git not installed, or timeout).
async fn collect_git_log(workspace: &std::path::Path, limit: usize) -> String {
    let limit_arg = format!("-{limit}");
    let mut cmd = tokio::process::Command::new("git");
    cmd.arg("log")
        .arg("--oneline")
        .arg(limit_arg)
        .current_dir(workspace)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    cmd.kill_on_drop(true);

    let Ok(child) = cmd.spawn() else {
        return String::new();
    };

    match tokio::time::timeout(GIT_LOG_TIMEOUT, child.wait_with_output()).await {
        Ok(Ok(output)) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).into_owned()
        }
        _ => String::new(),
    }
}

/// Approximate token count for `text` using
/// [`TokenCounter::estimate_tokens`] (the chars-per-4 heuristic).
///
/// We deliberately do not consult the LLM-backed `/tokenize` endpoint
/// here: the bootstrap path runs synchronously inside the tool dispatch,
/// and shelling out to the LLM server during startup is undesirable.
fn estimate_tokens(text: &str) -> usize {
    TokenCounter::estimate_tokens(text)
}

/// Trim the payload until its serialized form fits within `max_tokens`.
///
/// Strategy: shed older progress sessions first, then trim the git log
/// from its tail (oldest commits), then drop `last_session_hint` as a
/// last resort. **`working_directory` is preserved unconditionally** —
/// it is never trimmed, so the post-truncation token count is bounded by
/// `max_tokens + estimate_tokens(working_directory) + framing overhead`
/// rather than `max_tokens` alone.
///
/// Sets `payload.truncated = true` when any reduction was applied.
fn truncate_to_budget(payload: &mut BootstrapPayload, max_tokens: usize) {
    let initial = estimate_tokens(&serialize_for_size(payload));
    if initial <= max_tokens {
        return;
    }
    payload.truncated = true;

    // 1) Drop oldest progress sessions one at a time.
    while estimate_tokens(&serialize_for_size(payload)) > max_tokens {
        if !drop_oldest_progress_session(&mut payload.progress_summary) {
            break;
        }
    }

    // 2) Trim the git log from the bottom (oldest commit) by lines.
    while estimate_tokens(&serialize_for_size(payload)) > max_tokens
        && !payload.recent_git_log.is_empty()
    {
        let mut lines: Vec<&str> = payload.recent_git_log.lines().collect();
        if lines.is_empty() {
            break;
        }
        lines.pop();
        if lines.is_empty() {
            payload.recent_git_log.clear();
            break;
        }
        payload.recent_git_log = lines.join("\n");
        payload.recent_git_log.push('\n');
    }

    // 3) Last resort: clear the hint to free up budget.
    if estimate_tokens(&serialize_for_size(payload)) > max_tokens {
        payload.last_session_hint = None;
    }
}

/// Serialize the payload for size estimation. Failures fall back to a
/// best-effort string representation; we never want size-estimation to
/// abort the bootstrap.
fn serialize_for_size(payload: &BootstrapPayload) -> String {
    serde_json::to_string(payload).unwrap_or_else(|_| {
        format!(
            "{}\n{}\n{}\n{}",
            payload.working_directory,
            payload.recent_git_log,
            payload.progress_summary,
            payload.last_session_hint.as_deref().unwrap_or(""),
        )
    })
}

/// Remove the oldest `## Session ...` block from `progress_summary`.
///
/// Returns `true` if a session was dropped, `false` otherwise.
///
/// **Precondition:** by construction, `progress_summary` is the output
/// of [`ProgressLog::read_recent_sessions`](crate::artifacts::ProgressLog::read_recent_sessions),
/// which always starts at a `## Session ` header (or is empty). We
/// therefore do not need to scan for the first header — we drain from
/// byte `0` to the next header. An assertion guards the precondition
/// in debug builds.
fn drop_oldest_progress_session(progress_summary: &mut String) -> bool {
    if progress_summary.is_empty() {
        return false;
    }
    debug_assert!(
        progress_summary.starts_with("## Session "),
        "drop_oldest_progress_session expects content to begin at a session header; \
         got: {:?}",
        &progress_summary[..progress_summary.len().min(40)],
    );

    // Find the next header after the first one; drop everything up to
    // that point. If there is no next header, the entire summary is a
    // single session block — drop it all.
    let after_first = "## Session".len();
    if let Some(next_rel) = progress_summary[after_first..].find("\n## Session ") {
        let next_idx = after_first + next_rel + 1; // skip the leading '\n'
        progress_summary.drain(..next_idx);
    } else {
        progress_summary.clear();
    }
    true
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use crate::run::RunScope;
    use crate::session::SessionEndTrigger;
    use crate::store::RunStore;
    use chrono::Utc;

    fn make_runner() -> (tempfile::TempDir, Arc<Mutex<RunRunner>>) {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let store = Arc::new(RunStore::new(tmp.path().join("runs")));
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut runner = RunRunner::new(run, store);
        let _ = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        (tmp, Arc::new(Mutex::new(runner)))
    }

    #[tokio::test]
    async fn ad_hoc_bootstrap_returns_all_four_fields() {
        let (_tmp, runner) = make_runner();

        // Seed progress.md with a session.
        {
            let guard = runner.lock().await;
            guard
                .progress_log()
                .append_entry("did a thing")
                .unwrap_or_else(|e| panic!("{e}"));
        }

        // Seed a previous session with a hint.
        {
            let guard = runner.lock().await;
            let mut prev = crate::session::Session::begin(0);
            prev.next_session_hint = Some("focus on tests next".to_owned());
            prev.end(SessionEndTrigger::Completed);
            guard
                .session_log()
                .save(&prev)
                .unwrap_or_else(|e| panic!("{e}"));
        }

        let tool = SessionBootstrapTool::new(Arc::clone(&runner));
        let payload = tool.run_once().await.unwrap_or_else(|e| panic!("{e}"));

        // Working directory matches workspace_path.
        let expected_workspace = runner.lock().await.workspace_path_owned();
        assert_eq!(
            payload.working_directory,
            expected_workspace.display().to_string()
        );

        // recent_git_log is a string (possibly empty when not in a git
        // repo), but the field is always present.
        let _ = payload.recent_git_log.clone();

        // progress_summary contains the seeded entry.
        assert!(
            payload.progress_summary.contains("did a thing"),
            "progress_summary missing seeded entry: {}",
            payload.progress_summary
        );

        // last_session_hint is the one we wrote.
        assert_eq!(
            payload.last_session_hint.as_deref(),
            Some("focus on tests next")
        );
    }

    #[tokio::test]
    async fn execute_returns_json_string() {
        let (_tmp, runner) = make_runner();
        let tool = SessionBootstrapTool::new(runner);
        let res = tool
            .execute(serde_json::json!({}))
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(!res.is_error);
        let parsed: serde_json::Value =
            serde_json::from_str(&res.output).unwrap_or_else(|e| panic!("{e}"));
        for key in [
            "working_directory",
            "recent_git_log",
            "progress_summary",
            "last_session_hint",
        ] {
            assert!(parsed.get(key).is_some(), "missing key {key} in {parsed}");
        }
    }

    #[tokio::test]
    async fn truncates_when_over_budget() {
        let (_tmp, runner) = make_runner();

        // Force a tiny budget and seed a fat progress log.
        {
            let mut guard = runner.lock().await;
            guard.set_bootstrap_max_tokens(64);
            let scope = RunScope::AdHoc;
            let ts = Utc::now();
            for i in 1..=10_u32 {
                guard
                    .progress_log()
                    .append_session_header(i, &scope, ts)
                    .unwrap_or_else(|e| panic!("{e}"));
                for _ in 0..50 {
                    guard
                        .progress_log()
                        .append_entry(&"x".repeat(80))
                        .unwrap_or_else(|e| panic!("{e}"));
                }
            }
        }

        let tool = SessionBootstrapTool::new(Arc::clone(&runner));
        let payload = tool.run_once().await.unwrap_or_else(|e| panic!("{e}"));
        assert!(payload.truncated, "payload should be marked truncated");
        // `working_directory` is preserved unconditionally by
        // `truncate_to_budget`, so the post-truncation payload's
        // estimated token count must fit within
        // `budget + estimate_tokens(working_directory)`. Allow a small
        // constant for JSON framing overhead (`{}`, quotes, field names);
        // 64 tokens is comfortably above what the heuristic produces for
        // the bookkeeping fields.
        let serialized = serialize_for_size(&payload);
        let actual = TokenCounter::estimate_tokens(&serialized);
        let budget = 64;
        let preserved = TokenCounter::estimate_tokens(&payload.working_directory);
        let framing_overhead = 64;
        assert!(
            actual <= budget + preserved + framing_overhead,
            "post-truncation token count {actual} exceeds budget {budget} + \
             working_directory {preserved} + framing {framing_overhead}; \
             serialized len {} bytes",
            serialized.len(),
        );
    }

    /// `bootstrap_max_tokens = 0` is documented as "no truncation":
    /// even when the payload would normally be over budget, every
    /// session is preserved and `truncated` stays `false`.
    #[tokio::test]
    async fn zero_budget_disables_truncation() {
        let (_tmp, runner) = make_runner();

        // Same fat seed as `truncates_when_over_budget`, but with the
        // budget pinned to 0.
        {
            let mut guard = runner.lock().await;
            guard.set_bootstrap_max_tokens(0);
            let scope = RunScope::AdHoc;
            let ts = Utc::now();
            for i in 1..=10_u32 {
                guard
                    .progress_log()
                    .append_session_header(i, &scope, ts)
                    .unwrap_or_else(|e| panic!("{e}"));
                for _ in 0..50 {
                    guard
                        .progress_log()
                        .append_entry(&"x".repeat(80))
                        .unwrap_or_else(|e| panic!("{e}"));
                }
            }
        }

        let tool = SessionBootstrapTool::new(Arc::clone(&runner));
        let payload = tool.run_once().await.unwrap_or_else(|e| panic!("{e}"));
        assert!(!payload.truncated, "zero budget must skip truncation");
        // The full progress.md (5 sessions ish) survives.
        assert!(
            payload.progress_summary.contains("Session #6"),
            "older sessions should be preserved when budget is 0"
        );
    }

    #[test]
    fn drop_oldest_progress_session_drops_first_block() {
        let mut s =
            String::from("## Session #1 (a) [ad-hoc]\n- e1\n## Session #2 (b) [ad-hoc]\n- e2\n");
        let dropped = drop_oldest_progress_session(&mut s);
        assert!(dropped);
        assert!(s.starts_with("## Session #2"), "{s}");
        assert!(!s.contains("Session #1"), "{s}");
    }

    #[test]
    fn drop_oldest_progress_session_clears_when_only_one() {
        let mut s = String::from("## Session #1 (a) [ad-hoc]\n- e1\n");
        let dropped = drop_oldest_progress_session(&mut s);
        assert!(dropped);
        assert!(s.is_empty());
    }

    /// Bootstrap injection is observable: `BootstrapPayload` serializes
    /// stably so the same string the CLI feeds into
    /// `AgentLoop::push_system_message` is reconstructible from the
    /// payload. This is the deterministic half of the integration test;
    /// the other half (pushing into the agent's history) is covered by
    /// `tmg-core`'s `push_system_message_appends_to_history` test.
    #[tokio::test]
    async fn bootstrap_payload_round_trips_for_history_injection() {
        let (_tmp, runner) = make_runner();
        // Seed something in progress.md so the payload is non-trivial.
        runner
            .lock()
            .await
            .progress_log()
            .append_entry("seed entry")
            .unwrap_or_else(|e| panic!("{e}"));

        let tool = SessionBootstrapTool::new(Arc::clone(&runner));
        let payload = tool.run_once().await.unwrap_or_else(|e| panic!("{e}"));

        let json = serde_json::to_string_pretty(&payload).unwrap_or_else(|e| panic!("{e}"));
        let injected = format!("[session_bootstrap]\n{json}\n[/session_bootstrap]");

        // The injected content carries every documented field so the
        // LLM can act on it; this is what would land in
        // AgentLoop::history as a system message.
        assert!(injected.contains("\"working_directory\""), "{injected}");
        assert!(injected.contains("\"recent_git_log\""), "{injected}");
        assert!(injected.contains("\"progress_summary\""), "{injected}");
        assert!(injected.contains("\"last_session_hint\""), "{injected}");
        assert!(
            injected.contains("seed entry"),
            "progress.md content missing from injection: {injected}"
        );
        assert!(injected.starts_with("[session_bootstrap]\n"));
        assert!(injected.ends_with("\n[/session_bootstrap]"));
    }
}
