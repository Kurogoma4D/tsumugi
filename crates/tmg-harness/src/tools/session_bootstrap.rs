//! `session_bootstrap` tool: returns the SPEC §9.7 fields used to warm
//! up a session at startup.
//!
//! ## Ad-hoc scope
//!
//! For ad-hoc runs, the tool emits four fields:
//!
//! - `working_directory`
//! - `recent_git_log` (output of `git log --oneline -N`)
//! - `progress_summary` (last K sessions of `progress.md`)
//! - `last_session_hint` (most recent `next_session_hint`)
//!
//! ## Harnessed scope
//!
//! For harnessed runs, the same fields are emitted **plus**:
//!
//! - `features_summary` (compact view of `features.json`)
//! - `init_script_status` (exit code + tail of stdout/stderr from
//!   running `init.sh` with the `[harness] default_session_timeout`
//!   budget)
//! - `smoke_test_result` (stub `{ "tester_unavailable": true }` until
//!   the tester subagent ships in a follow-up issue)
//!
//! When the combined output exceeds `bootstrap_max_tokens`, harnessed
//! payloads first trim `features_summary` (older / lower-priority
//! entries), then fall through to the ad-hoc trimming order
//! (older `progress.md` sessions → tail of git log → `last_session_hint`).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use tmg_core::TokenCounter;
use tmg_sandbox::{SandboxConfig, SandboxContext, SandboxMode};
use tmg_tools::{Tool, ToolError, ToolResult};
use tokio::sync::Mutex;

use crate::artifacts::{FeatureList, FeaturesSummary, InitScript, InitScriptOutput};
use crate::run::RunScope;
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

/// Default number of features included in the `features_summary`.
const DEFAULT_FEATURES_TOP_N: usize = 20;

/// Default time budget for running `init.sh` during harnessed bootstrap.
///
/// The script is meant to be cheap (`yarn install`-class workloads), so
/// 60 seconds is a reasonable upper bound; longer-running setup should
/// move into a dedicated subagent.
const DEFAULT_INIT_SCRIPT_TIMEOUT: Duration = Duration::from_secs(60);

/// Number of stdout/stderr tail lines to surface for `init.sh`.
const INIT_SCRIPT_TAIL_LINES: usize = 20;

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
/// Field order follows SPEC §9.7. Harnessed-only fields default to
/// `None` for ad-hoc runs and `skip_serializing_if = "Option::is_none"`,
/// so they do not appear in ad-hoc output at all.
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
    /// Compact view of `features.json` (harnessed scope only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub features_summary: Option<FeaturesSummary>,
    /// Result of running `init.sh` (harnessed scope only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_script_status: Option<InitScriptStatus>,
    /// Result of running the tester subagent (harnessed scope only).
    ///
    /// Currently always `Some(SmokeTestResult::Unavailable { .. })` for
    /// harnessed runs because the tester subagent is not yet
    /// implemented; activated once the tester ships.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smoke_test_result: Option<SmokeTestResult>,
    /// Whether the payload was truncated to fit `bootstrap_max_tokens`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub truncated: bool,
}

/// Tail-line view of an `init.sh` invocation surfaced through
/// [`BootstrapPayload::init_script_status`].
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct InitScriptStatus {
    /// Exit code of the script (`-1` when killed by signal/timeout).
    pub exit_code: i32,
    /// Last `INIT_SCRIPT_TAIL_LINES` of stdout, joined with `\n`.
    pub tail_stdout: String,
    /// Last `INIT_SCRIPT_TAIL_LINES` of stderr, joined with `\n`.
    pub tail_stderr: String,
    /// Whether the script was killed because it exceeded the timeout
    /// budget.
    pub timed_out: bool,
}

/// Result of the harnessed-scope smoke test.
///
/// The tester subagent is tracked in a follow-up issue; until it lands,
/// the bootstrap always returns the [`Unavailable`](Self::Unavailable)
/// variant so the LLM can reason about its absence rather than seeing
/// no field at all.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum SmokeTestResult {
    /// Tester subagent is not yet wired up; `tester_unavailable` is
    /// always `true`.
    Unavailable {
        /// Always `true`; present so the LLM sees a stable, easily
        /// matched signal.
        tester_unavailable: bool,
        /// Reason the tester wasn't run (kept short and actionable).
        reason: String,
    },
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
    /// `Some` only when the run is harnessed.
    harness_inputs: Option<HarnessBootstrapInputs>,
}

/// Inputs needed to populate the harnessed-only fields. Held in a
/// separate struct so `BootstrapInputs` stays the same shape for ad-hoc
/// runs.
struct HarnessBootstrapInputs {
    features: FeatureList,
    init_script: InitScript,
}

async fn run_bootstrap(runner: &Arc<Mutex<RunRunner>>) -> Result<BootstrapPayload, ToolError> {
    // Read everything we need under a single short lock so we never
    // hold the runner while shelling out to `git log` or running
    // `init.sh`.
    let inputs: BootstrapInputs = {
        let guard = runner.lock().await;
        let last_hint = guard.session_log().last_hint().map_err(|e| {
            ToolError::io(
                "reading last_session_hint",
                std::io::Error::other(e.to_string()),
            )
        })?;
        let harness_inputs = match guard.scope() {
            RunScope::Harnessed { .. } => Some(HarnessBootstrapInputs {
                features: guard.features().clone(),
                init_script: guard.init_script().clone(),
            }),
            RunScope::AdHoc => None,
        };
        BootstrapInputs {
            workspace_path: guard.workspace_path_owned(),
            progress_log_path: guard.progress_log().path().to_path_buf(),
            bootstrap_max_tokens: guard.bootstrap_max_tokens(),
            last_hint,
            harness_inputs,
        }
    };

    let working_directory = inputs.workspace_path.display().to_string();
    let recent_git_log = collect_git_log(&inputs.workspace_path, DEFAULT_GIT_LOG_LIMIT).await;
    let progress_summary =
        read_recent_progress(&inputs.progress_log_path, DEFAULT_PROGRESS_SESSIONS)?;

    let (features_summary, init_script_status, smoke_test_result) =
        if let Some(harness) = inputs.harness_inputs.as_ref() {
            let features_summary = collect_features_summary(&harness.features);
            let init_script_status =
                collect_init_script_status(&harness.init_script, &inputs.workspace_path).await;
            let smoke_test_result = Some(tester_unavailable_placeholder());
            (features_summary, init_script_status, smoke_test_result)
        } else {
            (None, None, None)
        };

    let mut payload = BootstrapPayload {
        working_directory,
        recent_git_log,
        progress_summary,
        last_session_hint: inputs.last_hint,
        features_summary,
        init_script_status,
        smoke_test_result,
        truncated: false,
    };

    let max_tokens = inputs.bootstrap_max_tokens;
    if max_tokens > 0 {
        truncate_to_budget(&mut payload, max_tokens);
    }

    Ok(payload)
}

/// Read the features summary, returning `None` if `features.json` does
/// not exist or fails to parse. Bootstrap is best-effort: a missing or
/// corrupt features file should not abort the whole bundle.
fn collect_features_summary(features: &FeatureList) -> Option<FeaturesSummary> {
    if !features.path().exists() {
        return None;
    }
    match features.summary(DEFAULT_FEATURES_TOP_N) {
        Ok(summary) => Some(summary),
        Err(err) => {
            tracing::warn!(
                path = %features.path().display(),
                error = %err,
                "skipping features_summary in session_bootstrap"
            );
            None
        }
    }
}

/// Run `init.sh` and convert the result into an [`InitScriptStatus`].
///
/// Returns `None` when `init.sh` does not exist on disk; the agent will
/// see no `init_script_status` field rather than a confusing
/// not-found error.
async fn collect_init_script_status(
    init_script: &InitScript,
    workspace: &Path,
) -> Option<InitScriptStatus> {
    if !init_script.exists() {
        return None;
    }

    // Use a `Full`-mode sandbox here: this issue is only about wiring
    // the `init.sh` execution path. A future issue will tighten the
    // policy to `WorkspaceWrite` once we are confident that
    // initialiser scripts only need to touch the workspace.
    let sandbox_config = SandboxConfig::new(workspace).with_mode(SandboxMode::Full);
    let sandbox = SandboxContext::new(sandbox_config);

    match init_script.run(&sandbox, DEFAULT_INIT_SCRIPT_TIMEOUT).await {
        Ok(output) => Some(init_script_status_from_output(&output)),
        Err(err) => {
            tracing::warn!(
                path = %init_script.path().display(),
                error = %err,
                "init.sh execution failed in session_bootstrap"
            );
            None
        }
    }
}

fn init_script_status_from_output(output: &InitScriptOutput) -> InitScriptStatus {
    InitScriptStatus {
        exit_code: output.exit_code,
        tail_stdout: output.tail_stdout(INIT_SCRIPT_TAIL_LINES),
        tail_stderr: output.tail_stderr(INIT_SCRIPT_TAIL_LINES),
        timed_out: output.timed_out,
    }
}

/// Placeholder smoke-test result emitted while the tester subagent is
/// not yet implemented.
//
// TODO(#tester-subagent): Replace this stub with a real invocation of
// the tester subagent once that issue lands. The replacement should
// honour `[harness] smoke_test_every_n_sessions` and gate execution by
// `RunRunner::session_count()`.
fn tester_unavailable_placeholder() -> SmokeTestResult {
    SmokeTestResult::Unavailable {
        tester_unavailable: true,
        reason: "tester subagent not yet implemented".to_owned(),
    }
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
/// Strategy:
/// 1. **Harnessed scope only**: shed `features_summary` entries (older /
///    lower-priority — i.e. the *passing* ones at the tail of the
///    summary list) before touching ad-hoc fields. `total` and
///    `passing` counts are preserved so the LLM still sees the
///    cardinality even when individual entries are dropped.
/// 2. Drop the oldest `progress.md` sessions one at a time.
/// 3. Trim the `git log` from the tail (oldest commits) by lines.
/// 4. Last resort: clear `last_session_hint`.
///
/// `working_directory`, `init_script_status`, and `smoke_test_result`
/// are preserved unconditionally — they are bounded in size by
/// construction (`tail_stdout`/`tail_stderr` already cap at
/// `INIT_SCRIPT_TAIL_LINES`), so the post-truncation token count is
/// bounded by `max_tokens + framing_overhead + size_of_preserved_fields`
/// rather than `max_tokens` alone.
///
/// Sets `payload.truncated = true` when any reduction was applied.
fn truncate_to_budget(payload: &mut BootstrapPayload, max_tokens: usize) {
    let initial = estimate_tokens(&serialize_for_size(payload));
    if initial <= max_tokens {
        return;
    }
    payload.truncated = true;

    // 1) Harnessed-only: shed features_summary entries from the tail
    // (passing entries are placed last by `FeatureList::summary`, so
    // they go first; failing entries survive longer).
    if payload.features_summary.is_some() {
        while estimate_tokens(&serialize_for_size(payload)) > max_tokens {
            let dropped = drop_features_summary_tail(&mut payload.features_summary);
            if !dropped {
                break;
            }
        }
    }

    // 2) Drop oldest progress sessions one at a time.
    while estimate_tokens(&serialize_for_size(payload)) > max_tokens {
        if !drop_oldest_progress_session(&mut payload.progress_summary) {
            break;
        }
    }

    // 3) Trim the git log from the bottom (oldest commit) by lines.
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

    // 4) Last resort: clear the hint to free up budget.
    if estimate_tokens(&serialize_for_size(payload)) > max_tokens {
        payload.last_session_hint = None;
    }
}

/// Drop one entry from the tail of `features_summary.entries`.
///
/// Returns `true` when an entry was dropped, `false` when there is
/// nothing more to remove (the field is `None`, the list is empty, or
/// the field has been replaced with `None` by a previous full drain).
///
/// Once `entries` is empty, the entire `features_summary` field is set
/// to `None` so the field disappears from the serialized payload (one
/// extra byte savings via `skip_serializing_if`).
fn drop_features_summary_tail(features_summary: &mut Option<FeaturesSummary>) -> bool {
    let Some(summary) = features_summary.as_mut() else {
        return false;
    };
    if summary.entries.is_empty() {
        // No more rows; drop the field entirely.
        *features_summary = None;
        return true;
    }
    summary.entries.pop();
    summary.truncated = true;
    if summary.entries.is_empty() {
        // All rows shed: drop the now-empty summary so it stops
        // wasting tokens.
        *features_summary = None;
    }
    true
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

    /// Helper: turn an existing ad-hoc runner into a harnessed one,
    /// optionally seeding `features.json` and `init.sh` in the run
    /// directory.
    async fn promote_to_harnessed(
        runner: &Arc<Mutex<RunRunner>>,
        features_json: Option<&str>,
        init_sh: Option<&str>,
    ) {
        let mut guard = runner.lock().await;
        let new_scope = RunScope::Harnessed {
            workflow_id: "wf".to_owned(),
            max_sessions: None,
        };
        guard.run_mut().scope = new_scope.clone();
        guard.run_mut().workflow_id = Some("wf".to_owned());
        let features_path = guard.features().path().to_path_buf();
        let init_path = guard.init_script().path().to_path_buf();
        drop(guard);

        if let Some(content) = features_json {
            std::fs::write(&features_path, content).unwrap_or_else(|e| panic!("{e}"));
        }
        if let Some(content) = init_sh {
            std::fs::write(&init_path, content).unwrap_or_else(|e| panic!("{e}"));
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt as _;
                let mut perms = std::fs::metadata(&init_path)
                    .unwrap_or_else(|e| panic!("{e}"))
                    .permissions();
                perms.set_mode(0o755);
                std::fs::set_permissions(&init_path, perms).unwrap_or_else(|e| panic!("{e}"));
            }
        }
    }

    fn sample_features_json() -> &'static str {
        r#"{
  "features": [
    {
      "id": "feat-001",
      "category": "auth",
      "description": "Login",
      "steps": ["A"],
      "passes": false
    },
    {
      "id": "feat-002",
      "category": "auth",
      "description": "Logout",
      "steps": ["B"],
      "passes": false
    },
    {
      "id": "feat-003",
      "category": "billing",
      "description": "Invoice",
      "steps": ["C"],
      "passes": true
    }
  ]
}"#
    }

    /// Acceptance: ad-hoc runs do not surface the harnessed-only fields
    /// in either the typed payload or the JSON.
    #[tokio::test]
    async fn ad_hoc_payload_omits_harnessed_fields() {
        let (_tmp, runner) = make_runner();
        let tool = SessionBootstrapTool::new(Arc::clone(&runner));
        let payload = tool.run_once().await.unwrap_or_else(|e| panic!("{e}"));
        assert!(payload.features_summary.is_none());
        assert!(payload.init_script_status.is_none());
        assert!(payload.smoke_test_result.is_none());

        let res = tool
            .execute(serde_json::json!({}))
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        let parsed: serde_json::Value =
            serde_json::from_str(&res.output).unwrap_or_else(|e| panic!("{e}"));
        for key in [
            "features_summary",
            "init_script_status",
            "smoke_test_result",
        ] {
            assert!(
                parsed.get(key).is_none(),
                "ad-hoc payload should not contain `{key}`: {parsed}"
            );
        }
    }

    /// Acceptance: harnessed runs return the harnessed-only fields.
    #[tokio::test]
    async fn harnessed_payload_includes_features_and_smoke_test() {
        let (_tmp, runner) = make_runner();
        promote_to_harnessed(&runner, Some(sample_features_json()), None).await;

        let tool = SessionBootstrapTool::new(Arc::clone(&runner));
        let payload = tool.run_once().await.unwrap_or_else(|e| panic!("{e}"));

        let summary = payload
            .features_summary
            .as_ref()
            .unwrap_or_else(|| panic!("expected features_summary"));
        assert_eq!(summary.total, 3);
        assert_eq!(summary.passing, 1);
        assert!(!summary.entries.is_empty());

        // No init.sh seeded -> field absent.
        assert!(payload.init_script_status.is_none());

        let smoke = payload
            .smoke_test_result
            .as_ref()
            .unwrap_or_else(|| panic!("expected smoke_test_result"));
        match smoke {
            SmokeTestResult::Unavailable {
                tester_unavailable,
                reason,
            } => {
                assert!(tester_unavailable);
                assert!(!reason.is_empty());
            }
        }
    }

    /// Acceptance: when `init.sh` exists in a harnessed run, it is
    /// executed and its tail is captured.
    #[tokio::test]
    async fn harnessed_payload_runs_init_script() {
        let (_tmp, runner) = make_runner();
        let init_content = "#!/bin/sh\necho init-ran\n";
        promote_to_harnessed(&runner, Some(sample_features_json()), Some(init_content)).await;

        let tool = SessionBootstrapTool::new(Arc::clone(&runner));
        let payload = tool.run_once().await.unwrap_or_else(|e| panic!("{e}"));
        let status = payload
            .init_script_status
            .as_ref()
            .unwrap_or_else(|| panic!("expected init_script_status"));
        assert_eq!(status.exit_code, 0);
        assert!(!status.timed_out);
        assert!(
            status.tail_stdout.contains("init-ran"),
            "{}",
            status.tail_stdout
        );
    }

    /// `bootstrap_max_tokens` truncation in harnessed runs prefers
    /// trimming `features_summary` before touching ad-hoc fields.
    /// Acceptance test for issue #34.
    #[tokio::test]
    async fn truncation_prefers_features_summary() {
        use std::fmt::Write as _;

        let (_tmp, runner) = make_runner();

        // Build a harnessed run with a fat features list. Many
        // entries, all distinct ids, so the JSON is large.
        let mut features_json = String::from("{\n  \"features\": [\n");
        for i in 0..50 {
            if i > 0 {
                features_json.push_str(",\n");
            }
            write!(
                features_json,
                "    {{ \"id\": \"feat-{i:03}\", \"category\": \"cat\", \
                 \"description\": \"description that takes up some space {}\", \
                 \"steps\": [\"step1\", \"step2\"], \"passes\": false }}",
                "x".repeat(40)
            )
            .unwrap_or_else(|e| panic!("write failed: {e}"));
        }
        features_json.push_str("\n  ]\n}\n");

        promote_to_harnessed(&runner, Some(&features_json), None).await;

        // Seed a small progress.md entry so we can later confirm it
        // survives features-summary trimming.
        {
            let guard = runner.lock().await;
            guard
                .progress_log()
                .append_entry("progress sentinel")
                .unwrap_or_else(|e| panic!("{e}"));
        }

        // Force a tight budget that is comfortably bigger than the
        // bookkeeping fields but smaller than the unrolled
        // `features_summary`.
        {
            let mut guard = runner.lock().await;
            guard.set_bootstrap_max_tokens(128);
        }

        let tool = SessionBootstrapTool::new(Arc::clone(&runner));
        let payload = tool.run_once().await.unwrap_or_else(|e| panic!("{e}"));
        assert!(payload.truncated, "expected payload to be truncated");

        // The features_summary entries should have been trimmed (or
        // dropped entirely) before the progress.md sentinel disappears.
        let entries_after = payload
            .features_summary
            .as_ref()
            .map_or(0, |s| s.entries.len());
        assert!(
            entries_after < 50,
            "features_summary should have been trimmed, got {entries_after} entries"
        );
        assert!(
            payload.progress_summary.contains("progress sentinel"),
            "progress.md must survive features-first trimming: {}",
            payload.progress_summary,
        );
    }

    #[test]
    fn drop_features_summary_tail_progressively_empties_entries() {
        let mut summary = Some(FeaturesSummary {
            total: 3,
            passing: 0,
            entries: vec![
                FeaturesSummaryEntry {
                    id: "a".to_owned(),
                    category: "x".to_owned(),
                    passes: false,
                },
                FeaturesSummaryEntry {
                    id: "b".to_owned(),
                    category: "x".to_owned(),
                    passes: false,
                },
            ],
            truncated: false,
        });
        assert!(drop_features_summary_tail(&mut summary));
        assert_eq!(
            summary.as_ref().map(|s| s.entries.len()),
            Some(1),
            "first drop reduces to 1 entry"
        );
        assert!(drop_features_summary_tail(&mut summary));
        assert!(
            summary.is_none(),
            "after the last entry is shed the field should be cleared"
        );
        assert!(
            !drop_features_summary_tail(&mut summary),
            "no-op once cleared"
        );
    }

    use crate::artifacts::{FeaturesSummary, FeaturesSummaryEntry};
}
