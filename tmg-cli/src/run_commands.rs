//! Implementation of the `tmg run` subcommand family (issue #43).
//!
//! Each command resolves the target [`tmg_harness::Run`] (defaulting to
//! [`tmg_harness::RunStore::current`] when no `run_id` is supplied)
//! and either prints structured output to stdout (`list`, `status`)
//! or mutates `run.toml` via [`tmg_harness::commands`]. The TUI-bound
//! commands (`resume`, `new-session`) delegate back to the regular
//! TUI startup path; the rest can be invoked outside an active TUI
//! session because they only need read/write access to the run store.
//!
//! The list of commands surfaced here is the SPEC §9.8 minimum:
//! Resume / List / Status / Upgrade / Downgrade / Pause / Abort /
//! Shell / `NewSession`. The TUI slash-command equivalents are tracked
//! in #46.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context as _;
use chrono::{DateTime, Utc};
use tmg_harness::{
    RunId, RunRunner, RunScope, RunStatus, RunStore, RunSummary, StatusReport,
    commands as harness_commands,
};

/// Resolve the run id for a CLI command that takes an optional
/// `run_id` argument.
///
/// Resolution order:
/// 1. CLI-supplied id (validated via [`RunId::parse`]).
/// 2. The `current` pointer in the runs-dir, when present **and** its
///    target run's stored `workspace_path` matches the canonical cwd.
///    A mismatch falls through to the next step so a stale pointer
///    from a different project does not silently route the command.
/// 3. [`RunStore::latest_resumable`] for the supplied workspace, when
///    present.
///
/// Returns an error when no run can be resolved or when the
/// CLI-supplied id is malformed.
pub(crate) fn resolve_run_id(
    store: &RunStore,
    explicit: Option<&str>,
    workspace_path: Option<&Path>,
) -> anyhow::Result<RunId> {
    if let Some(raw) = explicit {
        return RunId::parse(raw.to_owned()).with_context(|| format!("invalid run id {raw:?}"));
    }
    if let Some(current) = store.current().context("reading current run pointer")? {
        // Validate that the `current` pointer agrees with the cwd we
        // were launched from. Without this, an out-of-date pointer
        // (e.g. left over from a different project that shares the
        // same `runs_dir`) would silently route mutations into an
        // unrelated run. Mirror the safety net `latest_resumable`
        // already provides.
        let agrees = match (workspace_path, store.load(&current)) {
            (Some(expected), Ok(loaded)) => loaded.workspace_path == expected,
            // Without an expected workspace, accept the pointer as-is
            // (caller did not opt in to the workspace check).
            (None, Ok(_)) => true,
            // Pointer is dangling (load failed). Fall through to the
            // next step so we don't propagate a confusing error from
            // a `current` pointer that should be self-healing.
            (_, Err(e)) => {
                tracing::debug!(
                    run_id = current.as_str(),
                    error = %e,
                    "current pointer is dangling; falling back to latest_resumable",
                );
                false
            }
        };
        if agrees {
            return Ok(current);
        }
    }
    if let Some(summary) = store
        .latest_resumable(workspace_path)
        .context("finding the most recent resumable run")?
    {
        return Ok(summary.id);
    }
    anyhow::bail!(
        "no run id supplied and no current/recent run could be resolved; \
         try `tmg run list` to inspect available runs"
    );
}

/// Print a fixed-width table of all runs to stdout.
///
/// Columns: `ID  SCOPE  STATUS  SESSIONS  PROGRESS  STARTED`. The
/// `PROGRESS` column shows `passing/total` for harnessed runs and
/// `-` for ad-hoc runs.
#[expect(
    clippy::print_stdout,
    reason = "tmg run list prints to stdout by design"
)]
pub(crate) fn cmd_list(store: &RunStore) -> anyhow::Result<()> {
    let summaries = harness_commands::list(store).context("listing runs")?;
    let now = Utc::now();

    if summaries.is_empty() {
        println!("(no runs found)");
        return Ok(());
    }

    let rows: Vec<TableRow> = summaries
        .iter()
        .map(|s| TableRow::from_summary(store, s, now))
        .collect();

    let header = ["ID", "SCOPE", "STATUS", "SESSIONS", "PROGRESS", "STARTED"];
    let mut widths = header.map(str::len);
    for row in &rows {
        let cells = row.cells();
        for (i, cell) in cells.iter().enumerate() {
            widths[i] = widths[i].max(cell.len());
        }
    }

    let mut out = std::io::stdout().lock();
    print_row(&mut out, &header, &widths)?;
    for row in &rows {
        print_row(&mut out, &row.cells(), &widths)?;
    }
    Ok(())
}

/// Pretty-print a [`StatusReport`] for the given run.
#[expect(
    clippy::print_stdout,
    reason = "tmg run status prints to stdout by design"
)]
pub(crate) fn cmd_status(store: &RunStore, run_id: &RunId) -> anyhow::Result<()> {
    let report: StatusReport =
        harness_commands::status(store, run_id, 20).context("building status report")?;

    let run = &report.run;
    let scope_label = run.scope.label();
    let workflow_id = match &run.scope {
        RunScope::Harnessed { workflow_id, .. } => workflow_id.as_str(),
        RunScope::AdHoc => "-",
    };
    let started = humantime_relative(run.created_at, Utc::now());
    let status_label = format_status(&run.status);
    let sessions_cell = format_sessions(run.session_count, run.max_sessions);

    println!("Run: {} ({})", run.id, scope_label);
    println!("  Status:    {status_label}");
    println!("  Workflow:  {workflow_id}");
    println!("  Sessions:  {sessions_cell}");
    println!("  Started:   {started}");
    println!("  Workspace: {}", run.workspace_path.display());

    if let Some(hist) = report.feature_histogram {
        println!(
            "  Features:  {passing}/{total} passing ({failing} remaining)",
            passing = hist.passing,
            total = hist.total,
            failing = hist.failing(),
        );
    }

    if let Some(session) = &report.last_session {
        let ended = session.ended_at.map_or_else(
            || "(active)".to_owned(),
            |t| humantime_relative(t, Utc::now()),
        );
        println!(
            "  Last session #{idx}: {tools} tool calls, {files} files modified, ended {ended}",
            idx = session.index,
            tools = session.tool_calls_count,
            files = session.files_modified.len(),
        );
    }

    if report.progress_tail.is_empty() {
        println!("\nprogress.md: (empty)");
    } else {
        println!("\nprogress.md (last {} lines):", report.progress_tail_lines);
        println!("{}", report.progress_tail);
    }
    Ok(())
}

/// Mutate `run.toml` so the target run is harnessed.
#[expect(
    clippy::print_stdout,
    reason = "tmg run upgrade prints a one-line confirmation"
)]
pub(crate) fn cmd_upgrade(store: &Arc<RunStore>, run_id: &RunId) -> anyhow::Result<()> {
    let run = store.load(run_id).context("loading run")?;
    if matches!(run.scope, RunScope::Harnessed { .. }) {
        println!(
            "Run {} is already harnessed; no action taken.",
            run.id.short()
        );
        return Ok(());
    }
    let mut runner = RunRunner::new(run, Arc::clone(store));
    harness_commands::upgrade(&mut runner).context("upgrading run to harnessed")?;
    println!("Upgraded run {} -> harnessed", runner.run_id().short());
    Ok(())
}

/// Mutate `run.toml` so the target run is back to ad-hoc.
#[expect(
    clippy::print_stdout,
    reason = "tmg run downgrade prints a one-line confirmation"
)]
pub(crate) fn cmd_downgrade(store: &Arc<RunStore>, run_id: &RunId) -> anyhow::Result<()> {
    let run = store.load(run_id).context("loading run")?;
    if matches!(run.scope, RunScope::AdHoc) {
        println!("Run {} is already ad-hoc; no action taken.", run.id.short());
        return Ok(());
    }
    let mut runner = RunRunner::new(run, Arc::clone(store));
    harness_commands::downgrade(&mut runner).context("downgrading run to ad-hoc")?;
    println!("Downgraded run {} -> ad_hoc", runner.run_id().short());
    Ok(())
}

/// Flip the run status to `Paused`.
#[expect(
    clippy::print_stdout,
    reason = "tmg run pause prints a one-line confirmation"
)]
pub(crate) fn cmd_pause(store: &Arc<RunStore>, run_id: &RunId) -> anyhow::Result<()> {
    let run = store.load(run_id).context("loading run")?;
    let mut runner = RunRunner::new(run, Arc::clone(store));
    harness_commands::pause(&mut runner).context("pausing run")?;
    println!("Paused run {}", runner.run_id().short());
    Ok(())
}

/// Flip the run status to `Failed { reason }`.
#[expect(
    clippy::print_stdout,
    reason = "tmg run abort prints a one-line confirmation"
)]
pub(crate) fn cmd_abort(store: &Arc<RunStore>, run_id: &RunId) -> anyhow::Result<()> {
    let run = store.load(run_id).context("loading run")?;
    let mut runner = RunRunner::new(run, Arc::clone(store));
    harness_commands::abort(&mut runner, "user aborted").context("aborting run")?;
    println!("Aborted run {} (status=failed)", runner.run_id().short());
    Ok(())
}

/// Spawn an interactive shell rooted at the current run's workspace.
///
/// The command does NOT apply any sandbox; this is an explicit
/// operator action that should reach the workspace exactly as the
/// user expects.
///
/// **Exit code propagation**: the function exits the calling process
/// with the shell's exit status (or `1` when the shell was killed by
/// a signal and `code()` therefore returns `None`). Callers should
/// treat the function as `-> !`-equivalent on success; we only
/// return `Result` so failures *before* the shell starts (load /
/// missing workspace / spawn) flow through `anyhow` to the
/// top-level handler.
pub(crate) fn cmd_shell(store: &RunStore, run_id: &RunId) -> anyhow::Result<()> {
    let run = store.load(run_id).context("loading run")?;
    let workspace = run.workspace_path.clone();
    if !workspace.exists() {
        anyhow::bail!("workspace path does not exist: {}", workspace.display());
    }
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_owned());
    let status = std::process::Command::new(&shell)
        .current_dir(&workspace)
        .status()
        .with_context(|| format!("spawning interactive shell {shell}"))?;
    // Propagate the shell's exit code so the user's `$?` reflects
    // what they saw inside the shell. `code()` returns `None` when
    // the process was killed by a signal; `1` is the conventional
    // fallback in that case (mirrors `bash` exiting on SIGTERM).
    std::process::exit(status.code().unwrap_or(1));
}

/// Format the SESSIONS column: `count/max` when a max is set,
/// otherwise just `count`.
fn format_sessions(count: u32, max: Option<u32>) -> String {
    match max {
        Some(m) => format!("{count}/{m}"),
        None => format!("{count}"),
    }
}

/// Stringify a [`RunStatus`] for column display.
fn format_status(status: &RunStatus) -> String {
    match status {
        RunStatus::Running => "running".to_owned(),
        RunStatus::Paused => "paused".to_owned(),
        RunStatus::Completed => "completed".to_owned(),
        RunStatus::Exhausted => "exhausted".to_owned(),
        RunStatus::Failed { reason } => format!("failed ({reason})"),
    }
}

/// Format `created_at` as a human-friendly relative time (e.g.
/// `"2 hours ago"`).
fn humantime_relative(then: DateTime<Utc>, now: DateTime<Utc>) -> String {
    let duration = now.signed_duration_since(then);
    if duration.num_seconds() < 0 {
        return "in the future".to_owned();
    }
    let Ok(std_duration) = duration.to_std() else {
        return "(invalid time)".to_owned();
    };
    // Clip to whole seconds so humantime prints `2h 5m` rather than
    // millisecond precision.
    let truncated = std::time::Duration::from_secs(std_duration.as_secs());
    let formatted = humantime::format_duration(truncated).to_string();
    if formatted.is_empty() {
        "just now".to_owned()
    } else {
        // Take the most significant component for compactness:
        // `2h 5m 30s` → `2h ago`. Falls back to the full string when
        // splitting fails for any reason.
        let head = formatted.split_whitespace().next().unwrap_or(&formatted);
        format!("{head} ago")
    }
}

/// One row of the `tmg run list` table.
struct TableRow {
    id: String,
    scope: String,
    status: String,
    sessions: String,
    progress: String,
    started: String,
}

impl TableRow {
    fn from_summary(store: &RunStore, summary: &RunSummary, now: DateTime<Utc>) -> Self {
        // Look up max_sessions / feature counts on demand. Loading the
        // full run is cheap (single TOML file) and rare (only the rows
        // shown by `tmg run list`).
        let (max_sessions, progress, scope_token) = match store.load(&summary.id) {
            Ok(run) => {
                let max = run.max_sessions;
                let progress = match &run.scope {
                    RunScope::Harnessed { .. } => format_progress(store, &summary.id),
                    RunScope::AdHoc => "-".to_owned(),
                };
                let scope_token = match &run.scope {
                    RunScope::AdHoc => "ad_hoc".to_owned(),
                    RunScope::Harnessed { .. } => "harnessed".to_owned(),
                };
                (max, progress, scope_token)
            }
            Err(_) => (None, "-".to_owned(), summary.scope_label.replace('-', "_")),
        };
        Self {
            id: summary.short_id().to_owned(),
            scope: scope_token,
            status: status_token(&summary.status),
            sessions: format_sessions(summary.session_count, max_sessions),
            progress,
            started: humantime_relative(summary.created_at, now),
        }
    }

    fn cells(&self) -> [&str; 6] {
        [
            self.id.as_str(),
            self.scope.as_str(),
            self.status.as_str(),
            self.sessions.as_str(),
            self.progress.as_str(),
            self.started.as_str(),
        ]
    }
}

fn status_token(status: &RunStatus) -> String {
    match status {
        RunStatus::Running => "running".to_owned(),
        RunStatus::Paused => "paused".to_owned(),
        RunStatus::Completed => "completed".to_owned(),
        RunStatus::Exhausted => "exhausted".to_owned(),
        RunStatus::Failed { .. } => "failed".to_owned(),
    }
}

fn format_progress(store: &RunStore, run_id: &RunId) -> String {
    let path = store.features_file(run_id);
    if !path.exists() {
        return "-".to_owned();
    }
    let list = tmg_harness::FeatureList::new(path);
    match list.read() {
        Ok(features) => {
            let total = features.features.len();
            let passing = features.features.iter().filter(|f| f.passes).count();
            format!("{passing}/{total}")
        }
        Err(_) => "-".to_owned(),
    }
}

fn print_row(out: &mut impl Write, cells: &[&str; 6], widths: &[usize; 6]) -> std::io::Result<()> {
    writeln!(
        out,
        "{:<w0$}  {:<w1$}  {:<w2$}  {:<w3$}  {:<w4$}  {:<w5$}",
        cells[0],
        cells[1],
        cells[2],
        cells[3],
        cells[4],
        cells[5],
        w0 = widths[0],
        w1 = widths[1],
        w2 = widths[2],
        w3 = widths[3],
        w4 = widths[4],
        w5 = widths[5],
    )
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tmg_harness::RunStatus;

    fn make_store() -> (tempfile::TempDir, Arc<RunStore>) {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let store = Arc::new(RunStore::new(tmp.path().join("runs")));
        (tmp, store)
    }

    #[test]
    fn resolve_run_id_uses_explicit_first() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        store
            .create_ad_hoc(workspace.clone(), None)
            .unwrap_or_else(|e| panic!("{e}"));

        let resolved = resolve_run_id(&store, Some("abc12345"), Some(&workspace))
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(resolved.as_str(), "abc12345");
    }

    /// `tmg run status ../foo` must surface "invalid run id"; we
    /// never construct a [`RunId`] from path-traversal-shaped input.
    #[test]
    fn resolve_run_id_rejects_malformed_explicit_id() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        match resolve_run_id(&store, Some("../foo"), Some(&workspace)) {
            Err(err) => {
                let msg = format!("{err:#}");
                assert!(
                    msg.contains("invalid run id"),
                    "expected error to mention `invalid run id`, got {msg:?}",
                );
            }
            Ok(id) => panic!(
                "expected error for malformed run id, got Ok({})",
                id.as_str(),
            ),
        }
    }

    /// When the `current` pointer references a run whose workspace
    /// disagrees with the cwd we were launched from, fall through to
    /// `latest_resumable` instead of silently routing the command.
    #[test]
    fn resolve_run_id_falls_through_when_current_workspace_mismatch() {
        let (tmp, store) = make_store();
        let workspace_a = tmp.path().join("workspace-a");
        let workspace_b = tmp.path().join("workspace-b");
        std::fs::create_dir_all(&workspace_a).unwrap_or_else(|e| panic!("{e}"));
        std::fs::create_dir_all(&workspace_b).unwrap_or_else(|e| panic!("{e}"));

        // Create one run for each workspace.
        let run_a = store
            .create_ad_hoc(workspace_a.clone(), None)
            .unwrap_or_else(|e| panic!("{e}"));
        std::thread::sleep(std::time::Duration::from_millis(5));
        let run_b = store
            .create_ad_hoc(workspace_b.clone(), None)
            .unwrap_or_else(|e| panic!("{e}"));

        // After the second create_ad_hoc, `current` points at run_b.
        // From cwd=workspace_a, the resolver should NOT return run_b
        // via the `current` shortcut; instead it should fall through
        // and pick run_a via `latest_resumable`.
        let resolved =
            resolve_run_id(&store, None, Some(&workspace_a)).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(resolved, run_a.id);
        // Sanity: from workspace_b, the same logic returns run_b.
        let resolved_b =
            resolve_run_id(&store, None, Some(&workspace_b)).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(resolved_b, run_b.id);
    }

    #[test]
    fn resolve_run_id_falls_back_to_current() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace.clone(), None)
            .unwrap_or_else(|e| panic!("{e}"));

        let resolved =
            resolve_run_id(&store, None, Some(&workspace)).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(resolved, run.id);
    }

    #[test]
    fn resolve_run_id_errors_when_no_run() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        let err = resolve_run_id(&store, None, Some(&workspace));
        assert!(err.is_err());
    }

    #[test]
    fn cmd_pause_writes_paused_status() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));

        cmd_pause(&store, &run.id).unwrap_or_else(|e| panic!("{e}"));
        let reloaded = store.load(&run.id).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(reloaded.status, RunStatus::Paused);
    }

    #[test]
    fn cmd_abort_writes_failed_status() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));

        cmd_abort(&store, &run.id).unwrap_or_else(|e| panic!("{e}"));
        let reloaded = store.load(&run.id).unwrap_or_else(|e| panic!("{e}"));
        match reloaded.status {
            RunStatus::Failed { reason } => assert_eq!(reason, "user aborted"),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn cmd_upgrade_flips_scope_and_cmd_downgrade_flips_back() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));

        // Plant a fake features.json BEFORE calling cmd_upgrade so
        // the precondition check passes.
        let features = store.features_file(&run.id);
        std::fs::write(
            &features,
            r#"{"features":[{"id":"f","category":"c","description":"d","steps":[],"passes":false}]}"#,
        )
        .unwrap_or_else(|e| panic!("{e}"));

        cmd_upgrade(&store, &run.id).unwrap_or_else(|e| panic!("{e}"));
        let upgraded = store.load(&run.id).unwrap_or_else(|e| panic!("{e}"));
        assert!(matches!(upgraded.scope, RunScope::Harnessed { .. }));

        cmd_downgrade(&store, &run.id).unwrap_or_else(|e| panic!("{e}"));
        let downgraded = store.load(&run.id).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(downgraded.scope, RunScope::AdHoc);
        assert!(features.exists(), "features.json must be preserved");
    }

    /// `cmd_upgrade` surfaces a clear precondition error when
    /// `features.json` is missing. Mirror the harness-level test in
    /// `commands.rs`; this one exercises the CLI wrapper.
    #[test]
    fn cmd_upgrade_without_features_returns_error() {
        let (tmp, store) = make_store();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));

        match cmd_upgrade(&store, &run.id) {
            Err(err) => {
                let msg = format!("{err:#}");
                assert!(
                    msg.contains("features.json not found"),
                    "expected error to mention missing features.json, got {msg:?}",
                );
            }
            Ok(()) => panic!("expected precondition failure but cmd_upgrade succeeded"),
        }
        // Scope must remain ad-hoc after the refused upgrade.
        let reloaded = store.load(&run.id).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(reloaded.scope, RunScope::AdHoc);
    }

    #[test]
    fn humantime_relative_handles_recent() {
        let now = Utc::now();
        let then = now - chrono::Duration::seconds(45);
        let result = humantime_relative(then, now);
        assert!(
            result.ends_with(" ago") || result == "just now",
            "got {result:?}",
        );
    }

    #[test]
    fn format_sessions_includes_max_when_set() {
        assert_eq!(format_sessions(3, None), "3");
        assert_eq!(format_sessions(3, Some(50)), "3/50");
    }

    #[test]
    fn cmd_list_runs_without_panicking_on_empty_store() {
        let (_tmp, store) = make_store();
        // Just exercise the path; output goes to stdout but we only
        // check that no error is returned.
        cmd_list(&store).unwrap_or_else(|e| panic!("{e}"));
    }
}
