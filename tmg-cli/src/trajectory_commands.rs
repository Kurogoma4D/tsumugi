//! `tmg trajectory ...` subcommand implementations (issue #55).
//!
//! Five operations:
//!
//! - `enable`  — flip `[trajectory] enabled = true` in the project
//!   `tsumugi.toml` (with a one-time consent warning).
//! - `disable` — flip it back to `false`.
//! - `list`    — enumerate trajectory files already on disk.
//! - `export`  — reconstruct trajectories from `session_log/*.json`
//!   plus optional `event_log/*.jsonl`.
//! - `bundle`  — pack every trajectory into a `tar.zst` archive.
//!
//! These commands operate on the project store directly; none of them
//! touch the agent loop.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, anyhow};
use chrono::{DateTime, Utc};
use tmg_harness::RunStore;
use tmg_trajectory::ExportFilter;

use crate::config::TsumugiConfig;

/// Resolve the project root + runs dir the same way the rest of the
/// CLI does. Centralised so the trajectory commands and the live
/// recorder agree on which `.tsumugi/runs/` they target.
fn resolve_paths(config: &TsumugiConfig) -> anyhow::Result<(PathBuf, PathBuf)> {
    let cwd = std::env::current_dir().context("reading current working directory")?;
    let canonical = cwd.canonicalize().unwrap_or_else(|_| cwd.clone());
    let runs_dir = if config.harness.runs_dir.is_absolute() {
        config.harness.runs_dir.clone()
    } else {
        canonical.join(&config.harness.runs_dir)
    };
    Ok((canonical, runs_dir))
}

/// One-time consent message shown by `tmg trajectory enable`.
pub const ENABLE_CONSENT_MESSAGE: &str = "\
WARNING: trajectory recording stores the FULL conversation text and \
code diffs (including system prompts, user input, assistant output, \
tool call arguments, and tool result payloads) under \
`.tsumugi/runs/<run-id>/trajectories/`.\n\
\n\
Default redaction masks API keys / tokens, but other sensitive \
content (file contents, command output, internal URLs) will be \
captured verbatim unless you set `[trajectory] include_tool_results \
= \"summary_only\"` or extend `redact_extra_patterns`.\n\
\n\
Trajectories are intended for SFT/RL training datasets. Do NOT \
share `.tsumugi/runs/<run-id>/trajectories/` outside trusted \
machines without reviewing the contents first.\n";

/// `tmg trajectory enable` — flip the master switch in
/// `.tsumugi/tsumugi.toml`. Prints the consent warning once before
/// proceeding; `--yes` skips the prompt for CI / scripted use.
#[expect(
    clippy::print_stdout,
    reason = "documented stdout surface for CLI subcommands"
)]
pub fn cmd_enable(config_path: Option<&Path>, yes: bool) -> anyhow::Result<()> {
    println!("{ENABLE_CONSENT_MESSAGE}");
    if !yes {
        println!(
            "Re-run with `--yes` to confirm and write `[trajectory] enabled = true` to your tsumugi.toml.",
        );
        return Ok(());
    }
    set_trajectory_enabled(config_path, true)?;
    println!(
        "trajectory recording ENABLED. New runs will write to `.tsumugi/runs/<run-id>/trajectories/`."
    );
    Ok(())
}

/// `tmg trajectory disable` — flip the master switch back to `false`.
#[expect(
    clippy::print_stdout,
    reason = "documented stdout surface for CLI subcommands"
)]
pub fn cmd_disable(config_path: Option<&Path>) -> anyhow::Result<()> {
    set_trajectory_enabled(config_path, false)?;
    println!(
        "trajectory recording DISABLED. Existing files under `.tsumugi/runs/<run-id>/trajectories/` are preserved."
    );
    Ok(())
}

/// Locate (and create if missing) the project `tsumugi.toml`, then
/// rewrite the `[trajectory]` block to set `enabled` to the supplied
/// value. The rest of the file is preserved verbatim — we use
/// `toml_edit` semantics by deserialising into `toml::Value`,
/// patching, and re-serialising. Comments are not preserved (a TODO
/// for `toml_edit`); the project config is small enough that this is
/// acceptable for the first cut.
fn set_trajectory_enabled(config_path: Option<&Path>, value: bool) -> anyhow::Result<()> {
    let path = config_path.map_or_else(default_project_config_path, Path::to_path_buf);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut value_table: toml::Value = if path.exists() {
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?
    } else {
        toml::Value::Table(toml::value::Table::new())
    };
    let toml::Value::Table(table) = &mut value_table else {
        return Err(anyhow!(
            "expected a TOML table at {}; refusing to overwrite a non-table top level",
            path.display(),
        ));
    };
    let traj_entry = table
        .entry("trajectory".to_owned())
        .or_insert_with(|| toml::Value::Table(toml::value::Table::new()));
    let toml::Value::Table(traj_table) = traj_entry else {
        return Err(anyhow!("[trajectory] in {} is not a table", path.display()));
    };
    traj_table.insert("enabled".to_owned(), toml::Value::Boolean(value));
    let serialized =
        toml::to_string_pretty(&value_table).context("serialising updated tsumugi.toml")?;
    std::fs::write(&path, serialized).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn default_project_config_path() -> PathBuf {
    PathBuf::from(".tsumugi").join("tsumugi.toml")
}

/// `tmg trajectory list` — print one line per trajectory file.
#[expect(
    clippy::print_stdout,
    reason = "documented stdout surface for CLI subcommands"
)]
pub fn cmd_list(config: &TsumugiConfig) -> anyhow::Result<()> {
    let (_, runs_dir) = resolve_paths(config)?;
    let entries = tmg_trajectory::list_trajectories(&runs_dir, &config.trajectory)?;
    if entries.is_empty() {
        println!(
            "(no trajectories under {} — enable with `tmg trajectory enable --yes`)",
            runs_dir.display()
        );
        return Ok(());
    }
    println!("{:<10}  {:>6}  {:>10}  PATH", "RUN", "SESSION", "BYTES");
    for e in entries {
        println!(
            "{:<10}  {:>6}  {:>10}  {}",
            e.run_id,
            e.session_num,
            e.size_bytes,
            e.path.display()
        );
    }
    Ok(())
}

/// `tmg trajectory export` — reconstruct trajectories from on-disk
/// `session_log` + (optional) `event_log` artifacts.
#[expect(
    clippy::print_stdout,
    reason = "documented stdout surface for CLI subcommands"
)]
pub fn cmd_export(
    config: &TsumugiConfig,
    run: Option<String>,
    out: Option<&Path>,
    all: bool,
    since: Option<String>,
) -> anyhow::Result<()> {
    let (_, runs_dir) = resolve_paths(config)?;
    let store = RunStore::new(&runs_dir);
    let since_dt: Option<DateTime<Utc>> = match since {
        Some(s) => Some(
            DateTime::parse_from_rfc3339(&s)
                .with_context(|| format!("parsing --since {s:?}"))?
                .with_timezone(&Utc),
        ),
        None => None,
    };
    let filter = ExportFilter {
        run_id: run,
        all,
        since: since_dt,
    };
    let mut effective = config.trajectory.clone();
    // Export should NOT require enabled=true: it is the way you
    // backfill historical sessions. We override locally so the
    // recorder constructed inside the export path opens its files.
    effective.enabled = true;
    let summaries = tmg_trajectory::export(&store, &filter, out, &effective)?;
    if summaries.is_empty() {
        println!("(no runs matched the selection)");
        return Ok(());
    }
    for s in summaries {
        let fallback = if s.had_summary_only_fallback {
            " (some sessions used summary-only fallback)"
        } else {
            ""
        };
        println!(
            "exported {} sessions for run {}{}",
            s.sessions_exported, s.run_id, fallback,
        );
    }
    Ok(())
}

/// `tmg trajectory bundle` — pack every trajectory under the runs
/// dir into one `tar.zst` archive.
#[expect(
    clippy::print_stdout,
    reason = "documented stdout surface for CLI subcommands"
)]
pub fn cmd_bundle(config: &TsumugiConfig, out: &Path, compression: i32) -> anyhow::Result<()> {
    let (_, runs_dir) = resolve_paths(config)?;
    let count = tmg_trajectory::bundle::bundle(&runs_dir, &config.trajectory, out, compression)?;
    println!(
        "bundled {count} session trajectory file(s) → {} (compression level {compression})",
        out.display()
    );
    Ok(())
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn enable_writes_trajectory_block() {
        let tmp = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let cfg = tmp.path().join("tsumugi.toml");
        set_trajectory_enabled(Some(&cfg), true).unwrap_or_else(|e| panic!("{e}"));
        let body = std::fs::read_to_string(&cfg).unwrap_or_else(|e| panic!("{e}"));
        assert!(body.contains("[trajectory]"), "{body}");
        assert!(body.contains("enabled = true"), "{body}");
    }

    #[test]
    fn disable_flips_back() {
        let tmp = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let cfg = tmp.path().join("tsumugi.toml");
        set_trajectory_enabled(Some(&cfg), true).unwrap_or_else(|e| panic!("{e}"));
        set_trajectory_enabled(Some(&cfg), false).unwrap_or_else(|e| panic!("{e}"));
        let body = std::fs::read_to_string(&cfg).unwrap_or_else(|e| panic!("{e}"));
        assert!(body.contains("enabled = false"), "{body}");
    }

    /// Existing TOML content (other sections) must survive the
    /// `enable` round-trip.
    #[test]
    fn enable_preserves_other_sections() {
        let tmp = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let cfg = tmp.path().join("tsumugi.toml");
        std::fs::write(
            &cfg,
            "[llm]\nendpoint = \"http://localhost:8080\"\nmodel = \"foo\"\n",
        )
        .unwrap_or_else(|e| panic!("{e}"));
        set_trajectory_enabled(Some(&cfg), true).unwrap_or_else(|e| panic!("{e}"));
        let body = std::fs::read_to_string(&cfg).unwrap_or_else(|e| panic!("{e}"));
        assert!(body.contains("[llm]"), "{body}");
        assert!(
            body.contains("endpoint = \"http://localhost:8080\""),
            "{body}"
        );
        assert!(body.contains("[trajectory]"), "{body}");
        assert!(body.contains("enabled = true"), "{body}");
    }
}
