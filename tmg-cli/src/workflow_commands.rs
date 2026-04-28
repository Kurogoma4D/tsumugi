//! Implementation of the `tmg workflow` subcommand family (issue #44).
//!
//! The three operations:
//!
//! - `list`: discover and tabulate available workflows.
//! - `validate`: parse every YAML in the discovery scope and report
//!   each failure with `path:line:col` precision.
//! - `run`: load a workflow and execute it via either
//!   [`WorkflowEngine`] or [`LongRunningExecutor`] depending on
//!   `mode:`.
//!
//! All three commands are CI-friendly: structured output to stdout,
//! errors to stderr, non-zero exit codes on failure.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context as _;
use serde_json::Value;
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

use tmg_harness::{RunRunner, RunStore};
use tmg_workflow::{
    LongRunningExecutor, RunStatus, WorkflowDef, WorkflowEngine, WorkflowError, WorkflowMode,
    WorkflowProgress, discover_workflows, parse_workflow_file,
};

use crate::config::{HarnessConfig, TsumugiConfig};

/// Parse a single `--input k=v` pair into `(key, value)`.
///
/// `v` is first parsed as JSON; if that fails the raw string is used.
/// This means CLI users can write `--input flag=true`,
/// `--input count=42`, or `--input note="hello"` and get the natural
/// JSON typing.
pub(crate) fn parse_input_pair(raw: &str) -> anyhow::Result<(String, Value)> {
    let Some((key, value)) = raw.split_once('=') else {
        anyhow::bail!("invalid --input '{raw}': expected key=value form");
    };
    if key.is_empty() {
        anyhow::bail!("invalid --input '{raw}': key must not be empty");
    }
    let parsed: Value =
        serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_owned()));
    Ok((key.to_owned(), parsed))
}

/// `tmg workflow list` — print a fixed-width table of every discovered
/// workflow.
#[expect(
    clippy::print_stdout,
    reason = "tmg workflow list prints a table to stdout by design"
)]
pub(crate) fn cmd_list(config: &TsumugiConfig) -> anyhow::Result<()> {
    let project_root = std::env::current_dir().context("reading current working directory")?;
    let canonical = project_root.canonicalize().unwrap_or(project_root);

    let rt = tokio::runtime::Runtime::new().context("creating tokio runtime")?;
    let metas = rt
        .block_on(discover_workflows(&canonical, &config.workflow))
        .context("discovering workflows")?;

    if metas.is_empty() {
        println!("(no workflows found)");
        println!("Run `tmg init --all` to install built-in workflow templates.");
        return Ok(());
    }

    let header = ["ID", "DESCRIPTION", "SOURCE"];
    let rows: Vec<[String; 3]> = metas
        .iter()
        .map(|m| {
            [
                m.id.clone(),
                m.description.clone().unwrap_or_else(|| "-".to_owned()),
                m.source_path.display().to_string(),
            ]
        })
        .collect();
    let mut widths = header.map(str::len);
    for row in &rows {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(cell.len());
        }
    }

    let mut out = std::io::stdout().lock();
    writeln!(
        out,
        "{:<w0$}  {:<w1$}  {:<w2$}",
        header[0],
        header[1],
        header[2],
        w0 = widths[0],
        w1 = widths[1],
        w2 = widths[2],
    )?;
    for row in &rows {
        writeln!(
            out,
            "{:<w0$}  {:<w1$}  {:<w2$}",
            row[0],
            row[1],
            row[2],
            w0 = widths[0],
            w1 = widths[1],
            w2 = widths[2],
        )?;
    }
    Ok(())
}

/// `tmg workflow validate` — parse every workflow YAML in the
/// discovery scope and report failures.
///
/// On success, prints a one-line summary and returns `Ok(())`. On any
/// failure, prints each diagnostic to stderr (with `path:line:col`
/// when the underlying error carries it) and returns an error so the
/// process exits non-zero.
#[expect(
    clippy::print_stdout,
    reason = "tmg workflow validate prints success summary to stdout"
)]
#[expect(
    clippy::print_stderr,
    reason = "validation diagnostics belong on stderr"
)]
pub(crate) fn cmd_validate(config: &TsumugiConfig) -> anyhow::Result<()> {
    let project_root = std::env::current_dir().context("reading current working directory")?;
    let canonical = project_root.canonicalize().unwrap_or(project_root);

    let rt = tokio::runtime::Runtime::new().context("creating tokio runtime")?;
    let report = rt.block_on(async {
        let metas = discover_workflows(&canonical, &config.workflow).await;
        let metas = match metas {
            Ok(m) => m,
            Err(e) => return ValidationReport::discovery_failed(e),
        };
        let mut report = ValidationReport::default();
        for meta in metas {
            match parse_workflow_file(&meta.source_path).await {
                Ok(_) => report.passed.push(meta.source_path.clone()),
                Err(err) => report.failed.push((meta.source_path.clone(), err)),
            }
        }
        report
    });

    if let Some(err) = report.discovery_error {
        eprintln!("workflow discovery failed: {err}");
        anyhow::bail!("validation aborted: discovery failure");
    }

    if report.passed.is_empty() && report.failed.is_empty() {
        println!("(no workflows found to validate)");
        return Ok(());
    }

    if report.failed.is_empty() {
        println!(
            "validated {n} workflow{s}",
            n = report.passed.len(),
            s = if report.passed.len() == 1 { "" } else { "s" },
        );
        return Ok(());
    }

    eprintln!(
        "{} of {} workflow file(s) failed to validate:",
        report.failed.len(),
        report.passed.len() + report.failed.len(),
    );
    for (path, err) in &report.failed {
        eprintln!("\n--- {} ---", path.display());
        // The parser's YAML errors include line/col automatically via
        // `serde_yml::Error`. Rendering the error chain surfaces the
        // path:line:col context in a stable shape.
        eprintln!("{err}");
    }
    anyhow::bail!("workflow validation failed");
}

/// `tmg workflow run <id> --input k=v ...` — load a workflow and run
/// it through the appropriate executor.
///
/// `inputs_raw` is the unparsed `Vec<String>` produced by clap; we
/// fold it into a `BTreeMap` here so a mistyped pair surfaces with a
/// clean error before any I/O happens.
pub(crate) fn cmd_run(
    workflow_id: &str,
    inputs_raw: &[String],
    config: &TsumugiConfig,
) -> anyhow::Result<()> {
    // 1. Parse the inputs first so we fail fast on bad CLI input.
    let mut inputs: BTreeMap<String, Value> = BTreeMap::new();
    for raw in inputs_raw {
        let (k, v) = parse_input_pair(raw)?;
        if inputs.insert(k.clone(), v).is_some() {
            anyhow::bail!("duplicate --input key '{k}'");
        }
    }

    let project_root = std::env::current_dir().context("reading current working directory")?;
    let canonical = project_root.canonicalize().unwrap_or(project_root);

    let rt = tokio::runtime::Runtime::new().context("creating tokio runtime")?;
    rt.block_on(async move {
        // 2. Discover and locate the named workflow.
        let metas = discover_workflows(&canonical, &config.workflow)
            .await
            .context("discovering workflows")?;
        let Some(meta) = metas.into_iter().find(|m| m.id == workflow_id) else {
            anyhow::bail!(
                "workflow '{workflow_id}' not found. Run `tmg workflow list` to see available workflows.",
            );
        };

        // 3. Parse it into a canonical WorkflowDef.
        let workflow = parse_workflow_file(&meta.source_path)
            .await
            .with_context(|| format!("parsing {}", meta.source_path.display()))?;

        // 4. Validate inputs against the declared InputDef shape.
        validate_inputs_shape(&workflow, &inputs)?;

        // 5. Resolve runs_dir / store for any path the executor needs.
        let runs_dir = resolve_runs_dir(&config.harness, &canonical);
        let store = Arc::new(RunStore::new(runs_dir));

        match workflow.mode {
            WorkflowMode::Normal => run_normal(&workflow, inputs, &canonical, config).await,
            WorkflowMode::LongRunning => {
                run_long_running(&workflow, inputs, &canonical, &store, config).await
            }
            other => anyhow::bail!(
                "unsupported workflow mode for this CLI: {other:?}. \
                 The CLI handler may need an update.",
            ),
        }
    })
}

/// Validate that every required input is present and that no obviously
/// type-incompatible value was supplied for a declared input.
///
/// We accept extras (forward compatibility — see [`tmg_workflow`]'s
/// `resolve_inputs`). Shape checks are intentionally lenient: the
/// `type` field is free-form per SPEC §8.10, so we only enforce the
/// rough "string vs number vs bool vs array vs object vs null" shape
/// for the well-known type names.
fn validate_inputs_shape(
    workflow: &WorkflowDef,
    inputs: &BTreeMap<String, Value>,
) -> anyhow::Result<()> {
    for (name, def) in &workflow.inputs {
        if !inputs.contains_key(name) && def.required && def.default.is_none() {
            anyhow::bail!("missing required --input '{name}'");
        }
        if let Some(value) = inputs.get(name) {
            if !value_matches_type(&def.r#type, value) {
                anyhow::bail!(
                    "--input '{name}={shown}' does not match declared type '{ty}'",
                    shown = render_value(value),
                    ty = def.r#type,
                );
            }
        }
    }
    Ok(())
}

/// Soft type-check helper. Returns `true` for unknown / loose types
/// so workflows can declare semantic types like `"path"` without us
/// rejecting any string.
fn value_matches_type(ty: &str, value: &Value) -> bool {
    match ty {
        "string" => value.is_string(),
        "integer" => value.is_i64() || value.is_u64(),
        "number" | "float" => value.is_number(),
        "boolean" | "bool" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        // Loose types: anything goes.
        _ => true,
    }
}

fn render_value(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Run a `mode: normal` workflow against an ad-hoc engine. This is a
/// CLI-driven path: we don't have a TUI to render `human` steps, so
/// any `human` step that the workflow contains will block until
/// timeout / cancellation. That is acceptable for the SPEC-described
/// CLI workflows but should be flagged in docs.
#[expect(
    clippy::print_stderr,
    reason = "WorkflowProgress events stream to stderr per SPEC §9.13"
)]
async fn run_normal(
    workflow: &WorkflowDef,
    inputs: BTreeMap<String, Value>,
    canonical: &Path,
    config: &TsumugiConfig,
) -> anyhow::Result<()> {
    let llm_pool = Arc::new(
        tmg_llm::LlmPool::new(
            &tmg_llm::PoolConfig::single(&config.llm.endpoint),
            config.llm.model.clone(),
        )
        .context("constructing LlmPool")?,
    );
    let cancel = CancellationToken::new();
    {
        let cancel_for_handler = cancel.clone();
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                cancel_for_handler.cancel();
            }
        });
    }
    let llm_client = tmg_llm::LlmClient::new(tmg_llm::LlmClientConfig::new(
        &config.llm.endpoint,
        &config.llm.model,
    ))
    .context("constructing LlmClient")?;
    let sandbox = Arc::new(tmg_sandbox::SandboxContext::new(
        tmg_sandbox::SandboxConfig::new(canonical)
            .with_mode(tmg_sandbox::SandboxMode::WorkspaceWrite),
    ));
    let subagent_manager = Arc::new(Mutex::new(tmg_agents::SubagentManager::new(
        llm_client,
        cancel.clone(),
        &config.llm.endpoint,
        &config.llm.model,
        Arc::clone(&sandbox),
    )));
    let registry = Arc::new(tmg_tools::ToolRegistry::new());
    let engine = WorkflowEngine::new(
        llm_pool,
        sandbox,
        registry,
        subagent_manager,
        config.workflow.clone(),
        Value::Null,
    );

    let (tx, mut rx) = mpsc::channel::<WorkflowProgress>(64);
    let stream = async move {
        while let Some(ev) = rx.recv().await {
            match ev {
                WorkflowProgress::StepStarted { step_id, step_type } => {
                    eprintln!("[step:start] {step_id} ({step_type})");
                }
                WorkflowProgress::StepCompleted { step_id, .. } => {
                    eprintln!("[step:done]  {step_id}");
                }
                WorkflowProgress::StepFailed { step_id, error } => {
                    eprintln!("[step:fail]  {step_id}: {error}");
                }
                WorkflowProgress::LoopIteration {
                    step_id,
                    iteration,
                    max,
                } => {
                    eprintln!("[loop]       {step_id} iteration {iteration}/{max}");
                }
                WorkflowProgress::HumanInputRequired { step_id, .. } => {
                    eprintln!(
                        "[human]      {step_id}: waiting for human input (CLI cannot answer; will time out)",
                    );
                }
                WorkflowProgress::WorkflowCompleted { outputs } => {
                    eprintln!("[workflow:done] {} output(s)", outputs.values.len());
                }
                _ => {}
            }
        }
    };

    let exec = engine.run_with_cancel(workflow, inputs, tx, cancel.clone());
    let (outcome, ()) = tokio::join!(exec, stream);
    let outputs = outcome.context("workflow execution failed")?;
    eprintln!("\nworkflow '{}' completed", workflow.id);
    if !outputs.values.is_empty() {
        eprintln!("outputs:");
        for (k, v) in &outputs.values {
            eprintln!("  {k} = {v}");
        }
    }
    Ok(())
}

/// Run a `mode: long_running` workflow through [`LongRunningExecutor`].
///
/// We synthesize a transient ad-hoc run (the executor will escalate it
/// to harnessed during the init phase). The chosen workspace is the
/// CLI's current cwd. This is the SPEC-prescribed bootstrap path for
/// `tmg workflow run <long-running-id>` invocations from a fresh
/// project.
#[expect(clippy::print_stderr, reason = "executor diagnostics stream to stderr")]
async fn run_long_running(
    workflow: &WorkflowDef,
    inputs: BTreeMap<String, Value>,
    canonical: &Path,
    store: &Arc<RunStore>,
    config: &TsumugiConfig,
) -> anyhow::Result<()> {
    let cancel = CancellationToken::new();
    {
        let cancel_for_handler = cancel.clone();
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                cancel_for_handler.cancel();
            }
        });
    }
    // Use the latest resumable run for this workspace if one exists,
    // otherwise create a fresh ad-hoc run. This mirrors the resume
    // policy of the TUI startup path.
    let run = match store
        .latest_resumable(Some(canonical))
        .context("looking up latest resumable run")?
    {
        Some(summary) => store.load(&summary.id).context("loading resumable run")?,
        None => store
            .create_ad_hoc(canonical.to_path_buf(), None)
            .context("creating ad-hoc run for long-running workflow")?,
    };
    eprintln!("[long-running] using run {}", run.id.as_str());

    let runner = Arc::new(Mutex::new(RunRunner::new(run, Arc::clone(store))));
    let llm_pool = Arc::new(
        tmg_llm::LlmPool::new(
            &tmg_llm::PoolConfig::single(&config.llm.endpoint),
            config.llm.model.clone(),
        )
        .context("constructing LlmPool")?,
    );
    let llm_client = tmg_llm::LlmClient::new(tmg_llm::LlmClientConfig::new(
        &config.llm.endpoint,
        &config.llm.model,
    ))
    .context("constructing LlmClient")?;
    let sandbox = Arc::new(tmg_sandbox::SandboxContext::new(
        tmg_sandbox::SandboxConfig::new(canonical)
            .with_mode(tmg_sandbox::SandboxMode::WorkspaceWrite),
    ));
    let subagent_manager = Arc::new(Mutex::new(tmg_agents::SubagentManager::new(
        llm_client,
        cancel.clone(),
        &config.llm.endpoint,
        &config.llm.model,
        Arc::clone(&sandbox),
    )));
    let registry = Arc::new(tmg_tools::ToolRegistry::new());
    let engine = Arc::new(WorkflowEngine::new(
        llm_pool,
        sandbox,
        registry,
        subagent_manager,
        config.workflow.clone(),
        Value::Null,
    ));
    let executor = LongRunningExecutor::new(Arc::clone(&engine), Arc::clone(&runner));
    let status = executor
        .run(workflow, inputs)
        .await
        .context("long-running workflow execution failed")?;
    match status {
        RunStatus::Completed => eprintln!("[long-running] workflow completed"),
        RunStatus::Exhausted { reason } => eprintln!("[long-running] exhausted ({reason})"),
        RunStatus::Failed { error } => {
            eprintln!("[long-running] failed: {error}");
            anyhow::bail!("long-running workflow failed");
        }
        other => {
            // The variant is `#[non_exhaustive]`; defensively log
            // unknown future variants rather than panicking.
            eprintln!("[long-running] unknown status: {other:?}");
        }
    }
    Ok(())
}

/// Resolve the runs-dir against the canonical cwd.
fn resolve_runs_dir(harness: &HarnessConfig, canonical: &Path) -> PathBuf {
    if harness.runs_dir.is_absolute() {
        harness.runs_dir.clone()
    } else {
        canonical.join(&harness.runs_dir)
    }
}

/// Aggregated outcome of `cmd_validate`.
#[derive(Default)]
struct ValidationReport {
    passed: Vec<PathBuf>,
    failed: Vec<(PathBuf, WorkflowError)>,
    discovery_error: Option<WorkflowError>,
}

impl ValidationReport {
    fn discovery_failed(err: WorkflowError) -> Self {
        Self {
            passed: Vec::new(),
            failed: Vec::new(),
            discovery_error: Some(err),
        }
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions")]
mod tests {
    use super::*;

    #[test]
    fn parse_input_pair_string() {
        let (k, v) = parse_input_pair("name=hello").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(k, "name");
        assert_eq!(v, Value::String("hello".to_owned()));
    }

    #[test]
    fn parse_input_pair_integer() {
        let (k, v) = parse_input_pair("count=42").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(k, "count");
        assert_eq!(v.as_i64(), Some(42));
    }

    #[test]
    fn parse_input_pair_bool() {
        let (k, v) = parse_input_pair("flag=true").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(k, "flag");
        assert_eq!(v.as_bool(), Some(true));
    }

    #[test]
    fn parse_input_pair_json_object() {
        let (_, v) = parse_input_pair(r#"obj={"a":1}"#).unwrap_or_else(|e| panic!("{e}"));
        assert!(v.is_object());
    }

    #[test]
    fn parse_input_pair_falls_back_to_string_for_invalid_json() {
        let (_, v) = parse_input_pair("note=hello world").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(v, Value::String("hello world".to_owned()));
    }

    #[test]
    fn parse_input_pair_rejects_missing_eq() {
        assert!(parse_input_pair("badpair").is_err());
    }

    #[test]
    fn parse_input_pair_rejects_empty_key() {
        assert!(parse_input_pair("=value").is_err());
    }

    #[test]
    fn value_matches_string_type() {
        assert!(value_matches_type("string", &Value::String("x".to_owned())));
        assert!(!value_matches_type("string", &Value::Bool(true)));
    }

    #[test]
    fn value_matches_integer_type() {
        assert!(value_matches_type("integer", &Value::from(7)));
        assert!(!value_matches_type(
            "integer",
            &Value::String("7".to_owned())
        ));
    }

    #[test]
    fn value_matches_unknown_type_is_lenient() {
        assert!(value_matches_type(
            "path",
            &Value::String("/tmp".to_owned())
        ));
    }

    #[test]
    fn validate_inputs_shape_passes_when_required_supplied() {
        use std::collections::BTreeMap;
        use tmg_workflow::InputDef;
        let mut declared = BTreeMap::new();
        declared.insert(
            "name".to_owned(),
            InputDef {
                r#type: "string".to_owned(),
                default: None,
                required: true,
                description: None,
            },
        );
        let workflow = tmg_workflow::WorkflowDef::new("wf".to_owned()).with_inputs(declared);
        let mut supplied = BTreeMap::new();
        supplied.insert("name".to_owned(), Value::String("alice".to_owned()));
        validate_inputs_shape(&workflow, &supplied).unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn validate_inputs_shape_rejects_missing_required() {
        use std::collections::BTreeMap;
        use tmg_workflow::InputDef;
        let mut declared = BTreeMap::new();
        declared.insert(
            "name".to_owned(),
            InputDef {
                r#type: "string".to_owned(),
                default: None,
                required: true,
                description: None,
            },
        );
        let workflow = tmg_workflow::WorkflowDef::new("wf".to_owned()).with_inputs(declared);
        let supplied: BTreeMap<String, Value> = BTreeMap::new();
        assert!(validate_inputs_shape(&workflow, &supplied).is_err());
    }

    #[test]
    fn validate_inputs_shape_accepts_optional_missing() {
        use std::collections::BTreeMap;
        use tmg_workflow::InputDef;
        let mut declared = BTreeMap::new();
        declared.insert(
            "verbose".to_owned(),
            InputDef {
                r#type: "boolean".to_owned(),
                default: Some(Value::Bool(false)),
                required: false,
                description: None,
            },
        );
        let workflow = tmg_workflow::WorkflowDef::new("wf".to_owned()).with_inputs(declared);
        let supplied: BTreeMap<String, Value> = BTreeMap::new();
        validate_inputs_shape(&workflow, &supplied).unwrap_or_else(|e| panic!("{e}"));
    }
}
