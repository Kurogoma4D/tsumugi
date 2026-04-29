use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context as _;
use clap::{Parser, Subcommand};
use tmg_harness::{
    RunRunner, RunRunnerToolProvider, RunStore, RunSummary, SessionBootstrapTool,
    SessionEndTrigger, register_run_tools,
};
use tmg_llm::ToolCallingMode;
use tokio::sync::Mutex;

mod config;
mod error;
mod harness_init;
mod init_command;
mod memory_commands;
mod pool_setup;
mod run_commands;
mod workflow_commands;

use config::{HarnessConfig, SandboxConfigSection, SearchConfig, SkillsSection, TsumugiConfig};

/// tsumugi - a local-LLM-powered coding agent
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Send a one-shot prompt to the LLM server and stream the response
    /// to stdout.
    #[arg(long, global = true)]
    prompt: Option<String>,

    /// LLM server endpoint URL. Overrides config file and environment.
    #[arg(long, global = true)]
    endpoint: Option<String>,

    /// Model name to use. Overrides config file and environment.
    #[arg(long, global = true)]
    model: Option<String>,

    /// Path to a `tsumugi.toml` configuration file.
    /// When specified, only this file is loaded (no global/project-local
    /// discovery).
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    /// Maximum context window tokens.
    #[arg(long, global = true)]
    max_context_tokens: Option<usize>,

    /// Context compression threshold (0.0-1.0). Compression auto-triggers
    /// when context usage exceeds this fraction of `max_context_tokens`.
    #[arg(long, global = true)]
    context_compression_threshold: Option<f64>,

    /// Maximum tokens for a single tool result before truncation.
    #[arg(long, global = true)]
    max_tool_result_tokens: Option<usize>,

    /// Token threshold above which `file_read` tool results are
    /// rewritten via tree-sitter signature extraction (issue #49).
    /// Set to 0 to disable signature-based compression.
    #[arg(long, global = true)]
    signature_threshold_tokens: Option<usize>,

    /// Tool calling mode: "native", "prompt_based", or "auto".
    ///
    /// - native: use OpenAI-compatible function calling API
    /// - prompt_based: inject tool descriptions in system prompt
    /// - auto (default): try native first, fall back to prompt_based parsing
    #[expect(
        clippy::doc_markdown,
        reason = "clap renders doc comments as --help text; backticks would show literally"
    )]
    #[arg(long, global = true)]
    tool_calling: Option<ToolCallingMode>,

    /// Path to write structured event log (JSON Lines format).
    /// Enables diagnostics by recording every agent event (tokens,
    /// tool calls, results) to the specified file.
    #[arg(long, global = true)]
    event_log: Option<PathBuf>,

    /// Top-level subcommand. When omitted, `tmg` launches the
    /// interactive TUI (or runs one-shot mode if `--prompt` was
    /// supplied).
    #[command(subcommand)]
    command: Option<Command>,
}

/// Top-level subcommand surface.
#[derive(Subcommand, Debug)]
enum Command {
    /// Run management (`tmg run <op>`). See SPEC §9.8.
    Run {
        /// Sub-operation on the active or specified run.
        #[command(subcommand)]
        op: RunCommand,
    },
    /// Workflow management (`tmg workflow <op>`). See SPEC §8.13.
    Workflow {
        /// Sub-operation on the workflow catalogue or runtime.
        #[command(subcommand)]
        op: WorkflowCommand,
    },
    /// Scaffold a `.tsumugi/` directory from the built-in templates.
    Init(InitArgs),
    /// Inspect or edit the project memory store (issue #52).
    Memory {
        /// Sub-operation on the memory store.
        #[command(subcommand)]
        op: MemoryCommand,
    },
    /// Cross-session full-text search over indexed session logs (issue #53).
    Search {
        /// Sub-operation on the search index.
        #[command(subcommand)]
        op: SearchCommand,
    },
}

/// Operations under `tmg search`. Issue #53.
#[derive(Subcommand, Debug)]
enum SearchCommand {
    /// Run an FTS5 query against the indexed session log.
    Query {
        /// FTS5 MATCH expression.
        query: String,
        /// Maximum number of hits to return (clamped to 1..=200).
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Restrict the search to one of `summary` / `turns` / `all`.
        #[arg(long, default_value = "all")]
        scope: String,
        /// Optional ISO8601 / RFC3339 lower bound on `started_at`.
        #[arg(long)]
        since: Option<String>,
    },
    /// Re-ingest every `session_NNN.json` under the project's runs
    /// directory that is not yet present in the index.
    Rebuild,
    /// Print DB statistics (size, session / turn counts).
    Stats,
}

/// Operations under `tmg memory`. Issue #52.
#[derive(Subcommand, Debug)]
enum MemoryCommand {
    /// List entry names in the project memory store.
    List,
    /// Print the body of a single memory entry.
    Show {
        /// Topic name (file stem, without `.md`).
        name: String,
    },
    /// Open the named entry in `$EDITOR` for manual editing.
    Edit {
        /// Topic name (file stem). Created on first save.
        name: String,
    },
}

/// Operations under `tmg workflow`. SPEC §8.13.
#[derive(Subcommand, Debug)]
enum WorkflowCommand {
    /// List discovered workflows in a fixed-width table.
    List,
    /// Parse every discovered workflow and report failures.
    ///
    /// Exits non-zero when at least one workflow fails to parse.
    Validate,
    /// Run the named workflow with `--input k=v` pairs.
    Run {
        /// Workflow id (matches the `id:` field in the YAML).
        workflow: String,
        /// Repeatable `--input k=v` pair. The value is parsed as JSON
        /// when it looks like JSON, falling back to a plain string.
        #[arg(long = "input")]
        inputs: Vec<String>,
    },
}

/// Flags for `tmg init`.
#[derive(clap::Args, Debug)]
struct InitArgs {
    /// Workflow templates to install (comma-separated).
    #[arg(long, value_delimiter = ',')]
    workflows: Vec<String>,
    /// Long-running harness templates to install (comma-separated).
    #[arg(long, value_delimiter = ',')]
    harness: Vec<String>,
    /// Install every built-in template.
    #[arg(long)]
    all: bool,
    /// Overwrite existing files when conflicts are detected. Without
    /// this flag, `tmg init` aborts and prints the conflict list.
    #[arg(long)]
    force: bool,
}

/// Operations under `tmg run`. SPEC §9.8.
#[derive(Subcommand, Debug)]
enum RunCommand {
    /// Resume a run in the interactive TUI.
    ///
    /// With no argument, resumes the run pointed at by
    /// `.tsumugi/runs/current` (or the most recent resumable run as a
    /// backwards-compat fallback).
    Resume {
        /// Optional explicit run id.
        run_id: Option<String>,
    },
    /// List all runs as a fixed-width table on stdout.
    List,
    /// Pretty-print detailed status for one run.
    Status {
        /// Optional run id; defaults to current.
        run_id: Option<String>,
    },
    /// Force-promote a run to harnessed scope.
    Upgrade {
        /// Optional run id; defaults to current.
        run_id: Option<String>,
    },
    /// Demote a run back to ad-hoc scope (preserves features.json).
    Downgrade {
        /// Optional run id; defaults to current.
        run_id: Option<String>,
    },
    /// Mark a run as paused without launching the TUI.
    ///
    /// Best-effort outside an active TUI: when a tmg TUI is currently
    /// attached to the run (`.tsumugi/runs/<id>/.tui-pid` sentinel),
    /// the command refuses with an error. Use the in-TUI `/run pause`
    /// slash command (issue #46) to pause an attached run.
    Pause {
        /// Optional run id; defaults to current.
        run_id: Option<String>,
    },
    /// Mark a run as failed (`reason = "user aborted"`).
    ///
    /// Best-effort outside an active TUI: refuses while a tmg TUI is
    /// attached to the run, for the same reason as `pause`.
    Abort {
        /// Optional run id; defaults to current.
        run_id: Option<String>,
    },
    /// Open an interactive shell rooted at the run's workspace path.
    Shell {
        /// Optional run id; defaults to current.
        run_id: Option<String>,
    },
    /// Force-rotate to a new session. Must be issued from inside an
    /// active TUI session; outside, prints an explanatory error.
    NewSession,
}

impl Cli {
    /// Apply CLI overrides to the loaded configuration.
    ///
    /// CLI options have the highest priority in the merge chain.
    fn apply_to(&self, config: &mut TsumugiConfig) {
        if let Some(ref endpoint) = self.endpoint {
            config.llm.endpoint.clone_from(endpoint);
        }
        if let Some(ref model) = self.model {
            config.llm.model.clone_from(model);
        }
        if let Some(max_ctx) = self.max_context_tokens {
            config.llm.max_context_tokens = max_ctx;
        }
        if let Some(threshold) = self.context_compression_threshold {
            config.llm.compression_threshold = threshold;
        }
        if let Some(max_tool) = self.max_tool_result_tokens {
            config.llm.max_tool_result_tokens = max_tool;
        }
        if let Some(sig_threshold) = self.signature_threshold_tokens {
            config.llm.signature_threshold_tokens = sig_threshold;
        }
        if let Some(tc) = self.tool_calling {
            config.llm.tool_calling = tc;
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialise the global tracing subscriber so workspace-wide
    // `tracing::warn!` / `tracing::debug!` calls land in the user's
    // terminal when `TMG_LOG` is set. The default filter is `warn` so
    // production runs only see escalation-grade messages; setting
    // `TMG_LOG=debug` (or finer-grained `TMG_LOG=tmg_llm=debug,warn`)
    // surfaces endpoint-resolution and pool-selection diagnostics.
    //
    // The TUI captures stderr indirectly via crossterm; in TUI mode the
    // tracing output stays out of the way until the TUI exits, at
    // which point it is flushed normally. One-shot mode renders the
    // logs interleaved with the streamed response (acceptable noise
    // for a debug session).
    init_tracing();

    // Load and merge configuration: global -> project-local -> env -> CLI.
    let mut config =
        config::load_config(cli.config.as_deref()).context("loading tsumugi configuration")?;
    cli.apply_to(&mut config);
    config.validate().context("validating configuration")?;

    let context_config = tmg_core::ContextConfig {
        max_context_tokens: config.llm.max_context_tokens,
        compression_threshold: config.llm.compression_threshold,
        max_tool_result_tokens: config.llm.max_tool_result_tokens,
        signature_threshold_tokens: config.llm.signature_threshold_tokens,
    };

    match cli.command {
        Some(Command::Run { op }) => dispatch_run_command(op, &config),
        Some(Command::Workflow { op }) => {
            dispatch_workflow_command(op, &config, cli.event_log.as_deref())
        }
        Some(Command::Init(args)) => {
            init_command::cmd_init(&args.workflows, &args.harness, args.all, args.force)
        }
        Some(Command::Memory { op }) => {
            dispatch_memory_command(op, &config, cli.event_log.as_deref())
        }
        Some(Command::Search { op }) => dispatch_search_command(op, &config),
        None => {
            if let Some(prompt) = cli.prompt {
                run_prompt(
                    &config.llm.endpoint,
                    &config.llm.model,
                    &prompt,
                    cli.event_log.as_deref(),
                    &config.memory,
                    &config.search,
                    &config.sandbox,
                    context_config,
                    config.llm.tool_calling,
                )
            } else {
                run_tui(
                    &config.llm.endpoint,
                    &config.llm.model,
                    context_config,
                    config.llm.tool_calling,
                    cli.event_log,
                    &config.harness,
                    &config.sandbox,
                    &config.workflow,
                    &config.skills,
                    &config.memory,
                    &config.search,
                    config.llm.subagent_pool.as_ref(),
                    None,
                )
            }
        }
    }
}

// `subagent_pool_from_config` lives in `crate::pool_setup` so the
// workflow command path (`tmg workflow run`) can reuse it.

/// Initialise the global `tracing` subscriber.
///
/// The subscriber is built from the `TMG_LOG` environment variable
/// (using the standard `EnvFilter` syntax: `TMG_LOG=debug` for global
/// debug, `TMG_LOG=tmg_llm=debug,warn` for crate-scoped overrides).
/// When `TMG_LOG` is unset the filter defaults to `warn`, which keeps
/// the live TUI quiet. Issue #50.
///
/// Calling this twice is harmless: the second `try_init` is a no-op
/// because the global default is already installed, so the function
/// silently absorbs the error. This matters for tests that may build
/// a `Cli` and reuse the runtime.
fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_env("TMG_LOG").unwrap_or_else(|_| EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false)
        .try_init();
}

/// Resolve the runs-dir against the canonical cwd, mirroring the
/// logic used by [`run_tui`] so the CLI surfaces and the live TUI
/// agree on which `.tsumugi/runs/` they target.
fn resolve_runs_dir(harness_config: &HarnessConfig) -> anyhow::Result<(PathBuf, PathBuf)> {
    let cwd = std::env::current_dir().context("reading current working directory")?;
    let canonical_cwd = cwd.canonicalize().unwrap_or_else(|_| cwd.clone());
    let runs_dir = if harness_config.runs_dir.is_absolute() {
        harness_config.runs_dir.clone()
    } else {
        canonical_cwd.join(&harness_config.runs_dir)
    };
    Ok((runs_dir, canonical_cwd))
}

/// Dispatch one `tmg memory <op>` invocation.
fn dispatch_memory_command(
    op: MemoryCommand,
    config: &TsumugiConfig,
    event_log: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    memory_commands::dispatch(op, config, event_log)
}

/// Dispatch one `tmg search <op>` invocation. Issue #53.
fn dispatch_search_command(op: SearchCommand, config: &TsumugiConfig) -> anyhow::Result<()> {
    let cwd = std::env::current_dir().context("reading current working directory")?;
    let canonical_cwd = cwd.canonicalize().unwrap_or_else(|_| cwd.clone());
    if !config.search.enabled {
        anyhow::bail!("[search] is disabled in tsumugi.toml; enable it before running tmg search");
    }
    let db_path = config.search.resolve_db_path(&canonical_cwd);
    let index = tmg_search::SearchIndex::open(&db_path)
        .with_context(|| format!("opening search index at {}", db_path.display()))?;

    match op {
        SearchCommand::Query {
            query,
            limit,
            scope,
            since,
        } => search_commands_query(&index, &query, limit, &scope, since.as_deref()),
        SearchCommand::Rebuild => search_commands_rebuild(&index, config, &canonical_cwd),
        SearchCommand::Stats => search_commands_stats(&index),
    }
}

#[expect(
    clippy::print_stdout,
    reason = "stdout is the contract for `tmg search` CLI output"
)]
fn search_commands_query(
    index: &tmg_search::SearchIndex,
    query: &str,
    limit: usize,
    scope: &str,
    since: Option<&str>,
) -> anyhow::Result<()> {
    let scope = tmg_search::SearchScope::parse(scope);
    let since_dt = match since {
        Some(s) => Some(
            chrono::DateTime::parse_from_rfc3339(s)
                .with_context(|| format!("parsing --since {s:?}"))?
                .with_timezone(&chrono::Utc),
        ),
        None => None,
    };
    let hits = index
        .query(query, limit, scope, since_dt)
        .context("running search query")?;
    if hits.is_empty() {
        println!("(no matches for query {query:?})");
        return Ok(());
    }
    for hit in hits {
        println!(
            "[{:.3}] {}#{}  {}\n  summary: {}\n  snippet: {}\n",
            hit.score, hit.run_id, hit.session_num, hit.started_at, hit.summary, hit.snippet,
        );
    }
    Ok(())
}

#[expect(
    clippy::print_stdout,
    reason = "stdout is the contract for `tmg search` CLI output"
)]
fn search_commands_rebuild(
    index: &tmg_search::SearchIndex,
    config: &TsumugiConfig,
    cwd: &std::path::Path,
) -> anyhow::Result<()> {
    let runs_dir = if config.harness.runs_dir.is_absolute() {
        config.harness.runs_dir.clone()
    } else {
        cwd.join(&config.harness.runs_dir)
    };
    let count = index
        .rebuild_from_disk(&runs_dir)
        .context("rebuilding search index from disk")?;
    println!(
        "ingested {count} new session(s) from {}",
        runs_dir.display()
    );
    Ok(())
}

#[expect(
    clippy::print_stdout,
    reason = "stdout is the contract for `tmg search` CLI output"
)]
fn search_commands_stats(index: &tmg_search::SearchIndex) -> anyhow::Result<()> {
    let stats = index.stats().context("reading search index stats")?;
    println!("db_path:  {}", stats.db_path.display());
    println!("size:     {} bytes", stats.size_bytes);
    println!("sessions: {}", stats.sessions);
    println!("turns:    {}", stats.turns);
    Ok(())
}

// `load_memory_index_for_prompt` was the bespoke helper used by the
// pre-AgentLoop one-shot path. Issue #2 of PR #76 review moved
// one-shot mode through the full agent loop, which already injects
// the index via `AgentLoop::inject_memory_index_now`, so the helper
// is no longer needed.

/// Dispatch one `tmg workflow <op>` invocation. None of these
/// operations attach to an active TUI; they all read or run workflows
/// against the configured discovery scope.
///
/// `event_log` is threaded only into `Run` because `List` and
/// `Validate` do not spawn subagents; the resolver / pool / tokenize
/// hooks are no-ops outside the live engine path.
fn dispatch_workflow_command(
    op: WorkflowCommand,
    config: &TsumugiConfig,
    event_log: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    match op {
        WorkflowCommand::List => workflow_commands::cmd_list(config),
        WorkflowCommand::Validate => workflow_commands::cmd_validate(config),
        WorkflowCommand::Run { workflow, inputs } => {
            workflow_commands::cmd_run(&workflow, &inputs, config, event_log)
        }
    }
}

/// Dispatch one `tmg run <op>` invocation. The TUI-bound operations
/// (`Resume`, `NewSession`) delegate back into [`run_tui`] (or print
/// an explanatory error). The rest mutate `run.toml` directly.
fn dispatch_run_command(op: RunCommand, config: &TsumugiConfig) -> anyhow::Result<()> {
    let (runs_dir, canonical_cwd) = resolve_runs_dir(&config.harness)?;
    let store = Arc::new(RunStore::new(runs_dir));

    match op {
        RunCommand::Resume { run_id } => {
            let context_config = tmg_core::ContextConfig {
                max_context_tokens: config.llm.max_context_tokens,
                compression_threshold: config.llm.compression_threshold,
                max_tool_result_tokens: config.llm.max_tool_result_tokens,
                signature_threshold_tokens: config.llm.signature_threshold_tokens,
            };
            // When the user supplied an explicit id, validate its
            // shape at the boundary. We thread the validated id
            // through to `run_tui` so the TUI loads exactly that run
            // (bypassing `harness_init::resolve_startup_run`'s
            // latest-resumable fallback). If no id was supplied, we
            // try `current` as a hint and fall back to the resolver.
            let resolved = if let Some(id) = run_id {
                Some(
                    tmg_harness::RunId::parse(id)
                        .context("parsing explicit run id for `tmg run resume`")?,
                )
            } else {
                // Best-effort: prefer `current` so subsequent
                // argument-less commands inherit the same target.
                store.current().context("reading current run pointer")?
            };
            // If we resolved a run, point `current` at it before
            // launching so the TUI shutdown / re-open paths agree on
            // the active id.
            if let Some(ref id) = resolved
                && let Err(e) = store.set_current(id)
            {
                tracing::warn!(error = %e, "failed to update current run pointer");
            }
            run_tui(
                &config.llm.endpoint,
                &config.llm.model,
                context_config,
                config.llm.tool_calling,
                None,
                &config.harness,
                &config.sandbox,
                &config.workflow,
                &config.skills,
                &config.memory,
                &config.search,
                config.llm.subagent_pool.as_ref(),
                resolved.as_ref(),
            )
        }
        RunCommand::List => run_commands::cmd_list(&store),
        RunCommand::Status { run_id } => {
            let id = run_commands::resolve_run_id(&store, run_id.as_deref(), Some(&canonical_cwd))?;
            run_commands::cmd_status(&store, &id)
        }
        RunCommand::Upgrade { run_id } => {
            let id = run_commands::resolve_run_id(&store, run_id.as_deref(), Some(&canonical_cwd))?;
            run_commands::cmd_upgrade(&store, &id)
        }
        RunCommand::Downgrade { run_id } => {
            let id = run_commands::resolve_run_id(&store, run_id.as_deref(), Some(&canonical_cwd))?;
            run_commands::cmd_downgrade(&store, &id)
        }
        RunCommand::Pause { run_id } => {
            let id = run_commands::resolve_run_id(&store, run_id.as_deref(), Some(&canonical_cwd))?;
            run_commands::cmd_pause(&store, &id)
        }
        RunCommand::Abort { run_id } => {
            let id = run_commands::resolve_run_id(&store, run_id.as_deref(), Some(&canonical_cwd))?;
            run_commands::cmd_abort(&store, &id)
        }
        RunCommand::Shell { run_id } => {
            let id = run_commands::resolve_run_id(&store, run_id.as_deref(), Some(&canonical_cwd))?;
            run_commands::cmd_shell(&store, &id)
        }
        RunCommand::NewSession => {
            // `new-session` rotates the *active* TUI session. Outside
            // a TUI we have no live runner, so we surface an
            // explanatory error rather than silently writing the
            // on-disk session. The TUI slash-command equivalent
            // (`/run new-session`) is tracked in #46. The single
            // `bail!` carries the entire message so anyhow's error
            // chain is the sole source of truth.
            anyhow::bail!(
                "tmg run new-session must be invoked from inside an active TUI session. \
                 Use the `/run new-session` slash command from within `tmg` (see #46).",
            );
        }
    }
}

/// Run a one-shot streaming prompt against the LLM server.
///
/// One-shot mode runs through the full [`tmg_core::AgentLoop`] (issue
/// #2 of PR #76 review) so memory `add` / `read` and the standard
/// file/grep tool surface are available the same way they are in the
/// TUI. Output is streamed to stdout via a custom [`StdoutSink`].
#[expect(
    clippy::too_many_arguments,
    reason = "linear setup helper takes the merged config sections as discrete refs"
)]
fn run_prompt(
    endpoint: &str,
    model: &str,
    prompt: &str,
    event_log: Option<&std::path::Path>,
    memory_config: &config::MemoryConfig,
    search_config: &SearchConfig,
    sandbox_config: &SandboxConfigSection,
    context_config: tmg_core::ContextConfig,
    tool_calling_mode: tmg_llm::ToolCallingMode,
) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let config = tmg_llm::LlmClientConfig::new(endpoint, model);
        let client = tmg_llm::LlmClient::new(config)?;

        let cancel = tokio_util::sync::CancellationToken::new();
        let cwd = std::env::current_dir()?;
        let canonical_cwd = cwd.canonicalize().unwrap_or_else(|_| cwd.clone());
        let project_root = canonical_cwd.clone();

        // Build the project memory store from the same `[memory]`
        // config the TUI consumes, so one-shot mode and TUI agree on
        // the index payload and the directory layout.
        let memory_store = Arc::new(tmg_memory::MemoryStore::with_dirs(
            memory_config.resolve_project_dir(&canonical_cwd),
            memory_config.resolve_global_dir(),
            memory_config.to_budget(),
        ));

        // Build the sandbox context from the operator's `[sandbox]`
        // config. We don't `activate()` it because one-shot mode is
        // expected to be the user's interactive integration smoke
        // test; activating would require platform-specific privileges
        // (Landlock, netns) that the user might not have.
        let sandbox_ctx = Arc::new(tmg_sandbox::SandboxContext::new(
            sandbox_config.to_sandbox_config(&canonical_cwd),
        ));

        // Issue #53: open the cross-session search index for read-only
        // use in one-shot mode. The DB is **never** written to from
        // here — one-shot mode does not construct a `RunRunner`, so no
        // session-end hook is registered and `ingest_session` is never
        // called. The agent gets a search tool over whatever the TUI /
        // workflow paths previously indexed.
        let search_index_for_oneshot: Option<Arc<tmg_search::SearchIndex>> = if search_config
            .enabled
        {
            let db_path = search_config.resolve_db_path(&canonical_cwd);
            match tmg_search::SearchIndex::open(&db_path) {
                Ok(idx) => Some(Arc::new(idx)),
                Err(e) => {
                    tracing::warn!(
                        path = %db_path.display(),
                        error = %e,
                        "failed to open search index for one-shot prompt; tool returns soft errors"
                    );
                    None
                }
            }
        } else {
            None
        };

        // Build the tool registry: built-ins + (optionally) the memory
        // tool. No subagent registration in one-shot mode because
        // there's no Run/Workflow/Subagent context here; the goal is
        // a self-contained tool surface that lets the model curate
        // memory and read/write files.
        let mut registry = tmg_tools::default_registry();
        if memory_config.enabled {
            registry.register(tmg_memory::MemoryTool::new(Arc::clone(&memory_store)));
        }
        // Issue #53: register the cross-session search tool in
        // one-shot mode so the agent can still query historical
        // sessions even though no new sessions are being ingested
        // (one-shot mode never constructs a RunRunner). Index opening
        // is best-effort — a failure surfaces as a soft tool error.
        if search_config.enabled
            && let Some(idx) = search_index_for_oneshot.as_ref()
        {
            registry.register(tmg_search::SessionSearchTool::new(Arc::clone(idx)));
        } else if search_config.enabled {
            registry.register(tmg_search::SessionSearchTool::disabled());
        }

        let mut agent = tmg_core::AgentLoop::with_context_config(
            client,
            registry,
            cancel.clone(),
            &project_root,
            &canonical_cwd,
            context_config,
            tool_calling_mode,
            Arc::clone(&sandbox_ctx),
        )?;

        // Inject the memory index into the system prompt the same way
        // the TUI does so the model sees the curated `MEMORY.md` rows
        // before its first reply.
        if memory_config.enabled {
            match memory_store.read_merged_index() {
                Ok(index) if !index.trim().is_empty() => {
                    let report = memory_store.budget_report().ok();
                    let mut payload = index.trim().to_owned();
                    if let Some(r) = report
                        && let Some(nudge) = tmg_memory::capacity_nudge(&r, memory_store.budget())
                    {
                        payload.push_str(&nudge);
                    }
                    let summary = format!(
                        "merged index: {} chars / {} lines",
                        payload.len(),
                        payload.lines().count(),
                    );
                    agent.set_memory_index(Some(payload));
                    agent.inject_memory_index_now();
                    if let Some(path) = event_log
                        && let Ok(mut writer) = tmg_core::EventLogWriter::open_append(path)
                    {
                        writer.write_memory("memory_prompt_inject", &summary);
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "failed to read memory index for one-shot prompt");
                }
            }
        }

        // The agent's `turn` method takes `&mut impl StreamSink` which
        // is `Sized`, so we cannot use `Box<dyn StreamSink>` here.
        // Branch on the event-log presence and call `turn` separately
        // in each arm so the sink type is statically known.
        if let Some(path) = event_log {
            let log = tmg_core::EventLogWriter::new(path).context("opening event log file")?;
            let mut sink = tmg_core::TeeStreamSink::new(StdoutSink::default(), log);
            agent
                .turn(prompt, &mut sink)
                .await
                .context("running one-shot agent turn")?;
        } else {
            let mut sink = StdoutSink::default();
            agent
                .turn(prompt, &mut sink)
                .await
                .context("running one-shot agent turn")?;
        }

        Ok::<(), anyhow::Error>(())
    })
}

/// One-shot stdout sink that mirrors [`StreamEvent`] output to the
/// terminal. Tool calls and warnings are surfaced as bracketed lines
/// so the user can see what the agent is doing without the TUI.
#[derive(Default)]
struct StdoutSink {
    /// True while the model is still emitting content tokens. Used to
    /// add a newline before printing tool-call markers / warnings so
    /// they don't fuse onto the same line as the streamed text.
    streaming_text: bool,
}

impl tmg_core::agent_loop::StreamSink for StdoutSink {
    #[expect(
        clippy::print_stdout,
        reason = "stdout is the contract for one-shot mode"
    )]
    fn on_token(&mut self, token: &str) -> Result<(), tmg_core::CoreError> {
        use std::io::Write as _;
        self.streaming_text = true;
        print!("{token}");
        let _ = std::io::stdout().flush();
        Ok(())
    }

    #[expect(
        clippy::print_stdout,
        reason = "stdout is the contract for one-shot mode"
    )]
    fn on_done(&mut self) -> Result<(), tmg_core::CoreError> {
        if self.streaming_text {
            println!();
            self.streaming_text = false;
        }
        Ok(())
    }

    #[expect(
        clippy::print_stdout,
        reason = "stdout is the contract for one-shot mode"
    )]
    fn on_tool_call(
        &mut self,
        _call_id: &str,
        name: &str,
        arguments: &str,
    ) -> Result<(), tmg_core::CoreError> {
        if self.streaming_text {
            println!();
            self.streaming_text = false;
        }
        println!("[tool_call] {name}({arguments})");
        Ok(())
    }

    #[expect(
        clippy::print_stdout,
        reason = "stdout is the contract for one-shot mode"
    )]
    fn on_tool_result(
        &mut self,
        _call_id: &str,
        name: &str,
        output: &str,
        is_error: bool,
    ) -> Result<(), tmg_core::CoreError> {
        let tag = if is_error {
            "tool_error"
        } else {
            "tool_result"
        };
        // Truncate noisy tool output to one line so the terminal stays
        // readable; the full payload is in the event log when one is
        // configured.
        let preview: String = output.chars().take(160).collect();
        let suffix = if output.len() > preview.len() {
            "…"
        } else {
            ""
        };
        println!("[{tag}] {name}: {preview}{suffix}");
        Ok(())
    }

    #[expect(
        clippy::print_stdout,
        reason = "stdout is the contract for one-shot mode"
    )]
    fn on_warning(&mut self, message: &str) -> Result<(), tmg_core::CoreError> {
        if self.streaming_text {
            println!();
            self.streaming_text = false;
        }
        println!("[warning] {message}");
        Ok(())
    }
}

/// Concrete [`tmg_agents::MemoryToolProvider`] used by the CLI to
/// register a memory tool into every spawned subagent's registry.
/// Issue #3 of PR #76 review: previously the memory tool was wired
/// only into the top-level agent's registry, so subagents could not
/// curate the project memory.
///
/// Each invocation constructs a fresh [`tmg_memory::MemoryTool`] that
/// shares the parent's [`MemoryStore`] via `Arc` and the same
/// pending-swap slot. When a subagent successfully writes through
/// `memory`, the parent agent loop's next `turn` picks up the new
/// snapshot just as if the parent had performed the write itself.
struct CliMemoryToolProvider {
    store: Arc<tmg_memory::MemoryStore>,
    pending: Option<Arc<std::sync::Mutex<Option<String>>>>,
}

impl tmg_agents::MemoryToolProvider for CliMemoryToolProvider {
    fn register_memory_tool(&self, registry: &mut tmg_tools::ToolRegistry) {
        if let Some(slot) = self.pending.clone() {
            let store_for_cb = Arc::clone(&self.store);
            let cb: tmg_memory::tool::MemoryRefreshCallback = Arc::new(move |_index: &str| {
                let snapshot = match store_for_cb.read_merged_index() {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::debug!(
                            error = %e,
                            "subagent memory refresh: read_merged_index failed",
                        );
                        return;
                    }
                };
                if let Ok(mut guard) = slot.lock() {
                    *guard = Some(snapshot);
                }
            });
            registry.register(tmg_memory::MemoryTool::with_refresh(
                Arc::clone(&self.store),
                cb,
            ));
        } else {
            // No shared slot configured (memory was disabled for the
            // top-level loop); register without a refresh hook so
            // the subagent still has read access.
            registry.register(tmg_memory::MemoryTool::new(Arc::clone(&self.store)));
        }
    }
}

/// Run the TUI-based interactive session.
///
/// Launches a ratatui terminal interface with a chat pane, header,
/// and input area. The TUI handles multi-turn conversations with
/// streaming LLM responses. Includes subagent support via
/// `spawn_agent` tool, with custom agent definitions from TOML.
///
/// Initialises a [`RunStore`] and resolves the active [`Run`] following
/// the policy described in [`harness_init::resolve_startup_run`]: when
/// `harness.auto_resume_on_start` is `true` and a resumable run exists,
/// the most recent one is loaded; otherwise a fresh ad-hoc run is
/// created. The run is wrapped in a [`RunRunner`] which begins one
/// session that lives for the duration of the TUI; the session is ended
/// with a normal-completion trigger when the TUI returns and aborted
/// with `UserCancelled` if the cancellation token fired or the TUI
/// returned an error.
#[expect(
    clippy::too_many_lines,
    reason = "linear startup wiring (config -> store -> manager -> registry -> agent loop); splitting obscures the single-shot launch sequence"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "startup helper takes the merged config sections as discrete refs; bundling into a struct would need the same fields"
)]
fn run_tui(
    endpoint: &str,
    model: &str,
    context_config: tmg_core::ContextConfig,
    tool_calling_mode: tmg_core::ToolCallingMode,
    event_log: Option<PathBuf>,
    harness_config: &HarnessConfig,
    sandbox_config: &SandboxConfigSection,
    workflow_config: &tmg_workflow::WorkflowConfig,
    skills_section: &SkillsSection,
    memory_config: &config::MemoryConfig,
    search_config: &SearchConfig,
    subagent_pool: Option<&tmg_llm::PoolConfig>,
    explicit_run_id: Option<&tmg_harness::RunId>,
) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let config = tmg_llm::LlmClientConfig::new(endpoint, model);
        let client = tmg_llm::LlmClient::new(config)?;

        let cancel = tokio_util::sync::CancellationToken::new();

        // Set up Ctrl-C handler for graceful shutdown.
        let cancel_for_handler = cancel.clone();
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                cancel_for_handler.cancel();
            }
        });

        let cwd = std::env::current_dir()?;
        // Canonicalise so the persisted `workspace_path` and the
        // `workspace` symlink target are absolute and stable across
        // shells that may have entered via a symlinked path. Falls
        // back to the raw cwd if canonicalisation fails (e.g. on
        // unusual filesystems).
        let canonical_cwd = cwd.canonicalize().unwrap_or_else(|_| cwd.clone());
        // Use canonical cwd as project root for now (could be improved with git root detection).
        let project_root = canonical_cwd.clone();

        // Resolve `runs_dir`: relative paths are interpreted against cwd
        // so the on-disk layout matches `.tsumugi/runs/<run-id>` for the
        // current project.
        let runs_dir = if harness_config.runs_dir.is_absolute() {
            harness_config.runs_dir.clone()
        } else {
            canonical_cwd.join(&harness_config.runs_dir)
        };
        let store = Arc::new(RunStore::new(runs_dir));
        // `select_startup_run` honours an explicit id from
        // `tmg run resume <id>` before falling back to the
        // auto-resume / create-fresh policy. The explicit-id path
        // bypasses `latest_resumable` so the user genuinely resumes
        // the run they typed, not "whatever was newest".
        let run = harness_init::select_startup_run(
            harness_config,
            &store,
            canonical_cwd.clone(),
            explicit_run_id,
        )
        .context("resolving startup run")?;
        // Whichever path we took, lock the resolved id into `current`
        // so out-of-TUI CLI mutators agree with the live runner.
        if let Err(e) = store.set_current(&run.id) {
            tracing::warn!(error = %e, "failed to update current run pointer at startup");
        }
        // Stamp the TUI sentinel so `tmg run pause` / `tmg run abort`
        // refuse to mutate `run.toml` while we hold the live runner.
        // Failures here are best-effort (a permission glitch on the
        // run dir should not abort startup); the worst case is the
        // CLI mutators not refusing.
        let sentinel_dir = store.run_dir(&run.id);
        if let Err(e) = tmg_harness::tui_sentinel::write(&sentinel_dir) {
            tracing::warn!(error = %e, "failed to write TUI sentinel; CLI mutators may race the live runner");
        }
        // Guard ensures the sentinel is removed on every exit path
        // (including panic propagation through tokio::block_on).
        let _sentinel_guard = TuiSentinelGuard::new(sentinel_dir);
        let mut runner = RunRunner::new(run, Arc::clone(&store));
        runner.set_bootstrap_max_tokens(harness_config.bootstrap_max_tokens);
        runner.set_default_session_timeout(harness_config.default_session_timeout);
        runner.set_session_log_compress_after(harness_config.session_log_compress_after);
        // The runner stores the threshold as `f32` (matching the
        // SessionState observers); narrow once at the boundary.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "threshold is in (0.0, 1.0]; precision loss bounded by f32 epsilon, acceptable for a coarse threshold check."
        )]
        runner.set_context_force_rotate_threshold(
            harness_config.context_force_rotate_threshold as f32,
        );
        // One-time warning when the user's `[sandbox] mode` is stricter
        // than the harnessed `init.sh` execution path honours; see
        // `harness_init::warn_if_sandbox_mode_mismatch` for details.
        if matches!(runner.scope(), tmg_harness::RunScope::Harnessed { .. }) {
            harness_init::warn_if_sandbox_mode_mismatch(sandbox_config);
        }

        // Per-session timeout channel: the watchdog feeds
        // `SessionEndTrigger::Timeout` here. Live rotation hand-off
        // is wired in #46; until then we spawn a loud-fallback
        // consumer that emits a `tracing::warn!` whenever the
        // deadline elapses so operators can see the timeout fired
        // even though the rotation is not yet executed.
        let (timeout_tx, mut timeout_rx) =
            tokio::sync::mpsc::channel::<tmg_harness::SessionEndTrigger>(4);
        // Register the (sender, duration) pair so EVERY future
        // `begin_session` (including the implicit one inside
        // `end_session_with_rotation`) re-arms the watchdog. Without
        // this, sessions after the first rotation would have no
        // wall-clock protection.
        let session_timeout = runner.default_session_timeout();
        if matches!(runner.scope(), tmg_harness::RunScope::Harnessed { .. }) {
            runner.set_session_timeout_config(Some((timeout_tx, session_timeout)));
        }
        // Loud-fallback consumer: drain the receiver and warn on
        // every Timeout trigger. Replace this spawn with the real
        // rotation hand-off once #46 lands.
        tokio::spawn(async move {
            while let Some(trigger) = timeout_rx.recv().await {
                tracing::warn!(
                    ?trigger,
                    "session timeout fired but live hand-off is not yet wired (issue #46)",
                );
            }
        });

        // Cross-session search index (issue #53). Initialise the SQLite
        // + FTS5 store under `<project>/.tsumugi/state.db`, run the
        // startup rebuild scan so any session_log files written before
        // the search feature was wired are picked up, then install a
        // session-end hook on the runner so live sessions stream into
        // the index. Failures in any of these steps degrade gracefully
        // — the index becomes `None`, the tool reports a soft error,
        // and the rest of the agent flow continues normally.
        let search_index: Option<Arc<tmg_search::SearchIndex>> = if search_config.enabled {
            let db_path = search_config.resolve_db_path(&canonical_cwd);
            match tmg_search::SearchIndex::open(&db_path) {
                Ok(idx) => {
                    let runs_dir_for_rebuild = store.runs_dir().to_path_buf();
                    if let Err(e) = idx.rebuild_from_disk(&runs_dir_for_rebuild) {
                        tracing::warn!(
                            error = %e,
                            "search index rebuild_from_disk failed; continuing with whatever was indexed"
                        );
                    }
                    Some(Arc::new(idx))
                }
                Err(e) => {
                    tracing::warn!(
                        path = %db_path.display(),
                        error = %e,
                        "failed to open search index; session_search tool will return soft errors"
                    );
                    None
                }
            }
        } else {
            None
        };
        if search_config.enabled
            && search_config.ingest_on_session_end
            && let Some(index) = search_index.as_ref()
        {
            let index_for_hook = Arc::clone(index);
            let hook: tmg_harness::SessionEndHook =
                Arc::new(move |run_id: &str, session: &tmg_harness::Session| {
                    if let Err(e) = index_for_hook.ingest_session(run_id, session) {
                        tracing::warn!(
                            run_id = %run_id,
                            session = session.index,
                            error = %e,
                            "search index ingest failed at session end"
                        );
                    }
                });
            runner.set_session_end_hook(Some(hook));
        }

        let session_handle = runner
            .begin_session()
            .context("beginning harness session")?;
        let run_summary: RunSummary = runner.summary();

        // Wrap the runner in Arc<Mutex<...>> so the Run-scoped tools and
        // the bootstrap path share one source of truth for the active
        // session.
        let runner = Arc::new(Mutex::new(runner));

        // Discover custom agent definitions.
        let custom_agent_metas = tmg_agents::discover_custom_agents(&project_root)
            .await
            .context("discovering custom agents")?;
        let custom_agent_defs: Vec<tmg_agents::CustomAgentDef> =
            custom_agent_metas.iter().map(|m| m.def().clone()).collect();

        // Build the parent [`SandboxContext`] from the merged
        // `[sandbox]` configuration. This context is shared with
        // `AgentLoop` (for top-level tool dispatch) and
        // `SubagentManager` (which derives a per-subagent context with
        // each agent kind's `sandbox_mode`).
        //
        // The workspace is the canonicalised cwd so workspace-write
        // checks line up with where the user invoked `tmg`. The OS
        // restrictions (`activate()`) are best-effort: on Linux,
        // Landlock + network-namespace setup is applied; on other
        // platforms it's a no-op and the software-level path checks
        // remain the enforcement boundary.
        let sandbox_ctx = Arc::new(tmg_sandbox::SandboxContext::new(
            sandbox_config.to_sandbox_config(&canonical_cwd),
        ));
        if let Err(e) = sandbox_ctx.activate().await {
            // Activation failure is fatal: the operator-configured
            // sandbox MUST be in force before any tool is dispatched.
            return Err(e).context("activating sandbox");
        }

        // Create the subagent manager. The escalator's
        // endpoint/model/disable knobs flow through
        // [`tmg_agents::EscalatorOverrides`] into the centralised
        // [`tmg_agents::EndpointResolver`] (issue #50) so the manager
        // delegates every spawn-time precedence decision to that
        // module.
        let escalator_overrides = tmg_agents::EscalatorOverrides::from_strings(
            harness_config.escalator.endpoint.clone(),
            harness_config.escalator.model.clone(),
            harness_config.escalator.disable,
        );
        // `[llm.subagent_pool]` is wired into the resolver here so
        // non-escalator builtin agents fan across the configured pool
        // endpoints. The validator already deduped URLs and reported
        // an empty list as "pool disabled" via the relaxed-validation
        // path; we honour that by only constructing a pool when the
        // post-validation list has at least one entry.
        let subagent_pool: Option<Arc<tmg_llm::LlmPool>> =
            pool_setup::subagent_pool_from_config(subagent_pool, model);
        // Capture the strategy BEFORE moving `subagent_pool` into the
        // resolver so the `--event-log` `pool_selected` records can
        // stamp the active strategy on every emitted event.
        let pool_strategy_label: Option<&'static str> =
            subagent_pool.as_ref().map(|p| p.strategy().as_str());
        let resolver = tmg_agents::EndpointResolver::new(endpoint, model)
            .with_escalator_overrides(escalator_overrides)
            .with_pool(subagent_pool);
        let subagent_manager = Arc::new(Mutex::new(tmg_agents::SubagentManager::new(
            client.clone(),
            cancel.clone(),
            resolver,
            Arc::clone(&sandbox_ctx),
        )));

        // Wire the `--event-log` observers on the subagent manager
        // and the global tokenize fallback path. Issue #50 review:
        // the previous wiring claim in the PR description was not
        // backed by code — the writer methods were defined but never
        // called. The hooks open the log file in append mode so we
        // share the same file the TUI itself writes into.
        if let Some(path) = event_log.as_deref() {
            match pool_setup::open_event_log(path) {
                Ok(log_writer) => {
                    pool_setup::install_event_log_hooks(
                        Arc::clone(&subagent_manager),
                        Arc::clone(&log_writer),
                        pool_strategy_label,
                    )
                    .await;
                    pool_setup::install_tokenize_failure_hook(log_writer);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to open event log for hooks; resolver/tokenize events will not be recorded");
                }
            }
        }

        // Wire the active `RunRunner` into the subagent manager so
        // harnessed-run subagents (initializer / tester / qa) get
        // their Run-aware tools (`progress_append`, `feature_list_*`)
        // registered when they spawn. The provider's scope flag
        // mirrors `register_run_tools` so the main agent and any
        // subagent see the same scope-gated tool set.
        let run_tool_provider: Arc<dyn tmg_agents::RunToolProvider> =
            Arc::new(RunRunnerToolProvider::new(Arc::clone(&runner)).await);
        subagent_manager
            .lock()
            .await
            .set_run_tool_provider(Some(Arc::clone(&run_tool_provider)));

        // Build the project memory store from `[memory]` config so the
        // `memory` tool and the prompt injection share one source of
        // truth. Construction is cheap (no I/O) and is always done so
        // the index can be re-read mid-session by the slash-command
        // path without re-parsing config.
        let memory_store = Arc::new(tmg_memory::MemoryStore::with_dirs(
            memory_config.resolve_project_dir(&canonical_cwd),
            memory_config.resolve_global_dir(),
            memory_config.to_budget(),
        ));

        // Shared pending-swap slot bridging the `memory` tool's
        // refresh callback (which fires from the JoinSet task that
        // dispatched the tool call) and the agent loop (which drains
        // the slot at the top of every `turn`). When memory is
        // disabled, we keep the option `None` so the agent loop's
        // `set_pending_memory_index(None)` keeps the cost zero.
        let pending_memory_index: Option<Arc<std::sync::Mutex<Option<String>>>> =
            memory_config
                .enabled
                .then(|| Arc::new(std::sync::Mutex::new(None)));

        // When memory is enabled, wire a [`MemoryToolProvider`] into
        // the subagent manager so every spawned subagent's registry
        // contains the `memory` tool. Issue #3 of PR #76 review.
        if memory_config.enabled {
            let provider: Arc<dyn tmg_agents::MemoryToolProvider> =
                Arc::new(CliMemoryToolProvider {
                    store: Arc::clone(&memory_store),
                    pending: pending_memory_index.clone(),
                });
            subagent_manager
                .lock()
                .await
                .set_memory_tool_provider(Some(provider));
        }

        // Create the tool registry: built-ins, then `memory`, then
        // spawn_agent, then the Run-scoped harness tools.
        // `register_run_tools` is the single authoritative place where
        // the harnessed-vs-ad-hoc tool set is decided; the CLI does
        // not branch on `scope()` itself so the gating logic lives in
        // one place.
        let mut registry = tmg_tools::default_registry();
        if memory_config.enabled {
            // Top-level agent gets a `MemoryTool` carrying a refresh
            // callback that re-reads the merged index after every
            // successful write and parks the snapshot in the shared
            // pending-swap slot. The agent loop drains the slot at
            // the top of its next `turn` so the LLM observes the
            // post-mutation state. Issue #4 of PR #76 review.
            let store_for_cb = Arc::clone(&memory_store);
            let pending_for_cb = pending_memory_index.clone();
            let cb: tmg_memory::tool::MemoryRefreshCallback =
                Arc::new(move |_index: &str| {
                    let Some(slot) = pending_for_cb.as_ref() else {
                        return;
                    };
                    let snapshot = match store_for_cb.read_merged_index() {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::debug!(
                                error = %e,
                                "memory refresh: read_merged_index failed",
                            );
                            return;
                        }
                    };
                    match slot.lock() {
                        Ok(mut guard) => {
                            *guard = Some(snapshot);
                        }
                        Err(e) => {
                            tracing::debug!(error = %e, "memory refresh: pending slot poisoned");
                        }
                    }
                });
            registry.register(tmg_memory::MemoryTool::with_refresh(
                Arc::clone(&memory_store),
                cb,
            ));
        }
        registry.register(tmg_agents::SpawnAgentTool::with_custom_agents(
            Arc::clone(&subagent_manager),
            custom_agent_defs.clone(),
        ));
        // Issue #53: register the cross-session FTS5 search tool when
        // enabled. When the index couldn't be opened we still register
        // a `disabled` tool so the LLM gets a soft error rather than a
        // missing-tool dispatch failure.
        if search_config.enabled {
            match search_index.as_ref() {
                Some(idx) => {
                    registry.register(tmg_search::SessionSearchTool::new(Arc::clone(idx)));
                }
                None => {
                    registry.register(tmg_search::SessionSearchTool::disabled());
                }
            }
        }
        // Issue #54: register `skill_manage` so the LLM (and the
        // autonomous skill_critic flow that reuses it via
        // `SkillsRuntime::apply_verdict`) can mutate
        // `.tsumugi/skills/`. Gated on `[skills.critic] enabled` so
        // operators can opt out by flipping a single flag in
        // `tsumugi.toml`.
        if skills_section.critic.enabled {
            registry.register(tmg_skills::SkillManageTool);
        }
        register_run_tools(&mut registry, Arc::clone(&runner)).await;

        // Workflow discovery + `run_workflow` / `workflow_status` tool
        // registration (issue #41). When at least one workflow is
        // discovered we build a [`WorkflowEngine`], install its
        // [`tmg_workflow::WorkflowIndex`], and register both tools.
        // When no workflows are present we skip registration entirely
        // so the LLM tool catalogue stays minimal.
        let workflow_metas = match tmg_workflow::discover_workflows(&project_root, workflow_config)
            .await
        {
            Ok(list) => list,
            Err(e) => {
                tracing::warn!(error = %e, "workflow discovery failed; tools not registered");
                Vec::new()
            }
        };
        // Background-runs handle. `Some` only if we actually
        // registered the workflow tools below; the TUI shutdown path
        // fires `cancel_all_background_runs` on it so in-flight
        // workflows terminate promptly instead of being aborted at the
        // runtime boundary.
        let mut background_runs: Option<tmg_workflow::BackgroundRunsHandle> = None;
        // Optional receiver of [`tmg_workflow::WorkflowProgress`]
        // events forwarded by the foreground `run_workflow` path; the
        // TUI subscribes via `set_workflow_progress_rx`. Only wired
        // when at least one workflow has been discovered (otherwise
        // there is nothing to listen to). The buffer size matches the
        // engine's internal channel cap so back-pressure is bounded
        // without blocking engine progress; saturation simply drops
        // observer events.
        let mut workflow_progress_rx: Option<
            tokio::sync::mpsc::Receiver<tmg_workflow::WorkflowProgress>,
        > = None;
        if !workflow_metas.is_empty() {
            // Eagerly parse each discovered workflow into a
            // canonical [`tmg_workflow::WorkflowDef`]. Parse errors
            // are non-fatal: a single bad workflow file should not
            // prevent the rest from loading. We log per-file failures
            // and skip them.
            let mut index_map: std::collections::HashMap<String, tmg_workflow::WorkflowDef> =
                std::collections::HashMap::new();
            for meta in &workflow_metas {
                match tmg_workflow::parse_workflow_file(&meta.source_path).await {
                    Ok(def) => {
                        index_map.insert(def.id.clone(), def);
                    }
                    Err(e) => {
                        tracing::warn!(
                            path = %meta.source_path.display(),
                            error = %e,
                            "skipping workflow that failed to parse",
                        );
                    }
                }
            }
            if !index_map.is_empty() {
                let workflow_index = Arc::new(tokio::sync::RwLock::new(index_map));
                let llm_pool_for_engine = Arc::new(tmg_llm::LlmPool::new(
                    &tmg_llm::PoolConfig::single(endpoint),
                    model,
                )?);
                // The engine's sandbox uses the same `WorkspaceWrite`
                // policy as the rest of the run for now; future work
                // (#42) will wire per-workflow sandbox overrides.
                let workflow_sandbox = Arc::new(tmg_sandbox::SandboxContext::new(
                    tmg_sandbox::SandboxConfig::new(&canonical_cwd)
                        .with_mode(tmg_sandbox::SandboxMode::WorkspaceWrite),
                ));
                let workflow_tool_registry = Arc::new(tmg_tools::ToolRegistry::new());
                let engine = Arc::new(
                    tmg_workflow::WorkflowEngine::new(
                        llm_pool_for_engine,
                        workflow_sandbox,
                        workflow_tool_registry,
                        Arc::clone(&subagent_manager),
                        workflow_config.clone(),
                        serde_json::Value::Null,
                    )
                    .with_workflow_index(Arc::clone(&workflow_index)),
                );
                let bg_runs = tmg_workflow::tools::new_background_runs();
                // Build a TUI-side observer channel. The foreground
                // `run_workflow` path fans every progress event into
                // `tui_progress_tx` in addition to draining it
                // internally; the TUI's activity pane drains the
                // matching receiver every tick. Saturation is dropped
                // (best-effort) so a stalled TUI never blocks engine
                // progress.
                let (tui_progress_tx, tui_progress_rx) =
                    tokio::sync::mpsc::channel::<tmg_workflow::WorkflowProgress>(64);
                tmg_workflow::register_workflow_tools_with_observer(
                    &mut registry,
                    &engine,
                    &bg_runs,
                    Some(tui_progress_tx),
                );
                workflow_progress_rx = Some(tui_progress_rx);
                // Stash a clone for the TUI shutdown path so we can
                // fire the cancellation tokens on every still-running
                // background workflow before the runtime tears down.
                background_runs = Some(bg_runs);
                tracing::info!(
                    count = workflow_metas.len(),
                    "registered run_workflow and workflow_status tools",
                );
            }
        }

        let max_context_tokens = context_config.max_context_tokens;
        // Install the operator-configured sandbox so file / shell
        // tools dispatched at the top level run under the same policy
        // the subagent manager hands its children. The sandbox is a
        // required constructor argument (issue #47 follow-up) so a
        // future code path cannot silently fall back to an
        // unrestricted default by forgetting to install one.
        let mut agent = tmg_core::AgentLoop::with_context_config(
            client,
            registry,
            cancel.clone(),
            &project_root,
            &canonical_cwd,
            context_config,
            tool_calling_mode,
            Arc::clone(&sandbox_ctx),
        )?;

        // Inject the memory index into the system prompt for this
        // session. Reading the merged index is best-effort: an I/O
        // error here downgrades to "no memory in prompt" so the agent
        // still starts. The capacity nudge is appended when the store
        // is at or above the configured caps so the agent knows to
        // curate before adding new entries.
        if memory_config.enabled {
            // Plug the shared pending-swap slot into the agent loop
            // BEFORE the first turn, so post-write refreshes from
            // `MemoryTool::with_refresh` are picked up automatically.
            // Issue #4 of PR #76 review.
            agent.set_pending_memory_index(pending_memory_index.clone());

            match memory_store.read_merged_index() {
                Ok(index) if !index.trim().is_empty() => {
                    let report = memory_store.budget_report().ok();
                    let mut payload = index.trim().to_owned();
                    if let Some(r) = report
                        && let Some(nudge) =
                            tmg_memory::capacity_nudge(&r, memory_store.budget())
                    {
                        payload.push_str(&nudge);
                    }
                    let summary = format!(
                        "merged index: {} chars / {} lines",
                        payload.len(),
                        payload.lines().count(),
                    );
                    agent.set_memory_index(Some(payload));
                    agent.inject_memory_index_now();
                    // Issue #12 of PR #76 review: record prompt-time
                    // memory injection so log readers can audit what
                    // the LLM saw at the start of the session.
                    if let Some(path) = event_log.as_deref()
                        && let Ok(mut writer) = tmg_core::EventLogWriter::open_append(path)
                    {
                        writer.write_memory("memory_prompt_inject", &summary);
                    }
                }
                Ok(_) => {
                    // Index empty / absent: nothing to inject.
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to read memory index for prompt");
                }
            }
        }

        // Wire the auto-promotion gate (issue #37): after every turn,
        // the harness observer records the turn metrics into
        // `RunRunner::session_state` so the
        // [`tmg_harness::EscalationEvaluator`] sees the SPEC §9.10
        // trigger inputs.
        //
        // The evaluator itself is **not** wired here yet: the TUI
        // event loop is the natural integration point for the async
        // background task that drives `evaluate` + `escalate_to_harnessed`,
        // and the TUI module owns the necessary `tokio::spawn` plumbing.
        // For now the runner's `session_state` is updated synchronously
        // so a future `tmg run upgrade` command can read the same
        // signals; the auto-evaluation hookup is tracked as a follow-up.
        install_turn_observer(&mut agent, Arc::clone(&runner), max_context_tokens);

        // Run session_bootstrap once and inject its output as a system
        // message so the LLM has the SPEC §9.7 context bundle for its
        // first turn. Failure is non-fatal: the agent simply starts
        // without the auto-injected bundle. The sandbox context is
        // threaded in so the bootstrap's `git log` shell-out runs
        // under the same policy as `shell_exec` (issue #47 follow-up
        // #3).
        inject_bootstrap(&mut agent, Arc::clone(&runner), Arc::clone(&sandbox_ctx)).await;

        // Subscribe to the runner's progress channel before handing
        // ownership of the runner mutex to the TUI. The channel is
        // bounded (16); the activity pane drains it every tick so
        // back-pressure on the runner is bounded by one TUI frame.
        let run_progress_rx = {
            let mut guard = runner.lock().await;
            Some(guard.progress_channel())
        };

        // Skill discovery for the `/skills` slash command listing.
        // Failure is non-fatal: the TUI starts with an empty list and
        // `/skills` simply reports no skills installed.
        let skill_metas =
            match tmg_skills::discover_skills_with_config(
                &project_root,
                skills_section.discovery(),
            )
            .await
            {
                Ok(list) => list,
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "skill discovery failed; /skills will report an empty list",
                    );
                    Vec::new()
                }
            };

        let tui_cancel = cancel.clone();
        let memory_store_for_tui = if memory_config.enabled {
            Some(Arc::clone(&memory_store))
        } else {
            None
        };
        let search_index_for_tui = search_index.as_ref().map(Arc::clone);

        // Issue #54: compute the startup banner text from any
        // unacknowledged `provenance: agent` skills the previous
        // session left behind. Best-effort; failure here just suppresses
        // the banner.
        let startup_banner = if skills_section.critic.enabled {
            let acked = match tmg_skills::load_acknowledged(
                project_root.join(".tsumugi").join("skills"),
            )
            .await
            {
                Ok(set) => set,
                Err(e) => {
                    tracing::debug!(error = %e, "load_acknowledged failed");
                    std::collections::BTreeSet::new()
                }
            };
            let pending = tmg_skills::pending_banner_names(&skill_metas, &acked);
            tmg_skills::format_banner(&pending)
        } else {
            None
        };

        let tui_result = tmg_tui::run(
            agent,
            model,
            tui_cancel,
            project_root,
            canonical_cwd,
            Some(subagent_manager),
            custom_agent_defs,
            event_log,
            Some(run_summary),
            Some(Arc::clone(&runner)),
            run_progress_rx,
            workflow_progress_rx,
            workflow_metas.clone(),
            skill_metas,
            memory_store_for_tui,
            search_index_for_tui,
            startup_banner,
        )
        .await;

        // Fire cancellation tokens on every still-running background
        // workflow so they observe the shutdown signal before the
        // runtime drops them. This is best-effort: a workflow whose
        // current await point does not consult `EngineCtx::cancel`
        // will still finish on its own, but at least the spawn
        // closure's `select!` loop sees the token in the next branch
        // it visits.
        if let Some(handle) = background_runs.as_ref() {
            let n = tmg_workflow::tools::cancel_all_background_runs(handle).await;
            if n > 0 {
                tracing::info!(count = n, "fired cancellation on background workflow runs");
            }
        }

        // Close the harness session before propagating the TUI result so
        // that `last_session_at` / `session_count` are persisted even on
        // error or cancellation paths. Capture the underlying error
        // message in the `Errored` trigger so post-mortem inspection of
        // the run sees the real failure rather than a generic string.
        let trigger = if let Err(ref e) = tui_result {
            SessionEndTrigger::Errored {
                message: format!("TUI returned an error: {e:#}"),
            }
        } else if cancel.is_cancelled() {
            SessionEndTrigger::UserCancelled
        } else {
            SessionEndTrigger::Completed
        };
        let end_result = {
            let mut guard = runner.lock().await;
            guard.end_session(&session_handle, trigger)
        };
        if let Err(e) = end_result {
            log_end_session_error(&e);
        }

        tui_result?;
        Ok::<(), anyhow::Error>(())
    })
}

/// RAII guard that clears the TUI-attached sentinel for a run on
/// `Drop`.
///
/// Created right after the sentinel is written so every exit path
/// from `run_tui` (clean return, error propagation, panic) removes
/// the file. Without the guard, a panic during agent setup would
/// leave a stale `.tui-pid` behind that would block CLI mutators
/// until the recorded PID is reused or the file is manually removed.
struct TuiSentinelGuard {
    run_dir: PathBuf,
}

impl TuiSentinelGuard {
    fn new(run_dir: PathBuf) -> Self {
        Self { run_dir }
    }
}

impl Drop for TuiSentinelGuard {
    fn drop(&mut self) {
        if let Err(e) = tmg_harness::tui_sentinel::clear(&self.run_dir) {
            tracing::warn!(error = %e, "failed to clear TUI sentinel on shutdown");
        }
    }
}

/// Surface a non-fatal error from `RunRunner::end_session` to the
/// operator. Persisting on shutdown is best-effort, so we never abort
/// the process — but a `SessionMismatch` here would indicate a
/// programming error (the handle threaded through `run_tui` should
/// always match the active session) and is called out explicitly.
fn log_end_session_error(err: &tmg_harness::HarnessError) {
    match err {
        tmg_harness::HarnessError::SessionMismatch { expected, actual } => {
            eprintln_warning(&format!(
                "internal error: end_session handle mismatch (expected {expected}, got {actual}); \
                 active session not persisted",
            ));
        }
        other => {
            eprintln_warning(&format!("failed to persist run state on shutdown: {other}",));
        }
    }
}

/// Print a warning to stderr without violating the workspace
/// `print_stderr` lint.
#[expect(
    clippy::print_stderr,
    reason = "best-effort shutdown warning; surfaced to operator without disturbing the TUI exit code"
)]
fn eprintln_warning(message: &str) {
    eprintln!("[tmg] warning: {message}");
}

/// Install the per-turn observer that feeds [`tmg_harness::SessionState`].
///
/// The observer captures the runner Arc plus the configured context
/// budget so that `RunRunner::after_turn` is called with both pieces
/// after every `AgentLoop::turn`.
///
/// **Async work via spawned task.** The closure itself is synchronous
/// (the agent loop runs it inline at turn-end). To keep the SPEC §9.10
/// triggers wired with real values, the closure spawns a tokio task
/// that:
///
/// 1. Reads the active session's `files_modified` list under a short
///    lock and computes the delta against a per-observer high-water
///    mark (so we only count files newly written this turn).
/// 2. Shells out to `git diff --shortstat --no-color HEAD` with a
///    2-second timeout to estimate the cumulative diff size, falling
///    back to `0` on any error (`git` not available, not a repo,
///    parse failure, timeout). A `tracing::debug!` is emitted on
///    fallback so operators can investigate.
/// 3. Re-acquires the runner lock and invokes `after_turn` with the
///    populated [`tmg_harness::TurnSummary`].
///
/// The observer detaches from the agent loop's execution: by the time
/// `turn` returns, the spawned task may not have completed yet. This
/// is acceptable because the auto-promotion evaluator only consults
/// `session_state` between turns, and a one-turn lag on a signal is
/// preferable to blocking the agent loop on `git`.
///
/// Note: the auto-promotion evaluator hookup (`detect_signals` →
/// `evaluate` → `escalate_to_harnessed`) is intentionally not part
/// of this wire-up. Wiring the async escalator round-trip into the
/// TUI is a follow-up; see issue #46 for the banner integration.
fn install_turn_observer(
    agent: &mut tmg_core::AgentLoop,
    runner: Arc<Mutex<RunRunner>>,
    max_context_tokens: usize,
) {
    // High-water mark on `session.files_modified.len()` from the
    // previous turn, so each call counts only the files newly recorded
    // by this turn rather than the cumulative session list.
    let files_high_water = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let observer: tmg_core::TurnObserver = Box::new(move |summary: &tmg_core::TurnSummary| {
        let runner = Arc::clone(&runner);
        let high_water = Arc::clone(&files_high_water);
        let tokens_used = summary.tokens_used;
        let tool_calls = summary.tool_calls;
        let user_message = summary.user_message.clone();

        // The closure runs inside `AgentLoop::turn`, which is `async`
        // and therefore inside the tokio runtime; `tokio::spawn` is
        // safe here. A failure to spawn (no current runtime) would be
        // a programming error rather than a recoverable condition;
        // we simply fall back to the synchronous update path so the
        // observer still updates `session_state` with the cheap
        // fields.
        if tokio::runtime::Handle::try_current().is_err() {
            update_session_state_sync(
                &runner,
                tokens_used,
                tool_calls,
                user_message,
                max_context_tokens,
            );
            return;
        }

        tokio::spawn(async move {
            // Step 1: drain the per-turn files_modified delta. Use a
            // short non-blocking lock acquisition so a long-running
            // tool that's still holding the lock cannot stall the
            // observer task; the rare contention case skips one
            // turn's metrics.
            let (workspace, files_modified) = if let Ok(guard) = runner.try_lock() {
                let workspace = guard.workspace_path_owned();
                let files = guard
                    .active_session()
                    .map(|s| s.files_modified.clone())
                    .unwrap_or_default();
                (workspace, files)
            } else {
                tracing::trace!("RunRunner busy; skipping after_turn update for one turn",);
                return;
            };

            let prev_mark =
                high_water.swap(files_modified.len(), std::sync::atomic::Ordering::SeqCst);
            let delta_files = if files_modified.len() > prev_mark {
                files_modified[prev_mark..].to_vec()
            } else {
                Vec::new()
            };

            // Step 2: shell out for diff size with a 2s timeout.
            let diff_lines = compute_diff_lines(&workspace).await;

            let harness_summary = tmg_harness::TurnSummary {
                tokens_used,
                tool_calls,
                files_modified: delta_files,
                diff_lines,
                user_message,
            };

            // Step 3: feed the summary into the runner and check
            // whether the SPEC §2.3 force-rotate threshold has been
            // exceeded. The actual `clear_history` / `begin_session`
            // sequence is owned by the higher-level TUI loop because
            // it requires mutable access to the live `AgentLoop`; we
            // only flip the rotation flag here so the consumer can
            // observe it on the next iteration.
            let mut guard = runner.lock().await;
            guard.after_turn(&harness_summary, max_context_tokens);

            let usage = if max_context_tokens == 0 {
                0.0
            } else {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "usage is clamped to [0.0, 1.0] before consumption"
                )]
                let raw = tokens_used as f64 / max_context_tokens as f64;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "ratio clamped to [0.0, 1.0] which fits in f32"
                )]
                {
                    raw.clamp(0.0, 1.0) as f32
                }
            };
            // Side-effect first: stamp the active session's
            // `context_usage_peak`. Then the pure check decides
            // whether to fire the rotation banner.
            guard.record_context_usage(usage);
            if guard.should_force_rotate(usage) {
                // Emit the SessionEnded event channel-side so a
                // downstream consumer wakes; the live rotation
                // (clear_history + begin_session) is wired by the
                // CLI / TUI follow-up (#46) which owns the agent.
                tracing::warn!(
                    usage,
                    "context usage exceeded force-rotate threshold; \
                     rotation must be driven by the TUI consumer",
                );
            }
        });
    });
    agent.set_turn_observer(Some(observer));
}

/// Synchronous fallback for [`install_turn_observer`] when no tokio
/// runtime is available. Updates `session_state` with the
/// always-known fields (`tokens_used`, `tool_calls`, `user_message`)
/// and leaves the I/O-derived fields at zero. Should only fire in
/// degraded test setups; production paths always run inside the
/// tokio runtime.
fn update_session_state_sync(
    runner: &Arc<Mutex<RunRunner>>,
    tokens_used: usize,
    tool_calls: u32,
    user_message: String,
    max_context_tokens: usize,
) {
    let harness_summary = tmg_harness::TurnSummary {
        tokens_used,
        tool_calls,
        files_modified: Vec::new(),
        diff_lines: 0,
        user_message,
    };
    match runner.try_lock() {
        Ok(mut guard) => guard.after_turn(&harness_summary, max_context_tokens),
        Err(_) => {
            tracing::trace!("RunRunner busy; skipping after_turn update for one turn",);
        }
    }
}

/// Estimate the cumulative diff size in `workspace` by shelling out
/// to `git diff --shortstat --no-color HEAD` with a 2-second timeout.
///
/// Returns `0` on any failure: workspace is not a git repo, `git`
/// isn't on PATH, the command timed out, or the shortstat line could
/// not be parsed. A `tracing::debug!` is emitted in the failure paths
/// so operators can investigate without the agent loop being
/// disturbed.
async fn compute_diff_lines(workspace: &std::path::Path) -> u32 {
    // Quick precondition: skip the spawn entirely when the workspace
    // does not look like a git repo. Cheap stat, avoids paying the
    // `git` startup cost for non-git projects.
    if !workspace.join(".git").exists() {
        return 0;
    }

    let mut command = tokio::process::Command::new("git");
    command
        .arg("diff")
        .arg("--shortstat")
        .arg("--no-color")
        .arg("HEAD")
        .current_dir(workspace)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());

    let output =
        match tokio::time::timeout(std::time::Duration::from_secs(2), command.output()).await {
            Ok(Ok(out)) => out,
            Ok(Err(e)) => {
                tracing::debug!(?e, "git diff --shortstat failed; diff_lines=0");
                return 0;
            }
            Err(_) => {
                tracing::debug!("git diff --shortstat timed out; diff_lines=0");
                return 0;
            }
        };

    if !output.status.success() {
        tracing::debug!(
            status = ?output.status,
            "git diff --shortstat returned non-zero; diff_lines=0",
        );
        return 0;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_shortstat(&stdout)
}

/// Parse a `git diff --shortstat` line of the form
/// `" 3 files changed, 42 insertions(+), 17 deletions(-)"` and return
/// `insertions + deletions`.
///
/// Returns `0` for empty input (no diff) or unparseable input. The
/// parse is intentionally lenient: insertions / deletions are
/// optional (a delete-only or insert-only diff omits the missing
/// half), and either may appear before the other.
fn parse_shortstat(s: &str) -> u32 {
    // Take the first non-empty line; `--shortstat` emits exactly one
    // line, but be defensive against trailing newlines.
    let Some(line) = s.lines().find(|l| !l.trim().is_empty()) else {
        return 0;
    };
    let mut total: u32 = 0;
    for part in line.split(',') {
        let trimmed = part.trim();
        if let Some(num_str) = trimmed.split_whitespace().next()
            && let Ok(n) = num_str.parse::<u32>()
            && (trimmed.contains("insertion") || trimmed.contains("deletion"))
        {
            total = total.saturating_add(n);
        }
    }
    total
}

/// Run `session_bootstrap` once and push its output into the agent's
/// history as a system message.
///
/// Failures (e.g. a missing `git` binary or a serialization error) are
/// surfaced as a stderr warning but never abort startup; the TUI is
/// usable without the bootstrap bundle.
async fn inject_bootstrap(
    agent: &mut tmg_core::AgentLoop,
    runner: Arc<Mutex<RunRunner>>,
    sandbox: Arc<tmg_sandbox::SandboxContext>,
) {
    let bootstrap_tool = SessionBootstrapTool::new(runner);
    match bootstrap_tool.run_once(&sandbox).await {
        Ok(payload) => match serde_json::to_string_pretty(&payload) {
            Ok(json) => {
                let injected = format!("[session_bootstrap]\n{json}\n[/session_bootstrap]");
                // Insert immediately after `history[0]` (the initial
                // system prompt) so chat templates that reject system
                // messages following a user turn (Mistral / Qwen /
                // Gemma) accept the bootstrap.
                agent.insert_bootstrap_system_message(injected);
            }
            Err(e) => eprintln_warning(&format!("failed to serialize bootstrap payload: {e}")),
        },
        Err(e) => eprintln_warning(&format!("session_bootstrap failed: {e}")),
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use clap::CommandFactory as _;

    // ---------------------------------------------------------------
    // CLI subcommand parsing (issue #43)
    // ---------------------------------------------------------------

    /// `clap` macro hygiene: the derive output should compile and
    /// validate without `debug_assert` failing.
    #[test]
    fn cli_definition_validates() {
        Cli::command().debug_assert();
    }

    /// `tmg` with no arguments parses to no subcommand and no prompt;
    /// this is the legacy "launch the TUI" code path.
    #[test]
    fn cli_no_args_launches_tui() {
        let cli = Cli::try_parse_from(["tmg"]).unwrap_or_else(|e| panic!("{e}"));
        assert!(cli.command.is_none());
        assert!(cli.prompt.is_none());
    }

    /// `tmg --prompt "hello"` parses with `prompt` set and no
    /// subcommand — the legacy one-shot mode.
    #[test]
    fn cli_prompt_only_runs_one_shot() {
        let cli =
            Cli::try_parse_from(["tmg", "--prompt", "hello"]).unwrap_or_else(|e| panic!("{e}"));
        assert!(cli.command.is_none());
        assert_eq!(cli.prompt.as_deref(), Some("hello"));
    }

    #[test]
    fn cli_run_list_parses() {
        let cli = Cli::try_parse_from(["tmg", "run", "list"]).unwrap_or_else(|e| panic!("{e}"));
        match cli.command {
            Some(Command::Run {
                op: RunCommand::List,
            }) => {}
            other => panic!("expected Run(List), got {other:?}"),
        }
    }

    #[test]
    fn cli_run_status_with_id_parses() {
        let cli = Cli::try_parse_from(["tmg", "run", "status", "abc12345"])
            .unwrap_or_else(|e| panic!("{e}"));
        match cli.command {
            Some(Command::Run {
                op: RunCommand::Status { run_id },
            }) => {
                assert_eq!(run_id.as_deref(), Some("abc12345"));
            }
            other => panic!("expected Run(Status), got {other:?}"),
        }
    }

    #[test]
    fn cli_run_resume_without_id_parses() {
        let cli = Cli::try_parse_from(["tmg", "run", "resume"]).unwrap_or_else(|e| panic!("{e}"));
        match cli.command {
            Some(Command::Run {
                op: RunCommand::Resume { run_id },
            }) => {
                assert!(run_id.is_none());
            }
            other => panic!("expected Run(Resume), got {other:?}"),
        }
    }

    #[test]
    fn cli_run_upgrade_parses() {
        let cli = Cli::try_parse_from(["tmg", "run", "upgrade"]).unwrap_or_else(|e| panic!("{e}"));
        match cli.command {
            Some(Command::Run {
                op: RunCommand::Upgrade { run_id },
            }) => {
                assert!(run_id.is_none());
            }
            other => panic!("expected Run(Upgrade), got {other:?}"),
        }
    }

    #[test]
    fn cli_run_pause_abort_downgrade_parse() {
        for sub in ["pause", "abort", "downgrade", "shell"] {
            let cli = Cli::try_parse_from(["tmg", "run", sub]).unwrap_or_else(|e| panic!("{e}"));
            assert!(matches!(cli.command, Some(Command::Run { .. })));
        }
    }

    #[test]
    fn cli_run_new_session_parses() {
        let cli =
            Cli::try_parse_from(["tmg", "run", "new-session"]).unwrap_or_else(|e| panic!("{e}"));
        match cli.command {
            Some(Command::Run {
                op: RunCommand::NewSession,
            }) => {}
            other => panic!("expected Run(NewSession), got {other:?}"),
        }
    }

    /// `--prompt` is `global = true` so it can be provided either
    /// before or after the subcommand. We only need the legacy
    /// "before any subcommand" form to keep working; this asserts
    /// global-flag plumbing didn't accidentally break it.
    #[test]
    fn cli_global_prompt_with_subcommand_parses() {
        let cli = Cli::try_parse_from(["tmg", "--prompt", "hi", "run", "list"])
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(cli.prompt.as_deref(), Some("hi"));
        assert!(matches!(
            cli.command,
            Some(Command::Run {
                op: RunCommand::List
            })
        ));
    }

    /// Parse the standard `git diff --shortstat` line shape.
    #[test]
    fn parse_shortstat_handles_canonical_line() {
        let line = " 3 files changed, 42 insertions(+), 17 deletions(-)\n";
        assert_eq!(parse_shortstat(line), 59);
    }

    /// `git diff --shortstat` may omit one of insertions / deletions
    /// when the diff is one-sided.
    #[test]
    fn parse_shortstat_handles_insertions_only() {
        assert_eq!(parse_shortstat(" 1 file changed, 5 insertions(+)\n"), 5,);
    }

    #[test]
    fn parse_shortstat_handles_deletions_only() {
        assert_eq!(parse_shortstat(" 2 files changed, 9 deletions(-)\n"), 9,);
    }

    #[test]
    fn parse_shortstat_returns_zero_for_empty_or_garbage() {
        assert_eq!(parse_shortstat(""), 0);
        assert_eq!(parse_shortstat("\n  \n"), 0);
        assert_eq!(parse_shortstat("not a shortstat line"), 0);
    }

    /// End-to-end: a sink-driven `files_modified` history flows
    /// through the per-turn observer's spawned task, lands on
    /// `SessionState`, and once the threshold is reached the
    /// [`tmg_harness::EscalationEvaluator::detect_signals`] fires
    /// the `SameFileEdit` rule.
    #[tokio::test]
    async fn observer_populates_files_modified_and_signals_fire() {
        use tmg_core::StreamSink as _;
        use tmg_harness::{
            EscalationConfig, EscalationEvaluator, EscalationSignal, HarnessStreamSink, RunStore,
            TurnSummary,
        };

        // Inner null sink — we only care about the wrapping side
        // effects on the runner's session state.
        struct NullSink;
        impl tmg_core::StreamSink for NullSink {
            fn on_token(&mut self, _t: &str) -> Result<(), tmg_core::CoreError> {
                Ok(())
            }
        }

        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let store = Arc::new(RunStore::new(tmp.path().join("runs")));
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut runner = RunRunner::new(run, Arc::clone(&store));
        let _ = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        let runner = Arc::new(Mutex::new(runner));

        // Drive the sink to record five edits to the same file. The
        // sink dedupes per-path, so the session sees exactly one
        // entry; we re-drive the runner's `after_turn` directly to
        // exercise the `same_file_edit` counter (which the sink does
        // not own).
        let mut sink = HarnessStreamSink::new(NullSink, Arc::clone(&runner));
        sink.on_tool_result(
            "call_1",
            "file_write",
            "Successfully wrote 5 bytes to '/tmp/hot.rs'",
            false,
        )
        .unwrap_or_else(|e| panic!("{e}"));

        // Simulate five turns that each touched `/tmp/hot.rs`. We feed
        // the same delta `vec!["/tmp/hot.rs"]` into `after_turn` five
        // times so the per-file edit tally reaches the threshold.
        {
            let mut guard = runner.lock().await;
            for _ in 0..5 {
                guard.after_turn(
                    &TurnSummary {
                        files_modified: vec!["/tmp/hot.rs".to_owned()],
                        ..Default::default()
                    },
                    8192,
                );
            }
        }

        // Now the SPEC §9.10 evaluator should observe the
        // SameFileEdit threshold trip.
        let evaluator = EscalationEvaluator::new(EscalationConfig::default(), None);
        let snapshot = {
            let guard = runner.lock().await;
            guard.session_state().clone()
        };
        let signals = evaluator.detect_signals(&snapshot);
        assert!(
            signals
                .iter()
                .any(|s| matches!(s, EscalationSignal::SameFileEdit { .. })),
            "expected SameFileEdit signal after 5 same-file edits, got {signals:?}",
        );
    }

    /// Files-modified delta accounting: when the observer runs twice
    /// against a sink-populated session, the second turn sees only
    /// the files newly added since the first turn.
    #[tokio::test]
    async fn files_modified_delta_advances_high_water_mark() {
        use tmg_core::StreamSink as _;
        use tmg_harness::{HarnessStreamSink, RunStore};

        struct NullSink;
        impl tmg_core::StreamSink for NullSink {
            fn on_token(&mut self, _t: &str) -> Result<(), tmg_core::CoreError> {
                Ok(())
            }
        }

        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let store = Arc::new(RunStore::new(tmp.path().join("runs")));
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut runner = RunRunner::new(run, Arc::clone(&store));
        let _ = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        let runner = Arc::new(Mutex::new(runner));

        let mut sink = HarnessStreamSink::new(NullSink, Arc::clone(&runner));
        sink.on_tool_result(
            "call_1",
            "file_write",
            "Successfully wrote 1 bytes to '/tmp/a.rs'",
            false,
        )
        .unwrap_or_else(|e| panic!("{e}"));

        // Snapshot 1: simulate the observer's high-water computation.
        let high_water = std::sync::atomic::AtomicUsize::new(0);
        let files_after_turn_1 = {
            let guard = runner.lock().await;
            guard
                .active_session()
                .unwrap_or_else(|| panic!("active session"))
                .files_modified
                .clone()
        };
        let prev = high_water.swap(
            files_after_turn_1.len(),
            std::sync::atomic::Ordering::SeqCst,
        );
        let delta_1 = files_after_turn_1[prev..].to_vec();
        assert_eq!(delta_1, vec!["/tmp/a.rs".to_owned()]);

        // Append a new file via the sink.
        sink.on_tool_result(
            "call_2",
            "file_write",
            "Successfully wrote 2 bytes to '/tmp/b.rs'",
            false,
        )
        .unwrap_or_else(|e| panic!("{e}"));

        // Snapshot 2: only the new file is in the delta; the high-
        // water mark prevents the previous file from being counted
        // again.
        let files_after_turn_2 = {
            let guard = runner.lock().await;
            guard
                .active_session()
                .unwrap_or_else(|| panic!("active session"))
                .files_modified
                .clone()
        };
        let prev = high_water.swap(
            files_after_turn_2.len(),
            std::sync::atomic::Ordering::SeqCst,
        );
        let delta_2 = files_after_turn_2[prev..].to_vec();
        assert_eq!(delta_2, vec!["/tmp/b.rs".to_owned()]);
    }

    /// `compute_diff_lines` short-circuits to `0` when the workspace
    /// has no `.git` directory, regardless of whether `git` is on
    /// PATH.
    #[tokio::test]
    async fn compute_diff_lines_returns_zero_outside_git() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let workspace = tmp.path().join("not-a-repo");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let result = compute_diff_lines(&workspace).await;
        assert_eq!(result, 0);
    }
}
