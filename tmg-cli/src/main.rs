use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context as _;
use clap::Parser;
use tmg_harness::{
    RunRunner, RunRunnerToolProvider, RunStore, RunSummary, SessionBootstrapTool,
    SessionEndTrigger, register_run_tools,
};
use tmg_llm::ToolCallingMode;
use tokio::sync::Mutex;

mod config;
mod error;
mod harness_init;

use config::{HarnessConfig, SandboxConfigSection, TsumugiConfig};

/// tsumugi - a local-LLM-powered coding agent
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Send a one-shot prompt to the LLM server and stream the response
    /// to stdout.
    #[arg(long)]
    prompt: Option<String>,

    /// LLM server endpoint URL. Overrides config file and environment.
    #[arg(long)]
    endpoint: Option<String>,

    /// Model name to use. Overrides config file and environment.
    #[arg(long)]
    model: Option<String>,

    /// Path to a `tsumugi.toml` configuration file.
    /// When specified, only this file is loaded (no global/project-local
    /// discovery).
    #[arg(long)]
    config: Option<PathBuf>,

    /// Maximum context window tokens.
    #[arg(long)]
    max_context_tokens: Option<usize>,

    /// Context compression threshold (0.0-1.0). Compression auto-triggers
    /// when context usage exceeds this fraction of `max_context_tokens`.
    #[arg(long)]
    context_compression_threshold: Option<f64>,

    /// Maximum tokens for a single tool result before truncation.
    #[arg(long)]
    max_tool_result_tokens: Option<usize>,

    /// Tool calling mode: "native", "prompt_based", or "auto".
    ///
    /// - native: use OpenAI-compatible function calling API
    /// - prompt_based: inject tool descriptions in system prompt
    /// - auto (default): try native first, fall back to prompt_based parsing
    #[expect(
        clippy::doc_markdown,
        reason = "clap renders doc comments as --help text; backticks would show literally"
    )]
    #[arg(long)]
    tool_calling: Option<ToolCallingMode>,

    /// Path to write structured event log (JSON Lines format).
    /// Enables diagnostics by recording every agent event (tokens,
    /// tool calls, results) to the specified file.
    #[arg(long)]
    event_log: Option<PathBuf>,
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
        if let Some(tc) = self.tool_calling {
            config.llm.tool_calling = tc;
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Load and merge configuration: global -> project-local -> env -> CLI.
    let mut config =
        config::load_config(cli.config.as_deref()).context("loading tsumugi configuration")?;
    cli.apply_to(&mut config);
    config.validate().context("validating configuration")?;

    let context_config = tmg_core::ContextConfig {
        max_context_tokens: config.llm.max_context_tokens,
        compression_threshold: config.llm.compression_threshold,
        max_tool_result_tokens: config.llm.max_tool_result_tokens,
    };

    if let Some(prompt) = cli.prompt {
        run_prompt(
            &config.llm.endpoint,
            &config.llm.model,
            &prompt,
            cli.event_log.as_deref(),
        )?;
    } else {
        run_tui(
            &config.llm.endpoint,
            &config.llm.model,
            context_config,
            config.llm.tool_calling,
            cli.event_log,
            &config.harness,
            &config.sandbox,
        )?;
    }

    Ok(())
}

/// Run a one-shot streaming prompt against the LLM server.
///
/// This function is the integration check for issue #2: it sends a prompt
/// to the llama-server and streams the response token-by-token to stdout.
#[expect(
    clippy::print_stdout,
    reason = "integration check: intentional stdout streaming output"
)]
fn run_prompt(
    endpoint: &str,
    model: &str,
    prompt: &str,
    event_log: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    use std::io::Write as _;
    use tokio_stream::StreamExt as _;

    let mut log_writer = event_log
        .map(tmg_core::EventLogWriter::new)
        .transpose()
        .context("opening event log file")?;

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let config = tmg_llm::LlmClientConfig::new(endpoint, model);
        let client = tmg_llm::LlmClient::new(config)?;

        let messages = vec![tmg_llm::ChatMessage {
            role: tmg_llm::Role::User,
            content: Some(prompt.to_owned()),
            tool_calls: None,
            tool_call_id: None,
        }];

        let cancel = tokio_util::sync::CancellationToken::new();
        let mut stream = client.chat_streaming(messages, vec![], cancel).await?;

        while let Some(event) = stream.next().await {
            match event? {
                tmg_llm::StreamEvent::ThinkingDelta(token) => {
                    if let Some(ref mut log) = log_writer {
                        log.write_thinking(&token);
                    }
                    // Thinking tokens are not displayed in one-shot mode.
                }
                tmg_llm::StreamEvent::ContentDelta(text) => {
                    if let Some(ref mut log) = log_writer {
                        log.write_token(&text);
                    }
                    print!("{text}");
                    std::io::stdout().flush()?;
                }
                tmg_llm::StreamEvent::ToolCallComplete(tc) => {
                    if let Some(ref mut log) = log_writer {
                        log.write_tool_call(&tc.function.name, &tc.function.arguments);
                    }
                    println!(
                        "\n[tool_call] {}({})",
                        tc.function.name, tc.function.arguments
                    );
                }
                tmg_llm::StreamEvent::Done(reason) => {
                    if let Some(ref mut log) = log_writer {
                        log.write_done();
                    }
                    println!();
                    if let Some(r) = reason {
                        println!("[done: {r}]");
                    }
                }
            }
        }

        Ok::<(), anyhow::Error>(())
    })
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
fn run_tui(
    endpoint: &str,
    model: &str,
    context_config: tmg_core::ContextConfig,
    tool_calling_mode: tmg_core::ToolCallingMode,
    event_log: Option<PathBuf>,
    harness_config: &HarnessConfig,
    sandbox_config: &SandboxConfigSection,
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
        let run = harness_init::resolve_startup_run(harness_config, &store, canonical_cwd.clone())
            .context("resolving startup run")?;
        let mut runner = RunRunner::new(run, Arc::clone(&store));
        runner.set_bootstrap_max_tokens(harness_config.bootstrap_max_tokens);
        runner.set_default_session_timeout(harness_config.default_session_timeout);
        // One-time warning when the user's `[sandbox] mode` is stricter
        // than the harnessed `init.sh` execution path honours; see
        // `harness_init::warn_if_sandbox_mode_mismatch` for details.
        if matches!(runner.scope(), tmg_harness::RunScope::Harnessed { .. }) {
            harness_init::warn_if_sandbox_mode_mismatch(sandbox_config);
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

        // Create the subagent manager. The escalator's
        // endpoint/model/disable knobs flow through
        // [`tmg_agents::EscalatorOverrides`] so the resolver in
        // `SubagentManager` is the single source of truth for
        // precedence (see `crates/tmg-agents/src/manager.rs` module
        // docs).
        let escalator_overrides = tmg_agents::EscalatorOverrides::from_strings(
            harness_config.escalator.endpoint.clone(),
            harness_config.escalator.model.clone(),
            harness_config.escalator.disable,
        );
        let subagent_manager = Arc::new(Mutex::new(
            tmg_agents::SubagentManager::new(client.clone(), cancel.clone(), endpoint, model)
                .with_escalator_overrides(escalator_overrides),
        ));

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

        // Create the tool registry: built-ins, then spawn_agent, then
        // the Run-scoped harness tools. `register_run_tools` is the
        // single authoritative place where the harnessed-vs-ad-hoc
        // tool set is decided; the CLI does not branch on `scope()`
        // itself so the gating logic lives in one place.
        let mut registry = tmg_tools::default_registry();
        registry.register(tmg_agents::SpawnAgentTool::with_custom_agents(
            Arc::clone(&subagent_manager),
            custom_agent_defs.clone(),
        ));
        register_run_tools(&mut registry, Arc::clone(&runner)).await;

        let max_context_tokens = context_config.max_context_tokens;
        let mut agent = tmg_core::AgentLoop::with_context_config(
            client,
            registry,
            cancel.clone(),
            &project_root,
            &canonical_cwd,
            context_config,
            tool_calling_mode,
        )?;

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
        // without the auto-injected bundle.
        inject_bootstrap(&mut agent, Arc::clone(&runner)).await;

        let tui_cancel = cancel.clone();
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
        )
        .await;

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
/// after every `AgentLoop::turn`. The callback uses `try_lock` so a
/// long-running tool that's still holding the runner lock cannot
/// stall the agent loop; in the rare contention case, one turn's
/// metrics are skipped and a `tracing::trace!` is emitted.
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
    let observer: tmg_core::TurnObserver = Box::new(move |summary: &tmg_core::TurnSummary| {
        // The Arc clone is cheap; the lock guard is dropped quickly so
        // the harness sink and the run-scoped tools can re-acquire it
        // immediately.
        let harness_summary = tmg_harness::TurnSummary {
            tokens_used: summary.tokens_used,
            tool_calls: summary.tool_calls,
            files_modified: Vec::new(),
            diff_lines: 0,
            user_message: summary.user_message.clone(),
        };
        match runner.try_lock() {
            Ok(mut guard) => guard.after_turn(&harness_summary, max_context_tokens),
            Err(_) => {
                tracing::trace!("RunRunner busy; skipping after_turn update for one turn",);
            }
        }
    });
    agent.set_turn_observer(Some(observer));
}

/// Run `session_bootstrap` once and push its output into the agent's
/// history as a system message.
///
/// Failures (e.g. a missing `git` binary or a serialization error) are
/// surfaced as a stderr warning but never abort startup; the TUI is
/// usable without the bootstrap bundle.
async fn inject_bootstrap(agent: &mut tmg_core::AgentLoop, runner: Arc<Mutex<RunRunner>>) {
    let bootstrap_tool = SessionBootstrapTool::new(runner);
    match bootstrap_tool.run_once().await {
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
