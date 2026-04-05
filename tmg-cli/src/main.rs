use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context as _;
use clap::Parser;
use tmg_llm::ToolCallingMode;
use tokio::sync::Mutex;

mod config;
mod error;

use config::TsumugiConfig;

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
        run_prompt(&config.llm.endpoint, &config.llm.model, &prompt)?;
    } else {
        run_tui(
            &config.llm.endpoint,
            &config.llm.model,
            context_config,
            config.llm.tool_calling,
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
fn run_prompt(endpoint: &str, model: &str, prompt: &str) -> anyhow::Result<()> {
    use std::io::Write as _;
    use tokio_stream::StreamExt as _;

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
                tmg_llm::StreamEvent::ThinkingDelta(_) => {
                    // Thinking tokens are not displayed in one-shot mode.
                }
                tmg_llm::StreamEvent::ContentDelta(text) => {
                    print!("{text}");
                    std::io::stdout().flush()?;
                }
                tmg_llm::StreamEvent::ToolCallComplete(tc) => {
                    println!(
                        "\n[tool_call] {}({})",
                        tc.function.name, tc.function.arguments
                    );
                }
                tmg_llm::StreamEvent::Done(reason) => {
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
fn run_tui(
    endpoint: &str,
    model: &str,
    context_config: tmg_core::ContextConfig,
    tool_calling_mode: tmg_core::ToolCallingMode,
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
        // Use cwd as project root for now (could be improved with git root detection).
        let project_root = cwd.clone();

        // Discover custom agent definitions.
        let custom_agent_metas = tmg_agents::discover_custom_agents(&project_root)
            .await
            .context("discovering custom agents")?;
        let custom_agent_defs: Vec<tmg_agents::CustomAgentDef> =
            custom_agent_metas.iter().map(|m| m.def().clone()).collect();

        // Create the subagent manager.
        let subagent_manager = Arc::new(Mutex::new(tmg_agents::SubagentManager::new(
            client.clone(),
            cancel.clone(),
            endpoint,
            model,
        )));

        // Create the tool registry with spawn_agent tool (including custom agents).
        let mut registry = tmg_tools::default_registry();
        registry.register(tmg_agents::SpawnAgentTool::with_custom_agents(
            Arc::clone(&subagent_manager),
            custom_agent_defs.clone(),
        ));

        let agent = tmg_core::AgentLoop::with_context_config(
            client,
            registry,
            cancel.clone(),
            &project_root,
            &cwd,
            context_config,
            tool_calling_mode,
        )?;

        tmg_tui::run(
            agent,
            model,
            cancel,
            project_root,
            cwd,
            Some(subagent_manager),
            custom_agent_defs,
        )
        .await?;

        Ok::<(), anyhow::Error>(())
    })
}
