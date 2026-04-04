use std::sync::Arc;

use anyhow::Context as _;
use clap::Parser;
use tokio::sync::Mutex;

/// tsumugi - a local-LLM-powered coding agent
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Send a one-shot prompt to the LLM server and stream the response to stdout.
    #[arg(long)]
    prompt: Option<String>,

    /// LLM server endpoint URL.
    #[arg(long, default_value = "http://localhost:8080")]
    endpoint: String,

    /// Model name to use.
    #[arg(long, default_value = "default")]
    model: String,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if let Some(prompt) = cli.prompt {
        run_prompt(&cli.endpoint, &cli.model, &prompt)?;
    } else {
        run_tui(&cli.endpoint, &cli.model)?;
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
fn run_tui(endpoint: &str, model: &str) -> anyhow::Result<()> {
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

        let agent =
            tmg_core::AgentLoop::new(client, registry, cancel.clone(), &project_root, &cwd)?;

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
