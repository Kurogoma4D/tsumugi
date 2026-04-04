use clap::Parser;

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
        run_interactive(&cli.endpoint, &cli.model)?;
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

/// Run an interactive multi-turn conversation loop.
///
/// Reads user input from stdin line-by-line, sends each line to the LLM
/// via [`tmg_core::AgentLoop`], and streams the response to stdout.
/// The loop exits on EOF (Ctrl-D) or when the `CancellationToken` is
/// triggered (Ctrl-C).
#[expect(
    clippy::print_stderr,
    reason = "CLI interactive mode: stderr used for user-facing prompts and status"
)]
fn run_interactive(endpoint: &str, model: &str) -> anyhow::Result<()> {
    use std::io::{BufRead as _, Write as _};

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

        let mut agent = tmg_core::AgentLoop::new(client, cancel.clone(), &project_root, &cwd)?;

        let stdin = std::io::stdin();
        let mut reader = stdin.lock();

        let mut sink = StdoutSink;

        loop {
            if cancel.is_cancelled() {
                eprintln!("\n[cancelled]");
                break;
            }

            eprint!("> ");
            std::io::stderr().flush()?;

            let mut input = String::new();
            let bytes_read = reader.read_line(&mut input)?;

            // EOF (Ctrl-D)
            if bytes_read == 0 {
                eprintln!("\n[bye]");
                break;
            }

            let trimmed = input.trim();
            if trimmed.is_empty() {
                continue;
            }

            match agent.turn(trimmed, &mut sink).await {
                Ok(()) => {}
                Err(tmg_core::CoreError::Cancelled) => {
                    eprintln!("\n[cancelled]");
                    break;
                }
                Err(e) => {
                    return Err(anyhow::Error::from(e));
                }
            }
        }

        Ok::<(), anyhow::Error>(())
    })
}

/// A [`tmg_core::StreamSink`] that writes tokens to stdout.
struct StdoutSink;

#[expect(
    clippy::print_stdout,
    reason = "CLI streaming output sink: intentional stdout writes"
)]
impl tmg_core::StreamSink for StdoutSink {
    fn on_token(&mut self, token: &str) -> Result<(), tmg_core::CoreError> {
        use std::io::Write as _;
        print!("{token}");
        std::io::stdout().flush()?;
        Ok(())
    }

    fn on_done(&mut self) -> Result<(), tmg_core::CoreError> {
        use std::io::Write as _;
        println!();
        std::io::stdout().flush()?;
        Ok(())
    }
}
