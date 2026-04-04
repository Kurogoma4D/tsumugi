use clap::Parser;

/// tsumugi - a local-LLM-powered coding agent
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {}

#[expect(
    clippy::unnecessary_wraps,
    reason = "main will propagate errors once features are implemented"
)]
fn main() -> anyhow::Result<()> {
    let _cli = Cli::parse();
    Ok(())
}
