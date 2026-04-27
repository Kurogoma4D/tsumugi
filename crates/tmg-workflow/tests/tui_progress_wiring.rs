//! Integration test for the TUI wire-up (issue #41).
//!
//! The TUI side of #41 is intentionally minimal: an
//! `mpsc::Receiver<WorkflowProgress>` is attached to `App` and drained
//! on each event-loop tick. The deliverable is the wire-up — full
//! rendering arrives in #45.
//!
//! This test exercises the wire-up end-to-end: it runs a tiny
//! foreground workflow against the engine and asserts that an external
//! consumer (standing in for the TUI's `try_recv` loop) receives
//! progress events. The test does NOT spin up the real ratatui frontend
//! — that requires a TTY — but it validates the contract that the TUI
//! depends on (events arrive through the channel as the engine runs).

#![expect(clippy::unwrap_used, reason = "test assertions")]

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

use tmg_agents::SubagentManager;
use tmg_llm::{LlmPool, PoolConfig};
use tmg_sandbox::{SandboxConfig, SandboxContext, SandboxMode};
use tmg_tools::ToolRegistry;
use tmg_workflow::{WorkflowConfig, WorkflowEngine, WorkflowProgress, parse_workflow_str};

fn build_engine(workspace: &Path) -> WorkflowEngine {
    let llm_pool =
        Arc::new(LlmPool::new(&PoolConfig::single("http://localhost:9999"), "test-model").unwrap());
    let sandbox = Arc::new(SandboxContext::new(
        SandboxConfig::new(workspace).with_mode(SandboxMode::WorkspaceWrite),
    ));
    let tool_registry = Arc::new(ToolRegistry::new());
    let llm_client_cfg = tmg_llm::LlmClientConfig::new("http://localhost:9999", "test-model");
    let llm_client = tmg_llm::LlmClient::new(llm_client_cfg).unwrap();
    let cancel = CancellationToken::new();
    let manager = SubagentManager::new(llm_client, cancel, "http://localhost:9999", "test-model");
    let subagent_manager = Arc::new(Mutex::new(manager));
    WorkflowEngine::new(
        llm_pool,
        sandbox,
        tool_registry,
        subagent_manager,
        WorkflowConfig::default(),
        serde_json::Value::Null,
    )
}

/// A foreground workflow run delivers `StepStarted`, `StepCompleted`,
/// and `WorkflowCompleted` events through the channel — exactly what
/// the TUI's `drain_workflow_progress` loop consumes.
#[tokio::test]
async fn foreground_workflow_emits_progress_events_to_consumer() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path());

    let yaml = r#"
id: tui_smoke
steps:
  - id: tap
    type: shell
    command: "echo hello"
outputs:
  out: "${{ steps.tap.exit_code }}"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();

    let (tx, mut rx) = mpsc::channel(32);
    // Standing in for the TUI: spawn a consumer that records the event
    // kinds it sees.
    let consumer = tokio::spawn(async move {
        let mut kinds: Vec<&'static str> = Vec::new();
        while let Some(ev) = rx.recv().await {
            let label = match ev {
                WorkflowProgress::StepStarted { .. } => "step_started",
                WorkflowProgress::StepCompleted { .. } => "step_completed",
                WorkflowProgress::StepFailed { .. } => "step_failed",
                WorkflowProgress::StepOutput { .. } => "step_output",
                WorkflowProgress::LoopIteration { .. } => "loop_iteration",
                WorkflowProgress::HumanInputRequired { .. } => "human_input_required",
                WorkflowProgress::WorkflowCompleted { .. } => "workflow_completed",
                // `WorkflowProgress` is `#[non_exhaustive]`; future
                // variants are observed but unnamed here.
                _ => "other",
            };
            kinds.push(label);
        }
        kinds
    });

    engine.run(&wf, BTreeMap::new(), tx).await.unwrap();
    let kinds = consumer.await.unwrap();

    // We expect at least the canonical lifecycle: started → completed →
    // workflow completed.
    assert!(kinds.contains(&"step_started"), "got: {kinds:?}");
    assert!(kinds.contains(&"step_completed"), "got: {kinds:?}");
    assert!(kinds.contains(&"workflow_completed"), "got: {kinds:?}");
}
