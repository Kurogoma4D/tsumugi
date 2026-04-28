//! Integration test for the foreground `run_workflow` → TUI activity
//! pane wiring (PR #69 review finding 2).
//!
//! `tmg-cli` constructs an mpsc channel, hands the sender to
//! [`tmg_workflow::register_workflow_tools_with_observer`], and hands
//! the receiver to [`tmg_tui::run`] which forwards it to
//! [`tmg_tui::App::set_workflow_progress_rx`]. This test exercises
//! that loop end-to-end without spinning up the real ratatui frontend
//! (which requires a TTY): build a `RunWorkflowTool` with the
//! observer plumbed in, invoke its foreground execute path, and apply
//! every event the receiver yields to a fresh
//! [`tmg_tui::ActivityPane`]. We then assert the pane briefly held a
//! populated `workflow_progress` section as `StepStarted` events
//! flowed through.

#![expect(clippy::unwrap_used, reason = "test assertions")]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use serde_json::Value;
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio_util::sync::CancellationToken;

use tmg_agents::SubagentManager;
use tmg_llm::{LlmPool, PoolConfig};
use tmg_sandbox::{SandboxConfig, SandboxContext, SandboxMode};
use tmg_tools::{Tool, ToolRegistry};
use tmg_tui::ActivityPane;
use tmg_workflow::{
    RunWorkflowTool, WorkflowConfig, WorkflowEngine, WorkflowProgress, parse_workflow_str, tools,
};

fn build_engine(
    workspace: &Path,
    workflows: Vec<(&str, &str)>,
) -> (
    Arc<WorkflowEngine>,
    Arc<RwLock<HashMap<String, tmg_workflow::WorkflowDef>>>,
) {
    let llm_pool =
        Arc::new(LlmPool::new(&PoolConfig::single("http://localhost:9999"), "test-model").unwrap());
    let sandbox = Arc::new(SandboxContext::new(
        SandboxConfig::new(workspace).with_mode(SandboxMode::WorkspaceWrite),
    ));
    let tool_registry = Arc::new(ToolRegistry::new());

    let llm_client_cfg = tmg_llm::LlmClientConfig::new("http://localhost:9999", "test-model");
    let llm_client = tmg_llm::LlmClient::new(llm_client_cfg).unwrap();
    let cancel = CancellationToken::new();
    let manager = SubagentManager::new(
        llm_client,
        cancel,
        "http://localhost:9999",
        "test-model",
        Arc::clone(&sandbox),
    );
    let subagent_manager = Arc::new(Mutex::new(manager));

    let mut index_map: HashMap<String, tmg_workflow::WorkflowDef> = HashMap::new();
    for (id, yaml) in workflows {
        let wf = parse_workflow_str(yaml, format!("<{id}>")).unwrap();
        index_map.insert(id.to_owned(), wf);
    }
    let index = Arc::new(RwLock::new(index_map));

    let engine = Arc::new(
        WorkflowEngine::new(
            llm_pool,
            sandbox,
            tool_registry,
            subagent_manager,
            WorkflowConfig::default(),
            Value::Null,
        )
        .with_workflow_index(Arc::clone(&index)),
    );
    (engine, index)
}

/// End-to-end: a foreground `run_workflow` invocation publishes every
/// progress event to the observer channel; replaying those events
/// through `ActivityPane::apply_workflow_event` reproduces the live
/// `workflow_progress` section the TUI would render.
#[tokio::test]
async fn run_workflow_tool_drives_activity_pane_through_observer() {
    let tmp = tempfile::tempdir().unwrap();
    let yaml = r#"
id: simple
steps:
  - id: produce
    type: shell
    command: "echo hello"
outputs:
  exit: "${{ steps.produce.exit_code }}"
"#;
    let (engine, index) = build_engine(tmp.path(), vec![("simple", yaml)]);
    let bg = tools::new_background_runs();
    let (tx, mut rx) = mpsc::channel::<WorkflowProgress>(64);
    let tool = RunWorkflowTool::with_progress_observer(engine, index, bg, tx);

    // Spawn a consumer that mirrors the TUI's drain: it applies every
    // event to a fresh `ActivityPane` and records whether the
    // `workflow_progress` section was ever populated. The pane clears
    // the section on `WorkflowCompleted`, so we cannot rely on the
    // post-run snapshot — instead we observe the live transition.
    let consumer = tokio::spawn(async move {
        let mut pane = ActivityPane::new();
        let mut saw_workflow_section = false;
        let mut saw_step_started = false;
        while let Some(ev) = rx.recv().await {
            if matches!(ev, WorkflowProgress::StepStarted { .. }) {
                saw_step_started = true;
            }
            pane.apply_workflow_event(&ev);
            if pane.workflow_progress.is_some() {
                saw_workflow_section = true;
            }
        }
        (saw_workflow_section, saw_step_started, pane)
    });

    let ctx = SandboxContext::test_default();
    let result = tool
        .execute(serde_json::json!({"workflow": "simple"}), &ctx)
        .await
        .unwrap();
    assert!(!result.is_error, "tool reported error: {}", result.output);

    // Drop the tool so its observer `Sender` is dropped, closing the
    // channel and letting the consumer's `recv` loop exit. Without
    // this the test would hang waiting for further events that will
    // never arrive.
    drop(tool);

    let (saw_workflow_section, saw_step_started, final_pane) = consumer.await.unwrap();
    assert!(
        saw_step_started,
        "observer should have received a StepStarted event during the run",
    );
    assert!(
        saw_workflow_section,
        "ActivityPane.workflow_progress should be populated during the run",
    );
    // Post-run, `WorkflowCompleted` clears the section — confirms the
    // state machine reaches the terminal arm rather than getting
    // stuck.
    assert!(
        final_pane.workflow_progress.is_none(),
        "WorkflowCompleted should clear workflow_progress",
    );
}
