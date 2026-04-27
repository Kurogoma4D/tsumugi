//! Integration tests for [`WorkflowEngine`] focused on sequential
//! step execution and the `${{ ... }}` chaining behaviour.
//!
//! These tests deliberately exercise *only* the shell + `write_file`
//! subset so they don't depend on a live LLM. The agent path has its
//! own coverage in `engine_agent_isolation.rs` (which checks the
//! freshness invariant by inspecting the spawned `SubagentRunner`'s
//! conversation history).

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
use tmg_workflow::{
    WorkflowConfig, WorkflowDef, WorkflowEngine, WorkflowProgress, parse_workflow_str,
};

/// Build the engine + a temp workspace + matching sandbox for tests.
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

#[tokio::test]
async fn shell_then_write_file_chain() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path());

    let yaml = r#"
id: shell_chain
description: "shell -> write_file"
inputs:
  greeting:
    type: string
    default: "hello"
steps:
  - id: produce
    type: shell
    command: "echo --tag-line--"
  - id: persist
    type: write_file
    path: "out.txt"
    content: "${{ inputs.greeting }} | exit=${{ steps.produce.exit_code }}"
outputs:
  produced: "${{ steps.produce.exit_code }}"
  saved_to: "${{ steps.persist.output.path }}"
"#;
    let wf: WorkflowDef = parse_workflow_str(yaml, "<inline>").unwrap();
    let inputs = BTreeMap::new();
    let (tx, mut rx) = mpsc::channel(32);

    let outputs = engine.run(&wf, inputs, tx).await.unwrap();

    // Sequential progress events: started/completed pairs in declared order.
    let mut events = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }
    let order: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            WorkflowProgress::StepStarted { step_id, .. } => Some(format!("started:{step_id}")),
            WorkflowProgress::StepCompleted { step_id, .. } => Some(format!("completed:{step_id}")),
            _ => None,
        })
        .collect();
    assert_eq!(
        order,
        vec![
            "started:produce",
            "completed:produce",
            "started:persist",
            "completed:persist",
        ],
    );

    assert_eq!(
        outputs.values.get("produced").map(String::as_str),
        Some("0")
    );
    assert_eq!(
        outputs.values.get("saved_to").map(String::as_str),
        Some("out.txt")
    );

    // The file was written.
    let written = std::fs::read_to_string(tmp.path().join("out.txt")).unwrap();
    assert!(written.starts_with("hello | exit=0"), "got: {written}");
}

#[tokio::test]
async fn when_clause_skips_step() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path());

    let yaml = r#"
id: skip_test
steps:
  - id: should_run
    type: shell
    command: "echo run-me"
  - id: should_skip
    type: shell
    command: "echo will-not-run"
    when: "steps.should_run.exit_code != 0"
outputs:
  marker: "${{ steps.should_run.exit_code }}"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, mut rx) = mpsc::channel(32);

    let outputs = engine.run(&wf, BTreeMap::new(), tx).await.unwrap();
    assert_eq!(outputs.values.get("marker").map(String::as_str), Some("0"));

    // Verify only one step's started event was emitted.
    let mut started = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        if let WorkflowProgress::StepStarted { step_id, .. } = ev {
            started.push(step_id);
        }
    }
    assert_eq!(started, vec!["should_run".to_owned()]);
}

#[tokio::test]
async fn missing_required_input_is_error() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path());

    let yaml = r"
id: needs_input
inputs:
  must_have:
    type: string
    required: true
steps: []
";
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, _rx) = mpsc::channel(32);

    let err = engine.run(&wf, BTreeMap::new(), tx).await.unwrap_err();
    assert!(format!("{err}").contains("must_have"), "{err}");
}

#[tokio::test]
async fn output_uses_template_substitution() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path());

    let yaml = r#"
id: chained
inputs:
  who:
    type: string
    default: "world"
steps:
  - id: t
    type: shell
    command: "true"
outputs:
  greeting: "hello, ${{ inputs.who }}! exit=${{ steps.t.exit_code }}"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, _rx) = mpsc::channel(32);
    let outputs = engine.run(&wf, BTreeMap::new(), tx).await.unwrap();
    assert_eq!(
        outputs.values.get("greeting").map(String::as_str),
        Some("hello, world! exit=0"),
    );
}
