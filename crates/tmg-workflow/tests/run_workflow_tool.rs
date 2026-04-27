//! Integration tests for the `run_workflow` and `workflow_status`
//! tools (issue #41).
//!
//! These tests build a `WorkflowEngine` against an in-memory workflow
//! index (no on-disk discovery needed) and exercise the foreground /
//! background paths through the tool surface.

#![expect(clippy::unwrap_used, reason = "test assertions")]
#![expect(clippy::panic, reason = "test assertions")]
#![expect(clippy::manual_assert, reason = "explicit panic clearer here")]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;

use tmg_agents::SubagentManager;
use tmg_llm::{LlmPool, PoolConfig};
use tmg_sandbox::{SandboxConfig, SandboxContext, SandboxMode};
use tmg_tools::{Tool, ToolRegistry};
use tmg_workflow::{
    RunWorkflowTool, WorkflowConfig, WorkflowEngine, WorkflowStatusTool, parse_workflow_str, tools,
};

/// Build an engine + index with the given workflows.
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
    let manager = SubagentManager::new(llm_client, cancel, "http://localhost:9999", "test-model");
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

/// Foreground `run_workflow` returns the workflow's outputs as a JSON
/// object embedded in the tool's textual output.
#[tokio::test]
async fn foreground_returns_outputs() {
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
    let tool = RunWorkflowTool::new(engine, index, bg);

    let result = tool
        .execute(serde_json::json!({"workflow": "simple"}))
        .await
        .unwrap();
    assert!(!result.is_error, "got error result: {}", result.output);
    let parsed: Value = serde_json::from_str(&result.output).unwrap();
    assert_eq!(parsed.get("exit").and_then(Value::as_str), Some("0"));
}

/// `run_workflow` with an unknown workflow id returns a clear error
/// `ToolResult` (not a hard `ToolError`).
#[tokio::test]
async fn unknown_workflow_returns_clear_error() {
    let tmp = tempfile::tempdir().unwrap();
    let (engine, index) = build_engine(tmp.path(), vec![]);
    let bg = tools::new_background_runs();
    let tool = RunWorkflowTool::new(engine, index, bg);

    let result = tool
        .execute(serde_json::json!({"workflow": "nope"}))
        .await
        .unwrap();
    assert!(result.is_error);
    assert!(result.output.contains("unknown workflow"));
}

/// Background mode returns a `run_id` immediately and `workflow_status`
/// transitions from `running` to `completed`.
#[tokio::test]
async fn background_returns_run_id_and_completes() {
    let tmp = tempfile::tempdir().unwrap();
    let yaml = r#"
id: bg
steps:
  - id: produce
    type: shell
    command: "echo bg-done"
outputs:
  signal: "${{ steps.produce.exit_code }}"
"#;
    let (engine, index) = build_engine(tmp.path(), vec![("bg", yaml)]);
    let bg = tools::new_background_runs();
    let run_tool = RunWorkflowTool::new(Arc::clone(&engine), index, Arc::clone(&bg));
    let status_tool = WorkflowStatusTool::new(Arc::clone(&bg));

    let start = run_tool
        .execute(serde_json::json!({"workflow": "bg", "background": true}))
        .await
        .unwrap();
    assert!(!start.is_error);
    let parsed: Value = serde_json::from_str(&start.output).unwrap();
    let run_id = parsed
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap()
        .to_owned();
    assert_eq!(run_id.len(), 8);
    assert_eq!(
        parsed.get("status").and_then(Value::as_str),
        Some("running")
    );

    // Poll until completed (with a generous bound).
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let result = status_tool
            .execute(serde_json::json!({"run_id": run_id.clone()}))
            .await
            .unwrap();
        let parsed: Value = serde_json::from_str(&result.output).unwrap();
        let status = parsed.get("status").and_then(Value::as_str).unwrap_or("");
        if status == "completed" {
            assert_eq!(
                parsed
                    .get("outputs")
                    .and_then(Value::as_object)
                    .and_then(|m| m.get("signal"))
                    .and_then(Value::as_str),
                Some("0")
            );
            return;
        }
        assert!(status == "running" || status == "completed");
        if std::time::Instant::now() > deadline {
            panic!("background workflow did not complete in time; last status = {parsed}");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Multiple concurrent background runs are addressable independently.
#[tokio::test]
async fn multiple_background_runs_addressable_independently() {
    let tmp = tempfile::tempdir().unwrap();
    let yaml = r#"
id: bg
steps:
  - id: produce
    type: shell
    command: "echo done"
outputs:
  ok: "${{ steps.produce.exit_code }}"
"#;
    let (engine, index) = build_engine(tmp.path(), vec![("bg", yaml)]);
    let bg = tools::new_background_runs();
    let run_tool = RunWorkflowTool::new(Arc::clone(&engine), index, Arc::clone(&bg));
    let status_tool = WorkflowStatusTool::new(Arc::clone(&bg));

    let mut run_ids = Vec::new();
    for _ in 0..3 {
        let start = run_tool
            .execute(serde_json::json!({"workflow": "bg", "background": true}))
            .await
            .unwrap();
        let parsed: Value = serde_json::from_str(&start.output).unwrap();
        run_ids.push(
            parsed
                .get("run_id")
                .and_then(Value::as_str)
                .unwrap()
                .to_owned(),
        );
    }
    // All run ids are unique.
    let mut sorted = run_ids.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 3);

    // Each is queryable.
    for rid in &run_ids {
        let r = status_tool
            .execute(serde_json::json!({"run_id": rid}))
            .await
            .unwrap();
        assert!(!r.is_error, "{}", r.output);
    }

    // Wait for completion.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let mut all_done = true;
        for rid in &run_ids {
            let r = status_tool
                .execute(serde_json::json!({"run_id": rid}))
                .await
                .unwrap();
            let parsed: Value = serde_json::from_str(&r.output).unwrap();
            if parsed.get("status").and_then(Value::as_str) != Some("completed") {
                all_done = false;
                break;
            }
        }
        if all_done {
            return;
        }
        if std::time::Instant::now() > deadline {
            panic!("not all background runs completed in time");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// `workflow_status` for an unknown `run_id` returns a clear error.
#[tokio::test]
async fn unknown_run_id_returns_clear_error() {
    let bg = tools::new_background_runs();
    let status_tool = WorkflowStatusTool::new(bg);

    let result = status_tool
        .execute(serde_json::json!({"run_id": "deadbeef"}))
        .await
        .unwrap();
    assert!(result.is_error);
    assert!(result.output.contains("unknown run_id"));
}
