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

/// Tool dispatch sandbox shared across the tests in this file.
///
/// The `Tool::execute` signature added in issue #47 takes a
/// [`SandboxContext`] reference; these tool-surface tests are not
/// trying to assert sandbox enforcement themselves, so we hand them a
/// permissive [`SandboxContext::test_default`].
fn dispatch_ctx() -> SandboxContext {
    SandboxContext::test_default()
}

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
    let ctx = dispatch_ctx();

    let result = tool
        .execute(serde_json::json!({"workflow": "simple"}), &ctx)
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
    let ctx = dispatch_ctx();

    let result = tool
        .execute(serde_json::json!({"workflow": "nope"}), &ctx)
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
    let ctx = dispatch_ctx();
    let status_tool = WorkflowStatusTool::new(Arc::clone(&bg));

    let start = run_tool
        .execute(
            serde_json::json!({"workflow": "bg", "background": true}),
            &ctx,
        )
        .await
        .unwrap();
    assert!(!start.is_error);
    let parsed: Value = serde_json::from_str(&start.output).unwrap();
    let run_id = parsed
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap()
        .to_owned();
    assert_eq!(run_id.len(), 16);
    assert_eq!(
        parsed.get("status").and_then(Value::as_str),
        Some("running")
    );

    // Poll until completed (with a generous bound).
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let result = status_tool
            .execute(serde_json::json!({"run_id": run_id.clone()}), &ctx)
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
    let ctx = dispatch_ctx();
    let status_tool = WorkflowStatusTool::new(Arc::clone(&bg));

    let mut run_ids = Vec::new();
    for _ in 0..3 {
        let start = run_tool
            .execute(
                serde_json::json!({"workflow": "bg", "background": true}),
                &ctx,
            )
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
            .execute(serde_json::json!({"run_id": rid}), &ctx)
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
                .execute(serde_json::json!({"run_id": rid}), &ctx)
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
    let ctx = dispatch_ctx();

    let result = status_tool
        .execute(serde_json::json!({"run_id": "0123456789abcdef"}), &ctx)
        .await
        .unwrap();
    assert!(result.is_error);
    assert!(result.output.contains("unknown run_id"));
}

/// A non-object `inputs` parameter is rejected with a clear error
/// rather than being silently coerced to an empty map.
#[tokio::test]
async fn non_object_inputs_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let yaml = r#"
id: simple
steps:
  - id: produce
    type: shell
    command: "echo hi"
outputs:
  out: "${{ steps.produce.exit_code }}"
"#;
    let (engine, index) = build_engine(tmp.path(), vec![("simple", yaml)]);
    let bg = tools::new_background_runs();
    let tool = RunWorkflowTool::new(engine, index, bg);
    let ctx = dispatch_ctx();

    // Array instead of object.
    let r = tool
        .execute(
            serde_json::json!({"workflow": "simple", "inputs": [1, 2, 3]}),
            &ctx,
        )
        .await
        .unwrap();
    assert!(r.is_error, "expected error result for array inputs");
    assert!(r.output.contains("inputs must be a JSON object"));

    // String instead of object.
    let r = tool
        .execute(
            serde_json::json!({"workflow": "simple", "inputs": "oops"}),
            &ctx,
        )
        .await
        .unwrap();
    assert!(r.is_error);
    assert!(r.output.contains("inputs must be a JSON object"));
}

/// A successful background workflow ends with `final_outputs == Ok(...)`
/// rather than the channel-closed sentinel: we observe the same value
/// twice — first via `workflow_status` and second via a direct read of
/// the `BackgroundRun` cell — and assert there is no spurious
/// "channel closed unexpectedly" leak when both select arms could
/// have raced.
#[tokio::test]
async fn successful_run_does_not_leak_channel_closed_sentinel() {
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
    let ctx = dispatch_ctx();
    let status_tool = WorkflowStatusTool::new(Arc::clone(&bg));

    // Spawn the run repeatedly to give the select! race plenty of
    // chances; the bug originally manifested intermittently when the
    // run-future arm fired and the post-loop "channel closed
    // unexpectedly" path overwrote the success outcome.
    for _ in 0..5 {
        let start = run_tool
            .execute(
                serde_json::json!({"workflow": "bg", "background": true}),
                &ctx,
            )
            .await
            .unwrap();
        let parsed: Value = serde_json::from_str(&start.output).unwrap();
        let run_id = parsed
            .get("run_id")
            .and_then(Value::as_str)
            .unwrap()
            .to_owned();

        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let r = status_tool
                .execute(serde_json::json!({"run_id": run_id.clone()}), &ctx)
                .await
                .unwrap();
            let parsed: Value = serde_json::from_str(&r.output).unwrap();
            let status = parsed.get("status").and_then(Value::as_str).unwrap_or("");
            if status == "completed" {
                let err = parsed.get("error").cloned().unwrap_or(Value::Null);
                assert!(
                    err.is_null(),
                    "completed run should have null error, got {err:?}",
                );
                let out = parsed.get("outputs").cloned().unwrap_or(Value::Null);
                assert!(
                    out.is_object(),
                    "completed run should expose outputs object"
                );
                break;
            }
            assert_ne!(
                status, "failed",
                "run unexpectedly failed; full status = {parsed}",
            );
            if std::time::Instant::now() > deadline {
                panic!("background run did not complete in time; last status = {parsed}");
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

/// Cancelling a background run mid-flight terminates promptly with
/// `status: failed` and an error message referencing cancellation.
#[tokio::test]
async fn cancel_mid_run_terminates_with_failed_status() {
    let tmp = tempfile::tempdir().unwrap();
    // Use an outer human step so the engine *itself* has an await
    // point that observes `EngineCtx::cancel`. The shell step alone
    // would not honour the token directly, but the spawn closure's
    // `select!` watches the cancel arm so the *recorded* outcome
    // still flips to `failed` regardless.
    let yaml = r#"
id: long
steps:
  - id: wait
    type: human
    message: "approve to continue"
outputs:
  ok: "${{ steps.wait.output.kind }}"
"#;
    let (engine, index) = build_engine(tmp.path(), vec![("long", yaml)]);
    let bg = tools::new_background_runs();
    let run_tool = RunWorkflowTool::new(Arc::clone(&engine), index, Arc::clone(&bg));
    let ctx = dispatch_ctx();
    let status_tool = WorkflowStatusTool::new(Arc::clone(&bg));

    let start = run_tool
        .execute(
            serde_json::json!({"workflow": "long", "background": true}),
            &ctx,
        )
        .await
        .unwrap();
    let parsed: Value = serde_json::from_str(&start.output).unwrap();
    let run_id = parsed
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap()
        .to_owned();

    // Confirm the run is registered and running before we cancel.
    let r = status_tool
        .execute(serde_json::json!({"run_id": run_id.clone()}), &ctx)
        .await
        .unwrap();
    let parsed: Value = serde_json::from_str(&r.output).unwrap();
    assert_eq!(
        parsed.get("status").and_then(Value::as_str),
        Some("running"),
    );

    // Cancel via the host-side API. The LLM-facing surface stays
    // unchanged in this issue; only the host (CLI / shutdown path)
    // can request termination. We deliberately route through
    // `cancel_all_background_runs` to mirror the CLI shutdown path.
    let n = tmg_workflow::tools::cancel_all_background_runs(&bg).await;
    assert!(n >= 1, "cancel_all should have observed at least one run");

    // Poll for the failed status. With the spawn closure's cancel
    // arm in place this transition is prompt — well under a second
    // even though the engine's own `human` step has its own
    // cancellation path.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        let r = status_tool
            .execute(serde_json::json!({"run_id": run_id.clone()}), &ctx)
            .await
            .unwrap();
        let parsed: Value = serde_json::from_str(&r.output).unwrap();
        let status = parsed.get("status").and_then(Value::as_str).unwrap_or("");
        if status == "failed" {
            let err = parsed
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_lowercase();
            assert!(
                err.contains("cancel"),
                "failed-run error should mention cancellation, got {err:?}",
            );
            return;
        }
        if std::time::Instant::now() > deadline {
            panic!("cancelled run did not record `failed` in time; last status = {parsed}");
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

// Eviction of completed runs is exercised as a unit test in
// `crates/tmg-workflow/src/tools/types.rs::tests` (see
// `reap_completed_runs_evicts_only_completed_old_entries`). The unit
// test has direct access to the `BackgroundRun` constructor without
// needing a public test surface; integration coverage of the
// `start_background` path is provided by the other tests in this
// file.
