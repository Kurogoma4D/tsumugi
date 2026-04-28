//! Integration tests for the `inject_files` field on agent steps.
//!
//! These tests cover the traversal rejection and template-expansion
//! invariants without requiring a live LLM. The agent step's
//! `inject_files` handling runs *before* the subagent is spawned, so
//! a `..`-containing path (or a sandbox-rejected path) surfaces as a
//! step failure before any network I/O.

#![expect(clippy::unwrap_used, reason = "test assertions")]
#![expect(clippy::expect_used, reason = "test assertions")]

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

use tmg_agents::{EndpointResolver, SubagentManager};
use tmg_llm::{LlmPool, PoolConfig};
use tmg_sandbox::{SandboxConfig, SandboxContext, SandboxMode};
use tmg_tools::ToolRegistry;
use tmg_workflow::{WorkflowConfig, WorkflowEngine, WorkflowError, parse_workflow_str};

/// Build the engine + a workspace-rooted sandbox for tests.
fn build_engine(workspace: &Path) -> WorkflowEngine {
    let endpoint = "http://127.0.0.1:1";
    let llm_pool = Arc::new(LlmPool::new(&PoolConfig::single(endpoint), "test-model").unwrap());
    let sandbox = Arc::new(SandboxContext::new(
        SandboxConfig::new(workspace).with_mode(SandboxMode::WorkspaceWrite),
    ));
    let tool_registry = Arc::new(ToolRegistry::new());

    let llm_client_cfg = tmg_llm::LlmClientConfig::new(endpoint, "test-model");
    let llm_client = tmg_llm::LlmClient::new(llm_client_cfg).unwrap();
    let cancel = CancellationToken::new();
    let manager = SubagentManager::new(
        llm_client,
        cancel,
        EndpointResolver::new(endpoint, "test-model"),
        Arc::clone(&sandbox),
    );
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

/// `..` traversal in `inject_files` must be rejected before the
/// subagent is spawned.
#[tokio::test]
async fn inject_files_rejects_parent_dir_traversal() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path());

    let yaml = r#"
id: traversal_check
steps:
  - id: probe
    type: agent
    subagent: explore
    prompt: "look around"
    inject_files:
      - "../../../etc/passwd"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, _rx) = mpsc::channel(32);
    let err = engine
        .run(&wf, BTreeMap::new(), tx)
        .await
        .expect_err("expected traversal rejection");

    // Engine re-tags non-StepFailed errors with `step_id` (see
    // fix #8). The underlying error must explain that `..` is not
    // allowed.
    let msg = err.to_string();
    assert!(
        msg.contains("'..'") || msg.contains("traversal"),
        "expected traversal-related message, got: {msg}"
    );
    // And the specific InvalidPath variant must be reachable in the
    // chain (either directly or wrapped via StepFailed).
    let raw_chain = format!("{err:?}");
    assert!(
        raw_chain.contains("InvalidPath") || raw_chain.contains("traversal"),
        "expected InvalidPath reachable, got: {raw_chain}"
    );
}

/// `inject_files` entries are templates: `${{ inputs.file }}` should
/// expand to the input value. We confirm the template ran correctly
/// by referencing a file that *does* exist via the input — the agent
/// step then proceeds past the file-read and fails on the unreachable
/// LLM endpoint, which proves the file was located via template
/// expansion.
#[tokio::test]
async fn inject_files_expands_templates() {
    let tmp = tempfile::tempdir().unwrap();
    // Create a real file inside the workspace.
    let target = tmp.path().join("hello.txt");
    std::fs::write(&target, "hi from inject").unwrap();

    let engine = build_engine(tmp.path());

    let yaml = r#"
id: tmpl_inject
inputs:
  file:
    type: string
    default: "hello.txt"
steps:
  - id: probe
    type: agent
    subagent: explore
    prompt: "use the file"
    inject_files:
      - "${{ inputs.file }}"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, _rx) = mpsc::channel(32);
    let err = engine
        .run(&wf, BTreeMap::new(), tx)
        .await
        .expect_err("LLM endpoint is unreachable");

    // If template expansion failed (e.g. literal `${{ inputs.file }}`
    // path), we'd see an InvalidPath / IO error pointing at the raw
    // template string. We expect to *bypass* file loading and reach
    // the LLM-call path, which fails with an `Agent` error.
    // Engine wraps every step error into StepFailed (#8 fix), so we
    // see StepFailed with the underlying agent/llm message threaded
    // through.
    let raw = format!("{err:?}");
    assert!(
        matches!(err, WorkflowError::StepFailed { .. }),
        "expected StepFailed (template expanded, file read OK, then unreachable LLM), got: {raw}"
    );
    // The error should NOT mention the un-expanded `${{` template.
    assert!(
        !raw.contains("${{"),
        "expected template to be expanded before path resolution: {raw}",
    );
}
