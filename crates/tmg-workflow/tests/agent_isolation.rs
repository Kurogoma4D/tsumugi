//! Integration test for the agent step's context-isolation invariant
//! (SPEC §8.7).
//!
//! Two complementary tests live in this directory:
//!
//! - this file: structural smoke test that exercises the engine →
//!   manager → runner wiring against an unreachable LLM endpoint.
//! - `agent_two_step_isolation.rs`: behavioural test against an
//!   in-process mock SSE server that captures the chat-completions
//!   payloads and asserts the second step's `messages` array contains
//!   exactly one user-role turn.
//!
//! Together they check that:
//!
//! 1. Each agent step calls `SubagentManager::spawn_with_notify`,
//!    which internally constructs a fresh `SubagentRunner` (see
//!    `tmg_agents::manager::SubagentManager::spawn_inner`). That
//!    runner's history is initialized with **only** the system prompt
//!    for the agent kind — there is no API surface that shares
//!    history across spawns. The freshness invariant is therefore
//!    guaranteed by construction.
//! 2. The engine actually goes through this code path. A failing
//!    agent step against an unreachable endpoint surfaces as
//!    `WorkflowError::StepFailed { step_id: "a", .. }` (the engine
//!    re-tags every step-handler error with the current `step_id` so
//!    the returned error and the `StepFailed` progress event agree).
//!    The error message keeps the original `agent error: ...` /
//!    `llm error: ...` chain for diagnostics.

#![expect(clippy::unwrap_used, reason = "test assertions")]
#![expect(clippy::panic, reason = "test assertions")]

use std::collections::BTreeMap;
use std::sync::Arc;

use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

use tmg_agents::SubagentManager;
use tmg_llm::{LlmPool, PoolConfig};
use tmg_sandbox::{SandboxConfig, SandboxContext, SandboxMode};
use tmg_tools::ToolRegistry;
use tmg_workflow::{WorkflowConfig, WorkflowEngine, WorkflowError, parse_workflow_str};

#[tokio::test]
async fn agent_step_routes_through_subagent_manager() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path().to_path_buf();

    // Use an unreachable port so the LLM call fails fast. We rely on
    // tokio's connect timeout to surface the failure as
    // `LlmError::Http` / `LlmError::Stream`.
    let endpoint = "http://127.0.0.1:1";
    let llm_pool = Arc::new(LlmPool::new(&PoolConfig::single(endpoint), "test-model").unwrap());
    let sandbox = Arc::new(SandboxContext::new(
        SandboxConfig::new(workspace.clone()).with_mode(SandboxMode::WorkspaceWrite),
    ));
    let tool_registry = Arc::new(ToolRegistry::new());

    let llm_client_cfg = tmg_llm::LlmClientConfig::new(endpoint, "test-model");
    let llm_client = tmg_llm::LlmClient::new(llm_client_cfg).unwrap();
    let cancel = CancellationToken::new();
    let manager = SubagentManager::new(llm_client, cancel, endpoint, "test-model");
    let subagent_manager = Arc::new(Mutex::new(manager));

    let engine = WorkflowEngine::new(
        llm_pool,
        sandbox,
        tool_registry,
        subagent_manager,
        WorkflowConfig::default(),
        serde_json::Value::Null,
    );

    // Single agent step against an unreachable endpoint. The engine
    // re-tags step-handler errors with the current `step_id`, so the
    // failure must surface as `StepFailed { step_id: "a", .. }` with
    // an "agent error: ..." / "llm error: ..." message threaded
    // through, demonstrating the engine → manager wiring.
    let yaml = r#"
id: agent_smoke
steps:
  - id: a
    type: agent
    subagent: explore
    prompt: "look around"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, _rx) = mpsc::channel(32);
    let result = engine.run(&wf, BTreeMap::new(), tx).await;

    let Err(err) = result else {
        panic!("expected error when LLM endpoint is unreachable");
    };
    // The engine re-tags every step-handler error with the current
    // step id, so the surfaced error is `StepFailed { step_id: "a",
    // .. }`. The original cause (an `AgentError::Llm`) is preserved
    // in the message string.
    let WorkflowError::StepFailed {
        ref step_id,
        ref message,
    } = err
    else {
        panic!("expected WorkflowError::StepFailed, got {err:?}");
    };
    assert_eq!(step_id, "a");
    assert!(
        message.contains("agent error") || message.contains("llm error"),
        "expected agent/llm cause in message, got: {message}"
    );
}

#[tokio::test]
async fn unknown_subagent_name_is_step_failure() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path().to_path_buf();

    let endpoint = "http://127.0.0.1:1";
    let llm_pool = Arc::new(LlmPool::new(&PoolConfig::single(endpoint), "test-model").unwrap());
    let sandbox = Arc::new(SandboxContext::new(
        SandboxConfig::new(workspace).with_mode(SandboxMode::WorkspaceWrite),
    ));
    let tool_registry = Arc::new(ToolRegistry::new());

    let llm_client_cfg = tmg_llm::LlmClientConfig::new(endpoint, "test-model");
    let llm_client = tmg_llm::LlmClient::new(llm_client_cfg).unwrap();
    let cancel = CancellationToken::new();
    let manager = SubagentManager::new(llm_client, cancel, endpoint, "test-model");
    let subagent_manager = Arc::new(Mutex::new(manager));

    let engine = WorkflowEngine::new(
        llm_pool,
        sandbox,
        tool_registry,
        subagent_manager,
        WorkflowConfig::default(),
        serde_json::Value::Null,
    );

    let yaml = r#"
id: unknown_agent
steps:
  - id: bad
    type: agent
    subagent: not_a_real_agent
    prompt: "hi"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, _rx) = mpsc::channel(32);
    let err = engine.run(&wf, BTreeMap::new(), tx).await.unwrap_err();
    assert!(
        matches!(err, WorkflowError::StepFailed { ref step_id, ref message }
            if step_id == "bad" && message.contains("unknown subagent")),
        "expected StepFailed for unknown subagent, got {err:?}"
    );
}
