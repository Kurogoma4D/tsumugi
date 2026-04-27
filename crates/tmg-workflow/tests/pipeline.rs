//! Integration tests for the declarative-pipeline form (`stages:`)
//! introduced in issue #41.
//!
//! Pipelines are workflows whose steps are
//! [`tmg_workflow::StepDef::Workflow`] entries. The engine resolves each
//! stage's target via the workflow index, runs it via
//! `WorkflowEngine::run`, and exposes the result as
//! `${{ stages.<id>.outputs.<key> }}` for downstream stages.

#![expect(clippy::unwrap_used, reason = "test assertions")]
#![expect(clippy::panic, reason = "test assertions")]

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;

use serde_json::Value;
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio_util::sync::CancellationToken;

use tmg_agents::SubagentManager;
use tmg_llm::{LlmPool, PoolConfig};
use tmg_sandbox::{SandboxConfig, SandboxContext, SandboxMode};
use tmg_tools::ToolRegistry;
use tmg_workflow::{
    StepDef, WorkflowConfig, WorkflowDef, WorkflowEngine, WorkflowProgress, parse_workflow_str,
};

fn build_engine(
    workspace: &Path,
    workflows: Vec<(&str, &str)>,
) -> (
    Arc<WorkflowEngine>,
    Arc<RwLock<HashMap<String, WorkflowDef>>>,
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

    let mut index_map: HashMap<String, WorkflowDef> = HashMap::new();
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

/// A two-stage pipeline chains output from the first stage into the
/// second via `${{ stages.<id>.outputs.<key> }}`.
#[tokio::test]
async fn two_stage_pipeline_chains_via_stage_outputs() {
    let tmp = tempfile::tempdir().unwrap();
    let stage_a = r#"
id: stage_a
inputs:
  greeting: { type: string, default: "hi" }
steps:
  - id: produce
    type: shell
    command: "echo prepared"
outputs:
  message: "${{ inputs.greeting }} from a"
"#;
    let stage_b = r#"
id: stage_b
inputs:
  carry: { type: string, default: "" }
steps:
  - id: consume
    type: shell
    command: "echo consume"
outputs:
  echo: "received: ${{ inputs.carry }}"
"#;
    let pipeline = r#"
id: pipeline
stages:
  - id: first
    workflow: stage_a
    inputs:
      greeting: "hello"
  - id: second
    workflow: stage_b
    inputs:
      carry: "${{ stages.first.outputs.message }}"
outputs:
  final: "${{ stages.second.outputs.echo }}"
"#;
    let (engine, _index) = build_engine(
        tmp.path(),
        vec![
            ("stage_a", stage_a),
            ("stage_b", stage_b),
            ("pipeline", pipeline),
        ],
    );

    let pipeline_def = parse_workflow_str(pipeline, "<pipeline>").unwrap();
    // Sanity: the parser produced a Workflow step variant.
    match &pipeline_def.steps[0] {
        StepDef::Workflow {
            id,
            workflow_id,
            inputs,
            ..
        } => {
            assert_eq!(id, "first");
            assert_eq!(workflow_id, "stage_a");
            assert_eq!(inputs.get("greeting").map(String::as_str), Some("hello"));
        }
        other => panic!("expected Workflow step, got {other:?}"),
    }

    let (tx, _rx) = mpsc::channel(64);
    let outputs = engine
        .run(&pipeline_def, BTreeMap::new(), tx)
        .await
        .unwrap();
    assert_eq!(
        outputs.values.get("final").map(String::as_str),
        Some("received: hello from a")
    );
}

/// A pipeline stage wrapped in `loop:` runs the target workflow up to
/// its `max_iterations` cap and emits `LoopIteration` events.
#[tokio::test]
async fn pipeline_loop_iterates_until_satisfied() {
    let tmp = tempfile::tempdir().unwrap();
    // A trivial stage workflow whose output is constant; the loop will
    // run until `max_iterations` since `until` never matches.
    let counter_yaml = r#"
id: counter
steps:
  - id: tick
    type: shell
    command: "echo tick"
outputs:
  size: "${{ steps.tick.stdout }}"
"#;
    let pipeline_yaml = r#"
id: looper
stages:
  - id: grow
    workflow: counter
    loop:
      max_iterations: 3
      until: "stages.grow.outputs.size == \"never\""
outputs:
  reached: "${{ stages.grow.outputs.size }}"
"#;
    let (engine, _index) = build_engine(
        tmp.path(),
        vec![("counter", counter_yaml), ("looper", pipeline_yaml)],
    );
    let pipeline_def = parse_workflow_str(pipeline_yaml, "<looper>").unwrap();

    let (tx, mut rx) = mpsc::channel(128);
    let outputs = engine
        .run(&pipeline_def, BTreeMap::new(), tx)
        .await
        .unwrap();
    // The loop hits max-iterations since `until` never matches.
    assert!(
        outputs.values.get("reached").is_some_and(|v| !v.is_empty()),
        "expected non-empty reached output, got: {outputs:?}"
    );

    // The progress stream should include at least one LoopIteration
    // event addressed to the `grow` stage.
    let mut events = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }
    let loop_count = events
        .iter()
        .filter(|ev| {
            matches!(
                ev,
                WorkflowProgress::LoopIteration { step_id, .. } if step_id == "grow"
            )
        })
        .count();
    assert!(
        loop_count >= 1,
        "expected at least one LoopIteration event for stage `grow`, got: {events:#?}"
    );
}

/// Referencing a stage that hasn't been declared raises a clear error
/// from the expression evaluator.
#[tokio::test]
async fn unknown_stage_reference_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let stage_yaml = r#"
id: stage_a
steps:
  - id: ok
    type: shell
    command: "echo a"
outputs:
  done: "yes"
"#;
    let pipeline_yaml = r#"
id: bad_pipeline
stages:
  - id: first
    workflow: stage_a
    inputs:
      from_missing: "${{ stages.never.outputs.x }}"
outputs:
  any: "x"
"#;
    let (engine, _index) = build_engine(
        tmp.path(),
        vec![("stage_a", stage_yaml), ("bad_pipeline", pipeline_yaml)],
    );
    let pipeline_def = parse_workflow_str(pipeline_yaml, "<bad>").unwrap();

    let (tx, _rx) = mpsc::channel(16);
    let result = engine.run(&pipeline_def, BTreeMap::new(), tx).await;
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("unknown stage 'never'") || msg.contains("rendering inputs"),
        "expected unknown-stage error, got: {msg}"
    );
}

/// Referencing a workflow id that's not in the index fails with a
/// clear error.
#[tokio::test]
async fn pipeline_unknown_workflow_id_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let pipeline_yaml = r#"
id: missing_target
stages:
  - id: first
    workflow: ghost
outputs:
  any: "x"
"#;
    let (engine, _index) = build_engine(tmp.path(), vec![("missing_target", pipeline_yaml)]);
    let pipeline_def = parse_workflow_str(pipeline_yaml, "<missing>").unwrap();

    let (tx, _rx) = mpsc::channel(16);
    let err = engine
        .run(&pipeline_def, BTreeMap::new(), tx)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("unknown workflow"), "got: {err}");
}
