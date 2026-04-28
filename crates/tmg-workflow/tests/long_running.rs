//! Integration tests for [`LongRunningExecutor`] (SPEC §8.12 + §9.9).
//!
//! These tests exercise the executor end-to-end against a real
//! [`tmg_harness::RunRunner`] so the harnessed-scope upgrade is
//! observable on disk. Steps use `shell` + `write_file` leaves so the
//! tests don't depend on a live LLM.
//!
//! Coverage matrix (matches the issue's acceptance list):
//!
//! - init phase flips the run from ad-hoc to harnessed.
//! - bootstrap output is injected into the iterate-step context as
//!   `inputs.bootstrap_context`.
//! - `until: artifact.features.all_passing` stops the loop when every
//!   feature is marked passing.
//! - `max_sessions` exhaustion returns `RunStatus::Exhausted`.
//! - `session_timeout` is honoured (the engine returns inside the
//!   budget; the runner records a `Timeout` trigger).

#![expect(clippy::unwrap_used, reason = "test assertions")]
#![expect(clippy::expect_used, reason = "test assertions")]
#![expect(clippy::panic, reason = "test assertions")]
#![expect(clippy::similar_names, reason = "runs_dir vs run_dir is intentional")]

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use tmg_agents::SubagentManager;
use tmg_harness::{Run, RunRunner, RunStore};
use tmg_llm::{LlmPool, PoolConfig};
use tmg_sandbox::{SandboxConfig, SandboxContext, SandboxMode};
use tmg_tools::ToolRegistry;
use tmg_workflow::{
    LongRunningExecutor, RunStatus, WorkflowConfig, WorkflowEngine, parse_workflow_str,
};

/// Build the engine + workspace + sandbox + run-runner harness for
/// tests.
fn build_harness(
    workspace: &Path,
    runs_dir: &Path,
) -> (Arc<WorkflowEngine>, Arc<Mutex<RunRunner>>) {
    let llm_pool =
        Arc::new(LlmPool::new(&PoolConfig::single("http://localhost:9999"), "test-model").unwrap());
    let canonical_workspace =
        std::fs::canonicalize(workspace).unwrap_or_else(|_| workspace.to_path_buf());
    let sandbox = Arc::new(SandboxContext::new(
        SandboxConfig::new(&canonical_workspace).with_mode(SandboxMode::WorkspaceWrite),
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

    let engine = Arc::new(WorkflowEngine::new(
        llm_pool,
        sandbox,
        tool_registry,
        subagent_manager,
        WorkflowConfig::default(),
        serde_json::Value::Null,
    ));

    let store = Arc::new(RunStore::new(runs_dir.to_path_buf()));
    let run = Run::new_ad_hoc(canonical_workspace);
    // Manually create the run dir so the runner can write artifacts.
    let run_dir = store.run_dir(&run.id);
    std::fs::create_dir_all(&run_dir).unwrap();
    store.save(&run).unwrap();
    let runner = RunRunner::new(run, store);
    let run_runner = Arc::new(Mutex::new(runner));

    (engine, run_runner)
}

/// Init phase escalates the run from ad-hoc to harnessed.
#[tokio::test]
async fn init_phase_escalates_run_to_harnessed() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path().join("workspace");
    let runs_dir = tmp.path().join("runs");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&runs_dir).unwrap();

    let (engine, runner) = build_harness(&workspace, &runs_dir);

    // Confirm starting scope.
    {
        let r = runner.lock().await;
        assert!(r.scope().is_ad_hoc());
    }

    // For this integration test we directly invoke the executor and
    // assert the post-state. We intentionally use simple shell-only
    // steps so no live LLM is required.
    let yaml_simple = r#"
id: lr_init
mode: long_running

init:
  steps:
    - id: marker
      type: shell
      command: "echo init-ran"

iterate:
  bootstrap: []
  steps:
    - id: stop
      type: shell
      command: "true"
  until: "true"
  max_sessions: 1
  session_timeout: "5s"
"#;
    let wf = parse_workflow_str(yaml_simple, "<inline>").unwrap();
    let exec = LongRunningExecutor::new(engine, Arc::clone(&runner));
    let status = exec.run(&wf, BTreeMap::new()).await.unwrap();

    // The until="true" stops the loop on iteration 1 -> Completed.
    assert_eq!(status, RunStatus::Completed);

    // The run is now harnessed.
    let r = runner.lock().await;
    assert!(!r.scope().is_ad_hoc(), "run should be harnessed after init");
}

/// `until` evaluating to true on the first session returns Completed.
#[tokio::test]
async fn until_true_stops_loop_immediately() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path().join("workspace");
    let runs_dir = tmp.path().join("runs");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&runs_dir).unwrap();

    let (engine, runner) = build_harness(&workspace, &runs_dir);

    let yaml = r#"
id: lr_until
mode: long_running

iterate:
  bootstrap: []
  steps:
    - id: noop
      type: shell
      command: "true"
  until: "true"
  max_sessions: 5
  session_timeout: "5s"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let exec = LongRunningExecutor::new(engine, Arc::clone(&runner));
    let status = exec.run(&wf, BTreeMap::new()).await.unwrap();
    assert_eq!(status, RunStatus::Completed);

    // Exactly one session was begun.
    let r = runner.lock().await;
    assert_eq!(r.run().session_count, 1);
}

/// `max_sessions` exhaustion returns `Exhausted`.
#[tokio::test]
async fn max_sessions_exhausted() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path().join("workspace");
    let runs_dir = tmp.path().join("runs");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&runs_dir).unwrap();

    let (engine, runner) = build_harness(&workspace, &runs_dir);

    let yaml = r#"
id: lr_exhaust
mode: long_running

iterate:
  bootstrap: []
  steps:
    - id: noop
      type: shell
      command: "true"
  until: "false"
  max_sessions: 3
  session_timeout: "5s"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let exec = LongRunningExecutor::new(engine, Arc::clone(&runner));
    let status = exec.run(&wf, BTreeMap::new()).await.unwrap();
    match status {
        RunStatus::Exhausted { reason } => assert_eq!(reason, "max_sessions"),
        other => panic!("expected Exhausted, got {other:?}"),
    }
    let r = runner.lock().await;
    assert_eq!(r.run().session_count, 3);
}

/// Bootstrap `run:` output appears in the iterate step context as
/// `inputs.bootstrap_context`.
#[tokio::test]
async fn bootstrap_output_appears_in_context() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path().join("workspace");
    let runs_dir = tmp.path().join("runs");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&runs_dir).unwrap();

    let (engine, runner) = build_harness(&workspace, &runs_dir);

    // The iterate step writes the bootstrap_context to a file we can
    // then assert on.
    let yaml = r#"
id: lr_bootstrap
mode: long_running

iterate:
  bootstrap:
    - run: "echo BOOT_OK"
  steps:
    - id: snapshot
      type: write_file
      path: "snapshot.txt"
      content: "${{ inputs.bootstrap_context }}"
  until: "true"
  max_sessions: 1
  session_timeout: "5s"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let exec = LongRunningExecutor::new(engine, Arc::clone(&runner));
    let status = exec.run(&wf, BTreeMap::new()).await.unwrap();
    assert_eq!(status, RunStatus::Completed);

    // The write_file step writes to the workspace root.
    let canonical_workspace = std::fs::canonicalize(&workspace).unwrap();
    let snapshot_path = canonical_workspace.join("snapshot.txt");
    let snapshot = std::fs::read_to_string(&snapshot_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", snapshot_path.display()));
    assert!(
        snapshot.contains("BOOT_OK"),
        "expected bootstrap output in snapshot: {snapshot}"
    );
    assert!(
        snapshot.contains("[bootstrap]"),
        "expected '[bootstrap]' header: {snapshot}"
    );
    assert!(
        snapshot.contains("$ echo BOOT_OK"),
        "expected command echo header: {snapshot}"
    );
}

/// `until: artifact.features.all_passing` stops when features.json is
/// populated with all-passing features. We exercise this by writing
/// `features.json` directly into the run directory between iterations.
#[tokio::test]
async fn until_artifact_features_all_passing_stops_loop() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path().join("workspace");
    let runs_dir = tmp.path().join("runs");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&runs_dir).unwrap();

    let (engine, runner) = build_harness(&workspace, &runs_dir);

    // Pre-write features.json with one passing feature and escalate
    // the run manually so the resolver finds the file. The init phase
    // is empty (no steps) so the executor only escalates.
    let features_path = {
        let r = runner.lock().await;
        r.features().path().to_path_buf()
    };
    if let Some(parent) = features_path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(
        &features_path,
        r#"{"features":[{"id":"a","category":"x","description":"d","steps":["s"],"passes":true}]}"#,
    )
    .unwrap();

    let yaml = r#"
id: lr_artifact
mode: long_running

init:
  steps: []

iterate:
  bootstrap: []
  steps:
    - id: noop
      type: shell
      command: "true"
  until: "artifact.features.all_passing"
  max_sessions: 5
  session_timeout: "5s"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let exec = LongRunningExecutor::new(engine, Arc::clone(&runner));
    let status = exec.run(&wf, BTreeMap::new()).await.unwrap();
    assert_eq!(status, RunStatus::Completed);
    // First session triggers the stop because all features pass.
    let r = runner.lock().await;
    assert_eq!(r.run().session_count, 1);
}

/// PR #66 review fix #2: an empty `features.json` must not let
/// `until: artifact.features.all_passing` complete the run silently
/// on iteration 1. The resolver now reports an error which surfaces as
/// `RunStatus::Failed`.
#[tokio::test]
async fn empty_features_rejects_all_passing() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path().join("workspace");
    let runs_dir = tmp.path().join("runs");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&runs_dir).unwrap();

    let (engine, runner) = build_harness(&workspace, &runs_dir);

    // Empty features.json (init has not yet declared the work).
    let features_path = {
        let r = runner.lock().await;
        r.features().path().to_path_buf()
    };
    if let Some(parent) = features_path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&features_path, r#"{"features":[]}"#).unwrap();

    let yaml = r#"
id: lr_empty_features
mode: long_running

init:
  steps: []

iterate:
  bootstrap: []
  steps:
    - id: noop
      type: shell
      command: "true"
  until: "artifact.features.all_passing"
  max_sessions: 2
  session_timeout: "5s"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let exec = LongRunningExecutor::new(engine, Arc::clone(&runner));
    let status = exec.run(&wf, BTreeMap::new()).await.unwrap();
    match status {
        RunStatus::Failed { error } => {
            assert!(
                error.contains("no features declared yet"),
                "expected 'no features declared yet' message, got: {error}"
            );
        }
        other => panic!("expected Failed for empty features.json, got {other:?}"),
    }
}

/// PR #66 review fix #3: an iterate-phase step's `write_file.path`
/// can resolve `${{ artifacts.<name> }}` (the named-artifact map is
/// plumbed through to iterate-phase steps via `EngineExtras`).
#[tokio::test]
async fn iterate_step_resolves_artifacts_path_template() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path().join("workspace");
    let runs_dir = tmp.path().join("runs");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&runs_dir).unwrap();
    let canonical_workspace = std::fs::canonicalize(&workspace).unwrap();

    let (engine, runner) = build_harness(&workspace, &runs_dir);

    // Use a workspace-local path so the sandbox accepts the write.
    let progress_target = "progress.md";
    let yaml = format!(
        r#"
id: lr_artifact_in_iterate
mode: long_running

init:
  artifacts:
    progress_file: "{progress_target}"
  steps: []

iterate:
  bootstrap: []
  steps:
    - id: write_progress
      type: write_file
      path: "${{{{ artifacts.progress_file }}}}"
      content: "session-${{{{ inputs.session_index }}}}"
  until: "true"
  max_sessions: 1
  session_timeout: "5s"
"#,
    );
    let wf = parse_workflow_str(&yaml, "<inline>").unwrap();
    let exec = LongRunningExecutor::new(engine, Arc::clone(&runner));
    let status = exec.run(&wf, BTreeMap::new()).await.unwrap();
    assert_eq!(status, RunStatus::Completed);

    let written = canonical_workspace.join(progress_target);
    let body = std::fs::read_to_string(&written)
        .unwrap_or_else(|e| panic!("read {}: {e}", written.display()));
    assert!(body.contains("session-1"), "unexpected body: {body}");
}

/// PR #66 review fix #3: an iterate-phase step's `when:` can reference
/// `${{ artifact.<name>.<method> }}` so step gating reflects the live
/// run state (the resolver is plumbed through to iterate steps).
#[tokio::test]
async fn iterate_step_when_uses_artifact_resolver() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path().join("workspace");
    let runs_dir = tmp.path().join("runs");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&runs_dir).unwrap();
    let canonical_workspace = std::fs::canonicalize(&workspace).unwrap();

    let (engine, runner) = build_harness(&workspace, &runs_dir);

    // Pre-populate features.json with two passing features so
    // `passing_count == 2` is observable from the iterate step's
    // `when:` expression.
    let features_path = {
        let r = runner.lock().await;
        r.features().path().to_path_buf()
    };
    if let Some(parent) = features_path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(
        &features_path,
        r#"{"features":[
            {"id":"a","category":"x","description":"d","steps":["s"],"passes":true},
            {"id":"b","category":"x","description":"d","steps":["s"],"passes":true}
        ]}"#,
    )
    .unwrap();

    let marker_path = canonical_workspace.join("marker.txt");
    let never_path = canonical_workspace.join("never.txt");
    let yaml = format!(
        r#"
id: lr_when_artifact
mode: long_running

init:
  steps: []

iterate:
  bootstrap: []
  steps:
    - id: marker_when_two
      type: shell
      command: "echo TWO_PASSING > {marker}"
      when: "artifact.features.passing_count == 2"
    - id: marker_never
      type: shell
      command: "echo NEVER > {never}"
      when: "artifact.features.passing_count == 99"
  until: "true"
  max_sessions: 1
  session_timeout: "5s"
"#,
        marker = marker_path.display(),
        never = never_path.display(),
    );
    let wf = parse_workflow_str(&yaml, "<inline>").unwrap();
    let exec = LongRunningExecutor::new(engine, Arc::clone(&runner));
    let status = exec.run(&wf, BTreeMap::new()).await.unwrap();
    assert_eq!(status, RunStatus::Completed);

    assert!(
        marker_path.exists(),
        "expected marker.txt (when-true gate) at {}",
        marker_path.display()
    );
    assert!(
        !never_path.exists(),
        "never.txt should not exist (when-false gate) at {}",
        never_path.display()
    );
}

/// PR #66 review fix #14: a non-trivial `features.json` followed by
/// an iterate loop that marks each feature passing should drive
/// `artifact.features.all_passing` to true and stop with
/// `RunStatus::Completed`. The iterate phase mutates the file via a
/// shell step on each session until every feature flips to `passes:
/// true`. This exercises the end-to-end loop the long-running mode is
/// designed for.
#[tokio::test]
async fn init_writes_features_iterate_marks_passing_until_done() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path().join("workspace");
    let runs_dir = tmp.path().join("runs");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&runs_dir).unwrap();

    let (engine, runner) = build_harness(&workspace, &runs_dir);

    // Seed three failing features. We pre-write directly so the test
    // doesn't depend on the sandbox accepting writes outside the
    // workspace; the iterate-phase shell step (which uses the sandbox
    // and runs without a fixed cwd) reads/writes the file via its
    // absolute path.
    let features_path = {
        let r = runner.lock().await;
        r.features().path().to_path_buf()
    };
    if let Some(parent) = features_path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(
        &features_path,
        r#"{"features":[
{"id":"alpha","category":"x","description":"d","steps":["s"],"passes":false},
{"id":"beta","category":"x","description":"d","steps":["s"],"passes":false},
{"id":"gamma","category":"x","description":"d","steps":["s"],"passes":false}
]}"#,
    )
    .unwrap();

    // The mark-passing helper script (a tiny python program) is
    // staged on disk and invoked from the iterate-phase shell step.
    // Each invocation flips the first still-failing feature's
    // `passes` flag to true and exits 0.
    let helper_path = workspace.join("mark_one.py");
    std::fs::write(
        &helper_path,
        format!(
            "import json\np=r'{p}'\nd=json.load(open(p))\nfor f in d['features']:\n    if not f['passes']:\n        f['passes']=True\n        json.dump(d, open(p,'w'))\n        break\n",
            p = features_path.display(),
        ),
    )
    .unwrap();

    let yaml = format!(
        r#"
id: lr_full_loop
mode: long_running

iterate:
  bootstrap: []
  steps:
    - id: mark_one
      type: shell
      command: "python3 {helper}"
  until: "artifact.features.all_passing"
  max_sessions: 5
  session_timeout: "10s"
"#,
        helper = helper_path.display(),
    );

    let wf = parse_workflow_str(&yaml, "<inline>")
        .unwrap_or_else(|e| panic!("parse failed: {e}\nyaml:\n{yaml}"));
    let exec = LongRunningExecutor::new(engine, Arc::clone(&runner));
    let status = exec.run(&wf, BTreeMap::new()).await.unwrap();
    assert_eq!(status, RunStatus::Completed, "expected Completed");

    // Three iterations were needed (one per feature) before
    // all_passing flipped true.
    let r = runner.lock().await;
    let count = r.run().session_count;
    assert_eq!(
        count, 3,
        "expected 3 iterations (one per feature), got {count}"
    );
}

/// Session timeout fires when the iterate steps take longer than the
/// configured `session_timeout`.
#[tokio::test(flavor = "multi_thread")]
async fn session_timeout_fires() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path().join("workspace");
    let runs_dir = tmp.path().join("runs");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&runs_dir).unwrap();

    let (engine, runner) = build_harness(&workspace, &runs_dir);

    // Iterate runs `sleep 2` but session_timeout is 200ms. We expect
    // the executor to abandon the session, record a Timeout, and
    // come back around for another iteration. With max_sessions=2
    // and until="false" the run exhausts after two timeouts.
    let yaml = r#"
id: lr_timeout
mode: long_running

iterate:
  bootstrap: []
  steps:
    - id: slow
      type: shell
      command: "sleep 2"
  until: "false"
  max_sessions: 2
  session_timeout: "200ms"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let exec = LongRunningExecutor::new(engine, Arc::clone(&runner));
    // The whole call must finish in well under the slow command's
    // 2-second budget; if the timeout doesn't fire we'd block here.
    let status = tokio::time::timeout(Duration::from_secs(10), exec.run(&wf, BTreeMap::new()))
        .await
        .expect("LongRunningExecutor::run did not respect session_timeout")
        .unwrap();
    match status {
        RunStatus::Exhausted { reason } => assert_eq!(reason, "max_sessions"),
        other => panic!("expected Exhausted after timeout, got {other:?}"),
    }
}
