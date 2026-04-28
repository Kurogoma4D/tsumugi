//! Integration tests for the control-flow step set added in issue #40
//! (`loop`, `branch`, `parallel`, `group`, `human`).
//!
//! These exercise the engine end-to-end via `shell` + `write_file`
//! leaves so they don't depend on a live LLM. The agent path is
//! validated separately in `agent_isolation.rs`.

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
use tmg_workflow::{
    HumanResponse, WorkflowConfig, WorkflowEngine, WorkflowProgress, parse_workflow_str,
};

/// Build the engine + a temp workspace + matching sandbox for tests.
fn build_engine(workspace: &Path, max_parallel_agents: u32) -> WorkflowEngine {
    let llm_pool =
        Arc::new(LlmPool::new(&PoolConfig::single("http://localhost:9999"), "test-model").unwrap());
    // Canonicalize the workspace path so the sandbox's write-check
    // (which canonicalizes target paths) sees a matching prefix. On
    // macOS `/var/folders/...` is a symlink to `/private/var/...`,
    // and a re-write of the same file fails the prefix check
    // otherwise.
    let workspace_canonical =
        std::fs::canonicalize(workspace).unwrap_or_else(|_| workspace.to_path_buf());
    let sandbox = Arc::new(SandboxContext::new(
        SandboxConfig::new(&workspace_canonical).with_mode(SandboxMode::WorkspaceWrite),
    ));
    let tool_registry = Arc::new(ToolRegistry::new());

    let llm_client_cfg = tmg_llm::LlmClientConfig::new("http://localhost:9999", "test-model");
    let llm_client = tmg_llm::LlmClient::new(llm_client_cfg).unwrap();
    let cancel = CancellationToken::new();
    let manager = SubagentManager::new(
        llm_client,
        cancel,
        EndpointResolver::new("http://localhost:9999", "test-model"),
        Arc::clone(&sandbox),
    );
    let subagent_manager = Arc::new(Mutex::new(manager));

    let config = WorkflowConfig {
        max_parallel_agents,
        ..WorkflowConfig::default()
    };
    WorkflowEngine::new(
        llm_pool,
        sandbox,
        tool_registry,
        subagent_manager,
        config,
        serde_json::Value::Null,
    )
}

fn drain<T>(rx: &mut mpsc::Receiver<T>) -> Vec<T> {
    let mut out = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        out.push(ev);
    }
    out
}

// ---------------------------------------------------------------------
// Loop
// ---------------------------------------------------------------------

#[tokio::test]
async fn loop_until_succeeds_after_third_iteration() {
    // Use a counter file in the workspace; each iteration appends a
    // line, and `until` becomes true once the counter file contains
    // the magic third line. Since the engine evaluates `until` over
    // the most recent iteration's `steps.<id>.exit_code`, we use shell
    // exit codes directly: iter1/iter2 -> exit 1, iter3 -> exit 0.
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path(), 2);

    let counter = tmp.path().join("counter");
    std::fs::write(&counter, "0").unwrap();
    let counter_path = counter.to_string_lossy().to_string();

    let yaml = format!(
        r#"
id: loop_succeeds
steps:
  - id: l
    type: loop
    max_iterations: 5
    until: "steps.bump.exit_code == 0"
    steps:
      - id: bump
        type: shell
        command: |
          n=$(cat {counter_path})
          n=$((n+1))
          echo $n > {counter_path}
          if [ $n -ge 3 ]; then exit 0; else exit 1; fi
"#
    );
    let wf = parse_workflow_str(&yaml, "<inline>").unwrap();
    let (tx, mut rx) = mpsc::channel(64);

    engine.run(&wf, BTreeMap::new(), tx).await.unwrap();

    let events = drain(&mut rx);
    let iters = events
        .iter()
        .filter(|e| matches!(e, WorkflowProgress::LoopIteration { .. }))
        .count();
    assert_eq!(iters, 3, "expected exactly three iterations");

    // Ensure the loop step's StepResult reports no max_iterations_reached.
    let final_loop_result = events
        .iter()
        .find_map(|e| match e {
            WorkflowProgress::StepCompleted { step_id, result } if step_id == "l" => Some(result),
            _ => None,
        })
        .expect("loop step completion event");
    assert_eq!(
        final_loop_result.output["max_iterations_reached"],
        serde_json::json!(false)
    );
    assert_eq!(final_loop_result.output["iterations"], serde_json::json!(3));
}

#[tokio::test]
async fn loop_max_iterations_reached_branch() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path(), 2);
    let yaml = r#"
id: loop_caps
steps:
  - id: l
    type: loop
    max_iterations: 2
    until: "false"
    steps:
      - id: noop
        type: shell
        command: "true"
outputs:
  hit_cap: "${{ steps.l.output.max_iterations_reached }}"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, mut rx) = mpsc::channel(32);
    let outputs = engine.run(&wf, BTreeMap::new(), tx).await.unwrap();
    let events = drain(&mut rx);
    let iters = events
        .iter()
        .filter(|e| matches!(e, WorkflowProgress::LoopIteration { .. }))
        .count();
    assert_eq!(iters, 2);
    assert_eq!(
        outputs.values.get("hit_cap").map(String::as_str),
        Some("true")
    );
}

#[tokio::test]
async fn loop_ref_substitution_runs_referenced_step() {
    // A `ref:` to a previously-defined shell step should clone the
    // step into the loop body and run it with iteration suffixes.
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path(), 2);
    let yaml = r#"
id: loop_ref
steps:
  - id: ping
    type: shell
    command: "echo pong"
  - id: l
    type: loop
    max_iterations: 2
    until: "false"
    steps:
      - ref: ping
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, mut rx) = mpsc::channel(32);
    engine.run(&wf, BTreeMap::new(), tx).await.unwrap();

    let events = drain(&mut rx);
    // Each iteration tags the inner ping step as `ping[1]`, `ping[2]`.
    let mut tagged: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            WorkflowProgress::StepStarted { step_id, .. } if step_id.starts_with("ping[") => {
                Some(step_id.clone())
            }
            _ => None,
        })
        .collect();
    tagged.sort();
    assert_eq!(tagged, vec!["ping[1]".to_owned(), "ping[2]".to_owned()]);
}

// ---------------------------------------------------------------------
// Branch
// ---------------------------------------------------------------------

#[tokio::test]
async fn branch_first_match_wins() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path(), 2);
    let yaml = r#"
id: br
inputs:
  flag:
    type: integer
    default: 2
steps:
  - id: choose
    type: branch
    conditions:
      - when: "inputs.flag == 1"
        steps:
          - id: a
            type: write_file
            path: "a.txt"
            content: "A"
      - when: "inputs.flag == 2"
        steps:
          - id: b
            type: write_file
            path: "b.txt"
            content: "B"
      - when: "true"
        steps:
          - id: c
            type: write_file
            path: "c.txt"
            content: "C"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, _rx) = mpsc::channel(32);
    engine.run(&wf, BTreeMap::new(), tx).await.unwrap();
    assert!(!tmp.path().join("a.txt").exists());
    assert!(tmp.path().join("b.txt").exists());
    assert!(!tmp.path().join("c.txt").exists());
}

#[tokio::test]
async fn branch_default_runs_when_no_match() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path(), 2);
    let yaml = r#"
id: brd
steps:
  - id: choose
    type: branch
    conditions:
      - when: "false"
        steps:
          - id: a
            type: write_file
            path: "a.txt"
            content: "A"
    default:
      - id: d
        type: write_file
        path: "d.txt"
        content: "D"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, _rx) = mpsc::channel(32);
    engine.run(&wf, BTreeMap::new(), tx).await.unwrap();
    assert!(!tmp.path().join("a.txt").exists());
    assert!(tmp.path().join("d.txt").exists());
}

#[tokio::test]
async fn branch_no_default_is_noop() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path(), 2);
    let yaml = r#"
id: brn
steps:
  - id: choose
    type: branch
    conditions:
      - when: "false"
        steps:
          - id: a
            type: write_file
            path: "a.txt"
            content: "A"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, _rx) = mpsc::channel(32);
    engine.run(&wf, BTreeMap::new(), tx).await.unwrap();
    assert!(!tmp.path().join("a.txt").exists());
}

// ---------------------------------------------------------------------
// Parallel
// ---------------------------------------------------------------------

#[tokio::test]
async fn parallel_shell_children_run_concurrently() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path(), 1); // agent cap doesn't matter for shell
    let yaml = r#"
id: parsh
steps:
  - id: p
    type: parallel
    steps:
      - id: a
        type: write_file
        path: "a.txt"
        content: "A"
      - id: b
        type: write_file
        path: "b.txt"
        content: "B"
      - id: c
        type: write_file
        path: "c.txt"
        content: "C"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, _rx) = mpsc::channel(64);
    engine.run(&wf, BTreeMap::new(), tx).await.unwrap();
    assert!(tmp.path().join("a.txt").exists());
    assert!(tmp.path().join("b.txt").exists());
    assert!(tmp.path().join("c.txt").exists());
}

#[tokio::test]
async fn parallel_failing_child_cancels_siblings() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path(), 2);
    // One child writes a file, the other exits non-zero. The shell
    // step's exit_code is recorded but does not raise StepFailed —
    // shell failures don't error the workflow. To force a hard
    // failure we use `write_file` with a bad path traversal.
    let yaml = r#"
id: par_fail
steps:
  - id: p
    type: parallel
    steps:
      - id: ok
        type: write_file
        path: "ok.txt"
        content: "ok"
      - id: bad
        type: write_file
        path: "../escape.txt"
        content: "boom"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, _rx) = mpsc::channel(64);
    let result = engine.run(&wf, BTreeMap::new(), tx).await;
    assert!(result.is_err(), "parallel must surface child failure");
}

// ---------------------------------------------------------------------
// Group
// ---------------------------------------------------------------------

#[tokio::test]
async fn group_abort_propagates_failure() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path(), 2);
    let yaml = r#"
id: g_abort
steps:
  - id: g
    type: group
    on_failure: abort
    steps:
      - id: bad
        type: write_file
        path: "../escape.txt"
        content: "boom"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, _rx) = mpsc::channel(32);
    let res = engine.run(&wf, BTreeMap::new(), tx).await;
    assert!(res.is_err());
}

#[tokio::test]
async fn group_continue_logs_failure_and_proceeds() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path(), 2);
    let yaml = r#"
id: g_cont
steps:
  - id: g
    type: group
    on_failure: continue
    steps:
      - id: bad
        type: write_file
        path: "../escape.txt"
        content: "boom"
  - id: after
    type: write_file
    path: "after.txt"
    content: "OK"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, _rx) = mpsc::channel(32);
    let res = engine.run(&wf, BTreeMap::new(), tx).await;
    assert!(res.is_ok(), "continue must not abort");
    assert!(tmp.path().join("after.txt").exists());
}

#[tokio::test]
async fn group_retry_succeeds_after_failures() {
    // The retry test needs a *deterministic* failure that goes away
    // after N attempts. We use:
    //   step `tick`   — increments a counter file (always succeeds).
    //   step `gate`   — write_file whose templated path is `..`
    //                   (sandbox-rejected) until the counter reaches
    //                   the threshold, after which it points to a
    //                   workspace-relative file.
    //
    // The engine restores `step_results` to the pre-group snapshot
    // before each retry, but the *filesystem* counter persists across
    // attempts, so the retry condition makes monotonic progress.
    let tmp = tempfile::tempdir().unwrap();
    let workspace = std::fs::canonicalize(tmp.path()).unwrap();
    let engine = build_engine(&workspace, 2);
    let counter = workspace.join("retry_counter");
    let path_marker = workspace.join("path.txt");
    std::fs::write(&counter, "0").unwrap();
    let counter_path = counter.to_string_lossy().to_string();
    let path_marker_str = path_marker.to_string_lossy().to_string();
    let yaml = format!(
        r#"
id: g_retry
steps:
  - id: g
    type: group
    on_failure: retry
    max_retries: 5
    steps:
      - id: tick
        type: shell
        command: |
          n=$(cat {counter_path})
          n=$((n+1))
          echo $n > {counter_path}
          if [ $n -ge 3 ]; then echo good > {path_marker_str}; else echo "../escape" > {path_marker_str}; fi
      - id: route
        type: shell
        command: "cat {path_marker_str}"
      - id: maybe_fail
        type: write_file
        path: "${{{{ steps.route.stdout }}}}.txt"
        content: ""
"#
    );
    // route.stdout has a trailing newline so the path becomes
    // "good\n.txt" or "../escape\n.txt". The "../escape" prefix is
    // rejected by the path-traversal check; the "good" case writes
    // "good\n.txt" which succeeds (newline is a valid path char on
    // Unix). Counter increments to 3 to break the retry loop.
    let wf = parse_workflow_str(&yaml, "<inline>").unwrap();
    let (tx, _rx) = mpsc::channel(64);
    let res = engine.run(&wf, BTreeMap::new(), tx).await;
    assert!(
        res.is_ok(),
        "group retry should eventually succeed; got {res:?}"
    );
    // Counter must have been bumped at least to the success threshold.
    let final_n: i32 = std::fs::read_to_string(&counter)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert!(final_n >= 3, "counter advanced to {final_n}, expected >= 3");
}

// ---------------------------------------------------------------------
// Human
// ---------------------------------------------------------------------

#[tokio::test]
async fn human_approve_continues_workflow() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path(), 2);
    let yaml = r#"
id: h_approve
steps:
  - id: prep
    type: write_file
    path: "before.txt"
    content: "before"
  - id: r
    type: human
    message: "ok?"
    options: [approve, reject]
  - id: after
    type: write_file
    path: "after.txt"
    content: "after"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, mut rx) = mpsc::channel(64);

    let run_handle = tokio::spawn({
        let inputs = BTreeMap::new();
        let wf = wf.clone();
        async move { engine.run(&wf, inputs, tx).await }
    });

    // Drain progress events until we see the HumanInputRequired.
    while let Some(ev) = rx.recv().await {
        if let WorkflowProgress::HumanInputRequired { response_tx, .. } = ev {
            let sender = response_tx.lock().await.take().unwrap();
            sender.send(HumanResponse::approve()).unwrap();
            break;
        }
    }

    // Drain the rest in the background (otherwise the engine blocks
    // on the channel) by spawning a no-op consumer.
    let _drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });
    run_handle.await.unwrap().unwrap();

    assert!(tmp.path().join("before.txt").exists());
    assert!(tmp.path().join("after.txt").exists());
}

#[tokio::test]
async fn human_reject_aborts_workflow() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path(), 2);
    let yaml = r#"
id: h_reject
steps:
  - id: r
    type: human
    message: "decide"
    options: [approve, reject]
  - id: never
    type: write_file
    path: "never.txt"
    content: "x"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, mut rx) = mpsc::channel(64);

    let run_handle = tokio::spawn({
        let wf = wf.clone();
        async move { engine.run(&wf, BTreeMap::new(), tx).await }
    });

    while let Some(ev) = rx.recv().await {
        if let WorkflowProgress::HumanInputRequired { response_tx, .. } = ev {
            let sender = response_tx.lock().await.take().unwrap();
            sender.send(HumanResponse::reject()).unwrap();
            break;
        }
    }
    let _drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });
    let res = run_handle.await.unwrap();
    assert!(res.is_err(), "reject must abort the workflow");
    assert!(!tmp.path().join("never.txt").exists());
}

#[tokio::test]
async fn human_revise_rewinds_to_target() {
    // The classic "design -> review (revise) -> design re-runs" flow.
    // We use write_file as a stand-in for the design agent. After the
    // first revise, the design step must run again, so the workspace
    // ends up with `design.txt` rewritten. To make the revise
    // *terminate* we send `revise` once and `approve` the second
    // time.
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path(), 2);
    let yaml = r#"
id: h_revise
inputs:
  iteration:
    type: integer
    default: 1
steps:
  - id: design
    type: write_file
    path: "design.txt"
    content: "designed"
  - id: r
    type: human
    message: "review the design"
    options: [approve, revise]
    revise_target: design
  - id: ship
    type: write_file
    path: "ship.txt"
    content: "shipped"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, mut rx) = mpsc::channel(128);

    let run_handle = tokio::spawn({
        let wf = wf.clone();
        async move { engine.run(&wf, BTreeMap::new(), tx).await }
    });

    let mut seen_revise = false;
    let mut human_seen = 0u32;
    while let Some(ev) = rx.recv().await {
        if let WorkflowProgress::HumanInputRequired { response_tx, .. } = ev {
            human_seen += 1;
            let sender = response_tx.lock().await.take().unwrap();
            if seen_revise {
                sender.send(HumanResponse::approve()).unwrap();
                break;
            }
            seen_revise = true;
            sender.send(HumanResponse::revise("design")).unwrap();
        }
    }
    let _drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });
    run_handle.await.unwrap().unwrap();

    assert_eq!(
        human_seen, 2,
        "the human step should fire twice — once before revise, once before approve"
    );
    assert!(tmp.path().join("design.txt").exists());
    assert!(tmp.path().join("ship.txt").exists());
}

// ---------------------------------------------------------------------
// SPEC §8.4 review_loop YAML — end-to-end parse + run with shell mocks
// ---------------------------------------------------------------------

#[tokio::test]
async fn spec_review_loop_yaml_parses_and_runs() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path(), 2);
    let counter = tmp.path().join("verify_counter");
    std::fs::write(&counter, "0").unwrap();
    let counter_path = counter.to_string_lossy().to_string();
    // Approximation of the SPEC §8.4 sample: top-level `verify` and
    // `fix_errors` are real shell steps; the loop body refs them.
    let yaml = format!(
        r#"
id: review_loop
description: "Verify -> fix -> verify cycle"
steps:
  - id: verify
    type: shell
    command: |
      n=$(cat {counter_path})
      n=$((n+1))
      echo $n > {counter_path}
      if [ $n -ge 2 ]; then exit 0; else exit 1; fi
  - id: fix_errors
    type: shell
    command: "echo applying fixes"
  - id: review
    type: loop
    max_iterations: 3
    until: "steps.verify.exit_code == 0"
    steps:
      - ref: verify
      - ref: fix_errors
"#
    );
    let wf = parse_workflow_str(&yaml, "<inline>").unwrap();
    let (tx, mut rx) = mpsc::channel(64);
    engine.run(&wf, BTreeMap::new(), tx).await.unwrap();

    let events = drain(&mut rx);
    let iters = events
        .iter()
        .filter(|e| matches!(e, WorkflowProgress::LoopIteration { .. }))
        .count();
    // Counter starts at 0; first attempt outside the loop -> 1
    // (verify exits 1), then first loop iter -> 2 (verify exits 0),
    // so we expect exactly one LoopIteration event.
    assert_eq!(iters, 1);
    let counter_final: i32 = std::fs::read_to_string(&counter)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(counter_final, 2, "verify ran exactly twice");
}

// ---------------------------------------------------------------------
// Regression tests for PR #64 review fixes
// ---------------------------------------------------------------------

/// Fix #1: every step (leaf or control-flow) must produce exactly one
/// `StepStarted` event. Pre-fix, control-flow handlers double-emitted.
#[tokio::test]
async fn each_step_emits_exactly_one_step_started_event() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path(), 2);
    // Workflow exercises every control-flow variant + a leaf so we
    // catch a regression in any handler.
    let yaml = r#"
id: one_started_per_step
steps:
  - id: g
    type: group
    on_failure: abort
    steps:
      - id: in_group
        type: write_file
        path: "g.txt"
        content: "g"
  - id: br
    type: branch
    conditions:
      - when: "true"
        steps:
          - id: in_branch
            type: write_file
            path: "br.txt"
            content: "br"
  - id: par
    type: parallel
    steps:
      - id: in_par
        type: write_file
        path: "par.txt"
        content: "par"
  - id: lp
    type: loop
    max_iterations: 1
    until: "true"
    steps:
      - id: in_loop
        type: write_file
        path: "lp.txt"
        content: "lp"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, mut rx) = mpsc::channel(64);
    engine.run(&wf, BTreeMap::new(), tx).await.unwrap();
    let events = drain(&mut rx);

    let mut counts: BTreeMap<String, u32> = BTreeMap::new();
    for ev in &events {
        if let WorkflowProgress::StepStarted { step_id, .. } = ev {
            *counts.entry(step_id.clone()).or_default() += 1;
        }
    }
    for (id, n) in &counts {
        assert_eq!(
            *n, 1,
            "step '{id}' emitted {n} StepStarted events; expected exactly 1"
        );
    }
    // Sanity: every control-flow step we declared above is in there.
    for expected in [
        "g",
        "br",
        "par",
        "lp",
        "in_group",
        "in_branch",
        "in_par",
        "in_loop[1]",
    ] {
        assert!(
            counts.contains_key(expected),
            "missing StepStarted for '{expected}' (got: {counts:?})"
        );
    }
}

/// Fix #5: when iteration N skips its inner leaf via `when=false`, the
/// loop must NOT preserve iteration N-1's value under the bare id.
/// Otherwise `until:` (which reads the bare id) would observe stale
/// data and could exit early on a stale truthy result.
///
/// Strategy:
/// - The loop's `until` is `steps.bump.exit_code == 0`.
/// - `guard` (always-running shell) toggles each iteration based on a
///   counter file: iter 1 exits 0; iter 2 exits 1 (so bump skips).
/// - `bump`'s `when: steps.guard.exit_code == 0` controls whether the
///   inner step runs.
///
/// Pre-fix behavior: iter 1 sets `bump` (bare-id mirror) to a
/// successful exit code (0). Iter 2 skips bump but the bare-id mirror
/// stays as iter 1's result. After iter 2 the loop's
/// `until="steps.bump.exit_code == 0"` is **true** because of the
/// stale mirror, so the loop exits early at iter 2 with `until` hit.
///
/// Post-fix: at the end of iter 2 we *clear* the bare-id mirror
/// because no `bump[2]` was written. `until="steps.bump.exit_code == 0"`
/// then errors-out (`steps.bump` missing), which surfaces as a
/// `WorkflowError::StepFailed` with `until-expression error:` in the
/// message. This is the visible signal we assert.
///
/// (A nicer post-fix UX would be to treat a missing id as falsy; we
/// keep the existing strict semantics here to avoid silently masking
/// typos. The test pins the actual current behaviour.)
#[tokio::test]
async fn loop_skipped_inner_clears_bare_id_mirror() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = build_engine(tmp.path(), 2);
    let counter = tmp.path().join("loop_skip_counter");
    std::fs::write(&counter, "0").unwrap();
    let counter_path = counter.to_string_lossy().to_string();
    // The `until` requires BOTH bump.exit_code == 0 AND
    // guard.exit_code != 0. Iter 1: guard exit 0 -> until false.
    // Iter 2: guard exit 1, bump skipped. Pre-fix: stale bare-id
    // mirror keeps bump.exit_code == 0, until evaluates true and the
    // loop exits cleanly. Post-fix: mirror is cleared, until errors,
    // and the engine returns Err.
    let yaml = format!(
        r#"
id: skip_clears_mirror
steps:
  - id: lp
    type: loop
    max_iterations: 3
    until: "steps.bump.exit_code == 0 && steps.guard.exit_code != 0"
    steps:
      - id: guard
        type: shell
        command: |
          n=$(cat {counter_path})
          n=$((n+1))
          echo $n > {counter_path}
          if [ $n -ge 2 ]; then exit 1; else exit 0; fi
      - id: bump
        type: shell
        when: "steps.guard.exit_code == 0"
        command: "true"
"#
    );
    let wf = parse_workflow_str(&yaml, "<inline>").unwrap();
    let (tx, mut rx) = mpsc::channel(64);
    let res = engine.run(&wf, BTreeMap::new(), tx).await;
    let _ = drain(&mut rx);

    // Pre-fix: iter 1 runs guard (exit 0) and bump (exit 0). bare-id
    // mirror has bump.exit_code=0. until evaluates: 0==0 (true) &&
    // 0!=0 (false) -> false. Iter 2: guard exits 1; bump skipped.
    // Pre-fix bare-id mirror still says exit_code=0. until: 0==0
    // (true) && 1!=0 (true) -> true. Loop exits at iter 2 with `Ok`.
    //
    // Post-fix: iter 2 clears the bump bare-id mirror. until errors
    // because `steps.bump` is missing. The loop returns Err.
    //
    // We assert post-fix behaviour: the run errored with an
    // until-expression error mentioning bump.
    let err = res.expect_err("post-fix #5: until must error on missing bump after skipped iter 2");
    let msg = err.to_string();
    assert!(
        msg.contains("until-expression") && msg.contains("bump"),
        "expected an until-expression error referencing 'bump', got: {msg}"
    );
}
