#![expect(
    clippy::expect_used,
    clippy::too_many_lines,
    reason = "integration test: linear narrative > splitting into helpers"
)]
//! End-to-end integration test for the autonomous skill creation
//! pipeline (issue #54, round-2 review).
//!
//! Round-1 added a `SkillsRuntime`-only test (in
//! `crates/tmg-skills/tests/critic_pipeline.rs`); the reviewer flagged
//! that the wiring between [`HarnessStreamSink`] (which observes
//! per-tool outcomes), the [`TurnOutcomeRecorder`] (which buffers
//! them across a turn), the [`SkillsRuntime`] (which evaluates
//! signals and applies verdicts), and a real [`RunRunner`] turn-
//! observer was not exercised end-to-end. This test fills that gap.
//!
//! The LLM round-trip itself (the `skill_critic` subagent) is
//! intentionally stubbed: we feed a hand-crafted JSON verdict directly
//! to `parse_verdict` + `apply_verdict`. This mirrors the production
//! data flow that follows the live LLM call without making the test
//! depend on `llama-server`.
//!
//! What we exercise:
//!
//! 1. Construct a real [`RunRunner`] + active session.
//! 2. Wrap a downstream sink with [`HarnessStreamSink`] and install a
//!    [`TurnOutcomeRecorder`] on it.
//! 3. Drive the sink through five successful tool calls
//!    (`on_tool_call` + `on_tool_result(is_error=false)`) — this is
//!    the fingerprint of a `SuccessfulComplexTask` trigger.
//! 4. Drain the recorder into a `tmg_skills::TurnSummary` and call
//!    `RunRunner::after_turn` to update the harness session state
//!    (mirrors the `tmg-cli` `install_turn_observer` path).
//! 5. Call `SkillsRuntime::record_turn` to evaluate triggers and
//!    confirm a signal fires.
//! 6. Apply a hand-crafted critic verdict via
//!    `SkillsRuntime::apply_verdict`, which invokes
//!    `SkillManageTool::execute_inner` and writes a `provenance: agent`
//!    SKILL.md.
//! 7. Confirm `discover_skills_with_config` re-finds the new skill.
//! 8. Confirm `pending_banner_names` surfaces it.
//! 9. Drive the use-skill-metrics path: record `use_skill` on the
//!    recorder, then call `record_use_skill_outcome` and assert the
//!    metrics file is updated.

use std::sync::Arc;

use tmg_core::StreamSink;
use tmg_harness::{HarnessStreamSink, RunRunner, RunStore, TurnSummary};
use tmg_sandbox::{SandboxConfig, SandboxContext, SandboxMode};
use tmg_skills::{
    SkillCriticConfig, SkillCriticVerdict, SkillsConfig, SkillsRuntime, TriggerKind,
    TurnOutcomeRecorder, UseSkillOutcome, discover_skills_with_config, load_acknowledged,
    parse_verdict, pending_banner_names,
};
use tokio::sync::Mutex;

/// Bare-bones [`StreamSink`] used as the inner sink in the test. It
/// swallows every callback so we can focus on the [`HarnessStreamSink`]
///   + [`TurnOutcomeRecorder`] integration without worrying about
///     channel back-pressure.
struct NullSink;

impl StreamSink for NullSink {
    fn on_token(&mut self, _token: &str) -> Result<(), tmg_core::CoreError> {
        Ok(())
    }
}

fn make_ctx(workspace: &std::path::Path) -> SandboxContext {
    SandboxContext::new(SandboxConfig::new(workspace).with_mode(SandboxMode::Full))
}

#[tokio::test]
async fn end_to_end_runner_skill_pipeline() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let project_root = tmp.path().to_path_buf();
    let runs_dir = project_root.join(".tsumugi").join("runs");
    let workspace = project_root.clone();
    std::fs::create_dir_all(&workspace).expect("workspace");

    // ---- Step 1: construct a real RunRunner + session.
    let store = Arc::new(RunStore::new(runs_dir));
    let run = store
        .create_ad_hoc(workspace.clone(), None)
        .expect("create_ad_hoc");
    let mut runner = RunRunner::new(run, store);
    let session_handle = runner.begin_session().expect("begin_session");
    let runner = Arc::new(Mutex::new(runner));

    // ---- Step 2: build the runtime + recorder, wrap the sink.
    let critic_cfg = SkillCriticConfig::default();
    let runtime = Arc::new(Mutex::new(SkillsRuntime::new(
        project_root.clone(),
        critic_cfg,
    )));
    let recorder = Arc::new(TurnOutcomeRecorder::new());
    let mut sink = HarnessStreamSink::new(NullSink, Arc::clone(&runner))
        .with_skill_outcome_recorder(Arc::clone(&recorder));

    // ---- Step 3: drive five successful tool calls through the sink.
    for (i, tool) in [
        "file_read",
        "file_read",
        "shell_exec",
        "shell_exec",
        "file_write",
    ]
    .iter()
    .enumerate()
    {
        let call_id = format!("call_{i}");
        sink.on_tool_call(&call_id, tool, "{}").expect("call");
        sink.on_tool_result(&call_id, tool, "ok", false)
            .expect("result");
    }

    // The recorder should now hold five successful outcomes.
    // ---- Step 4: drain the recorder and feed the harness session
    //              state, mimicking the install_turn_observer flow.
    let skills_summary = recorder.take_for_turn(0).expect("non-empty turn");
    assert_eq!(skills_summary.tool_calls.len(), 5);
    assert!(skills_summary.tool_calls.iter().all(|c| c.success));
    assert!(!skills_summary.turn_errored);

    // Harness side: feed a TurnSummary into the runner. We use the
    // simpler harness summary shape (token / tool count / user
    // message) since the SessionState observer reads only that.
    let harness_summary = TurnSummary {
        tokens_used: 1000,
        tool_calls: 5,
        user_message: "deploy a rust crate".to_owned(),
        ..TurnSummary::default()
    };
    {
        let mut guard = runner.lock().await;
        guard.after_turn(&harness_summary, 8192);
    }

    // ---- Step 5: record_turn fires SuccessfulComplexTask.
    let signals = {
        let mut guard = runtime.lock().await;
        guard.record_turn(&skills_summary)
    };
    assert!(!signals.is_empty(), "expected a signal: {signals:?}");
    let chosen = SkillsRuntime::highest_priority(&signals)
        .expect("highest_priority")
        .clone();
    assert_eq!(chosen.kind, TriggerKind::SuccessfulComplexTask);

    // ---- Step 6: apply a stubbed verdict via the runtime.
    // The JSON shape mirrors what skill_critic would return live.
    let stub_json = r#"{
        "create": true,
        "name": "auto-deploy-runner",
        "description": "Deploy a Rust crate via the runner pipeline",
        "draft": "Steps:\n1. cargo publish --dry-run\n2. cargo publish\n",
        "reason": "Multi-step procedure observed across five successful tool calls."
    }"#;
    let verdict: SkillCriticVerdict = parse_verdict(stub_json).expect("parse");
    let sandbox = make_ctx(&workspace);
    let new_dir = {
        let mut guard = runtime.lock().await;
        guard
            .apply_verdict(&verdict, &sandbox)
            .await
            .expect("apply_verdict")
            .expect("create=true must produce a path")
    };
    assert!(new_dir.ends_with("auto-deploy-runner"));

    // The SKILL.md must carry `provenance: agent` so the banner
    // detection loop can pick it up.
    let body = std::fs::read_to_string(new_dir.join("SKILL.md")).expect("read SKILL.md");
    assert!(
        body.contains("provenance: agent"),
        "expected agent provenance in: {body}",
    );

    // ---- Step 7: discover_skills_with_config re-finds it.
    let skills_cfg = SkillsConfig {
        compat_claude: false,
        compat_agent_skills: false,
        ..Default::default()
    };
    let metas = discover_skills_with_config(&project_root, &skills_cfg)
        .await
        .expect("discover");
    assert!(
        metas
            .iter()
            .any(|m| m.name.as_str() == "auto-deploy-runner"),
        "expected auto-deploy-runner in discovered skills",
    );

    // ---- Step 8: pending_banner_names lists it (no acks yet).
    let acked = {
        let guard = runtime.lock().await;
        load_acknowledged(guard.skills_root())
            .await
            .expect("load_acknowledged")
    };
    let pending = pending_banner_names(&metas, &acked);
    assert!(
        pending.contains(&"auto-deploy-runner".to_owned()),
        "expected pending banner: {pending:?}",
    );

    // ---- Step 9: use_skill metrics roundtrip through the runtime.
    // Record a use_skill invocation via the recorder, drain it, then
    // record an outcome.
    recorder.record_use_skill("auto-deploy-runner");
    let invoked = recorder.take_use_skill_invocations();
    assert_eq!(invoked, vec!["auto-deploy-runner".to_owned()]);
    let m = {
        let guard = runtime.lock().await;
        guard
            .record_use_skill_outcome("auto-deploy-runner", UseSkillOutcome::Success)
            .await
            .expect("record_use_skill_outcome")
    };
    assert_eq!(m.success_count, 1);
    assert_eq!(m.failure_count, 0);

    // After the run, end the session so the harness state stays
    // consistent.
    {
        let mut guard = runner.lock().await;
        guard
            .end_session(&session_handle, tmg_harness::SessionEndTrigger::Completed)
            .ok();
    }
}

/// Verify the `/skill capture` manual flow: triggering `manual_capture`
/// on the runtime synthesises a `Manual` signal on the next
/// `record_turn` even when the per-session budget would otherwise
/// block (i.e. after `record_auto_generated`).
#[tokio::test]
async fn manual_capture_bypasses_budget() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let project_root = tmp.path().to_path_buf();
    let mut runtime = SkillsRuntime::new(project_root.clone(), SkillCriticConfig::default());

    // First, exhaust the budget by applying one auto verdict.
    let sandbox = make_ctx(&project_root);
    let v = SkillCriticVerdict {
        create: true,
        name: Some("first".into()),
        description: Some("d".into()),
        draft: Some("body".into()),
        reason: "r".into(),
    };
    runtime.apply_verdict(&v, &sandbox).await.expect("apply");
    assert!(!runtime.auto_creation_allowed());

    // Now request a manual capture and feed any turn — the manual
    // signal must surface despite the consumed budget.
    runtime.trigger_manual_capture();
    let summary = tmg_skills::TurnSummary {
        turn_index: 5,
        tool_calls: vec![tmg_skills::ToolCallOutcome {
            tool: "file_read".into(),
            success: true,
        }],
        feedback_memory_written: false,
        turn_errored: false,
    };
    let signals = runtime.record_turn(&summary);
    assert!(
        signals.iter().any(|s| s.kind == TriggerKind::Manual),
        "manual signal must surface despite budget exhaustion: {signals:?}",
    );
    // The manual flag must be cleared after one drain.
    assert!(!runtime.manual_capture_pending());
}

/// Verify the harness sink propagates `use_skill` invocations to the
/// recorder. Round-2 review item #d.
#[tokio::test]
async fn harness_sink_records_use_skill_invocation() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let runs_dir = tmp.path().join(".tsumugi").join("runs");
    let workspace = tmp.path().join("workspace");
    std::fs::create_dir_all(&workspace).expect("ws");

    let store = Arc::new(RunStore::new(runs_dir));
    let run = store.create_ad_hoc(workspace, None).expect("ad_hoc");
    let mut runner = RunRunner::new(run, store);
    runner.begin_session().expect("begin_session");
    let runner = Arc::new(Mutex::new(runner));

    let recorder = Arc::new(TurnOutcomeRecorder::new());
    let mut sink = HarnessStreamSink::new(NullSink, Arc::clone(&runner))
        .with_skill_outcome_recorder(Arc::clone(&recorder));

    sink.on_tool_call(
        "call_use_1",
        "use_skill",
        r#"{"skill_name": "deploy-rust"}"#,
    )
    .expect("call");
    sink.on_tool_result("call_use_1", "use_skill", "ok", false)
        .expect("result");

    let invoked = recorder.take_use_skill_invocations();
    assert_eq!(invoked, vec!["deploy-rust".to_owned()]);

    // The on_tool_result also recorded the call outcome, so
    // take_for_turn should produce a non-empty summary.
    let summary = recorder.take_for_turn(0).expect("non-empty");
    assert_eq!(summary.tool_calls.len(), 1);
    assert!(summary.tool_calls[0].success);
}
