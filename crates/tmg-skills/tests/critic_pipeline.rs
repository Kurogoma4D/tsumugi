#![expect(
    clippy::expect_used,
    reason = "integration test: clarity of failure message > production lints"
)]
//! End-to-end integration test for the autonomous skill creation
//! pipeline (issue #54, code-review item #1.i).
//!
//! The test exercises the full chain offline (no LLM round-trip) by
//! feeding a hand-crafted critic verdict — the same JSON shape the
//! `skill_critic` subagent would emit — and asserting that:
//!
//! 1. A trigger fires when [`SignalCollector::record_turn`] sees a
//!    qualifying turn (5+ consecutive successful tool calls, no error).
//! 2. The runtime gates on the per-session budget so the next turn does
//!    NOT also fire after `record_auto_generated`.
//! 3. [`SkillsRuntime::apply_verdict`] invokes [`SkillManageTool`]
//!    behind the scenes and produces a `provenance: agent` skill on
//!    disk.
//! 4. [`discover_skills_with_config`] re-finds the freshly-created
//!    skill.
//! 5. [`pending_banner_names`] surfaces it as needing a banner.
//! 6. [`format_banner`] produces a non-empty banner string.
//! 7. After [`SkillsRuntime::acknowledge_banner`], the banner list goes
//!    empty (round-trip).
//!
//! The test is offline — it never touches a live `llama-server` — so
//! it can run on any developer machine in `cargo test --workspace`.

use std::path::Path;

use tmg_sandbox::{SandboxConfig, SandboxContext, SandboxMode};
use tmg_skills::{
    SignalCollector, SkillCriticConfig, SkillCriticVerdict, SkillManageTool, SkillsConfig,
    SkillsRuntime, ToolCallOutcome, TriggerKind, TurnSummary, discover_skills_with_config,
    format_banner, load_acknowledged, parse_verdict, pending_banner_names,
};
use tmg_tools::ToolRegistry;

fn make_sandbox(workspace: &Path) -> SandboxContext {
    SandboxContext::new(SandboxConfig::new(workspace).with_mode(SandboxMode::Full))
}

#[expect(
    clippy::too_many_lines,
    reason = "linear end-to-end script: splitting into helpers obscures the step-by-step verification flow"
)]
#[tokio::test]
async fn end_to_end_autonomous_skill_creation_chain() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let project_root = tmp.path().to_path_buf();
    let ctx = make_sandbox(&project_root);

    // Disable compat sources so discovery only sees `.tsumugi/skills/`.
    let skills_cfg = SkillsConfig {
        compat_claude: false,
        compat_agent_skills: false,
        ..Default::default()
    };
    let critic_cfg = SkillCriticConfig::default();
    let mut runtime = SkillsRuntime::new(project_root.clone(), critic_cfg);

    // ---- Step 1: drive the SignalCollector with a qualifying turn.
    // Five consecutive successes within one turn, no error. The
    // SuccessfulComplexTask trigger should fire.
    let qualifying_turn = TurnSummary {
        turn_index: 0,
        tool_calls: vec![
            ToolCallOutcome {
                tool: "file_read".into(),
                success: true,
            },
            ToolCallOutcome {
                tool: "file_read".into(),
                success: true,
            },
            ToolCallOutcome {
                tool: "shell_exec".into(),
                success: true,
            },
            ToolCallOutcome {
                tool: "shell_exec".into(),
                success: true,
            },
            ToolCallOutcome {
                tool: "file_write".into(),
                success: true,
            },
        ],
        feedback_memory_written: false,
        turn_errored: false,
    };
    let signals = runtime.record_turn(&qualifying_turn);
    assert!(
        !signals.is_empty(),
        "expected at least one signal from a qualifying turn",
    );
    let chosen = SkillsRuntime::highest_priority(&signals).expect("priority");
    assert_eq!(chosen.kind, TriggerKind::SuccessfulComplexTask);
    assert!(runtime.auto_creation_allowed());

    // ---- Step 2: parse a stub critic verdict (the same shape the
    // skill_critic subagent would return) and apply it.
    let stub_verdict_json = r#"{
        "create": true,
        "name": "auto-deploy",
        "description": "Deploy a Rust crate",
        "draft": "Steps:\n1. cargo publish\n",
        "reason": "Multi-step deploy procedure with retries."
    }"#;
    let verdict: SkillCriticVerdict = parse_verdict(stub_verdict_json).expect("parse");
    let new_dir = runtime
        .apply_verdict(&verdict, &ctx)
        .await
        .expect("apply_verdict")
        .expect("create=true should produce a path");
    assert!(new_dir.ends_with("auto-deploy"));

    // Budget consumed.
    assert!(
        !runtime.auto_creation_allowed(),
        "per-session budget should be consumed after one autonomous create",
    );

    // ---- Step 3: skill_manage wrote SKILL.md with provenance=agent
    let skill_md = new_dir.join("SKILL.md");
    let body = std::fs::read_to_string(&skill_md).expect("read SKILL.md");
    assert!(
        body.contains("provenance: agent"),
        "skill body should be marked agent-authored: {body}",
    );

    // ---- Step 4: discover_skills_with_config re-finds the skill.
    let metas = discover_skills_with_config(&project_root, &skills_cfg)
        .await
        .expect("discover_skills");
    let names: Vec<&str> = metas.iter().map(|m| m.name.as_str()).collect();
    assert!(
        names.contains(&"auto-deploy"),
        "expected auto-deploy in discovered skills: {names:?}",
    );

    // ---- Step 5: pending_banner_names surfaces the new skill.
    let acked = load_acknowledged(runtime.skills_root())
        .await
        .expect("load_acknowledged");
    let pending = pending_banner_names(&metas, &acked);
    assert!(
        pending.contains(&"auto-deploy".to_owned()),
        "expected pending banner for auto-deploy: {pending:?}",
    );

    // ---- Step 6: format_banner produces a usable banner string.
    let banner = format_banner(&pending).expect("banner formatted");
    assert!(
        banner.starts_with('\u{26a1}'),
        "banner needs prefix: {banner}"
    );
    assert!(banner.contains("auto-deploy"));

    // ---- Step 7: ack roundtrip clears it.
    runtime
        .acknowledge_banner(&pending)
        .await
        .expect("acknowledge_banner");
    let post_ack = runtime
        .compute_pending_banner(&skills_cfg)
        .await
        .expect("compute_pending_banner");
    assert!(
        post_ack.is_empty(),
        "expected empty banner list after ack: {post_ack:?}",
    );
}

#[tokio::test]
async fn budget_blocks_second_auto_creation_in_same_session() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ctx = make_sandbox(tmp.path());
    let mut runtime = SkillsRuntime::new(tmp.path().to_path_buf(), SkillCriticConfig::default());

    // First create succeeds.
    let v1 = SkillCriticVerdict {
        create: true,
        name: Some("first".into()),
        description: Some("d".into()),
        draft: Some("body".into()),
        reason: "r".into(),
    };
    runtime.apply_verdict(&v1, &ctx).await.expect("first");
    assert!(!runtime.auto_creation_allowed());

    // The runtime no longer permits auto creation. The harness wiring
    // should consult `auto_creation_allowed` before spawning the
    // critic; we simulate that here by checking the gate explicitly.
    assert!(
        !runtime.auto_creation_allowed(),
        "budget exhausted, harness must skip the critic for this session",
    );
}

#[tokio::test]
async fn rejection_removes_skill_directory() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ctx = make_sandbox(tmp.path());
    let mut runtime = SkillsRuntime::new(tmp.path().to_path_buf(), SkillCriticConfig::default());

    let v = SkillCriticVerdict {
        create: true,
        name: Some("ephemeral".into()),
        description: Some("d".into()),
        draft: Some("body".into()),
        reason: "r".into(),
    };
    let dir = runtime
        .apply_verdict(&v, &ctx)
        .await
        .expect("apply")
        .expect("path");
    assert!(dir.exists());

    runtime
        .apply_rejection("ephemeral", &ctx)
        .await
        .expect("reject");
    assert!(
        !dir.exists(),
        "skill directory should be removed by rejection"
    );
}

#[tokio::test]
async fn skill_manage_tool_registers_and_dispatches_under_skill_manage_name() {
    // Issue #54 review item #1.b / #3: SkillManageTool must register
    // under the canonical name `skill_manage` and dispatch through the
    // ToolRegistry execute path. This catches regressions where the
    // tool is dropped from the project-scope registry assembly.
    let mut registry = ToolRegistry::new();
    registry.register(SkillManageTool);

    let tool = registry
        .get("skill_manage")
        .expect("skill_manage must be registered");
    assert_eq!(tool.name(), "skill_manage");

    // Execute a `create` action through the registry to confirm the
    // dispatch path is wired (not just `get`). The tool's
    // `execute_inner` direct-call path is independently exercised by
    // its unit tests; this test confirms the registry hookup itself.
    let tmp = tempfile::tempdir().expect("tempdir");
    let ctx = make_sandbox(tmp.path());
    let result = registry
        .execute(
            "skill_manage",
            serde_json::json!({
                "action": "create",
                "name": "registry-test",
                "description": "via registry",
            }),
            &ctx,
        )
        .await
        .expect("dispatch");
    assert!(!result.is_error, "dispatch failed: {}", result.output);
    assert!(
        tmp.path()
            .join(".tsumugi")
            .join("skills")
            .join("registry-test")
            .join("SKILL.md")
            .exists(),
        "skill_manage create through registry must produce a SKILL.md",
    );
}

#[tokio::test]
async fn signal_collector_emits_both_recovery_and_user_correction() {
    // Bonus regression coverage for the MEDIUM #6 review fix: when a
    // turn both fires ErrorRecovery (recovery completed) AND
    // UserCorrection (feedback memory written), both signals must
    // surface — neither is dropped.
    let mut c = SignalCollector::new(1);
    // Turn 0: same tool fails twice.
    let _ = c.record_turn(&TurnSummary {
        turn_index: 0,
        tool_calls: vec![
            ToolCallOutcome {
                tool: "shell_exec".into(),
                success: false,
            },
            ToolCallOutcome {
                tool: "shell_exec".into(),
                success: false,
            },
        ],
        feedback_memory_written: false,
        turn_errored: true,
    });
    // Turn 1: a different tool succeeds AND the user writes feedback.
    let signals = c.record_turn(&TurnSummary {
        turn_index: 1,
        tool_calls: vec![ToolCallOutcome {
            tool: "file_read".into(),
            success: true,
        }],
        feedback_memory_written: true,
        turn_errored: false,
    });
    assert_eq!(signals.len(), 2, "expected both signals: {signals:?}");
    assert!(signals.iter().any(|s| s.kind == TriggerKind::ErrorRecovery));
    assert!(
        signals
            .iter()
            .any(|s| s.kind == TriggerKind::UserCorrection)
    );
}
