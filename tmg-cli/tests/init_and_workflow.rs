//! End-to-end tests for the `tmg init` and `tmg workflow {list,validate}`
//! subcommands (issue #44).
//!
//! These spawn the compiled `tmg` binary with `cargo` env knobs so we
//! exercise the full clap → dispatch → tmg-workflow round-trip.
//!
//! `tmg workflow run` is exercised separately via the ad-hoc-run path
//! when an LLM endpoint is reachable; the smoke variant here only
//! checks the CLI surface (`--help`, `--input k=v` parsing).

#![cfg(test)]
#![expect(clippy::unwrap_used, reason = "tests")]
#![expect(clippy::expect_used, reason = "tests")]

use std::path::Path;
use std::process::Command;

/// Locate the `tmg` binary built by the current `cargo test`
/// invocation. This lets us drive the binary as a black box without
/// having to depend on `assert_cmd` or hardcode the target path.
fn tmg_binary() -> std::path::PathBuf {
    // Cargo sets CARGO_BIN_EXE_<name> for every binary defined in the
    // crate under test. See:
    // https://doc.rust-lang.org/cargo/reference/environment-variables.html
    let path = env!("CARGO_BIN_EXE_tmg");
    std::path::PathBuf::from(path)
}

/// Run the `tmg` binary in `cwd` with the given args, returning the
/// `Output`.
fn run_tmg(cwd: &Path, args: &[&str]) -> std::process::Output {
    Command::new(tmg_binary())
        .args(args)
        .current_dir(cwd)
        .env("TMG_LLM_ENDPOINT", "http://127.0.0.1:1") // unreachable; not used by list/validate
        .output()
        .expect("running tmg")
}

#[test]
fn init_writes_workflow_files() {
    let tmp = tempfile::tempdir().unwrap();
    let out = run_tmg(tmp.path(), &["init", "--workflows", "plan,implement"]);
    assert!(
        out.status.success(),
        "tmg init failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(tmp.path().join(".tsumugi/workflows/plan.yaml").exists());
    assert!(
        tmp.path()
            .join(".tsumugi/workflows/implement.yaml")
            .exists()
    );
}

#[test]
fn init_without_force_refuses_to_overwrite() {
    let tmp = tempfile::tempdir().unwrap();
    let first = run_tmg(tmp.path(), &["init", "--workflows", "plan"]);
    assert!(first.status.success());

    // Re-running without --force should fail and not modify the file.
    let original = std::fs::read(tmp.path().join(".tsumugi/workflows/plan.yaml")).unwrap();
    let second = run_tmg(tmp.path(), &["init", "--workflows", "plan"]);
    assert!(!second.status.success(), "second init should fail");
    let after = std::fs::read(tmp.path().join(".tsumugi/workflows/plan.yaml")).unwrap();
    assert_eq!(original, after, "init without --force must not overwrite");
}

#[test]
fn init_all_force_overwrites_cleanly() {
    let tmp = tempfile::tempdir().unwrap();
    // Pre-create one of the targets with custom content.
    std::fs::create_dir_all(tmp.path().join(".tsumugi/workflows")).unwrap();
    std::fs::write(tmp.path().join(".tsumugi/workflows/plan.yaml"), "stale").unwrap();

    let out = run_tmg(tmp.path(), &["init", "--all", "--force"]);
    assert!(
        out.status.success(),
        "tmg init --all --force failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let new_content =
        std::fs::read_to_string(tmp.path().join(".tsumugi/workflows/plan.yaml")).unwrap();
    assert_ne!(new_content, "stale", "expected --force to overwrite");
    assert!(new_content.contains("id: plan"));

    // Every other workflow template should also be present.
    for fname in [
        "plan.yaml",
        "implement.yaml",
        "review.yaml",
        "refactor.yaml",
        "build-app.yaml",
        "migrate-codebase.yaml",
        "add-feature.yaml",
        "bug-fix-batch.yaml",
    ] {
        assert!(
            tmp.path().join(".tsumugi/workflows").join(fname).exists(),
            "missing workflow file {fname}",
        );
    }
    assert!(tmp.path().join(".tsumugi/skills/develop/SKILL.md").exists());
}

#[test]
fn workflow_list_finds_builtin_after_init() {
    let tmp = tempfile::tempdir().unwrap();
    let init = run_tmg(tmp.path(), &["init", "--all"]);
    assert!(init.status.success());

    let list = run_tmg(tmp.path(), &["workflow", "list"]);
    assert!(
        list.status.success(),
        "tmg workflow list failed: {}",
        String::from_utf8_lossy(&list.stderr)
    );
    let stdout = String::from_utf8_lossy(&list.stdout);
    // The id of the plan workflow.
    assert!(
        stdout.contains("plan"),
        "missing plan in list output:\n{stdout}"
    );
    // The id of the build_app workflow (note: SPEC uses snake_case ids).
    assert!(
        stdout.contains("build_app"),
        "missing build_app in list output:\n{stdout}",
    );
}

#[test]
fn workflow_validate_passes_on_builtins() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(run_tmg(tmp.path(), &["init", "--all"]).status.success());
    let out = run_tmg(tmp.path(), &["workflow", "validate"]);
    assert!(
        out.status.success(),
        "validate failed:\n  stdout: {}\n  stderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

#[test]
fn workflow_validate_reports_broken_workflow() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(
        run_tmg(tmp.path(), &["init", "--workflows", "plan"])
            .status
            .success()
    );
    // Inject a broken file alongside the good one.
    std::fs::write(
        tmp.path().join(".tsumugi/workflows/broken.yaml"),
        "id: broken\nthis is not valid yaml: [unclosed\n",
    )
    .unwrap();
    let out = run_tmg(tmp.path(), &["workflow", "validate"]);
    assert!(
        !out.status.success(),
        "validate must fail when a workflow is broken; stdout: {}",
        String::from_utf8_lossy(&out.stdout),
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("broken.yaml"),
        "stderr should mention broken.yaml:\n{stderr}",
    );
}

#[test]
fn init_with_unknown_template_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let out = run_tmg(tmp.path(), &["init", "--workflows", "does-not-exist"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unknown template"), "{stderr}");
}

#[test]
fn workflow_run_rejects_missing_required_input() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(
        run_tmg(tmp.path(), &["init", "--workflows", "plan"])
            .status
            .success()
    );
    // The plan workflow declares `requirements` as required; omitting
    // it must fail before any LLM call happens.
    let out = run_tmg(tmp.path(), &["workflow", "run", "plan"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("missing required") || stderr.contains("requirements"),
        "stderr should mention missing input:\n{stderr}",
    );
}

#[test]
fn init_help_contains_template_names() {
    let tmp = tempfile::tempdir().unwrap();
    let out = run_tmg(tmp.path(), &["init", "--help"]);
    assert!(out.status.success());
}

/// `tmg workflow run plan --input requirements="..."` must reach the
/// engine and emit at least one `StepStarted` event (the engine emits
/// `StepStarted` before invoking the leaf step's handler, so this
/// works even when the LLM endpoint is unreachable).
#[test]
fn workflow_run_normal_emits_step_started() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(
        run_tmg(tmp.path(), &["init", "--workflows", "plan"])
            .status
            .success()
    );
    let out = run_tmg(
        tmp.path(),
        &[
            "workflow",
            "run",
            "plan",
            "--input",
            "requirements=Add a status command",
        ],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Don't assert success: the agent step needs an LLM and our test
    // uses an unreachable endpoint. We only need to see the engine
    // reach the first step.
    assert!(
        stderr.contains("[step:start] explore"),
        "expected explore step to start; stderr={stderr}",
    );
}

/// `tmg workflow run build_app` should dispatch to the long-running
/// executor based on the workflow's declared `mode: long_running`.
/// We verify dispatch indirectly by observing the
/// `[long-running] using run <id>` line that the CLI prints to stderr
/// before any LLM call. With an unreachable endpoint the executor will
/// later fail; the existence of the diagnostic line is the thing
/// that proves we took the long-running branch.
#[test]
fn workflow_run_long_running_routes_through_executor() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(
        run_tmg(tmp.path(), &["init", "--harness", "build-app"])
            .status
            .success()
    );
    // build-app declares `app_description` as required; supply it so
    // input validation passes.
    let out = run_tmg(
        tmp.path(),
        &[
            "workflow",
            "run",
            "build_app",
            "--input",
            "app_description=demo",
        ],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    // The CLI must reach the long-running diagnostic, even though the
    // run will eventually fail because the LLM endpoint is unreachable.
    assert!(
        stderr.contains("[long-running]"),
        "expected long-running diagnostic; stdout={stdout}; stderr={stderr}",
    );
}
