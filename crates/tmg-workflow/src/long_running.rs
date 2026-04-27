//! Long-running workflow executor (SPEC §8.12 + §9.9).
//!
//! Drives the `mode: long_running` workflow lifecycle on top of the
//! existing [`WorkflowEngine`]:
//!
//! 1. **Init phase** — runs once per harnessed run on first attach.
//!    Typically calls the `initializer` builtin subagent through an
//!    `agent` step to produce `features.json` / `init.sh` /
//!    `progress.md`. After the steps complete the executor invokes
//!    [`tmg_harness::RunRunner::escalate_to_harnessed`] so the run
//!    flips to harnessed scope and the on-disk artifacts are recorded.
//!
//! 2. **Iterate phase** — the per-session loop. Each iteration begins
//!    a new harnessed session, resolves the `bootstrap` items into a
//!    context block, runs `iterate.steps`, then evaluates the `until`
//!    expression. Sessions stop the loop on:
//!    - `until` truthy → [`RunStatus::Completed`]
//!    - `session_count >= max_sessions` →
//!      [`RunStatus::Exhausted { reason: "max_sessions" }`]
//!    - `session_timeout` exceeded → fires
//!      [`tmg_harness::SessionEndTrigger::Timeout`] via the existing
//!      watchdog channel and continues with the next iteration.
//!
//! ## Cancellation safety
//!
//! Every await point is cancel-safe at the session boundary. The
//! per-session loop honours the engine's progress channel for visibility
//! but does not block on a UI consumer; a slow consumer back-pressures
//! progress events without stalling the loop.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::sync::{Mutex, mpsc};

use tmg_harness::{RunRunner, SessionEndTrigger};

use crate::def::{BootstrapItem, StepDef, StepResult, WorkflowDef, WorkflowMode};
use crate::engine::WorkflowEngine;
use crate::error::{Result, WorkflowError};
use crate::expr::{self, ArtifactResolver, ExprContext};
use crate::progress::WorkflowProgress;

/// Terminal state of [`LongRunningExecutor::run`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RunStatus {
    /// `until:` evaluated truthy and the run completed normally.
    Completed,
    /// Hit a hard stop (e.g. `max_sessions`) without completing.
    Exhausted {
        /// Short tag identifying the stop reason (e.g.
        /// `"max_sessions"`).
        reason: String,
    },
    /// A step in the iterate phase failed; the carried message is the
    /// formatted [`WorkflowError`].
    Failed {
        /// Error message.
        error: String,
    },
}

/// Stateful executor that drives a `mode: long_running` workflow.
///
/// Constructed once per CLI / TUI invocation and given a shared
/// [`WorkflowEngine`] plus the run's [`RunRunner`].
pub struct LongRunningExecutor {
    engine: Arc<WorkflowEngine>,
    run_runner: Arc<Mutex<RunRunner>>,
}

impl LongRunningExecutor {
    /// Build a new executor.
    #[must_use]
    pub fn new(engine: Arc<WorkflowEngine>, run_runner: Arc<Mutex<RunRunner>>) -> Self {
        Self { engine, run_runner }
    }

    /// Drive the workflow to its terminal [`RunStatus`].
    ///
    /// Returns:
    ///
    /// - [`RunStatus::Completed`] when the iterate phase's `until`
    ///   expression evaluates truthy at the end of a session.
    /// - [`RunStatus::Exhausted`] when `iterate.max_sessions` is
    ///   reached without `until` ever firing.
    /// - [`RunStatus::Failed`] when an init or iterate step propagates
    ///   a [`WorkflowError`] up.
    ///
    /// # Errors
    ///
    /// Returns a [`WorkflowError`] only for setup-level problems (the
    /// workflow is not `mode: long_running`, the `iterate` phase is
    /// missing, or the harness escalation I/O fails). Step-level
    /// failures inside the iterate loop are surfaced as
    /// [`RunStatus::Failed`] so the caller can persist the error to
    /// `run.toml` without unwinding.
    #[expect(
        clippy::too_many_lines,
        reason = "linear init-then-loop sequence; splitting it into helpers would scatter the session lifecycle across tiny methods"
    )]
    pub async fn run(
        &self,
        workflow: &WorkflowDef,
        inputs: BTreeMap<String, Value>,
    ) -> Result<RunStatus> {
        if workflow.mode != WorkflowMode::LongRunning {
            return Err(WorkflowError::invalid_workflow(
                workflow.id.clone(),
                "LongRunningExecutor::run requires a `mode: long_running` workflow",
            ));
        }
        let Some(iterate) = workflow.iterate.as_ref() else {
            return Err(WorkflowError::invalid_workflow(
                workflow.id.clone(),
                "long_running workflow has no `iterate:` phase",
            ));
        };

        // ---------- Init phase ----------
        // Run init.steps once when the run is still ad-hoc, then
        // escalate to harnessed. Idempotency: if the run is already
        // harnessed (e.g. resumed from a previous attempt) we skip
        // this entire branch and go straight to the iterate loop.
        let init_artifacts = workflow
            .init
            .as_ref()
            .map(|init| init.artifacts.clone())
            .unwrap_or_default();

        let must_init = {
            let runner = self.run_runner.lock().await;
            runner.scope().is_ad_hoc()
        };
        if must_init {
            if let Some(init) = workflow.init.as_ref() {
                let init_inputs = inputs.clone();
                let init_workflow = synthesize_phase_workflow(
                    &workflow.id,
                    "init",
                    &init.steps,
                    &workflow.inputs,
                    &init_artifacts,
                );
                if let Err(err) = self.run_phase(&init_workflow, init_inputs).await {
                    return Ok(RunStatus::Failed {
                        error: err.to_string(),
                    });
                }
            }
            // Escalate even when there are no init.steps — the SPEC
            // §9.9 invariant is that long_running runs are harnessed
            // by the time the iterate loop begins.
            self.escalate_run().await?;
        }

        // ---------- Iterate phase ----------
        let mut session_idx: u32 = 0;
        loop {
            session_idx += 1;
            if session_idx > iterate.max_sessions {
                return Ok(RunStatus::Exhausted {
                    reason: "max_sessions".to_owned(),
                });
            }

            // 1. Begin a new harnessed session and arm the watchdog.
            let (timeout_tx, mut timeout_rx) = mpsc::channel::<SessionEndTrigger>(2);
            let session_handle = {
                let mut runner = self.run_runner.lock().await;
                runner
                    .begin_session()
                    .map_err(|e| WorkflowError::StepFailed {
                        step_id: "iterate".to_owned(),
                        message: format!("begin_session: {e}"),
                    })?
            };

            // 2. Resolve bootstrap items.
            let bootstrap_block = match self
                .resolve_bootstrap(&iterate.bootstrap, &init_artifacts, &inputs)
                .await
            {
                Ok(b) => b,
                Err(err) => {
                    let mut runner = self.run_runner.lock().await;
                    let _ = runner.end_session(&session_handle, SessionEndTrigger::UserExit);
                    return Ok(RunStatus::Failed {
                        error: format!("bootstrap failed: {err}"),
                    });
                }
            };

            // 3. Execute iterate.steps with the bootstrap block + the
            //    artifacts map injected as `bootstrap_input`.
            let mut iterate_inputs = inputs.clone();
            iterate_inputs.insert(
                "bootstrap_context".to_owned(),
                Value::String(bootstrap_block),
            );
            iterate_inputs.insert(
                "session_index".to_owned(),
                Value::Number(serde_json::Number::from(session_idx)),
            );
            let iter_workflow = synthesize_phase_workflow(
                &workflow.id,
                "iterate",
                &iterate.steps,
                &workflow.inputs,
                &init_artifacts,
            );

            // Arm the per-session watchdog. The runner's existing
            // helper spawns a tokio sleep + try_send task; we still
            // wrap the engine's `run` in a `tokio::time::timeout` so
            // an aborted-watchdog path still terminates the loop in
            // bounded time. The watchdog drives the *trigger
            // attribution* (so the persisted session record carries
            // `Timeout`); the local `tokio::time::timeout` drives the
            // actual unwind.
            self.arm_watchdog(timeout_tx.clone(), iterate.session_timeout)
                .await;

            let exec_future =
                run_engine_drain(Arc::clone(&self.engine), iter_workflow, iterate_inputs);

            let exec_outcome = tokio::time::timeout(iterate.session_timeout, exec_future).await;

            // Drain the watchdog channel non-blockingly to see whether
            // it fired. If it did, the session ended on a Timeout.
            let watchdog_fired = matches!(timeout_rx.try_recv(), Ok(SessionEndTrigger::Timeout));
            let timeout_fired = exec_outcome.is_err() || watchdog_fired;
            // If the steps returned an error (not a timeout), surface
            // it as Failed rather than continuing.
            if let Ok(Err(err)) = exec_outcome {
                let mut runner = self.run_runner.lock().await;
                runner.disarm_timeout_watchdog();
                let _ = runner.end_session(&session_handle, SessionEndTrigger::UserExit);
                return Ok(RunStatus::Failed {
                    error: err.to_string(),
                });
            }

            // Disarm the watchdog and end the session.
            {
                let mut runner = self.run_runner.lock().await;
                runner.disarm_timeout_watchdog();
                let trigger = if timeout_fired {
                    SessionEndTrigger::Timeout
                } else {
                    SessionEndTrigger::Completed
                };
                runner.end_session(&session_handle, trigger).map_err(|e| {
                    WorkflowError::StepFailed {
                        step_id: "iterate".to_owned(),
                        message: format!("end_session: {e}"),
                    }
                })?;
            }

            // If the session timed out, skip the `until` evaluation
            // and just begin the next iteration.
            if timeout_fired {
                continue;
            }

            // 4. Evaluate `until` against the run's resolver.
            let resolver = RunRunnerArtifactResolver {
                runner: Arc::clone(&self.run_runner),
            };
            let inputs_value = Value::Object(map_to_obj(&inputs));
            let env: BTreeMap<String, String> = std::env::vars().collect();
            let empty_steps: BTreeMap<String, StepResult> = BTreeMap::new();
            let until_ctx = ExprContext::new(&inputs_value, &empty_steps, &Value::Null, &env)
                .with_artifacts(&init_artifacts)
                .with_artifact_resolver(&resolver);
            let stop = expr::eval_bool(&iterate.until, &until_ctx).map_err(|e| {
                WorkflowError::StepFailed {
                    step_id: "iterate.until".to_owned(),
                    message: e.to_string(),
                }
            })?;
            if stop {
                return Ok(RunStatus::Completed);
            }
        }
    }

    /// Escalate the active run from ad-hoc to harnessed scope.
    ///
    /// This is the SPEC §9.9 forced upgrade for `mode: long_running`.
    /// We synthesize an [`tmg_harness::EscalationDecision::Escalate`]
    /// (no estimated feature count yet — the count populates from
    /// `features.json` once the initializer has produced it) and feed
    /// the runner empty `features_json` / `init.sh` text. The runner
    /// truncates over any existing files via `std::fs::write`, so
    /// writing empty strings is benign when the init phase already
    /// produced the real artifacts on its own. (For workflows that
    /// don't run an initializer step the empty files are sufficient
    /// to record the harnessed transition; SPEC §9.3 considers the
    /// scope flip the source of truth.)
    async fn escalate_run(&self) -> Result<()> {
        let mut runner = self.run_runner.lock().await;
        if !runner.scope().is_ad_hoc() {
            return Ok(());
        }
        // Read whatever the init phase wrote (if anything) and feed it
        // to escalate_to_harnessed. The runner's contract is that the
        // strings are written verbatim into `features.json` / `init.sh`,
        // overwriting any existing content. We re-feed the existing
        // contents so a successful init phase isn't clobbered with
        // empty bytes.
        let features_path = runner.features().path().to_path_buf();
        let init_path = runner.init_script().path().to_path_buf();
        let features_json = read_or_default(&features_path, "{\"features\":[]}");
        let init_script = read_or_default(&init_path, "#!/bin/sh\n");

        let decision = tmg_harness::EscalationDecision::Escalate {
            reason: "mode: long_running forced harnessed scope".to_owned(),
            estimated_features: None,
        };
        runner
            .escalate_to_harnessed(&features_json, &init_script, &decision)
            .map_err(|e| WorkflowError::StepFailed {
                step_id: "init".to_owned(),
                message: format!("escalate_to_harnessed: {e}"),
            })?;
        Ok(())
    }

    /// Run a single phase (init or iterate) by delegating to the
    /// engine's standard `run` path.
    async fn run_phase(
        &self,
        phase_workflow: &WorkflowDef,
        inputs: BTreeMap<String, Value>,
    ) -> Result<()> {
        let (tx, mut rx) = mpsc::channel::<WorkflowProgress>(64);
        // Spawn a drain task so the engine can keep emitting progress
        // even when no UI is attached.
        let drain = tokio::spawn(async move {
            while rx.recv().await.is_some() {
                // Discard for now; SPEC §8.10 wires this to the TUI.
            }
        });
        let _outputs = self.engine.run(phase_workflow, inputs, tx).await?;
        drain.abort();
        Ok(())
    }

    /// Resolve bootstrap items into a single context block string.
    async fn resolve_bootstrap(
        &self,
        items: &[BootstrapItem],
        artifacts: &BTreeMap<String, PathBuf>,
        inputs: &BTreeMap<String, Value>,
    ) -> Result<String> {
        if items.is_empty() {
            return Ok(String::new());
        }
        let mut buf = String::new();
        buf.push_str("[bootstrap]\n");
        let inputs_value = Value::Object(map_to_obj(inputs));
        let env: BTreeMap<String, String> = std::env::vars().collect();
        let empty_steps: BTreeMap<String, StepResult> = BTreeMap::new();
        let resolver = RunRunnerArtifactResolver {
            runner: Arc::clone(&self.run_runner),
        };
        for item in items {
            let ctx = ExprContext::new(&inputs_value, &empty_steps, &Value::Null, &env)
                .with_artifacts(artifacts)
                .with_artifact_resolver(&resolver);
            match item {
                BootstrapItem::Run(cmd_template) => {
                    let cmd = expr::eval_string(cmd_template, &ctx)?;
                    let runner = self.run_runner.lock().await;
                    let workspace = runner.workspace_path_owned();
                    drop(runner);
                    let output = run_shell_in(&workspace, &cmd).await?;
                    let _ = writeln!(buf, "$ {cmd}");
                    let _ = writeln!(buf, "{}", output.trim_end());
                }
                BootstrapItem::Read(path_template) => {
                    let path_text = expr::eval_string(path_template, &ctx)?;
                    let target = if Path::new(&path_text).is_absolute() {
                        PathBuf::from(&path_text)
                    } else {
                        let runner = self.run_runner.lock().await;
                        runner.workspace_path_owned().join(&path_text)
                    };
                    match tokio::fs::read_to_string(&target).await {
                        Ok(text) => {
                            let _ = writeln!(buf, "# read: {path_text}");
                            let _ = writeln!(buf, "{text}");
                        }
                        Err(e) => {
                            let _ = writeln!(buf, "# read: {path_text} FAILED: {e}");
                        }
                    }
                }
                BootstrapItem::SmokeTest(step) => {
                    let summary = self.run_smoke_test(step, inputs).await?;
                    let _ = writeln!(buf, "# smoke_test: {summary}");
                }
            }
        }
        Ok(buf)
    }

    /// Dispatch a single smoke-test step through a one-shot synthetic
    /// workflow and return a short summary string.
    async fn run_smoke_test(
        &self,
        step: &StepDef,
        inputs: &BTreeMap<String, Value>,
    ) -> Result<String> {
        let phase = synthesize_phase_workflow(
            "smoke_test",
            "smoke_test",
            std::slice::from_ref(step),
            &BTreeMap::new(),
            &BTreeMap::new(),
        );
        let (tx, mut rx) = mpsc::channel::<WorkflowProgress>(32);
        let id = step.id().to_owned();
        let drain = tokio::spawn(async move {
            // Capture a one-line summary from the StepCompleted event.
            let mut last_exit: Option<i32> = None;
            while let Some(evt) = rx.recv().await {
                if let WorkflowProgress::StepCompleted { result, step_id } = evt
                    && step_id == id
                {
                    last_exit = Some(result.exit_code);
                }
            }
            last_exit
        });
        let outcome = self.engine.run(&phase, inputs.clone(), tx).await;
        let exit = drain.await.unwrap_or(None);
        match outcome {
            Ok(_) => Ok(format!(
                "{step_id} exit={exit}",
                step_id = step.id(),
                exit = exit.map_or_else(|| "?".to_owned(), |c| c.to_string()),
            )),
            Err(e) => Ok(format!("{step_id} ERROR: {e}", step_id = step.id())),
        }
    }

    /// Arm the runner's per-session watchdog so a Timeout trigger is
    /// delivered on `tx` after `duration` elapses.
    async fn arm_watchdog(&self, tx: mpsc::Sender<SessionEndTrigger>, duration: Duration) {
        let mut runner = self.run_runner.lock().await;
        runner.arm_session_timeout_watchdog(duration, tx);
    }
}

/// Run the workflow engine and drain the progress channel locally so
/// the engine stays unblocked even when no UI is attached.
async fn run_engine_drain(
    engine: Arc<WorkflowEngine>,
    workflow: WorkflowDef,
    inputs: BTreeMap<String, Value>,
) -> Result<crate::def::WorkflowOutputs> {
    let (tx, mut rx) = mpsc::channel::<WorkflowProgress>(64);
    let drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });
    let outputs = engine.run(&workflow, inputs, tx).await?;
    drain.abort();
    Ok(outputs)
}

/// Synthesize a single-phase [`WorkflowDef`] from `steps` so the
/// engine's existing `run()` path can drive it.
fn synthesize_phase_workflow(
    parent_id: &str,
    phase: &str,
    steps: &[StepDef],
    inputs: &BTreeMap<String, crate::def::InputDef>,
    _artifacts: &BTreeMap<String, PathBuf>,
) -> WorkflowDef {
    WorkflowDef {
        id: format!("{parent_id}__{phase}"),
        description: Some(format!("{phase} phase of {parent_id}")),
        // The synthetic phase workflow runs as a normal workflow under
        // the engine; long-running orchestration is handled by the
        // outer LongRunningExecutor.
        mode: WorkflowMode::Normal,
        inputs: inputs.clone(),
        steps: steps.to_vec(),
        outputs: BTreeMap::new(),
        init: None,
        iterate: None,
    }
}

/// Bridge a [`tmg_harness::RunRunner`] into the [`ArtifactResolver`]
/// trait, exposing the run-level data sources to the expression
/// evaluator.
///
/// Currently supported `tail` values:
///
/// - `features.all_passing` → bool (vacuously true on an empty list).
/// - `features.passing_count` → integer.
///
/// Unknown tails return a clear error message that lists the known
/// names. To extend: add another arm in `resolve` and document it
/// here.
struct RunRunnerArtifactResolver {
    runner: Arc<Mutex<RunRunner>>,
}

impl ArtifactResolver for RunRunnerArtifactResolver {
    fn resolve(&self, tail: &str) -> Result<Value> {
        // We need a sync read but the runner sits behind a tokio Mutex.
        // `try_lock` succeeds in the synchronous expression evaluation
        // path because the executor releases the mutex before calling
        // into the evaluator. In the rare contended case we fall back
        // to a clear error rather than blocking the runtime.
        let guard = self.runner.try_lock().map_err(|_| {
            WorkflowError::expression(
                "could not acquire RunRunner lock for artifact.* resolution; \
                 the long_running executor releases the lock before evaluating expressions",
            )
        })?;
        match tail {
            "features.all_passing" => {
                let v = guard
                    .features()
                    .all_passing()
                    .map_err(|e| WorkflowError::expression(format!("features.all_passing: {e}")))?;
                Ok(json!(v))
            }
            "features.passing_count" => {
                // Read full features list and count `passes == true`.
                let parsed = guard.features().read().map_err(|e| {
                    WorkflowError::expression(format!("features.passing_count: {e}"))
                })?;
                let count = parsed.features.iter().filter(|f| f.passes).count();
                Ok(json!(count))
            }
            other => Err(WorkflowError::expression(format!(
                "unknown artifact resolver method 'artifact.{other}' \
                 (supported: features.all_passing, features.passing_count)"
            ))),
        }
    }
}

/// Convert a `BTreeMap<String, Value>` into a `serde_json::Map`
/// preserving ordering.
fn map_to_obj(map: &BTreeMap<String, Value>) -> serde_json::Map<String, Value> {
    map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// Run a shell command in `workspace` and return combined stdout/stderr.
async fn run_shell_in(workspace: &Path, cmd: &str) -> Result<String> {
    let output = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .current_dir(workspace)
        .output()
        .await
        .map_err(|e| WorkflowError::io(format!("running bootstrap command '{cmd}'"), e))?;
    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    if !output.stderr.is_empty() {
        combined.push_str("\n--- stderr ---\n");
        combined.push_str(&String::from_utf8_lossy(&output.stderr));
    }
    Ok(combined)
}

/// Read a file or fall back to a default literal when the file is
/// missing / unreadable.
fn read_or_default(path: &Path, fallback: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|_| fallback.to_owned())
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions")]
mod tests {
    use super::*;

    #[test]
    fn run_status_completed_eq() {
        assert_eq!(RunStatus::Completed, RunStatus::Completed);
    }

    #[test]
    fn run_status_exhausted_carries_reason() {
        let s = RunStatus::Exhausted {
            reason: "max_sessions".to_owned(),
        };
        if let RunStatus::Exhausted { reason } = s {
            assert_eq!(reason, "max_sessions");
        } else {
            panic!("expected Exhausted");
        }
    }

    #[test]
    fn synthesize_phase_workflow_carries_steps() {
        let steps = vec![StepDef::Shell {
            id: "x".to_owned(),
            command: "true".to_owned(),
            timeout: None,
            when: None,
        }];
        let wf = synthesize_phase_workflow(
            "build_app",
            "init",
            &steps,
            &BTreeMap::new(),
            &BTreeMap::new(),
        );
        assert_eq!(wf.id, "build_app__init");
        assert_eq!(wf.mode, WorkflowMode::Normal);
        assert_eq!(wf.steps.len(), 1);
    }
}
