//! `parallel` step handler (SPEC §8.4).
//!
//! Dispatches each child step concurrently via `tokio::spawn`. The
//! engine-wide [`tokio::sync::Semaphore`] (sized by
//! `[workflow] max_parallel_agents`) caps **agent** leaves only —
//! shell / `write_file` children run uncapped, matching the SPEC
//! ("agent ステップだけがカウント対象").
//!
//! ## Child isolation
//!
//! Each spawned task gets a fresh `BTreeMap<String, StepResult>` seeded
//! from the parent's snapshot at dispatch time. After all children
//! complete (or one fails) the produced results are merged back into
//! the parent map. This avoids needing a `Mutex` around `step_results`
//! for the duration of a parallel block — children that produce
//! disjoint id sets see no interference. If two children produce the
//! same id, the later one (by `JoinSet` completion order) wins; in
//! practice the parser's id-uniqueness check makes that case
//! impossible.
//!
//! ## Cancellation
//!
//! A private [`tokio_util::sync::CancellationToken`] is fired on the
//! first child failure. Spawned step handlers do not currently observe
//! this token directly (the LLM and shell layers manage their own
//! tokens), so cancellation is best-effort: in-flight blocking work
//! continues until its natural completion. We still abort outstanding
//! `JoinSet` tasks so the *workflow* observes prompt failure.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::json;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::def::{StepDef, StepResult};
use crate::engine::{EngineCtx, StepOutcome, dispatch_step};
use crate::error::{Result, WorkflowError};
use crate::progress::WorkflowProgress;

/// Outcome carried back from a single child task: the child's id and
/// either the new step-results snapshot it produced, or the failure
/// it raised. Aliased here because clippy flags the bare tuple as
/// `type_complexity` once it's nested in `JoinSet<...>`.
type ChildJoin = (
    String,
    std::result::Result<BTreeMap<String, StepResult>, WorkflowError>,
);

pub(crate) async fn execute(
    ctx: &EngineCtx,
    step: &StepDef,
    step_results: &mut BTreeMap<String, StepResult>,
) -> Result<StepOutcome> {
    let StepDef::Parallel { id, steps: body } = step else {
        return Err(WorkflowError::StepFailed {
            step_id: step.id().to_owned(),
            message: "internal error: parallel::execute called with non-parallel step".to_owned(),
        });
    };

    // NOTE: `StepStarted` is emitted by `engine::dispatch_step_inner`
    // for every step before delegating to the per-type handler;
    // emitting it again here would deliver duplicates.

    let cancel = CancellationToken::new();
    let mut joinset: JoinSet<ChildJoin> = JoinSet::new();

    // Snapshot the parent's results so each child sees the same
    // baseline. We wrap in `Arc` to share cheaply across tasks.
    let baseline = Arc::new(step_results.clone());

    for child in body {
        let ctx_clone = ctx.clone();
        let child_clone = child.clone();
        let cancel_clone = cancel.clone();
        let baseline_clone = Arc::clone(&baseline);
        let child_id = child.id().to_owned();
        joinset.spawn(async move {
            // Each child gets a private mutable map seeded from the
            // baseline. Owned by the spawned task so we can pass
            // `&mut local` straight into `dispatch_step` without a
            // lock — no lock is needed because no other task touches
            // this map.
            let mut local = (*baseline_clone).clone();
            let result = tokio::select! {
                biased;
                () = cancel_clone.cancelled() => {
                    Err(WorkflowError::StepFailed {
                        step_id: child_id.clone(),
                        message: "cancelled by sibling failure".to_owned(),
                    })
                }
                outcome = dispatch_step(&ctx_clone, &child_clone, &mut local) => {
                    outcome.map(|_| ())
                }
            };
            (child_id, result.map(|()| local))
        });
    }

    // Drain the join set, collecting per-child maps. On first failure
    // fire the cancellation token and explicitly `shutdown().await` the
    // join set so remaining children are aborted promptly (rather than
    // waiting for them to observe the cancel token).
    let mut child_maps: Vec<(String, BTreeMap<String, StepResult>)> =
        Vec::with_capacity(body.len());
    let mut first_error: Option<WorkflowError> = None;
    while let Some(joined) = joinset.join_next().await {
        match joined {
            Ok((child_id, Ok(map))) => child_maps.push((child_id, map)),
            Ok((child_id, Err(err))) => {
                if first_error.is_none() {
                    first_error = Some(err);
                    cancel.cancel();
                    joinset.shutdown().await;
                }
                tracing::debug!(child_id, "parallel child failed; siblings cancelled");
            }
            Err(join_err) => {
                if first_error.is_none() {
                    first_error = Some(WorkflowError::StepFailed {
                        step_id: id.clone(),
                        message: format!("parallel child task panicked: {join_err}"),
                    });
                    cancel.cancel();
                    joinset.shutdown().await;
                }
            }
        }
    }

    if let Some(err) = first_error {
        let _ = ctx
            .progress_tx
            .send(WorkflowProgress::StepFailed {
                step_id: id.clone(),
                error: err.to_string(),
            })
            .await;
        return Err(err);
    }

    // Merge children's new entries back into the parent. Each child's
    // map already includes the baseline; we only copy entries the
    // child added (or modified). The parser's id-uniqueness rule
    // rules out genuine merge collisions.
    for (_child_id, map) in child_maps {
        for (k, v) in map {
            step_results.insert(k, v);
        }
    }

    let result = StepResult {
        output: json!({"children": body.len()}),
        exit_code: 0,
        stdout: String::new(),
        stderr: String::new(),
        changed_files: Vec::new(),
    };

    let _ = ctx
        .progress_tx
        .send(WorkflowProgress::StepCompleted {
            step_id: id.clone(),
            result: result.clone(),
        })
        .await;
    step_results.insert(id.clone(), result);

    Ok(StepOutcome::Completed)
}
