//! Shared types for the workflow tools (`run_workflow`, `workflow_status`).

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, OnceCell};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::def::WorkflowOutputs;
use crate::progress::WorkflowProgress;

/// 16-character lowercase hex run identifier.
///
/// Wrapping in a newtype rather than aliasing to `String` so misuse
/// (e.g. passing a workflow id where a run id was expected) becomes a
/// compile error. The string form is stable across logs and the
/// `workflow_status` tool surface.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkflowRunId(String);

/// Length, in characters, of a stringified [`WorkflowRunId`].
const RUN_ID_HEX_LEN: usize = 16;

impl WorkflowRunId {
    /// Generate a fresh random 16-char hex id.
    ///
    /// Uses `rand::random::<u64>()` which yields 64 bits of entropy →
    /// 16 hex chars. Probability of collision across N concurrent runs
    /// is roughly `N^2 / 2^65`; for billions of runs in one process
    /// this is still negligible. We do not deduplicate against the
    /// existing run map because the cost of a check would dwarf the
    /// expected clash rate.
    #[must_use]
    pub fn generate() -> Self {
        let n: u64 = rand::random();
        Self(format!("{n:016x}"))
    }

    /// Parse a string as a [`WorkflowRunId`], returning `None` if the
    /// input does not match the canonical shape (16 lowercase hex
    /// characters). Used by the `workflow_status` tool to validate
    /// LLM-supplied run ids before performing the map lookup.
    #[must_use]
    pub(crate) fn parse_lookup_key(s: &str) -> Option<Self> {
        if s.len() != RUN_ID_HEX_LEN {
            return None;
        }
        if !s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
            return None;
        }
        Some(Self(s.to_owned()))
    }

    /// Borrow the inner string form.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for WorkflowRunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Bookkeeping for one background-mode workflow run.
pub struct BackgroundRun {
    /// Wall-clock start time, used to compute `elapsed`.
    pub started_at: Instant,
    /// Bounded ring of recent progress events.
    ///
    /// We keep a small buffer rather than the full event history so
    /// long-running workflows don't grow a per-run memory leak. The
    /// `workflow_status` tool reads the *most recent* event for the
    /// `current_step` field; older events are useful only for human
    /// inspection and are discarded once the ring fills.
    ///
    /// Wrapped in `Arc` so the spawned run task and the registered
    /// `BackgroundRun` share the same buffer (the task pushes events,
    /// the status tool reads them).
    pub progress_buffer: Arc<Mutex<VecDeque<WorkflowProgress>>>,
    /// Join handle for the spawned run task. Held so callers can poll
    /// completion via [`JoinHandle::is_finished`] and so the runtime
    /// knows the task is owned. The handle is wrapped in `Mutex` so
    /// the field stays observable across the `Arc<BackgroundRun>`
    /// shared with the status tool — `JoinHandle` itself is not
    /// `Sync`. Test code joins on this handle to await completion
    /// deterministically by `take()`-ing the handle.
    pub join_handle: Mutex<Option<JoinHandle<()>>>,
    /// Per-run cancellation token. Firing this requests the workflow
    /// run to terminate promptly; any leaf step honouring
    /// `EngineCtx::cancel` will observe the cancellation. The
    /// background spawn closure observes the token in its `select!`
    /// loop so the run records a `failed` status with a clear error
    /// message even if the engine itself does not honour cancellation
    /// at every await point.
    pub cancel: CancellationToken,
    /// Final outcome of the run, set exactly once when the workflow
    /// finishes (success or failure). Stored as a `Result<...,
    /// String>` so the entire run record is `Send + 'static` without
    /// dragging the workflow error type through a public boundary.
    /// Wrapped in `Arc` so the spawn closure can write to the same
    /// cell the registered struct exposes.
    pub final_outputs: Arc<OnceCell<Result<WorkflowOutputs, String>>>,
}

/// Maximum number of progress events kept per background run.
pub(crate) const PROGRESS_BUFFER_CAP: usize = 64;

/// How long completed background runs are kept in the runs map before
/// being evicted. Five minutes is a balance between leaving the LLM
/// enough time to poll for outcomes and not growing unbounded across a
/// long-running session. The reaper runs opportunistically on every
/// `start_background` and `workflow_status` call; there is no
/// dedicated background sweeper task.
pub(crate) const BACKGROUND_RUN_RETENTION: Duration = Duration::from_secs(300);

/// Shared map of background runs.
///
/// Owned jointly by the `RunWorkflowTool` (writer) and the
/// `WorkflowStatusTool` (reader). We use `tokio::sync::Mutex` instead
/// of `std::sync::Mutex` so callers can hold the lock across `.await`
/// in the spawn closure that drains the per-run progress receiver.
///
/// ## Eviction policy
///
/// Completed runs (those whose `final_outputs` has been set) are
/// evicted after they have been resident for
/// [`BACKGROUND_RUN_RETENTION`]. Sweeping is opportunistic: every
/// `start_background` and `workflow_status` call walks the map and
/// drops any expired terminal runs before doing its own work. Running
/// (non-terminal) entries are never evicted.
pub type BackgroundRunsHandle = Arc<Mutex<HashMap<WorkflowRunId, Arc<BackgroundRun>>>>;

/// Sweep `runs` of completed entries older than `retention`. Caller
/// must already hold the map's lock.
pub(crate) fn reap_completed_runs(
    runs: &mut HashMap<WorkflowRunId, Arc<BackgroundRun>>,
    retention: Duration,
    now: Instant,
) {
    runs.retain(|_, entry| {
        // Only completed runs are candidates for eviction. A `running`
        // entry has `final_outputs.get() == None` and must be kept
        // regardless of how long it has been alive.
        if entry.final_outputs.get().is_none() {
            return true;
        }
        // Saturating subtraction guards against pathological clock
        // skew: if `started_at` is somehow in the future, treat the
        // age as zero and keep the entry.
        let age = now.saturating_duration_since(entry.started_at);
        age <= retention
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::def::WorkflowOutputs;
    use std::collections::{BTreeMap, HashMap};
    use tokio::task::JoinHandle;

    /// Build a `BackgroundRun` whose `final_outputs` is already set
    /// to a successful outcome and whose `started_at` is `started`.
    fn completed_run_at(started: Instant) -> BackgroundRun {
        let cell = Arc::new(OnceCell::new());
        let _ = cell.set(Ok(WorkflowOutputs {
            values: BTreeMap::new(),
        }));
        BackgroundRun {
            started_at: started,
            progress_buffer: Arc::new(Mutex::new(VecDeque::new())),
            join_handle: Mutex::new(None::<JoinHandle<()>>),
            cancel: CancellationToken::new(),
            final_outputs: cell,
        }
    }

    /// Build a `BackgroundRun` whose `final_outputs` has not been
    /// set (i.e. the run is still considered "running").
    fn running_run_at(started: Instant) -> BackgroundRun {
        BackgroundRun {
            started_at: started,
            progress_buffer: Arc::new(Mutex::new(VecDeque::new())),
            join_handle: Mutex::new(None::<JoinHandle<()>>),
            cancel: CancellationToken::new(),
            final_outputs: Arc::new(OnceCell::new()),
        }
    }

    /// Generated ids are 16 lowercase hex characters and roughly
    /// unique across many calls (we don't *require* uniqueness; the
    /// test just guards against a pathological generator that returns
    /// the same value every time).
    #[test]
    fn generated_ids_are_16_hex_chars() {
        let id = WorkflowRunId::generate();
        let s = id.as_str();
        assert_eq!(s.len(), 16);
        assert!(
            s.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
    }

    #[test]
    fn many_generated_ids_are_distinct() {
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1000 {
            seen.insert(WorkflowRunId::generate().0);
        }
        // We expect ~1000 distinct ids; allow up to one collision in
        // 1000 to keep the test non-flaky on any plausible generator.
        assert!(seen.len() >= 999, "got {} distinct ids", seen.len());
    }

    /// Eviction policy: completed entries older than the retention
    /// are dropped; running entries are never dropped regardless of
    /// age; recent completed entries are kept until they age out.
    #[test]
    fn reap_completed_runs_evicts_only_completed_old_entries() {
        let now = Instant::now();
        let mut runs: HashMap<WorkflowRunId, Arc<BackgroundRun>> = HashMap::new();

        // Use `checked_sub` to dodge clippy's
        // `unchecked_time_subtraction` lint; in practice `Instant::now`
        // is far enough from the epoch that these saturating cases
        // can't trigger, but the explicit form makes the intent clear.
        let old = now
            .checked_sub(Duration::from_secs(60))
            .unwrap_or_else(Instant::now);
        let nearly_now = now
            .checked_sub(Duration::from_millis(1))
            .unwrap_or_else(Instant::now);

        // Old + completed: should be evicted.
        let id_old_done = WorkflowRunId("0000000000000001".to_owned());
        runs.insert(id_old_done.clone(), Arc::new(completed_run_at(old)));
        // Old + running: must survive (running entries never expire).
        let id_old_running = WorkflowRunId("0000000000000002".to_owned());
        runs.insert(id_old_running.clone(), Arc::new(running_run_at(old)));
        // Recent + completed: must survive (still inside retention).
        let id_fresh_done = WorkflowRunId("0000000000000003".to_owned());
        runs.insert(
            id_fresh_done.clone(),
            Arc::new(completed_run_at(nearly_now)),
        );

        // Tiny retention so the 60-second-old completed entry is
        // unambiguously past the cutoff while the 1ms-old one is
        // unambiguously inside it.
        reap_completed_runs(&mut runs, Duration::from_millis(10), now);

        assert!(
            !runs.contains_key(&id_old_done),
            "old completed entry should be evicted",
        );
        assert!(
            runs.contains_key(&id_old_running),
            "running entries are never evicted",
        );
        assert!(
            runs.contains_key(&id_fresh_done),
            "recent completed entries are kept until aged out",
        );
    }

    #[test]
    fn parse_lookup_key_validates_shape() {
        // 16 lowercase hex chars: accepted.
        assert!(WorkflowRunId::parse_lookup_key("0123456789abcdef").is_some());
        // Wrong length: rejected.
        assert!(WorkflowRunId::parse_lookup_key("deadbeef").is_none());
        assert!(WorkflowRunId::parse_lookup_key("0123456789abcdef0").is_none());
        // Uppercase hex: rejected.
        assert!(WorkflowRunId::parse_lookup_key("0123456789ABCDEF").is_none());
        // Non-hex characters: rejected.
        assert!(WorkflowRunId::parse_lookup_key("0123456789abcdeg").is_none());
    }
}
