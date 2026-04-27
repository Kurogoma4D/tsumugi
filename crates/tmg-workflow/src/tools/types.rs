//! Shared types for the workflow tools (`run_workflow`, `workflow_status`).

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{Mutex, OnceCell};
use tokio::task::JoinHandle;

use crate::def::WorkflowOutputs;
use crate::progress::WorkflowProgress;

/// 8-character lowercase hex run identifier.
///
/// Wrapping in a newtype rather than aliasing to `String` so misuse
/// (e.g. passing a workflow id where a run id was expected) becomes a
/// compile error. The string form is stable across logs and the
/// `workflow_status` tool surface.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkflowRunId(String);

impl WorkflowRunId {
    /// Generate a fresh random 8-char hex id.
    ///
    /// Uses `rand::random::<u32>()` which yields 32 bits of entropy →
    /// 8 hex chars. Probability of collision across N concurrent runs
    /// is roughly `N^2 / 2^33`; for thousands of runs in one process
    /// this is still negligible. We do not deduplicate against the
    /// existing run map because the cost of a one-in-a-million
    /// retry-on-clash check would exceed the cost of the clash itself.
    #[must_use]
    pub fn generate() -> Self {
        let n: u32 = rand::random();
        Self(format!("{n:08x}"))
    }

    /// Wrap an existing 8-char hex string. The constructor is
    /// `pub(crate)` because external callers are expected to receive
    /// run ids from [`Self::generate`] or from the
    /// `workflow_status`-tool surface; constructing arbitrary strings
    /// here would defeat the type-safety motivation.
    #[must_use]
    pub(crate) fn from_str_unchecked(s: impl Into<String>) -> Self {
        Self(s.into())
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
    pub progress_buffer: Mutex<VecDeque<WorkflowProgress>>,
    /// Join handle for the spawned run task. Held only so the runtime
    /// knows the task is owned; we read the result via
    /// [`Self::final_outputs`] which is set by the spawn closure.
    pub join_handle: JoinHandle<()>,
    /// Final outcome of the run, set exactly once when the workflow
    /// finishes (success or failure). Stored as a `Result<...,
    /// String>` so the entire run record is `Send + 'static` without
    /// dragging the workflow error type through a public boundary.
    pub final_outputs: OnceCell<Result<WorkflowOutputs, String>>,
}

/// Maximum number of progress events kept per background run.
pub(crate) const PROGRESS_BUFFER_CAP: usize = 64;

/// Shared map of background runs.
///
/// Owned jointly by the `RunWorkflowTool` (writer) and the
/// `WorkflowStatusTool` (reader). We use `tokio::sync::Mutex` instead
/// of `std::sync::Mutex` so callers can hold the lock across `.await`
/// in the spawn closure that drains the per-run progress receiver.
pub type BackgroundRunsHandle = Arc<Mutex<HashMap<WorkflowRunId, Arc<BackgroundRun>>>>;

#[cfg(test)]
mod tests {
    use super::*;

    /// Generated ids are 8 lowercase hex characters and roughly unique
    /// across many calls (we don't *require* uniqueness; the test just
    /// guards against a pathological generator that returns the same
    /// value every time).
    #[test]
    fn generated_ids_are_8_hex_chars() {
        let id = WorkflowRunId::generate();
        let s = id.as_str();
        assert_eq!(s.len(), 8);
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
}
