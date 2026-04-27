//! Run runner used by the CLI startup sequence.
//!
//! [`RunRunner`] owns the active [`Run`] plus the artifact handles
//! ([`ProgressLog`], [`SessionLog`]) that the Run-scoped tools mutate.
//! It is shared across the CLI / TUI / tools as `Arc<Mutex<RunRunner>>`
//! so multiple async tasks can update the active session safely.
//!
//! Responsibilities:
//!
//! - hold the active [`Run`] and a reference-counted [`RunStore`] so
//!   that session lifecycle updates can be persisted from anywhere in
//!   the CLI / TUI.
//! - bump `session_count` and `last_session_at` on
//!   [`begin_session`](RunRunner::begin_session), append a
//!   `## Session #N` header to `progress.md`, and persist `run.toml`.
//! - on [`end_session`](RunRunner::end_session), close the active
//!   [`Session`], write `session_NNN.json` via [`SessionLog::save`],
//!   then re-persist `run.toml`.
//! - expose mutators ([`active_session_mut`](RunRunner::active_session_mut),
//!   [`save_active_session`](RunRunner::save_active_session)) so the
//!   `progress_append` / `session_summary_save` tools can update the
//!   live session record from inside the agent loop.
//!
//! Workflow integration and escalation are out of scope for this issue
//! and tracked separately.

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use chrono::Utc;
use tokio::sync::mpsc;

use crate::artifacts::{FeatureList, InitScript, ProgressLog, SessionLog};
use crate::error::HarnessError;
use crate::escalation::EscalationDecision;
use crate::run::{Run, RunScope, RunSummary};
use crate::session::{Session, SessionEndTrigger, SessionHandle};
use crate::state::{SessionState, TurnSummary};
use crate::store::RunStore;

/// Default max tokens for `session_bootstrap` output before truncation.
pub const DEFAULT_BOOTSTRAP_MAX_TOKENS: usize = 4096;

/// Capacity of the [`RunProgressEvent`] channel created by
/// [`RunRunner::progress_channel`].
///
/// Sized to accommodate the small number of background events the
/// auto-promotion gate produces (one upgrade event, plus a few status
/// transitions). The channel is intentionally bounded so a slow
/// consumer back-pressures the producer rather than buffering events
/// indefinitely.
const RUN_PROGRESS_CHANNEL_CAPACITY: usize = 16;

/// Async events emitted by [`RunRunner`] for higher-level surfaces
/// (TUI banner, CLI status line, ...).
///
/// The variants correspond to events surfaced by SPEC §9.3 and §9.10
/// that a UI may want to display without polling the runner. The TUI
/// banner integration is tracked separately (#46); the channel itself
/// is wired here so the harness can fire events even before the UI
/// subscribes.
#[derive(Debug, Clone, PartialEq)]
pub enum RunProgressEvent {
    /// The active run was just promoted from ad-hoc to harnessed.
    /// Carries the feature count from the
    /// [`crate::escalation::EscalationDecision::Escalate`] verdict (or
    /// `None` when the escalator declined to estimate).
    ScopeUpgraded {
        /// Optional `estimated_features` from the verdict.
        features_count: Option<u32>,
    },
}

/// Receiver returned by [`RunRunner::progress_channel`].
///
/// Wraps `tokio::sync::mpsc::Receiver<RunProgressEvent>` with a
/// dedicated newtype so consumers do not depend on tokio's channel
/// type directly.
pub type RunProgressReceiver = mpsc::Receiver<RunProgressEvent>;

/// Default wall-clock budget for one harnessed session.
///
/// Mirrors the CLI's `[harness] default_session_timeout` default
/// (`30m`); used as a fallback when the runner has not been
/// configured with a value from `tsumugi.toml`.
pub const DEFAULT_SESSION_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// Run runner wrapping a [`Run`] plus its artifacts.
///
/// **Concurrency:** like [`RunStore`], this runner assumes the owning
/// process is the only writer to the underlying `runs_dir`. There is
/// no inter-process locking around session begin/end; running multiple
/// `tmg` processes against the same `runs_dir` can race on
/// `session_count` updates. A locking layer is tracked as a follow-up.
#[derive(Debug)]
pub struct RunRunner {
    run: Run,
    store: Arc<RunStore>,
    progress: ProgressLog,
    session_log: SessionLog,
    /// Handle to the harnessed-only `features.json`.
    ///
    /// Constructed eagerly from the store's path layout for both ad-hoc
    /// and harnessed runs; the underlying file only exists for harnessed
    /// runs whose initializer has populated it. Callers must check
    /// [`scope`](Self::scope) before assuming the file exists.
    features: FeatureList,
    /// Handle to the harnessed-only `init.sh`.
    ///
    /// Same eager-construction contract as [`features`](Self::features).
    init_script: InitScript,
    /// Active in-flight session, populated between
    /// [`begin_session`](RunRunner::begin_session) and
    /// [`end_session`](RunRunner::end_session).
    active_session: Option<Session>,
    /// Token budget for `session_bootstrap` output.
    ///
    /// Reads from `[harness] bootstrap_max_tokens` in `tsumugi.toml`.
    /// `0` disables truncation entirely; see
    /// [`set_bootstrap_max_tokens`](Self::set_bootstrap_max_tokens) for
    /// the full semantics.
    bootstrap_max_tokens: usize,
    /// Wall-clock budget for one harnessed session.
    ///
    /// Reads from `[harness] default_session_timeout` in `tsumugi.toml`
    /// (humantime string, e.g. `"30m"`). Used by the harnessed
    /// `session_bootstrap` to cap `init.sh` execution so the inner
    /// sandbox timer matches the outer `tokio::time::timeout` deadline.
    /// Defaults to [`DEFAULT_SESSION_TIMEOUT`].
    default_session_timeout: Duration,

    /// Aggregate state observed across the active session, fed by the
    /// harness's [`AgentLoop::set_turn_observer`](tmg_core::AgentLoop::set_turn_observer)
    /// callback via [`Self::after_turn`]. Read by the
    /// [`EscalationEvaluator`](crate::escalation::EscalationEvaluator)
    /// to decide whether to fire the SPEC §9.10 trigger table.
    session_state: SessionState,

    /// Optional sender for [`RunProgressEvent`]s.
    ///
    /// Lazily allocated by [`Self::progress_channel`]; remains `None`
    /// until a consumer subscribes. When `None`, the runner does not
    /// allocate the channel — the event publication path is a no-op.
    progress_tx: Option<mpsc::Sender<RunProgressEvent>>,

    /// Concurrency gate for [`Self::escalate_to_harnessed`].
    ///
    /// Set to `true` while a promotion is in progress (or after the
    /// runner has already been promoted) so a second escalation
    /// evaluation triggered in quick succession cannot double-promote.
    /// SPEC §9.3 expects exactly one promotion per run; this atomic
    /// gate enforces that contract independently of the higher-level
    /// `Arc<Mutex<RunRunner>>` that wraps the runner.
    escalation_in_progress: AtomicBool,
}

impl RunRunner {
    /// Construct a runner over the given run and store, deriving artifact
    /// handles from the store's directory layout.
    #[must_use]
    pub fn new(run: Run, store: Arc<RunStore>) -> Self {
        let progress = ProgressLog::new(store.progress_file(&run.id));
        let session_log = SessionLog::new(store.session_log_dir(&run.id));
        let features = FeatureList::new(store.features_file(&run.id));
        let init_script = InitScript::new(store.init_script_file(&run.id));
        let session_state = SessionState::new(run.scope.clone());
        Self {
            run,
            store,
            progress,
            session_log,
            features,
            init_script,
            active_session: None,
            bootstrap_max_tokens: DEFAULT_BOOTSTRAP_MAX_TOKENS,
            default_session_timeout: DEFAULT_SESSION_TIMEOUT,
            session_state,
            progress_tx: None,
            escalation_in_progress: AtomicBool::new(false),
        }
    }

    /// Set the `bootstrap_max_tokens` budget.
    ///
    /// **Semantics:**
    /// - `0` means **no truncation** — the bootstrap payload is emitted
    ///   in full regardless of size. Use this only when you trust the
    ///   inputs (e.g. tests or short-lived projects).
    /// - any value `>= 1` is treated as a hard budget; older
    ///   `progress.md` sessions and tail-end git log lines are shed
    ///   until the serialized payload's estimated token count fits.
    ///
    /// Defaults to [`DEFAULT_BOOTSTRAP_MAX_TOKENS`].
    pub fn set_bootstrap_max_tokens(&mut self, n: usize) {
        self.bootstrap_max_tokens = n;
    }

    /// Return the current `bootstrap_max_tokens` budget.
    ///
    /// A return value of `0` indicates "no truncation"; any value
    /// `>= 1` is the hard token budget enforced by `session_bootstrap`.
    /// See [`set_bootstrap_max_tokens`](Self::set_bootstrap_max_tokens)
    /// for full semantics.
    #[must_use]
    pub fn bootstrap_max_tokens(&self) -> usize {
        self.bootstrap_max_tokens
    }

    /// Set the per-session wall-clock budget.
    ///
    /// This is the value plumbed into `session_bootstrap`'s `init.sh`
    /// execution so that the inner sandbox timer (`with_timeout`) and
    /// the outer `tokio::time::timeout` share the same deadline.
    /// Defaults to [`DEFAULT_SESSION_TIMEOUT`].
    pub fn set_default_session_timeout(&mut self, timeout: Duration) {
        self.default_session_timeout = timeout;
    }

    /// Return the current per-session wall-clock budget.
    ///
    /// Read by `session_bootstrap` to bound `init.sh` execution; see
    /// [`set_default_session_timeout`](Self::set_default_session_timeout)
    /// for the contract.
    #[must_use]
    pub fn default_session_timeout(&self) -> Duration {
        self.default_session_timeout
    }

    /// Borrow the active run.
    #[must_use]
    pub fn run(&self) -> &Run {
        &self.run
    }

    /// Borrow the active run mutably.
    pub fn run_mut(&mut self) -> &mut Run {
        &mut self.run
    }

    /// Lightweight summary of the active run, for header display etc.
    #[must_use]
    pub fn summary(&self) -> RunSummary {
        RunSummary::from_run(&self.run)
    }

    /// Borrow the run's [`ProgressLog`].
    #[must_use]
    pub fn progress_log(&self) -> &ProgressLog {
        &self.progress
    }

    /// Borrow the run's [`SessionLog`].
    #[must_use]
    pub fn session_log(&self) -> &SessionLog {
        &self.session_log
    }

    /// Borrow the run's [`FeatureList`] handle.
    ///
    /// The underlying file (`features.json`) only exists for harnessed
    /// runs after the `initializer` subagent has created it. Use
    /// [`scope`](Self::scope) to gate access at the call site.
    #[must_use]
    pub fn features(&self) -> &FeatureList {
        &self.features
    }

    /// Borrow the run's [`InitScript`] handle.
    ///
    /// Like [`features`](Self::features), the underlying file
    /// (`init.sh`) only exists for harnessed runs.
    #[must_use]
    pub fn init_script(&self) -> &InitScript {
        &self.init_script
    }

    /// Borrow the run's [`RunScope`].
    ///
    /// Convenience method for tools that need to gate registration or
    /// behaviour on whether the run is ad-hoc or harnessed.
    #[must_use]
    pub fn scope(&self) -> &RunScope {
        &self.run.scope
    }

    /// Whether the active run is harnessed.
    #[must_use]
    pub fn is_harnessed(&self) -> bool {
        matches!(self.run.scope, RunScope::Harnessed { .. })
    }

    /// Borrow the workspace path for this run.
    #[must_use]
    pub fn workspace_path(&self) -> &std::path::Path {
        &self.run.workspace_path
    }

    /// Workspace path as an owned `PathBuf` for tools that want to
    /// move the value into a spawned task.
    #[must_use]
    pub fn workspace_path_owned(&self) -> PathBuf {
        self.run.workspace_path.clone()
    }

    /// Borrow the active session, if any.
    #[must_use]
    pub fn active_session(&self) -> Option<&Session> {
        self.active_session.as_ref()
    }

    /// Borrow the active session mutably (used by Run-scoped tools to
    /// update `tool_calls_count`, `files_modified`, etc.).
    pub fn active_session_mut(&mut self) -> Option<&mut Session> {
        self.active_session.as_mut()
    }

    /// Persist the active session to `session_NNN.json` without ending
    /// it. Used by `session_summary_save` so the saved summary is
    /// durable even if the process is killed before `end_session`.
    pub fn save_active_session(&self) -> Result<(), HarnessError> {
        if let Some(session) = self.active_session.as_ref() {
            self.session_log.save(session)?;
        }
        Ok(())
    }

    /// Begin a new session.
    ///
    /// - Increments `session_count` and stamps `last_session_at` on the run.
    /// - Persists `run.toml`.
    /// - Appends a `## Session #N (timestamp) [scope]` header to
    ///   `progress.md`.
    /// - Writes the initial `session_NNN.json` so external readers see a
    ///   consistent record (with `ended_at = None`) immediately.
    pub fn begin_session(&mut self) -> Result<SessionHandle, HarnessError> {
        let now = Utc::now();
        self.run.session_count = self.run.session_count.saturating_add(1);
        self.run.last_session_at = Some(now);
        self.store.save(&self.run)?;

        self.progress
            .append_session_header(self.run.session_count, &self.run.scope, now)?;

        let mut session = Session::begin(self.run.session_count);
        // begin() stamps with Utc::now(); harmonise so both timestamps match.
        session.started_at = now;
        // Persist initial record (ended_at still None so readers can tell
        // it's the live session).
        self.session_log.save(&session)?;

        let handle = SessionHandle {
            index: session.index,
        };
        self.active_session = Some(session);
        Ok(handle)
    }

    /// End the given session, marking the trigger, persisting the final
    /// `session_NNN.json` and the run record.
    ///
    /// If `handle.index` does not match the currently-active session,
    /// returns [`HarnessError::SessionMismatch`] **without** mutating
    /// the active session or persisting any state. This avoids silently
    /// overwriting the active session with an incorrect index, which
    /// would corrupt `session_NNN.json` for the real active session.
    ///
    /// The active session is also left untouched (not consumed) so the
    /// caller can recover by retrying with the correct handle, e.g.
    /// the one returned by [`begin_session`](Self::begin_session).
    pub fn end_session(
        &mut self,
        handle: &SessionHandle,
        trigger: SessionEndTrigger,
    ) -> Result<(), HarnessError> {
        // Validate the handle against the active session before taking
        // it out of `self`, so a mismatched handle is rejected without
        // disturbing in-memory state.
        let Some(session_ref) = self.active_session.as_ref() else {
            tracing::warn!(
                handle_index = handle.index,
                "end_session called without an active session"
            );
            self.store.save(&self.run)?;
            return Ok(());
        };
        if session_ref.index != handle.index {
            return Err(HarnessError::SessionMismatch {
                expected: session_ref.index,
                actual: handle.index,
            });
        }

        // Indices match — take the session out and finalize it.
        if let Some(mut session) = self.active_session.take() {
            session.end(trigger);
            self.session_log.save(&session)?;
        }
        self.store.save(&self.run)?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // SessionState observers and the auto-promotion path (issue #37)
    // -----------------------------------------------------------------

    /// Borrow the active [`SessionState`].
    ///
    /// The state is updated on every call to [`Self::after_turn`].
    /// Callers that need a snapshot (e.g. the escalation evaluator)
    /// should clone the returned reference.
    #[must_use]
    pub fn session_state(&self) -> &SessionState {
        &self.session_state
    }

    /// Replace the active [`SessionState`] in-place.
    ///
    /// Mostly useful for tests that want to seed specific signal
    /// values; production callers feed the state via
    /// [`Self::after_turn`].
    pub fn update_session_state(&mut self, state: SessionState) {
        self.session_state = state;
    }

    /// Subscribe to [`RunProgressEvent`]s from this runner.
    ///
    /// Lazily allocates the underlying `mpsc::channel`; the first
    /// caller wins, subsequent calls return a fresh receiver while
    /// the runner keeps publishing on the same sender. (Practically
    /// only one consumer subscribes — the TUI banner pipeline — so
    /// re-subscribing is a config error rather than a supported
    /// case.)
    pub fn progress_channel(&mut self) -> RunProgressReceiver {
        let (tx, rx) = mpsc::channel(RUN_PROGRESS_CHANNEL_CAPACITY);
        self.progress_tx = Some(tx);
        rx
    }

    /// Publish an event on the [`RunProgressEvent`] channel if a
    /// consumer is subscribed; otherwise drop it silently.
    ///
    /// Uses `try_send` so a slow consumer does not block the runner.
    /// A full channel emits a `tracing::warn!` and drops the event;
    /// the operator can investigate via the trace logs.
    fn publish_progress(&self, event: RunProgressEvent) {
        let Some(tx) = self.progress_tx.as_ref() else {
            return;
        };
        if let Err(err) = tx.try_send(event) {
            tracing::warn!(?err, "RunProgressEvent dropped (channel full or closed)");
        }
    }

    /// Update [`SessionState`] from a [`TurnSummary`].
    ///
    /// `max_context_tokens` is the active LLM budget, used to derive
    /// `context_usage = tokens_used / max_context_tokens`. A `0`
    /// budget short-circuits to `0.0` so a misconfigured run does not
    /// fire the context-pressure signal.
    pub fn after_turn(&mut self, summary: &TurnSummary, max_context_tokens: usize) {
        self.session_state.observe(summary, max_context_tokens);
    }

    /// Drive the SPEC §9.3 7-step auto-promotion flow.
    ///
    /// **Caller contract:**
    ///
    /// - The active run must be [`RunScope::AdHoc`]; otherwise this
    ///   returns `Ok(false)` without action.
    /// - `features_json` and `init_script` must be the artifacts the
    ///   `Initializer` subagent produced for this run; they are
    ///   written verbatim under the run directory.
    /// - `decision` must be the
    ///   [`EscalationDecision::Escalate`](crate::escalation::EscalationDecision::Escalate)
    ///   variant. `Skip` is rejected with `Ok(false)` (the harness
    ///   should never call this method on a Skip; the check is
    ///   defensive).
    ///
    /// **Atomicity:** the gate at [`Self::escalation_in_progress`]
    /// prevents double-promotion when two escalation evaluations
    /// fire in quick succession. The first caller takes the gate,
    /// drives the flow, and on success leaves the gate set. The
    /// second caller observes the existing harnessed scope and
    /// returns `Ok(false)`.
    ///
    /// Returns `Ok(true)` when the runner was promoted, `Ok(false)`
    /// when no action was needed, or an error when the I/O fails.
    ///
    /// # Errors
    ///
    /// - [`HarnessError::Io`] / [`HarnessError::Serialize`] for
    ///   write failures of `features.json`, `init.sh`, `progress.md`
    ///   or `run.toml`.
    pub fn escalate_to_harnessed(
        &mut self,
        features_json: &str,
        init_script: &str,
        decision: &EscalationDecision,
    ) -> Result<bool, HarnessError> {
        // Gate: bail if another escalation is already in flight or has
        // already completed. The CAS pattern is sufficient because the
        // RunRunner sits behind an `Arc<Mutex<...>>` in production —
        // the gate's job is to handle the rare double-call from the
        // higher-level evaluator, not to be the primary lock.
        if self
            .escalation_in_progress
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            tracing::info!("auto-promotion already in progress / completed; skipping");
            return Ok(false);
        }

        let EscalationDecision::Escalate {
            reason,
            estimated_features,
        } = decision
        else {
            // Defensive: a Skip decision should never reach here.
            self.escalation_in_progress.store(false, Ordering::SeqCst);
            return Ok(false);
        };

        if !self.run.scope.is_ad_hoc() {
            tracing::info!("run already harnessed; skipping auto-promotion");
            // Leave the gate set so subsequent calls also short-circuit.
            return Ok(false);
        }

        // Step 1: Initializer's artifacts (`features.json`, `init.sh`)
        // are persisted to the run directory. The CLI feeds us the
        // text the Initializer subagent produced; if a future revision
        // wants the runner itself to spawn the Initializer, that
        // wiring stays in `tmg-cli` so the runner remains
        // synchronous and crate-boundary clean.
        let features_path = self.features.path().to_path_buf();
        let init_path = self.init_script.path().to_path_buf();
        if let Some(parent) = features_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| HarnessError::io(parent, e))?;
        }
        std::fs::write(&features_path, features_json)
            .map_err(|e| HarnessError::io(&features_path, e))?;
        std::fs::write(&init_path, init_script).map_err(|e| HarnessError::io(&init_path, e))?;
        // Best-effort: make init.sh executable. Failure is non-fatal
        // because `run` invokes `sh <path>` rather than `<path>` directly.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            if let Ok(meta) = std::fs::metadata(&init_path) {
                let mut perms = meta.permissions();
                perms.set_mode(0o755);
                let _ = std::fs::set_permissions(&init_path, perms);
            }
        }

        // Step 2: flip `run.toml` to the harnessed scope, recording
        // upgrade metadata so a later resume can attribute the
        // promotion correctly.
        let upgraded_run =
            self.store
                .upgrade_to_harnessed(&self.run.id, self.run.session_count, reason)?;
        self.run = upgraded_run;
        self.session_state.set_scope(self.run.scope.clone());

        // Step 3: append the SPEC §9.3 marker to progress.md.
        let timestamp = Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
        let count_label = estimated_features
            .map(|n| format!("estimated_features={n}"))
            .unwrap_or_else(|| "estimated_features=?".to_owned());
        let upgrade_block = format!(
            "\n## Session #{n} ({ts}) [SCOPE UPGRADE]\n\
             - reason: {reason}\n\
             - {count_label}\n",
            n = self.run.session_count,
            ts = timestamp,
            reason = reason,
            count_label = count_label,
        );
        // We use the path-level `OpenOptions::append` shape because
        // `ProgressLog`'s public API does not yet have a generic
        // multi-line append. This keeps the change minimal; a future
        // refactor could move the helper into ProgressLog itself.
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(self.progress.path())
            .map_err(|e| HarnessError::io(self.progress.path(), e))?;
        file.write_all(upgrade_block.as_bytes())
            .map_err(|e| HarnessError::io(self.progress.path(), e))?;
        file.flush()
            .map_err(|e| HarnessError::io(self.progress.path(), e))?;

        // Step 4: emit the TUI-facing event. Banner UI is out of scope
        // (#46), but the event channel is wired now so a consumer can
        // subscribe via `progress_channel()`.
        self.publish_progress(RunProgressEvent::ScopeUpgraded {
            features_count: *estimated_features,
        });

        // Step 5 (caller-side): re-register the harnessed-only tools
        // onto the live ToolRegistry. The runner does not hold the
        // registry directly — that's a higher-level concern wired
        // through `RunRunnerToolProvider`. The CLI is responsible for
        // installing a fresh provider after this call returns.
        //
        // Reset the per-turn signals so the same conditions do not
        // immediately re-trigger.
        self.session_state.reset_after_promotion();

        // Leave `escalation_in_progress = true` permanently so a
        // racing second call observes "already harnessed" and bails
        // with `Ok(false)`. There is exactly one promotion per run.
        Ok(true)
    }

    /// Look up the active run id for use by the higher-level CLI when
    /// driving the [`RunStore::upgrade_to_harnessed`] step out-of-band.
    #[must_use]
    pub fn run_id(&self) -> &crate::run::RunId {
        &self.run.id
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;

    fn make_runner() -> (tempfile::TempDir, RunRunner) {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let store = Arc::new(RunStore::new(tmp.path().join("runs")));
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap_or_else(|e| panic!("{e}"));
        let run = store
            .create_ad_hoc(workspace, None)
            .unwrap_or_else(|e| panic!("{e}"));
        let runner = RunRunner::new(run, store);
        (tmp, runner)
    }

    #[test]
    fn begin_session_increments_count() {
        let (_tmp, mut runner) = make_runner();
        assert_eq!(runner.run().session_count, 0);
        let handle = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(handle.index, 1);
        assert_eq!(runner.run().session_count, 1);
        assert!(runner.run().last_session_at.is_some());
        assert!(runner.active_session().is_some());
    }

    #[test]
    fn begin_session_appends_progress_header() {
        let (_tmp, mut runner) = make_runner();
        let _ = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        let progress =
            std::fs::read_to_string(runner.progress_log().path()).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            progress.starts_with("# Progress Log\n"),
            "progress.md missing init header: {progress}"
        );
        assert!(
            progress.contains("## Session #1 ("),
            "progress.md missing session header: {progress}"
        );
        assert!(progress.contains("[ad-hoc]"), "{progress}");
    }

    #[test]
    fn end_session_writes_session_log_json() {
        let (_tmp, mut runner) = make_runner();
        let handle = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        runner
            .end_session(&handle, SessionEndTrigger::Completed)
            .unwrap_or_else(|e| panic!("{e}"));
        let entries = runner
            .session_log()
            .list()
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].index, 1);

        let session = runner
            .session_log()
            .load(1)
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_else(|| panic!("session 1 should exist"));
        assert_eq!(session.end_trigger, Some(SessionEndTrigger::Completed));
        assert!(session.ended_at.is_some());
    }

    #[test]
    fn end_session_persists_run() {
        let (_tmp, mut runner) = make_runner();
        let handle = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        runner
            .end_session(&handle, SessionEndTrigger::Completed)
            .unwrap_or_else(|e| panic!("{e}"));
        let store = Arc::clone(&runner.store);
        let id = runner.run().id.clone();
        let loaded = store.load(&id).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(loaded.session_count, 1);
        assert_eq!(loaded.workspace_path, runner.run().workspace_path);
        assert_eq!(loaded.id, id);
    }

    #[test]
    fn end_session_mismatched_handle_returns_error_without_persisting() {
        let (_tmp, mut runner) = make_runner();
        let real_handle = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        let bad_handle = SessionHandle {
            index: real_handle.index + 7,
        };

        let result = runner.end_session(&bad_handle, SessionEndTrigger::Completed);
        assert!(
            matches!(
                result,
                Err(HarnessError::SessionMismatch { expected, actual })
                    if expected == real_handle.index && actual == bad_handle.index
            ),
            "expected SessionMismatch, got {result:?}",
        );

        // Active session still alive: the real handle still works.
        assert!(
            runner.active_session().is_some(),
            "active session must not be dropped on mismatch",
        );
        runner
            .end_session(&real_handle, SessionEndTrigger::Completed)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(runner.active_session().is_none());
    }

    #[test]
    fn save_active_session_persists_summary_without_ending() {
        let (_tmp, mut runner) = make_runner();
        let _ = runner.begin_session().unwrap_or_else(|e| panic!("{e}"));
        if let Some(s) = runner.active_session_mut() {
            s.summary = "halfway summary".to_owned();
        }
        runner
            .save_active_session()
            .unwrap_or_else(|e| panic!("{e}"));
        let session = runner
            .session_log()
            .load(1)
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_else(|| panic!("session 1 should exist"));
        assert_eq!(session.summary, "halfway summary");
        assert!(
            session.ended_at.is_none(),
            "save_active_session must not end the session"
        );
    }

    // ----- auto-promotion tests (issue #37) -----

    #[test]
    fn after_turn_updates_session_state() {
        let (_tmp, mut runner) = make_runner();
        let summary = TurnSummary {
            tokens_used: 4096,
            tool_calls: 3,
            files_modified: Vec::new(),
            diff_lines: 0,
            user_message: "アプリ全体 を フルスクラッチ で書き直したい".to_owned(),
        };
        runner.after_turn(&summary, 8192);
        let state = runner.session_state();
        assert!((state.context_usage - 0.5).abs() < 1e-6);
        assert_eq!(state.pending_subtasks, 3);
        assert!(state.last_user_input_size_signal);
    }

    /// Acceptance: a successful escalation flips the on-disk scope to
    /// harnessed, writes both `features.json` / `init.sh`, and
    /// appends a `[SCOPE UPGRADE]` block to `progress.md`.
    #[test]
    fn escalate_to_harnessed_promotes_run_and_artifacts() {
        let (_tmp, mut runner) = make_runner();
        // Pre-condition: ad-hoc.
        assert!(runner.run().scope.is_ad_hoc());

        let features = r#"{"features":[]}"#;
        let init_sh = "#!/bin/sh\necho ok\n";
        let decision = EscalationDecision::Escalate {
            reason: "spans crates".to_owned(),
            estimated_features: Some(36),
        };

        let promoted = runner
            .escalate_to_harnessed(features, init_sh, &decision)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(promoted, "first call must promote");
        assert!(matches!(runner.run().scope, RunScope::Harnessed { .. }));

        // Artifacts on disk.
        let features_on_disk =
            std::fs::read_to_string(runner.features().path()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(features_on_disk, features);
        let init_on_disk =
            std::fs::read_to_string(runner.init_script().path()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(init_on_disk, init_sh);

        // progress.md was extended.
        let progress =
            std::fs::read_to_string(runner.progress_log().path()).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            progress.contains("[SCOPE UPGRADE]"),
            "missing SCOPE UPGRADE block: {progress}",
        );
        assert!(
            progress.contains("spans crates"),
            "reason not recorded: {progress}",
        );
        assert!(
            progress.contains("estimated_features=36"),
            "estimate not recorded: {progress}",
        );
    }

    /// Concurrent invocation safety: a second `escalate_to_harnessed`
    /// call after a successful promotion must not double-promote.
    #[test]
    fn escalate_is_idempotent_after_first_success() {
        let (_tmp, mut runner) = make_runner();
        let decision = EscalationDecision::Escalate {
            reason: "first".to_owned(),
            estimated_features: None,
        };
        assert!(
            runner
                .escalate_to_harnessed("{\"features\":[]}", "#!/bin/sh\n", &decision)
                .unwrap_or_else(|e| panic!("{e}"))
        );

        // A second call (e.g. if a racing background task finishes
        // late) must be a no-op.
        let promoted_again = runner
            .escalate_to_harnessed("{\"features\":[]}", "#!/bin/sh\n", &decision)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(!promoted_again, "second call must not double-promote");
    }

    /// Skip decisions are rejected defensively even though the harness
    /// is supposed to filter them out before calling this method.
    #[test]
    fn escalate_to_harnessed_skip_decision_returns_false() {
        let (_tmp, mut runner) = make_runner();
        let decision = EscalationDecision::Skip {
            reason: "too small".to_owned(),
        };
        let promoted = runner
            .escalate_to_harnessed("{}", "#!/bin/sh\n", &decision)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(!promoted);
        assert!(runner.run().scope.is_ad_hoc());
    }

    /// SPEC §9.3 step 4: the scope-upgrade event reaches the
    /// progress channel when a consumer is subscribed.
    #[tokio::test]
    async fn progress_channel_observes_scope_upgrade() {
        let (_tmp, mut runner) = make_runner();
        let mut rx = runner.progress_channel();
        let decision = EscalationDecision::Escalate {
            reason: "x".to_owned(),
            estimated_features: Some(7),
        };
        runner
            .escalate_to_harnessed("{\"features\":[]}", "#!/bin/sh\n", &decision)
            .unwrap_or_else(|e| panic!("{e}"));

        let event = rx.recv().await.unwrap_or_else(|| panic!("no event"));
        assert_eq!(
            event,
            RunProgressEvent::ScopeUpgraded {
                features_count: Some(7)
            }
        );
    }

    /// End-to-end mock-launcher integration: the
    /// [`EscalationEvaluator`](crate::escalation::EscalationEvaluator)
    /// returns `escalate=true`, the runner promotes the run, and a
    /// fresh [`RunRunnerToolProvider`] sees the harnessed-only
    /// `feature_list_*` tools become registrable.
    #[tokio::test]
    async fn end_to_end_mock_launcher_promotes_and_reregisters_tools() {
        use crate::escalation::{
            EscalationConfig, EscalationEvaluator, EscalationSignal,
            launcher::testing::MockLauncher,
        };
        use crate::tools::RunRunnerToolProvider;
        use std::sync::Arc as StdArc;
        use tmg_agents::RunToolProvider;
        use tmg_tools::ToolRegistry;
        use tokio::sync::Mutex;

        let (_tmp, runner) = make_runner();
        let runner = StdArc::new(Mutex::new(runner));

        // Subscribe before kicking off so the upgrade event is captured.
        let mut progress_rx = {
            let mut guard = runner.lock().await;
            guard.progress_channel()
        };

        // Mock launcher returning a positive verdict; the evaluator
        // parses it and we feed the resulting decision into
        // `escalate_to_harnessed`.
        let launcher = StdArc::new(MockLauncher {
            response: r#"{"escalate":true,"reason":"large refactor","estimated_features":42}"#
                .to_owned(),
        });
        let evaluator = EscalationEvaluator::new(EscalationConfig::default(), Some(launcher));

        let decision = evaluator
            .evaluate(vec![EscalationSignal::UserInputSize], "n/a")
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(decision.should_escalate());

        // Drive the promotion through the runner.
        {
            let mut guard = runner.lock().await;
            let promoted = guard
                .escalate_to_harnessed(r#"{"features":[]}"#, "#!/bin/sh\nexit 0\n", &decision)
                .unwrap_or_else(|e| panic!("{e}"));
            assert!(promoted);
            assert!(matches!(guard.run().scope, RunScope::Harnessed { .. }));
        }

        // The progress event reached the consumer.
        let event = progress_rx
            .recv()
            .await
            .unwrap_or_else(|| panic!("no progress event"));
        assert_eq!(
            event,
            RunProgressEvent::ScopeUpgraded {
                features_count: Some(42)
            },
        );

        // Step 5: a fresh `RunRunnerToolProvider` constructed against
        // the now-harnessed runner sees the harnessed-only tools.
        // (The CLI installs this fresh provider; in a unit test we
        // construct it directly.)
        let provider = RunRunnerToolProvider::new(StdArc::clone(&runner)).await;
        let mut registry = ToolRegistry::new();
        assert!(provider.register_run_tool(&mut registry, "feature_list_read"));
        assert!(provider.register_run_tool(&mut registry, "feature_list_mark_passing"));
        assert!(registry.get("feature_list_read").is_some());
        assert!(registry.get("feature_list_mark_passing").is_some());
    }
}
