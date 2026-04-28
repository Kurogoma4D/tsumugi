//! Structured Activity Pane state (SPEC §7.1 / §7.2, issue #45).
//!
//! This module collects the per-section state the Activity Pane
//! renders. It is intentionally just plain data: rendering lives in
//! [`crate::ui`] and event ingestion lives on [`crate::App`].
//!
//! The previous flat `Vec<ToolActivityEntry>` model on `App` is now a
//! field of [`ActivityPane::tool_log`]; subagent summaries (already
//! provided by `tmg-agents`) move under [`ActivityPane::subagents`];
//! [`RunProgressSection`] / [`WorkflowProgressSection`] are net-new.

use std::time::{Duration, Instant};

use tmg_agents::SubagentSummary;
use tmg_harness::{RunScope, RunSummary};

use crate::diff::DiffPreview;

/// One log entry in the tool activity sub-section.
///
/// This used to live directly on `App` as `ToolActivityEntry` (and the
/// type alias is re-exported there for backward compatibility) — see
/// [`crate::app::ToolActivityEntry`].
#[derive(Debug, Clone)]
pub struct ToolActivityEntry {
    /// Unique LLM-issued tool-call identifier. Used to pair a `ToolCall`
    /// activity entry with the matching `ToolResult` /
    /// `ToolResultCompressed` entries even when concurrent calls of
    /// the same `tool_name` interleave (issue #49 review #6).
    ///
    /// Empty when the entry was created from a path that did not carry
    /// a call id (currently no such path exists; reserved for forward
    /// compatibility).
    pub call_id: String,
    /// Tool name.
    pub tool_name: String,
    /// Display summary (call params or truncated result).
    pub summary: String,
    /// Whether this entry is an error.
    pub is_error: bool,
    /// Whether the recorded (history-bound) tool result was rewritten
    /// via tree-sitter signature extraction (issue #49). Drives the
    /// `[compressed via tree-sitter: N symbols]` hint in the renderer.
    pub compressed: bool,
    /// Number of symbols extracted by tree-sitter when `compressed`
    /// is true. Zero otherwise.
    pub compressed_symbol_count: usize,
}

/// Aggregated state for the Activity Pane.
///
/// `Clone` is intentionally not derived: the pane is owned by the
/// `App` for the lifetime of a TUI session, mutated in place by the
/// drain methods, and never duplicated. Adding `Clone` would also
/// require it on [`crate::diff::DiffPreview`], which now holds a
/// `Mutex`-backed render cache that is not meaningfully cloneable.
#[derive(Debug)]
pub struct ActivityPane {
    /// Always present: progress for the active run.
    pub run_progress: RunProgressSection,
    /// Optional: present whenever a workflow is in flight.
    pub workflow_progress: Option<WorkflowProgressSection>,
    /// Subagent summaries (cached from `tmg_agents::SubagentManager`).
    pub subagents: Vec<SubagentSummary>,
    /// Tool activity log (newest entries pushed at the back).
    pub tool_log: Vec<ToolActivityEntry>,
    /// Most recent diff preview, or `None` if no diff-producing tool
    /// has fired yet in this session.
    pub diff_preview: Option<DiffPreview>,
}

impl ActivityPane {
    /// Construct an empty pane suitable for an ad-hoc run with no
    /// active session yet.
    #[must_use]
    pub fn new() -> Self {
        Self {
            run_progress: RunProgressSection::default(),
            workflow_progress: None,
            subagents: Vec::new(),
            tool_log: Vec::new(),
            diff_preview: None,
        }
    }

    /// Apply a single [`tmg_workflow::WorkflowProgress`] event to the
    /// pane's `workflow_progress` section.
    ///
    /// This is the pure side-effect of [`crate::App::drain_workflow_progress`]
    /// extracted so unit tests can exercise the state transitions
    /// without spinning up a full `App`.
    pub fn apply_workflow_event(&mut self, ev: &tmg_workflow::WorkflowProgress) {
        match ev {
            tmg_workflow::WorkflowProgress::StepStarted { step_id, .. } => {
                let entry = self
                    .workflow_progress
                    .get_or_insert_with(|| WorkflowProgressSection::new(""));
                entry.current_step.clone_from(step_id);
                entry.iteration = None;
            }
            tmg_workflow::WorkflowProgress::LoopIteration {
                step_id,
                iteration,
                max,
            } => {
                let entry = self
                    .workflow_progress
                    .get_or_insert_with(|| WorkflowProgressSection::new(""));
                entry.current_step.clone_from(step_id);
                entry.iteration = Some((*iteration, *max));
            }
            tmg_workflow::WorkflowProgress::WorkflowCompleted { .. }
            | tmg_workflow::WorkflowProgress::StepFailed { .. } => {
                // The engine emits `StepFailed` when a step fails; the
                // run aborts unless the failure is wrapped in a
                // `Group { on_failure: continue }`. There is no
                // separate `WorkflowFailed` variant, so we treat any
                // unhandled `StepFailed` as a terminal signal and
                // clear the section so the activity pane stops
                // showing stale progress.
                self.workflow_progress = None;
            }
            _ => {}
        }
    }
}

impl Default for ActivityPane {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-run progress data driving the run-progress sub-section.
///
/// SPEC §7.1 specifies two layouts: a thicker harnessed variant with a
/// progress bar and feature lists, and a compact ad-hoc variant. The
/// renderer in [`crate::ui`] picks a variant based on
/// [`Self::scope`].
#[derive(Debug, Clone)]
pub struct RunProgressSection {
    /// Run scope. The renderer switches presentation based on this.
    pub scope: RunScope,
    /// 1-indexed session number under this run.
    pub session_num: u32,
    /// Optional `max_sessions` cap for harnessed runs.
    pub session_max: Option<u32>,
    /// Number of commits made in this run (best-effort; #46 wires the
    /// real counter).
    pub commits: u32,
    /// Number of features marked passing.
    pub features_done: u32,
    /// Total number of features in `features.json` (0 for ad-hoc).
    pub features_total: u32,
    /// Currently-targeted feature id, if known.
    pub current_feature: Option<String>,
    /// Upcoming feature ids in file order.
    pub upcoming_features: Vec<String>,
    /// Number of LLM turns observed so far in the active session
    /// (drives the ad-hoc variant's "N turns" line).
    pub turns: u32,
}

impl RunProgressSection {
    /// Apply a [`RunSummary`] snapshot, refreshing the scope label and
    /// session counter without disturbing harnessed-only fields.
    pub fn apply_summary(&mut self, summary: &RunSummary) {
        // The summary carries `scope_label` (a `&'static str`); the
        // full `RunScope` enum is mirrored by callers that hold the
        // run record. We map back to an `AdHoc` / `harnessed` shell so
        // the renderer's match still works without making the summary
        // carry the heavyweight scope variant.
        if summary.scope_label == "harnessed" {
            // Preserve the existing harnessed payload if any; else
            // synthesise an empty one. The `workflow_id` carried by
            // the summary lets us produce a usable harnessed scope
            // even if `apply_summary` is called before a richer
            // source updates the field.
            if !matches!(self.scope, RunScope::Harnessed { .. }) {
                let workflow_id = summary
                    .workflow_id
                    .clone()
                    .unwrap_or_else(|| "(unknown)".to_owned());
                self.scope = RunScope::harnessed(workflow_id, None);
            }
        } else {
            self.scope = RunScope::AdHoc;
        }
        self.session_num = summary.session_count.max(1);
    }

    /// Replace the harnessed feature counters.
    pub fn set_features(&mut self, done: u32, total: u32) {
        self.features_done = done;
        self.features_total = total;
    }

    /// Increment the turn counter for the ad-hoc variant.
    pub fn record_turn(&mut self) {
        self.turns = self.turns.saturating_add(1);
    }

    /// Whether the current scope is harnessed.
    #[must_use]
    pub fn is_harnessed(&self) -> bool {
        matches!(self.scope, RunScope::Harnessed { .. })
    }
}

impl Default for RunProgressSection {
    fn default() -> Self {
        Self {
            scope: RunScope::AdHoc,
            session_num: 1,
            session_max: None,
            commits: 0,
            features_done: 0,
            features_total: 0,
            current_feature: None,
            upcoming_features: Vec::new(),
            turns: 0,
        }
    }
}

/// Per-workflow progress data driving the workflow-progress sub-section.
///
/// `Clone` is intentionally not derived: this section is created by
/// [`ActivityPane::apply_workflow_event`] on the first relevant
/// progress event and replaced wholesale (or cleared) on terminal
/// events; no caller needs to duplicate it.
#[derive(Debug)]
pub struct WorkflowProgressSection {
    /// Workflow id (mirrors `WorkflowDef::id`).
    pub workflow_id: String,
    /// The most recently observed step id.
    pub current_step: String,
    /// `(iteration, max)` from the latest [`tmg_workflow::WorkflowProgress::LoopIteration`].
    pub iteration: Option<(u32, u32)>,
    /// Wall-clock time the section was created (i.e. when the first
    /// progress event arrived).
    pub started_at: Instant,
}

impl WorkflowProgressSection {
    /// Construct a fresh section. The workflow id is left empty when
    /// unknown; `StepStarted` events do not carry the workflow id, so
    /// it is filled in best-effort by the renderer or an outer wiring
    /// step.
    #[must_use]
    pub fn new(workflow_id: impl Into<String>) -> Self {
        Self {
            workflow_id: workflow_id.into(),
            current_step: String::new(),
            iteration: None,
            started_at: Instant::now(),
        }
    }

    /// Wall-clock duration since the section was created.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }
}

/// Compact `RunHeader` for the top header bar (SPEC §7.1).
///
/// The header bar previously showed only `model_name` /
/// `context_usage`. Issue #45 adds a `[run: <id> <scope>]` /
/// `[session: #N]` block, plus `<features_done>/<features_total>` for
/// harnessed runs. This struct is the minimal data the renderer
/// needs; it is built from a [`RunSummary`] (and feature counters
/// when harnessed) at the same point [`RunProgressSection`] is
/// updated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunHeader {
    /// 8-char short id (`RunSummary::short_id`).
    pub id_short: String,
    /// Scope label, e.g. `"ad-hoc"` or `"harnessed"`.
    pub scope_label: &'static str,
    /// 1-indexed session number.
    pub session_num: u32,
    /// `Some((done, total))` for harnessed scope; `None` for ad-hoc.
    pub features: Option<(u32, u32)>,
}

impl RunHeader {
    /// Build a header from a [`RunSummary`] without harnessed feature
    /// counters (caller fills those in later).
    #[must_use]
    pub fn from_summary(summary: &RunSummary) -> Self {
        Self {
            id_short: summary.short_id().to_owned(),
            scope_label: summary.scope_label,
            session_num: summary.session_count.max(1),
            features: None,
        }
    }

    /// Format the header as the SPEC §7.1 string. Used by the
    /// renderer; broken out so tests can assert on the exact output
    /// without spinning up a `Frame`.
    #[must_use]
    pub fn format(&self) -> String {
        match self.features {
            Some((done, total)) => format!(
                "[run: {id} {scope} {done}/{total}] [session: #{n}]",
                id = self.id_short,
                scope = self.scope_label,
                n = self.session_num,
            ),
            None => format!(
                "[run: {id} {scope}] [session: #{n}]",
                id = self.id_short,
                scope = self.scope_label,
                n = self.session_num,
            ),
        }
    }
}

/// Convenience: re-export of `ToolActivityEntry` for the activity
/// pane's caller. Defined here so [`ActivityPane`] is self-contained.
pub use ToolActivityEntry as ActivityToolEntry;

/// Diff preview file path, used by the activity pane when picking up
/// a `file_write` /  `file_patch` tool result.
///
/// Exposed so the renderer can format the preview's source path as a
/// short filename without re-implementing path-truncation logic. Kept
/// as a free function rather than a method so callers don't need to
/// take a [`DiffPreview`] reference (they can pass any `&Path`).
#[must_use]
pub fn short_path_label(path: &std::path::Path) -> String {
    let mut parts: Vec<String> = path
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    if parts.len() > 3 {
        let tail = parts.split_off(parts.len() - 3);
        return format!(".../{}", tail.join("/"));
    }
    parts.join("/")
}

/// Re-export so consumers can build their own diff previews without
/// importing the diff module separately.
pub use crate::diff::DiffPreview as ActivityDiffPreview;

#[expect(
    clippy::expect_used,
    clippy::single_char_pattern,
    reason = "tests use expect/single-char patterns for clarity; the workspace policy denies them in production code"
)]
#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn make_ad_hoc_summary() -> RunSummary {
        let run = tmg_harness::Run::new_ad_hoc(PathBuf::from("/tmp/ws"));
        RunSummary::from_run(&run)
    }

    fn make_harnessed_summary() -> RunSummary {
        let mut run = tmg_harness::Run::new_ad_hoc(PathBuf::from("/tmp/ws"));
        run.scope = RunScope::harnessed("wf-build", Some(50));
        run.session_count = 12;
        let mut summary = RunSummary::from_run(&run);
        summary.workflow_id = Some("wf-build".to_owned());
        summary.scope_label = "harnessed";
        summary.session_count = 12;
        summary
    }

    #[test]
    fn run_header_ad_hoc_format() {
        let mut header = RunHeader::from_summary(&make_ad_hoc_summary());
        header.features = None;
        let formatted = header.format();
        assert!(formatted.contains("ad-hoc"));
        assert!(formatted.contains("session: #1"));
        assert!(!formatted.contains("/"));
    }

    #[test]
    fn run_header_harnessed_features() {
        let mut header = RunHeader::from_summary(&make_harnessed_summary());
        header.features = Some((22, 47));
        let formatted = header.format();
        assert!(formatted.contains("harnessed 22/47"));
        assert!(formatted.contains("session: #12"));
    }

    #[test]
    fn run_progress_section_switches_on_apply_summary() {
        let mut section = RunProgressSection::default();
        assert!(!section.is_harnessed());
        section.apply_summary(&make_harnessed_summary());
        assert!(section.is_harnessed());
        assert_eq!(section.session_num, 12);
        section.apply_summary(&make_ad_hoc_summary());
        assert!(!section.is_harnessed());
    }

    #[test]
    fn run_progress_section_features_setter() {
        let mut section = RunProgressSection::default();
        section.set_features(3, 10);
        assert_eq!(section.features_done, 3);
        assert_eq!(section.features_total, 10);
    }

    #[test]
    fn workflow_progress_section_records_iteration() {
        let mut section = WorkflowProgressSection::new("wf-x");
        section.current_step = "build".to_owned();
        section.iteration = Some((3, 10));
        assert_eq!(section.workflow_id, "wf-x");
        assert_eq!(section.iteration, Some((3, 10)));
    }

    #[test]
    fn short_path_label_truncates_long_paths() {
        let p = std::path::Path::new("/a/b/c/d/e/f.txt");
        let s = short_path_label(p);
        assert!(s.starts_with("..."));
        assert!(s.ends_with("f.txt"));
    }

    #[test]
    fn short_path_label_keeps_short_paths() {
        let p = std::path::Path::new("a/b.txt");
        let s = short_path_label(p);
        assert_eq!(s, "a/b.txt");
    }

    #[test]
    fn workflow_section_appears_on_step_started_and_clears_on_completed() {
        use tmg_workflow::{WorkflowOutputs, WorkflowProgress};

        let mut pane = ActivityPane::new();
        assert!(pane.workflow_progress.is_none());

        // StepStarted should create the section.
        pane.apply_workflow_event(&WorkflowProgress::StepStarted {
            step_id: "build".to_owned(),
            step_type: "shell",
        });
        let section = pane
            .workflow_progress
            .as_ref()
            .expect("section should be populated after StepStarted");
        assert_eq!(section.current_step, "build");

        // LoopIteration should refresh and add iteration counter.
        pane.apply_workflow_event(&WorkflowProgress::LoopIteration {
            step_id: "iterate".to_owned(),
            iteration: 3,
            max: 10,
        });
        let section = pane.workflow_progress.as_ref().expect("section");
        assert_eq!(section.current_step, "iterate");
        assert_eq!(section.iteration, Some((3, 10)));

        // WorkflowCompleted should clear the section.
        pane.apply_workflow_event(&WorkflowProgress::WorkflowCompleted {
            outputs: WorkflowOutputs::default(),
        });
        assert!(pane.workflow_progress.is_none());
    }

    #[test]
    fn workflow_section_appears_on_step_started_and_clears_on_step_failed() {
        use tmg_workflow::WorkflowProgress;

        let mut pane = ActivityPane::new();
        assert!(pane.workflow_progress.is_none());

        pane.apply_workflow_event(&WorkflowProgress::StepStarted {
            step_id: "build".to_owned(),
            step_type: "shell",
        });
        assert!(pane.workflow_progress.is_some());

        // StepFailed should clear the section since the engine does
        // not emit a separate `WorkflowFailed` variant — an unhandled
        // step failure aborts the run.
        pane.apply_workflow_event(&WorkflowProgress::StepFailed {
            step_id: "build".to_owned(),
            error: "oops".to_owned(),
        });
        assert!(pane.workflow_progress.is_none());
    }
}
