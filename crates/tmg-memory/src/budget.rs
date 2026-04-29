//! Capacity limits for the memory store.
//!
//! The store is intentionally small: the index is part of every system
//! prompt and individual entries are read into context on demand. The
//! limits here express what "near capacity" means; exceeding them
//! emits a warning rather than a hard error so the agent can still
//! curate its way back under the threshold.

use serde::{Deserialize, Serialize};

/// Capacity caps for the memory store.
///
/// Values here mirror the issue spec defaults:
/// - `index_max_lines = 200`
/// - `entry_max_chars = 600`
/// - `total_files_limit = 50`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryBudget {
    /// Maximum number of lines in the `MEMORY.md` index.
    pub index_max_lines: usize,
    /// Maximum body character count for any one entry.
    pub entry_max_chars: usize,
    /// Maximum number of memory files (excluding `MEMORY.md`).
    pub total_files_limit: usize,
}

impl MemoryBudget {
    /// Default budget matching the issue spec.
    #[must_use]
    pub const fn default_const() -> Self {
        Self {
            index_max_lines: 200,
            entry_max_chars: 600,
            total_files_limit: 50,
        }
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self::default_const()
    }
}

/// Snapshot of the store's current size relative to the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetReport {
    /// Current line count of `MEMORY.md`.
    pub index_lines: usize,
    /// Current number of entry files.
    pub file_count: usize,
    /// `true` when at least one limit is met or exceeded.
    pub near_capacity: bool,
}

impl BudgetReport {
    /// Compute a [`BudgetReport`] for the given measurements.
    ///
    /// "Near capacity" fires at 80% of either limit (rounded down):
    /// the agent is nudged to curate before any limit is met, so the
    /// store has room to accept the next legitimate entry. Issue #9
    /// of PR #76 raised that the previous `>= limit` threshold fired
    /// only after the store was already full.
    #[must_use]
    pub fn from_measurements(index_lines: usize, file_count: usize, budget: &MemoryBudget) -> Self {
        let lines_threshold = (budget.index_max_lines * 4) / 5;
        let files_threshold = (budget.total_files_limit * 4) / 5;
        let near_capacity = index_lines >= lines_threshold || file_count >= files_threshold;
        Self {
            index_lines,
            file_count,
            near_capacity,
        }
    }

    /// Return `true` when the agent should be nudged to curate.
    #[must_use]
    pub const fn near_capacity(&self) -> bool {
        self.near_capacity
    }
}

/// Build the system-prompt snippet that nudges the agent to curate
/// when [`BudgetReport::near_capacity`] is `true`. Returns `None` when
/// the store is comfortably under budget.
#[must_use]
pub fn capacity_nudge(report: &BudgetReport, budget: &MemoryBudget) -> Option<String> {
    if !report.near_capacity {
        return None;
    }
    Some(format!(
        "\n\n[memory] near capacity (index {} / {} lines, {} / {} files). \
         Consider integrating overlapping entries or removing stale ones \
         via the `memory` tool's `update` / `remove` actions.",
        report.index_lines, budget.index_max_lines, report.file_count, budget.total_files_limit,
    ))
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;

    #[test]
    fn budget_defaults_match_spec() {
        let b = MemoryBudget::default();
        assert_eq!(b.index_max_lines, 200);
        assert_eq!(b.entry_max_chars, 600);
        assert_eq!(b.total_files_limit, 50);
    }

    #[test]
    fn report_near_capacity_lines() {
        let b = MemoryBudget::default();
        // 200 lines: well past the 80% threshold of 160 lines.
        let r = BudgetReport::from_measurements(200, 10, &b);
        assert!(r.near_capacity());
    }

    #[test]
    fn report_near_capacity_files() {
        let b = MemoryBudget::default();
        // 50 files: at the limit, well past the 80% threshold of 40.
        let r = BudgetReport::from_measurements(10, 50, &b);
        assert!(r.near_capacity());
    }

    /// Issue #9 regression: nudge fires at 80% of limit, not just at
    /// 100%, so the agent has time to curate before being blocked.
    #[test]
    fn nudge_fires_at_80_percent_lines() {
        let b = MemoryBudget::default();
        // 80% of 200 = 160 — exactly at the threshold.
        let r = BudgetReport::from_measurements(160, 5, &b);
        assert!(r.near_capacity(), "nudge must fire at 80% of index lines");
    }

    #[test]
    fn nudge_fires_at_80_percent_files() {
        let b = MemoryBudget::default();
        // 80% of 50 = 40 — exactly at the threshold.
        let r = BudgetReport::from_measurements(10, 40, &b);
        assert!(r.near_capacity(), "nudge must fire at 80% of file count");
    }

    #[test]
    fn report_under_budget() {
        let b = MemoryBudget::default();
        // Well under 80% on both axes.
        let r = BudgetReport::from_measurements(10, 5, &b);
        assert!(!r.near_capacity());
        assert!(capacity_nudge(&r, &b).is_none());
    }

    #[test]
    fn nudge_emitted_when_full() {
        let b = MemoryBudget::default();
        let r = BudgetReport::from_measurements(200, 50, &b);
        let msg = capacity_nudge(&r, &b).expect("nudge present");
        assert!(msg.contains("near capacity"));
    }
}
