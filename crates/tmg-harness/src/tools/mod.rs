//! Run-scoped [`Tool`](tmg_tools::Tool) implementations.
//!
//! These tools differ from those in [`tmg_tools::default_registry`]
//! because they hold a reference to the active [`RunRunner`]: they
//! mutate the run's `progress.md` / active `Session` rather than
//! interacting with the workspace filesystem directly.
//!
//! Wiring: the CLI registers them on the global [`ToolRegistry`] after
//! creating the [`RunRunner`]; they are intentionally **not** included
//! in `default_registry()` so a Run-less code path (e.g. `--prompt`
//! one-shot mode) does not accidentally surface them to the LLM.

mod progress_append;
mod session_bootstrap;
mod session_summary_save;

pub use progress_append::ProgressAppendTool;
pub use session_bootstrap::SessionBootstrapTool;
pub use session_summary_save::SessionSummarySaveTool;
