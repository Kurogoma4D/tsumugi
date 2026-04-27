//! Run-scoped artifact I/O.
//!
//! These types own the on-disk artifacts described in SPEC §9.6:
//!
//! - [`ProgressLog`]: append-only `progress.md` (one human-readable
//!   markdown timeline per run).
//! - [`SessionLog`]: directory of `session_NNN.json` records (one per
//!   session, schema in SPEC §9.4).
//!
//! Both types live alongside `run.toml` inside the run directory
//! (`<runs_dir>/<run-id>/`).

pub mod progress;
pub mod session_log;

pub use progress::ProgressLog;
pub use session_log::{SessionLog, SessionLogEntry};
