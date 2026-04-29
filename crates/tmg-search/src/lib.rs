//! tmg-search: cross-session full-text search for the tsumugi agent.
//!
//! Provides:
//!
//! - [`SearchIndex`] — `SQLite` + FTS5 store of session summaries (and
//!   optional turn transcripts) at `<project>/.tsumugi/state.db`.
//! - [`SessionSearchTool`] — agent-facing tool wired into the standard
//!   [`tmg_tools::ToolRegistry`].
//! - [`redact_secrets`] — credential-masking helper shared with the
//!   trajectory recorder (#55).
//!
//! Issue #53 motivates the design: the agent needs a way to ask
//! "have I solved this before?" without depending on the full memory
//! curation flow (#52). The search layer is "raw indexed history",
//! while memory is "curated long-term knowledge"; they are
//! complementary.
//!
//! # Quick start
//!
//! ```no_run
//! use std::sync::Arc;
//! use tmg_search::{SearchIndex, SessionSearchTool};
//!
//! let index = Arc::new(SearchIndex::open(".tsumugi/state.db")?);
//! let mut registry = tmg_tools::default_registry();
//! registry.register(SessionSearchTool::new(Arc::clone(&index)));
//! # Ok::<(), tmg_search::SearchError>(())
//! ```

pub mod error;
pub mod index;
pub mod redact;
pub mod tool;

pub use error::SearchError;
pub use index::{SearchHit, SearchIndex, SearchScope, SearchStats, TurnRecord};
pub use redact::{REDACTION_TOKEN, redact_secrets};
pub use tool::{SESSION_SEARCH_TOOL_NAME, SessionSearchTool};
