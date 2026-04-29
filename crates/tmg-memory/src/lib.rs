//! tmg-memory: project-scoped persistent memory for the tsumugi agent.
//!
//! The memory layer is a flat directory of markdown files plus a single
//! `MEMORY.md` index. Both layers (project and global) use the same
//! schema; the index is loaded into every system prompt and individual
//! entry bodies are read on demand via the `memory` tool.
//!
//! # Layout
//!
//! ```text
//! <project_root>/.tsumugi/memory/
//!   MEMORY.md                       # 1 row per entry, capped at 200 lines
//!   project_layout.md               # YAML frontmatter + body
//!   feedback_review_style.md
//!   ...
//!
//! ~/.config/tsumugi/memory/         # optional low-priority layer
//!   MEMORY.md
//!   ...
//! ```
//!
//! # Quick start
//!
//! ```no_run
//! use std::sync::Arc;
//! use tmg_memory::{MemoryStore, MemoryTool, MemoryType};
//!
//! let store = Arc::new(MemoryStore::open("."));
//! // Register the tool with the standard registry:
//! let mut registry = tmg_tools::default_registry();
//! registry.register(MemoryTool::new(Arc::clone(&store)));
//! ```

pub mod budget;
pub mod entry;
pub mod error;
pub mod store;
pub mod tool;

pub use budget::{BudgetReport, MemoryBudget, capacity_nudge};
pub use entry::{Frontmatter, MemoryEntry, MemoryType, parse_entry};
pub use error::MemoryError;
pub use store::{INDEX_FILE_NAME, MemoryScope, MemoryStore};
pub use tool::{MEMORY_TOOL_NAME, MemoryTool};
