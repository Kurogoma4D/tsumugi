//! Built-in tool implementations.

mod file_patch;
mod file_read;
mod file_write;
mod grep_search;
mod list_dir;
mod shell_exec;

pub use file_patch::FilePatchTool;
pub use file_read::FileReadTool;
pub use file_write::FileWriteTool;
pub use grep_search::GrepSearchTool;
pub use list_dir::ListDirTool;
pub use shell_exec::ShellExecTool;

use crate::types::ToolRegistry;

/// Create a [`ToolRegistry`] pre-populated with all built-in tools.
pub fn default_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(FileReadTool);
    registry.register(FileWriteTool);
    registry.register(FilePatchTool);
    registry.register(GrepSearchTool);
    registry.register(ListDirTool);
    registry.register(ShellExecTool);
    registry
}
