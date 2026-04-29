//! System prompt assembly from TSUMUGI.md, AGENTS.md, and the memory
//! index (`MEMORY.md`).
//!
//! Loads prompt files from:
//! - `~/.config/tsumugi/TSUMUGI.md` (global)
//! - Each directory from project root down to the current working directory
//! - Same paths for `AGENTS.md` (compatibility alias)
//!
//! The memory index ([`MEMORY_INDEX_FILE_NAMES`]) is loaded alongside
//! the prompt files when [`load_prompt_files_with_memory`] is used:
//! global memory first, then project memory, both wrapped in a header
//! that explains their role and the `memory` tool surface.
//!
//! Missing files are silently skipped. I/O errors on existing files are
//! propagated with context.

use std::path::{Path, PathBuf};

use crate::message::Message;

/// Header prefixed to the memory index when injected into the prompt.
///
/// Public so call sites that want to render the index identically (e.g.
/// the `/memory` slash command) can reuse the same framing.
pub const MEMORY_PROMPT_HEADER: &str = "\
[memory] Project & global persistent memory index. Each row is \
`- [topic](file.md) — one-line hook`. Use the `memory` tool's `read` \
action with `name = topic` to load a body before starting work; \
use `add` / `update` / `remove` to curate.";

/// Memory index file name. Matches [`tmg_memory::INDEX_FILE_NAME`] but
/// is duplicated here so `tmg-core` does not depend on `tmg-memory`.
const MEMORY_INDEX_FILE_NAME: &str = "MEMORY.md";

/// File names to search for prompt content.
const PROMPT_FILE_NAMES: &[&str] = &["TSUMUGI.md", "AGENTS.md"];

/// Load all prompt files and return them as user-role messages to inject
/// at the start of the conversation.
///
/// The order is:
/// 1. Global config directory (`~/.config/tsumugi/`)
/// 2. Each directory from `project_root` down to `cwd` (inclusive)
///
/// Within each directory, `TSUMUGI.md` is loaded before `AGENTS.md`.
///
/// # Errors
///
/// Returns an I/O error if a file exists but cannot be read.
pub fn load_prompt_files(project_root: &Path, cwd: &Path) -> Result<Vec<Message>, std::io::Error> {
    load_prompt_files_with_memory(project_root, cwd, None)
}

/// Like [`load_prompt_files`], but also injects the merged
/// [`tmg_memory`] index ahead of any further user content when the
/// caller passes one.
///
/// `memory_index` is the rendered output of
/// [`tmg_memory::MemoryStore::read_merged_index`]. When `None` or
/// blank, the memory block is skipped entirely.
///
/// Order: prompt files (global → project) → memory index. The index
/// is added last so chat templates that scan from the top see the
/// invariant project context first.
///
/// # Errors
///
/// Returns an I/O error if a prompt file exists but cannot be read.
pub fn load_prompt_files_with_memory(
    project_root: &Path,
    cwd: &Path,
    memory_index: Option<&str>,
) -> Result<Vec<Message>, std::io::Error> {
    let mut messages = Vec::new();

    // 1. Global config directory
    if let Some(config_dir) = dirs::config_dir() {
        let global_dir = config_dir.join("tsumugi");
        load_from_directory(&global_dir, &mut messages)?;
    }

    // 2. Walk from project_root down to cwd
    let directories = directories_from_root_to_cwd(project_root, cwd);
    for dir in &directories {
        load_from_directory(dir, &mut messages)?;
    }

    // 3. Memory index (project + optional global, already merged by
    //    the caller). The header makes the role of the index clear to
    //    the LLM so it knows to consult the `memory` tool.
    if let Some(index) = memory_index
        && !index.trim().is_empty()
    {
        let payload = format!("{MEMORY_PROMPT_HEADER}\n\n{index}");
        messages.push(Message::user(payload));
    }

    Ok(messages)
}

/// Convenience: read `MEMORY.md` from a memory directory if it exists.
///
/// Used by integration code that wants to thread the project-only
/// index into the prompt without depending on `tmg-memory`. Returns
/// `Ok(None)` when the file is missing.
///
/// # Errors
///
/// Surfaces I/O errors other than `NotFound`.
pub fn read_memory_index(memory_dir: &Path) -> Result<Option<String>, std::io::Error> {
    let path = memory_dir.join(MEMORY_INDEX_FILE_NAME);
    match std::fs::read_to_string(&path) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// Attempt to load prompt files from a single directory.
///
/// Missing files are silently skipped; other I/O errors are returned.
fn load_from_directory(dir: &Path, messages: &mut Vec<Message>) -> Result<(), std::io::Error> {
    for &filename in PROMPT_FILE_NAMES {
        let path = dir.join(filename);
        match std::fs::read_to_string(&path) {
            Ok(content) if !content.trim().is_empty() => {
                messages.push(Message::user(content));
            }
            Ok(_) => {
                // Empty file, skip.
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // File does not exist, skip silently.
            }
            Err(e) => {
                return Err(std::io::Error::new(
                    e.kind(),
                    format!("failed to read {}: {e}", path.display()),
                ));
            }
        }
    }
    Ok(())
}

/// Compute the list of directories from `root` down to `target` (inclusive).
///
/// If `target` is not a descendant of `root`, only `root` is returned.
fn directories_from_root_to_cwd(root: &Path, target: &Path) -> Vec<PathBuf> {
    // Canonicalize-like normalization without following symlinks is tricky.
    // We use the paths as-is and rely on `strip_prefix`.
    let Ok(relative) = target.strip_prefix(root) else {
        // target is not under root; just return root.
        return vec![root.to_path_buf()];
    };

    let mut dirs = Vec::new();
    let mut current = root.to_path_buf();
    dirs.push(current.clone());

    for component in relative.components() {
        current.push(component);
        dirs.push(current.clone());
    }

    dirs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directories_from_root_to_cwd_basic() {
        let root = Path::new("/home/user/project");
        let cwd = Path::new("/home/user/project/src/core");

        let dirs = directories_from_root_to_cwd(root, cwd);
        assert_eq!(
            dirs,
            vec![
                PathBuf::from("/home/user/project"),
                PathBuf::from("/home/user/project/src"),
                PathBuf::from("/home/user/project/src/core"),
            ]
        );
    }

    #[test]
    fn directories_from_root_to_cwd_same_dir() {
        let root = Path::new("/home/user/project");
        let cwd = Path::new("/home/user/project");

        let dirs = directories_from_root_to_cwd(root, cwd);
        assert_eq!(dirs, vec![PathBuf::from("/home/user/project")]);
    }

    #[test]
    fn directories_from_root_to_cwd_not_descendant() {
        let root = Path::new("/home/user/project");
        let cwd = Path::new("/tmp/other");

        let dirs = directories_from_root_to_cwd(root, cwd);
        assert_eq!(dirs, vec![PathBuf::from("/home/user/project")]);
    }

    #[test]
    fn load_from_directory_missing_dir() {
        let mut messages = Vec::new();
        let result = load_from_directory(Path::new("/nonexistent/path"), &mut messages);
        assert!(result.is_ok());
        assert!(messages.is_empty());
    }

    #[test]
    fn load_prompt_files_empty_dirs() {
        // Use a temp dir with no prompt files
        let tmp = std::env::temp_dir().join("tmg_test_prompt_empty");
        let _ = std::fs::create_dir_all(&tmp);

        let result = load_prompt_files(&tmp, &tmp);
        assert!(result.is_ok());
        // Should have no messages from the temp dir (global config may contribute)
        // We just verify no error occurred.

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
