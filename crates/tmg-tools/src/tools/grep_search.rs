//! `grep_search` tool: pattern search across files.

use std::fmt::Write;
use std::path::{Path, PathBuf};

use tmg_sandbox::SandboxContext;

use crate::error::ToolError;
use crate::path_util::{validate_and_resolve, validate_path};
use crate::types::{Tool, ToolResult};

/// Maximum number of matching lines to return.
const MAX_MATCHES: usize = 500;

/// Maximum directory recursion depth to prevent infinite loops from symlinks.
const MAX_DEPTH: usize = 32;

/// Search files for a regex pattern.
pub struct GrepSearchTool;

impl Tool for GrepSearchTool {
    fn name(&self) -> &'static str {
        "grep_search"
    }

    fn description(&self) -> &'static str {
        "Search files for lines matching a regex pattern. Returns matching lines with \
         file paths and line numbers."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for."
                },
                "path": {
                    "type": "string",
                    "description": "The directory or file to search in. Defaults to current directory."
                },
                "include": {
                    "type": "string",
                    "description": "File name filter (glob pattern, e.g. '*.rs'). Only files matching this pattern are searched."
                }
            },
            "required": ["pattern"],
            "additionalProperties": false
        })
    }

    fn execute<'a>(
        &'a self,
        params: serde_json::Value,
        ctx: &'a SandboxContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + 'a>,
    > {
        Box::pin(self.execute_inner(params, ctx))
    }
}

impl GrepSearchTool {
    async fn execute_inner(
        &self,
        params: serde_json::Value,
        ctx: &SandboxContext,
    ) -> Result<ToolResult, ToolError> {
        let Some(pattern_str) = params.get("pattern").and_then(serde_json::Value::as_str) else {
            return Err(ToolError::invalid_params(
                "missing required parameter: pattern",
            ));
        };

        let regex = regex::Regex::new(pattern_str)?;

        // When no path is supplied, default to the current directory.
        // The current directory is then resolved via the same
        // `validate_and_resolve` path to surface a clear sandbox
        // denial if the cwd is outside the workspace.
        let search_path = if let Some(p) = params.get("path").and_then(serde_json::Value::as_str) {
            let resolved = validate_and_resolve(p, ctx)?;
            ctx.check_path_access(&resolved)?;
            resolved
        } else {
            let default = PathBuf::from(".");
            let _ = validate_path(&default)?;
            let resolved = if default.is_absolute() {
                default
            } else {
                ctx.workspace().join(default)
            };
            ctx.check_path_access(&resolved)?;
            resolved
        };

        let include_filter = params
            .get("include")
            .and_then(serde_json::Value::as_str)
            .map(String::from);

        // Snapshot the workspace path so the blocking task can do its
        // own per-file sandbox check without holding `&SandboxContext`
        // (which is `!Send` into `spawn_blocking`'s `'static` closure
        // because it borrows `ctx`). The check is structurally
        // identical to `SandboxContext::check_path_access` for a
        // workspace-scoped sandbox: workspace prefix OR system-read
        // allowlist OR tsumugi config dir.
        //
        // We deliberately pull the data we need rather than doing
        // arbitrary `clone()`s on `SandboxConfig`; the workspace path
        // is `Arc`-friendly via [`SandboxContext::workspace`].
        let workspace = ctx.workspace().to_path_buf();
        let mode_unrestricted = ctx.mode().is_unrestricted();

        // Run the search in a blocking task since it does synchronous I/O.
        let result = tokio::task::spawn_blocking(move || {
            search_files(
                &search_path,
                &regex,
                include_filter.as_deref(),
                &workspace,
                mode_unrestricted,
            )
        })
        .await
        .map_err(|e| ToolError::io("grep search task panicked", std::io::Error::other(e)))??;

        Ok(result)
    }
}

/// Perform the actual file search.
fn search_files(
    root: &Path,
    regex: &regex::Regex,
    include: Option<&str>,
    workspace: &Path,
    mode_unrestricted: bool,
) -> Result<ToolResult, ToolError> {
    let mut matches = Vec::new();
    let mut truncated = false;

    if root.is_file() {
        search_single_file(root, regex, &mut matches, &mut truncated);
    } else if root.is_dir() {
        walk_and_search(
            root,
            regex,
            include,
            &mut matches,
            &mut truncated,
            0,
            workspace,
            mode_unrestricted,
        )?;
    } else {
        return Err(ToolError::io(
            format!("'{}' is not a file or directory", root.display()),
            std::io::Error::new(std::io::ErrorKind::NotFound, "not found"),
        ));
    }

    if matches.is_empty() {
        return Ok(ToolResult::success("No matches found."));
    }

    let mut output = matches.join("\n");
    if truncated {
        let _ = write!(output, "\n... (truncated at {MAX_MATCHES} matches)");
    }

    Ok(ToolResult::success(output))
}

/// Decide whether a file the walker landed on is allowed to be read
/// under the sandbox boundary.
///
/// Conservatively re-implements the workspace-prefix branch of
/// `SandboxContext::check_path_access` so the blocking walk thread
/// does not need to thread an `&ctx` reference past
/// `spawn_blocking`. A canonical `path` is required: the caller is
/// expected to pass the resolved (symlink-followed) absolute form.
///
/// We only allow paths under the workspace prefix here; system-read
/// paths (`/usr`, `/bin`, …) are intentionally excluded from the
/// recursive grep walk, because legitimate searches over `/usr` /
/// `/bin` are vanishingly rare and the resulting noise (binary
/// files, generated headers, etc.) is far more likely to be a
/// misconfiguration than a useful query. The single-root branch of
/// `execute_inner` already calls `check_path_access` so an explicit
/// `path: "/usr/include"` request still passes the front gate.
fn is_within_workspace(path: &Path, workspace: &Path, mode_unrestricted: bool) -> bool {
    if mode_unrestricted {
        return true;
    }
    path.starts_with(workspace)
}

/// Walk a directory and search matching files.
#[expect(
    clippy::too_many_arguments,
    reason = "snake-cased state for workspace boundary + recursion limits; restructuring as a struct adds noise without clarifying intent"
)]
fn walk_and_search(
    dir: &Path,
    regex: &regex::Regex,
    include: Option<&str>,
    matches: &mut Vec<String>,
    truncated: &mut bool,
    depth: usize,
    workspace: &Path,
    mode_unrestricted: bool,
) -> Result<(), ToolError> {
    if *truncated || depth > MAX_DEPTH {
        return Ok(());
    }

    let read_dir = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => return Ok(()),
        Err(e) => {
            return Err(ToolError::io(
                format!("reading directory '{}'", dir.display()),
                e,
            ));
        }
    };

    let mut entries: Vec<std::fs::DirEntry> = Vec::new();
    for entry in read_dir {
        let Ok(entry) = entry else { continue };
        entries.push(entry);
    }
    entries.sort_by_key(std::fs::DirEntry::file_name);

    for entry in &entries {
        if *truncated {
            break;
        }

        let name = entry.file_name();
        let name = name.to_string_lossy();

        // Skip hidden files/directories.
        if name.starts_with('.') {
            continue;
        }

        // Use entry.file_type() which does not follow symlinks. This
        // both prevents infinite loops on symlink cycles AND closes
        // the symlink-escape sandbox bypass: a file entry that is
        // really a symlink (e.g. workspace/.../passwd ->
        // /etc/passwd) reports `is_symlink()` here and falls through
        // every branch, so its target is never read.
        let Ok(file_type) = entry.file_type() else {
            continue;
        };

        let entry_path = entry.path();

        if file_type.is_dir() {
            // Recurse only into real directories (not symlinked ones).
            walk_and_search(
                &entry_path,
                regex,
                include,
                matches,
                truncated,
                depth + 1,
                workspace,
                mode_unrestricted,
            )?;
        } else if file_type.is_file() && matches_include_filter(&name, include) {
            // Belt-and-suspenders: even though `is_symlink()` filters
            // out symlinks above, double-check that the canonical
            // path of every regular file we are about to read still
            // sits inside the sandbox boundary. This catches edge
            // cases such as a hard link that resolves outside the
            // workspace, and matches the per-file invariant the
            // reviewer asked for in issue-#47 follow-up #4.
            let canonical =
                std::fs::canonicalize(&entry_path).unwrap_or_else(|_| entry_path.clone());
            if !is_within_workspace(&canonical, workspace, mode_unrestricted) {
                continue;
            }
            search_single_file(&entry_path, regex, matches, truncated);
        }
    }

    Ok(())
}

/// Check if a filename matches the include glob pattern.
///
/// Uses a simple glob matching: `*` matches any sequence of non-separator chars.
fn matches_include_filter(filename: &str, include: Option<&str>) -> bool {
    let Some(pattern) = include else {
        return true;
    };
    simple_glob_match(pattern, filename)
}

/// A minimal glob matcher supporting `*` as a wildcard.
fn simple_glob_match(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();

    if parts.len() == 1 {
        // No wildcard, exact match.
        return pattern == text;
    }

    let mut pos = 0;

    // First part must be a prefix.
    let Some(first) = parts.first() else {
        return true;
    };
    if !first.is_empty() {
        if !text.starts_with(*first) {
            return false;
        }
        pos = first.len();
    }

    // Last part must be a suffix.
    let Some(last) = parts.last() else {
        return true;
    };
    if !last.is_empty() && !text[pos..].ends_with(*last) {
        return false;
    }
    let end = if last.is_empty() {
        text.len()
    } else {
        text.len() - last.len()
    };

    if pos > end {
        return false;
    }

    // Middle parts must appear in order.
    for part in &parts[1..parts.len() - 1] {
        if part.is_empty() {
            continue;
        }
        let Some(idx) = text[pos..end].find(*part) else {
            return false;
        };
        pos += idx + part.len();
    }

    true
}

/// Search a single file for matching lines.
fn search_single_file(
    path: &Path,
    regex: &regex::Regex,
    matches: &mut Vec<String>,
    truncated: &mut bool,
) {
    // Skip binary or unreadable files.
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };

    for (line_no, line) in content.lines().enumerate() {
        if regex.is_match(line) {
            matches.push(format!("{}:{}: {line}", path.display(), line_no + 1));
            if matches.len() >= MAX_MATCHES {
                *truncated = true;
                return;
            }
        }
    }
}

#[expect(clippy::unwrap_used, reason = "test assertions")]
#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> SandboxContext {
        SandboxContext::test_default()
    }

    #[tokio::test]
    async fn grep_basic_search() {
        let dir = std::env::temp_dir().join("tmg_tools_test_grep");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("a.txt"), "hello world\nfoo bar\nhello again\n").ok();
        std::fs::write(dir.join("b.txt"), "no match here\n").ok();

        let tool = GrepSearchTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "hello",
                    "path": dir.to_str().unwrap()
                }),
                &sandbox,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.output.contains("hello world"));
        assert!(result.output.contains("hello again"));
        assert!(!result.output.contains("no match"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn grep_with_include_filter() {
        let dir = std::env::temp_dir().join("tmg_tools_test_grep_include");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("a.rs"), "fn main() {}\n").ok();
        std::fs::write(dir.join("b.txt"), "fn main() {}\n").ok();

        let tool = GrepSearchTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "fn main",
                    "path": dir.to_str().unwrap(),
                    "include": "*.rs"
                }),
                &sandbox,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.output.contains("a.rs"));
        assert!(!result.output.contains("b.txt"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn grep_no_matches() {
        let dir = std::env::temp_dir().join("tmg_tools_test_grep_none");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("a.txt"), "hello\n").ok();

        let tool = GrepSearchTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "zzzzz",
                    "path": dir.to_str().unwrap()
                }),
                &sandbox,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.output.contains("No matches"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn grep_invalid_regex() {
        let tool = GrepSearchTool;
        let sandbox = ctx();
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "[invalid",
                    "path": "/tmp"
                }),
                &sandbox,
            )
            .await;
        assert!(result.is_err());
    }

    /// Issue #47 follow-up #4: a symlink inside the workspace
    /// pointing at a sensitive path outside the workspace must NOT be
    /// dereferenced by the recursive walk. We use `Full` mode for the
    /// workspace lookup so `check_path_access` doesn't reject the
    /// search root, but the per-file walker still needs to refuse to
    /// follow workspace-internal symlinks pointing outside.
    #[cfg(unix)]
    #[tokio::test]
    async fn grep_does_not_follow_symlink_to_outside_file() {
        use tmg_sandbox::{SandboxConfig, SandboxMode};

        let dir = std::env::temp_dir().join("tmg_tools_test_grep_symlink_escape");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok();

        // Write a benign file inside the workspace so the walker has
        // something to find under normal conditions.
        std::fs::write(dir.join("ok.txt"), "marker-found-inside\n").ok();

        // Place an "evil" file outside the workspace and link to it
        // from inside.
        let outside_dir = std::env::temp_dir().join("tmg_tools_test_grep_symlink_escape_outside");
        let _ = std::fs::remove_dir_all(&outside_dir);
        std::fs::create_dir_all(&outside_dir).ok();
        let outside_file = outside_dir.join("secret.txt");
        std::fs::write(&outside_file, "marker-secret-from-outside\n").ok();
        let _ = std::os::unix::fs::symlink(&outside_file, dir.join("escape.txt"));

        // Workspace is `dir`; symlink target lives outside it.
        let canonical_workspace = std::fs::canonicalize(&dir).unwrap();
        let sandbox = SandboxContext::new(
            SandboxConfig::new(&canonical_workspace).with_mode(SandboxMode::WorkspaceWrite),
        );

        let tool = GrepSearchTool;
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "marker",
                    "path": canonical_workspace.to_str().unwrap()
                }),
                &sandbox,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(
            result.output.contains("marker-found-inside"),
            "expected legitimate workspace match, got: {}",
            result.output
        );
        assert!(
            !result.output.contains("marker-secret-from-outside"),
            "symlink escape: walker followed escape.txt out of the workspace; output: {}",
            result.output
        );

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&outside_dir);
    }

    #[test]
    fn glob_matching() {
        assert!(simple_glob_match("*.rs", "main.rs"));
        assert!(simple_glob_match("*.rs", "lib.rs"));
        assert!(!simple_glob_match("*.rs", "main.txt"));
        assert!(simple_glob_match("test_*", "test_foo"));
        assert!(!simple_glob_match("test_*", "foo_test"));
        assert!(simple_glob_match("*", "anything"));
        assert!(simple_glob_match("exact", "exact"));
        assert!(!simple_glob_match("exact", "other"));
        // Bounds check: long pattern match against short string.
        assert!(!simple_glob_match("*longmatch", "short"));
    }
}
