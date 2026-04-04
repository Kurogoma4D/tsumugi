//! `grep_search` tool: pattern search across files.

use std::fmt::Write;
use std::path::{Path, PathBuf};

use crate::error::ToolError;
use crate::path_util::validate_path;
use crate::types::{Tool, ToolResult};

/// Maximum number of matching lines to return.
const MAX_MATCHES: usize = 500;

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

    fn execute(
        &self,
        params: serde_json::Value,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + '_>,
    > {
        Box::pin(self.execute_inner(params))
    }
}

impl GrepSearchTool {
    async fn execute_inner(&self, params: serde_json::Value) -> Result<ToolResult, ToolError> {
        let Some(pattern_str) = params.get("pattern").and_then(serde_json::Value::as_str) else {
            return Err(ToolError::invalid_params(
                "missing required parameter: pattern",
            ));
        };

        let regex = regex::Regex::new(pattern_str)?;

        let search_path = match params.get("path").and_then(serde_json::Value::as_str) {
            Some(p) => validate_path(p)?,
            None => PathBuf::from("."),
        };

        let include_filter = params
            .get("include")
            .and_then(serde_json::Value::as_str)
            .map(String::from);

        // Run the search in a blocking task since it does synchronous I/O.
        let result = tokio::task::spawn_blocking(move || {
            search_files(&search_path, &regex, include_filter.as_deref())
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
) -> Result<ToolResult, ToolError> {
    let mut matches = Vec::new();
    let mut truncated = false;

    if root.is_file() {
        search_single_file(root, regex, &mut matches, &mut truncated);
    } else if root.is_dir() {
        walk_and_search(root, regex, include, &mut matches, &mut truncated)?;
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

/// Walk a directory and search matching files.
fn walk_and_search(
    dir: &Path,
    regex: &regex::Regex,
    include: Option<&str>,
    matches: &mut Vec<String>,
    truncated: &mut bool,
) -> Result<(), ToolError> {
    if *truncated {
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

    let mut entries: Vec<PathBuf> = Vec::new();
    for entry in read_dir {
        let Ok(entry) = entry else { continue };
        entries.push(entry.path());
    }
    entries.sort();

    for entry in &entries {
        if *truncated {
            break;
        }

        let name = entry
            .file_name()
            .map(|n| n.to_string_lossy())
            .unwrap_or_default();

        // Skip hidden files/directories.
        if name.starts_with('.') {
            continue;
        }

        if entry.is_dir() {
            walk_and_search(entry, regex, include, matches, truncated)?;
        } else if entry.is_file() && matches_include_filter(&name, include) {
            search_single_file(entry, regex, matches, truncated);
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

    #[tokio::test]
    async fn grep_basic_search() {
        let dir = std::env::temp_dir().join("tmg_tools_test_grep");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("a.txt"), "hello world\nfoo bar\nhello again\n").ok();
        std::fs::write(dir.join("b.txt"), "no match here\n").ok();

        let tool = GrepSearchTool;
        let result = tool
            .execute(serde_json::json!({
                "pattern": "hello",
                "path": dir.to_str().unwrap()
            }))
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
        let result = tool
            .execute(serde_json::json!({
                "pattern": "fn main",
                "path": dir.to_str().unwrap(),
                "include": "*.rs"
            }))
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
        let result = tool
            .execute(serde_json::json!({
                "pattern": "zzzzz",
                "path": dir.to_str().unwrap()
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.output.contains("No matches"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn grep_invalid_regex() {
        let tool = GrepSearchTool;
        let result = tool
            .execute(serde_json::json!({
                "pattern": "[invalid",
                "path": "/tmp"
            }))
            .await;
        assert!(result.is_err());
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
    }
}
