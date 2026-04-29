//! Memory entry types: [`MemoryEntry`], [`MemoryType`], [`Frontmatter`].

use std::str::FromStr;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::MemoryError;

/// The frontmatter delimiter used in memory entry files.
pub(crate) const FRONTMATTER_DELIMITER: &str = "---";

/// Vocabulary for the `type` frontmatter field.
///
/// The four variants are taken from the Hermes / Claude Code curation
/// philosophy and tagged at the entry level so future retrieval logic
/// can filter by category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryType {
    /// Owner / collaborator preferences and roles.
    User,
    /// Feedback received from the human (with `why` and `when`).
    Feedback,
    /// Project structure, in-flight tasks, organisational knowledge.
    Project,
    /// Pointers to external resources (Linear, Grafana, Notion, etc.).
    Reference,
}

impl MemoryType {
    /// Return the canonical lowercase string representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Feedback => "feedback",
            Self::Project => "project",
            Self::Reference => "reference",
        }
    }
}

impl std::fmt::Display for MemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for MemoryType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "user" => Ok(Self::User),
            "feedback" => Ok(Self::Feedback),
            "project" => Ok(Self::Project),
            "reference" => Ok(Self::Reference),
            other => Err(other.to_owned()),
        }
    }
}

/// YAML frontmatter on a memory entry file.
///
/// `name`, `description`, and `type` are required. Timestamps are
/// optional in the on-disk format but are populated on every write so
/// that a file written by tsumugi always carries them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Frontmatter {
    /// Topic name (matches the file stem `<name>.md`).
    pub name: String,
    /// One-line description that surfaces in the `MEMORY.md` index.
    pub description: String,
    /// Categorical type tag.
    #[serde(rename = "type")]
    pub kind: MemoryType,
    /// Creation timestamp (RFC 3339).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,
    /// Last update timestamp (RFC 3339).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
}

/// A loaded memory entry: frontmatter plus body text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryEntry {
    /// Parsed frontmatter.
    pub frontmatter: Frontmatter,
    /// The free-form body text following the closing `---` delimiter.
    pub body: String,
}

impl MemoryEntry {
    /// Render the entry to its on-disk markdown form.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Yaml`] if the frontmatter cannot be
    /// serialised (effectively impossible for the typed schema, but
    /// surfaced so callers do not need to `unwrap`).
    pub fn to_markdown(&self) -> Result<String, MemoryError> {
        let yaml = serde_yml::to_string(&self.frontmatter).map_err(|e| MemoryError::Yaml {
            path: format!("<{}>", self.frontmatter.name),
            source: e,
        })?;
        // serde_yml emits a trailing newline; ensure a single newline
        // between the YAML block and the closing delimiter so the
        // round-trip is deterministic.
        let yaml_block = yaml.trim_end_matches('\n');
        let body = self.body.trim_start_matches(['\n', '\r']);
        Ok(format!(
            "{FRONTMATTER_DELIMITER}\n{yaml_block}\n{FRONTMATTER_DELIMITER}\n\n{body}"
        ))
    }
}

/// Parse a memory entry markdown string into [`MemoryEntry`].
///
/// `path_hint` is used purely for error messages.
///
/// # Errors
///
/// Returns [`MemoryError::InvalidFrontmatter`] for missing / malformed
/// delimiters, [`MemoryError::Yaml`] for bad YAML, and
/// [`MemoryError::InvalidType`] when `type` is outside the vocabulary
/// (caught via deserialisation).
pub fn parse_entry(content: &str, path_hint: &str) -> Result<MemoryEntry, MemoryError> {
    let trimmed = content.trim_start();
    let Some(rest) = trimmed.strip_prefix(FRONTMATTER_DELIMITER) else {
        return Err(MemoryError::invalid_frontmatter(
            path_hint,
            "file does not start with '---'",
        ));
    };

    let Some(end_idx) = find_closing_delimiter(rest) else {
        return Err(MemoryError::invalid_frontmatter(
            path_hint,
            "closing '---' delimiter not found",
        ));
    };

    let yaml_content = &rest[..end_idx];
    let body_start = end_idx + FRONTMATTER_DELIMITER.len();
    let body = rest
        .get(body_start..)
        .unwrap_or("")
        .trim_start_matches(['\n', '\r'])
        .to_owned();

    let frontmatter: Frontmatter =
        serde_yml::from_str(yaml_content).map_err(|e| MemoryError::Yaml {
            path: path_hint.to_owned(),
            source: e,
        })?;

    if frontmatter.name.is_empty() {
        return Err(MemoryError::invalid_frontmatter(
            path_hint,
            "frontmatter `name` is empty",
        ));
    }
    if frontmatter.description.is_empty() {
        return Err(MemoryError::invalid_frontmatter(
            path_hint,
            "frontmatter `description` is empty",
        ));
    }

    Ok(MemoryEntry { frontmatter, body })
}

/// Find the byte offset of the closing `---` delimiter within the text
/// after the opening delimiter has been stripped.
///
/// The closing delimiter must appear at the start of a line. Mirrors
/// the helper used in `tmg-skills::parse`.
fn find_closing_delimiter(text: &str) -> Option<usize> {
    let mut offset = 0;
    for (idx, line) in text.split('\n').enumerate() {
        if idx == 0 {
            offset += line.len() + 1;
            continue;
        }
        if line.trim() == FRONTMATTER_DELIMITER {
            return Some(offset);
        }
        offset += line.len() + 1;
    }
    None
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;

    #[test]
    fn memory_type_roundtrip() {
        for variant in [
            MemoryType::User,
            MemoryType::Feedback,
            MemoryType::Project,
            MemoryType::Reference,
        ] {
            let s = variant.as_str();
            let back: MemoryType = s.parse().expect("parses");
            assert_eq!(back, variant);
        }
    }

    #[test]
    fn parse_valid_entry() {
        let content = "\
---
name: project_layout
description: Layout description
type: project
---

Body text here.
";
        let entry = parse_entry(content, "test").expect("parses");
        assert_eq!(entry.frontmatter.name, "project_layout");
        assert_eq!(entry.frontmatter.kind, MemoryType::Project);
        assert!(entry.body.starts_with("Body text"));
    }

    #[test]
    fn rejects_missing_opening_delimiter() {
        let content = "name: foo\n---\nbody\n";
        let err = parse_entry(content, "test").expect_err("must reject");
        assert!(matches!(err, MemoryError::InvalidFrontmatter { .. }));
    }

    #[test]
    fn rejects_missing_closing_delimiter() {
        let content = "---\nname: foo\ndescription: bar\ntype: user\n";
        let err = parse_entry(content, "test").expect_err("must reject");
        assert!(matches!(err, MemoryError::InvalidFrontmatter { .. }));
    }

    #[test]
    fn rejects_invalid_type_value() {
        let content = "---\nname: foo\ndescription: bar\ntype: invalid_kind\n---\nbody\n";
        let err = parse_entry(content, "test").expect_err("must reject");
        // Bad enum value comes back as a YAML error.
        assert!(matches!(err, MemoryError::Yaml { .. }));
    }

    #[test]
    fn rejects_empty_required_field() {
        let content = "---\nname: \"\"\ndescription: bar\ntype: user\n---\nbody\n";
        let err = parse_entry(content, "test").expect_err("must reject");
        assert!(matches!(err, MemoryError::InvalidFrontmatter { .. }));
    }

    #[test]
    fn render_then_parse_roundtrips() {
        let entry = MemoryEntry {
            frontmatter: Frontmatter {
                name: "x".to_owned(),
                description: "y".to_owned(),
                kind: MemoryType::Feedback,
                created_at: None,
                updated_at: None,
            },
            body: "hello\n".to_owned(),
        };
        let text = entry.to_markdown().expect("renders");
        let back = parse_entry(&text, "test").expect("parses");
        assert_eq!(back.frontmatter.name, "x");
        assert_eq!(back.frontmatter.kind, MemoryType::Feedback);
        assert!(back.body.starts_with("hello"));
    }
}
