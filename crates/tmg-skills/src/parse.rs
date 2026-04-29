//! SKILL.md frontmatter parsing.
//!
//! A SKILL.md file has the format:
//!
//! ```text
//! ---
//! name: my-skill
//! description: A short description
//! invocation: auto
//! allowed_tools:
//!   - file_read
//!   - grep_search
//! ---
//!
//! Instruction body goes here...
//! ```
//!
//! The frontmatter is delimited by `---` lines at the start of the file.

use crate::error::SkillError;
use crate::types::SkillFrontmatter;

/// The frontmatter delimiter.
const FRONTMATTER_DELIMITER: &str = "---";

/// Parse a SKILL.md file into its frontmatter and body.
///
/// Returns `(frontmatter, body)` where `body` is everything after the
/// closing `---` delimiter.
///
/// # Errors
///
/// Returns [`SkillError::InvalidFrontmatter`] if the file does not contain
/// valid frontmatter delimiters, or [`SkillError::YamlParse`] if the YAML
/// content cannot be deserialized.
pub fn parse_skill_md(
    content: &str,
    file_path: &str,
) -> Result<(SkillFrontmatter, String), SkillError> {
    let trimmed = content.trim_start();

    // The file must start with `---`.
    let Some(rest) = trimmed.strip_prefix(FRONTMATTER_DELIMITER) else {
        return Err(SkillError::invalid_frontmatter(
            file_path,
            "file does not start with '---'",
        ));
    };

    // Find the closing `---`.
    let Some(end_idx) = find_closing_delimiter(rest) else {
        return Err(SkillError::invalid_frontmatter(
            file_path,
            "closing '---' delimiter not found",
        ));
    };

    let yaml_content = &rest[..end_idx];
    let body_start = end_idx + FRONTMATTER_DELIMITER.len();
    let body = rest
        .get(body_start..)
        .unwrap_or("")
        .trim_start_matches(['\n', '\r']);

    let frontmatter: SkillFrontmatter =
        serde_yml::from_str(yaml_content).map_err(|e| SkillError::YamlParse {
            path: file_path.to_owned(),
            source: e,
        })?;

    Ok((frontmatter, body.to_owned()))
}

/// Find the byte offset of the closing `---` delimiter within the text
/// after the opening delimiter has been stripped.
///
/// The closing delimiter must appear at the start of a line. Uses
/// cumulative byte offsets to avoid mismatch when YAML content contains
/// `---` substrings.
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
#[expect(clippy::unwrap_used, reason = "test assertions")]
#[expect(clippy::panic, reason = "test assertions")]
mod tests {
    use super::*;
    use crate::types::InvocationPolicy;

    /// Serialize a `SkillFrontmatter` back to YAML string (for roundtrip testing).
    fn serialize_frontmatter(frontmatter: &SkillFrontmatter) -> Result<String, serde_yml::Error> {
        serde_yml::to_string(frontmatter)
    }

    /// Format a full SKILL.md file from frontmatter and body.
    fn format_skill_md(
        frontmatter: &SkillFrontmatter,
        body: &str,
    ) -> Result<String, serde_yml::Error> {
        let yaml = serialize_frontmatter(frontmatter)?;
        Ok(format!("---\n{yaml}---\n\n{body}"))
    }

    #[test]
    fn parse_basic_frontmatter() {
        let content = "\
---
name: test-skill
description: A test skill
---

This is the body.
";
        let (fm, body) = parse_skill_md(content, "test.md").unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(fm.name, "test-skill");
        assert_eq!(fm.description, "A test skill");
        assert_eq!(fm.invocation, InvocationPolicy::Auto);
        assert!(fm.allowed_tools.is_none());
        // Optional fields added in #54 are absent on legacy SKILL.md
        // files (this is the agentskills.io compat case).
        assert!(fm.version.is_none());
        assert!(fm.created_at.is_none());
        assert!(fm.updated_at.is_none());
        assert!(fm.provenance.is_none());
        assert_eq!(body, "This is the body.\n");
    }

    #[test]
    fn parse_frontmatter_with_provenance() {
        // Auto-generated skills from `skill_critic` carry a `provenance:
        // agent` tag plus version/timestamps. Confirm the parser
        // accepts the extended shape.
        let content = "\
---
name: deploy-rust
description: Publish a Rust crate to crates.io
version: \"1.0.0\"
provenance: agent
---

Steps...
";
        let (fm, _) = parse_skill_md(content, "test.md").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(fm.version.as_deref(), Some("1.0.0"));
        assert_eq!(fm.provenance, Some(crate::types::Provenance::Agent));
    }

    #[test]
    fn parse_full_frontmatter() {
        let content = "\
---
name: code-review
description: Reviews code for best practices
invocation: explicit_only
allowed_tools:
  - file_read
  - grep_search
---

Review the code carefully.
";
        let (fm, body) = parse_skill_md(content, "test.md").unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(fm.name, "code-review");
        assert_eq!(fm.description, "Reviews code for best practices");
        assert_eq!(fm.invocation, InvocationPolicy::ExplicitOnly);
        assert_eq!(
            fm.allowed_tools,
            Some(vec!["file_read".to_owned(), "grep_search".to_owned()])
        );
        assert_eq!(body, "Review the code carefully.\n");
    }

    #[test]
    fn parse_missing_opening_delimiter() {
        let content = "name: test\n---\nbody";
        let err = parse_skill_md(content, "bad.md").unwrap_err();
        assert!(err.to_string().contains("does not start with '---'"));
    }

    #[test]
    fn parse_missing_closing_delimiter() {
        let content = "---\nname: test\n";
        let err = parse_skill_md(content, "bad.md").unwrap_err();
        assert!(
            err.to_string()
                .contains("closing '---' delimiter not found")
        );
    }

    #[test]
    fn parse_invalid_yaml() {
        let content = "---\n[invalid yaml\n---\nbody";
        let err = parse_skill_md(content, "bad.md").unwrap_err();
        assert!(err.to_string().contains("yaml parse error"));
    }

    #[test]
    fn roundtrip_frontmatter() {
        let original = SkillFrontmatter {
            name: "roundtrip-test".to_owned(),
            description: "Tests roundtrip".to_owned(),
            invocation: InvocationPolicy::ExplicitOnly,
            allowed_tools: Some(vec!["file_read".to_owned()]),
            version: None,
            created_at: None,
            updated_at: None,
            provenance: None,
        };

        let yaml = serialize_frontmatter(&original).unwrap_or_else(|e| panic!("{e}"));
        let parsed: SkillFrontmatter = serde_yml::from_str(&yaml).unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(original, parsed);
    }

    #[test]
    fn parse_closing_delimiter_on_last_line_without_trailing_newline() {
        let content = "---\nname: x\ndescription: y\n---";
        let (fm, body) = parse_skill_md(content, "f.md").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(fm.name, "x");
        assert_eq!(fm.description, "y");
        assert_eq!(body, "");
    }

    #[test]
    fn format_and_reparse() {
        let original = SkillFrontmatter {
            name: "format-test".to_owned(),
            description: "Tests formatting".to_owned(),
            invocation: InvocationPolicy::Auto,
            allowed_tools: None,
            version: None,
            created_at: None,
            updated_at: None,
            provenance: None,
        };
        let body = "Do the thing.\n";

        let md = format_skill_md(&original, body).unwrap_or_else(|e| panic!("{e}"));
        let (parsed_fm, parsed_body) =
            parse_skill_md(&md, "formatted.md").unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(original, parsed_fm);
        assert_eq!(parsed_body, body);
    }

    #[expect(clippy::expect_used, reason = "proptest assertions")]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_invocation_policy() -> impl Strategy<Value = InvocationPolicy> {
            prop_oneof![
                Just(InvocationPolicy::Auto),
                Just(InvocationPolicy::ExplicitOnly),
            ]
        }

        /// Generate a non-empty string that does not contain `---` on its own
        /// line and has no leading/trailing whitespace that would be lost in
        /// YAML roundtrip.
        fn arb_yaml_safe_string() -> impl Strategy<Value = String> {
            "[a-zA-Z0-9][a-zA-Z0-9 _-]{0,30}[a-zA-Z0-9]"
        }

        fn arb_allowed_tools() -> impl Strategy<Value = Option<Vec<String>>> {
            prop_oneof![
                Just(None),
                proptest::collection::vec("[a-z_]{1,20}", 1..5).prop_map(Some),
            ]
        }

        fn arb_frontmatter() -> impl Strategy<Value = SkillFrontmatter> {
            (
                arb_yaml_safe_string(),
                arb_yaml_safe_string(),
                arb_invocation_policy(),
                arb_allowed_tools(),
            )
                .prop_map(|(name, description, invocation, allowed_tools)| {
                    SkillFrontmatter {
                        name,
                        description,
                        invocation,
                        allowed_tools,
                        version: None,
                        created_at: None,
                        updated_at: None,
                        provenance: None,
                    }
                })
        }

        proptest! {
            #[test]
            fn frontmatter_roundtrip(fm in arb_frontmatter()) {
                let yaml = serialize_frontmatter(&fm).expect("serialize");
                let parsed: SkillFrontmatter =
                    serde_yml::from_str(&yaml).expect("deserialize");
                prop_assert_eq!(&fm, &parsed);
            }

            #[test]
            fn format_parse_roundtrip(fm in arb_frontmatter(), body in "[a-zA-Z0-9 .!?\n]{0,100}") {
                let md = format_skill_md(&fm, &body).expect("format");
                let (parsed_fm, parsed_body) =
                    parse_skill_md(&md, "prop.md").expect("parse");
                prop_assert_eq!(&fm, &parsed_fm);
                // Body is trimmed of leading newlines by the parser.
                let expected_body = body.trim_start_matches(['\n', '\r']);
                prop_assert_eq!(parsed_body, expected_body);
            }
        }
    }
}
