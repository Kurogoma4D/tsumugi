//! Property-based tests for the prompt-based tool call parser.
//!
//! Uses proptest to verify parser robustness against various
//! `<tool_call>` tag formats and malformed inputs.

use proptest::prelude::*;
use tmg_llm::parse_tool_calls;

/// Strategy to generate valid tool names (alphanumeric + underscore).
fn tool_name_strategy() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9_]{0,30}"
}

/// Strategy to generate simple JSON argument objects.
fn simple_args_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("{}".to_owned()),
        "[a-z_]{1,10}".prop_map(|key| format!(r#"{{"{key}": "value"}}"#)),
        "[a-z_]{1,10}".prop_map(|key| format!(r#"{{"{key}": 42}}"#)),
        "[a-z_]{1,10}".prop_map(|key| format!(r#"{{"{key}": true}}"#)),
    ]
}

/// Strategy to generate a valid `<tool_call>` block.
fn valid_tool_call_strategy() -> impl Strategy<Value = String> {
    (tool_name_strategy(), simple_args_strategy()).prop_map(|(name, args)| {
        format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>")
    })
}

proptest! {
    /// Valid tool_call tags should always parse without panicking and
    /// produce exactly one tool call.
    #[test]
    fn valid_tool_call_always_parses(tc in valid_tool_call_strategy()) {
        let (_, calls, errors) = parse_tool_calls(&tc);
        prop_assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        prop_assert_eq!(calls.len(), 1, "expected exactly 1 call, got {}", calls.len());
    }

    /// Arbitrary text without tool_call tags should pass through unchanged
    /// and produce no tool calls or errors.
    #[test]
    fn text_without_tags_passes_through(text in "[^<>]{0,500}") {
        let (cleaned, calls, errors) = parse_tool_calls(&text);
        prop_assert!(calls.is_empty());
        prop_assert!(errors.is_empty());
        prop_assert_eq!(cleaned.trim(), text.trim());
    }

    /// The parser should never panic on arbitrary input, even with
    /// malformed or nested tags.
    #[test]
    fn never_panics_on_arbitrary_input(text in "(.|\n){0,500}") {
        // Just ensure no panic occurs.
        let _ = parse_tool_calls(&text);
    }

    /// Multiple valid tool calls in sequence should all parse.
    #[test]
    fn multiple_valid_calls_parse(
        tc1 in valid_tool_call_strategy(),
        tc2 in valid_tool_call_strategy(),
    ) {
        let combined = format!("{tc1}\n{tc2}");
        let (_, calls, errors) = parse_tool_calls(&combined);
        prop_assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        prop_assert_eq!(calls.len(), 2);
    }

    /// Valid tool calls surrounded by arbitrary (non-tag) text should
    /// still be extracted, and the surrounding text preserved.
    #[test]
    fn tool_call_with_surrounding_text(
        prefix in "[a-zA-Z0-9 .,!?]{0,100}",
        tc in valid_tool_call_strategy(),
        suffix in "[a-zA-Z0-9 .,!?]{0,100}",
    ) {
        let input = format!("{prefix}\n{tc}\n{suffix}");
        let (cleaned, calls, errors) = parse_tool_calls(&input);
        prop_assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        prop_assert_eq!(calls.len(), 1);
        // Cleaned text should contain the prefix and suffix but not the tag.
        prop_assert!(!cleaned.contains("<tool_call>"));
        if !prefix.trim().is_empty() {
            prop_assert!(cleaned.contains(prefix.trim()), "missing prefix in cleaned text");
        }
        if !suffix.trim().is_empty() {
            prop_assert!(cleaned.contains(suffix.trim()), "missing suffix in cleaned text");
        }
    }

    /// Malformed JSON inside tool_call tags should produce parse errors,
    /// not panics.
    #[test]
    fn malformed_json_produces_error(garbage in "[^{}\"]{1,50}") {
        let input = format!("<tool_call>\n{garbage}\n</tool_call>");
        let (_, calls, errors) = parse_tool_calls(&input);
        // Should produce an error (not necessarily empty calls since
        // some random strings might be valid JSON).
        prop_assert!(
            !errors.is_empty() || !calls.is_empty(),
            "expected either an error or a parsed call"
        );
    }

    /// Unclosed tool_call tags should be treated as plain text.
    #[test]
    fn unclosed_tag_treated_as_text(content in "[a-z]{1,50}") {
        let input = format!("<tool_call>{content}");
        let (cleaned, calls, errors) = parse_tool_calls(&input);
        prop_assert!(calls.is_empty());
        prop_assert!(errors.is_empty());
        prop_assert!(cleaned.contains("<tool_call>"));
    }

    /// The parser should handle varying amounts of whitespace inside tags.
    #[test]
    fn whitespace_variations(
        name in tool_name_strategy(),
        pre_ws in "[ \t\n]{0,5}",
        post_ws in "[ \t\n]{0,5}",
    ) {
        let input = format!(
            "<tool_call>{pre_ws}{{\"name\": \"{name}\", \"arguments\": {{}}}}{post_ws}</tool_call>"
        );
        let (_, calls, errors) = parse_tool_calls(&input);
        prop_assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        prop_assert_eq!(calls.len(), 1);
        prop_assert_eq!(&calls[0].function.name, &name);
    }
}
