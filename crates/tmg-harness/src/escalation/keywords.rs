//! Bilingual scale-keyword detector for the auto-promotion size signal.
//!
//! [`detect_size_signal`] tokenises a user message and returns `true`
//! when **two or more distinct entries** from a small bilingual
//! dictionary are present. The two-hit rule suppresses false positives
//! that a single common phrase would otherwise produce — e.g.
//! `"OAuth対応"` alone is not enough; `"OAuth対応 + アプリ全体"` is.
//!
//! The dictionary is intentionally small and curated; coverage is biased
//! toward phrases that signal "implement an entire app / multi-feature
//! flow / from scratch / large project". Adding a phrase here is cheap;
//! the goal is precision rather than recall, since the escalator
//! subagent has the final say.
//!
//! ASCII keywords are matched case-insensitively. Mixed
//! Japanese-with-ASCII-prefix keywords (e.g. `OAuth対応`) are matched
//! against the ASCII-folded haystack: `to_ascii_lowercase` leaves
//! Japanese codepoints untouched but normalises the ASCII prefix, so a
//! user typing `oauth対応` still matches `OAuth対応`. The dictionary
//! itself is stored verbatim — only the haystack is folded.

/// Bilingual scale-indicating phrases. Each entry is one keyword; a
/// match counts toward the two-hit threshold.
const KEYWORDS: &[&str] = &[
    // Japanese
    "アプリ全体",
    "全機能",
    "OAuth対応",
    "フルスクラッチ",
    "全体実装",
    "多機能",
    "大規模",
    "全画面",
    "全コンポーネント",
    "ゼロから",
    "ゼロベース",
    "刷新",
    "再構築",
    "全面リニューアル",
    "システム全体",
    "プロジェクト全体",
    // English
    "multi-feature",
    "entire app",
    "entire application",
    "full implementation",
    "large project",
    "from scratch",
    "ground up",
    "full rewrite",
    "rebuild the entire",
    "across the codebase",
    "every module",
    "all features",
    "wide refactor",
    "system-wide",
    "end-to-end",
    "oauth flow",
    "oauth integration",
];

/// Return `true` when `user_message` contains at least two distinct
/// scale-indicating phrases from the bilingual dictionary.
///
/// **Strategy:**
///
/// - ASCII keywords match case-insensitively against an ASCII-folded
///   copy of the message. Folding is done once up front to keep the
///   inner loop cheap.
/// - Non-ASCII (Japanese) keywords match verbatim against the original
///   message — Japanese has no case fold and our dictionary entries are
///   already in canonical form.
/// - Each keyword is counted at most once even if it appears multiple
///   times, so spamming the same phrase does not bypass the two-hit
///   gate.
#[must_use]
pub fn detect_size_signal(user_message: &str) -> bool {
    if user_message.is_empty() {
        return false;
    }

    // ASCII-fold once for cheap case-insensitive matching. Japanese
    // characters pass through `to_ascii_lowercase` unchanged because
    // it only touches ASCII codepoints, so folding is also safe to
    // use against mixed Japanese-with-ASCII-prefix keywords like
    // `OAuth対応`.
    let folded = user_message.to_ascii_lowercase();

    let mut hits = 0_u32;
    for keyword in KEYWORDS {
        // Fold the keyword too so the comparison is symmetric. ASCII
        // entries become lowercase; Japanese codepoints are
        // unchanged. This makes `oauth対応` match the dictionary's
        // `OAuth対応` entry while leaving pure-Japanese entries like
        // `アプリ全体` unaffected.
        let folded_keyword = keyword.to_ascii_lowercase();
        if folded.contains(&folded_keyword) {
            hits = hits.saturating_add(1);
            if hits >= 2 {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_keyword_returns_false() {
        assert!(!detect_size_signal("アプリ全体を直したい"));
        assert!(!detect_size_signal("from scratch please"));
        assert!(!detect_size_signal("just a small fix"));
    }

    #[test]
    fn two_keywords_returns_true() {
        assert!(detect_size_signal(
            "アプリ全体 を フルスクラッチ で書き直したい"
        ));
        assert!(detect_size_signal(
            "Build the entire app from scratch please"
        ));
    }

    #[test]
    fn ascii_case_insensitive() {
        assert!(detect_size_signal(
            "FROM SCRATCH and a Full Implementation across the whole repo"
        ));
        assert!(detect_size_signal("Entire App, multi-feature, redesign"));
    }

    #[test]
    fn mixed_japanese_english_counts_as_two() {
        assert!(detect_size_signal(
            "アプリ全体 を multi-feature で再構築したい"
        ));
        // A single Japanese keyword + a single English keyword counts as 2.
        assert!(detect_size_signal("OAuth対応 を from scratch でやりたい"));
    }

    #[test]
    fn empty_message_returns_false() {
        assert!(!detect_size_signal(""));
    }

    #[test]
    fn duplicate_keyword_still_one_hit() {
        // Same keyword repeated must NOT cross the two-hit gate.
        assert!(!detect_size_signal(
            "from scratch from scratch from scratch"
        ));
    }

    #[test]
    fn benign_short_message_returns_false() {
        assert!(!detect_size_signal("hi"));
        assert!(!detect_size_signal("fix the typo"));
        assert!(!detect_size_signal("rename the variable"));
    }

    /// Japanese-with-ASCII-prefix keywords (`OAuth対応`) are matched
    /// case-insensitively on their ASCII portion: a user typing
    /// `oauth対応` (lowercase ASCII prefix) plus another keyword must
    /// fire the size signal.
    #[test]
    fn mixed_prefix_keyword_is_case_insensitive_on_ascii_portion() {
        // `oauth対応` (lowercase ASCII) + `アプリ全体` → 2 hits.
        assert!(detect_size_signal("oauth対応 と アプリ全体 をやりたい"));
        // Sanity: the original casing still matches.
        assert!(detect_size_signal("OAuth対応 と アプリ全体 をやりたい"));
        // Single keyword still does not fire even with case-insensitivity.
        assert!(!detect_size_signal("oauth対応 だけ"));
    }
}
