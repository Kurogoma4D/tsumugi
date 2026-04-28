//! Transient overlay banners (SPEC §7.1 scope-upgrade notification).
//!
//! A [`TransientBanner`] is a short-lived message displayed at the top
//! of the chat pane. The TUI inserts one whenever it observes a
//! [`tmg_harness::RunProgressEvent::ScopeUpgraded`] event; the renderer
//! draws it as an overlay until [`TransientBanner::is_expired`]
//! returns `true`, at which point the App clears the field.

use std::time::{Duration, Instant};

/// Default lifetime for a scope-upgrade banner.
pub const DEFAULT_BANNER_TTL: Duration = Duration::from_secs(3);

/// A short-lived message overlaid on the chat pane.
#[derive(Debug, Clone)]
pub struct TransientBanner {
    /// Display text. Rendered as a single line.
    pub text: String,
    /// Wall-clock time the banner was created.
    pub started_at: Instant,
    /// How long the banner should remain visible.
    pub ttl: Duration,
}

impl TransientBanner {
    /// Create a banner with the [default TTL][DEFAULT_BANNER_TTL].
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self::with_ttl(text, DEFAULT_BANNER_TTL)
    }

    /// Create a banner with an explicit TTL.
    #[must_use]
    pub fn with_ttl(text: impl Into<String>, ttl: Duration) -> Self {
        Self {
            text: text.into(),
            started_at: Instant::now(),
            ttl,
        }
    }

    /// Whether the banner has outlived its TTL.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.started_at.elapsed() >= self.ttl
    }

    /// Construct the canonical scope-upgrade banner text from the
    /// optional feature count carried by [`tmg_harness::RunProgressEvent::ScopeUpgraded`].
    #[must_use]
    pub fn scope_upgrade(features_count: Option<u32>) -> Self {
        let text = match features_count {
            Some(n) => format!("Run upgraded to harnessed ({n} features tracked)"),
            None => "Run upgraded to harnessed".to_owned(),
        };
        Self::new(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_upgrade_text_with_count() {
        let b = TransientBanner::scope_upgrade(Some(47));
        assert!(b.text.contains("47 features"));
    }

    #[test]
    fn scope_upgrade_text_without_count() {
        let b = TransientBanner::scope_upgrade(None);
        assert_eq!(b.text, "Run upgraded to harnessed");
    }

    #[test]
    fn fresh_banner_is_not_expired() {
        let b = TransientBanner::new("hello");
        assert!(!b.is_expired());
    }

    #[test]
    fn zero_ttl_banner_is_immediately_expired() {
        let b = TransientBanner::with_ttl("hello", Duration::from_nanos(0));
        // After at least one tick of the clock the banner is expired.
        std::thread::sleep(Duration::from_millis(1));
        assert!(b.is_expired());
    }
}
