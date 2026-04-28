//! Shared tokenize helpers for the rest of the workspace.
//!
//! [`count_tokens_or_estimate`] is the canonical entry point for any
//! caller that wants an "exact tokenize, fall back to a heuristic on
//! failure" count. The original implementation lived in two places
//! (`tmg-core::TokenCounter::request_update` and the harness
//! `session_bootstrap` helper); centralising it here gives us:
//!
//! - one place to apply the chars/4 heuristic,
//! - one place to emit a `tracing::warn!` on the first tokenize
//!   failure (and `tracing::debug!` for subsequent failures so the
//!   logs do not spam when the LLM server is genuinely down),
//! - one place that downstream call sites can depend on without
//!   reaching into `LlmClient::tokenize` directly.
//!
//! The heuristic budget is a deliberate `chars / 4` integer divide
//! with a `.max(1)` floor so a non-empty string never reports zero
//! tokens. This matches the pre-existing
//! [`crate::estimate_tokens_heuristic`] helper used by callers that
//! cannot be made async.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::client::LlmClient;

/// Tracks whether we have already emitted a `warn!` for a tokenize
/// failure. The first failure is loud; subsequent failures drop to
/// `debug!` so the agent loop is not flooded when the LLM endpoint is
/// genuinely down. This is a process-wide latch — re-arming would
/// require a more nuanced policy than the SPEC currently mandates.
static TOKENIZE_WARNED: AtomicBool = AtomicBool::new(false);

/// Estimate the token count of `text` using a `chars / 4` heuristic.
///
/// Returns `1` for the empty string so the caller never has to
/// special-case zero-budget paths.
#[must_use]
pub fn estimate_tokens_heuristic(text: &str) -> usize {
    (text.len() / 4).max(1)
}

/// Count the tokens in `text` using the LLM server's `POST /tokenize`
/// endpoint, falling back to [`estimate_tokens_heuristic`] on failure.
///
/// The first failure is logged at `warn!`; subsequent failures land at
/// `debug!`. The caller never sees an error: a heuristic estimate is
/// returned instead so token-budget decisions can keep flowing even
/// when the server is unreachable.
///
/// # Cancel safety
///
/// This call is cancel-safe: dropping the future after it has issued
/// the HTTP request simply discards the response. The fallback path is
/// pure CPU and cannot be cancelled, but it returns instantly.
pub async fn count_tokens_or_estimate(client: &LlmClient, text: &str) -> usize {
    match client.tokenize(text).await {
        Ok(count) => {
            tracing::trace!(
                token_count = count,
                text_len = text.len(),
                "tokenize succeeded",
            );
            count
        }
        Err(e) => {
            let estimate = estimate_tokens_heuristic(text);
            // Compare-and-swap so only the first racing caller emits
            // the loud warning. All subsequent (and concurrent)
            // failures land at debug.
            if TOKENIZE_WARNED
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                tracing::warn!(
                    error = %e,
                    text_len = text.len(),
                    estimate,
                    "tokenize failed; using chars/4 heuristic (further failures will be logged at debug level)",
                );
            } else {
                tracing::debug!(
                    error = %e,
                    text_len = text.len(),
                    estimate,
                    "tokenize failed; using chars/4 heuristic",
                );
            }
            estimate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_tokens_heuristic_floor_one() {
        assert_eq!(estimate_tokens_heuristic(""), 1);
        assert_eq!(estimate_tokens_heuristic("a"), 1);
        assert_eq!(estimate_tokens_heuristic("abcd"), 1);
        assert_eq!(estimate_tokens_heuristic("abcde"), 1);
        assert_eq!(estimate_tokens_heuristic(&"a".repeat(8)), 2);
        assert_eq!(estimate_tokens_heuristic(&"a".repeat(400)), 100);
    }

    /// When the LLM endpoint is unreachable, [`count_tokens_or_estimate`]
    /// must surface the heuristic estimate rather than an error. We
    /// deliberately point at a closed loopback port so the request
    /// fails fast.
    #[tokio::test]
    async fn falls_back_to_estimate_on_unreachable_endpoint() {
        let cfg = crate::client::LlmClientConfig::new("http://127.0.0.1:1", "test");
        let Ok(client) = crate::client::LlmClient::new(cfg) else {
            return;
        };
        let text = "hello world";
        let n = count_tokens_or_estimate(&client, text).await;
        assert_eq!(n, estimate_tokens_heuristic(text));
    }
}
