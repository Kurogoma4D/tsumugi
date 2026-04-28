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

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::client::LlmClient;

/// Tracks whether we have already emitted a `warn!` for a tokenize
/// failure. The first failure is loud; subsequent failures drop to
/// `debug!` so the agent loop is not flooded when the LLM endpoint is
/// genuinely down. The latch re-arms the moment a tokenize call
/// succeeds again so the next outage produces a fresh warning instead
/// of being silent forever (issue #50 review feedback).
static TOKENIZE_WARNED: AtomicBool = AtomicBool::new(false);

/// Synchronous observer fired on every tokenize fallback.
///
/// The hook is intentionally lightweight (one call site, four
/// arguments) so the CLI's `--event-log` writer can subscribe without
/// pulling `tmg-core` into the `tmg-llm` dependency graph. Issue #50.
///
/// Arguments: `endpoint`, `text_len`, `estimate`, `error`.
pub type TokenizeFailureHook = Arc<dyn Fn(&str, usize, usize, &str) + Send + Sync + 'static>;

/// Process-wide slot for the [`TokenizeFailureHook`]. The CLI installs
/// one when an `--event-log` path is supplied; tests / library users
/// leave it unset.
static TOKENIZE_FAILURE_HOOK: OnceLock<Mutex<Option<TokenizeFailureHook>>> = OnceLock::new();

/// Install (or replace) the global [`TokenizeFailureHook`].
///
/// Calling with `None` clears any previously-installed hook. The hook
/// is process-wide because the underlying tokenize helper is also
/// process-wide; the CLI is the only caller and it installs the hook
/// once at startup.
pub fn set_tokenize_failure_hook(hook: Option<TokenizeFailureHook>) {
    let slot = TOKENIZE_FAILURE_HOOK.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = slot.lock() {
        *guard = hook;
    }
}

/// Borrow the currently-installed [`TokenizeFailureHook`], if any.
fn current_tokenize_failure_hook() -> Option<TokenizeFailureHook> {
    TOKENIZE_FAILURE_HOOK
        .get()?
        .lock()
        .ok()?
        .as_ref()
        .map(Arc::clone)
}

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
            // Re-arm the warning latch on every recovery so the next
            // outage produces a fresh `warn!` instead of being silent
            // forever (issue #50 review feedback). `Relaxed` is fine
            // — losing a recovery to a racing failure just means the
            // next failure logs at `debug!`, which is acceptable.
            TOKENIZE_WARNED.store(false, Ordering::Relaxed);
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
            // Notify any installed observer (typically the CLI's
            // `--event-log` writer). Hook invocation runs on the same
            // task as the agent loop; installations should be cheap.
            if let Some(hook) = current_tokenize_failure_hook() {
                let err_str = e.to_string();
                hook(client.endpoint(), text.len(), estimate, &err_str);
            }
            estimate
        }
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
#[expect(
    clippy::type_complexity,
    reason = "the captured-tuple type is local to one test and wrapping it in a `type` alias would obscure what each field means in the assertion site"
)]
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

    /// Issue #50 review: a failure path must fire the installed
    /// [`TokenizeFailureHook`] so the CLI's `--event-log` writer can
    /// observe tokenize fallbacks. The hook stores the captured
    /// arguments in a shared `Vec` so the test can assert on them.
    #[tokio::test]
    async fn failure_hook_observes_fallback() {
        // Reset the hook slot before the test in case another test
        // installed one. (Tests run on a single tokio runtime; the
        // process-wide slot is shared.)
        set_tokenize_failure_hook(None);

        let captured: Arc<Mutex<Vec<(String, usize, usize, String)>>> =
            Arc::new(Mutex::new(Vec::new()));
        let captured_for_hook = Arc::clone(&captured);
        set_tokenize_failure_hook(Some(Arc::new(
            move |endpoint: &str, len: usize, estimate: usize, err: &str| {
                if let Ok(mut g) = captured_for_hook.lock() {
                    g.push((endpoint.to_owned(), len, estimate, err.to_owned()));
                }
            },
        )));

        let cfg = crate::client::LlmClientConfig::new("http://127.0.0.1:1", "test");
        let Ok(client) = crate::client::LlmClient::new(cfg) else {
            set_tokenize_failure_hook(None);
            return;
        };
        let _ = count_tokens_or_estimate(&client, "hello world").await;

        let events = captured.lock().expect("hook lock");
        assert!(
            !events.is_empty(),
            "tokenize failure hook must fire on unreachable endpoint",
        );
        let (endpoint, len, estimate, _err) = &events[0];
        assert_eq!(endpoint, "http://127.0.0.1:1");
        assert_eq!(*len, "hello world".len());
        assert_eq!(*estimate, estimate_tokens_heuristic("hello world"));
        drop(events);

        // Cleanup: clear the hook so subsequent tests do not observe
        // stale state.
        set_tokenize_failure_hook(None);
    }
}
