//! Shared CLI helpers for `[llm.subagent_pool]` wiring.
//!
//! Moved out of `main.rs` so the workflow command path
//! (`tmg workflow run`) can reuse the same construction logic instead
//! of building an [`tmg_agents::EndpointResolver`] without a pool —
//! that mismatch silently dropped the operator's pool config (issue
//! #50 review).
//!
//! In addition to pool construction, this module also installs the
//! `--event-log` observers (`endpoint_resolved`, `pool_selected`,
//! `tokenize_failure`) on the [`tmg_agents::SubagentManager`]
//! (and via [`tmg_llm::set_tokenize_failure_hook`] on the global
//! tokenize fallback path). Centralising the wiring keeps both the
//! TUI and CLI paths in lock-step so a future code change cannot
//! forget to install one of the hooks on one of the entry points.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context as _;
use tokio::sync::Mutex as AsyncMutex;

use tmg_agents::{EndpointResolvedEvent, EndpointResolvedHook, SubagentManager};
use tmg_core::EventLogWriter;
use tmg_llm::{LlmPool, PoolConfig, TokenizeFailureHook};

/// Build the [`LlmPool`] (if any) for the subagent resolver.
///
/// Returns `None` when no `[llm.subagent_pool]` section was supplied
/// or when the operator left `endpoints = []` (the relaxed validator
/// already logged the "pool disabled" notice). Construction errors are
/// downgraded to a `tracing::warn!` so a malformed pool does not abort
/// startup — the resolver still has a valid main fallback.
pub(crate) fn subagent_pool_from_config(
    pool_cfg: Option<&PoolConfig>,
    model: &str,
) -> Option<Arc<LlmPool>> {
    let cfg = pool_cfg?;
    let report = match cfg.validate_relaxed() {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(error = %e, "subagent pool config invalid; not constructing pool");
            return None;
        }
    };
    if report.disabled {
        return None;
    }
    let deduped = PoolConfig {
        endpoints: report.deduped_endpoints,
        strategy: cfg.strategy,
    };
    match LlmPool::new(&deduped, model) {
        Ok(pool) => Some(Arc::new(pool)),
        Err(e) => {
            tracing::warn!(error = %e, "subagent pool construction failed; falling back to main endpoint");
            None
        }
    }
}

/// Open an [`EventLogWriter`] in append mode.
///
/// `append` is preferred over `new` here because both the TUI startup
/// path and `tmg workflow run` may write into the same `--event-log`
/// path during a single shell session, and truncating between
/// invocations would lose the prior run's events.
///
/// Returned errors carry context describing the path so the operator
/// sees what file actually failed to open.
pub(crate) fn open_event_log(path: &Path) -> anyhow::Result<Arc<std::sync::Mutex<EventLogWriter>>> {
    let writer = EventLogWriter::open_append(path)
        .with_context(|| format!("opening event log at {}", path.display()))?;
    Ok(Arc::new(std::sync::Mutex::new(writer)))
}

/// Install the `endpoint_resolved` and `pool_selected` hooks on the
/// supplied [`SubagentManager`].
///
/// `pool_selected` events are emitted from inside the
/// `endpoint_resolved` hook when the resolver reports `source ==
/// "pool"` — that's the one branch where a multi-endpoint pool
/// actually picked a URL, so emitting both records lets the
/// `--event-log` reader correlate the spawn with the pool selection
/// without a separate observer plumb. The `strategy` string is
/// stamped from the pool's configured [`tmg_llm::LoadBalanceStrategy`]
/// at install time, which is stable for the lifetime of the manager.
pub(crate) async fn install_event_log_hooks(
    manager: Arc<AsyncMutex<SubagentManager>>,
    log: Arc<std::sync::Mutex<EventLogWriter>>,
    pool_strategy: Option<&'static str>,
) {
    let log_for_resolved = Arc::clone(&log);
    let strategy_label: &'static str = pool_strategy.unwrap_or("");
    let hook: EndpointResolvedHook = Arc::new(move |ev: &EndpointResolvedEvent<'_>| {
        if let Ok(mut writer) = log_for_resolved.lock() {
            writer.write_endpoint_resolved(ev.agent_kind, ev.endpoint, ev.model, ev.source);
            // `pool_selected` is the structurally-richer record for the
            // pool case; emit it whenever the resolver reports `pool`.
            if ev.source == "pool" {
                writer.write_pool_selected(ev.endpoint, strategy_label);
            }
        }
    });
    manager.lock().await.set_endpoint_resolved_hook(Some(hook));
}

/// Install the global tokenize-failure observer.
///
/// The hook is process-wide (see [`tmg_llm::set_tokenize_failure_hook`])
/// because the underlying tokenize helper is also process-wide. The
/// CLI is the only caller and installs the hook once at startup; the
/// returned guard intentionally does not exist because a tokenize
/// failure that fires after the CLI exits is harmless.
pub(crate) fn install_tokenize_failure_hook(log: Arc<std::sync::Mutex<EventLogWriter>>) {
    let hook: TokenizeFailureHook = Arc::new(
        move |endpoint: &str, text_len: usize, estimate: usize, error: &str| {
            if let Ok(mut writer) = log.lock() {
                writer.write_tokenize_failure(endpoint, text_len, estimate, error);
            }
        },
    );
    tmg_llm::set_tokenize_failure_hook(Some(hook));
}
