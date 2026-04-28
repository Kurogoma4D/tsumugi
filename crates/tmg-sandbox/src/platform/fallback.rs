//! Fallback sandbox implementation for non-Linux platforms.
//!
//! All sandbox operations are no-ops that emit warnings. The agent
//! functions normally but without OS-level security restrictions.
//!
//! Function signatures intentionally match the Linux module so the
//! platform dispatch in `mod.rs` works uniformly. In particular the
//! Linux-side network ACL is split into a pre-netns DNS resolution
//! step ([`resolve_domains`]) and a post-netns iptables installation
//! step ([`install_iptables_chain`]); the same split is exposed here as
//! no-ops so callers can use a single code path on every platform.

use std::net::IpAddr;

use crate::config::SandboxConfig;
use crate::error::SandboxError;

/// Outcome of [`install_iptables_chain`] on non-Linux platforms.
///
/// Mirrors [`crate::platform::NetworkAcl`] on Linux so callers can
/// handle the result type uniformly.
#[derive(Debug, Clone, Default)]
pub struct NetworkAcl {
    /// Always empty on non-Linux: no rules are installed.
    pub resolved_ips: Vec<IpAddr>,
}

impl NetworkAcl {
    /// Number of distinct IP addresses authorized by this ACL.
    pub fn len(&self) -> usize {
        self.resolved_ips.len()
    }

    /// Always `true` on non-Linux.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.resolved_ips.is_empty()
    }
}

/// Emit a warning that Landlock is not available on this platform.
///
/// Returns `Ok(())` -- the caller should proceed without restrictions.
#[expect(
    clippy::unnecessary_wraps,
    reason = "signature must match linux::apply_landlock for platform dispatch"
)]
pub fn apply_landlock(config: &SandboxConfig) -> Result<(), SandboxError> {
    if config.mode.is_restricted() {
        emit_warning(
            "Landlock filesystem sandbox is not available on this platform; running without filesystem restrictions",
        );
    }
    Ok(())
}

/// Emit a warning that network namespaces are not available on this platform.
///
/// Returns `Ok(())` -- the caller should proceed without network restrictions.
#[expect(
    clippy::unnecessary_wraps,
    reason = "signature must match linux::create_network_namespace for platform dispatch"
)]
pub fn create_network_namespace() -> Result<(), SandboxError> {
    emit_warning(
        "Network namespace isolation is not available on this platform; \
         running without network restrictions",
    );
    Ok(())
}

/// No-op loopback bring-up for non-Linux platforms.
///
/// On Linux this would run `ip link set lo up` inside the freshly-created
/// netns via `tokio::process::Command`. Outside Linux the agent is not
/// in a netns, so there is nothing to do.
pub async fn bring_up_loopback() -> Result<(), SandboxError> {
    // Yield once so the function is genuinely async and matches the
    // Linux signature without tripping the `unused_async` lint.
    tokio::task::yield_now().await;
    Ok(())
}

/// No-op DNS resolution: always returns an empty IP list, regardless
/// of `allowed_domains`.
///
/// On Linux this would resolve each domain via `tokio::net::lookup_host`
/// **before** entering the netns. On other platforms there is no netns,
/// no `iptables`, and nothing to enforce, so we skip resolution
/// entirely and let the caller proceed.
///
/// # Errors
///
/// This implementation never returns an error. The signature returns
/// `Result` only to match the Linux variant.
pub async fn resolve_domains(allowed_domains: &[String]) -> Result<Vec<IpAddr>, SandboxError> {
    // Yield once so the function is genuinely async and matches the
    // Linux signature without tripping the `unused_async` lint.
    tokio::task::yield_now().await;

    if !allowed_domains.is_empty() {
        emit_warning(
            "Network ACL enforcement is not available on this platform; \
             allowed_domains is recorded but not enforced",
        );
    }
    Ok(Vec::new())
}

/// No-op iptables installation: returns an empty [`NetworkAcl`] without
/// touching any kernel state.
///
/// # Errors
///
/// This implementation never returns an error. The signature returns
/// `Result` only to match the Linux variant.
pub async fn install_iptables_chain(_resolved_ips: &[IpAddr]) -> Result<NetworkAcl, SandboxError> {
    // Yield once so the function is genuinely async and matches the
    // Linux signature without tripping the `unused_async` lint.
    tokio::task::yield_now().await;
    Ok(NetworkAcl::default())
}

/// Always returns `false` on non-Linux: there is no kernel-level
/// `CAP_NET_ADMIN` concept to query.
#[must_use]
pub fn has_cap_net_admin() -> bool {
    false
}

/// No-op: OOM score adjustment is Linux-only.
///
/// Returns `Ok(())`.
#[expect(
    clippy::unnecessary_wraps,
    reason = "signature must match linux::adjust_oom_score for platform dispatch"
)]
pub fn adjust_oom_score(_pid: u32, _score: i32) -> Result<(), SandboxError> {
    // OOM score adjustment is a Linux-specific feature. Silently skip
    // on other platforms as it is a best-effort optimization.
    Ok(())
}

/// Emit a warning message via `eprintln`.
///
/// On non-Linux platforms this is the primary way to inform the user
/// that sandbox features are unavailable.
#[expect(
    clippy::print_stderr,
    reason = "intentional warning output for unsupported platform"
)]
fn emit_warning(msg: &str) {
    eprintln!("[tmg-sandbox] WARNING: {msg}");
}
