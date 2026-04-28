//! Fallback sandbox implementation for non-Linux platforms.
//!
//! All sandbox operations are no-ops that emit warnings. The agent
//! functions normally but without OS-level security restrictions.
//!
//! Function signatures intentionally match the Linux module so the
//! platform dispatch in `mod.rs` works uniformly.

use std::net::IpAddr;

use crate::config::SandboxConfig;
use crate::error::SandboxError;

/// Outcome of [`apply_network_acl`] on non-Linux platforms.
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
    #[must_use]
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
/// netns. Outside Linux the agent is not in a netns, so there is nothing
/// to do.
#[expect(
    clippy::unnecessary_wraps,
    reason = "signature must match linux::bring_up_loopback for platform dispatch"
)]
pub fn bring_up_loopback() -> Result<(), SandboxError> {
    Ok(())
}

/// No-op network ACL: emits a warning when `allowed_domains` is
/// non-empty and returns an empty [`NetworkAcl`].
///
/// On Linux this would resolve `allowed_domains` to IPs and install
/// iptables rules inside the netns. On other platforms there is no
/// equivalent and we simply skip enforcement so the agent can keep
/// running with full network access.
///
/// # Errors
///
/// This implementation never returns an error. The signature returns
/// `Result` only to match the Linux variant.
pub async fn apply_network_acl(allowed_domains: &[String]) -> Result<NetworkAcl, SandboxError> {
    // Yield once so the function is genuinely async and matches the
    // Linux signature without tripping the `unused_async` lint.
    tokio::task::yield_now().await;

    if !allowed_domains.is_empty() {
        emit_warning(
            "Network ACL enforcement is not available on this platform; \
             allowed_domains is recorded but not enforced",
        );
    }
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
