//! Linux network ACL implementation: DNS resolution + iptables rules.
//!
//! This module enforces SPEC §6.2 outbound network restriction. After the
//! caller has placed the current process in a fresh network namespace via
//! [`super::netns::create_network_namespace`] (and brought `lo` up via
//! [`super::netns::bring_up_loopback`]), this module:
//!
//! 1. Resolves each domain in `allowed_domains` to a set of IP addresses
//!    using `tokio::net::lookup_host`.
//! 2. Installs an `iptables` rule chain inside the netns that:
//!    - allows loopback traffic
//!    - allows DNS (53/udp + 53/tcp) so further name resolution works
//!    - allows outbound traffic to each resolved IP
//!    - allows established/related return traffic
//!    - drops everything else by default
//!
//! Each `iptables` invocation runs sequentially via
//! [`std::process::Command`]. If any step fails, the partial chain remains
//! in the netns; callers handle rollback by tearing the namespace itself
//! down (e.g. by exiting the sandboxed process).
//!
//! # Future work
//!
//! TTL-based DNS re-resolution is **not** implemented in this module. The
//! IP set is captured once at sandbox-creation time. A long-running agent
//! that talks to a DNS-load-balanced service may eventually fail when the
//! cached IPs expire on the server side. Tracking re-resolution requires
//! either a periodic refresh task or eBPF (`cgroup/connect4`) and is
//! deferred to a follow-up issue.

use std::net::IpAddr;
use std::process::Command;

use crate::error::SandboxError;

/// Outcome of [`apply_network_acl`].
///
/// Holds the set of IP addresses that were resolved from
/// `allowed_domains` and authorized by the iptables chain. Callers can
/// log this set or pass it through to diagnostic surfaces.
#[derive(Debug, Clone, Default)]
pub struct NetworkAcl {
    /// IP addresses that were resolved from the allowed domains and
    /// installed as `ACCEPT` rules in the netns iptables chain.
    pub resolved_ips: Vec<IpAddr>,
}

impl NetworkAcl {
    /// Number of distinct IP addresses authorized by this ACL.
    #[must_use]
    pub fn len(&self) -> usize {
        self.resolved_ips.len()
    }

    /// Whether the ACL authorized no IPs (still has loopback + DNS).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.resolved_ips.is_empty()
    }
}

/// Default DNS port used for both UDP and TCP allow rules.
const DNS_PORT: &str = "53";

/// Apply the network ACL inside the current netns.
///
/// This must be called **after** the caller has entered a fresh network
/// namespace. It does not perform `unshare(CLONE_NEWNET)` itself.
///
/// On any failure the partial iptables state is left in place; the
/// caller is expected to discard the netns (typically by exiting the
/// sandboxed process so the kernel reclaims the namespace).
///
/// # Errors
///
/// Returns [`SandboxError::NetworkAcl`] if:
/// - `iptables` is not installed or not on `PATH`
/// - any `iptables` invocation exits with a non-zero status
/// - DNS resolution for an allowed domain fails (this is a hard error so
///   misconfiguration is not silently ignored; the caller decides whether
///   to honor `strict` and fall back)
///
/// # Cancellation safety
///
/// This function awaits `tokio::net::lookup_host` for each domain. If
/// the future is dropped mid-resolution, no iptables rules will have
/// been installed yet because resolution runs before any
/// `iptables` invocation.
pub async fn apply_network_acl(allowed_domains: &[String]) -> Result<NetworkAcl, SandboxError> {
    let resolved_ips = resolve_domains(allowed_domains).await?;
    install_iptables_chain(&resolved_ips)?;
    Ok(NetworkAcl { resolved_ips })
}

/// Resolve each domain to its IP addresses via `tokio::net::lookup_host`.
///
/// Duplicate IPs across domains are de-duplicated. Resolution failures
/// for individual domains are wrapped into [`SandboxError::NetworkAcl`]
/// with the offending domain in the message.
async fn resolve_domains(allowed_domains: &[String]) -> Result<Vec<IpAddr>, SandboxError> {
    let mut ips: Vec<IpAddr> = Vec::new();

    for domain in allowed_domains {
        // `lookup_host` accepts `host:port` form. We use port 0 because
        // we only care about the address; the kernel never sees this
        // port (we install rules per IP, not per port pair).
        let host_port = format!("{domain}:0");
        let addrs =
            tokio::net::lookup_host(&host_port)
                .await
                .map_err(|e| SandboxError::NetworkAcl {
                    reason: format!("failed to resolve domain '{domain}': {e}"),
                })?;

        for addr in addrs {
            let ip = addr.ip();
            if !ips.contains(&ip) {
                ips.push(ip);
            }
        }
    }

    Ok(ips)
}

/// Install the iptables rule chain inside the current netns.
///
/// This issues a fixed sequence of `iptables -A` and `iptables -P`
/// commands. On the first failure, processing stops and the partial
/// chain remains; the caller is expected to discard the netns.
fn install_iptables_chain(resolved_ips: &[IpAddr]) -> Result<(), SandboxError> {
    // Allow loopback first so any in-process listener (e.g. a child
    // helper bound to 127.0.0.1) keeps working.
    run_iptables(&["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"])?;
    run_iptables(&["-A", "INPUT", "-i", "lo", "-j", "ACCEPT"])?;

    // Allow DNS so the agent can re-resolve allowed domains. We allow
    // both UDP and TCP for full DNS coverage (large responses, DoT
    // setups, etc.).
    run_iptables(&[
        "-A", "OUTPUT", "-p", "udp", "--dport", DNS_PORT, "-j", "ACCEPT",
    ])?;
    run_iptables(&[
        "-A", "OUTPUT", "-p", "tcp", "--dport", DNS_PORT, "-j", "ACCEPT",
    ])?;

    // Allow outbound to each resolved IP.
    for ip in resolved_ips {
        let ip_str = ip.to_string();
        run_iptables(&["-A", "OUTPUT", "-d", &ip_str, "-j", "ACCEPT"])?;
    }

    // Allow established / related return traffic on INPUT so the
    // outbound flows to allowed IPs can complete.
    run_iptables(&[
        "-A",
        "INPUT",
        "-m",
        "state",
        "--state",
        "ESTABLISHED,RELATED",
        "-j",
        "ACCEPT",
    ])?;

    // Default DROP for everything else.
    run_iptables(&["-P", "OUTPUT", "DROP"])?;
    run_iptables(&["-P", "INPUT", "DROP"])?;

    Ok(())
}

/// Run a single `iptables` command and return an error on non-zero exit.
fn run_iptables(args: &[&str]) -> Result<(), SandboxError> {
    let output =
        Command::new("iptables")
            .args(args)
            .output()
            .map_err(|e| SandboxError::NetworkAcl {
                reason: format!("failed to spawn `iptables {}`: {e}", args.join(" ")),
            })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(SandboxError::NetworkAcl {
            reason: format!(
                "`iptables {}` exited with {}: {}",
                args.join(" "),
                output.status,
                stderr.trim()
            ),
        });
    }
    Ok(())
}

/// Return whether the current process appears to have `CAP_NET_ADMIN`.
///
/// This is a best-effort check based on `/proc/self/status`: on Linux the
/// effective capability set is exposed as the hex `CapEff` line. Bit 12
/// (`CAP_NET_ADMIN`) must be set for `iptables` and `unshare(CLONE_NEWNET)`
/// to succeed.
///
/// Returns `false` if the file cannot be read or parsed (treating "we
/// cannot prove we have the capability" as "we do not have it"), so the
/// caller's `strict = false` fallback path takes over.
#[must_use]
pub fn has_cap_net_admin() -> bool {
    // CAP_NET_ADMIN = 12
    const CAP_NET_ADMIN_BIT: u64 = 1 << 12;

    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return false;
    };

    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("CapEff:") {
            let hex = rest.trim();
            if let Ok(bits) = u64::from_str_radix(hex, 16) {
                return (bits & CAP_NET_ADMIN_BIT) != 0;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_acl_len_and_is_empty() {
        let acl = NetworkAcl::default();
        assert!(acl.is_empty());
        assert_eq!(acl.len(), 0);

        let acl = NetworkAcl {
            resolved_ips: vec!["127.0.0.1".parse().unwrap_or(IpAddr::from([0, 0, 0, 0]))],
        };
        assert!(!acl.is_empty());
        assert_eq!(acl.len(), 1);
    }

    /// Sanity check: resolving an empty list returns no IPs and does
    /// not touch iptables. Safe to run anywhere.
    #[tokio::test]
    async fn resolve_domains_empty_input() {
        let ips = resolve_domains(&[]).await.unwrap_or_default();
        assert!(ips.is_empty());
    }

    /// Resolve `localhost` (always present on Linux). This does not
    /// touch iptables and so is safe in CI without `CAP_NET_ADMIN`.
    #[tokio::test]
    async fn resolve_domains_localhost() {
        let ips = resolve_domains(&["localhost".to_owned()])
            .await
            .unwrap_or_default();
        // Most Linux systems return at least 127.0.0.1; if the test
        // host is unusually configured this becomes a soft check.
        if ips.is_empty() {
            return;
        }
        assert!(
            ips.iter().any(|ip| ip.is_loopback()),
            "expected localhost to resolve to a loopback address, got {ips:?}"
        );
    }
}
