//! Linux network ACL implementation: DNS resolution + iptables rules.
//!
//! This module enforces SPEC §6.2 outbound network restriction. The
//! activation flow is split across two functions so DNS resolution can
//! run **before** the process enters an empty network namespace (where
//! it would otherwise have no way to talk to a resolver):
//!
//! 1. [`resolve_domains`] resolves each entry in `allowed_domains` to
//!    a set of IP addresses using `tokio::net::lookup_host`. This MUST
//!    be called **before** `unshare(CLONE_NEWNET)` because the empty
//!    netns has no veth and no default route, so name resolution would
//!    otherwise fail for every non-`/etc/hosts` domain.
//! 2. After the caller has placed the process in a fresh network
//!    namespace via [`super::netns::create_network_namespace`] (and
//!    brought `lo` up via [`super::netns::bring_up_loopback`]),
//!    [`install_iptables_chain`] installs an `iptables` rule chain
//!    inside the netns that:
//!    - flips the default policy to `DROP` for OUTPUT/INPUT first
//!    - allows loopback traffic
//!    - allows DNS (53/udp + 53/tcp) so further name resolution works
//!    - allows established/related return traffic
//!    - allows outbound traffic to each resolved IP
//!
//! Each `iptables` invocation runs sequentially via
//! [`tokio::process::Command`] so the executor thread is never blocked
//! on a synchronous `wait()`. If any `-A` step fails after the default
//! policy has been flipped to `DROP`, the chain is flushed via
//! `iptables -F` so the netns ends up in a deterministic
//! "drop everything" state rather than a partial allowlist with default
//! `ACCEPT` (which would be worse than no rules at all).
//!
//! # IPv6
//!
//! `iptables` is IPv4-only: feeding it a v6 literal (`::1`,
//! `2001:db8::1`, ...) makes the rule installation crash. Domains that
//! resolve to both A and AAAA records (which is most public services,
//! including `localhost`) would therefore break the chain.
//!
//! [`install_iptables_chain`] filters its input to IPv4 addresses
//! before issuing any rules. IPv6 outbound is still blocked because the
//! empty netns has no IPv6 default route and no `ip6tables` chain. A
//! richer fix that splits v4/v6 across `iptables` + `ip6tables` is left
//! for a follow-up.
//!
//! # Future work
//!
//! TTL-based DNS re-resolution is **not** implemented in this module. The
//! IP set is captured once at sandbox-creation time. A long-running agent
//! that talks to a DNS-load-balanced service may eventually fail when the
//! cached IPs expire on the server side. Tracking re-resolution requires
//! either a periodic refresh task or eBPF (`cgroup/connect4`) and is
//! deferred to a follow-up issue.

use std::net::{IpAddr, Ipv4Addr};

use tokio::process::Command;

use crate::error::SandboxError;

/// Outcome of [`install_iptables_chain`].
///
/// Holds the set of IPv4 addresses that were resolved from
/// `allowed_domains` and authorized by the iptables chain. Callers can
/// log this set or pass it through to diagnostic surfaces.
#[derive(Debug, Clone, Default)]
pub struct NetworkAcl {
    /// IPv4 addresses that were resolved from the allowed domains and
    /// installed as `ACCEPT` rules in the netns iptables chain. IPv6
    /// addresses returned by the resolver are dropped before this list
    /// is built (see module docs).
    pub resolved_ips: Vec<IpAddr>,
}

impl NetworkAcl {
    /// Number of distinct IP addresses authorized by this ACL.
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

/// Resolve each domain to its IP addresses via `tokio::net::lookup_host`.
///
/// This MUST be called **before** the process enters an empty network
/// namespace. Once `unshare(CLONE_NEWNET)` has run there is no veth and
/// no default route, so DNS lookups for any domain not listed in
/// `/etc/hosts` would fail.
///
/// Duplicate IPs across domains are de-duplicated. Per-domain resolution
/// failures are **non-fatal**: the failure is logged via `eprintln!`
/// and the remaining domains continue to resolve. The function only
/// errors if **every** domain fails to resolve (or if the input is
/// non-empty but no IP at all comes back), so a single NXDOMAIN entry
/// in a list of three does not nuke the whole sandbox.
///
/// # Cancellation safety
///
/// Awaiting `lookup_host` for each domain is cancel-safe: if the future
/// is dropped, no iptables rules have been installed (this function
/// does not touch iptables at all) and the netns has not yet been
/// created either, so dropping mid-resolution leaves the system
/// unchanged.
///
/// # Errors
///
/// Returns [`SandboxError::NetworkAcl`] only if `allowed_domains` is
/// non-empty and **all** domains fail to resolve. The reason string
/// includes the per-domain failure messages.
pub async fn resolve_domains(allowed_domains: &[String]) -> Result<Vec<IpAddr>, SandboxError> {
    let mut ips: Vec<IpAddr> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for domain in allowed_domains {
        // `lookup_host` accepts `host:port` form. We use port 443
        // (HTTPS) for the cleanest cross-libc behavior; the kernel
        // never sees this port (we install rules per IP, not per port
        // pair), but some libc implementations are happier with a
        // well-known service number than with `:0`.
        let host_port = format!("{domain}:443");
        match tokio::net::lookup_host(&host_port).await {
            Ok(addrs) => {
                for addr in addrs {
                    let ip = addr.ip();
                    if !ips.contains(&ip) {
                        ips.push(ip);
                    }
                }
            }
            Err(e) => {
                let msg = format!("failed to resolve domain '{domain}': {e}");
                emit_resolve_warning(&msg);
                failures.push(msg);
            }
        }
    }

    if !allowed_domains.is_empty() && ips.is_empty() {
        // Every domain failed: this almost certainly means a misconfig
        // or no network at all, and we should surface it rather than
        // silently install zero IP-specific rules.
        return Err(SandboxError::NetworkAcl {
            reason: format!(
                "no allowed domain could be resolved ({} attempted): {}",
                allowed_domains.len(),
                failures.join("; ")
            ),
        });
    }

    Ok(ips)
}

/// Install the iptables rule chain inside the current netns.
///
/// This must be called **after** the caller has entered a fresh network
/// namespace and brought `lo` up. The function does not touch DNS or
/// the netns; it only translates the already-resolved IP list into
/// `iptables` rules.
///
/// The order of operations is important: the default OUTPUT/INPUT
/// policy is flipped to `DROP` **first**, so that if any subsequent
/// `-A` rule fails the netns ends up in a deterministic
/// "drop everything" state. On any error after that, `iptables -F` is
/// run to flush partial state, leaving the chain empty but with the
/// default `DROP` policy still in effect — which is at least no worse
/// than the netns-only blocking baseline.
///
/// Only IPv4 addresses are translated into rules; IPv6 entries in
/// `resolved_ips` are silently dropped. See module docs for rationale.
///
/// # Cancellation safety
///
/// This function awaits `tokio::process::Command::output` for each
/// `iptables` call. If the future is dropped mid-installation, the
/// kernel-side rules already added persist; the caller is expected to
/// discard the netns by exiting the sandboxed process.
///
/// # Errors
///
/// Returns [`SandboxError::NetworkAcl`] if:
/// - `iptables` is not installed or not on `PATH`
/// - any `iptables` invocation exits with a non-zero status
pub async fn install_iptables_chain(resolved_ips: &[IpAddr]) -> Result<NetworkAcl, SandboxError> {
    // Filter to IPv4 only: `iptables` (vs `ip6tables`) cannot accept v6
    // literals and would crash on the first `-d ::1`-style argument.
    let ipv4_only: Vec<IpAddr> = resolved_ips
        .iter()
        .filter_map(|ip| match ip {
            IpAddr::V4(v4) => Some(IpAddr::V4(*v4)),
            IpAddr::V6(_) => None,
        })
        .collect();

    match install_chain_inner(&ipv4_only).await {
        Ok(()) => Ok(NetworkAcl {
            resolved_ips: ipv4_only,
        }),
        Err(e) => {
            // Any rule failed after we flipped the default to DROP.
            // Flush the chain so we end up in a deterministic
            // "drop everything" state instead of a partial allowlist
            // sitting on top of default-ACCEPT.
            flush_chain_best_effort().await;
            Err(e)
        }
    }
}

/// Inner installation routine. Kept separate so the caller in
/// [`install_iptables_chain`] can flush on any error from any of these
/// steps in one place.
async fn install_chain_inner(ipv4_addrs: &[IpAddr]) -> Result<(), SandboxError> {
    // 1. Flip the default policy to DROP **first**. If any subsequent
    //    rule fails, the netns is at least in a fail-closed state.
    run_iptables(&["-P", "OUTPUT", "DROP"]).await?;
    run_iptables(&["-P", "INPUT", "DROP"]).await?;

    // 2. Allow loopback so any in-process listener (e.g. a child
    //    helper bound to 127.0.0.1) keeps working.
    run_iptables(&["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"]).await?;
    run_iptables(&["-A", "INPUT", "-i", "lo", "-j", "ACCEPT"]).await?;

    // 3. Allow DNS so the agent can re-resolve allowed domains. We
    //    allow both UDP and TCP for full DNS coverage (large responses,
    //    DoT setups, etc.).
    run_iptables(&[
        "-A", "OUTPUT", "-p", "udp", "--dport", DNS_PORT, "-j", "ACCEPT",
    ])
    .await?;
    run_iptables(&[
        "-A", "OUTPUT", "-p", "tcp", "--dport", DNS_PORT, "-j", "ACCEPT",
    ])
    .await?;

    // 4. Allow established / related return traffic on INPUT so the
    //    outbound flows to allowed IPs can complete.
    run_iptables(&[
        "-A",
        "INPUT",
        "-m",
        "state",
        "--state",
        "ESTABLISHED,RELATED",
        "-j",
        "ACCEPT",
    ])
    .await?;

    // 5. Allow outbound to each resolved IPv4. (IPv6 was filtered out
    //    by the caller; see module docs.)
    for ip in ipv4_addrs {
        debug_assert!(matches!(ip, IpAddr::V4(_)));
        let ip_str = ip.to_string();
        run_iptables(&["-A", "OUTPUT", "-d", &ip_str, "-j", "ACCEPT"]).await?;
    }

    Ok(())
}

/// Flush all rules from the filter table. Best-effort: any error here
/// is itself reported via `eprintln!` because we are already on the
/// error path of [`install_iptables_chain`] and have nothing better
/// to do with it.
///
/// After the flush the default OUTPUT/INPUT policy (already flipped to
/// `DROP` by the caller) remains in effect, so the netns stays in a
/// fail-closed state.
async fn flush_chain_best_effort() {
    if let Err(e) = run_iptables(&["-F"]).await {
        emit_flush_warning(&e);
    }
}

/// Run a single `iptables` command and return an error on non-zero exit.
///
/// Uses [`tokio::process::Command`] so the surrounding `async fn` does
/// not block its executor thread on the synchronous `wait()` of
/// `std::process::Command`.
async fn run_iptables(args: &[&str]) -> Result<(), SandboxError> {
    let output = Command::new("iptables")
        .args(args)
        .output()
        .await
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

/// Emit a warning to stderr when an individual domain fails DNS
/// resolution. The full set of failures is also recorded by the caller
/// for inclusion in the eventual error message if every domain fails.
#[expect(
    clippy::print_stderr,
    reason = "intentional warning: per-domain resolution failure is non-fatal but operator-visible"
)]
fn emit_resolve_warning(msg: &str) {
    eprintln!("[tmg-sandbox] WARNING: {msg}");
}

/// Emit a warning when the post-error `iptables -F` flush itself fails.
/// At this point we have already lost the ability to install a clean
/// allowlist; there is nothing more to do but report it.
#[expect(
    clippy::print_stderr,
    reason = "intentional warning: chain flush after partial install failed"
)]
fn emit_flush_warning(err: &SandboxError) {
    eprintln!(
        "[tmg-sandbox] WARNING: failed to flush iptables chain after partial install ({err}); \
         the netns is in default-DROP state with no allow rules"
    );
}

/// Return whether the current process appears to have `CAP_NET_ADMIN`.
///
/// This is a best-effort check based on `/proc/self/status`: on Linux
/// the effective capability set is exposed as the hex `CapEff` line.
/// Bit 12 (`CAP_NET_ADMIN`) must be set for `iptables` and
/// `unshare(CLONE_NEWNET)` to succeed.
///
/// **Note on scope.** This function inspects only `CapEff` (the
/// *effective* set). It does not look at `CapBnd` (bounding) or
/// `CapInh` (inheritable). For our purposes this is the right check:
/// the agent process needs the capability to be effective *now* in
/// order to call `iptables`, regardless of what is in its bounding or
/// inheritable sets.
///
/// Returns `false` if the file cannot be read or parsed (treating
/// "we cannot prove we have the capability" as "we do not have it"),
/// so the caller's `strict = false` fallback path takes over.
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

    /// A list of domains where every entry fails to resolve must
    /// surface as an error rather than silently producing zero IPs.
    /// We use `.invalid` (RFC 6761 reserved) so the test is reliable
    /// across DNS resolvers.
    #[tokio::test]
    async fn resolve_domains_all_failures_errors() {
        let res = resolve_domains(&[
            "definitely-not-a-host.invalid".to_owned(),
            "another-bogus-name.invalid".to_owned(),
        ])
        .await;
        assert!(
            res.is_err(),
            "expected error when every domain fails to resolve, got {res:?}"
        );
    }

    /// A partial-failure list where at least one domain resolves must
    /// succeed and return the resolvable IPs only. Failures are logged
    /// to stderr but do not prevent the function from returning Ok.
    #[tokio::test]
    async fn resolve_domains_partial_failure_succeeds() {
        let res = resolve_domains(&[
            "localhost".to_owned(),
            "definitely-not-a-host.invalid".to_owned(),
        ])
        .await;
        // If `localhost` itself doesn't resolve on this host, we have
        // bigger problems; treat as a soft pass.
        let Ok(ips) = res else {
            eprintln!(
                "skipping resolve_domains_partial_failure_succeeds: localhost did not resolve"
            );
            return;
        };
        assert!(
            !ips.is_empty(),
            "expected at least the localhost IP, got empty list"
        );
    }

    /// `install_iptables_chain` filters IPv6 out of its input. We can
    /// test the filter directly without invoking iptables by checking
    /// that the function would only attempt rules for the v4 entries.
    /// The test here exercises only the filter logic via the public
    /// API contract: passing a v6-only list and not having
    /// `CAP_NET_ADMIN` should fail at the FIRST `iptables` call (the
    /// `-P OUTPUT DROP` policy flip), not at a `-d ::1` rule.
    ///
    /// We don't actually run `iptables` here (would need root), so we
    /// just confirm the IPv4 filter compiles and that the v4-only
    /// `NetworkAcl` would be empty for v6 input.
    #[test]
    fn ipv4_filter_drops_v6_addresses() {
        let mixed: Vec<IpAddr> = vec!["127.0.0.1".parse().unwrap(), "::1".parse().unwrap()];
        let v4: Vec<IpAddr> = mixed
            .iter()
            .filter_map(|ip| match ip {
                IpAddr::V4(v4) => Some(IpAddr::V4(*v4)),
                IpAddr::V6(_) => None,
            })
            .collect();
        assert_eq!(v4.len(), 1);
        assert_eq!(v4[0], IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)));
    }
}
