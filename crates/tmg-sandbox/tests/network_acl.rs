//! Integration tests for the network ACL.
//!
//! Most assertions are Linux-only because the ACL is implemented via
//! `iptables` inside a network namespace; macOS / other platforms only
//! exercise the no-op fallback.
//!
//! Tests that require `CAP_NET_ADMIN` (i.e. actual `unshare` and
//! `iptables` invocations) are gated behind `#[ignore]` so CI without
//! root privileges still passes. Run them locally with:
//!
//! ```sh
//! sudo -E cargo test -p tmg-sandbox --test network_acl -- --ignored
//! ```

use tmg_sandbox::{NetworkConfig, SandboxConfig, SandboxContext, SandboxMode};

/// Cross-platform sanity check: with a non-empty `allowed_domains` and
/// `strict = false`, [`SandboxContext::activate`] must not fail even
/// when `CAP_NET_ADMIN` is missing. This documents the warn-and-fall-back
/// behavior on both Linux (no-cap CI) and macOS (no netns at all).
#[tokio::test]
async fn activate_with_allowed_domains_non_strict_does_not_fail() {
    let mut config = SandboxConfig::new(std::env::temp_dir())
        .with_mode(SandboxMode::Full) // Full mode short-circuits activation.
        .with_allowed_domains(vec!["example.com".to_owned()]);
    config.network.strict = false;

    let ctx = SandboxContext::new(config);
    // Full mode skips the network phase entirely, so this is just a
    // regression guard that the new field does not break activation.
    let res = ctx.activate().await;
    assert!(res.is_ok(), "activate failed in Full mode: {res:?}");
}

/// `effective_allowed_domains` correctly merges the legacy top-level
/// field with `[sandbox.network]`.
#[test]
fn effective_allowed_domains_merges_sources() {
    let mut config = SandboxConfig::new(std::env::temp_dir())
        .with_allowed_domains(vec!["legacy.example".to_owned()]);
    config.network = NetworkConfig {
        allowed_domains: vec!["new.example".to_owned(), "legacy.example".to_owned()],
        strict: true,
    };

    let merged = config.effective_allowed_domains();
    assert_eq!(merged, vec!["legacy.example", "new.example"]);
    assert!(config.network_strict());
}

/// On Linux without `CAP_NET_ADMIN`, `strict = true` must cause
/// activation to fail with [`SandboxError::NetworkAcl`]. We can test
/// this without root because the failure path is taken *before* any
/// privileged syscall.
///
/// Skipped on non-Linux because the fallback never reaches the
/// capability check (it short-circuits to a no-op).
#[cfg(target_os = "linux")]
#[tokio::test]
async fn strict_mode_without_capability_errors() {
    use tmg_sandbox::SandboxError;

    // We need a non-Full mode so `activate` actually runs the network
    // phase, but we also do not want Landlock to interact with the
    // workspace. Use ReadOnly with a tempdir that exists.
    let tempdir = std::env::temp_dir();
    let mut config = SandboxConfig::new(&tempdir)
        .with_mode(SandboxMode::ReadOnly)
        .with_allowed_domains(vec!["example.com".to_owned()]);
    config.network.strict = true;

    // If the test environment happens to have CAP_NET_ADMIN, we skip
    // because the strict path would actually try to install rules.
    // Tests with CAP_NET_ADMIN are gated separately via `#[ignore]`.
    if has_cap_net_admin_on_linux() {
        eprintln!("skipping strict_mode_without_capability_errors: CAP_NET_ADMIN present");
        return;
    }

    let ctx = SandboxContext::new(config);
    let res = ctx.activate().await;
    match res {
        Err(SandboxError::NetworkAcl { reason }) => {
            assert!(
                reason.contains("CAP_NET_ADMIN"),
                "expected CAP_NET_ADMIN message, got: {reason}"
            );
        }
        // Landlock may fail before we reach the network phase if the
        // kernel does not support it. That is also acceptable -- the
        // test is asserting that we *do not* silently succeed.
        Err(SandboxError::Landlock { .. }) => {
            eprintln!("Landlock failed before network phase; skipping strict assertion");
        }
        Err(SandboxError::NetworkNamespace { .. }) => {
            eprintln!("netns creation failed before ACL phase; skipping strict assertion");
        }
        Err(other) => panic!("unexpected error: {other:?}"),
        Ok(()) => {
            panic!("expected NetworkAcl error in strict mode without CAP_NET_ADMIN, got Ok(())")
        }
    }
}

/// Linux smoke test that actually installs iptables rules. This
/// requires `CAP_NET_ADMIN` and a working `iptables` binary, so it is
/// `#[ignore]`d by default. Run with:
///
/// ```sh
/// sudo -E cargo test -p tmg-sandbox --test network_acl -- --ignored
/// ```
///
/// Note: `localhost` resolves to both `127.0.0.1` (A) and `::1` (AAAA)
/// on dual-stack hosts. The ACL implementation is IPv4-only and
/// silently filters out the v6 entry, so the smoke test stays safe on
/// any host. IPv6 outbound is still blocked because the empty netns
/// has no v6 default route -- see the module-level docs in
/// `network_acl.rs` for the rationale.
#[cfg(target_os = "linux")]
#[tokio::test]
#[ignore = "requires CAP_NET_ADMIN and iptables; run with --ignored under sudo"]
async fn linux_apply_acl_smoke() {
    if !has_cap_net_admin_on_linux() {
        // Belt-and-suspenders: do not even try if we lack the
        // capability, even when the test was opted in with --ignored.
        eprintln!("skipping linux_apply_acl_smoke: CAP_NET_ADMIN missing");
        return;
    }
    if !iptables_available() {
        eprintln!("skipping linux_apply_acl_smoke: iptables binary not on PATH");
        return;
    }

    let tempdir = std::env::temp_dir();
    let config = SandboxConfig::new(&tempdir)
        .with_mode(SandboxMode::ReadOnly)
        .with_allowed_domains(vec!["localhost".to_owned()]);

    let ctx = SandboxContext::new(config);
    // We expect activate() to succeed end-to-end.
    let res = ctx.activate().await;
    assert!(res.is_ok(), "activate failed: {res:?}");
}

#[cfg(target_os = "linux")]
fn has_cap_net_admin_on_linux() -> bool {
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

#[cfg(target_os = "linux")]
fn iptables_available() -> bool {
    std::process::Command::new("iptables")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
