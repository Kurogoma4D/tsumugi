//! Linux network namespace creation via `unshare(CLONE_NEWNET)`.
//!
//! This module is only compiled on `target_os = "linux"`.

use crate::error::SandboxError;

/// Create a new network namespace using `unshare(CLONE_NEWNET)`.
///
/// After this call, the current process has its own empty network
/// namespace with only a loopback interface. All network access is
/// blocked unless iptables rules are subsequently added.
///
/// # Errors
///
/// Returns [`SandboxError::NetworkNamespace`] if the `unshare` syscall fails
/// (e.g., due to insufficient privileges).
///
/// # Safety
///
/// Calls `libc::unshare(CLONE_NEWNET)` which is safe to call but modifies
/// the process's network namespace. This is an irreversible operation for
/// the current process.
#[expect(
    unsafe_code,
    reason = "FFI call to libc::unshare for network namespace isolation"
)]
pub fn create_network_namespace() -> Result<(), SandboxError> {
    // SAFETY: `unshare(CLONE_NEWNET)` is a Linux syscall that creates a new
    // network namespace for the calling process. It does not access any memory
    // unsafely; it only modifies kernel-level namespace state. The return value
    // is checked for errors.
    let ret = unsafe { libc::unshare(libc::CLONE_NEWNET) };
    if ret != 0 {
        return Err(SandboxError::NetworkNamespace {
            source: std::io::Error::last_os_error(),
        });
    }
    Ok(())
}

/// Bring up the loopback interface inside the current network namespace.
///
/// A freshly-created netns has a loopback (`lo`) device that starts in the
/// `DOWN` state. Without this, even loopback traffic (e.g. `127.0.0.1:8080`
/// connections that local tooling occasionally relies on) is dropped.
///
/// This is best-effort: failures are not fatal, because some test
/// environments do not run with `CAP_NET_ADMIN` and the caller may have
/// chosen to fall back gracefully. The returned [`SandboxError`] should
/// be propagated only when strict enforcement is desired.
///
/// # Errors
///
/// Returns [`SandboxError::NetworkAcl`] if `ip link set lo up` cannot be
/// executed or exits with a non-zero status.
pub fn bring_up_loopback() -> Result<(), SandboxError> {
    let output = std::process::Command::new("ip")
        .args(["link", "set", "dev", "lo", "up"])
        .output()
        .map_err(|e| SandboxError::NetworkAcl {
            reason: format!("failed to spawn `ip link set lo up`: {e}"),
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(SandboxError::NetworkAcl {
            reason: format!(
                "`ip link set lo up` exited with {}: {}",
                output.status,
                stderr.trim()
            ),
        });
    }
    Ok(())
}
