//! Fallback sandbox implementation for non-Linux platforms.
//!
//! All sandbox operations are no-ops that emit warnings. The agent
//! functions normally but without OS-level security restrictions.
//!
//! Function signatures intentionally match the Linux module so the
//! platform dispatch in `mod.rs` works uniformly.

use crate::config::SandboxConfig;
use crate::error::SandboxError;

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
