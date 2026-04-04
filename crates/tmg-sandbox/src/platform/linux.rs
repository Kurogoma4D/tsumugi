//! Linux-specific sandbox implementation using Landlock and network namespaces.
//!
//! This module is only compiled on `target_os = "linux"`.

use std::path::Path;

use landlock::{
    ABI, Access, AccessFs, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr,
};

use crate::config::SandboxConfig;
use crate::error::SandboxError;
use crate::mode::SandboxMode;

/// The best Landlock ABI version we target.
const LANDLOCK_ABI: ABI = ABI::V3;

/// System paths that receive read-only access in the Landlock ruleset.
const SYSTEM_READ_ONLY_PATHS: &[&str] = &["/usr", "/bin", "/lib", "/lib64"];

/// Apply Landlock filesystem restrictions based on the sandbox configuration.
///
/// This restricts the current process (and all future children) to:
/// - workspace directory: read-only or read+write depending on mode
/// - system paths (`/usr`, `/bin`, `/lib`, `/lib64`): read-only
/// - `~/.config/tsumugi/`: read-only
/// - everything else: no access
///
/// # Errors
///
/// Returns [`SandboxError::Landlock`] if Landlock is not supported by the
/// kernel or if rule creation/enforcement fails.
///
/// # Safety
///
/// This function uses Landlock which is a safe Linux security module API.
/// No `unsafe` code is required.
pub fn apply_landlock(config: &SandboxConfig) -> Result<(), SandboxError> {
    if config.mode == SandboxMode::Full {
        return Ok(());
    }

    let read_access = AccessFs::from_read(LANDLOCK_ABI);
    let write_access = AccessFs::from_write(LANDLOCK_ABI);
    let read_write_access = read_access | write_access;

    let mut ruleset = Ruleset::default()
        .handle_access(read_write_access)
        .map_err(|e| SandboxError::Landlock {
            reason: format!("failed to create ruleset: {e}"),
        })?
        .create()
        .map_err(|e| SandboxError::Landlock {
            reason: format!("failed to create ruleset: {e}"),
        })?;

    // Workspace directory access based on mode.
    let workspace_access = if config.mode.allows_workspace_write() {
        read_write_access
    } else {
        read_access
    };

    ruleset = add_path_rule(ruleset, &config.workspace, workspace_access)?;

    // System paths: read-only.
    for path_str in SYSTEM_READ_ONLY_PATHS {
        let path = Path::new(path_str);
        if path.exists() {
            ruleset = add_path_rule(ruleset, path, read_access)?;
        }
    }

    // ~/.config/tsumugi/: read-only.
    if let Some(config_dir) = dirs_config_path() {
        if config_dir.exists() {
            ruleset = add_path_rule(ruleset, &config_dir, read_access)?;
        }
    }

    ruleset
        .restrict_self()
        .map_err(|e| SandboxError::Landlock {
            reason: format!("failed to enforce ruleset: {e}"),
        })?;

    Ok(())
}

/// Add a Landlock path rule to the ruleset.
fn add_path_rule(
    mut ruleset: landlock::RulesetCreated,
    path: &Path,
    access: AccessFs,
) -> Result<landlock::RulesetCreated, SandboxError> {
    let fd = PathFd::new(path).map_err(|e| SandboxError::Landlock {
        reason: format!("failed to open path '{}': {e}", path.display()),
    })?;

    ruleset = ruleset
        .add_rule(PathBeneath::new(fd, access))
        .map_err(|e| SandboxError::Landlock {
            reason: format!("failed to add rule for '{}': {e}", path.display()),
        })?;

    Ok(ruleset)
}

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

/// Adjust the OOM score for a given process.
///
/// Writes to `/proc/{pid}/oom_score_adj` to influence the kernel's
/// OOM killer behavior.
///
/// # Errors
///
/// Returns [`SandboxError::OomAdjust`] if the write fails.
pub fn adjust_oom_score(pid: u32, score: i32) -> Result<(), SandboxError> {
    let path = format!("/proc/{pid}/oom_score_adj");
    std::fs::write(&path, score.to_string()).map_err(|e| SandboxError::OomAdjust { source: e })
}

/// Get the tsumugi configuration directory path.
///
/// Returns `~/.config/tsumugi/` if the home directory can be determined.
fn dirs_config_path() -> Option<std::path::PathBuf> {
    // Use $HOME directly as we cannot depend on the `dirs` crate here
    // without adding it as a dependency. In production, the config path
    // should be passed in from the caller.
    std::env::var_os("HOME").map(|home| {
        let mut path = std::path::PathBuf::from(home);
        path.push(".config");
        path.push("tsumugi");
        path
    })
}
