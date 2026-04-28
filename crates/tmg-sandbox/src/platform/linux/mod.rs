//! Linux-specific sandbox implementation using Landlock and network namespaces.
//!
//! This module is only compiled on `target_os = "linux"`.
//!
//! # Submodules
//!
//! - [`landlock`]: filesystem restriction via the Landlock LSM
//! - [`netns`]: network namespace creation (`unshare(CLONE_NEWNET)`) and
//!   loopback bring-up
//! - [`network_acl`]: DNS resolution + iptables ACL inside the netns

pub mod landlock;
pub mod netns;
pub mod network_acl;

pub use landlock::apply_landlock;
pub use netns::{bring_up_loopback, create_network_namespace};
pub use network_acl::{NetworkAcl, has_cap_net_admin, install_iptables_chain, resolve_domains};

use crate::error::SandboxError;

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
