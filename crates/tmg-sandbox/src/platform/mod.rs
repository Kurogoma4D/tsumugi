//! Platform-specific sandbox implementations.
//!
//! On Linux, this module delegates to Landlock and network namespace
//! APIs. On all other platforms, it provides no-op implementations
//! that emit warnings.

#[cfg(target_os = "linux")]
mod linux;

#[cfg(not(target_os = "linux"))]
mod fallback;

#[cfg(target_os = "linux")]
pub use linux::{
    NetworkAcl, adjust_oom_score, apply_landlock, bring_up_loopback, create_network_namespace,
    has_cap_net_admin, install_iptables_chain, resolve_domains,
};

#[cfg(not(target_os = "linux"))]
pub use fallback::{
    NetworkAcl, adjust_oom_score, apply_landlock, bring_up_loopback, create_network_namespace,
    has_cap_net_admin, install_iptables_chain, resolve_domains,
};
