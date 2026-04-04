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
pub use linux::{adjust_oom_score, apply_landlock, create_network_namespace};

#[cfg(not(target_os = "linux"))]
pub use fallback::{adjust_oom_score, apply_landlock, create_network_namespace};
