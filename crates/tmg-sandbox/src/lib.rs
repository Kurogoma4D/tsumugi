//! tmg-sandbox: Security sandbox for the tsumugi coding agent.
//!
//! This crate provides OS-level process isolation to restrict filesystem
//! access, network connectivity, and resource usage of agent-spawned
//! processes.
//!
//! # Platform support
//!
//! - **Linux**: Full support via Landlock (filesystem), network namespaces
//!   + iptables (network), and `/proc` (OOM score adjustment).
//! - **macOS / other**: All sandbox operations are no-ops that emit
//!   warnings. Path validation checks still function as a software-level
//!   safety net.
//!
//! # Architecture
//!
//! ```text
//! SandboxConfig ──▶ SandboxContext ──▶ platform::{linux, fallback}
//!                        │
//!                        ├── check_path_access()   (software check)
//!                        ├── check_write_access()  (software check)
//!                        ├── activate()            (OS-level restrictions)
//!                        └── run_command()          (timeout + OOM)
//! ```
//!
//! # Usage
//!
//! ```no_run
//! # use tmg_sandbox::{SandboxConfig, SandboxContext, SandboxMode};
//! # async fn example() -> Result<(), tmg_sandbox::SandboxError> {
//! let config = SandboxConfig::new("/home/user/project")
//!     .with_mode(SandboxMode::WorkspaceWrite)
//!     .with_allowed_domains(vec!["api.example.com".to_owned()])
//!     .with_timeout(60);
//!
//! let mut ctx = SandboxContext::new(config);
//! ctx.activate().await?;
//!
//! // Validate path access before operations.
//! ctx.check_path_access("/home/user/project/src/main.rs")?;
//! ctx.check_write_access("/home/user/project/output.txt")?;
//!
//! // Run a command within sandbox constraints.
//! let output = ctx.run_command("cargo build").await?;
//! assert!(output.success());
//! # Ok(())
//! # }
//! ```

pub mod config;
pub mod context;
pub mod error;
pub mod mode;
mod platform;
pub mod process;

pub use config::SandboxConfig;
pub use context::SandboxContext;
pub use error::SandboxError;
pub use mode::SandboxMode;
pub use process::CommandOutput;
