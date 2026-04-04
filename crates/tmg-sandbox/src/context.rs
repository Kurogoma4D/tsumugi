//! `SandboxContext`: the main entry point for sandbox setup and enforcement.

use std::path::{Path, PathBuf};

use crate::config::SandboxConfig;
use crate::error::SandboxError;
use crate::mode::SandboxMode;
use crate::platform;
use crate::process::{self, CommandOutput};

/// A configured sandbox environment.
///
/// `SandboxContext` holds the resolved sandbox configuration and provides
/// methods to:
///
/// - Apply OS-level restrictions (Landlock, network namespace)
/// - Validate filesystem access before tool execution
/// - Run commands with timeout and OOM score enforcement
///
/// # Platform behavior
///
/// On Linux, calling [`activate`](SandboxContext::activate) applies Landlock
/// filesystem restrictions and optionally creates a network namespace with
/// domain-based allowlisting.
///
/// On non-Linux platforms, `activate` emits warnings and proceeds without
/// OS-level restrictions. Path validation still functions as a software-level
/// check.
///
/// # Example
///
/// ```no_run
/// # use tmg_sandbox::{SandboxConfig, SandboxContext};
/// # async fn example() -> Result<(), tmg_sandbox::SandboxError> {
/// let config = SandboxConfig::new("/home/user/project");
/// let mut ctx = SandboxContext::new(config);
/// ctx.activate().await?;
///
/// // Run a command within sandbox constraints.
/// let output = ctx.run_command("ls -la").await?;
/// assert!(output.success());
/// # Ok(())
/// # }
/// ```
pub struct SandboxContext {
    /// The sandbox configuration.
    config: SandboxConfig,

    /// Whether OS-level restrictions have been activated.
    activated: bool,
}

impl SandboxContext {
    /// Create a new sandbox context from the given configuration.
    ///
    /// The sandbox is not active until [`activate`](Self::activate) is called.
    pub fn new(config: SandboxConfig) -> Self {
        Self {
            config,
            activated: false,
        }
    }

    /// Activate OS-level sandbox restrictions.
    ///
    /// On Linux, this:
    /// 1. Applies Landlock filesystem rules
    /// 2. Creates a network namespace (if `allowed_domains` is configured)
    /// 3. Applies iptables rules for the domain allowlist
    ///
    /// On non-Linux platforms, this emits warnings and returns `Ok(())`.
    ///
    /// This method is idempotent: calling it multiple times has no
    /// additional effect after the first successful activation.
    ///
    /// # Errors
    ///
    /// Returns [`SandboxError`] if any OS-level restriction fails to apply.
    pub async fn activate(&mut self) -> Result<(), SandboxError> {
        if self.activated {
            return Ok(());
        }

        if self.config.mode == SandboxMode::Full {
            self.activated = true;
            return Ok(());
        }

        // Apply filesystem restrictions.
        platform::apply_landlock(&self.config)?;

        // Apply network restrictions if domains are configured.
        if !self.config.allowed_domains.is_empty() {
            platform::create_network_namespace()?;
            platform::apply_network_allowlist(&self.config.allowed_domains).await?;
        }

        self.activated = true;
        Ok(())
    }

    /// Check whether a given path is accessible under the current sandbox mode.
    ///
    /// This performs a software-level check (not relying on Landlock):
    /// - In `Full` mode, all paths are allowed.
    /// - In `WorkspaceWrite` and `ReadOnly` modes, only paths within the
    ///   workspace directory (and system paths) are allowed.
    ///
    /// This does **not** distinguish between read and write access; use
    /// [`check_write_access`](Self::check_write_access) for write checks.
    pub fn check_path_access(&self, path: impl AsRef<Path>) -> Result<(), SandboxError> {
        if self.config.mode.is_unrestricted() {
            return Ok(());
        }

        let path = path.as_ref();
        let canonical = normalize_path(path);

        // Allow access to workspace directory.
        if canonical.starts_with(&self.config.workspace) {
            return Ok(());
        }

        // Allow access to system paths.
        for system_path in &["/usr", "/bin", "/lib", "/lib64"] {
            if canonical.starts_with(system_path) {
                return Ok(());
            }
        }

        // Allow access to tsumugi config directory.
        if let Some(config_dir) = tsumugi_config_dir() {
            if canonical.starts_with(&config_dir) {
                return Ok(());
            }
        }

        Err(SandboxError::AccessDenied {
            path: path.to_path_buf(),
        })
    }

    /// Check whether writing to a given path is allowed under the current mode.
    ///
    /// - `Full` mode: all writes allowed.
    /// - `WorkspaceWrite` mode: writes allowed only within the workspace.
    /// - `ReadOnly` mode: no writes allowed anywhere.
    pub fn check_write_access(&self, path: impl AsRef<Path>) -> Result<(), SandboxError> {
        if self.config.mode.is_unrestricted() {
            return Ok(());
        }

        let path = path.as_ref();

        if !self.config.mode.allows_workspace_write() {
            return Err(SandboxError::AccessDenied {
                path: path.to_path_buf(),
            });
        }

        // WorkspaceWrite mode: only allow writes within the workspace.
        let canonical = normalize_path(path);
        if canonical.starts_with(&self.config.workspace) {
            return Ok(());
        }

        Err(SandboxError::AccessDenied {
            path: path.to_path_buf(),
        })
    }

    /// Run a shell command within sandbox constraints.
    ///
    /// The command is executed with:
    /// - The configured timeout (default 30s)
    /// - OOM score adjustment (Linux only)
    /// - `kill_on_drop` for cleanup on cancellation
    ///
    /// # Errors
    ///
    /// Returns [`SandboxError::Timeout`] if the command exceeds the timeout,
    /// or [`SandboxError::Io`] if the command cannot be spawned.
    pub async fn run_command(&self, command: &str) -> Result<CommandOutput, SandboxError> {
        process::run_sandboxed_command(command, self.config.timeout_secs, self.config.oom_score_adj)
            .await
    }

    /// Return a reference to the sandbox configuration.
    pub fn config(&self) -> &SandboxConfig {
        &self.config
    }

    /// Return the current sandbox mode.
    pub fn mode(&self) -> SandboxMode {
        self.config.mode
    }

    /// Return the workspace path.
    pub fn workspace(&self) -> &Path {
        &self.config.workspace
    }

    /// Return whether the sandbox has been activated.
    pub fn is_activated(&self) -> bool {
        self.activated
    }
}

/// Normalize a path by resolving `.` and `..` components without
/// touching the filesystem (no symlink resolution).
///
/// This is a best-effort normalization for sandbox path checks.
fn normalize_path(path: &Path) -> PathBuf {
    use std::path::Component;

    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                normalized.pop();
            }
            Component::CurDir => {}
            other => normalized.push(other),
        }
    }
    normalized
}

/// Get the tsumugi configuration directory path (`~/.config/tsumugi/`).
fn tsumugi_config_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| {
        let mut path = PathBuf::from(home);
        path.push(".config");
        path.push("tsumugi");
        path
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SandboxConfig {
        SandboxConfig::new("/tmp/workspace")
    }

    #[test]
    fn full_mode_allows_everything() {
        let config = test_config().with_mode(SandboxMode::Full);
        let ctx = SandboxContext::new(config);

        assert!(ctx.check_path_access("/etc/passwd").is_ok());
        assert!(ctx.check_path_access("/root/.ssh/id_rsa").is_ok());
        assert!(ctx.check_write_access("/etc/shadow").is_ok());
    }

    #[test]
    fn workspace_write_allows_workspace_read() {
        let config = test_config().with_mode(SandboxMode::WorkspaceWrite);
        let ctx = SandboxContext::new(config);

        assert!(ctx.check_path_access("/tmp/workspace/src/main.rs").is_ok());
    }

    #[test]
    fn workspace_write_allows_workspace_write() {
        let config = test_config().with_mode(SandboxMode::WorkspaceWrite);
        let ctx = SandboxContext::new(config);

        assert!(ctx.check_write_access("/tmp/workspace/output.txt").is_ok());
    }

    #[test]
    fn workspace_write_denies_outside_write() {
        let config = test_config().with_mode(SandboxMode::WorkspaceWrite);
        let ctx = SandboxContext::new(config);

        assert!(ctx.check_write_access("/etc/passwd").is_err());
        assert!(ctx.check_write_access("/root/file.txt").is_err());
    }

    #[test]
    fn workspace_write_allows_system_path_read() {
        let config = test_config().with_mode(SandboxMode::WorkspaceWrite);
        let ctx = SandboxContext::new(config);

        assert!(ctx.check_path_access("/usr/bin/ls").is_ok());
        assert!(ctx.check_path_access("/bin/sh").is_ok());
        assert!(
            ctx.check_path_access("/lib/x86_64-linux-gnu/libc.so")
                .is_ok()
        );
    }

    #[test]
    fn workspace_write_denies_outside_read() {
        let config = test_config().with_mode(SandboxMode::WorkspaceWrite);
        let ctx = SandboxContext::new(config);

        assert!(ctx.check_path_access("/etc/passwd").is_err());
        assert!(ctx.check_path_access("/root/.bashrc").is_err());
        assert!(ctx.check_path_access("/home/other/secret").is_err());
    }

    #[test]
    fn read_only_denies_all_writes() {
        let config = test_config().with_mode(SandboxMode::ReadOnly);
        let ctx = SandboxContext::new(config);

        // Even workspace writes are denied.
        assert!(ctx.check_write_access("/tmp/workspace/file.txt").is_err());
        assert!(ctx.check_write_access("/etc/passwd").is_err());
    }

    #[test]
    fn read_only_allows_workspace_read() {
        let config = test_config().with_mode(SandboxMode::ReadOnly);
        let ctx = SandboxContext::new(config);

        assert!(ctx.check_path_access("/tmp/workspace/src/main.rs").is_ok());
    }

    #[test]
    fn read_only_allows_system_path_read() {
        let config = test_config().with_mode(SandboxMode::ReadOnly);
        let ctx = SandboxContext::new(config);

        assert!(ctx.check_path_access("/usr/local/bin/cargo").is_ok());
    }

    #[test]
    fn read_only_denies_outside_read() {
        let config = test_config().with_mode(SandboxMode::ReadOnly);
        let ctx = SandboxContext::new(config);

        assert!(ctx.check_path_access("/etc/shadow").is_err());
    }

    #[test]
    fn path_traversal_normalization() {
        let config = test_config().with_mode(SandboxMode::WorkspaceWrite);
        let ctx = SandboxContext::new(config);

        // Attempting to escape via `..` should be caught.
        assert!(
            ctx.check_path_access("/tmp/workspace/../../../etc/passwd")
                .is_err()
        );
    }

    #[test]
    fn normalize_path_handles_parent_dir() {
        let result = normalize_path(Path::new("/tmp/workspace/../secret"));
        assert_eq!(result, PathBuf::from("/tmp/secret"));
    }

    #[test]
    fn normalize_path_handles_current_dir() {
        let result = normalize_path(Path::new("/tmp/./workspace/./file"));
        assert_eq!(result, PathBuf::from("/tmp/workspace/file"));
    }

    #[tokio::test]
    async fn run_command_success() {
        let config = test_config();
        let ctx = SandboxContext::new(config);

        let output = ctx.run_command("echo sandbox-test").await;
        assert!(output.is_ok());
        let output = output.unwrap_or_else(|_| CommandOutput {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: -1,
        });
        assert!(output.success());
        assert!(output.stdout.contains("sandbox-test"));
    }

    #[tokio::test]
    async fn run_command_timeout() {
        let config = test_config().with_timeout(1);
        let ctx = SandboxContext::new(config);

        let result = ctx.run_command("sleep 60").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn activate_is_idempotent() {
        let config = test_config().with_mode(SandboxMode::Full);
        let mut ctx = SandboxContext::new(config);

        assert!(!ctx.is_activated());
        ctx.activate().await.unwrap_or(());
        assert!(ctx.is_activated());
        // Second activation should succeed without error.
        ctx.activate().await.unwrap_or(());
        assert!(ctx.is_activated());
    }
}
