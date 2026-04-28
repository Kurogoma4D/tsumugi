//! `SandboxContext`: the main entry point for sandbox setup and enforcement.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

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
/// let ctx = SandboxContext::new(config);
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
    ///
    /// Stored as an [`AtomicBool`] so [`activate`](Self::activate) can
    /// take `&self` and the activation flag can be safely set even
    /// after the context has been wrapped in `Arc<SandboxContext>`.
    activated: AtomicBool,
}

impl SandboxContext {
    /// Create a new sandbox context from the given configuration.
    ///
    /// The sandbox is not active until [`activate`](Self::activate) is called.
    pub fn new(config: SandboxConfig) -> Self {
        Self {
            config,
            activated: AtomicBool::new(false),
        }
    }

    /// Activate OS-level sandbox restrictions.
    ///
    /// On Linux, this:
    /// 1. Applies Landlock filesystem rules
    /// 2. Creates an isolated network namespace that blocks all external
    ///    network access
    ///
    /// On non-Linux platforms, this emits warnings and returns `Ok(())`.
    ///
    /// This method is idempotent: calling it multiple times has no
    /// additional effect after the first successful activation.
    ///
    /// # Network isolation
    ///
    /// Network restriction is achieved by placing the process into a new,
    /// empty network namespace via `unshare(CLONE_NEWNET)`. The new namespace
    /// contains only a loopback interface with no external connectivity.
    ///
    /// The `allowed_domains` configuration field is accepted but **not yet
    /// enforced** -- selective domain allowlisting requires veth pair setup
    /// or Landlock v4+ network access rules, which are planned for future
    /// work. Currently, when any network restriction is active, **all**
    /// external network access is blocked.
    ///
    /// # Errors
    ///
    /// Returns [`SandboxError`] if any OS-level restriction fails to apply.
    #[expect(
        clippy::unused_async,
        reason = "kept async for API stability; future network allowlist implementation will require async"
    )]
    pub async fn activate(&self) -> Result<(), SandboxError> {
        // Use `SeqCst` for the load/store pair so the "first activator
        // wins" check is sequentially consistent with the activation
        // side effects (`apply_landlock` / `create_network_namespace`)
        // observed by other threads. The activation path is rare
        // enough that the slightly stronger ordering is not a hot-path
        // concern.
        if self.activated.load(Ordering::SeqCst) {
            return Ok(());
        }

        if self.config.mode == SandboxMode::Full {
            self.activated.store(true, Ordering::SeqCst);
            return Ok(());
        }

        // Apply filesystem restrictions.
        platform::apply_landlock(&self.config)?;

        // Apply network restrictions: create an empty network namespace
        // that blocks all external connectivity.
        platform::create_network_namespace()?;

        self.activated.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Check whether a given path is accessible under the current sandbox mode.
    ///
    /// This performs a software-level check (not relying on Landlock):
    /// - In `Full` mode, all paths are allowed.
    /// - In `WorkspaceWrite` and `ReadOnly` modes, only paths within the
    ///   workspace directory (and system paths) are allowed.
    ///
    /// Both the candidate path and the workspace path are normalised
    /// to their canonical (symlink-resolved) form before comparison,
    /// so a workspace configured as `/tmp/workspace` matches a
    /// candidate that canonicalises through `/tmp -> /private/tmp` on
    /// macOS.
    ///
    /// This does **not** distinguish between read and write access; use
    /// [`check_write_access`](Self::check_write_access) for write checks.
    pub fn check_path_access(&self, path: impl AsRef<Path>) -> Result<(), SandboxError> {
        if self.config.mode.is_unrestricted() {
            return Ok(());
        }

        let path = path.as_ref();
        let canonical = normalize_path(path, &self.config.workspace);
        let canonical_workspace = normalize_path(&self.config.workspace, &self.config.workspace);

        // Allow access to workspace directory.
        if canonical.starts_with(&canonical_workspace) {
            return Ok(());
        }

        // Allow access to system paths (platform-specific).
        for system_path in system_read_paths() {
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
    ///
    /// Like [`check_path_access`](Self::check_path_access), this
    /// normalises both the candidate path and the workspace path
    /// before comparing them, so symlinked workspace prefixes (e.g.
    /// `/tmp -> /private/tmp` on macOS) line up correctly.
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
        let canonical = normalize_path(path, &self.config.workspace);
        let canonical_workspace = normalize_path(&self.config.workspace, &self.config.workspace);
        if canonical.starts_with(&canonical_workspace) {
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
    /// - The sandbox workspace as `cwd`
    /// - `kill_on_drop` for cleanup on cancellation
    ///
    /// # Errors
    ///
    /// Returns [`SandboxError::Timeout`] if the command exceeds the timeout,
    /// or [`SandboxError::Io`] if the command cannot be spawned.
    pub async fn run_command(&self, command: &str) -> Result<CommandOutput, SandboxError> {
        process::run_sandboxed_command(
            command,
            self.config.timeout_secs,
            self.config.oom_score_adj,
            &self.config.workspace,
        )
        .await
    }

    /// Run a shell command with a per-call timeout that may shorten
    /// (but never extend) the sandbox-wide default.
    ///
    /// Equivalent to [`run_command`](Self::run_command) but allows
    /// callers (e.g. `shell_exec`) to cap the timeout for a single
    /// invocation without cloning the [`SandboxConfig`] just to swap
    /// the `timeout_secs` field.
    ///
    /// The supplied `timeout_secs` is **clamped** to the sandbox-wide
    /// default: if it exceeds `self.config.timeout_secs`, the
    /// configured value is used instead. This preserves the invariant
    /// that the sandbox owns the upper bound on process budget.
    ///
    /// # Errors
    ///
    /// Returns [`SandboxError::Timeout`] if the command exceeds the
    /// effective timeout, or [`SandboxError::Io`] if the command
    /// cannot be spawned.
    pub async fn run_command_with_timeout(
        &self,
        command: &str,
        timeout_secs: u64,
    ) -> Result<CommandOutput, SandboxError> {
        let effective = timeout_secs.min(self.config.timeout_secs);
        process::run_sandboxed_command(
            command,
            effective,
            self.config.oom_score_adj,
            &self.config.workspace,
        )
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
        self.activated.load(Ordering::SeqCst)
    }

    /// Construct a permissive [`SandboxContext`] suitable for tests.
    ///
    /// The returned context uses [`SandboxMode::Full`] (no filesystem
    /// restrictions), an empty domain allowlist, and the system temp
    /// directory as the workspace. Path-access checks therefore always
    /// succeed, so tests that exercise tools through the registry do
    /// not need to thread their own [`SandboxConfig`] through every
    /// call site.
    ///
    /// This helper is **only** intended for unit and integration tests;
    /// production code paths should construct a [`SandboxContext`] from
    /// a real [`SandboxConfig`] derived from the user's
    /// `[sandbox]` configuration.
    #[must_use]
    pub fn test_default() -> Self {
        let workspace = std::env::temp_dir();
        let config = SandboxConfig::new(workspace).with_mode(SandboxMode::Full);
        Self::new(config)
    }

    /// Derive a child [`SandboxContext`] from this context with the
    /// given operating mode.
    ///
    /// The child inherits the parent's `workspace`, `allowed_domains`,
    /// `timeout_secs`, and `oom_score_adj` and overrides only the
    /// [`SandboxMode`]. The resulting context is **not** activated;
    /// subagents that need OS-level enforcement should call
    /// [`activate`](Self::activate) themselves.
    ///
    /// Used by [`SubagentManager`](https://docs.rs/tmg-agents) to
    /// spawn each subagent under the [`SandboxMode`] declared by its
    /// [`AgentType::sandbox_mode`](https://docs.rs/tmg-agents) (e.g.
    /// `WorkspaceWrite` for `worker` / `initializer` / `tester`,
    /// `ReadOnly` for `explore` / `plan` / `qa` / `escalator`).
    #[must_use]
    pub fn derive(&self, mode: SandboxMode) -> Self {
        let mut config = self.config.clone();
        config.mode = mode;
        Self::new(config)
    }
}

/// Normalize a path for sandbox access checks.
///
/// If `path` is relative, it is first joined with the given `base` directory
/// (typically the workspace or current working directory) so that the resulting
/// path is absolute and can be compared against the allowlist.
///
/// Resolution strategy:
/// 1. If the full path exists, [`std::fs::canonicalize`] is used so any
///    symlink along the chain is resolved to its real target.
/// 2. Otherwise, the **deepest existing ancestor** is canonicalized and
///    the remaining (non-existing) components are appended after lexical
///    normalization. This catches a symlinked *parent* directory pointing
///    outside the sandbox even when the leaf does not exist yet (e.g. a
///    file the agent is about to create).
/// 3. If even the root has no canonical form (highly unusual), the
///    fallback is purely lexical normalization.
fn normalize_path(path: &Path, base: &Path) -> PathBuf {
    let absolute = if path.is_relative() {
        base.join(path)
    } else {
        path.to_path_buf()
    };

    // Fast path: the entire path exists -- canonicalize resolves all
    // symlinks for us.
    if let Ok(canonical) = std::fs::canonicalize(&absolute) {
        return canonical;
    }

    // Slow path: the leaf (or several trailing components) does not
    // exist yet. Walk up to the deepest existing ancestor,
    // canonicalize it, and re-attach the missing tail. This way a
    // symlinked parent directory pointing outside the sandbox is
    // still detected.
    let lexical = lexical_normalize(&absolute);
    let mut deepest = lexical.as_path();
    let mut tail: Vec<&std::ffi::OsStr> = Vec::new();
    loop {
        if let Ok(canonical) = std::fs::canonicalize(deepest) {
            // Re-append the missing components in original order.
            let mut resolved = canonical;
            for component in tail.iter().rev() {
                resolved.push(component);
            }
            return resolved;
        }
        // Strip one component and retry. If we run out of ancestors,
        // fall through to the lexical fallback.
        match (deepest.parent(), deepest.file_name()) {
            (Some(parent), Some(name)) => {
                tail.push(name);
                deepest = parent;
            }
            _ => break,
        }
    }

    // Final fallback: purely lexical normalization. A symlinked
    // ancestor here would slip through, but this branch is only
    // reached when even the filesystem root cannot be canonicalized.
    lexical
}

/// Resolve `.` and `..` components without touching the filesystem.
fn lexical_normalize(path: &Path) -> PathBuf {
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

/// System paths that are granted read-only access in software path checks.
///
/// On Linux, these correspond to the standard FHS directories containing
/// system binaries and shared libraries. On macOS, Homebrew and system
/// framework paths are included instead.
#[cfg(target_os = "linux")]
fn system_read_paths() -> &'static [&'static str] {
    &["/usr", "/bin", "/lib", "/lib64"]
}

/// System paths that are granted read-only access in software path checks.
///
/// On macOS, these include Homebrew, system frameworks, and standard
/// Unix directories.
#[cfg(target_os = "macos")]
fn system_read_paths() -> &'static [&'static str] {
    &[
        "/usr",
        "/bin",
        "/sbin",
        "/opt/homebrew",
        "/System",
        "/Library",
    ]
}

/// Fallback system path allowlist for other platforms.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn system_read_paths() -> &'static [&'static str] {
    &["/usr", "/bin"]
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
        #[cfg(target_os = "linux")]
        {
            assert!(ctx.check_path_access("/bin/sh").is_ok());
            assert!(
                ctx.check_path_access("/lib/x86_64-linux-gnu/libc.so")
                    .is_ok()
            );
        }
        #[cfg(target_os = "macos")]
        {
            assert!(ctx.check_path_access("/bin/sh").is_ok());
            assert!(ctx.check_path_access("/opt/homebrew/bin/cargo").is_ok());
            assert!(ctx.check_path_access("/System/Library/Frameworks").is_ok());
            assert!(ctx.check_path_access("/Library/Developer").is_ok());
        }
    }

    #[test]
    fn workspace_write_denies_outside_read() {
        let config = test_config().with_mode(SandboxMode::WorkspaceWrite);
        let ctx = SandboxContext::new(config);

        assert!(ctx.check_path_access("/etc/passwd").is_err());
        assert!(ctx.check_path_access("/root/.bashrc").is_err());
        // `/home/...` on macOS canonicalises through
        // `/System/Volumes/Data/home/...` and ends up under the
        // `/System` system-read allowlist, which is correct platform
        // behaviour (the boundary follows real filesystem layout).
        // Use `/etc/passwd` and `/root/.bashrc` which are reliably
        // outside the allowlist on both Linux and macOS.
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

    /// Normalize paths to a single canonical form for cross-platform
    /// test assertions. On macOS the system temp `/tmp` is a symlink
    /// to `/private/tmp`, so the new symlink-resolving `normalize_path`
    /// returns the latter for any descendant whose ancestor exists.
    /// The tests below accept either form.
    fn paths_match(actual: &Path, expected_lexical: &Path) -> bool {
        if actual == expected_lexical {
            return true;
        }
        // macOS canonicalisation: `/tmp` -> `/private/tmp`.
        let mut prefixed = PathBuf::from("/private");
        for component in expected_lexical.components().skip(1) {
            prefixed.push(component);
        }
        actual == prefixed.as_path()
    }

    #[test]
    fn normalize_path_handles_parent_dir() {
        let base = Path::new("/tmp/workspace");
        // Absolute path with `..` -- base is unused.
        let result = normalize_path(Path::new("/tmp/workspace/../secret"), base);
        assert!(
            paths_match(&result, Path::new("/tmp/secret")),
            "got {result:?}"
        );
    }

    #[test]
    fn normalize_path_handles_current_dir() {
        let base = Path::new("/tmp/workspace");
        let result = normalize_path(Path::new("/tmp/./workspace/./file"), base);
        assert!(
            paths_match(&result, Path::new("/tmp/workspace/file")),
            "got {result:?}"
        );
    }

    #[test]
    fn normalize_path_resolves_relative_path() {
        let base = Path::new("/tmp/workspace");
        let result = normalize_path(Path::new("src/main.rs"), base);
        assert!(
            paths_match(&result, Path::new("/tmp/workspace/src/main.rs")),
            "got {result:?}"
        );
    }

    #[test]
    fn relative_path_access_within_workspace() {
        let config = test_config().with_mode(SandboxMode::WorkspaceWrite);
        let ctx = SandboxContext::new(config);

        // Relative paths should be resolved against the workspace.
        assert!(ctx.check_path_access("src/main.rs").is_ok());
        assert!(ctx.check_write_access("output.txt").is_ok());
    }

    #[test]
    fn relative_path_traversal_denied() {
        let config = test_config().with_mode(SandboxMode::WorkspaceWrite);
        let ctx = SandboxContext::new(config);

        // Relative path that escapes via `..` should be denied.
        assert!(ctx.check_path_access("../../etc/passwd").is_err());
    }

    #[tokio::test]
    async fn run_command_success() {
        // The new run_command anchors `cwd` at the workspace, so the
        // workspace directory MUST exist on disk for the spawn to
        // succeed. Use the system temp dir, which always exists,
        // rather than the static `/tmp/workspace` used by `test_config`.
        let config = SandboxConfig::new(std::env::temp_dir());
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
        let config = SandboxConfig::new(std::env::temp_dir()).with_timeout(1);
        let ctx = SandboxContext::new(config);

        let result = ctx.run_command("sleep 60").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn activate_is_idempotent() {
        let config = test_config().with_mode(SandboxMode::Full);
        let ctx = SandboxContext::new(config);

        assert!(!ctx.is_activated());
        ctx.activate().await.unwrap_or(());
        assert!(ctx.is_activated());
        // Second activation should succeed without error.
        ctx.activate().await.unwrap_or(());
        assert!(ctx.is_activated());
    }

    #[tokio::test]
    async fn activate_works_through_arc() {
        // Issue #1/#2: `Arc<SandboxContext>::activate()` must compile
        // and run. With the `&mut self` API this required taking a
        // mutex; the `&self` + `AtomicBool` design lets the wrapped
        // context be activated directly.
        let config = test_config().with_mode(SandboxMode::Full);
        let ctx = std::sync::Arc::new(SandboxContext::new(config));
        ctx.activate().await.unwrap_or(());
        assert!(ctx.is_activated());
    }
}
