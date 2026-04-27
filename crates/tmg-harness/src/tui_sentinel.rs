//! TUI-attached sentinel file management.
//!
//! When a `tmg` TUI starts up against a run, it writes a small file at
//! `<run-dir>/.tui-pid` containing its PID. CLI mutators (`tmg run
//! pause` / `tmg run abort`) check for this sentinel before writing
//! `run.toml`: if the sentinel exists and the recorded process is
//! still alive, the CLI refuses with [`HarnessError::TuiAttached`]
//! rather than racing the live runner. The TUI removes the file on
//! shutdown.
//!
//! ## Failure model
//!
//! - **Stale sentinel** (process no longer alive): treated as
//!   "no TUI attached"; the CLI proceeds with its mutation. The
//!   sentinel is removed best-effort on the way through.
//! - **Unparseable sentinel** (corrupted contents): logged at warn
//!   and treated as stale; same as above.
//! - **Sentinel write/remove I/O errors**: surfaced as
//!   [`HarnessError::Io`] from the corresponding [`write`] / [`clear`]
//!   helper. The TUI tolerates these (best-effort) so a permission
//!   problem on the run dir doesn't abort startup.
//!
//! ## Filename
//!
//! The leading dot keeps the file out of the way of casual `ls`
//! listings. Tests verify the constant rather than hard-coding the
//! string elsewhere.

use std::fs;
use std::path::{Path, PathBuf};

use crate::error::HarnessError;

/// Filename of the TUI-attached sentinel under each run directory.
pub const TUI_SENTINEL_FILENAME: &str = ".tui-pid";

/// Compute the sentinel path inside the run directory.
#[must_use]
pub fn sentinel_path(run_dir: &Path) -> PathBuf {
    run_dir.join(TUI_SENTINEL_FILENAME)
}

/// Write the current process's PID to the sentinel.
///
/// Atomically replaces any prior sentinel via a tmp-and-rename, so
/// concurrent CLI readers never observe a half-written file.
pub fn write(run_dir: &Path) -> Result<(), HarnessError> {
    let path = sentinel_path(run_dir);
    let tmp = path.with_extension("pid.tmp");
    let pid = std::process::id();
    if let Err(e) = fs::write(&tmp, format!("{pid}\n")) {
        let _ = fs::remove_file(&tmp);
        return Err(HarnessError::io(&tmp, e));
    }
    if let Err(e) = fs::rename(&tmp, &path) {
        let _ = fs::remove_file(&tmp);
        return Err(HarnessError::io(&path, e));
    }
    Ok(())
}

/// Remove the sentinel, if present. Missing-file is not an error.
pub fn clear(run_dir: &Path) -> Result<(), HarnessError> {
    let path = sentinel_path(run_dir);
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(HarnessError::io(&path, e)),
    }
}

/// Return the PID currently recorded in the sentinel, when one exists
/// and the recorded process is still alive.
///
/// Stale sentinels (recorded process exited) and corrupt sentinels
/// (non-numeric contents) resolve to `Ok(None)`. The stale case also
/// removes the file as a courtesy, so the next caller sees a clean
/// state. The cleanup is best-effort: a remove failure is logged and
/// otherwise ignored.
pub fn read_live_pid(run_dir: &Path) -> Result<Option<u32>, HarnessError> {
    let path = sentinel_path(run_dir);
    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(HarnessError::io(&path, e)),
    };
    let trimmed = content.trim();
    let Ok(pid) = trimmed.parse::<u32>() else {
        tracing::warn!(
            path = %path.display(),
            content = %trimmed,
            "TUI sentinel has non-numeric contents; treating as stale",
        );
        let _ = fs::remove_file(&path);
        return Ok(None);
    };
    if process_is_alive(pid) {
        Ok(Some(pid))
    } else {
        tracing::debug!(pid, "removing stale TUI sentinel for non-running process");
        let _ = fs::remove_file(&path);
        Ok(None)
    }
}

/// Cross-platform "is this PID currently alive?" probe.
///
/// On Unix, `kill(pid, 0)` returns 0 when the process exists (and the
/// caller has permission to signal it) and `-1` with `ESRCH` when the
/// PID is not in use. We treat any other errno (including `EPERM`,
/// which means "the process exists but you can't signal it") as
/// "alive" — being conservative is safer than silently overwriting a
/// live runner's `run.toml`.
///
/// On non-Unix platforms we currently assume the sentinel is live
/// when present; the TUI rotation flow has not been validated on
/// Windows. Refining this is tracked as a follow-up.
#[cfg(unix)]
#[expect(
    unsafe_code,
    reason = "FFI call to libc::kill(pid, 0) for a permission-only liveness check"
)]
fn process_is_alive(pid: u32) -> bool {
    // Negative or zero PIDs are not valid signal targets; treat as
    // "not alive" so the sentinel cleanup path runs.
    if pid == 0 {
        return false;
    }
    // `pid_t` is a signed integer; PIDs above `i32::MAX` are
    // impossible on every Unix in practice, but guard against
    // wrap-around on the cast just in case.
    let Ok(raw_pid) = libc::pid_t::try_from(pid) else {
        return false;
    };
    // SAFETY: `kill` is async-signal-safe and the only precondition
    // is a valid signal number; signal `0` means "permission check
    // only" per POSIX, so we never deliver a signal to the target
    // process. `kill` does not access any memory through pointers we
    // supply.
    let rc = unsafe { libc::kill(raw_pid, 0) };
    if rc == 0 {
        return true;
    }
    // `last_os_error` reads thread-local errno; it is the canonical
    // way to inspect errno after a failing libc call from Rust.
    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
    errno != libc::ESRCH
}

#[cfg(not(unix))]
fn process_is_alive(_pid: u32) -> bool {
    // Conservative fallback: assume alive when present so we never
    // clobber a live TUI's run.toml on platforms we have not yet
    // validated.
    true
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_live_pid_returns_self_pid() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        write(tmp.path()).unwrap_or_else(|e| panic!("{e}"));
        let read = read_live_pid(tmp.path()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(read, Some(std::process::id()));
    }

    #[test]
    fn missing_sentinel_resolves_to_none() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let read = read_live_pid(tmp.path()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(read, None);
    }

    #[test]
    fn clear_removes_sentinel_and_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        write(tmp.path()).unwrap_or_else(|e| panic!("{e}"));
        clear(tmp.path()).unwrap_or_else(|e| panic!("{e}"));
        assert!(!sentinel_path(tmp.path()).exists());
        // Second call is a no-op.
        clear(tmp.path()).unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn corrupt_sentinel_is_treated_as_stale() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(sentinel_path(tmp.path()), "not a pid\n").unwrap_or_else(|e| panic!("{e}"));
        let read = read_live_pid(tmp.path()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(read, None);
        // Corrupt sentinel was cleaned up.
        assert!(!sentinel_path(tmp.path()).exists());
    }

    #[cfg(unix)]
    #[test]
    fn stale_sentinel_for_dead_pid_resolves_to_none() {
        // PID 1 is init / launchd and is always alive on a running
        // system, so it is not safe for a "definitely dead" check.
        // Spawn a short-lived child and capture its PID; once
        // `wait()` returns we know that PID is no longer in use.
        let mut child = std::process::Command::new("true")
            .spawn()
            .unwrap_or_else(|e| panic!("{e}"));
        let pid = child.id();
        let _ = child.wait();
        // Give the OS a moment to clean up the zombie; on most
        // systems `wait` is enough.
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(sentinel_path(tmp.path()), format!("{pid}\n"))
            .unwrap_or_else(|e| panic!("{e}"));
        let read = read_live_pid(tmp.path()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            read, None,
            "expected stale PID {pid} to be reported as not alive"
        );
        assert!(
            !sentinel_path(tmp.path()).exists(),
            "stale sentinel should have been cleaned up",
        );
    }
}
