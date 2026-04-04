//! Error types for the TUI crate.

/// Errors produced by the TUI layer.
#[derive(Debug, thiserror::Error)]
pub enum TuiError {
    /// An I/O error (e.g. terminal setup/teardown).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// An error originating from the core orchestration layer.
    #[error("core error: {0}")]
    Core(#[from] tmg_core::CoreError),

    /// The TUI session was cancelled (e.g. Ctrl+C / `CancellationToken`).
    #[error("session cancelled")]
    Cancelled,
}
