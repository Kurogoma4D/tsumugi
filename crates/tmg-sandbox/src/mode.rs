//! Sandbox operating modes controlling filesystem access levels.

use serde::{Deserialize, Serialize};

/// Controls the level of filesystem access granted to sandboxed processes.
///
/// The mode determines which Landlock rules are applied:
///
/// - [`ReadOnly`](SandboxMode::ReadOnly) -- workspace directory is read-only,
///   system paths are read-only, everything else is denied.
/// - [`WorkspaceWrite`](SandboxMode::WorkspaceWrite) -- workspace directory is
///   read+write (default), system paths are read-only, everything else is denied.
/// - [`Full`](SandboxMode::Full) -- no filesystem restrictions applied (escape
///   hatch for trusted operations).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxMode {
    /// Workspace is read-only. No writes are permitted anywhere.
    ReadOnly,

    /// Workspace is read+write (default). System paths are read-only.
    #[default]
    WorkspaceWrite,

    /// No filesystem restrictions. Used for trusted operations or
    /// platforms where sandboxing is unavailable.
    Full,
}

impl SandboxMode {
    /// Returns `true` if writes to the workspace directory are allowed.
    pub fn allows_workspace_write(&self) -> bool {
        matches!(self, Self::WorkspaceWrite | Self::Full)
    }

    /// Returns `true` if arbitrary filesystem access is allowed.
    pub fn is_unrestricted(&self) -> bool {
        matches!(self, Self::Full)
    }

    /// Returns `true` if any filesystem restrictions are active.
    pub fn is_restricted(&self) -> bool {
        !self.is_unrestricted()
    }
}

impl std::fmt::Display for SandboxMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadOnly => write!(f, "read_only"),
            Self::WorkspaceWrite => write!(f, "workspace_write"),
            Self::Full => write!(f, "full"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_workspace_write() {
        assert_eq!(SandboxMode::default(), SandboxMode::WorkspaceWrite);
    }

    #[test]
    fn read_only_denies_writes() {
        let mode = SandboxMode::ReadOnly;
        assert!(!mode.allows_workspace_write());
        assert!(mode.is_restricted());
        assert!(!mode.is_unrestricted());
    }

    #[test]
    fn workspace_write_allows_writes() {
        let mode = SandboxMode::WorkspaceWrite;
        assert!(mode.allows_workspace_write());
        assert!(mode.is_restricted());
        assert!(!mode.is_unrestricted());
    }

    #[test]
    fn full_mode_unrestricted() {
        let mode = SandboxMode::Full;
        assert!(mode.allows_workspace_write());
        assert!(!mode.is_restricted());
        assert!(mode.is_unrestricted());
    }

    #[test]
    fn display_matches_serde() {
        assert_eq!(SandboxMode::ReadOnly.to_string(), "read_only");
        assert_eq!(SandboxMode::WorkspaceWrite.to_string(), "workspace_write");
        assert_eq!(SandboxMode::Full.to_string(), "full");
    }

    #[test]
    fn serde_roundtrip() {
        let modes = [
            SandboxMode::ReadOnly,
            SandboxMode::WorkspaceWrite,
            SandboxMode::Full,
        ];
        for mode in &modes {
            let json =
                serde_json::to_string(mode).unwrap_or_else(|_| String::from("serialize failed"));
            let deserialized: SandboxMode =
                serde_json::from_str(&json).unwrap_or(SandboxMode::Full);
            assert_eq!(*mode, deserialized);
        }
    }
}
