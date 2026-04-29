//! `tmg trajectory bundle`: pack every trajectory file into one
//! `tar.zst` archive.
//!
//! ## Format choice
//!
//! tar+zstd is the canonical "binary blob" format the ML community
//! consumes (`HuggingFace` `datasets`, Atropos, TRL, torchdata). We
//! prefer `tar` (uncompressed in-archive) wrapped in a zstd stream
//! over `.tar.gz` for the higher ratio on JSONL — JSONL has long
//! repeated key sequences (`{"type":"assistant"`...) that zstd
//! exploits well.
//!
//! ## Layout
//!
//! Inside the archive, every member uses
//! `<run_id>/session_NNN.jsonl` so consumers can group sessions by
//! run without parsing the records.

use std::fs::File;
use std::io::Write as _;
use std::path::Path;

use tar::Builder;

use crate::config::TrajectoryConfig;
use crate::error::TrajectoryError;
use crate::export::list_trajectories;

/// Pack every trajectory under `runs_dir` into a `tar.zst` archive at
/// `out_path`.
///
/// `compression_level` is the zstd quality level (1..=22). The default
/// passed by the CLI is `3`, which matches the `zstd` CLI default.
///
/// # Errors
///
/// Returns [`TrajectoryError::Bundle`] on tar/zstd failure and
/// [`TrajectoryError::Io`] on filesystem failure.
pub fn bundle(
    runs_dir: &Path,
    config: &TrajectoryConfig,
    out_path: &Path,
    compression_level: i32,
) -> Result<u32, TrajectoryError> {
    let entries = list_trajectories(runs_dir, config)?;
    if entries.is_empty() {
        return Err(TrajectoryError::Bundle {
            path: out_path.to_path_buf(),
            message: "no trajectories found to bundle".into(),
        });
    }
    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| TrajectoryError::io(parent, e))?;
    }
    let out_file = File::create(out_path).map_err(|e| TrajectoryError::io(out_path, e))?;
    // The zstd encoder writes a stream into the underlying writer; we
    // wrap it in a tar Builder so the call sequence is
    // `tar::append → zstd::write → fs::write`. Calling
    // `Encoder::finish()` is required to emit the trailing zstd frame.
    let encoder = zstd::stream::write::Encoder::new(out_file, compression_level)
        .map_err(|e| TrajectoryError::bundle(out_path, format!("creating zstd encoder: {e}")))?;
    let mut tar = Builder::new(encoder);
    let mut count: u32 = 0;
    for entry in &entries {
        let archive_name = format!("{}/session_{:03}.jsonl", entry.run_id, entry.session_num);
        tar.append_path_with_name(&entry.path, &archive_name)
            .map_err(|e| {
                TrajectoryError::bundle(
                    &entry.path,
                    format!("appending to tar as {archive_name}: {e}"),
                )
            })?;
        count = count.saturating_add(1);
    }
    let encoder = tar
        .into_inner()
        .map_err(|e| TrajectoryError::bundle(out_path, format!("finalising tar: {e}")))?;
    let mut out_file = encoder
        .finish()
        .map_err(|e| TrajectoryError::bundle(out_path, format!("finalising zstd: {e}")))?;
    out_file
        .flush()
        .map_err(|e| TrajectoryError::io(out_path, e))?;
    Ok(count)
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;
    use std::io::Read as _;
    use tempfile::TempDir;

    /// End-to-end: create a fake trajectory, bundle, then read the
    /// archive back via the standard zstd + tar APIs and verify the
    /// expected member is present.
    #[test]
    fn bundle_creates_valid_tar_zst() {
        let tmp = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let runs_dir = tmp.path().join("runs");
        let traj_dir = runs_dir.join("abc12345").join("trajectories");
        std::fs::create_dir_all(&traj_dir).unwrap_or_else(|e| panic!("{e}"));
        let session = traj_dir.join("session_001.jsonl");
        std::fs::write(&session, "{\"type\":\"meta\"}\n").unwrap_or_else(|e| panic!("{e}"));

        let out_path = tmp.path().join("traj.tar.zst");
        let count = bundle(&runs_dir, &TrajectoryConfig::default(), &out_path, 3)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(count, 1);
        assert!(out_path.exists());

        let f = File::open(&out_path).unwrap_or_else(|e| panic!("{e}"));
        let dec = zstd::stream::read::Decoder::new(f).unwrap_or_else(|e| panic!("{e}"));
        let mut tar_reader = tar::Archive::new(dec);
        let mut found = false;
        for entry in tar_reader.entries().unwrap_or_else(|e| panic!("{e}")) {
            let mut entry = entry.unwrap_or_else(|e| panic!("{e}"));
            let path_owned = entry.path().unwrap_or_else(|e| panic!("{e}")).into_owned();
            assert_eq!(path_owned.to_string_lossy(), "abc12345/session_001.jsonl");
            let mut body = String::new();
            entry
                .read_to_string(&mut body)
                .unwrap_or_else(|e| panic!("{e}"));
            assert!(body.contains(r#""type":"meta""#), "{body}");
            found = true;
        }
        assert!(found, "expected one archive entry");
    }

    #[test]
    fn bundle_errors_when_no_trajectories() {
        let tmp = TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let runs_dir = tmp.path().join("runs");
        std::fs::create_dir_all(&runs_dir).unwrap_or_else(|e| panic!("{e}"));
        let out_path = tmp.path().join("empty.tar.zst");
        let result = bundle(&runs_dir, &TrajectoryConfig::default(), &out_path, 3);
        assert!(matches!(result, Err(TrajectoryError::Bundle { .. })));
    }
}
