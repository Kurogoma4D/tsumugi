//! Workflow discovery (SPEC §8.2).
//!
//! Search order (highest priority first):
//! 1. Project-local: `<project_root>/.tsumugi/workflows/`
//! 2. User-global: `~/.config/tsumugi/workflows/`
//! 3. Custom paths from [`WorkflowConfig::discovery_paths`].
//!
//! Workflows are deduplicated by id; higher-priority sources shadow
//! lower-priority ones. Each workflow lives in its own `*.yaml` /
//! `*.yml` file; the file stem is *not* used as the id — the id
//! declared inside the YAML is canonical so renaming the file does not
//! change the lookup key. The file stem is only used for descriptive
//! diagnostics.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::config::WorkflowConfig;
use crate::def::WorkflowMeta;
use crate::error::{Result, WorkflowError};

/// Discover all workflow files reachable from the configured roots.
///
/// Returns the deduplicated list sorted by id (deterministic).
///
/// # Errors
///
/// Returns [`WorkflowError`] if a workflow file exists but cannot be
/// read or parsed enough to extract id/description. Missing
/// directories are tolerated — they simply contribute nothing.
pub async fn discover_workflows(
    project_root: impl AsRef<Path>,
    config: &WorkflowConfig,
) -> Result<Vec<WorkflowMeta>> {
    let project_root = project_root.as_ref();

    // Build the priority-ordered scan list.
    let mut scan_dirs: Vec<PathBuf> = Vec::new();
    scan_dirs.push(project_root.join(".tsumugi").join("workflows"));
    if let Some(config_dir) = dirs::config_dir() {
        scan_dirs.push(config_dir.join("tsumugi").join("workflows"));
    }
    for extra in &config.discovery_paths {
        scan_dirs.push(extra.clone());
    }

    // First-occurrence wins (project > user > custom).
    let mut by_id: BTreeMap<String, WorkflowMeta> = BTreeMap::new();
    for dir in scan_dirs {
        scan_directory(&dir, &mut by_id).await?;
    }

    Ok(by_id.into_values().collect())
}

async fn scan_directory(dir: &Path, by_id: &mut BTreeMap<String, WorkflowMeta>) -> Result<()> {
    let mut entries = match tokio::fs::read_dir(dir).await {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => return Ok(()),
        Err(e) => {
            return Err(WorkflowError::io(
                format!("reading workflow directory {}", dir.display()),
                e,
            ));
        }
    };

    loop {
        let entry = match entries.next_entry().await {
            Ok(Some(e)) => e,
            Ok(None) => return Ok(()),
            Err(e) => {
                return Err(WorkflowError::io("reading workflow dir entry", e));
            }
        };

        let path = entry.path();
        if !is_yaml_file(&path) {
            continue;
        }

        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => {
                return Err(WorkflowError::io(
                    format!("reading workflow {}", path.display()),
                    e,
                ));
            }
        };

        // Extract id + description from a minimal mirror so we don't
        // require the full step list to parse cleanly. Discovery is a
        // listing operation; a broken workflow should still appear via
        // its id (callers will hit the parse error on demand later).
        let head = match serde_yml::from_str::<WorkflowHead>(&content) {
            Ok(h) => h,
            Err(e) => {
                return Err(WorkflowError::YamlParse {
                    path: path.clone(),
                    source: e,
                });
            }
        };

        if head.id.is_empty() {
            return Err(WorkflowError::invalid_workflow(
                path.display().to_string(),
                "workflow `id` must not be empty",
            ));
        }

        if !by_id.contains_key(&head.id) {
            by_id.insert(
                head.id.clone(),
                WorkflowMeta {
                    id: head.id,
                    source_path: path,
                    description: head.description,
                },
            );
        }
    }
}

fn is_yaml_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|s| s.to_str()),
        Some("yaml" | "yml")
    )
}

/// Minimal projection of a workflow YAML used by discovery.
#[derive(Debug, Deserialize)]
struct WorkflowHead {
    id: String,
    #[serde(default)]
    description: Option<String>,
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions")]
mod tests {
    use super::*;

    fn write_workflow(dir: &Path, file: &str, id: &str, desc: &str) {
        std::fs::create_dir_all(dir).unwrap();
        let body = format!("id: {id}\ndescription: \"{desc}\"\nsteps: []\n");
        std::fs::write(dir.join(file), body).unwrap();
    }

    #[tokio::test]
    async fn discover_empty_project() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = WorkflowConfig::default();
        let workflows = discover_workflows(tmp.path(), &cfg).await.unwrap();
        assert!(workflows.is_empty());
    }

    #[tokio::test]
    async fn discover_project_workflow() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join(".tsumugi").join("workflows");
        write_workflow(&dir, "build.yaml", "build", "Build the project");
        let cfg = WorkflowConfig::default();
        let workflows = discover_workflows(tmp.path(), &cfg).await.unwrap();
        assert_eq!(workflows.len(), 1);
        assert_eq!(workflows[0].id, "build");
        assert_eq!(
            workflows[0].description.as_deref(),
            Some("Build the project")
        );
    }

    #[tokio::test]
    async fn discover_priority_project_over_custom() {
        let tmp = tempfile::tempdir().unwrap();
        let proj_dir = tmp.path().join(".tsumugi").join("workflows");
        let custom_dir = tmp.path().join("extra-workflows");
        write_workflow(&proj_dir, "implement.yaml", "implement", "from project");
        write_workflow(&custom_dir, "implement.yaml", "implement", "from custom");

        let cfg = WorkflowConfig {
            discovery_paths: vec![custom_dir.clone()],
            ..WorkflowConfig::default()
        };
        let workflows = discover_workflows(tmp.path(), &cfg).await.unwrap();
        assert_eq!(workflows.len(), 1);
        assert_eq!(workflows[0].id, "implement");
        assert_eq!(workflows[0].description.as_deref(), Some("from project"));
    }

    #[tokio::test]
    async fn discover_skips_non_yaml() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join(".tsumugi").join("workflows");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("README.md"), "not yaml").unwrap();
        write_workflow(&dir, "ok.yaml", "ok", "valid");
        let cfg = WorkflowConfig::default();
        let workflows = discover_workflows(tmp.path(), &cfg).await.unwrap();
        assert_eq!(workflows.len(), 1);
        assert_eq!(workflows[0].id, "ok");
    }

    #[tokio::test]
    async fn discover_includes_yml_extension() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join(".tsumugi").join("workflows");
        write_workflow(&dir, "yml.yml", "yml_wf", "yml ext");
        let cfg = WorkflowConfig::default();
        let workflows = discover_workflows(tmp.path(), &cfg).await.unwrap();
        assert_eq!(workflows.len(), 1);
        assert_eq!(workflows[0].id, "yml_wf");
    }
}
