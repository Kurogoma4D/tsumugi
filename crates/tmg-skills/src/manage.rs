//! `skill_manage` tool: programmatic CRUD over `.tsumugi/skills/`.
//!
//! This is the Tool implementation behind `skill_critic`'s autonomous
//! skill creation flow (issue #54) and behind explicit `/skill capture`
//! invocations. It deliberately *only* writes inside
//! `<workspace>/.tsumugi/skills/` so a misbehaving model cannot reach
//! into `~/.config/tsumugi/skills/` and clobber another project's
//! library.
//!
//! ## Actions
//!
//! - `create` — write a new `<name>/SKILL.md` (errors if it already
//!   exists).
//! - `patch` — strict search/replace inside SKILL.md, like the
//!   `file_patch` tool.
//! - `edit` — full-content replacement; bumps the semantic version's
//!   minor component.
//! - `remove` — delete the entire skill directory.
//! - `add_file` — write a helper file inside `scripts/` or
//!   `references/`.
//! - `remove_file` — delete a helper file inside `scripts/` or
//!   `references/`.
//!
//! After every successful write, `INDEX.md` (`.tsumugi/skills/INDEX.md`)
//! is regenerated so the cache stays in sync with the on-disk skills.
//!
//! ## Sandbox
//!
//! Every filesystem operation is gated through
//! [`SandboxContext::check_write_access`] (writes) or
//! [`SandboxContext::check_path_access`] (reads). The active mode also
//! decides whether the tool runs at all — `ReadOnly` is a hard refusal.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use chrono::Utc;
use tmg_sandbox::SandboxContext;
use tmg_tools::error::ToolError;
use tmg_tools::types::{Tool, ToolResult};

use crate::parse::parse_skill_md;
use crate::types::{InvocationPolicy, Provenance, SkillFrontmatter};

/// Maximum length of the `name` parameter (skill directory name).
const MAX_NAME_LEN: usize = 64;

/// Maximum length of the `path` parameter (auxiliary file path inside
/// the skill directory).
const MAX_PATH_LEN: usize = 256;

/// Subdirectories permitted as the parent of an auxiliary file.
const ALLOWED_AUX_SUBDIRS: &[&str] = &["scripts", "references"];

/// The `skill_manage` tool. Mutates `.tsumugi/skills/` under the
/// active workspace.
pub struct SkillManageTool;

impl Tool for SkillManageTool {
    fn name(&self) -> &'static str {
        "skill_manage"
    }

    fn description(&self) -> &'static str {
        "Create, update, or remove a project-local skill under \
         .tsumugi/skills/. Actions: create, patch, edit, remove, \
         add_file, remove_file. Writes ONLY inside the workspace's \
         .tsumugi/skills/ tree; never touches the global config."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "create",
                        "patch",
                        "edit",
                        "remove",
                        "add_file",
                        "remove_file"
                    ],
                },
                "name": { "type": "string" },
                "description": { "type": "string" },
                "invocation": {
                    "type": "string",
                    "enum": ["auto", "explicit_only"],
                },
                "old_str": { "type": "string" },
                "new_str": { "type": "string" },
                "content": { "type": "string" },
                "path": { "type": "string" },
                "provenance": {
                    "type": "string",
                    "enum": ["user", "agent"],
                },
            },
            "required": ["action", "name"],
            "additionalProperties": false
        })
    }

    fn execute<'a>(
        &'a self,
        params: serde_json::Value,
        ctx: &'a SandboxContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + 'a>,
    > {
        Box::pin(self.execute_inner(params, ctx))
    }
}

/// Action discriminator parsed from the JSON `action` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    Create,
    Patch,
    Edit,
    Remove,
    AddFile,
    RemoveFile,
}

impl Action {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "create" => Some(Self::Create),
            "patch" => Some(Self::Patch),
            "edit" => Some(Self::Edit),
            "remove" => Some(Self::Remove),
            "add_file" => Some(Self::AddFile),
            "remove_file" => Some(Self::RemoveFile),
            _ => None,
        }
    }
}

impl SkillManageTool {
    /// Run the tool's inner logic outside the [`Tool`] dispatch path.
    ///
    /// Public so the harness's autonomous skill creation flow
    /// (`SkillsRuntime::apply_verdict`) can invoke `skill_manage`
    /// directly with a `provenance: agent` create payload — the LLM
    /// itself does not need this tool in its registry to be able to
    /// land an auto-generated skill.
    ///
    /// # Errors
    ///
    /// Surfaces every variant of [`ToolError`] the regular
    /// [`Tool::execute`] path can produce.
    pub async fn execute_inner(
        &self,
        params: serde_json::Value,
        ctx: &SandboxContext,
    ) -> Result<ToolResult, ToolError> {
        let action_str = required_str(&params, "action")?;
        let Some(action) = Action::parse(action_str) else {
            return Err(ToolError::invalid_params(format!(
                "unknown action: {action_str:?}; expected one of \
                 create, patch, edit, remove, add_file, remove_file"
            )));
        };

        let name = required_str(&params, "name")?;
        validate_name(name)?;

        let skills_root = ctx.workspace().join(".tsumugi").join("skills");
        let skill_dir = skills_root.join(name);

        // Every action mutates the filesystem (or, for destructive
        // ones, requires confirmation that we *can* mutate). Gate
        // ReadOnly mode explicitly.
        ctx.check_write_access(&skills_root)?;

        match action {
            Action::Create => {
                self.do_create(&params, &skills_root, &skill_dir, name)
                    .await
            }
            Action::Patch => self.do_patch(&params, &skill_dir).await,
            Action::Edit => self.do_edit(&params, &skill_dir, name).await,
            Action::Remove => self.do_remove(&skills_root, &skill_dir, name).await,
            Action::AddFile => self.do_add_file(&params, &skill_dir).await,
            Action::RemoveFile => self.do_remove_file(&params, &skill_dir).await,
        }
    }

    async fn do_create(
        &self,
        params: &serde_json::Value,
        skills_root: &Path,
        skill_dir: &Path,
        name: &str,
    ) -> Result<ToolResult, ToolError> {
        if skill_dir.exists() {
            return Ok(ToolResult::error(format!(
                "skill {name:?} already exists at {}; use action=edit or action=patch to modify",
                skill_dir.display()
            )));
        }

        let description = required_str(params, "description")?;
        let invocation = parse_invocation(params)?;
        let provenance = parse_provenance(params).unwrap_or(Provenance::User);
        let body = optional_str(params, "content").unwrap_or("");

        let now = Utc::now();
        let frontmatter = SkillFrontmatter {
            name: name.to_owned(),
            description: description.to_owned(),
            invocation,
            allowed_tools: None,
            version: Some("0.1.0".to_owned()),
            created_at: Some(now),
            updated_at: Some(now),
            provenance: Some(provenance),
        };

        tokio::fs::create_dir_all(skill_dir).await.map_err(|e| {
            ToolError::io(
                format!("creating skill directory {}", skill_dir.display()),
                e,
            )
        })?;

        let skill_md_path = skill_dir.join("SKILL.md");
        let document = serialize_skill_md(&frontmatter, body)?;
        tokio::fs::write(&skill_md_path, document)
            .await
            .map_err(|e| ToolError::io(format!("writing {}", skill_md_path.display()), e))?;

        regenerate_index(skills_root).await?;

        Ok(ToolResult::success(format!(
            "Created skill {name:?} at {}",
            skill_md_path.display()
        )))
    }

    async fn do_patch(
        &self,
        params: &serde_json::Value,
        skill_dir: &Path,
    ) -> Result<ToolResult, ToolError> {
        let old_str = required_str(params, "old_str")?;
        let new_str = required_str(params, "new_str")?;
        let skill_md = skill_dir.join("SKILL.md");

        let content = read_existing_skill(&skill_md).await?;

        let match_count = content.matches(old_str).count();
        if match_count == 0 {
            return Ok(ToolResult::error(format!(
                "old_str not found in {}",
                skill_md.display()
            )));
        }
        if match_count > 1 {
            return Ok(ToolResult::error(format!(
                "old_str matches {match_count} locations in {}; it must match exactly once",
                skill_md.display()
            )));
        }

        let new_content = content.replacen(old_str, new_str, 1);
        let (mut frontmatter, body) = split_skill_md(&new_content, &skill_md)?;
        frontmatter.updated_at = Some(Utc::now());

        let serialized = serialize_skill_md(&frontmatter, &body)?;
        tokio::fs::write(&skill_md, serialized)
            .await
            .map_err(|e| ToolError::io(format!("writing {}", skill_md.display()), e))?;

        regenerate_index(parent_or_self(skill_dir)).await?;

        Ok(ToolResult::success(format!(
            "Patched {}",
            skill_md.display()
        )))
    }

    async fn do_edit(
        &self,
        params: &serde_json::Value,
        skill_dir: &Path,
        name: &str,
    ) -> Result<ToolResult, ToolError> {
        let new_body = required_str(params, "content")?;
        let skill_md = skill_dir.join("SKILL.md");

        let existing = read_existing_skill(&skill_md).await?;
        let (mut frontmatter, _body) = split_skill_md(&existing, &skill_md)?;

        // Optional metadata refresh on edit. An empty `description` is
        // rejected (consistent with `create`, where `description` is
        // required and non-empty); callers who want to clear the field
        // must use `patch` to delete the line explicitly.
        if let Some(desc) = optional_str(params, "description") {
            if desc.trim().is_empty() {
                return Err(ToolError::invalid_params(
                    "description must not be empty; omit the field to keep the existing value",
                ));
            }
            frontmatter.description = desc.to_owned();
        }
        if let Ok(invocation) = parse_invocation(params) {
            // Only overwrite when the caller actually supplied the
            // field; the helper returns Auto for an empty string,
            // which we don't want to silently apply.
            if optional_str(params, "invocation").is_some() {
                frontmatter.invocation = invocation;
            }
        }
        frontmatter.version = Some(bump_version(frontmatter.version.as_deref()));
        frontmatter.updated_at = Some(Utc::now());

        let document = serialize_skill_md(&frontmatter, new_body)?;
        tokio::fs::write(&skill_md, document)
            .await
            .map_err(|e| ToolError::io(format!("writing {}", skill_md.display()), e))?;

        regenerate_index(parent_or_self(skill_dir)).await?;

        Ok(ToolResult::success(format!(
            "Edited skill {name:?} (version -> {})",
            frontmatter.version.as_deref().unwrap_or("?"),
        )))
    }

    async fn do_remove(
        &self,
        skills_root: &Path,
        skill_dir: &Path,
        name: &str,
    ) -> Result<ToolResult, ToolError> {
        if !skill_dir.exists() {
            return Ok(ToolResult::error(format!(
                "skill {name:?} not found at {}",
                skill_dir.display()
            )));
        }

        // Belt-and-braces: ensure the resolved skill_dir is still
        // inside skills_root (rules out any path-traversal in `name`
        // that slipped past `validate_name`).
        if !skill_dir.starts_with(skills_root) {
            return Err(ToolError::invalid_params(
                "resolved skill path escapes .tsumugi/skills/",
            ));
        }

        tokio::fs::remove_dir_all(skill_dir)
            .await
            .map_err(|e| ToolError::io(format!("removing {}", skill_dir.display()), e))?;

        regenerate_index(skills_root).await?;

        Ok(ToolResult::success(format!("Removed skill {name:?}")))
    }

    async fn do_add_file(
        &self,
        params: &serde_json::Value,
        skill_dir: &Path,
    ) -> Result<ToolResult, ToolError> {
        let rel = required_str(params, "path")?;
        let content = required_str(params, "content")?;
        let target = resolve_aux_path(skill_dir, rel)?;

        if let Some(parent) = target.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                ToolError::io(format!("creating parent dir {}", parent.display()), e)
            })?;
        }

        tokio::fs::write(&target, content)
            .await
            .map_err(|e| ToolError::io(format!("writing {}", target.display()), e))?;

        Ok(ToolResult::success(format!(
            "Wrote auxiliary file {}",
            target.display()
        )))
    }

    async fn do_remove_file(
        &self,
        params: &serde_json::Value,
        skill_dir: &Path,
    ) -> Result<ToolResult, ToolError> {
        let rel = required_str(params, "path")?;
        let target = resolve_aux_path(skill_dir, rel)?;

        if !target.exists() {
            return Ok(ToolResult::error(format!(
                "auxiliary file not found: {}",
                target.display()
            )));
        }

        tokio::fs::remove_file(&target)
            .await
            .map_err(|e| ToolError::io(format!("removing {}", target.display()), e))?;

        Ok(ToolResult::success(format!(
            "Removed auxiliary file {}",
            target.display()
        )))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Required string parameter accessor.
fn required_str<'a>(params: &'a serde_json::Value, key: &str) -> Result<&'a str, ToolError> {
    params
        .get(key)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ToolError::invalid_params(format!("missing required parameter: {key}")))
}

/// Optional string parameter accessor.
fn optional_str<'a>(params: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    params.get(key).and_then(serde_json::Value::as_str)
}

fn parse_invocation(params: &serde_json::Value) -> Result<InvocationPolicy, ToolError> {
    match optional_str(params, "invocation") {
        Some("auto") | None => Ok(InvocationPolicy::Auto),
        Some("explicit_only") => Ok(InvocationPolicy::ExplicitOnly),
        Some(other) => Err(ToolError::invalid_params(format!(
            "invalid invocation: {other:?}; expected 'auto' or 'explicit_only'"
        ))),
    }
}

fn parse_provenance(params: &serde_json::Value) -> Option<Provenance> {
    match optional_str(params, "provenance") {
        Some("agent") => Some(Provenance::Agent),
        Some("user") => Some(Provenance::User),
        _ => None,
    }
}

/// Validate a skill name: non-empty, no path separators, no traversal,
/// no whitespace, length-bounded.
fn validate_name(name: &str) -> Result<(), ToolError> {
    if name.is_empty() {
        return Err(ToolError::invalid_params("skill name must not be empty"));
    }
    if name.len() > MAX_NAME_LEN {
        return Err(ToolError::invalid_params(format!(
            "skill name is too long ({} > {MAX_NAME_LEN} bytes)",
            name.len()
        )));
    }
    if name.contains('/') || name.contains('\\') {
        return Err(ToolError::invalid_params(
            "skill name must not contain path separators",
        ));
    }
    if name == "." || name == ".." || name.starts_with('.') {
        return Err(ToolError::invalid_params(
            "skill name must not start with '.'",
        ));
    }
    if name.chars().any(char::is_whitespace) {
        return Err(ToolError::invalid_params(
            "skill name must not contain whitespace",
        ));
    }
    Ok(())
}

/// Resolve an auxiliary path inside the skill directory. The path must:
///
/// - Be a relative path (no absolute roots, no `..` segments)
/// - Start with one of [`ALLOWED_AUX_SUBDIRS`] (`scripts/` or
///   `references/`)
/// - Be reasonably short
fn resolve_aux_path(skill_dir: &Path, rel: &str) -> Result<PathBuf, ToolError> {
    if rel.is_empty() {
        return Err(ToolError::invalid_params("path must not be empty"));
    }
    if rel.len() > MAX_PATH_LEN {
        return Err(ToolError::invalid_params(format!(
            "path is too long ({} > {MAX_PATH_LEN} bytes)",
            rel.len()
        )));
    }

    let candidate = Path::new(rel);
    if candidate.is_absolute() {
        return Err(ToolError::invalid_params(
            "path must be relative to the skill directory",
        ));
    }

    // Walk components: reject any `..` and any odd component.
    let mut components = candidate.components();
    let Some(first) = components.next() else {
        return Err(ToolError::invalid_params("path must not be empty"));
    };
    let first_str = first
        .as_os_str()
        .to_str()
        .ok_or_else(|| ToolError::invalid_params("path is not valid UTF-8"))?;
    if !ALLOWED_AUX_SUBDIRS.contains(&first_str) {
        return Err(ToolError::invalid_params(format!(
            "path must start with one of {ALLOWED_AUX_SUBDIRS:?}"
        )));
    }

    for component in components {
        match component {
            std::path::Component::Normal(_) => {}
            _ => {
                return Err(ToolError::invalid_params(
                    "path must contain only normal components (no '..' / absolute roots)",
                ));
            }
        }
    }

    Ok(skill_dir.join(rel))
}

async fn read_existing_skill(skill_md: &Path) -> Result<String, ToolError> {
    tokio::fs::read_to_string(skill_md)
        .await
        .map_err(|e| ToolError::io(format!("reading {}", skill_md.display()), e))
}

fn split_skill_md(content: &str, path: &Path) -> Result<(SkillFrontmatter, String), ToolError> {
    parse_skill_md(content, &path.display().to_string())
        .map_err(|e| ToolError::io(format!("parsing {}", path.display()), to_io_error(&e)))
}

/// Convert a [`crate::error::SkillError`] into a fake `io::Error` so it
/// can flow through [`ToolError::io`]. We deliberately preserve the
/// `Display` text in the message rather than carrying the original
/// error, since `ToolError` does not have a dedicated variant for
/// frontmatter parse failures.
fn to_io_error(err: &crate::error::SkillError) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, err.to_string())
}

fn serialize_skill_md(frontmatter: &SkillFrontmatter, body: &str) -> Result<String, ToolError> {
    let yaml = serde_yml::to_string(frontmatter).map_err(|e| {
        ToolError::io(
            "serializing SKILL.md frontmatter",
            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
        )
    })?;
    Ok(format!("---\n{yaml}---\n\n{body}"))
}

/// Bump the minor component of a `MAJOR.MINOR.PATCH` version. Falls
/// back to `0.1.0` when the input is missing or unparseable.
fn bump_version(current: Option<&str>) -> String {
    let Some(s) = current else {
        return "0.1.0".to_owned();
    };
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 3 {
        return "0.1.0".to_owned();
    }
    let major: u32 = parts[0].parse().unwrap_or(0);
    let minor: u32 = parts[1].parse().unwrap_or(0);
    // patch component is intentionally reset to 0 on a minor bump.
    format!("{major}.{}.0", minor + 1)
}

fn parent_or_self(skill_dir: &Path) -> &Path {
    skill_dir.parent().unwrap_or(skill_dir)
}

// ---------------------------------------------------------------------------
// INDEX.md regeneration
// ---------------------------------------------------------------------------

/// Regenerate `<skills_root>/INDEX.md` from the current state of the
/// directory.
///
/// The file mirrors `MEMORY.md`'s shape so the user can scan all skills
/// (and follow links to each `SKILL.md`) without spinning up the agent.
/// It is purely a cache — `discovery::discover_skills` does not read it.
pub async fn regenerate_index(skills_root: &Path) -> Result<(), ToolError> {
    let mut entries: Vec<(String, String, String)> = Vec::new();

    let mut read_dir = match tokio::fs::read_dir(skills_root).await {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Nothing to index. Make sure we don't leave a stale
            // INDEX.md behind.
            let index_path = skills_root.join("INDEX.md");
            if index_path.exists() {
                tokio::fs::remove_file(&index_path).await.map_err(|e| {
                    ToolError::io(format!("removing stale {}", index_path.display()), e)
                })?;
            }
            return Ok(());
        }
        Err(e) => {
            return Err(ToolError::io(
                format!("reading {}", skills_root.display()),
                e,
            ));
        }
    };

    while let Some(entry) = read_dir
        .next_entry()
        .await
        .map_err(|e| ToolError::io("reading skills directory entry", e))?
    {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let skill_md = path.join("SKILL.md");
        let content = match tokio::fs::read_to_string(&skill_md).await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(
                    path = %skill_md.display(),
                    error = %e,
                    "skipping skill in INDEX regeneration: SKILL.md unreadable"
                );
                continue;
            }
        };
        let display = skill_md.display().to_string();
        let (frontmatter, _body) = match parse_skill_md(&content, &display) {
            Ok(parsed) => parsed,
            Err(e) => {
                tracing::warn!(
                    path = %skill_md.display(),
                    error = %e,
                    "skipping skill in INDEX regeneration: SKILL.md frontmatter parse failed"
                );
                continue;
            }
        };
        let dir_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&frontmatter.name)
            .to_owned();
        entries.push((dir_name, frontmatter.name, frontmatter.description));
    }

    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut output = String::from("# Skills Index\n\n");
    if entries.is_empty() {
        output.push_str("(no skills installed)\n");
    } else {
        for (dir_name, _name, description) in entries {
            let _ = writeln!(
                output,
                "- [{dir_name}]({dir_name}/SKILL.md) — {description}"
            );
        }
    }

    let index_path = skills_root.join("INDEX.md");
    tokio::fs::write(&index_path, output)
        .await
        .map_err(|e| ToolError::io(format!("writing {}", index_path.display()), e))?;

    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests assert with unwrap for clarity; the workspace policy denies them in production code"
)]
mod tests {
    use super::*;

    fn make_ctx(workspace: &Path) -> SandboxContext {
        use tmg_sandbox::{SandboxConfig, SandboxMode};
        let config = SandboxConfig::new(workspace).with_mode(SandboxMode::Full);
        SandboxContext::new(config)
    }

    #[tokio::test]
    async fn create_writes_skill_and_index() {
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_ctx(tmp.path());
        let tool = SkillManageTool;

        let result = tool
            .execute_inner(
                serde_json::json!({
                    "action": "create",
                    "name": "deploy-rust",
                    "description": "Publish a Rust crate",
                    "invocation": "auto",
                    "provenance": "agent",
                    "content": "Steps:\n1. cargo publish\n",
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!result.is_error, "{}", result.output);

        let skill_md = tmp
            .path()
            .join(".tsumugi")
            .join("skills")
            .join("deploy-rust")
            .join("SKILL.md");
        let body = std::fs::read_to_string(&skill_md).unwrap();
        assert!(body.contains("name: deploy-rust"));
        assert!(body.contains("provenance: agent"));
        // serde_yml may emit the version as `'0.1.0'` (quoted) because
        // the string looks like a non-canonical YAML scalar; tolerate
        // both forms.
        assert!(body.contains("0.1.0"), "expected version 0.1.0 in:\n{body}");
        assert!(body.contains("Steps:"));

        // INDEX.md is regenerated.
        let index =
            std::fs::read_to_string(tmp.path().join(".tsumugi").join("skills").join("INDEX.md"))
                .unwrap();
        assert!(index.contains("[deploy-rust](deploy-rust/SKILL.md)"));
        assert!(index.contains("Publish a Rust crate"));
    }

    #[tokio::test]
    async fn create_rejects_existing_skill() {
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_ctx(tmp.path());
        let tool = SkillManageTool;

        for _ in 0..2 {
            let _ = tool
                .execute_inner(
                    serde_json::json!({
                        "action": "create",
                        "name": "dup",
                        "description": "x",
                    }),
                    &ctx,
                )
                .await
                .unwrap();
        }

        let result = tool
            .execute_inner(
                serde_json::json!({
                    "action": "create",
                    "name": "dup",
                    "description": "x",
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.output.contains("already exists"));
    }

    #[tokio::test]
    async fn patch_replaces_substring_and_bumps_updated_at() {
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_ctx(tmp.path());
        let tool = SkillManageTool;

        tool.execute_inner(
            serde_json::json!({
                "action": "create",
                "name": "patcher",
                "description": "before patch",
                "content": "alpha\nbeta\n",
            }),
            &ctx,
        )
        .await
        .unwrap();

        let result = tool
            .execute_inner(
                serde_json::json!({
                    "action": "patch",
                    "name": "patcher",
                    "old_str": "alpha",
                    "new_str": "ALPHA",
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!result.is_error, "{}", result.output);

        let skill_md = tmp
            .path()
            .join(".tsumugi")
            .join("skills")
            .join("patcher")
            .join("SKILL.md");
        let body = std::fs::read_to_string(&skill_md).unwrap();
        assert!(body.contains("ALPHA"));
        assert!(body.contains("updated_at:"));
    }

    #[tokio::test]
    async fn patch_errors_on_no_match() {
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_ctx(tmp.path());
        let tool = SkillManageTool;

        tool.execute_inner(
            serde_json::json!({
                "action": "create",
                "name": "patch-nm",
                "description": "x",
                "content": "alpha\n",
            }),
            &ctx,
        )
        .await
        .unwrap();

        let result = tool
            .execute_inner(
                serde_json::json!({
                    "action": "patch",
                    "name": "patch-nm",
                    "old_str": "missing",
                    "new_str": "found",
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.output.contains("not found"));
    }

    #[tokio::test]
    async fn edit_bumps_minor_version() {
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_ctx(tmp.path());
        let tool = SkillManageTool;

        tool.execute_inner(
            serde_json::json!({
                "action": "create",
                "name": "ver",
                "description": "v",
                "content": "before\n",
            }),
            &ctx,
        )
        .await
        .unwrap();

        let result = tool
            .execute_inner(
                serde_json::json!({
                    "action": "edit",
                    "name": "ver",
                    "content": "after\n",
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!result.is_error, "{}", result.output);
        assert!(result.output.contains("0.2.0"));

        let body = std::fs::read_to_string(
            tmp.path()
                .join(".tsumugi")
                .join("skills")
                .join("ver")
                .join("SKILL.md"),
        )
        .unwrap();
        assert!(body.contains("after"));
        // Tolerate quoted vs. unquoted YAML scalar emission.
        assert!(body.contains("0.2.0"), "expected version 0.2.0 in:\n{body}");
    }

    #[tokio::test]
    async fn remove_deletes_directory_and_updates_index() {
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_ctx(tmp.path());
        let tool = SkillManageTool;

        tool.execute_inner(
            serde_json::json!({
                "action": "create",
                "name": "rm-me",
                "description": "x",
            }),
            &ctx,
        )
        .await
        .unwrap();

        let result = tool
            .execute_inner(
                serde_json::json!({
                    "action": "remove",
                    "name": "rm-me",
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!result.is_error, "{}", result.output);

        let skill_dir = tmp.path().join(".tsumugi").join("skills").join("rm-me");
        assert!(!skill_dir.exists());

        let index =
            std::fs::read_to_string(tmp.path().join(".tsumugi").join("skills").join("INDEX.md"))
                .unwrap();
        assert!(!index.contains("rm-me"));
    }

    #[tokio::test]
    async fn add_and_remove_aux_file() {
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_ctx(tmp.path());
        let tool = SkillManageTool;

        tool.execute_inner(
            serde_json::json!({
                "action": "create",
                "name": "aux",
                "description": "x",
            }),
            &ctx,
        )
        .await
        .unwrap();

        let result = tool
            .execute_inner(
                serde_json::json!({
                    "action": "add_file",
                    "name": "aux",
                    "path": "scripts/run.sh",
                    "content": "#!/bin/bash\necho hi\n",
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!result.is_error, "{}", result.output);

        let target = tmp
            .path()
            .join(".tsumugi")
            .join("skills")
            .join("aux")
            .join("scripts")
            .join("run.sh");
        assert!(target.exists());

        let result = tool
            .execute_inner(
                serde_json::json!({
                    "action": "remove_file",
                    "name": "aux",
                    "path": "scripts/run.sh",
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!result.is_error, "{}", result.output);
        assert!(!target.exists());
    }

    #[tokio::test]
    async fn add_file_rejects_path_traversal() {
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_ctx(tmp.path());
        let tool = SkillManageTool;

        tool.execute_inner(
            serde_json::json!({
                "action": "create",
                "name": "trav",
                "description": "x",
            }),
            &ctx,
        )
        .await
        .unwrap();

        let result = tool
            .execute_inner(
                serde_json::json!({
                    "action": "add_file",
                    "name": "trav",
                    "path": "scripts/../../escape.sh",
                    "content": "x",
                }),
                &ctx,
            )
            .await;
        assert!(result.is_err(), "expected invalid_params error");
    }

    #[tokio::test]
    async fn add_file_rejects_unknown_subdir() {
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_ctx(tmp.path());
        let tool = SkillManageTool;

        tool.execute_inner(
            serde_json::json!({
                "action": "create",
                "name": "wrongdir",
                "description": "x",
            }),
            &ctx,
        )
        .await
        .unwrap();

        let result = tool
            .execute_inner(
                serde_json::json!({
                    "action": "add_file",
                    "name": "wrongdir",
                    "path": "secrets/leak.txt",
                    "content": "x",
                }),
                &ctx,
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn name_traversal_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_ctx(tmp.path());
        let tool = SkillManageTool;

        let result = tool
            .execute_inner(
                serde_json::json!({
                    "action": "create",
                    "name": "../escape",
                    "description": "x",
                }),
                &ctx,
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn read_only_sandbox_blocks_create() {
        use tmg_sandbox::{SandboxConfig, SandboxMode};
        let tmp = tempfile::tempdir().unwrap();
        let config = SandboxConfig::new(tmp.path()).with_mode(SandboxMode::ReadOnly);
        let ctx = SandboxContext::new(config);
        let tool = SkillManageTool;

        let result = tool
            .execute_inner(
                serde_json::json!({
                    "action": "create",
                    "name": "blocked",
                    "description": "x",
                }),
                &ctx,
            )
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn bump_version_minor() {
        assert_eq!(bump_version(Some("0.1.0")), "0.2.0");
        assert_eq!(bump_version(Some("1.4.7")), "1.5.0");
        assert_eq!(bump_version(None), "0.1.0");
        assert_eq!(bump_version(Some("garbage")), "0.1.0");
    }

    #[test]
    fn validate_name_rules() {
        assert!(validate_name("good-skill").is_ok());
        assert!(validate_name("good_123").is_ok());
        assert!(validate_name("").is_err());
        assert!(validate_name("../bad").is_err());
        assert!(validate_name("nested/foo").is_err());
        assert!(validate_name(".hidden").is_err());
        assert!(validate_name("with space").is_err());
        assert!(validate_name(&"x".repeat(MAX_NAME_LEN + 1)).is_err());
    }

    #[tokio::test]
    async fn regenerate_index_handles_missing_root() {
        let tmp = tempfile::tempdir().unwrap();
        let result = regenerate_index(&tmp.path().join("never-exists")).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn agentskills_io_compat_skill_remains_parseable_after_create() {
        // A skill written by `create` must still parse cleanly via the
        // generic `parse_skill_md` path used by the agentskills.io
        // compatibility layer (the same parser is shared).
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_ctx(tmp.path());
        let tool = SkillManageTool;

        tool.execute_inner(
            serde_json::json!({
                "action": "create",
                "name": "compat",
                "description": "compat test",
                "content": "Body content.\n",
            }),
            &ctx,
        )
        .await
        .unwrap();

        let path = tmp
            .path()
            .join(".tsumugi")
            .join("skills")
            .join("compat")
            .join("SKILL.md");
        let body = std::fs::read_to_string(&path).unwrap();
        let (fm, content) = parse_skill_md(&body, &path.display().to_string()).unwrap();
        assert_eq!(fm.name, "compat");
        assert_eq!(fm.description, "compat test");
        assert_eq!(fm.invocation, InvocationPolicy::Auto);
        // The new fields must be present but the document still parses
        // through the same unified parser used by the .agents/skills
        // compat path.
        assert_eq!(fm.provenance, Some(Provenance::User));
        assert!(content.contains("Body content."));
    }
}
