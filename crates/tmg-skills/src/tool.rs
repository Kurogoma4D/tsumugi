//! `use_skill` tool implementation.
//!
//! This tool loads a skill's SKILL.md body and returns it as a tool result,
//! enabling the LLM to incorporate skill instructions into its context.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tmg_sandbox::SandboxContext;
use tmg_tools::error::ToolError;
use tmg_tools::types::{Tool, ToolResult};

use crate::loader::{format_skill_for_tool_result, load_skill};
use crate::types::{SkillMeta, SkillName};

/// The `use_skill` tool: loads a skill's instruction body and returns it.
pub struct UseSkillTool {
    /// Index of available skills by name for lookup.
    skills: Arc<HashMap<SkillName, SkillMeta>>,
}

impl UseSkillTool {
    /// Create a new `UseSkillTool` from discovered skill metadata.
    pub fn new(skills: Vec<SkillMeta>) -> Self {
        let map: HashMap<SkillName, SkillMeta> =
            skills.into_iter().map(|s| (s.name.clone(), s)).collect();
        Self {
            skills: Arc::new(map),
        }
    }
}

impl Tool for UseSkillTool {
    fn name(&self) -> &'static str {
        "use_skill"
    }

    fn description(&self) -> &'static str {
        "Load a skill's instructions into context. Returns the SKILL.md body, \
         associated scripts/references file list, and allowed_tools constraints."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill to load."
                }
            },
            "required": ["skill_name"],
            "additionalProperties": false
        })
    }

    fn execute<'a>(
        &'a self,
        params: serde_json::Value,
        ctx: &'a SandboxContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult, ToolError>> + Send + 'a>> {
        Box::pin(self.execute_inner(params, ctx))
    }
}

impl UseSkillTool {
    async fn execute_inner(
        &self,
        params: serde_json::Value,
        ctx: &SandboxContext,
    ) -> Result<ToolResult, ToolError> {
        let Some(skill_name) = params.get("skill_name").and_then(serde_json::Value::as_str) else {
            return Err(ToolError::invalid_params(
                "missing required parameter: skill_name",
            ));
        };

        let name = SkillName::new(skill_name);
        let Some(meta) = self.skills.get(&name) else {
            let mut available: Vec<&str> = self.skills.keys().map(SkillName::as_str).collect();
            available.sort_unstable();
            return Ok(ToolResult::error(format!(
                "Skill not found: {skill_name}. Available skills: {}",
                available.join(", ")
            )));
        };

        // Loading the skill body is a filesystem read of the
        // discovered SKILL.md path; surface a clear sandbox denial if
        // the active mode forbids the read.
        ctx.check_path_access(meta.path.as_path())?;
        // Defense in depth: `load_skill` also enumerates the
        // `scripts/` and `references/` siblings of SKILL.md. Those
        // subdirectories live under the same parent we just approved,
        // but a maliciously-crafted symlink there could redirect the
        // walk outside the sandbox boundary. Approving the parent
        // directly catches that case (`check_path_access`
        // canonicalises the path, resolving any symlink), and
        // [`load_skill`] itself guards against symlinked files inside
        // the subdirectories by relying on `file_type()`, which does
        // not traverse links.
        if let Some(parent) = meta.path.as_path().parent() {
            ctx.check_path_access(parent)?;
        }

        match load_skill(meta).await {
            Ok(content) => {
                let result = format_skill_for_tool_result(&content);
                Ok(ToolResult::success(result))
            }
            Err(e) => Ok(ToolResult::error(format!(
                "Failed to load skill '{skill_name}': {e}"
            ))),
        }
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions")]
mod tests {
    use super::*;
    use crate::types::{InvocationPolicy, SkillFrontmatter, SkillPath, SkillSource};

    fn make_tool_with_skill(tmp_dir: &std::path::Path) -> UseSkillTool {
        let skill_dir = tmp_dir.join("my-skill");
        std::fs::create_dir_all(&skill_dir).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: my-skill\ndescription: A test\n---\n\nDo the thing.\n",
        )
        .unwrap_or_else(|e| panic!("{e}"));

        let meta = SkillMeta {
            name: SkillName::new("my-skill"),
            frontmatter: SkillFrontmatter {
                name: "my-skill".to_owned(),
                description: "A test".to_owned(),
                invocation: InvocationPolicy::Auto,
                allowed_tools: None,
                version: None,
                created_at: None,
                updated_at: None,
                provenance: None,
            },
            source: SkillSource::ProjectTsumugi,
            path: SkillPath::new(skill_dir.join("SKILL.md")),
        };

        UseSkillTool::new(vec![meta])
    }

    #[tokio::test]
    async fn use_skill_success() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let tool = make_tool_with_skill(tmp.path());
        let ctx = SandboxContext::test_default();

        let result = tool
            .execute_inner(serde_json::json!({ "skill_name": "my-skill" }), &ctx)
            .await
            .unwrap_or_else(|e| panic!("{e}"));

        assert!(!result.is_error);
        assert!(result.output.contains("Do the thing."));
        assert!(result.output.contains("# Skill: my-skill"));
    }

    #[tokio::test]
    async fn use_skill_not_found() {
        let tool = UseSkillTool::new(vec![]);
        let ctx = SandboxContext::test_default();
        let result = tool
            .execute_inner(serde_json::json!({ "skill_name": "nonexistent" }), &ctx)
            .await
            .unwrap_or_else(|e| panic!("{e}"));

        assert!(result.is_error);
        assert!(result.output.contains("Skill not found"));
    }

    #[tokio::test]
    async fn use_skill_missing_param() {
        let tool = UseSkillTool::new(vec![]);
        let ctx = SandboxContext::test_default();
        let result = tool.execute_inner(serde_json::json!({}), &ctx).await;
        assert!(result.is_err());
    }
}
