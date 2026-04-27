//! YAML → [`WorkflowDef`] parser (SPEC §8.3).
//!
//! The strategy is to deserialize into a permissive serde mirror, then
//! convert (and validate) into the canonical types in [`crate::def`].
//! Parse-time errors point at the offending field with a clear message.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;
use std::time::Duration;

use serde::Deserialize;

use crate::def::{InputDef, StepDef, WorkflowDef, WorkflowMode};
use crate::error::{Result, WorkflowError};

/// Identifier pattern enforced for workflow ids and step ids.
///
/// Snake-case-ish: must start with a lower-case ASCII letter, followed
/// by lower-case ASCII letters, digits, or underscores.
const ID_PATTERN: &str = r"^[a-z][a-z0-9_]*$";

/// Validate an id against [`ID_PATTERN`] without holding a `Regex`
/// outside of this function.
///
/// We avoid storing the compiled regex in a `LazyLock` (and the
/// associated initialization-failure branch) by performing the trivial
/// `snake_case` check by hand: the pattern `^[a-z][a-z0-9_]*$` is
/// straightforward enough that hand-rolling it avoids both the regex
/// initialization-failure footgun and the workspace's `unwrap_used`
/// lint. We still keep `regex` in the dependency list because it is a
/// workspace-wide tool — other expression validation paths may use it
/// later.
fn is_valid_id(id: &str) -> bool {
    let mut chars = id.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_lowercase() {
        return false;
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// Permissive serde mirror for the top-level workflow YAML object.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWorkflow {
    id: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    inputs: BTreeMap<String, RawInput>,
    #[serde(default)]
    steps: Vec<RawStep>,
    #[serde(default)]
    outputs: BTreeMap<String, String>,
}

/// Permissive serde mirror for an input definition.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawInput {
    #[serde(default = "default_input_type", rename = "type")]
    r#type: String,
    #[serde(default)]
    default: Option<serde_json::Value>,
    #[serde(default)]
    required: Option<bool>,
    #[serde(default)]
    description: Option<String>,
}

fn default_input_type() -> String {
    "string".to_owned()
}

/// Permissive serde mirror for a step.
///
/// `step_type` is captured under the YAML `type` key so we can validate
/// the supported set with a clean error message before dispatching to
/// the type-specific fields.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStep {
    id: String,
    #[serde(rename = "type")]
    step_type: String,

    // Common
    #[serde(default)]
    when: Option<String>,

    // Agent-only
    #[serde(default)]
    subagent: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    inject_files: Option<Vec<String>>,

    // Shell-only
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    timeout: Option<RawTimeout>,

    // WriteFile-only
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    content: Option<String>,
}

/// A timeout, accepted as either a number-of-seconds or a humantime-style
/// string ("30s", "2m"). Number is interpreted as seconds.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawTimeout {
    Seconds(u64),
    Humantime(String),
}

impl RawTimeout {
    fn into_duration(self, step_id: &str) -> Result<Duration> {
        match self {
            Self::Seconds(s) => Ok(Duration::from_secs(s)),
            Self::Humantime(s) => humantime::parse_duration(&s).map_err(|e| {
                WorkflowError::invalid_workflow(
                    format!("step '{step_id}'"),
                    format!("invalid timeout '{s}': {e}"),
                )
            }),
        }
    }
}

/// Parse a workflow YAML string into a validated [`WorkflowDef`].
///
/// `source_path` is used purely for error context.
pub fn parse_workflow_str(yaml: &str, source_path: impl AsRef<Path>) -> Result<WorkflowDef> {
    let path = source_path.as_ref();
    let raw: RawWorkflow = serde_yml::from_str(yaml).map_err(|e| WorkflowError::YamlParse {
        path: path.to_path_buf(),
        source: e,
    })?;
    finalize_workflow(raw, path)
}

/// Read and parse a workflow file from disk.
pub async fn parse_workflow_file(path: impl AsRef<Path>) -> Result<WorkflowDef> {
    let path = path.as_ref();
    let content = tokio::fs::read_to_string(path)
        .await
        .map_err(|e| WorkflowError::io(format!("reading workflow {}", path.display()), e))?;
    parse_workflow_str(&content, path)
}

fn finalize_workflow(raw: RawWorkflow, path: &Path) -> Result<WorkflowDef> {
    let path_display = path.display().to_string();

    // Validate workflow id.
    if raw.id.is_empty() {
        return Err(WorkflowError::invalid_workflow(
            &path_display,
            "workflow `id` must not be empty",
        ));
    }
    if !is_valid_id(&raw.id) {
        return Err(WorkflowError::invalid_workflow(
            &path_display,
            format!(
                "workflow id '{}' must match {ID_PATTERN} (lower-case letters, digits, underscores; starts with a letter)",
                raw.id
            ),
        ));
    }

    // Resolve mode.
    let mode = match raw.mode.as_deref() {
        None | Some("normal") => WorkflowMode::Normal,
        Some("long_running") => WorkflowMode::LongRunning,
        Some(other) => {
            return Err(WorkflowError::invalid_workflow(
                &path_display,
                format!("unknown workflow mode '{other}' (supported: normal, long_running)"),
            ));
        }
    };

    // Convert inputs.
    let mut inputs: BTreeMap<String, InputDef> = BTreeMap::new();
    for (name, raw_input) in raw.inputs {
        if name.is_empty() {
            return Err(WorkflowError::invalid_workflow(
                &path_display,
                "input names must not be empty",
            ));
        }
        let required = raw_input.required.unwrap_or(false);
        inputs.insert(
            name,
            InputDef {
                r#type: raw_input.r#type,
                default: raw_input.default,
                required,
                description: raw_input.description,
            },
        );
    }

    // Convert steps with id-uniqueness check.
    let mut seen_ids: BTreeSet<String> = BTreeSet::new();
    let mut steps: Vec<StepDef> = Vec::with_capacity(raw.steps.len());
    for raw_step in raw.steps {
        if raw_step.id.is_empty() {
            return Err(WorkflowError::invalid_workflow(
                &path_display,
                "step ids must not be empty",
            ));
        }
        if !is_valid_id(&raw_step.id) {
            return Err(WorkflowError::invalid_workflow(
                &path_display,
                format!("step id '{}' must match {ID_PATTERN}", raw_step.id),
            ));
        }
        if !seen_ids.insert(raw_step.id.clone()) {
            return Err(WorkflowError::invalid_workflow(
                &path_display,
                format!("duplicate step id '{}'", raw_step.id),
            ));
        }
        steps.push(convert_step(raw_step, &path_display)?);
    }

    Ok(WorkflowDef {
        id: raw.id,
        description: raw.description,
        mode,
        inputs,
        steps,
        outputs: raw.outputs,
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "linear per-step-type validation; splitting would scatter the supported-fields matrix across helpers"
)]
fn convert_step(raw: RawStep, path_display: &str) -> Result<StepDef> {
    let RawStep {
        id,
        step_type,
        when,
        subagent,
        prompt,
        model,
        inject_files,
        command,
        timeout,
        path,
        content,
    } = raw;

    match step_type.as_str() {
        "agent" => {
            let subagent = subagent.ok_or_else(|| {
                WorkflowError::invalid_workflow(
                    path_display,
                    format!("agent step '{id}' is missing required field 'subagent'"),
                )
            })?;
            let prompt = prompt.ok_or_else(|| {
                WorkflowError::invalid_workflow(
                    path_display,
                    format!("agent step '{id}' is missing required field 'prompt'"),
                )
            })?;
            // Forbid shell-only / write_file-only fields on agent steps.
            if command.is_some() || timeout.is_some() {
                return Err(WorkflowError::invalid_workflow(
                    path_display,
                    format!("agent step '{id}' must not set shell fields ('command'/'timeout')"),
                ));
            }
            if path.is_some() || content.is_some() {
                return Err(WorkflowError::invalid_workflow(
                    path_display,
                    format!("agent step '{id}' must not set write_file fields ('path'/'content')"),
                ));
            }

            Ok(StepDef::Agent {
                id,
                subagent,
                prompt,
                model,
                when,
                inject_files: inject_files.unwrap_or_default(),
            })
        }
        "shell" => {
            let command = command.ok_or_else(|| {
                WorkflowError::invalid_workflow(
                    path_display,
                    format!("shell step '{id}' is missing required field 'command'"),
                )
            })?;
            if subagent.is_some() || prompt.is_some() || model.is_some() || inject_files.is_some() {
                return Err(WorkflowError::invalid_workflow(
                    path_display,
                    format!("shell step '{id}' must not set agent fields"),
                ));
            }
            if path.is_some() || content.is_some() {
                return Err(WorkflowError::invalid_workflow(
                    path_display,
                    format!("shell step '{id}' must not set write_file fields"),
                ));
            }
            let timeout = match timeout {
                Some(t) => Some(t.into_duration(&id)?),
                None => None,
            };
            Ok(StepDef::Shell {
                id,
                command,
                timeout,
                when,
            })
        }
        "write_file" => {
            let path = path.ok_or_else(|| {
                WorkflowError::invalid_workflow(
                    path_display,
                    format!("write_file step '{id}' is missing required field 'path'"),
                )
            })?;
            let content = content.ok_or_else(|| {
                WorkflowError::invalid_workflow(
                    path_display,
                    format!("write_file step '{id}' is missing required field 'content'"),
                )
            })?;
            if subagent.is_some()
                || prompt.is_some()
                || model.is_some()
                || inject_files.is_some()
                || command.is_some()
                || timeout.is_some()
            {
                return Err(WorkflowError::invalid_workflow(
                    path_display,
                    format!("write_file step '{id}' must not set agent or shell fields"),
                ));
            }
            if when.is_some() {
                // SPEC §8.4 currently scopes `when` to agent/shell;
                // reject explicitly so tooling does not assume it works.
                return Err(WorkflowError::invalid_workflow(
                    path_display,
                    format!(
                        "write_file step '{id}' does not currently support 'when' (see SPEC §8.4)"
                    ),
                ));
            }
            Ok(StepDef::WriteFile { id, path, content })
        }
        other => Err(WorkflowError::invalid_workflow(
            path_display,
            format!(
                "unknown step type '{other}' for step '{id}' (supported: agent, shell, write_file; loop/branch/parallel/group/human are out of scope — see issue #40)"
            ),
        )),
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions")]
#[expect(clippy::unwrap_used, reason = "test assertions")]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn p() -> PathBuf {
        PathBuf::from("<inline>")
    }

    #[test]
    fn id_validator_matches_pattern() {
        assert!(is_valid_id("foo"));
        assert!(is_valid_id("foo_bar1"));
        assert!(!is_valid_id("Foo"));
        assert!(!is_valid_id("1foo"));
        assert!(!is_valid_id("foo-bar"));
        assert!(!is_valid_id(""));
    }

    /// SPEC §8.3 sample workflow (agent + shell subset).
    const SPEC_8_3_SAMPLE: &str = r#"
id: implement_feature
description: "Implement a feature with agent + verify"
mode: normal

inputs:
  feature_id:
    type: string
    required: true
    description: "Target feature id"
  budget:
    type: integer
    default: 50

steps:
  - id: plan
    type: agent
    subagent: plan
    prompt: "Plan the implementation of feature ${{ inputs.feature_id }}."

  - id: implement
    type: agent
    subagent: worker
    prompt: "Implement based on this plan:\n${{ steps.plan.output }}"
    inject_files:
      - "src/lib.rs"

  - id: verify
    type: shell
    command: "cargo test --workspace"
    timeout: 600

outputs:
  plan_summary: "${{ steps.plan.output }}"
  test_exit:   "${{ steps.verify.exit_code }}"
"#;

    #[test]
    fn parses_spec_sample() {
        let wf = parse_workflow_str(SPEC_8_3_SAMPLE, p())
            .unwrap_or_else(|e| panic!("parse failed: {e}"));
        assert_eq!(wf.id, "implement_feature");
        assert_eq!(
            wf.description.as_deref(),
            Some("Implement a feature with agent + verify")
        );
        assert_eq!(wf.mode, WorkflowMode::Normal);
        assert_eq!(wf.inputs.len(), 2);
        assert!(wf.inputs.get("feature_id").unwrap().required);
        assert_eq!(wf.steps.len(), 3);
        assert_eq!(wf.steps[0].id(), "plan");
        assert_eq!(wf.steps[1].step_type(), "agent");
        match &wf.steps[2] {
            StepDef::Shell {
                command, timeout, ..
            } => {
                assert_eq!(command, "cargo test --workspace");
                assert_eq!(*timeout, Some(Duration::from_secs(600)));
            }
            other => panic!("unexpected step: {other:?}"),
        }
        assert_eq!(wf.outputs.len(), 2);
    }

    #[test]
    fn parses_humantime_timeout() {
        let yaml = r#"
id: hm
steps:
  - id: t
    type: shell
    command: "true"
    timeout: "30s"
"#;
        let wf = parse_workflow_str(yaml, p()).unwrap();
        match &wf.steps[0] {
            StepDef::Shell { timeout, .. } => {
                assert_eq!(*timeout, Some(Duration::from_secs(30)));
            }
            _ => panic!("not shell"),
        }
    }

    #[test]
    fn rejects_duplicate_step_ids() {
        let yaml = r#"
id: dup
steps:
  - id: same
    type: shell
    command: "echo a"
  - id: same
    type: shell
    command: "echo b"
"#;
        let err = parse_workflow_str(yaml, p()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("duplicate step id"), "{msg}");
    }

    #[test]
    fn rejects_unknown_step_type() {
        let yaml = r#"
id: bad
steps:
  - id: s
    type: parallel
    command: "true"
"#;
        let err = parse_workflow_str(yaml, p()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown step type"), "{msg}");
        // Mention the supported set + #40 reference.
        assert!(msg.contains("agent"));
        assert!(msg.contains("issue #40"));
    }

    #[test]
    fn rejects_uppercase_id() {
        let yaml = r"
id: BAD
steps: []
";
        let err = parse_workflow_str(yaml, p()).unwrap_err();
        assert!(err.to_string().contains("workflow id"));
    }

    #[test]
    fn rejects_empty_id() {
        let yaml = r#"
id: ""
steps: []
"#;
        let err = parse_workflow_str(yaml, p()).unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn rejects_agent_with_command() {
        let yaml = r#"
id: bad
steps:
  - id: a
    type: agent
    subagent: worker
    prompt: "hi"
    command: "echo nope"
"#;
        let err = parse_workflow_str(yaml, p()).unwrap_err();
        assert!(err.to_string().contains("must not set shell fields"));
    }

    #[test]
    fn rejects_write_file_with_when() {
        let yaml = r#"
id: bad
steps:
  - id: w
    type: write_file
    path: "out.txt"
    content: "hi"
    when: "true"
"#;
        let err = parse_workflow_str(yaml, p()).unwrap_err();
        assert!(
            err.to_string()
                .contains("does not currently support 'when'")
        );
    }

    #[test]
    fn write_file_step_round_trips() {
        let yaml = r#"
id: ok
steps:
  - id: w
    type: write_file
    path: "${{ inputs.out }}"
    content: "hello ${{ inputs.name }}"
"#;
        let wf = parse_workflow_str(yaml, p()).unwrap();
        match &wf.steps[0] {
            StepDef::WriteFile { path, content, .. } => {
                assert_eq!(path, "${{ inputs.out }}");
                assert_eq!(content, "hello ${{ inputs.name }}");
            }
            _ => panic!("not write_file"),
        }
    }

    #[test]
    fn rejects_unknown_top_level_field() {
        let yaml = r"
id: extra
unknown_field: 1
steps: []
";
        let err = parse_workflow_str(yaml, p()).unwrap_err();
        assert!(matches!(err, WorkflowError::YamlParse { .. }));
    }

    #[test]
    fn long_running_mode_parses() {
        let yaml = r"
id: lr
mode: long_running
steps: []
";
        let wf = parse_workflow_str(yaml, p()).unwrap();
        assert_eq!(wf.mode, WorkflowMode::LongRunning);
    }
}
