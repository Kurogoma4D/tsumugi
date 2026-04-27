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

use crate::def::{FailurePolicy, InputDef, StepDef, WorkflowDef, WorkflowMode};
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
    steps: Vec<RawStepOrRef>,
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

/// A step entry — either a fully-specified step (with `id` + `type`)
/// or a `ref:` to a previously-defined step (only meaningful inside a
/// `loop`'s `steps:` block).
///
/// `Step` is boxed because [`RawStep`] is large and the unboxed
/// variant size dominates the enum. Boxing keeps the parser allocation
/// pattern stable when the supported step set grows.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawStepOrRef {
    /// `{ ref: previous_step_id }` shorthand. Resolved at parse time
    /// by cloning the referenced [`StepDef`] into the loop body.
    Ref(RawStepRef),
    /// Standard step definition.
    Step(Box<RawStep>),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStepRef {
    /// The id of a previously-declared step to clone in.
    r#ref: String,
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

    // Loop-only
    #[serde(default)]
    max_iterations: Option<u32>,
    #[serde(default)]
    until: Option<String>,
    #[serde(default)]
    steps: Option<Vec<RawStepOrRef>>,

    // Branch-only
    #[serde(default)]
    conditions: Option<Vec<RawBranchCondition>>,
    #[serde(default)]
    default: Option<Vec<RawStepOrRef>>,

    // Group-only
    #[serde(default)]
    on_failure: Option<String>,
    #[serde(default)]
    max_retries: Option<u32>,

    // Human-only — kept inline to avoid a separate `with:` wrapper.
    // SPEC §8.4 nests human-step fields under `with:` in the YAML, but
    // since `flatten` here would conflict with `deny_unknown_fields`
    // we accept them at the top level and document that.
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    show: Option<String>,
    #[serde(default)]
    options: Option<Vec<String>>,
    #[serde(default)]
    revise_target: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBranchCondition {
    /// `${{ ... }}` boolean expression (without surrounding braces).
    when: String,
    /// Steps to execute when `when` is truthy.
    steps: Vec<RawStepOrRef>,
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

    // Convert top-level steps with id-uniqueness and `ref:`-availability checks.
    //
    // `registry` tracks every step id seen so far at the top level so a
    // `loop` body can `ref:` them. Nested ids do *not* leak into the
    // registry (a loop's inner steps are loop-scoped).
    let mut registry: BTreeMap<String, StepDef> = BTreeMap::new();
    let mut seen_ids: BTreeSet<String> = BTreeSet::new();
    let mut steps: Vec<StepDef> = Vec::with_capacity(raw.steps.len());
    for entry in raw.steps {
        let RawStepOrRef::Step(raw_step) = entry else {
            return Err(WorkflowError::invalid_workflow(
                &path_display,
                "top-level `ref:` is not allowed; `ref:` may only appear inside a `loop` step's `steps:` block",
            ));
        };
        let step = convert_step(*raw_step, &path_display, &mut seen_ids, &registry)?;
        registry.insert(step.id().to_owned(), step.clone());
        steps.push(step);
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
fn convert_step(
    raw: RawStep,
    path_display: &str,
    seen_ids: &mut BTreeSet<String>,
    registry: &BTreeMap<String, StepDef>,
) -> Result<StepDef> {
    if raw.id.is_empty() {
        return Err(WorkflowError::invalid_workflow(
            path_display,
            "step ids must not be empty",
        ));
    }
    if !is_valid_id(&raw.id) {
        return Err(WorkflowError::invalid_workflow(
            path_display,
            format!("step id '{}' must match {ID_PATTERN}", raw.id),
        ));
    }
    if !seen_ids.insert(raw.id.clone()) {
        return Err(WorkflowError::invalid_workflow(
            path_display,
            format!("duplicate step id '{}'", raw.id),
        ));
    }

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
        max_iterations,
        until,
        steps: nested_steps,
        conditions,
        default,
        on_failure,
        max_retries,
        message,
        show,
        options,
        revise_target,
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
            forbid_control_flow_fields(
                &id,
                "agent",
                path_display,
                max_iterations,
                &until,
                &nested_steps,
                &conditions,
                &default,
                &on_failure,
                max_retries,
                &message,
                &show,
                &options,
                &revise_target,
            )?;

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
            forbid_control_flow_fields(
                &id,
                "shell",
                path_display,
                max_iterations,
                &until,
                &nested_steps,
                &conditions,
                &default,
                &on_failure,
                max_retries,
                &message,
                &show,
                &options,
                &revise_target,
            )?;
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
            forbid_control_flow_fields(
                &id,
                "write_file",
                path_display,
                max_iterations,
                &until,
                &nested_steps,
                &conditions,
                &default,
                &on_failure,
                max_retries,
                &message,
                &show,
                &options,
                &revise_target,
            )?;
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
        "loop" => {
            let max_iterations = max_iterations.ok_or_else(|| {
                WorkflowError::invalid_workflow(
                    path_display,
                    format!("loop step '{id}' is missing required field 'max_iterations'"),
                )
            })?;
            if max_iterations == 0 {
                return Err(WorkflowError::invalid_workflow(
                    path_display,
                    format!("loop step '{id}' max_iterations must be >= 1"),
                ));
            }
            let until = until.ok_or_else(|| {
                WorkflowError::invalid_workflow(
                    path_display,
                    format!("loop step '{id}' is missing required field 'until'"),
                )
            })?;
            let raw_steps = nested_steps.ok_or_else(|| {
                WorkflowError::invalid_workflow(
                    path_display,
                    format!("loop step '{id}' is missing required field 'steps'"),
                )
            })?;
            if when.is_some() {
                return Err(WorkflowError::invalid_workflow(
                    path_display,
                    format!("loop step '{id}' does not currently support 'when'"),
                ));
            }
            // Inner steps are scoped to this loop. We use a *fresh*
            // seen-set so a child can re-use a top-level id by `ref:`,
            // but still detect duplicate inline ids inside the loop.
            let mut inner_seen: BTreeSet<String> = BTreeSet::new();
            let inner = resolve_steps(raw_steps, path_display, &mut inner_seen, registry)?;
            Ok(StepDef::Loop {
                id,
                max_iterations,
                until,
                steps: inner,
            })
        }
        "branch" => {
            let raw_conditions = conditions.ok_or_else(|| {
                WorkflowError::invalid_workflow(
                    path_display,
                    format!("branch step '{id}' is missing required field 'conditions'"),
                )
            })?;
            if raw_conditions.is_empty() {
                return Err(WorkflowError::invalid_workflow(
                    path_display,
                    format!("branch step '{id}' must declare at least one condition"),
                ));
            }
            let mut converted = Vec::with_capacity(raw_conditions.len());
            for cond in raw_conditions {
                let mut inner_seen: BTreeSet<String> = BTreeSet::new();
                let body = resolve_steps(cond.steps, path_display, &mut inner_seen, registry)?;
                converted.push((cond.when, body));
            }
            let default_steps = match default {
                Some(raw) => {
                    let mut inner_seen: BTreeSet<String> = BTreeSet::new();
                    Some(resolve_steps(raw, path_display, &mut inner_seen, registry)?)
                }
                None => None,
            };
            Ok(StepDef::Branch {
                id,
                conditions: converted,
                default: default_steps,
            })
        }
        "parallel" => {
            let raw_steps = nested_steps.ok_or_else(|| {
                WorkflowError::invalid_workflow(
                    path_display,
                    format!("parallel step '{id}' is missing required field 'steps'"),
                )
            })?;
            if raw_steps.is_empty() {
                return Err(WorkflowError::invalid_workflow(
                    path_display,
                    format!("parallel step '{id}' must declare at least one inner step"),
                ));
            }
            let mut inner_seen: BTreeSet<String> = BTreeSet::new();
            let inner = resolve_steps(raw_steps, path_display, &mut inner_seen, registry)?;
            Ok(StepDef::Parallel { id, steps: inner })
        }
        "group" => {
            let raw_steps = nested_steps.ok_or_else(|| {
                WorkflowError::invalid_workflow(
                    path_display,
                    format!("group step '{id}' is missing required field 'steps'"),
                )
            })?;
            let policy = match on_failure.as_deref() {
                None | Some("abort") => FailurePolicy::Abort,
                Some("retry") => FailurePolicy::Retry,
                Some("continue") => FailurePolicy::Continue,
                Some(other) => {
                    return Err(WorkflowError::invalid_workflow(
                        path_display,
                        format!(
                            "group step '{id}' has unknown on_failure '{other}' (allowed: abort, retry, continue)"
                        ),
                    ));
                }
            };
            let max_retries_value = max_retries.unwrap_or(0);
            if matches!(policy, FailurePolicy::Retry) && max_retries_value == 0 {
                return Err(WorkflowError::invalid_workflow(
                    path_display,
                    format!(
                        "group step '{id}' has on_failure: retry but max_retries is 0 (must be >= 1)"
                    ),
                ));
            }
            let mut inner_seen: BTreeSet<String> = BTreeSet::new();
            let inner = resolve_steps(raw_steps, path_display, &mut inner_seen, registry)?;
            Ok(StepDef::Group {
                id,
                on_failure: policy,
                max_retries: max_retries_value,
                steps: inner,
            })
        }
        "human" => {
            let message = message.ok_or_else(|| {
                WorkflowError::invalid_workflow(
                    path_display,
                    format!("human step '{id}' is missing required field 'message'"),
                )
            })?;
            let options =
                options.unwrap_or_else(|| vec!["approve".to_owned(), "reject".to_owned()]);
            // If `revise` is in `options`, the `revise_target` must be set.
            if options.iter().any(|o| o == "revise") && revise_target.is_none() {
                return Err(WorkflowError::invalid_workflow(
                    path_display,
                    format!(
                        "human step '{id}' lists 'revise' in options but is missing 'revise_target'"
                    ),
                ));
            }
            Ok(StepDef::Human {
                id,
                message,
                show,
                options,
                revise_target,
            })
        }
        other => Err(WorkflowError::invalid_workflow(
            path_display,
            format!(
                "unknown step type '{other}' for step '{id}' (supported: agent, shell, write_file, loop, branch, parallel, group, human)"
            ),
        )),
    }
}

/// Resolve a `Vec<RawStepOrRef>` (typically a control-flow body) into
/// a `Vec<StepDef>`. `ref:` entries are looked up in `registry` and
/// cloned in. Inline entries are validated against `inner_seen`.
fn resolve_steps(
    raw: Vec<RawStepOrRef>,
    path_display: &str,
    inner_seen: &mut BTreeSet<String>,
    registry: &BTreeMap<String, StepDef>,
) -> Result<Vec<StepDef>> {
    let mut out = Vec::with_capacity(raw.len());
    for entry in raw {
        match entry {
            RawStepOrRef::Ref(r) => {
                let Some(found) = registry.get(&r.r#ref) else {
                    return Err(WorkflowError::invalid_workflow(
                        path_display,
                        format!(
                            "ref: '{}' does not match any previously-defined step id",
                            r.r#ref
                        ),
                    ));
                };
                // Cloning a leaf step is fine; a `ref:` to a control-flow
                // step is rejected because re-running a loop / human /
                // group inline raises ambiguity around id-disambiguation
                // and snapshot ownership. Flag it now with a clear error.
                match found {
                    StepDef::Agent { .. } | StepDef::Shell { .. } | StepDef::WriteFile { .. } => {
                        out.push(found.clone());
                    }
                    other => {
                        return Err(WorkflowError::invalid_workflow(
                            path_display,
                            format!(
                                "ref: '{}' resolves to a {} step; only agent/shell/write_file steps may be ref'd",
                                r.r#ref,
                                other.step_type()
                            ),
                        ));
                    }
                }
            }
            RawStepOrRef::Step(raw_step) => {
                // Nested control-flow does not pollute the parent's
                // step registry — fresh seen-set, but preserve the
                // outer registry so `ref:` still resolves.
                let step = convert_step(*raw_step, path_display, inner_seen, registry)?;
                out.push(step);
            }
        }
    }
    Ok(out)
}

/// Reject control-flow-only fields on leaf step types.
#[expect(
    clippy::too_many_arguments,
    reason = "validation needs every field; grouping into a struct would just shuffle the arguments"
)]
#[expect(
    clippy::ref_option,
    reason = "callers already hold the Option; passing &Option<T> avoids cloning the inner Vec/String"
)]
fn forbid_control_flow_fields(
    id: &str,
    step_type: &str,
    path_display: &str,
    max_iterations: Option<u32>,
    until: &Option<String>,
    nested_steps: &Option<Vec<RawStepOrRef>>,
    conditions: &Option<Vec<RawBranchCondition>>,
    default: &Option<Vec<RawStepOrRef>>,
    on_failure: &Option<String>,
    max_retries: Option<u32>,
    message: &Option<String>,
    show: &Option<String>,
    options: &Option<Vec<String>>,
    revise_target: &Option<String>,
) -> Result<()> {
    if max_iterations.is_some()
        || until.is_some()
        || nested_steps.is_some()
        || conditions.is_some()
        || default.is_some()
        || on_failure.is_some()
        || max_retries.is_some()
        || message.is_some()
        || show.is_some()
        || options.is_some()
        || revise_target.is_some()
    {
        return Err(WorkflowError::invalid_workflow(
            path_display,
            format!("{step_type} step '{id}' must not set control-flow fields"),
        ));
    }
    Ok(())
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
    type: mystery
    command: "true"
"#;
        let err = parse_workflow_str(yaml, p()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown step type"), "{msg}");
        // Mention the supported set.
        assert!(msg.contains("agent"));
        assert!(msg.contains("loop"));
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

    /// SPEC §8.4 `review_loop` sample: top-level steps + a loop body
    /// that uses `ref:` to reuse previous shell steps.
    #[test]
    fn parses_spec_review_loop_sample() {
        let yaml = r#"
id: review_loop
steps:
  - id: verify
    type: shell
    command: "cargo test --workspace"
  - id: fix_errors
    type: shell
    command: "echo fix"
  - id: loop_step
    type: loop
    max_iterations: 3
    until: "steps.verify.exit_code == 0"
    steps:
      - ref: verify
      - ref: fix_errors
"#;
        let wf = parse_workflow_str(yaml, p()).unwrap();
        assert_eq!(wf.steps.len(), 3);
        match &wf.steps[2] {
            StepDef::Loop {
                id,
                max_iterations,
                until,
                steps,
            } => {
                assert_eq!(id, "loop_step");
                assert_eq!(*max_iterations, 3);
                assert_eq!(until, "steps.verify.exit_code == 0");
                assert_eq!(steps.len(), 2);
                assert_eq!(steps[0].id(), "verify");
                assert_eq!(steps[1].id(), "fix_errors");
            }
            other => panic!("expected Loop, got {other:?}"),
        }
    }

    #[test]
    fn loop_ref_to_unknown_id_fails() {
        let yaml = r#"
id: bad_ref
steps:
  - id: l
    type: loop
    max_iterations: 1
    until: "true"
    steps:
      - ref: not_defined
"#;
        let err = parse_workflow_str(yaml, p()).unwrap_err();
        assert!(err.to_string().contains("does not match"), "{}", err);
    }

    #[test]
    fn parses_branch_with_default() {
        let yaml = r#"
id: bx
steps:
  - id: choose
    type: branch
    conditions:
      - when: "inputs.flag == 1"
        steps:
          - id: a
            type: shell
            command: "echo a"
      - when: "inputs.flag == 2"
        steps:
          - id: b
            type: shell
            command: "echo b"
    default:
      - id: c
        type: shell
        command: "echo c"
"#;
        let wf = parse_workflow_str(yaml, p()).unwrap();
        match &wf.steps[0] {
            StepDef::Branch {
                conditions,
                default,
                ..
            } => {
                assert_eq!(conditions.len(), 2);
                assert!(default.is_some());
            }
            _ => panic!("expected branch"),
        }
    }

    #[test]
    fn parses_parallel() {
        let yaml = r#"
id: par
steps:
  - id: p
    type: parallel
    steps:
      - id: a
        type: shell
        command: "echo a"
      - id: b
        type: shell
        command: "echo b"
"#;
        let wf = parse_workflow_str(yaml, p()).unwrap();
        match &wf.steps[0] {
            StepDef::Parallel { steps, .. } => assert_eq!(steps.len(), 2),
            _ => panic!("expected parallel"),
        }
    }

    #[test]
    fn parses_group_retry_with_max_retries() {
        let yaml = r#"
id: g
steps:
  - id: grp
    type: group
    on_failure: retry
    max_retries: 3
    steps:
      - id: s
        type: shell
        command: "true"
"#;
        let wf = parse_workflow_str(yaml, p()).unwrap();
        match &wf.steps[0] {
            StepDef::Group {
                on_failure,
                max_retries,
                ..
            } => {
                assert_eq!(*on_failure, FailurePolicy::Retry);
                assert_eq!(*max_retries, 3);
            }
            _ => panic!("expected group"),
        }
    }

    #[test]
    fn group_retry_without_max_retries_fails() {
        let yaml = r#"
id: g
steps:
  - id: grp
    type: group
    on_failure: retry
    steps:
      - id: s
        type: shell
        command: "true"
"#;
        let err = parse_workflow_str(yaml, p()).unwrap_err();
        assert!(err.to_string().contains("max_retries"), "{err}");
    }

    #[test]
    fn parses_human_step_with_revise() {
        let yaml = r#"
id: hh
steps:
  - id: design
    type: agent
    subagent: worker
    prompt: "design"
  - id: review
    type: human
    message: "review the design"
    show: "${{ steps.design.output }}"
    options: [approve, reject, revise]
    revise_target: design
"#;
        let wf = parse_workflow_str(yaml, p()).unwrap();
        match &wf.steps[1] {
            StepDef::Human {
                message,
                options,
                revise_target,
                ..
            } => {
                assert_eq!(message, "review the design");
                assert_eq!(options.len(), 3);
                assert_eq!(revise_target.as_deref(), Some("design"));
            }
            _ => panic!("expected human"),
        }
    }

    #[test]
    fn human_with_revise_but_no_target_fails() {
        let yaml = r#"
id: bad_human
steps:
  - id: r
    type: human
    message: "decide"
    options: [approve, revise]
"#;
        let err = parse_workflow_str(yaml, p()).unwrap_err();
        assert!(err.to_string().contains("revise_target"), "{err}");
    }
}
