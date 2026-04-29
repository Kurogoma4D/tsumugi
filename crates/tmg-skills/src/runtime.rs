//! `SkillsRuntime`: stateful orchestrator wiring the autonomous skill
//! creation pipeline.
//!
//! This module is the glue the harness / CLI talks to so the dead-code
//! warnings on `SignalCollector`, `SkillManageTool`, `SkillCriticConfig`,
//! `update_metrics`, `format_banner`, and the `App::take_*` accessors
//! disappear in production builds.
//!
//! The runtime is deliberately I/O-light: it owns the
//! [`SignalCollector`] state, holds an [`Arc<SkillManageTool>`] for
//! direct invocation, and exposes verbs the CLI / TUI can call between
//! turns:
//!
//! - [`SkillsRuntime::record_turn`] — feed a turn summary, get back the
//!   list of signals (deduped through [`SignalCollector::highest_priority`]).
//! - [`SkillsRuntime::apply_verdict`] — execute a parsed
//!   [`SkillCriticVerdict`] by invoking [`SkillManageTool`] under the
//!   right sandbox.
//! - [`SkillsRuntime::record_use_skill_outcome`] — call
//!   [`update_metrics`] after a `use_skill` invocation's follow-up turn.
//! - [`SkillsRuntime::apply_rejection`] — delete a rejected skill
//!   directory and (optionally) write a feedback note.
//! - [`SkillsRuntime::compute_pending_banner`] — list the auto-generated
//!   skills whose banner the user has not yet acknowledged.
//! - [`SkillsRuntime::acknowledge_banner`] — write `.banner_ack`.
//!
//! The harness is responsible for spawning `skill_critic` (an LLM
//! round-trip) and feeding the parsed verdict here. The runtime
//! intentionally does not own the LLM client.

use std::path::PathBuf;
use std::sync::Arc;

use tmg_sandbox::SandboxContext;
use tmg_tools::error::ToolError;

use crate::banner::{format_banner, load_acknowledged, pending_banner_names, save_acknowledged};
use crate::critic::{SkillCriticConfig, SkillCriticVerdict};
use crate::discovery::discover_skills_with_config;
use crate::emergence::{SignalCollector, SkillCandidacySignal, ToolCallOutcome, TurnSummary};
use crate::error::SkillError;
use crate::manage::SkillManageTool;
use crate::metrics::{SkillMetrics, update_metrics};

/// Outcome the runtime cares about for [`SkillsRuntime::record_use_skill_outcome`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UseSkillOutcome {
    /// The follow-up turn produced no error and no negation.
    Success,
    /// The follow-up turn errored or the user negated; the carried
    /// string is captured as an `improvement_hint` entry.
    Failure(String),
}

/// Stateful runtime owning the `skill_critic` + `skill_manage` flow.
pub struct SkillsRuntime {
    /// Project root (e.g. canonicalised cwd) used to resolve
    /// `.tsumugi/skills/`.
    project_root: PathBuf,
    /// Effective `[skills.critic]` configuration.
    critic_config: SkillCriticConfig,
    /// Per-process signal collector.
    collector: SignalCollector,
    /// Pre-built [`SkillManageTool`] (so the runtime can invoke it
    /// without going through a registry).
    manage_tool: Arc<SkillManageTool>,
}

impl SkillsRuntime {
    /// Construct a new runtime.
    #[must_use]
    pub fn new(project_root: impl Into<PathBuf>, critic_config: SkillCriticConfig) -> Self {
        let max = critic_config.max_per_session;
        Self {
            project_root: project_root.into(),
            critic_config,
            collector: SignalCollector::new(max),
            manage_tool: Arc::new(SkillManageTool),
        }
    }

    /// Project-local skills root (`<project>/.tsumugi/skills/`).
    #[must_use]
    pub fn skills_root(&self) -> PathBuf {
        self.project_root.join(".tsumugi").join("skills")
    }

    /// Borrow the critic config (used by harness wiring to look up the
    /// endpoint / model / system prompt).
    #[must_use]
    pub fn critic_config(&self) -> &SkillCriticConfig {
        &self.critic_config
    }

    /// Whether autonomous skill creation is allowed *right now*
    /// (budget remaining and not auto-disabled).
    #[must_use]
    pub fn auto_creation_allowed(&self) -> bool {
        self.critic_config.enabled && self.collector.auto_creation_allowed()
    }

    /// Mark auto-creation as disabled for the rest of the session
    /// (driven by the TUI's `/skills disable-auto`).
    pub fn disable_auto(&mut self) {
        self.collector.disable_auto();
    }

    /// Whether `/skills disable-auto` has been issued.
    #[must_use]
    pub fn auto_disabled(&self) -> bool {
        self.collector.auto_disabled()
    }

    /// Forward `record_turn` to the collector. Returns the list of
    /// signals produced.
    pub fn record_turn(&mut self, turn: &TurnSummary) -> Vec<SkillCandidacySignal> {
        self.collector.record_turn(turn)
    }

    /// Pick the highest-priority signal — `UserCorrection` >
    /// `ErrorRecovery` > `SuccessfulComplexTask` — from a list. This is
    /// the canonical "fire one critic per turn" projection.
    #[must_use]
    pub fn highest_priority(signals: &[SkillCandidacySignal]) -> Option<&SkillCandidacySignal> {
        SignalCollector::highest_priority(signals)
    }

    /// Construct a manual capture signal (driven by `/skill capture`).
    #[must_use]
    pub fn manual_capture(&self, turn_index: usize, tool_call_count: u32) -> SkillCandidacySignal {
        self.collector.manual_capture(turn_index, tool_call_count)
    }

    /// Apply a parsed [`SkillCriticVerdict`].
    ///
    /// When the verdict has `create=true`, the runtime invokes
    /// [`SkillManageTool`] under the supplied [`SandboxContext`] with
    /// `provenance=agent`, marks `record_auto_generated`, and returns
    /// the new skill directory. When `create=false`, returns
    /// `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Surfaces the [`ToolError`] returned by [`SkillManageTool`].
    pub async fn apply_verdict(
        &mut self,
        verdict: &SkillCriticVerdict,
        ctx: &SandboxContext,
    ) -> Result<Option<PathBuf>, ToolError> {
        if !verdict.create {
            return Ok(None);
        }
        let name = verdict.name.as_deref().unwrap_or("");
        let description = verdict.description.as_deref().unwrap_or("");
        let draft = verdict.draft.as_deref().unwrap_or("");

        let params = serde_json::json!({
            "action": "create",
            "name": name,
            "description": description,
            "invocation": "auto",
            "provenance": "agent",
            "content": draft,
        });
        let result = self.manage_tool.execute_inner(params, ctx).await?;
        if result.is_error {
            return Err(ToolError::invalid_params(format!(
                "skill_manage rejected verdict: {}",
                result.output
            )));
        }
        // Decrement the per-session budget.
        self.collector.record_auto_generated();

        let skill_dir = self.skills_root().join(name);
        Ok(Some(skill_dir))
    }

    /// Apply a `/skills reject <name>` request: delete the skill
    /// directory.
    ///
    /// The caller is responsible for writing a feedback note to memory
    /// (it owns the [`tmg_memory::MemoryStore`]); this method only
    /// handles the on-disk skill removal.
    ///
    /// # Errors
    ///
    /// Surfaces the [`ToolError`] returned by [`SkillManageTool`].
    pub async fn apply_rejection(&self, name: &str, ctx: &SandboxContext) -> Result<(), ToolError> {
        let params = serde_json::json!({
            "action": "remove",
            "name": name,
        });
        let _ = self.manage_tool.execute_inner(params, ctx).await?;
        Ok(())
    }

    /// Record a `use_skill` invocation outcome by updating the per-skill
    /// metrics file.
    ///
    /// # Errors
    ///
    /// Propagates [`SkillError`] from the metrics load/save.
    pub async fn record_use_skill_outcome(
        &self,
        skill_name: &str,
        outcome: UseSkillOutcome,
    ) -> Result<SkillMetrics, SkillError> {
        let skill_dir = self.skills_root().join(skill_name);
        update_metrics(&skill_dir, |m| match outcome {
            UseSkillOutcome::Success => m.record_success(),
            UseSkillOutcome::Failure(hint) => m.record_failure(hint),
        })
        .await
    }

    /// List the auto-generated skill names whose banner the user has
    /// not yet acknowledged.
    ///
    /// Re-discovers skills under the project root each call so a freshly
    /// created skill is picked up without restarting the runtime.
    ///
    /// # Errors
    ///
    /// Propagates [`SkillError`] from discovery and the
    /// `.banner_ack` read.
    pub async fn compute_pending_banner(
        &self,
        skills_config: &crate::config::SkillsConfig,
    ) -> Result<Vec<String>, SkillError> {
        let metas = discover_skills_with_config(&self.project_root, skills_config).await?;
        let acked = load_acknowledged(self.skills_root()).await?;
        Ok(pending_banner_names(&metas, &acked))
    }

    /// Format the pending banner for display. Returns `None` when
    /// nothing pending.
    #[must_use]
    pub fn format_pending_banner(names: &[String]) -> Option<String> {
        format_banner(names)
    }

    /// Persist that the user has dismissed the banner for `names`.
    ///
    /// # Errors
    ///
    /// Propagates [`SkillError`] from the file write.
    pub async fn acknowledge_banner(&self, names: &[String]) -> Result<(), SkillError> {
        let mut set = load_acknowledged(self.skills_root()).await?;
        for n in names {
            set.insert(n.clone());
        }
        save_acknowledged(self.skills_root(), &set).await
    }

    /// Convenience: for a [`tmg_core::TurnSummary`] (which only carries
    /// `tokens_used`, `tool_calls` count, `user_message`), build a
    /// [`TurnSummary`] (the skills version, which needs per-tool-call
    /// outcomes). Callers typically construct the per-call outcomes
    /// themselves; this helper is the simplest mapping for cases where
    /// outcomes are unknown — it produces a turn with `tool_calls.len()`
    /// successful entries for `success_calls`, `failed_calls` failures,
    /// and the supplied `feedback_memory_written` / `turn_errored`
    /// flags.
    ///
    /// Used by the harness wire-up where the streaming `StreamSink` has
    /// already counted tool successes / failures separately from
    /// [`tmg_core::TurnSummary`].
    #[must_use]
    pub fn build_skills_turn_summary(
        turn_index: usize,
        success_calls: &[String],
        failed_calls: &[String],
        feedback_memory_written: bool,
        turn_errored: bool,
    ) -> TurnSummary {
        let mut tool_calls = Vec::with_capacity(success_calls.len() + failed_calls.len());
        for name in success_calls {
            tool_calls.push(ToolCallOutcome {
                tool: name.clone(),
                success: true,
            });
        }
        for name in failed_calls {
            tool_calls.push(ToolCallOutcome {
                tool: name.clone(),
                success: false,
            });
        }
        TurnSummary {
            turn_index,
            tool_calls,
            feedback_memory_written,
            turn_errored,
        }
    }
}

// Re-export the trigger kind for convenience so callers don't need a
// separate `use tmg_skills::TriggerKind`.
pub use crate::emergence::TriggerKind as PublicTriggerKind;

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "tests assert with unwrap/expect for clarity; the workspace policy denies them in production code"
)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::emergence::TriggerKind;
    use tmg_sandbox::{SandboxConfig, SandboxMode};

    fn make_ctx(workspace: &Path) -> SandboxContext {
        SandboxContext::new(SandboxConfig::new(workspace).with_mode(SandboxMode::Full))
    }

    #[tokio::test]
    async fn apply_verdict_create_invokes_skill_manage_with_agent_provenance() {
        let tmp = tempfile::tempdir().unwrap();
        let mut runtime =
            SkillsRuntime::new(tmp.path().to_path_buf(), SkillCriticConfig::default());
        let ctx = make_ctx(tmp.path());

        let verdict = SkillCriticVerdict {
            create: true,
            name: Some("auto-deploy".to_owned()),
            description: Some("Deploy a Rust crate".to_owned()),
            draft: Some("Steps:\n1. cargo publish\n".to_owned()),
            reason: "non-trivial".to_owned(),
        };

        let dir = runtime
            .apply_verdict(&verdict, &ctx)
            .await
            .unwrap()
            .expect("create=true should produce a path");
        assert!(dir.ends_with(PathBuf::from("auto-deploy")));
        let body = std::fs::read_to_string(dir.join("SKILL.md")).unwrap();
        assert!(body.contains("provenance: agent"));
        assert!(body.contains("name: auto-deploy"));

        // Budget consumed.
        assert!(!runtime.auto_creation_allowed());
    }

    #[tokio::test]
    async fn apply_verdict_skip_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let mut runtime =
            SkillsRuntime::new(tmp.path().to_path_buf(), SkillCriticConfig::default());
        let ctx = make_ctx(tmp.path());

        let verdict = SkillCriticVerdict {
            create: false,
            name: None,
            description: None,
            draft: None,
            reason: "one-off".to_owned(),
        };
        let dir = runtime.apply_verdict(&verdict, &ctx).await.unwrap();
        assert!(dir.is_none());
        // Budget untouched.
        assert!(runtime.auto_creation_allowed());
    }

    #[tokio::test]
    async fn record_use_skill_outcome_updates_metrics() {
        let tmp = tempfile::tempdir().unwrap();
        let mut runtime =
            SkillsRuntime::new(tmp.path().to_path_buf(), SkillCriticConfig::default());
        let ctx = make_ctx(tmp.path());

        // Create a skill so the metrics file has a home.
        let v = SkillCriticVerdict {
            create: true,
            name: Some("metricked".to_owned()),
            description: Some("d".to_owned()),
            draft: Some("body".to_owned()),
            reason: "r".to_owned(),
        };
        runtime.apply_verdict(&v, &ctx).await.unwrap();

        let m = runtime
            .record_use_skill_outcome("metricked", UseSkillOutcome::Success)
            .await
            .unwrap();
        assert_eq!(m.success_count, 1);

        let m2 = runtime
            .record_use_skill_outcome("metricked", UseSkillOutcome::Failure("oops".to_owned()))
            .await
            .unwrap();
        assert_eq!(m2.success_count, 1);
        assert_eq!(m2.failure_count, 1);
        assert!(m2.improvement_hints.iter().any(|h| h == "oops"));
    }

    #[tokio::test]
    async fn pending_banner_round_trip_clears_after_ack() {
        let tmp = tempfile::tempdir().unwrap();
        let mut runtime =
            SkillsRuntime::new(tmp.path().to_path_buf(), SkillCriticConfig::default());
        let ctx = make_ctx(tmp.path());

        let v = SkillCriticVerdict {
            create: true,
            name: Some("freshly-baked".to_owned()),
            description: Some("d".to_owned()),
            draft: Some("body".to_owned()),
            reason: "r".to_owned(),
        };
        runtime.apply_verdict(&v, &ctx).await.unwrap();

        let cfg = crate::config::SkillsConfig {
            compat_claude: false,
            compat_agent_skills: false,
            ..Default::default()
        };
        let pending = runtime.compute_pending_banner(&cfg).await.unwrap();
        assert!(pending.contains(&"freshly-baked".to_owned()));

        let banner = SkillsRuntime::format_pending_banner(&pending).unwrap();
        assert!(banner.starts_with('\u{26a1}'));
        assert!(banner.contains("freshly-baked"));

        runtime.acknowledge_banner(&pending).await.unwrap();
        let post = runtime.compute_pending_banner(&cfg).await.unwrap();
        assert!(
            post.is_empty(),
            "expected empty pending list after ack: {post:?}"
        );
    }

    #[test]
    fn record_turn_forwards_to_collector() {
        let mut runtime = SkillsRuntime::new(
            std::env::temp_dir(),
            SkillCriticConfig {
                max_per_session: 3,
                ..SkillCriticConfig::default()
            },
        );
        let summary = TurnSummary {
            turn_index: 0,
            tool_calls: vec![ToolCallOutcome {
                tool: "x".to_owned(),
                success: true,
            }],
            feedback_memory_written: true,
            turn_errored: false,
        };
        let signals = runtime.record_turn(&summary);
        assert!(
            signals
                .iter()
                .any(|s| s.kind == TriggerKind::UserCorrection)
        );
    }

    #[test]
    fn build_skills_turn_summary_marshalls_outcomes() {
        let summary = SkillsRuntime::build_skills_turn_summary(
            7,
            &["file_read".to_owned(), "shell_exec".to_owned()],
            &["bad_tool".to_owned()],
            false,
            true,
        );
        assert_eq!(summary.turn_index, 7);
        assert!(summary.turn_errored);
        let n_success = summary.tool_calls.iter().filter(|c| c.success).count();
        let n_fail = summary.tool_calls.iter().filter(|c| !c.success).count();
        assert_eq!(n_success, 2);
        assert_eq!(n_fail, 1);
    }
}
