//! Modal UI for `human` workflow steps (SPEC §8.4 / issue #46).
//!
//! When the engine emits [`tmg_workflow::WorkflowProgress::HumanInputRequired`]
//! the TUI captures the event into a [`HumanPrompt`] and renders it as
//! an overlay over the chat pane until the user picks an option. The
//! response is sent back through the carried one-shot channel so the
//! workflow engine can resume.
//!
//! ## Selection model
//!
//! The prompt lists the engine-provided `options` (typically `approve`
//! / `reject` / `revise`). Selection is driven by:
//! * `Tab` / `Right` / `Down` and `Shift+Tab` / `Left` / `Up` — cycle
//!   focus through the option list.
//! * The first letter of each option (`a`, `r`, `v` for the canonical
//!   set) — jump-select that option.
//! * `Enter` — confirm the focused option and dispatch the response.
//! * `Esc` — cancel the prompt; the carried responder is dropped which
//!   the engine treats as a "no response" failure.
//!
//! ## Response mapping
//!
//! The mapping from option-string to [`tmg_workflow::HumanResponseKind`]
//! is intentionally lenient: any string starting with `app` becomes
//! `Approve`, any string starting with `rej` becomes `Reject`, any
//! string starting with `rev` becomes `Revise`. Workflow authors that
//! deviate from the canonical names will still get a sensible default.

use std::sync::Arc;

use tmg_workflow::{HumanResponder, HumanResponse, HumanResponseKind};
use tokio::sync::Mutex;

/// State for an in-flight `human` step prompt.
#[derive(Debug)]
pub struct HumanPrompt {
    /// Step id from the engine (used to address `revise` targets).
    pub step_id: String,
    /// Prompt body shown above the option list.
    pub message: String,
    /// Optional pre-rendered `show:` payload (e.g. the previous step's
    /// stdout).
    pub show: Option<String>,
    /// Allowed response keywords as the engine declared them. Each
    /// entry maps via [`option_kind`] to a typed response kind.
    pub options: Vec<String>,
    /// Currently focused option index. Always within
    /// `0..options.len()` while the prompt is alive.
    pub focused: usize,
    /// One-shot responder. Wrapped in `Arc<Mutex<Option<...>>>` per
    /// the [`HumanResponder`] contract; calling [`Self::respond`]
    /// `take()`s the inner sender.
    responder: HumanResponder,
    /// Optional `revise_target` provided by the workflow definition.
    /// When present and the user picks a `revise` option, the response
    /// carries this target.
    pub revise_target: Option<String>,
}

impl HumanPrompt {
    /// Build a prompt from a [`tmg_workflow::WorkflowProgress::HumanInputRequired`]
    /// event's fields.
    #[must_use]
    pub fn new(
        step_id: String,
        message: String,
        show: Option<String>,
        options: Vec<String>,
        responder: HumanResponder,
    ) -> Self {
        // Default focus on the first entry so a plain `Enter` picks
        // something deterministic. Empty option lists are rare in
        // practice (the workflow validator requires at least one),
        // but we still tolerate them by leaving `focused` at 0.
        Self {
            step_id,
            message,
            show,
            options,
            focused: 0,
            responder,
            revise_target: None,
        }
    }

    /// Attach a `revise_target` parsed from the workflow definition.
    #[must_use]
    pub fn with_revise_target(mut self, target: Option<String>) -> Self {
        self.revise_target = target;
        self
    }

    /// Move focus to the next option (wrap-around).
    pub fn focus_next(&mut self) {
        if self.options.is_empty() {
            return;
        }
        self.focused = (self.focused + 1) % self.options.len();
    }

    /// Move focus to the previous option (wrap-around).
    pub fn focus_prev(&mut self) {
        if self.options.is_empty() {
            return;
        }
        if self.focused == 0 {
            self.focused = self.options.len() - 1;
        } else {
            self.focused -= 1;
        }
    }

    /// Try to focus an option by its first character.
    ///
    /// Returns `true` when a match was found and focus was updated.
    /// Comparison is ASCII case-insensitive.
    pub fn focus_by_initial(&mut self, ch: char) -> bool {
        let target = ch.to_ascii_lowercase();
        for (i, opt) in self.options.iter().enumerate() {
            if let Some(first) = opt.chars().next()
                && first.to_ascii_lowercase() == target
            {
                self.focused = i;
                return true;
            }
        }
        false
    }

    /// Borrow the focused option string. Returns the empty string when
    /// the option list is empty (defensive — the caller should never
    /// see this).
    #[must_use]
    pub fn focused_option(&self) -> &str {
        self.options.get(self.focused).map_or("", String::as_str)
    }

    /// Send the carried response and consume the prompt.
    ///
    /// Returns `Ok(())` on a successful send; `Err(())` when the
    /// responder was already taken (double-take per the [`HumanResponder`]
    /// contract). The TUI logs the error and clears the prompt either
    /// way.
    pub async fn respond(self, response: HumanResponse) -> Result<(), ()> {
        let mut guard = self.responder.lock().await;
        let Some(tx) = guard.take() else {
            return Err(());
        };
        // Drop the guard before sending so the engine isn't blocked
        // waiting on the mutex in case it tries to read the responder
        // for diagnostics.
        drop(guard);
        tx.send(response).map_err(|_| ())
    }

    /// Take the responder out so the prompt can be dismissed without
    /// sending anything (e.g. on `Esc`). The engine will observe a
    /// missing response and fail the step.
    pub async fn take_responder(&self) {
        let mut guard = self.responder.lock().await;
        let _ = guard.take();
    }

    /// Convenience: build a default approval response targeted at the
    /// prompt's step. Used by tests.
    #[must_use]
    pub fn responder_handle(
        &self,
    ) -> Arc<Mutex<Option<tokio::sync::oneshot::Sender<HumanResponse>>>> {
        Arc::clone(&self.responder)
    }
}

/// Map an option string to a response kind.
///
/// Recognised prefixes:
/// * `app...` → [`HumanResponseKind::Approve`]
/// * `rej...` → [`HumanResponseKind::Reject`]
/// * `rev...` → [`HumanResponseKind::Revise`]
///
/// Anything else falls back to `Approve` so workflow authors using
/// ad-hoc affirmative options (`yes`, `ok`, ...) still see something
/// sensible.
#[must_use]
pub fn option_kind(option: &str) -> HumanResponseKind {
    let lower = option.to_ascii_lowercase();
    if lower.starts_with("rej") || lower.starts_with("no") {
        HumanResponseKind::Reject
    } else if lower.starts_with("rev") {
        HumanResponseKind::Revise
    } else {
        HumanResponseKind::Approve
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests assert with expect")]
mod tests {
    use super::*;
    use tokio::sync::oneshot;

    fn make_prompt(options: Vec<&str>) -> (HumanPrompt, oneshot::Receiver<HumanResponse>) {
        let (tx, rx) = oneshot::channel();
        let responder: HumanResponder = Arc::new(Mutex::new(Some(tx)));
        let prompt = HumanPrompt::new(
            "review".to_owned(),
            "approve?".to_owned(),
            None,
            options.into_iter().map(str::to_owned).collect(),
            responder,
        );
        (prompt, rx)
    }

    #[test]
    fn focus_cycles_forward_and_backward() {
        let (mut prompt, _rx) = make_prompt(vec!["approve", "reject", "revise"]);
        assert_eq!(prompt.focused, 0);
        prompt.focus_next();
        assert_eq!(prompt.focused, 1);
        prompt.focus_next();
        prompt.focus_next();
        assert_eq!(prompt.focused, 0);
        prompt.focus_prev();
        assert_eq!(prompt.focused, 2);
    }

    #[test]
    fn focus_by_initial_jumps() {
        // Use distinct first characters so jump-by-initial is
        // unambiguous. The TUI's canonical options are
        // approve / reject / revise — both reject & revise start with
        // 'r'. The intended UX is that pressing `r` selects the first
        // 'r' option encountered (reject) and the user can `Tab` to
        // revise; mapping a single key to a single option is the only
        // contract `focus_by_initial` makes.
        let (mut prompt, _rx) = make_prompt(vec!["approve", "build", "cancel"]);
        assert!(prompt.focus_by_initial('B'));
        assert_eq!(prompt.focused, 1);
        assert!(prompt.focus_by_initial('c'));
        assert_eq!(prompt.focused, 2);
        assert!(prompt.focus_by_initial('a'));
        assert_eq!(prompt.focused, 0);
        assert!(!prompt.focus_by_initial('x'));
    }

    #[test]
    fn focus_by_initial_picks_first_match() {
        // When multiple options share an initial, the first one wins.
        let (mut prompt, _rx) = make_prompt(vec!["approve", "reject", "revise"]);
        prompt.focus_next(); // jump off the first match so the test isn't trivial
        assert!(prompt.focus_by_initial('r'));
        assert_eq!(prompt.focused, 1, "first 'r' option should win");
    }

    #[test]
    fn option_kind_maps_canonical_strings() {
        assert_eq!(option_kind("approve"), HumanResponseKind::Approve);
        assert_eq!(option_kind("Approve"), HumanResponseKind::Approve);
        assert_eq!(option_kind("reject"), HumanResponseKind::Reject);
        assert_eq!(option_kind("revise"), HumanResponseKind::Revise);
        assert_eq!(option_kind("no"), HumanResponseKind::Reject);
        assert_eq!(option_kind("yes"), HumanResponseKind::Approve);
    }

    #[tokio::test]
    async fn respond_delivers_through_oneshot() {
        let (prompt, rx) = make_prompt(vec!["approve", "reject", "revise"]);
        prompt.respond(HumanResponse::approve()).await.expect("ok");
        let resp = rx.await.expect("response received");
        assert_eq!(resp.kind, HumanResponseKind::Approve);
    }

    #[tokio::test]
    async fn double_respond_returns_err() {
        let (prompt, _rx) = make_prompt(vec!["approve"]);
        let handle = prompt.responder_handle();
        // Manually drain so the second respond observes the empty slot.
        let mut guard = handle.lock().await;
        let _ = guard.take();
        drop(guard);
        let err = prompt.respond(HumanResponse::approve()).await;
        assert!(err.is_err());
    }
}
