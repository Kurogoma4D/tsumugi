---
name: auto-issue-worker
description: |
  Automatically consume open GitHub issues one by one. Fetches the oldest open issue,
  delegates implementation to the issue-implementer agent, runs code review via the
  code-reviewer agent, iterates on feedback, merges the PR, and moves on to the next issue.
  Invoke with `/auto-issue-worker`.
allowed-tools:
  - Bash
  - Task
---

# Auto Issue Worker

You are an autonomous issue-processing pipeline for the **tsumugi** repository (`Kurogoma4D/tsumugi`).
tsumugi is a local-LLM-powered coding agent written in Rust (Cargo workspace).
Your job is to pick up the oldest open issue, implement it, get it reviewed, merge the PR, and repeat
until no open issues remain.

## Workflow

Execute the following loop until there are no more open issues:

### Step 1 — Pick the next issue

```bash
gh issue list --repo Kurogoma4D/tsumugi --state open --limit 1 -S "sort:created-asc" --json number,title,labels
```

- If the result is empty, report "All issues are resolved" and stop.
- Otherwise, note the issue `number` and `title`.
- Check the issue's dependency section (依存). If it depends on unresolved issues, skip it and pick the next one.

### Step 2 — Implement the issue

Delegate implementation to the **github-issue-implementer** agent:

```
Task tool:
  subagent_type: github-issue-implementer
  prompt: "Implement issue #<number> for the tsumugi repository (Kurogoma4D/tsumugi)."
```

- The agent will create a worktree, implement the change, run quality checks (`cargo build/clippy/fmt/test --workspace`), and create a PR.
- Capture the resulting PR number/URL from the agent's output.

### Step 3 — Review the PR

Delegate code review to the **code-reviewer** agent:

```
Task tool:
  subagent_type: general-purpose
  prompt: |
    You are a code reviewer. Follow the instructions in .claude/agents/code-reviewer.md.
    Review PR #<pr-number> in the Kurogoma4D/tsumugi repository.
    Return a list of issues found, or "LGTM" if the code is acceptable.
```

### Step 4 — Fix review feedback (if any)

If the reviewer returned issues (not "LGTM"):

1. Resume the issue-implementer agent (or launch a new one) to address each review comment.
2. After fixes are pushed, go back to **Step 3** to re-review.
3. Repeat until the reviewer returns "LGTM".
4. Limit the review loop to **3 iterations** to prevent infinite cycles. If not resolved after 3 rounds, flag the issue to the user and move on.

### Step 5 — Merge the PR

```bash
gh pr merge <pr-number> --repo Kurogoma4D/tsumugi --squash --delete-branch
```

- Confirm the merge was successful.
- If merge fails (e.g., CI not passing), report the failure to the user and move on.

### Step 6 — Loop

Go back to **Step 1** to pick the next issue.

## Rules

- Always confirm each step's outcome before proceeding to the next.
- If any step fails unrecoverably, report the failure clearly and move on to the next issue.
- Do not modify issues that have the `wontfix` or `on-hold` label — skip them.
- Respect issue dependencies (依存 section): skip issues whose dependencies are not yet closed.
- Provide a brief progress summary after each issue is completed or skipped.
- At the end, provide a final summary of all issues processed and their outcomes.
