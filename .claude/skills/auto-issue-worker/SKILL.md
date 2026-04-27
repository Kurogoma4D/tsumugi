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
tsumugi is a local-LLM-powered coding agent written in Rust (2024 edition, Cargo workspace, MSRV 1.85.0+).
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

- The agent will create a worktree, implement the change, run quality checks (`cargo build/clippy --all-targets --all-features/fmt/test --workspace`, plus `cargo deny check` if dependencies changed), and create a PR.
- Capture the resulting PR number/URL from the agent's output.

### Step 2.5 — Runtime Verification (動作確認)

PR が作成されたら、ワンショット **と** TUI の両方で動作確認を行う。
LLM サーバーが稼働している場合のみ実行する。

```bash
# LLM サーバー疎通確認
curl -s --max-time 5 "${TMG_LLM_ENDPOINT:-http://localhost:8080}/health" && echo "LLM available" || echo "LLM unavailable"
```

**LLM が利用不可能な場合:** 両方スキップして「LLM unavailable — runtime verification skipped」と記録する。

**LLM が利用可能な場合、以下を順に実行:**

#### A. ワンショットモード検証

issue の内容に応じた適切なテストプロンプトで実行:

```bash
timeout 120 ./target/debug/tmg --prompt "<issue に関連するプロンプト>" --event-log /tmp/tmg-issue-<number>-oneshot.jsonl 2>&1
```

イベントログを読んで検証:
- `token` イベントが存在するか
- `is_error: true` のイベントがないか
- `done` イベントで正常終了しているか

#### B. TUI モード検証

`expect` で TUI を起動し、テストプロンプトを送って動作を検証する:

```bash
expect -c '
  set timeout 120
  spawn ./target/debug/tmg --event-log /tmp/tmg-issue-<number>-tui.jsonl
  sleep 3
  send "<issue に関連するプロンプト>\r"
  sleep 30
  send "\x03"
  expect eof
' 2>&1
```

イベントログ (`/tmp/tmg-issue-<number>-tui.jsonl`) を読んで検証:
- ワンショットと同様に `token` / `done` イベントの存在確認
- `is_error: true` がないこと
- TUI 経由でもエージェントループが正常に動作していること

**検証に失敗した場合:**
1. issue-implementer エージェントに修正を依頼
2. 修正後に再度 runtime verification を実行 (ワンショット + TUI)
3. 最大 2 回のリトライで解決しない場合はユーザーにフラグを立てて次の issue に進む

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
- Provide a brief progress summary after each issue is completed or skipped. Include runtime verification results (PASS / FAIL / SKIP).
- At the end, provide a final summary of all issues processed and their outcomes, including runtime verification status for each.
