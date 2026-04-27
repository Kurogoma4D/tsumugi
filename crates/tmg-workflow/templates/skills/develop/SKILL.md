# develop

A SPEC §8.5 method-1 sample skill for tsumugi.

This skill encapsulates the standard "implement -> verify -> review"
loop that most coding tasks need.  Drop it into
`.tsumugi/skills/develop/` and reference it from a workflow or
subagent prompt.

## When to use

Use when the user asks for a code change that:

- Has clear acceptance criteria (e.g. "tests should pass").
- Touches a contained area of the codebase (one or two crates).
- Does not require deep architectural decisions.

## Steps

1. Plan: enumerate the files to touch and the order to do them in.
2. Implement: produce a minimal change set.
3. Verify: run `cargo test --workspace` (or the project's equivalent).
4. Review: re-read the diff with a reviewer mindset; fix obvious
   issues.
5. Commit: prepare a Conventional-Commits-style message.

## Allowed tools

- `file_read`, `file_write`, `file_patch`
- `grep_search`, `list_dir`
- `shell_exec`

## Hand-off

When you finish, return:

- A short summary of the change.
- The exit code of the verify step.
- A list of files changed.
