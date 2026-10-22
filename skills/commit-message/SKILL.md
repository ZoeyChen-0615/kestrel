---
name: commit-message
description: Write a clear Conventional Commits message for the staged changes in a git repository.
requires:
  bins: [git]
---

# Commit message

Use when the user asks for a commit message (or to commit) for their current changes.

1. Run `git diff --staged --stat`, then `git diff --staged` to read what actually changed.
   If nothing is staged, say so instead of guessing.
2. Write the message in Conventional Commits form: `type(scope): summary`.
   - `type` is one of feat, fix, docs, refactor, test, chore.
   - The summary is imperative, lower-case, and under 72 characters.
3. Add a short body only if the *why* is not obvious from the summary.
4. Show the message and ask before running `git commit`.
