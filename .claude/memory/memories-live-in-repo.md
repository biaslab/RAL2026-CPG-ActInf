---
name: memories-live-in-repo
description: Memory files belong in the repo at .claude/memory/, symlinked from the harness path — other agents read them
metadata:
  type: feedback
---

Memory files must live **inside the project's git repo** at `<repo>/.claude/memory/`,
not in the default `~/.claude/projects/<slug>/memory/`.

**Why:** other agents work in these repos and need access to the memories. A store
under `~/.claude` is invisible to them and untracked by git, so knowledge written
there cannot be shared or reviewed.

**How to apply:** keep the real files in `<repo>/.claude/memory/` and make
`~/.claude/projects/<slug>/memory` a symlink to that directory — the harness loads
memory from the per-project path, so the symlink is what preserves auto-loading into
context each session. Set this up in any new project. Check `.claude/memory/` is not
gitignored, and flag uncommitted memory files when reporting repo state, since they
only reach other agents once committed. The same rule is recorded in
`~/.claude/CLAUDE.md` so it applies across projects.
