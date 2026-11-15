#!/bin/bash
# read-only: what does git in WSL see now that core.worktree is set in the shared config?
cd /mnt/c/Auto-research || exit 1
echo "toplevel: $(git rev-parse --show-toplevel 2>&1)"
echo "git-dir:  $(git rev-parse --git-dir 2>&1)"
echo "status lines (main checkout vs HEAD, first 5):"; git status --short 2>&1 | head -5
echo "worktrees:"; git worktree list 2>&1 | head -6
echo "config worktree: $(git config --get core.worktree)"
