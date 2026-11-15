#!/bin/bash
# stop only MY gate run (cwd in the integration worktree, GIT_WORK_TREE of it in the environment)
mine=""
for p in $(pgrep -f "run_gates|test_|gates_integration|python"); do
  if tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q '^GIT_WORK_TREE=/mnt/c/Auto-research/.claude/worktrees/integration$'; then mine="$mine $p"; fi
done
echo "mine:$mine"
[ -n "$mine" ] && kill $mine
sleep 2
for p in $mine; do kill -0 $p 2>/dev/null && { echo "still alive $p, kill -9"; kill -9 $p; }; done
echo "left:"; pgrep -af "tests/run_gates.sh" | cut -c1-100
