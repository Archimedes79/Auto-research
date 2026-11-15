#!/bin/bash
# which run_gates.sh processes exist and where they run
for p in $(pgrep -f "tests/run_gates.sh"); do
  echo "$p cwd=$(readlink /proc/$p/cwd) etime=$(ps -o etime= -p $p | tr -d ' ') ppid=$(ps -o ppid= -p $p | tr -d ' ')"
done
echo "--- children with GIT_DIR in their environment:"
for p in $(pgrep -f "gates_integration|run_gates|python"); do
  if tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q '^GIT_WORK_TREE=/mnt/c/Auto-research/.claude/worktrees/integration'; then echo "$p $(ps -o etime=,cmd= -p $p | cut -c1-110)"; fi
done
