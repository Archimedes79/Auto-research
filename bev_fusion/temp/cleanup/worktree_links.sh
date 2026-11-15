#!/bin/bash
# list the links inside the three worktrees (look only unless called with --rm)
W=/mnt/c/Auto-research/.claude/worktrees
for w in agent-a9bab4e4edc921066 integration agent-a6afb9dbe32b94670; do
  echo "=== $w"
  find $W/$w -type l -printf '%p -> %l\n' 2>/dev/null | grep -v '/__pycache__/' | head -20
done
if [ "$1" = "--rm" ]; then
  for w in agent-a9bab4e4edc921066 integration; do
    find $W/$w -type l ! -path '*/__pycache__/*' -print0 2>/dev/null | while IFS= read -r -d '' l; do rm "$l" && echo "unlinked $l"; done
  done
fi
echo "--- big files in coder A temp/deadcode"
find $W/agent-a9bab4e4edc921066/bev_fusion/temp/deadcode -type f -size +20M -printf '%s %p\n' 2>/dev/null | sort -n | tail -8
if [ "$1" = "--rm" ]; then
  mv $W/agent-a9bab4e4edc921066/bev_fusion/temp/deadcode /mnt/c/Auto-research/bev_fusion/temp/cleanup/coderA_deadcode && echo "moved coder A deadcode scratch to temp/cleanup/coderA_deadcode"
fi
