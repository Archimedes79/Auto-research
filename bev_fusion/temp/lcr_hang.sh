#!/bin/bash
cd /mnt/c/Auto-research || exit 1
ls .git/hooks | grep -v sample
echo "--- processes under the push:"
ps -eo pid,ppid,etime,pcpu,stat,cmd --sort=-etime | grep -E "git push|pre-push|git-remote|ssh|credential" | grep -v grep | head -8 | cut -c1-180
echo "--- hook:"; [ -f .git/hooks/pre-push ] && head -20 .git/hooks/pre-push | cut -c1-160
git config --get-regexp 'core.hooksPath|credential' | head
