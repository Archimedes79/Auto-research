#!/usr/bin/env bash
# One-off watcher of training logs that never holds a file open (a Git Bash `tail -F` outlives its monitor on Windows and
# locks the run folder): every 30 s print the new lines of runs/<name>/train.log that matter; exit when every named run has
# finished (.done or an "exit" line from run_cfg.sh is not visible here, so: a Traceback, or the final checkpoint step).
# Usage: bash temp/train/watch.sh <name> [<name> ...]
cd "$(dirname "$0")/../.." || exit 1
declare -A seen
for n in "$@"; do seen[$n]=0; done
pat='val total|Traceback|Error|error|non-finite|Killed|out of memory|^step 100:'
while true; do
  alive=0
  for n in "$@"; do
    f=runs/$n/train.log
    [ -f "$f" ] || { alive=1; continue; }
    total=$(wc -l < "$f")
    if [ "$total" -gt "${seen[$n]}" ]; then
      sed -n "$((seen[$n] + 1)),${total}p" "$f" | grep -E "$pat" | sed "s/^/[$n] /"
      seen[$n]=$total
    fi
    if [ -f runs/$n/.done ]; then echo "[$n] finished (.done)"; seen[$n]=999999999; continue; fi
    if grep -q Traceback "$f"; then echo "[$n] crashed"; seen[$n]=999999999; continue; fi
    [ "${seen[$n]}" -lt 999999999 ] && alive=1
  done
  [ $alive -eq 0 ] && exit 0
  sleep 30
done
