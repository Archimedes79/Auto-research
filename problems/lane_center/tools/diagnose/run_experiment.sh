#!/usr/bin/env bash
# One command for a lane experiment whose card is written and whose code is committed:
#   bash tools/diagnose/run_experiment.sh E<n>      (from the problem folder, WSL)
# check -> start -> wait until every run is finished (stops at the first failed run) -> judge --write (records not committed: the conclusion is
# written into the card first) -> the stage diagnosis of every run -> one table of the verdict and the stage numbers.
set -e
EXP=${1:?experiment id}
HERE=$(cd "$(dirname "$0")" && pwd)
PROBLEM=$(cd "$HERE/../.." && pwd)
CORE=$(cd "$PROBLEM/../../autoresearch" && pwd)
PY=${PY:-$HOME/miniconda3/envs/gcar/bin/python}
FOLDER=$(ls -d "$PROBLEM/experiments/${EXP}_"* | head -n 1)
cd "$CORE"
$PY -m research check lane_center "$EXP"
$PY -m research start lane_center "$EXP"
while true; do                      # a run.json being rewritten can be unreadable for a moment: count the ok runs, never the unread ones
  if grep -qsE '^ "status": "(failed|stopped)"' "$FOLDER"/runs/*/run.json; then
    echo "a run failed or stopped"; $PY -m research status lane_center | grep "$EXP"; exit 2
  fi
  runs=$(ls -d "$FOLDER"/runs/*/ | wc -l)
  ok=$(grep -lsE '^ "status": "ok"' "$FOLDER"/runs/*/run.json | wc -l)
  [ "$ok" -eq "$runs" ] && break
  sleep 30
done
$PY -m research judge lane_center "$EXP" --write --no-commit > /dev/null
for f in "$FOLDER"/runs/*/run.json; do
  run=$(basename "$(dirname "$f")")
  $PY -m research py lane_center -m lane_center.tools.diagnose.diagnose "$run" --worst 3 > /dev/null || echo "diagnosis of $run failed"
done
$PY -m research py lane_center -m lane_center.tools.diagnose.summary "$EXP"
