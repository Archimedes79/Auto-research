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
while true; do
  states=$(for f in "$FOLDER"/runs/*/run.json; do sed -n 's/^ "status": "\([a-z]*\)".*/\1/p' "$f"; done)
  echo "$states" | grep -qE 'failed|stopped' && { echo "a run failed or stopped"; $PY -m research status lane_center | grep "$EXP"; exit 2; }
  echo "$states" | grep -qvE '^ok$' || break
  sleep 30
done
$PY -m research judge lane_center "$EXP" --write --no-commit > /dev/null
for f in "$FOLDER"/runs/*/run.json; do
  run=$(basename "$(dirname "$f")")
  (cd "$PROBLEM" && $PY tools/diagnose/diagnose.py "$run" --worst 3 > /dev/null) || echo "diagnosis of $run failed"
done
$PY "$HERE/summary.py" "$EXP"
