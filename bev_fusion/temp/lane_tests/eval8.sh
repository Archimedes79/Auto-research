#!/bin/bash
# scratch (diagnostic): the standard evaluation of one checkpoint on the 8 TRAIN scenes of the overfit test (all their frames), into a run folder of its own.
#   bash temp/lane_tests/eval8.sh <run folder> <checkpoint folder>      (from anywhere; run.py starts it)
# For an existing run the folder is a pseudo run (config, gt.json and code copied) so that eval/ of the original run (its val and train tables) is not overwritten.
cd "$(dirname "$0")/../.." || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
run=${1%/}; ck=${2%/}
SC=$(cat temp/lane_tests/overfit8_scenes.txt)
if [ "$run" != "$ck" ]; then
  mkdir -p "$run" && cp "$ck/config.json" "$ck/gt.json" "$run/" && { [ -d "$ck/code" ] && cp -r "$ck/code" "$run/"; true; } || exit 1
fi
$PY evaluation/bev_suite.py "$run" --ckpt-dir "$ck" --split train --workers 2 eval --scenes $SC
rc=$?
echo "eval8 $run exit $rc $(date +%T)"
exit $rc
