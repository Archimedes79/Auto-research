#!/bin/bash
cd /mnt/c/Auto-research/autoresearch || exit 1
git log --oneline -3
git status --short -- evalsuite runsuite | head
echo "=== diff runner_api.py (Eval) vs predict.py (mine), ignoring the renames"
diff <(sed -e 's/def run(/def predict(/; s/def check(/def plan(/; s/def list(/def list_results(/' evalsuite/runner_api.py) runsuite/predict.py | cut -c1-230 | head -150
echo "=== Helpers._source in evalsuite/api.py"
grep -n "_source" evalsuite/api.py | head
