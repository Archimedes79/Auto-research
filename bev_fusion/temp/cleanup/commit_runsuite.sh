#!/bin/bash
cd /mnt/c/Auto-research || exit 1
python3 bev_fusion/temp/cleanup/blobs_runsuite.py || exit 1
P="autoresearch/runsuite/README.md autoresearch/runsuite/__init__.py autoresearch/runsuite/__main__.py autoresearch/runsuite/api.py autoresearch/runsuite/cli.py autoresearch/runsuite/engine.py autoresearch/runsuite/jobs.py autoresearch/runsuite/selftest.py \
bev_fusion/project.json \
bev_fusion/records/jobs/README.md bev_fusion/records/jobs/running/README.md bev_fusion/records/jobs/running/pred_t20_small.json bev_fusion/records/jobs/evaluation/README.md \
bev_fusion/tests/test_runner_predict.py"
git status --short -- $P autoresearch/trainsuite/api.py | cat
python3 autoresearch/tools/check_git.py --paths $P autoresearch/README.md bev_fusion/tests/README.md autoresearch/trainsuite/api.py || exit 1
echo "--- ready; blob for trainsuite/api.py must be made by hand (Training chat has hunks there)"
