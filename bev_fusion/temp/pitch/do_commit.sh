#!/bin/bash
cd /mnt/c/Auto-research
PY="$HOME/miniconda3/envs/gcar/bin/python"
$PY autoresearch/tools/commit_paths.py -F bev_fusion/temp/pitch/commit_sets_msg.txt \
  bev_fusion/evaluation/bev_sets.py bev_fusion/evaluation/bev_adapter.py bev_fusion/evaluation/bev_suite.py bev_fusion/evaluation/evaluate.py bev_fusion/evaluation/bev_selftest.py \
  bev_fusion/evaluation/README.md bev_fusion/tests/test_evaluate_door.py bev_fusion/tests/run_gates.sh bev_fusion/splits/README.md bev_fusion/FROZEN.md bev_fusion/frozen.json \
  --blob bev_fusion/tests/README.md=bev_fusion/temp/pitch/tests_README_head.md
