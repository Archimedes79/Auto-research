#!/bin/bash
cd /mnt/c/Auto-research
PY="$HOME/miniconda3/envs/gcar/bin/python"
P="bev_fusion/evaluation/bev_eval.py bev_fusion/evaluation/evalsuite.json bev_fusion/evaluation/bev_sets.py bev_fusion/evaluation/README.md bev_fusion/tests/test_evaluation_api.py bev_fusion/tests/run_gates.sh"
$PY autoresearch/tools/check_git.py --paths $P bev_fusion/tests/README.md | tail -3
$PY autoresearch/tools/commit_paths.py -F bev_fusion/temp/pitch/commit_b_msg.txt $P --blob bev_fusion/tests/README.md=bev_fusion/temp/pitch/tests_README_head4.md 2>&1 | tail -10
