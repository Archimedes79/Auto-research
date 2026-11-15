#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/tests/test_bev_free.py bev_fusion/tests/test_evaluate_door.py bev_fusion/tests/test_evaluation_api.py bev_fusion/tests/test_runner_real.py bev_fusion/tests/run_gates.sh bev_fusion/tests/README.md bev_fusion/README.md bev_fusion/runs/README.md bev_fusion/training/configs/README.md bev_fusion/training/configs/t20_base_nol4.json"
git diff --stat -- $P | tail -n 12 | cut -c1-110
bash -n bev_fusion/tests/run_gates.sh && echo "run_gates.sh syntax ok"
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/gates_msg.txt $P || exit 1
git log --format='%h %s' -1 | cut -c1-120
