#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/README.md bev_fusion/data/README.md bev_fusion/records/README.md bev_fusion/splits/README.md bev_fusion/tests/README.md bev_fusion/tests/test_gt_prepare.py bev_fusion/tests/test_splits.py bev_fusion/training/README.md bev_fusion/training/architecture.json bev_fusion/training/architecture.md bev_fusion/training/configs/README.md bev_fusion_lane_center/training/README.md bev_fusion_lane_center/training/configs/README.md bev_fusion_lane_center/training/startable.py bev_fusion_lane_center/training/train/run_config.py bev_fusion_lane_center/training/trainer.py"
git diff --stat -- $P | tail -n 20 | cut -c1-120
python3 -m py_compile bev_fusion_lane_center/training/startable.py bev_fusion_lane_center/training/train/run_config.py bev_fusion_lane_center/training/trainer.py bev_fusion/tests/test_splits.py bev_fusion/tests/test_gt_prepare.py && echo "py_compile ok"
find . -name '__pycache__' -newer bev_fusion/temp/cleanup/doc_pass_commit.sh -maxdepth 4 -prune 2>/dev/null | head -n 0
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/doc_pass_msg.txt $P || exit 1
git log --format='%h %s' -2 | cut -c1-120
