#!/bin/bash
cd /mnt/c/Auto-research
P="bev_fusion_lane_center/training/grid/jmaps.py bev_fusion_lane_center/training/train/jmaps_train.py bev_fusion_lane_center/training/grid/test_jmaps_dirx.py bev_fusion_lane_center/training/grid/README.md bev_fusion_lane_center/training/README.md bev_fusion_lane_center/tools/maintenance/run_all_tests.sh bev_fusion_lane_center/experiments/E319_heading-guided-exchange/experiment.md"
python3 autoresearch/tools/check_git.py --no-remote --paths $P 2>&1 | tail -15
echo "--- $1"
if [ "$1" = "dry" ]; then python3 autoresearch/tools/commit_paths.py --dry-run -m "E319 dirx" $P 2>&1 | tail -15; fi
if [ "$1" = "commit" ]; then python3 autoresearch/tools/commit_paths.py -F bev_fusion_lane_center/temp/e319/commit_msg.txt $P 2>&1 | tail -15; fi
