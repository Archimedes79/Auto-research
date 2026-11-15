#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/experiments/t25_one_frame_one_scan/experiment.md bev_fusion/tests/README.md bev_fusion/tests/run_gates.sh bev_fusion/training/README.md bev_fusion/training/augment.py bev_fusion/training/configs/README.md bev_fusion/training/interface.py bev_fusion/training/model.py bev_fusion/training/nuscenes_dataset.py bev_fusion/training/precompute.py bev_fusion/training/train.py bev_fusion/tests/test_radar_recursive.py bev_fusion/training/configs/t25_ctl.json bev_fusion/training/configs/t25_rec_r.json bev_fusion/training/configs/t25_rec_scan.json bev_fusion/training/configs/t25_rec_scan_mem.json"
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 2
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/t25code_msg.txt $P || exit 1
git log --format='%h %s' -1 | cut -c1-130
git fetch origin --quiet; echo "behind/ahead: $(git rev-list --left-right --count origin/main...HEAD)"
