#!/bin/bash
cd /mnt/c/Auto-research || exit 1
python3 autoresearch/tools/check_git.py --paths bev_fusion/training/model.py bev_fusion/training/feature_cache.py bev_fusion/training/student_encoder.py bev_fusion/training/configs/t22_B_s0.json bev_fusion/tests/test_student_chain.py bev_fusion/tests/README.md bev_fusion/experiments/t22_student_encoder/experiment.md 2>&1 | tail -8
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/t22/commit_msg3.txt bev_fusion/training/model.py bev_fusion/training/feature_cache.py bev_fusion/training/student_encoder.py bev_fusion/training/configs/t22_B_s0.json bev_fusion/tests/test_student_chain.py bev_fusion/tests/README.md bev_fusion/experiments/t22_student_encoder/experiment.md 2>&1 | tail -8
