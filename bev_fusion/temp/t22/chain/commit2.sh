#!/bin/bash
cd /mnt/c/Auto-research || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/t22/chain/commit_msg2.txt bev_fusion/training/model.py bev_fusion/training/README.md bev_fusion/tests/test_student_chain.py bev_fusion/tests/README.md
