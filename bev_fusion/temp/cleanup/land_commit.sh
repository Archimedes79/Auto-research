#!/bin/bash
cd /mnt/c/Auto-research || exit 1
S=bev_fusion/temp/cleanup
python3 autoresearch/tools/commit_paths.py --allow-scratch -F $S/land_msg.txt \
  --blob .gitignore=$S/blob_.gitignore \
  --blob bev_fusion/temp/README.md=$S/blob_bev_fusion_temp_README.md \
  --blob bev_fusion/tests/README.md=$S/blob_bev_fusion_tests_README.md \
  $(cat $S/landing_paths.txt | tr '\n' ' ') 2>&1 | tail -15
echo "rc=${PIPESTATUS[0]}"
git log -1 --format='%h %s' | cut -c1-100
