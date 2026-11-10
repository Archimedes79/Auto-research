#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
echo "== BEV:"; AR_PROJECT=../bev_fusion python3 ../autoresearch/tools/check_structure.py 2>&1 | cut -c1-170 | tail -n 25
cd ../bev_fusion_lane_center || exit 1
echo "== LANE (lines naming files I touched):"; AR_PROJECT=../bev_fusion_lane_center python3 ../autoresearch/tools/check_structure.py 2>&1 | grep -E 'training/(README|configs|startable|trainer|train/)' | cut -c1-170
