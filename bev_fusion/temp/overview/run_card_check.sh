#!/bin/bash
cd /mnt/c/Auto-research || exit 1
sed -i 's/\r$//' bev_fusion/temp/lane_tests/chain8.sh
bash -n bev_fusion/temp/lane_tests/chain8.sh && echo "chain8 syntax ok"
AR_PROJECT=../bev_fusion python3 autoresearch/tools/experiment.py check t24_lane_pull 2>&1 | tail -n 25
