#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion/data/cache/gt
find . -maxdepth 1 -name '*_6a7aa7864a.npz' -newer /mnt/c/Auto-research/bev_fusion/temp/lane_tests/paint_out/paint_result.json -printf '%f\n' > /mnt/c/Auto-research/bev_fusion/temp/overview/rp_newer_files.txt
wc -l /mnt/c/Auto-research/bev_fusion/temp/overview/rp_newer_files.txt
