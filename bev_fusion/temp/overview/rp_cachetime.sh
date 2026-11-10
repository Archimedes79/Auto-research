#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion/data/cache/gt
ref=/mnt/c/Auto-research/bev_fusion/temp/lane_tests/paint_out/paint_result.json
ls -l --time-style=long-iso $ref
echo "files with the signature:"
find . -maxdepth 1 -name '*_6a7aa7864a.npz' | wc -l
echo "newer than the paint result:"
find . -maxdepth 1 -name '*_6a7aa7864a.npz' -newer $ref | wc -l
echo "mtime day histogram:"
find . -maxdepth 1 -name '*_6a7aa7864a.npz' -printf '%TY-%Tm-%Td\n' | sort | uniq -c
