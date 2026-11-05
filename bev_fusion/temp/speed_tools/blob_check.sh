#!/bin/bash
cd /mnt/c/Auto-research || exit 1
python3 bev_fusion/temp/speed_tools/readme_rows.py
for f in training tests; do
  echo "== blob $f README vs HEAD: changed lines"
  git show HEAD:bev_fusion/$f/README.md > /tmp/head_$f.md
  diff /tmp/head_$f.md bev_fusion/temp/blob_${f}_README.md | grep '^[<>]' | cut -c1-110
done
