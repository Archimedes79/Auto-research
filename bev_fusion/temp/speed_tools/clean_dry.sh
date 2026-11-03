#!/bin/bash
# dry run of clean_temp.py for my two obsolete scratch trees (the speed patch is committed in 0b8da1c; the HEAD copy can be exported from git again)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
for n in speed speed_base; do
  echo "=== clean_temp $n (dry run)"
  timeout 120 python3 ../autoresearch/tools/clean_temp.py $n 2>&1 | tail -14 | cut -c1-200
done
