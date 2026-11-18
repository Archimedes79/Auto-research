#!/bin/bash
# E358 scratch, arm t (the fixture): the zero-parameter twin on the ATTACHED ends through the staged scorer, both seeds, one after the other (CPU, nice, 2 threads)
cd /mnt/c/Auto-research/bev_fusion_lane_center
for S in 0 1; do
  bash temp/e358/run_staged.sh jends_eval.py score --run e324_s_s$S --seed $S --thr 0.1 --attach --snap --snap-nogeom --wire twin --tag twinatt > temp/e358/twin_s$S.log 2>&1
  echo "seed $S exit $?" >> temp/e358/twin_done.txt
done
