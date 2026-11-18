#!/bin/bash
# E385 no-run check: stage 2 (jlive s4) 20 CPU steps, the landed code against the code of tag e3df44fb (temp/e385/old_tree: the 4 token modules of the tag), seed 0
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for t in new old; do
  if [ $t = new ]; then export U2_WT=$(pwd); else export U2_WT=$(pwd)/temp/e385/old_tree/bev_fusion_lane_center; fi
  echo "=== $t ($U2_WT)"
  nice -n 10 bash tools/env/_py14.sh temp/simplify/ux_run_with.py temp/e385/eq_s4.py 0 > temp/e385/eq_s4_$t.out 2>&1
  echo "rc $?"; grep -E "^EQ|Error|assert" temp/e385/eq_s4_$t.out | cut -c1-400
done
echo "eq_s4 done"
