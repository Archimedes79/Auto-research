#!/bin/bash
# U5 scratch: after the timing edit (JRT.timed -> JTG.timed): test_jtoken_eval and the repro on the mirror tree, jtiming from the mirror (its `timed` is a stand-in for the lane chat's landing)
cd "$(dirname "$0")/../.."
export U2_WT=/mnt/c/Auto-research_u5_tree/bev_fusion_lane_center CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export U2_EXTRA="jtiming=$U2_WT/evaluation/jtiming.py"
for s in evaluation/tests/test_jtoken_eval.py "temp/simplify/repro_score.py u5c"; do
  echo "=== $s"
  nice -n 10 bash tools/env/_py14.sh ux_run_with.py $s > temp/simplify/u5c_last.out 2>&1
  rc=$?
  grep -v "^u2_run" temp/simplify/u5c_last.out | grep -E "^timing|BIT|metrics|ok$|Error" | tail -n 6 | cut -c1-400
  grep "^u2_run_with: imported" temp/simplify/u5c_last.out | cut -c1-400
  echo "rc $rc"
done
echo "u5c done"
