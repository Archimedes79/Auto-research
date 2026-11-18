#!/bin/bash
# D121: the restored S2 hook of jtoken.py (worktree C:/Auto-research_wt_s2hook) against the live data: test_jtoken (incl. test_past), the jlive check, test_jtoken_eval, the repro
cd "$(dirname "$0")/../.."
export U2_WT=/mnt/c/Auto-research_wt_s2hook/bev_fusion_lane_center CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in "training/grid/test_jtoken.py" temp/simplify/u_jlive_check.py evaluation/tests/test_jtoken_eval.py "temp/simplify/repro_score.py s2h"; do
  n=$(basename ${s%% *} .py)
  echo "=== $s"
  nice -n 10 bash tools/env/_py14.sh temp/simplify/ux_run_with.py $s > temp/simplify/s2h_$n.out 2>&1
  rc=$?
  grep -v "^u2_run" temp/simplify/s2h_$n.out | grep -iE "^past:|BIT|metrics|ok$|Error|FAIL" | tail -n 5 | cut -c1-300
  echo "rc $rc"
done
echo "s2hook tests done"
