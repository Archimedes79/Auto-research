#!/bin/bash
# U6 scratch (after the U2-U5 critic's should-fixes in the WORKTREE: jlive refusal test, load_model names the run, with_tokens_np out, jtok2 text):
# test_jtoken, the jlive check (selftest, real check, mutants), test_jtoken_eval (jtiming stand-in from the mirror) and the repro, all against the worktree's modules and the live data
cd "$(dirname "$0")/../.."
export U2_WT=/mnt/c/Auto-research_wt_simplify/bev_fusion_lane_center CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export U2_EXTRA="jtiming=/mnt/c/Auto-research_u5_tree/bev_fusion_lane_center/evaluation/jtiming.py"
for s in training/grid/test_jtoken.py temp/simplify/u_jlive_check.py evaluation/tests/test_jtoken_eval.py "temp/simplify/repro_score.py u6"; do
  n=$(basename ${s%% *} .py)
  echo "=== $s"
  nice -n 10 bash tools/env/_py14.sh ux_run_with.py $s > temp/simplify/u6_$n.out 2>&1
  rc=$?
  grep -v "^u2_run" temp/simplify/u6_$n.out | grep -iE "BIT|metrics|ok$| ok|skip|Error|mutant|FAIL" | tail -n 8 | cut -c1-300
  echo "rc $rc"
done
echo "u6 done"
