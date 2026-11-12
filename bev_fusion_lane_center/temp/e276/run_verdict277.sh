#!/bin/bash
# scratch: verdict277 by hand (the scheduler left its loop at "every job is done" before it reached the verdict block)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
{ echo "== verdict277 u against B and against B + hard step (by hand, scheduler loop ended first) $(date)"; OMP_NUM_THREADS=2 bash tools/env/_py14.sh qgraph_run_eval.py verdict277 --var e277_u_s0.plain_t0.5 e277_u_s1.plain_t0.5 --ctl e273_p_s0 e273_p_s1 --hard e273_p_s0.close1.5_t0.5 e273_p_s1.close1.5_t0.5; } >> temp/e276/verdicts.log 2>&1
touch temp/e276/VERDICT_U_DONE
tail -90 temp/e276/verdicts.log
