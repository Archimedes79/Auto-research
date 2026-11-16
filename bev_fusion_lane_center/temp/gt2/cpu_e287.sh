#!/bin/bash
# CPU analyses on the E287 dumps (one thread, low priority): rule control on the same keypoints, precision-recall against the E284 head, the AUC split, failure splits of p1
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "start $(date)"
nice -n 10 bash tools/env/_py14.sh jrule_control.py score --run e287_p0_s0 e287_p1_s0 e287_p2_s0 e287_p0_s1 e287_p1_s1 e287_p2_s1 --kp-thr 0.5 > temp/gt2/e287_rule.log 2>&1; echo "rule done $(date)"
for S in 0 1; do
  for A in p0 p1 p2; do
    nice -n 10 bash tools/env/_py14.sh jpair_pr.py score --run e287_${A}_s$S --control e284_h2_s$S > temp/gt2/e287_pr_${A}_s$S.log 2>&1; echo "pr $A $S done $(date)"
  done
done
nice -n 10 bash tools/env/_py14.sh ../temp/gt2/pair_diag.py e287_p0_s0 e287_p1_s0 e287_p2_s1 > temp/gt2/e287_diag.log 2>&1; echo "diag done $(date)"
nice -n 10 bash tools/env/_py14.sh jsplits.py score --run e287_p1_s0 e287_p1_s1 --kp-thr 0.5 > temp/gt2/e287_splits.log 2>&1; echo "splits done $(date)"
echo CPU_E287_DONE
