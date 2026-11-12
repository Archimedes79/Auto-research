#!/bin/bash
# early CPU read of E287 p2 seed 0 (one thread): the dump (the chain re-dumps it later), the rule control, the failure splits, the precision-recall curve
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "start $(date)"
nice -n 10 bash tools/env/_py14.sh jends_eval.py dump --run e287_p2_s0 > temp/gt2/early_dump_p2_s0.log 2>&1; echo "dump done $(date)"
nice -n 10 bash tools/env/_py14.sh jrule_control.py score --run e287_p2_s0 --kp-thr 0.5 > temp/gt2/early_rule_p2_s0.log 2>&1; echo "rule control done $(date)"
nice -n 10 bash tools/env/_py14.sh jpair_pr.py score --run e287_p2_s0 --control e284_h2_s0 > temp/gt2/early_pr_p2_s0.log 2>&1; echo "pr done $(date)"
echo EARLY_DONE
