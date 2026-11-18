#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
L=temp/_session_scratch_e291
bash temp/e291/run_staged.sh jprior.py > $L/final_jprior.log 2>&1; echo "exit $?" >> $L/final_jprior.log
bash temp/e291/run_staged.sh jends_eval.py selftest > $L/final_eval_selftest.log 2>&1; echo "exit $?" >> $L/final_eval_selftest.log
bash temp/e291/run_staged.sh test_e291_staged.py --only identity_random,inject,keys,identity_real > $L/final_core.log 2>&1; echo "exit $?" >> $L/final_core.log
