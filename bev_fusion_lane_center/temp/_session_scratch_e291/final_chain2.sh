#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
L=temp/_session_scratch_e291
python3 temp/e291/test_e291_guards.py > $L/r2_guards.log 2>&1; echo "exit $?" >> $L/r2_guards.log
bash temp/e291/run_staged.sh jprior.py > $L/r2_jprior.log 2>&1; echo "exit $?" >> $L/r2_jprior.log
bash temp/e291/run_staged.sh test_e291_staged.py --only inject,identity_random,keys,shuffle,identity_real,bfsig,dump > $L/r2_core.log 2>&1; echo "exit $?" >> $L/r2_core.log
bash temp/e291/run_staged.sh test_e291_staged.py --only features_real,smoke > $L/r2_heavy.log 2>&1; echo "exit $?" >> $L/r2_heavy.log
bash temp/e291/run_staged.sh jends_eval.py selftest > $L/r2_evalself.log 2>&1; echo "exit $?" >> $L/r2_evalself.log
