#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
setsid nohup python3 ../autoresearch/tools/run.py --seed 0 --retries 0 --require-card --split val --id t26.base_eval_vel --hyp "The base's velocity rows on GT v13 (stationary flag, flow) differ from the stored card on GT v9; the control of the velocity clause is measured, not copied." --pred "centre-field EPE of moving objects within 0.05 of the stored 1.1037; every other row as in t26.base_eval" --rule "no verdict of its own: the control values of the JSON clauses of t26" -- bash temp/t26/job_base_eval_vel.sh > temp/t26/q4.log 2>&1 < /dev/null &
sleep 3; echo queued
