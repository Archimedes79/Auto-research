#!/bin/bash
# scratch: the commands of the README examples, as written (read-only ones)
PY=~/miniconda3/envs/gcar/bin/python
cd /mnt/c/Auto-research/bev_fusion || exit 1
echo "=== runner: methods"; $PY -m runner methods | head -2 | cut -c1-150; echo "exit ${PIPESTATUS[0]}"
echo "=== runner: Runner.plan runs/t23_C_s1 small.val"; $PY -m runner Runner.plan runs/t23_C_s1 small.val 2>&1 | head -14 | cut -c1-160; echo "exit ${PIPESTATUS[0]}"
cd /mnt/c/Auto-research/autoresearch || exit 1
echo "=== evaluation: Evaluation.check runs/t23_C_s1 small.val"; AR_PROJECT=../bev_fusion $PY -m evalsuite Evaluation.check runs/t23_C_s1 small.val 2>&1 | head -14 | cut -c1-160; echo "exit ${PIPESTATUS[0]}"
echo "=== training: Training.check example job"; AR_PROJECT=../bev_fusion $PY -m trainsuite Training.check ../bev_fusion/records/jobs/training/example_t23_lane_bce.json 2>&1 | head -8 | cut -c1-160; echo "exit ${PIPESTATUS[0]}"
