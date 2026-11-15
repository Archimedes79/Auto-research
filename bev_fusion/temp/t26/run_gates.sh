#!/bin/bash
# throw-away (t26 track A): the existing gates that touch the changed files, one after the other, 2 threads, nice 15; one summary line per gate in /tmp/t26_gates.log
cd /mnt/c/Auto-research/bev_fusion
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=~/miniconda3/envs/gcar/bin/python
: > /tmp/t26_gates.log
for t in "tests/test_student_chain.py" "tests/test_student_encoder.py" "tests/test_stream_tools.py" "tests/test_trainkit_equivalence.py" "tests/test_train_guards.py" "tests/test_grad_scaler.py" \
         "tests/test_host_batch.py" "tests/test_slots.py" "tests/test_stream_temporal.py" "tests/test_radar_recursive.py" "tests/test_lane_branch_chain.py" "tests/test_lane_branch_dual.py" \
         "training/interface.py" "training/lane_branch.py" "training/augment.py" "training/startable.py"; do
  s=$(date +%s)
  nice -n 15 timeout 2400 $PY $t > /tmp/t26_gate_$(basename $t .py).log 2>&1
  rc=$?
  echo "$t rc=$rc $(( $(date +%s) - s )) s: $(tail -1 /tmp/t26_gate_$(basename $t .py).log | cut -c1-200)" >> /tmp/t26_gates.log
done
echo ALLDONE >> /tmp/t26_gates.log
