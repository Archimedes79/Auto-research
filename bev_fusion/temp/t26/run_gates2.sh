#!/bin/bash
# throw-away (t26 track A, revision): the gates after the critic's fixes, one after the other, 2 threads, nice 15; one summary line per gate in /tmp/t26_gates2.log
cd /mnt/c/Auto-research/bev_fusion
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=~/miniconda3/envs/gcar/bin/python
: > /tmp/t26_gates2.log
for t in "tests/test_student_window.py" "tests/test_student_chain.py" "tests/test_stream_tools.py" "tests/test_train_guards.py" "training/startable.py" "training/lane_branch.py" "tests/test_trainkit_equivalence.py" \
         "tests/test_ab_timing.py" "tests/test_speed_flags.py" "tests/test_clip_split.py" "tests/test_lr_groups.py" "tests/test_lane_branch_chain.py" "training/interface.py"; do
  s=$(date +%s)
  nice -n 15 timeout 2400 $PY -u $t > /tmp/t26_gate2_$(basename $t .py).log 2>&1
  rc=$?
  echo "$t rc=$rc $(( $(date +%s) - s )) s: $(tail -1 /tmp/t26_gate2_$(basename $t .py).log | cut -c1-200)" >> /tmp/t26_gates2.log
done
echo ALLDONE >> /tmp/t26_gates2.log
