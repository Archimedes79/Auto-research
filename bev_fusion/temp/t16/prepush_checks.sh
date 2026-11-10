#!/bin/bash
# scratch: what runs, git hygiene, and the fast gates of both projects before the push Mirko asked for (2026-11-03)
cd /mnt/c/Auto-research || exit 1
echo "== running jobs"
ps -eo pid,etime,args | grep -E "run.py|train.py|grid_train|bev_suite|evaluate.py|trainsuite|run_ar" | grep -v grep | cut -c1-160
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "== check_git (known foreign tag lines hidden)"
python3 autoresearch/tools/check_git.py 2>&1 | grep -v "15f1f8da" | grep -v "^  pushing the tags" | head -n 14
echo "== bev_fusion fast gates"
(cd bev_fusion && nice -n 10 bash tests/run_gates.sh > temp/t16/gates5.log 2>&1; echo "rc=$?"; grep -c "^PASS" temp/t16/gates5.log; grep -E "^FAIL|Traceback" temp/t16/gates5.log | head -5)
echo "== lane fast gates"
(cd bev_fusion_lane_center && nice -n 10 bash tools/maintenance/run_gates.sh > /tmp/lane_gates.log 2>&1; echo "rc=$?"; tail -n 6 /tmp/lane_gates.log | cut -c1-160)
