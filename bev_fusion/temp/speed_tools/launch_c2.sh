#!/bin/bash
# speed.c2 (bf16 + cudnn.benchmark + 8-step profile window) and speed.a3 (baseline repeat) behind it; the ids speed.b / speed.c / speed.a2 are stale records of launchers I killed
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
for p in $(pgrep -f "tools/run.py --id speed.a2 "); do kill -9 $p && echo "old a2 launcher $p killed"; done
setsid nohup python3 ../autoresearch/tools/run.py --id speed.c2 --no-card --no-git --gpu --ram-mb 20000 --split train -- bash temp/speed_run.sh speed temp/compose/speed_c2.json > runs/speed_c2.launcher.log 2>&1 < /dev/null &
sleep 3
setsid nohup python3 ../autoresearch/tools/run.py --id speed.a3 --no-card --no-git --gpu --ram-mb 20000 --split train --after speed.c2 -- bash temp/speed_run.sh speed_base temp/compose/speed_a3.json > runs/speed_a3.launcher.log 2>&1 < /dev/null &
sleep 6
tail -n 3 runs/speed_c2.launcher.log runs/speed_a3.launcher.log | cut -c1-200
pgrep -af "tools/run.py" | cut -c1-120
