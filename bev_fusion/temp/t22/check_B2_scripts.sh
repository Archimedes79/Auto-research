#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion/temp/t22 || exit 1
for f in night_B2.sh start_B2.sh start_evalB2.sh start_smoke_B2full.sh dry_B2.sh; do bash -n $f && echo "syntax ok $f"; done
python3 window_gate.py --selftest | tail -2
cd /mnt/c/Auto-research/bev_fusion
echo "--- t23_released logic now (expect not released at $(date +%H:%M))"
python3 ../autoresearch/tools/schedule.py show 2>/dev/null | grep -E '^(done|cancelled)'
