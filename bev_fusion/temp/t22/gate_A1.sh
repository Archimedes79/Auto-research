#!/bin/bash
# scratch: wait for an away window with a free GPU, then run start_A1.sh (expected 0.5 h at most)
cd /mnt/c/Auto-research/bev_fusion || exit 1
exec python3 temp/t22/window_gate.py --hours 0.5 --stop-file runs/t22_A1_s0/CONTROL -- bash temp/t22/start_A1.sh
