#!/bin/bash
# scratch: wait for an away window with a free GPU, then run start_A2.sh (A1 ran 180 samples/s with the backbone frozen; A2 is 98k samples, so 0.5 h is the expectation, the stop file ends it cleanly if it runs over)
cd /mnt/c/Auto-research/bev_fusion || exit 1
exec python3 temp/t22/window_gate.py --night-only --hours 1.0 --stop-file runs/t22_A2_s0/CONTROL -- bash temp/t22/start_A2.sh
