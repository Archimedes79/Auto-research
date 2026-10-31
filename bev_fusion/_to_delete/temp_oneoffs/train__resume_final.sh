#!/usr/bin/env bash
# One-off: stop the combined training, keep its log, and resume it from the latest checkpoint with the config as it is now
# (steps changed 8000 -> 6000 at the step-1000 checkpoint; the cosine schedule follows cfg.steps, a negligible jump at 1000).
cd "$(dirname "$0")/../.." || exit 1
pkill -f "train.py.*t1_r3_combined.json"
sleep 8
pgrep -af "train.py.*t1_r3_combined" | head -2
cp runs/t1_r3_combined/train.log runs/t1_r3_combined/train_part1.log
ls runs/t1_r3_combined | grep ckpt
bash temp/train/run_cfg.sh training/configs/t1_r3_combined.json 04a7de9
