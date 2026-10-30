#!/bin/bash
# chain G: camera-only quick rerun with the per-class thresholds, after chain B (the flow-explains runs)
cd "$(dirname "$0")/../.."
sleep 60
while pgrep -f "[c]hain_b.sh" > /dev/null; do sleep 30; done
~/miniconda3/envs/gcar/bin/python -u tests/test_camera_only.py --quick > temp/checks/camera_only_quick_thr.log 2>&1
