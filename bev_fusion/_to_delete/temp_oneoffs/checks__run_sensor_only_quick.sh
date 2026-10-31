#!/usr/bin/env bash
# One-off: wait for the running object-head test, then run the radar-only and camera-only quick gates one after the
# other (one GPU job at a time). Logs next to this script. Run from the bev_fusion folder.
cd "$(dirname "$0")/../.."
while pgrep -f "[t]ests/test_object_head.py" >/dev/null; do sleep 10; done
PY=~/miniconda3/envs/gcar/bin/python
$PY -u tests/test_radar_only.py --quick > temp/checks/radar_only_quick.log 2>&1
$PY -u tests/test_camera_only.py --quick > temp/checks/camera_only_quick.log 2>&1
echo done
