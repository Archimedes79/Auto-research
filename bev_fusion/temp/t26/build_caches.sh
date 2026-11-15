#!/bin/bash
# scratch: the t26 image caches (CPU, one worker, nice 19): key frames of xl first, then the older frames of the window; resumable, markers written by the builders at the end
cd /mnt/c/Auto-research/bev_fusion
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
PY=~/miniconda3/envs/gcar/bin/python
CFG=training/configs/t26_one.json
echo "== $(date '+%F %T') build-cache"
nice -n 19 $PY training/lane_branch.py build-cache $CFG --workers 1 ; echo "== $(date '+%F %T') build-cache rc=$?"
echo "== $(date '+%F %T') build-cache-older"
nice -n 19 $PY training/lane_branch.py build-cache-older $CFG --workers 1 ; echo "== $(date '+%F %T') build-cache-older rc=$?"
echo "== $(date '+%F %T') check-cache"; nice -n 19 $PY training/lane_branch.py check-cache $CFG 2>&1 | tail -3
echo "== $(date '+%F %T') check-cache-older"; nice -n 19 $PY training/lane_branch.py check-cache-older $CFG 2>&1 | tail -3
echo "== $(date '+%F %T') all done"
