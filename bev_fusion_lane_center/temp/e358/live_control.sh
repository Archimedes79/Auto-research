#!/bin/bash
# E358 control of the live check: the SAME check on CPU for E312's keypoint records (E325's accepted path, which was run on the GPU): if it fails the same way, the device (CPU vs the GPU of the stored dumps) is the confound, not the attached ends
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
nice -n 10 bash tools/env/_py14.sh jwire_check.py live --seed 0 --pair-run e312_p_s0 --max 24 --workers 2 --device cpu > temp/e358/live_control_kp_s0.log 2>&1
echo "control kp seed 0 exit $? $(date +%H:%M:%S)" >> temp/e358/live_done.txt
