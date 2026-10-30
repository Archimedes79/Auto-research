#!/bin/bash
# chain X: the multi-keyframe recursive evaluation of the best nuScenes model (seed 0), CPU, after its training saved it
cd "$(dirname "$0")/../.."
until grep -qE "claims hold|Traceback" temp/checks/flow_explains_nusc_best_s0.log 2>/dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES= nice -n 15 ~/miniconda3/envs/gcar/bin/python -u tests/test_flow_explains.py --data nuscenes --sequences --n-seq 24 --length 5 --sharp --pyr --doppler --rawwarp 3.0 --prior 0.03 --smooth 0.05 > temp/checks/flow_sequences_nusc_best_s0.log 2>&1
