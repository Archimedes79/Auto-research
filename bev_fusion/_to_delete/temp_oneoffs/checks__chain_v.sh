#!/bin/bash
# chain V (2nd version): the best setup on nuScenes with the recursive evaluation, seed 1 (after chain U wrote the data cache)
cd "$(dirname "$0")/../.."
while [ ! -f temp/checks/flow_explains_nusc_test_16_v1.pt ]; do sleep 15; done
sleep 30
~/miniconda3/envs/gcar/bin/python -u tests/test_flow_explains.py --quick --data nuscenes --recursive --sharp --pyr --doppler --rawwarp 3.0 --prior 0.03 --smooth 0.05 --seed 1 > temp/checks/flow_explains_nusc_best_s1.log 2>&1
