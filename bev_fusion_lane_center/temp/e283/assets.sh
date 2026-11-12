#!/bin/bash
# scratch (system report 2026-11-08): the pictures and the two videos of the current best head, CPU, low priority
cd "$(dirname "$0")/../.." || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
(nice -n 12 bash tools/env/_py14.sh qgraph_results.py --run e279_w_s0 --id E279.w_s0 --scenes 11 39 --device cpu > temp/e283/videos.log 2>&1; touch temp/e283/VIDEOS_DONE) &
(nice -n 12 bash tools/env/_py14.sh compare_views.py "$PWD/results/E279/figures" 643,300,100,450 e273_p_s0 e273_p_s0+close e279_w_s0 > temp/e283/views.log 2>&1; touch temp/e283/VIEWS_DONE) &
wait
