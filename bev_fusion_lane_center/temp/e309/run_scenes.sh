#!/bin/bash
cd "$(dirname "$0")"
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" bash ../../tools/env/_py14.sh e309/render_scene.py 0 ../../results/l6_view_E312/videos 41 33 15 > scenes_run.log 2>&1
echo SCENES DONE >> scenes_run.log
