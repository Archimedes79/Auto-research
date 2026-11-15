#!/bin/bash
# scratch: stage-0 timing of the image side on the idle GPU (night window, nothing else running), log in temp/t22/time_student.log
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
$PY temp/t22/time_student.py > temp/t22/time_student.log 2>&1
echo "exit $?" >> temp/t22/time_student.log
