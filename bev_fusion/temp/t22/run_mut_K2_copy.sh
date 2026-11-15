#!/bin/bash
# scratch: planted bugs of the critic-round fixes on a PRIVATE COPY in ~/mutk2 (never in the shared tree)
set -e
SRC=/mnt/c/Auto-research
rm -rf ~/mutk2
mkdir -p ~/mutk2/bev_fusion ~/mutk2/autoresearch ~/mutk2/bev_fusion/temp/t22
cp -r $SRC/bev_fusion/training ~/mutk2/bev_fusion/
cp -r $SRC/bev_fusion/tests ~/mutk2/bev_fusion/
cp -r $SRC/bev_fusion/evaluation ~/mutk2/bev_fusion/
cp -r $SRC/bev_fusion/runner ~/mutk2/bev_fusion/ 2>/dev/null || true
cp -r $SRC/autoresearch/trainkit ~/mutk2/autoresearch/
cp -r $SRC/autogt ~/mutk2/autogt
cp $SRC/bev_fusion/temp/t22/mut_K2.py ~/mutk2/bev_fusion/temp/t22/
cd ~/mutk2/bev_fusion
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniconda3/envs/gcar/bin/python
setsid nohup nice -n 10 $PY temp/t22/mut_K2.py > ~/mutk2/mut_K2.log 2>&1 < /dev/null &
sleep 2
echo "started (log ~/mutk2/mut_K2.log)"
