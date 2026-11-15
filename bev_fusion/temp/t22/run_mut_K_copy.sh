#!/bin/bash
# scratch: the planted-bug run for section K on a PRIVATE COPY of training/, tests/, trainkit and autogt in ~/mutk (a mutated file must never sit in the shared tree: a run that starts meanwhile snapshots it)
set -e
SRC=/mnt/c/Auto-research
du -sh $SRC/autogt $SRC/bev_fusion/training $SRC/bev_fusion/tests $SRC/autoresearch/trainkit 2>/dev/null
rm -rf ~/mutk
mkdir -p ~/mutk/bev_fusion ~/mutk/autoresearch
cp -r $SRC/bev_fusion/training ~/mutk/bev_fusion/
cp -r $SRC/bev_fusion/tests ~/mutk/bev_fusion/
cp -r $SRC/autoresearch/trainkit ~/mutk/autoresearch/
cp -r $SRC/autogt ~/mutk/autogt
cp -r $SRC/bev_fusion/evaluation ~/mutk/bev_fusion/
cp -r $SRC/bev_fusion/runner ~/mutk/bev_fusion/ 2>/dev/null || true
mkdir -p ~/mutk/bev_fusion/temp/t22
cp $SRC/bev_fusion/temp/t22/mut_K.py ~/mutk/bev_fusion/temp/t22/
cd ~/mutk/bev_fusion
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniconda3/envs/gcar/bin/python
echo "baseline in the copy:"
nice -n 10 $PY tests/test_student_chain.py > ~/mutk/baseline.out 2>&1 && echo "baseline PASSED" || { echo "baseline FAILED"; tail -5 ~/mutk/baseline.out; exit 1; }
setsid nohup nice -n 10 $PY temp/t22/mut_K.py > ~/mutk/mut_K.log 2>&1 < /dev/null &
sleep 2
echo "mutants started (log ~/mutk/mut_K.log)"
