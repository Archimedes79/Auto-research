#!/bin/bash
# scratch: refresh the private copy ~/mutk2 (code, tests, evalsuite), run the BASELINE of the full test there (it must pass), then the five mutants whose first run was invalid; detached
SRC=/mnt/c/Auto-research
cd ~/mutk2/bev_fusion || exit 1
cp -r $SRC/bev_fusion/training/*.py training/
cp -r $SRC/bev_fusion/tests/*.py tests/
cp -r $SRC/bev_fusion/evaluation/*.py evaluation/
rm -rf ~/mutk2/autoresearch/evalsuite
cp -r $SRC/autoresearch/evalsuite ~/mutk2/autoresearch/evalsuite
cp $SRC/bev_fusion/temp/t22/mut_K5.py temp/t22/
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniconda3/envs/gcar/bin/python
nice -n 10 $PY tests/test_student_chain.py > ~/mutk2/baseline_full.out 2>&1 && echo "baseline of the full test in the copy: PASSED ($(grep -c '^  ok ' ~/mutk2/baseline_full.out) checks)" || { echo "baseline FAILED"; grep -E 'FAIL|Error' ~/mutk2/baseline_full.out | head -5 | cut -c1-240; exit 1; }
setsid nohup nice -n 10 $PY temp/t22/mut_K5.py > ~/mutk2/mut_K5.log 2>&1 < /dev/null &
sleep 1
echo "mutants started (log ~/mutk2/mut_K5.log)"
