#!/bin/bash
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "start $(date)"
nice -n 5 bash tools/env/_py14.sh jends_eval.py dump --run e287_p2_s1 > temp/e287/dump_p2_s1_redo.log 2>&1; echo "dump exit $? $(date)"
nice -n 5 bash tools/env/_py14.sh jends_eval.py score --run e287_p2_s1 --seed 1 > temp/e287/score_p2_s1_redo.log 2>&1; echo "score exit $? $(date)"
echo REDO_DONE
