#!/bin/bash
# scratch: runs tests/test_student_chain.py (CPU, 2 threads, low priority) and keeps the output in temp/t22/test_student_chain.out
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
nice -n 10 $HOME/miniconda3/envs/gcar/bin/python tests/test_student_chain.py > temp/t22/test_student_chain.out 2>&1
echo "exit $?"
grep -c '  ok ' temp/t22/test_student_chain.out
grep 'FAIL' temp/t22/test_student_chain.out | head -20
tail -4 temp/t22/test_student_chain.out | cut -c1-300
