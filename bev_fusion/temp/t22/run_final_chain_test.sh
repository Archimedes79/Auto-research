#!/bin/bash
# scratch: the final run of tests/test_student_chain.py (output kept, checks counted)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
nice -n 10 $HOME/miniconda3/envs/gcar/bin/python tests/test_student_chain.py > temp/t22/test_student_chain_final.out 2>&1
echo "exit $?"
echo "ok checks: $(grep -c '^  ok ' temp/t22/test_student_chain_final.out)"
echo "FAIL lines: $(grep -c 'FAIL' temp/t22/test_student_chain_final.out)"
tail -1 temp/t22/test_student_chain_final.out
