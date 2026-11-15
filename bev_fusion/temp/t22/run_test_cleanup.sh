#!/bin/bash
# scratch: runs tests/test_student_chain.py and counts the student_chain_* folders in temp/ before and after (the test must leave none of its own)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
ls -d temp/student_chain_* 2>/dev/null | sort > temp/t22/sc_before.txt
echo "folders before: $(wc -l < temp/t22/sc_before.txt)"
nice -n 10 $HOME/miniconda3/envs/gcar/bin/python tests/test_student_chain.py > temp/t22/test_student_chain.out 2>&1
echo "exit $?"
tail -1 temp/t22/test_student_chain.out | cut -c1-200
ls -d temp/student_chain_* 2>/dev/null | sort > temp/t22/sc_after.txt
echo "folders after: $(wc -l < temp/t22/sc_after.txt)"
echo "new ones:"; comm -13 temp/t22/sc_before.txt temp/t22/sc_after.txt
