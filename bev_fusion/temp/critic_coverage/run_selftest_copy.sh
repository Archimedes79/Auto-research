#!/bin/bash
cd /mnt/c/Auto-research || exit 1
S=$PWD/bev_fusion/temp/critic_coverage/newcopy
cd $S
PYTHONPATH=$S OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 15 ~/miniconda3/envs/gcar/bin/python -m autogt selftest > $S/../selftest_copy_out.txt 2>&1
echo "exit $?" >> $S/../selftest_copy_out.txt
