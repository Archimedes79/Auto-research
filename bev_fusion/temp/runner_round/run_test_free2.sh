#!/bin/bash
# scratch: the equivalence test on a run with a newer code snapshot (host_side, pin_memory)
cd /mnt/c/Auto-research/bev_fusion || exit 1
P=~/miniconda3/envs/gcar/bin/python
RUN=${1:-runs/t23_C_s1}
ls $RUN | head -12
grep -c host_side $RUN/code/train.py
uptime
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
$P tests/test_bev_free.py $RUN > temp/cleanup/test_free2.log 2>&1
echo "exit $?" >> temp/cleanup/test_free2.log
