#!/usr/bin/env bash
# one-off: the evaluation path on the rebuilt framework. A throw-away run folder in scratch whose code snapshot is today's training/
# (the hurdle chain, ckpt 5030 linked, not copied); bev_suite eval on 8 val frames writes only into that folder.
cd "$(dirname "$0")/../.." || exit 1
R=temp/arch_build/evalrun
rm -rf $R && mkdir -p $R
cp runs/t10_dual_long_s0/config.json runs/t10_dual_long_s0/gt.json $R/
ln -s "$(pwd)/runs/t10_dual_long_s0/ckpt_00005030.pt" $R/ckpt_00005030.pt
cp -r training $R/code && rm -rf $R/code/__pycache__
echo "snapshot: today's training/ ($(git -C .. rev-parse --short HEAD) + working tree)" > $R/code/SNAPSHOT.txt
export OMP_NUM_THREADS=4
timeout 2400 ~/miniconda3/envs/gcar/bin/python -u evaluation/bev_suite.py $R --workers 2 eval --n 8 2>&1 | grep -v Warning | tail -12
ls $R/eval 2>/dev/null
