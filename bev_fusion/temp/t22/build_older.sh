#!/bin/bash
# scratch: the image cache of the OLDER frames of the t21 P4 pair (n_cam 2, cam_stride 4) for the large split, into data/cache/img_front/sweeps (resumable, one process, low priority);
# a second run only checks that every file is there and writes sweeps/BUILT.json (the finished-build marker that startable.py asks for)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniconda3/envs/gcar/bin/python
setsid nohup nice -n 15 $PY training/lane_branch.py build-cache-older runs/t22_B_s0/config.json --workers 1 > temp/t22/build_older2.log 2>&1 < /dev/null &
sleep 3
ps -eo pid,etime,pcpu,rss,args | grep -E 'lane_branch' | grep -v grep | cut -c1-140
