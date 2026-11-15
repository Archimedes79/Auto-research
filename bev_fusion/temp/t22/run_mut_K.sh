#!/bin/bash
# scratch: the mutation run for section K, detached; the files are restored by mut_K.py itself (backups in temp/t22/mut_backup_k/)
cd /mnt/c/Auto-research/bev_fusion || exit 1
setsid nohup nice -n 10 $HOME/miniconda3/envs/gcar/bin/python temp/t22/mut_K.py > temp/t22/mut_K.log 2>&1 < /dev/null &
sleep 2
ps -eo pid,etime,args | grep mut_K | grep -v grep | cut -c1-100
