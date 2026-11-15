#!/usr/bin/env bash
# scratch: run a python script of bev_fusion under nice with 2 threads (the box is shared with lane trainings)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2
export PYTHONPATH="/mnt/c/Auto-research${PYTHONPATH:+:$PYTHONPATH}"
exec nice -n 15 "$HOME/miniconda3/envs/gcar/bin/python" "$@"
