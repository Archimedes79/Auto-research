#!/usr/bin/env bash
# run a python file of this folder with the training environment, from the bev_fusion folder: wsl bash temp/centreline_eval/py.sh <file> [args]
# CPU only (a training run holds the GPU)
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
exec "$HOME/miniconda3/envs/gcar/bin/python" "$@"
