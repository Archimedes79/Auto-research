#!/bin/bash
# run a scratch script of this folder with the gcar python from the bev_fusion folder
cd "$(dirname "$0")/../.." || exit 1
exec ~/miniconda3/envs/gcar/bin/python "$@"
