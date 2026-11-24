#!/bin/bash
# usage: run.sh <script path abs or name> args
cd /mnt/c/Auto-research/bev_fusion_lane_center
export CUDA_VISIBLE_DEVICES=
bash tools/env/_py14.sh "$@"
