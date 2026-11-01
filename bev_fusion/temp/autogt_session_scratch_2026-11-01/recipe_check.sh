#!/bin/bash
# scratch: try the rebuild recipe of an old GT version (README, section Versions) on nuScenes mini from the worktree of the tag autogt/v5
cd "$(dirname "$0")/../../.." || exit 1               # repository root
cd temp/gt_v5 || exit 1
export PYTHONPATH=bev_fusion OMP_NUM_THREADS=1
PY=~/miniconda3/envs/gcar/bin/python
$PY -c "import autogt; print(autogt.__file__)"
nice -n 15 $PY -m autogt build --dataroot ~/data/nuscenes --version v1.0-mini --scenes scene-0061 --stride 20 --out ../gt_v5_out 2>&1 | tail -4
ls ../gt_v5_out
