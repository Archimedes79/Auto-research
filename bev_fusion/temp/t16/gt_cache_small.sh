#!/bin/bash
# scratch of the t16 bring-up: the trainer's GT cache (signature of the live generator = v8) for the ten small-set scenes
cd "$(dirname "$0")/../../.." || exit 1
R=$HOME/data/nuscenes_full
C=$PWD/bev_fusion/data/cache/gt
t0=$(date +%s)
bash autogt/run_nice.sh bev_fusion/training/gt_cache_build.py $R v1.0-trainval $C --split train --scenes scene-0061 scene-0553 scene-0655 scene-0757 scene-0796 scene-1077 scene-1094 scene-1100 --workers 4
bash autogt/run_nice.sh bev_fusion/training/gt_cache_build.py $R v1.0-trainval $C --split val --scenes scene-0103 scene-0916 --workers 4
echo "total wall $(( $(date +%s) - t0 )) s"
