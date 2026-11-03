#!/bin/bash
# scratch of the t16 bring-up: the trainer's GT cache (signature of the live generator = v8) for the 428 / 82 scenes of t16_full.json
cd "$(dirname "$0")/../../.." || exit 1
R=$HOME/data/nuscenes_full
C=$PWD/bev_fusion/data/cache/gt
CFG=bev_fusion/training/configs/t16_full.json
TR=$(python3 -c "import json,sys; print(' '.join(json.load(open('$CFG'))['train_scenes']))")
VA=$(python3 -c "import json,sys; print(' '.join(json.load(open('$CFG'))['val_scenes']))")
t0=$(date +%s)
bash autogt/run_nice.sh bev_fusion/training/gt_cache_build.py $R v1.0-trainval $C --split val --scenes $VA --workers 4
bash autogt/run_nice.sh bev_fusion/training/gt_cache_build.py $R v1.0-trainval $C --split train --scenes $TR --workers 4
echo "total wall $(( $(date +%s) - t0 )) s"
