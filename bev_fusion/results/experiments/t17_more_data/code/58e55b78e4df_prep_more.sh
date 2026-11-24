#!/bin/bash
# scratch of the t16 follow-up (D46): the feature folders of the "more data" arm: all 850 scenes' DualBEV shards and the BEVFusion shards (510 exist, the 272 train
# scenes of nuScenes parts 7 - 10 are built on the GPU). Run from the WSL: bash bev_fusion/temp/t16/prep_more.sh links | extract
set -e
cd "$(dirname "$0")/../.."
C=data/cache
A=$C/all
mode=$1
if [ "$mode" = links ]; then
  mkdir -p $A/cam_feats $A/cam_feats_bevfusion $A/src_new
  for src in cam_feats cam_feats_parts7_10; do
    base=$(readlink -f $C/$src)
    for f in "$base"/*.npy "$base"/*.index.npz; do ln -sfn "$f" $A/cam_feats/$(basename "$f"); done
  done
  base=$(readlink -f $C/cam_feats_bevfusion)
  for f in "$base"/*.npy "$base"/*.index.npz; do ln -sfn "$f" $A/cam_feats_bevfusion/$(basename "$f"); done
  # the new train scenes (scene list of the config temp/t16/more_scenes.json: written by make_more.py): their DualBEV shards are the source of the BEVFusion build
  /home/ml/miniconda3/envs/gcar/bin/python - <<'PY'
import json, os
new = json.load(open("temp/t16/more_scenes.json"))["new_train"]
for s in new:
    for ext in ("npy", "index.npz"):
        dst = f"data/cache/all/src_new/{s}.{ext}"
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(f"data/cache/all/cam_feats/{s}.{ext}"), dst)
print(len(new), "scenes in src_new")
PY
  echo "all/cam_feats: $(ls $A/cam_feats | wc -l) files, all/cam_feats_bevfusion: $(ls $A/cam_feats_bevfusion | wc -l) files (1020 expected until the extraction ran)"
elif [ "$mode" = extract ]; then
  # keep only the scenes that have no BEVFusion shard yet (resumable per scene by the builder)
  /home/ml/miniconda3/envs/gcar/bin/python training/cache_bevfusion_features.py build --src $A/src_new --out $A/cam_feats_bevfusion --device cuda
  echo "BEVFusion files now: $(ls $A/cam_feats_bevfusion | wc -l) (expect 1020 + 544 = 1564)"
else
  echo "usage: prep_more.sh links | extract"; exit 2
fi
