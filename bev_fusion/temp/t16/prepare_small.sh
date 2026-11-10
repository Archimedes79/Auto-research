#!/bin/bash
# scratch of the t16 bring-up: the feature folders of autogt's small set (ten scenes) under data/cache/small/ and the BEVFusion shards
# that the six scenes of nuScenes parts 7 - 10 still lack. Run from bev_fusion/ inside the WSL: bash temp/t16/prepare_small.sh
set -e
cd "$(dirname "$0")/../.."
C=data/cache
S=$C/small
mkdir -p $S/cam_feats $S/cam_feats_bevfusion $S/src_missing
SCENES="scene-0061 scene-0103 scene-0553 scene-0655 scene-0757 scene-0796 scene-0916 scene-1077 scene-1094 scene-1100"
for s in $SCENES; do
  if [ -e $C/cam_feats/$s.npy ]; then src=cam_feats; else src=cam_feats_parts7_10; fi
  for ext in npy index.npz; do
    ln -sfn "$(readlink -f $C/$src)/$s.$ext" $S/cam_feats/$s.$ext
  done
  if [ -e $C/cam_feats_bevfusion/$s.npy ]; then
    for ext in npy index.npz; do ln -sfn "$(readlink -f $C/cam_feats_bevfusion)/$s.$ext" $S/cam_feats_bevfusion/$s.$ext; done
  else                                                        # BEVFusion features of this scene are missing: build from its DualBEV shard
    for ext in npy index.npz; do ln -sfn "$(readlink -f $S/cam_feats/$s.$ext)" $S/src_missing/$s.$ext; done
  fi
done
echo "linked: $(ls $S/cam_feats | wc -l) DualBEV files, $(ls $S/cam_feats_bevfusion | wc -l) BEVFusion files, $(ls $S/src_missing | wc -l) to build"
if [ "$(ls $S/src_missing | wc -l)" -gt 0 ]; then
  /home/ml/miniconda3/envs/gcar/bin/python training/cache_bevfusion_features.py build --src $S/src_missing --out $S/cam_feats_bevfusion --device cuda
fi
echo "BEVFusion files now: $(ls $S/cam_feats_bevfusion | wc -l) (expect 20)"
