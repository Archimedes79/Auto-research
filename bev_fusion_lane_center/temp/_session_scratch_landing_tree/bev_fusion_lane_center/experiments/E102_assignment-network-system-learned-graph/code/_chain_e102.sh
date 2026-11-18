#!/bin/bash
# E102 zero-shot chain: learned graphs of a trace network on all VAL / TEST rows -> src caches (prefix l) -> assignment samples -> the loop-9 assignment network (E87 recipe, trained on rule graphs) applied unchanged
# usage: _chain_e102.sh <trace model dir> <assignment model dir> <out name> [extra args of p10_infer.py]
M="$1"; A="$2"; O="$3"; shift 3
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
LOG=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/$O.log
echo "start $(date)" > $LOG
for sp in val_clean test_clean; do
  bash _pyr.sh p10_infer.py --name $M --split $sp ${TAG:+--tag ${sp}_$TAG} "$@" >> $LOG 2>&1
  bash _py.sh p10_tosrc.py --name $M --split $sp ${TAG:+--tag ${sp}_$TAG} >> $LOG 2>&1
done
bash _py.sh p8_dataset.py --plan dev --loc tmpl --srcprefix l --tag _segL --workers 6 >> $LOG 2>&1
bash _py.sh p8_dataset.py --plan test --loc tmpl --srcprefix l --tag _segL --workers 6 >> $LOG 2>&1
bash run_gcar.sh p8_assign.py --name $O --load $A --tag _segL --dev_tag _segL --test 1 >> $LOG 2>&1
echo "done $(date)" >> $LOG
touch /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/$O.done
