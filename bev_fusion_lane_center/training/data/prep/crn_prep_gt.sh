#!/usr/bin/env bash
# Soll-Daten fuer die Kamera-Radar-Experimente im Fahrzeug-System (GT_FRAME=ego, kein Lidar):
#   Karten-Knoten, Soll-Raster, exakte Abstandskarte.
# Aufruf: bash crn_prep_gt.sh <mini|b123>      Ausgabe: ~/exp_crn_<name>/{graph,gt,feats}
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate "${ENVNAME:-crn}"
export GT_FRAME=ego PYTHONWARNINGS=ignore
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
case "$1" in
  mini) ROOT=$HOME/data/nuscenes;      VER=v1.0-mini;     EXP=$HOME/exp_crn_mini ;;
  b123) ROOT=$HOME/data/nuscenes_full; VER=v1.0-trainval; EXP=$HOME/exp_crn_b123 ;;
  *) echo "mini oder b123"; exit 1 ;;
esac
mkdir -p "$EXP/graph" "$EXP/gt" "$EXP/feats"
export NUSC_ROOT=$ROOT NUSC_VERSION=$VER
echo "== Karten-Knoten"
python "/mnt/c/ML-Experiment/bev_fusion_lane_center/training/data/prep/prep_graph.py" "$EXP/graph" 2>&1 | grep -E "Samples"
echo "== Soll-Raster"
python "/mnt/c/ML-Experiment/bev_fusion_lane_center/training/data/prep/centerline_gt.py" --dataroot "$ROOT" --version "$VER" --out "$EXP/gt" \
  --infos "$ROOT/nuscenes_infos_train.pkl,$ROOT/nuscenes_infos_val.pkl" 2>&1 | grep -E "Samples"
echo "== exakte Abstandskarte"
python "/mnt/c/ML-Experiment/bev_fusion_lane_center/training/data/prep/make_dist_gt.py" "$EXP/feats" 2>&1 | grep -E "Samples"
echo "GT FERTIG ($1)"
