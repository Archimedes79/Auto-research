#!/usr/bin/env bash
# Radar-Punktwolken fuer CRN erzeugen (gen_radar_bev.py + gen_radar_pv.py aus dem CRN-Repo, mit angepassten Pfaden).
# Liest nur Radar-Dateien und Metadaten (Posen, Kalibrierung), keine Lidar-Punktwolken.
# Aufruf (Umgebung crn):  bash crn_gen_radar.sh <ROOT> <VERSION>
# Voraussetzung: <ROOT>/nuscenes_infos_{train,val}_crn.pkl (crn_make_infos.py), samples/RADAR_*, sweeps/RADAR_*
set -e
ROOT=$(readlink -f "$1"); VERSION=$2
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate "${ENVNAME:-crn}"
cd ~/work/CRN
for s in bev pv; do
  sed -e "s#^SPLIT = 'v1.0-trainval'#SPLIT = '$VERSION'#" \
      -e "s#^DATA_PATH = 'data/nuScenes'#DATA_PATH = '$ROOT'#" \
      -e "s#^info_paths = .*#info_paths = ['$ROOT/nuscenes_infos_train_crn.pkl', '$ROOT/nuscenes_infos_val_crn.pkl']#" \
      scripts/gen_radar_$s.py > /tmp/crn_gen_radar_$s.py
  grep -nE '^(SPLIT|DATA_PATH|info_paths)' /tmp/crn_gen_radar_$s.py
done
python /tmp/crn_gen_radar_bev.py
python /tmp/crn_gen_radar_pv.py
ls $ROOT/radar_pv_filter | wc -l
echo "RADAR FERTIG"
