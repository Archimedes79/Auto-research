#!/usr/bin/env bash
# Mini-Test der Datenkette fuer CRN (Infos, Radar, Soll-Daten) in der Umgebung 'bev' (Devkit + mmcv genuegen).
set -e
export ENVNAME=bev
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate bev
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
python crn_make_infos.py --root ~/data/nuscenes --version v1.0-mini --mini 2>&1 | grep -vi warn
bash crn_gen_radar.sh ~/data/nuscenes v1.0-mini 2>&1 | grep -vi warn | tail -8
bash crn_prep_gt.sh mini
echo "MINI-DATEN FERTIG"
