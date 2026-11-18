#!/usr/bin/env bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# Datenkette fuer CRN auf den Teilen 1-3 (Umgebung 'bev'): Infos (Szenen aus den bisherigen tokens.json), Radar, Soll-Daten im Ego-System.
set -e
export ENVNAME=bev
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate bev
cd $BFLC/training
python crn_make_infos.py --root ~/data/nuscenes_full --version v1.0-trainval --tokens ~/exp_b123/feats 2>&1 | grep -E "Szenen"
bash crn_gen_radar.sh ~/data/nuscenes_full v1.0-trainval 2>&1 | grep -vi -E "warn|it/s" | tail -8
bash crn_prep_gt.sh b123
echo "B123-DATEN FERTIG"
