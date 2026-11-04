#!/bin/bash
# all GPU runs of this session, one at a time, only while no other session trains (train.py / run_cfg.sh) and the GPU
# has room (total < 11 GB, CLAUDE.md) - on 2026-10-30 a peer's training grew next to these runs and passed 11 GB.
# Order: the answer to "can GT at +-dt generate v" first (full label / boxes, seed 0), then the correlation arm.
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/miniconda3/envs/gcar/bin/python
free() { ! pgrep -f "[r]un_cfg\.sh|[t]rain\.py" > /dev/null \
         && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt $((11000 - $1)) ]; }
wait_free() { until free $1 && sleep 60 && free $1; do sleep 60; done; }
timed() {   # timed <log> <need MB> <args...>
  local log=$1 need=$2; shift 2
  wait_free $need
  echo "$log start $(date +%T)"
  $PY -u tests/test_flow_timed.py "$@" > temp/checks/sup/$log.log 2>&1
  echo "$log exit $? $(date +%T)"
}
timed full_label_s0 1500 --label
timed full_boxes_s0 1500 --boxes
timed quick_label_corr 1500 --label --corr --quick
timed quick_boxes_corr 1500 --boxes --corr --quick
timed full_boxes_corr_s0 1500 --boxes --corr
timed full_label_corr_s0 1500 --label --corr
BEST="--sharp --pyr --doppler --rawwarp 3.0 --prior 0.03 --smooth 0.05 --gate-doppler"
wait_free 1500
echo "gatedop s1 start $(date +%T)"
$PY -u tests/test_flow_explains.py --quick --data nuscenes $BEST --seed 1 > temp/checks/flow_explains_nusc_gatedop_s1.log 2>&1
$PY -u tests/test_flow_explains.py --data nuscenes --sequences --n-seq 24 --length 5 $BEST --seed 1 > temp/checks/flow_sequences_nusc_gatedop_s1.log 2>&1
timed full_boxes_maps_s0 3000 --boxes --maps
timed full_label_s1 1500 --label --seed 1
timed full_boxes_s1 1500 --boxes --seed 1
timed full_boxes_corr_s1 1500 --boxes --corr --seed 1
timed full_label_corr_s1 1500 --label --corr --seed 1
echo "queue finished $(date +%T)"
