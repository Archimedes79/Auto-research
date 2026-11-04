#!/bin/bash
# Camera-slot gates (CAMERAS table): waits for free RAM, then the light tests, then tests/equivalence.py (old code 94464cd vs the live tree)
# for the two models the Architecture chat named (t10_dual_s0 step 3000; t10_dual_long_s0 step 5030, hurdle). Log: temp/camera_slot/gates.log
cd "$(dirname "$0")/../.." || exit 2                     # bev_fusion/
echo 1000 > /proc/self/oom_score_adj                      # if RAM runs out, this job dies first, not a training
WRAP=../bev_fusion_lane_center/tools/env/_py14.sh
OLD=$(cd ../temp/equiv_old/training && pwd)
LOG=temp/camera_slot/gates.log
avail_gb() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
wait_ram() { local need=$1 ok=0; while [ $ok -lt 3 ]; do if [ "$(avail_gb)" -ge "$need" ]; then ok=$((ok+1)); else ok=0; fi; sleep 20; done; }
step() { local name=$1; shift; echo "== $name $(date +%H:%M:%S) (RAM available $(avail_gb) GB)" | tee -a $LOG; "$@" >> $LOG 2>&1; local rc=$?; echo "-- $name exit $rc" | tee -a $LOG; return $rc; }
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 EQUIV_DEVICE=cpu
: > $LOG
wait_ram 4
step camera_features bash $WRAP tests/test_camera_features.py
step geometry bash $WRAP tests/test_dual_camera_geometry.py
step slots bash $WRAP tests/test_slots.py
wait_ram 12
for spec in "A runs/t10_dual_s0 ckpt_00003000.pt 4" "B runs/t10_dual_long_s0 ckpt_00005030.pt 3"; do
  set -- $spec
  export EQUIV_RUN=$2 EQUIV_CKPT=$3 EQUIV_REF=temp/camera_slot/ref_$1.pt
  step save_$1 env EQUIV_CODE=$OLD bash $WRAP tests/equivalence.py save --n $4
  step compare_$1 bash $WRAP tests/equivalence.py compare
done
echo "== done $(date +%H:%M:%S)" | tee -a $LOG
