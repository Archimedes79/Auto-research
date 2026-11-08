#!/bin/bash
# GaussianCaR vs our maps on the SAME 400 frames (CPU, 2 threads): the t6 procedure (temp/soa_gap/map_ref.py) for the dual-feature chain,
# its same-code control and the BEVFusion-feature chain. One run after the other; a finished json is not redone.
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
for r in t10_dual_s0 t8_ctl_s0 t8_bevfeat_s0; do
  [ -f temp/soa_gap/map_ref_${r}_400.json ] && continue
  OMP_NUM_THREADS=2 $PY -u temp/soa_gap/map_ref.py runs/$r temp/soa_gap/map_ref_${r}_400.json 400 > temp/soa_gap/map_ref_${r}_400.log 2>&1
  echo "$r exit $?"
done
echo "all done"
