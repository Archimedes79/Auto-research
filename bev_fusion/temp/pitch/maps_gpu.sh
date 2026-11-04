#!/bin/bash
# The like-for-like map scores of the case on the GPU (a few minutes per model instead of 45 minutes on the CPU): GaussianCaR vs our chains on the same 400 frames.
# Run it ONLY with the GPU free of trainings (it needs about 3 GB of GPU memory and 6 - 8 GB of host memory), through run.py with --gpu --gpu-max-mb 3000 (start_maps.sh).
# First a 8-frame check of the GPU path (the variant map_ref_dev.py was only tested on the CPU logic); a failure stops the chain at once.
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
MAPREF_DEVICE=cuda OMP_NUM_THREADS=2 $PY -u temp/soa_gap/map_ref_dev.py runs/t10_dual_s0 temp/soa_gap/map_ref_check8.json 8 > temp/soa_gap/map_ref_check8.log 2>&1
rc=$?
echo "check8 exit $rc"
[ $rc -eq 0 ] || { tail -5 temp/soa_gap/map_ref_check8.log; exit 1; }
for r in t10_dual_s0 t8_ctl_s0 t8_bevfeat_s0 t10_dual_s1; do
  [ -f temp/soa_gap/map_ref_${r}_400.json ] && continue
  MAPREF_DEVICE=cuda OMP_NUM_THREADS=2 $PY -u temp/soa_gap/map_ref_dev.py runs/$r temp/soa_gap/map_ref_${r}_400.json 400 > temp/soa_gap/map_ref_${r}_400.log 2>&1
  echo "$r exit $?"
done
echo "maps done"
