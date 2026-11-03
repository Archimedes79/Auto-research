#!/bin/bash
# The like-for-like map scores of the case for the 6000-step run: the same 400 frames and the same call as map_ref_<run>_400.json of the dual chain, so the shares of GaussianCaR can be compared one to one.
# Starts only behind the chained evaluation (start_maps_long.sh --after); needs about 3 GB of GPU memory and 6 - 8 GB of host memory. An 8-frame check of the new model class (object_moving head) runs first; a failure stops the chain at once.
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
r=t10_dual_long_s0
MAPREF_DEVICE=cuda OMP_NUM_THREADS=2 $PY -u temp/soa_gap/map_ref_dev.py runs/$r temp/soa_gap/map_ref_${r}_check8.json 8 > temp/soa_gap/map_ref_${r}_check8.log 2>&1
rc=$?
echo "check8 exit $rc"
[ $rc -eq 0 ] || { tail -5 temp/soa_gap/map_ref_${r}_check8.log; exit 1; }
MAPREF_DEVICE=cuda OMP_NUM_THREADS=2 $PY -u temp/soa_gap/map_ref_dev.py runs/$r temp/soa_gap/map_ref_${r}_400.json 400 > temp/soa_gap/map_ref_${r}_400.log 2>&1
echo "$r exit $?"
