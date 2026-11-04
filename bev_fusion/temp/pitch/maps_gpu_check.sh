#!/bin/bash
# Layer-1 check of the GPU map scoring: t1_r4_soa (the model of the stored CPU run temp/soa_gap/map_ref_r4soa_400.json) scored again on the GPU, 400 frames, then compared number by number.
# Run it through run.py (start_maps_check.sh) with the GPU free of trainings (about 3 GB GPU, 11 GB host memory, about 3 minutes).
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
MAPREF_DEVICE=cuda OMP_NUM_THREADS=2 $PY -u temp/soa_gap/map_ref_dev.py runs/t1_r4_soa temp/soa_gap/map_ref_r4soa_gpu_400.json 400 > temp/soa_gap/map_ref_r4soa_gpu_400.log 2>&1
rc=$?
echo "scoring exit $rc"
[ $rc -eq 0 ] || { tail -5 temp/soa_gap/map_ref_r4soa_gpu_400.log; exit 1; }
python3 temp/pitch/maps_cmp.py temp/soa_gap/map_ref_r4soa_400.json temp/soa_gap/map_ref_r4soa_gpu_400.json
