#!/bin/bash
# scratch: ms per frame of the three finished systems on an idle GPU (idle guard of the timing tool, patched in temp/t16/timing_patched2.py: own pid, 3 s settle before the second look)
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=/home/ml/miniconda3/envs/gcar/bin/python
for RUN in t16_full_s0 t17_data t18_fix; do
  echo "== timing $RUN $(date +%T)"
  $PY temp/t16/timing_patched2.py runs/$RUN --n 30 2>&1 | tail -n 14 | cut -c1-260
done
echo "== done $(date +%T)"
