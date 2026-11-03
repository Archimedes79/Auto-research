#!/bin/bash
# scratch: ms per frame of the three finished systems on an idle GPU (idle guard of the timing tool; runs as a job of the GPU line behind t18.fix.eval)
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=/home/ml/miniconda3/envs/gcar/bin/python
for RUN in t16_full_s0 t17_data t18_fix; do
  echo "== timing $RUN $(date +%T)"
  $PY evaluation/bev_suite.py runs/$RUN timing --n 30 2>&1 | tail -n 8 | cut -c1-240
done
echo "== done $(date +%T)"
