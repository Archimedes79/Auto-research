#!/bin/bash
# t26.ref_maps_base: the base's map layers against GaussianCaR on the claim region (the base was trained on GT v9, scored on v13: --allow-gt-mismatch, written into the json), CPU
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH=/mnt/c/Auto-research OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=""
# a courtesy gate (the Spurmitte chains share this box, a lane run died of ENOMEM at 01:10): start only when MemAvailable is at least 8 GB on 3 looks in a row
ok=0; while [ $ok -lt 3 ]; do a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); if [ "$a" -ge 8192 ]; then ok=$((ok+1)); else ok=0; fi; sleep 20; done
mkdir -p temp/t26/ref_maps_base
nice -n 15 ~/miniconda3/envs/gcar/bin/python evaluation/soa_maps.py temp/t26/base_run temp/t26/ref_maps_base/out.json --device cpu --allow-gt-mismatch
