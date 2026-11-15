#!/usr/bin/env bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
exec nice -n 15 ~/miniconda3/envs/gcar/bin/python temp/t26_b/with_autogt_alias.py evaluation/soa_maps.py runs/t16_full_s0 temp/t26_b/soa_repro_n40.json --n 40 \
  --frames-from temp/t16/soa/soa_maps_v8_t16_full_s0.json --gt-signature 6a7aa7864a --device cpu --check temp/t16/soa/soa_maps_v8_t16_full_s0.json
