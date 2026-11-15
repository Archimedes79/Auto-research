#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
{ echo "# radar_sweep_density.py, 2026-11-13 (D91): python studies/radar_sweep_density.py (defaults: nuScenes mini, newest sweep = age <= 0.09 s, margin 1 m)"; nice -n 15 ~/miniconda3/envs/gcar/bin/python studies/radar_sweep_density.py 2>&1 | grep -v '^WARNING'; } > records/radar_sweep_density_2026-11-13.txt
wc -l records/radar_sweep_density_2026-11-13.txt; sed -n 2,3p records/radar_sweep_density_2026-11-13.txt | cut -c1-200
