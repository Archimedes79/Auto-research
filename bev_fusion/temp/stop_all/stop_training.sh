#!/bin/bash
# clean stop of the running training r2_rad_raster64_s0 at the user's request (trainsuite stop protocol: the trainer sees CONTROL at the next step boundary, saves a full checkpoint, writes STOPPED, exits 75; run.py then puts the id on HOLD and the --after chain of the radar runs is blocked)
cd "$(dirname "$0")/../../../autoresearch" || exit 1
python3 -m trainsuite --root ../bev_fusion/runs status r2_rad_raster64_s0 2>&1 | tail -6
python3 -m trainsuite --root ../bev_fusion/runs stop r2_rad_raster64_s0 --reason "stopped at the user's request on 2026-11-01 (22:5x): please stop all training activities; resume only on the user's word" 2>&1 | tail -12
