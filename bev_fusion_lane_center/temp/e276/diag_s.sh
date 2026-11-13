#!/bin/bash
# scratch diagnostic (by hand, never a verdict): what does the failed arm s (phase-3 checkpoint) deliver? CPU, 60 frames, three node thresholds, scored against the sparse GT nodes
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 12 bash tools/env/_py14.sh qgraph_run_eval.py score --run e276_s_s0_p3 --seed 0 --n 60 --exp e276diag --readouts plain@0.5 plain@0.3 plain@0.2 --sparse 0.3 7.5 > temp/e276/diag_s.log 2>&1
touch temp/e276/DIAG_S_DONE
