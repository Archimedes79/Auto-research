cd /mnt/c/Auto-research/bev_fusion_lane_center
echo 1000 > /proc/self/oom_score_adj
T="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2"
# E264.mfl2: the val250 ladder 0.40 - 0.65 never reached the union's false length (s3 0.054 - 0.055 vs u23 0.062; s1 0.053 - 0.054 vs u01 0.059),
# so t* = 0.10, the lowest tau of the loop-22 grids (E265.eb), read on all 873 frames
python3 ../autoresearch/tools/run.py --id E264.mfl2 --require-card --split val \
  -- $T bash -c 'bash tools/env/_py14.sh evaluate.py e24_j1rc_s3 --tau 0.1 --tag mfl --compare e24_j1rc_s3 && bash tools/env/_py14.sh evaluate.py e24_j1rc_s1 --tau 0.1 --tag mfl --compare e24_j1rc_s1' > temp/loop22/E264mfl2.launcher.log 2>&1
echo "E264.mfl2: $(tail -n 1 temp/loop22/E264mfl2.launcher.log)"
python3 ../autoresearch/tools/run.py --id E264.mv --require-card --split val \
  -- $T bash -c 'for v in 1.5 1.7 1.9 2.1 2.3 2.5; do bash tools/env/_py14.sh evaluate.py ens_j1rc_23 --subset val250 --asm vtx_r=$v --tag vx$v || exit 1; done' > temp/loop22/E264mv.launcher.log 2>&1
echo "E264.mv: $(tail -n 1 temp/loop22/E264mv.launcher.log)"
