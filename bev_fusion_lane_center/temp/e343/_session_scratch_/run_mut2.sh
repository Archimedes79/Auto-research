#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 bash tools/env/_py14.sh temp/e343/_session_scratch_/mutate.py 'is_protected roots no casefold' 'precheck no seed check' 'get_dump flagged unchecked' 'get_dump default no sidecar ok' 'score_seed B flag unchecked' 'score_seed ctl ends unchecked' 'score_seed ctl B unchecked' 'get_dump no provenance' 'provenance ignores device' 'provenance ignores code' 'provenance ignores checkpoint' 'provenance ignores run' 'provenance missing sidecar ok' 'is_protected no casefold' 'is_protected off' > temp/e343/_session_scratch_/mutate_r2b.log 2>&1
