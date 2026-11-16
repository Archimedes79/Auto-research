#!/bin/bash
# critic finding of E320 (2026-11-12 23:45): score the E312 checkpoints with the matched-wiring read-outs of E311 (CPU, 1 thread, nice; new card file names, nothing is overwritten)
#   --snap --snap-nogeom : keypoint geometry + ONE nearest piece per end (the wiring of the snap without its geometry)  -> ends_card_snap_nogeom_thr0.1.json
#   --snap --snap-all    : snapped geometry + EVERY piece in reach (the proximity wiring of the attached read-out)        -> ends_card_snap_all_thr0.1.json
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
L=temp/jr/score_e312_wiring.log
echo "start $(date)" >> $L
for S in 0 1; do
  for OPT in snap-nogeom snap-all; do
    nice -n 19 bash tools/env/_py14.sh jends_eval.py score --run e312_p_s$S --seed $S --snap --$OPT --thr 0.1 >> temp/jr/score_e312_${OPT}_s$S.log 2>&1
    echo "$OPT s$S exit $? $(date)" >> $L
  done
done
echo "done $(date)" >> $L
touch temp/jr/score_e312_wiring.done
