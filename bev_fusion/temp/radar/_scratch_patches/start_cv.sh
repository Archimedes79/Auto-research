#!/usr/bin/env bash
# start the radar codec cross-validation through the launcher (waits for a GPU slot); run from the bev_fusion folder
cd "$(dirname "$0")/../../.." || exit 1
AR_PROJECT=../bev_fusion nohup python3 ../autoresearch/tools/run.py --id r1.codec_cv --gpu --wait-gpu 720 \
  --hyp "Residual (delta) and physics-derived (phys) coding of range / direction keep more radar detections than the absolute coding, also across scenes" \
  --pred "delta - abs kept/oracle at tolerances x0.25: mean over 5 scene folds >= +5 points, wins >= 4 of 5; phys - delta within 2 points; overfitting excess < 3 points for every codec" \
  --rule "A better than B if the mean paired fold difference of kept/oracle exceeds max(2 points, seed SD) and t > 2.776 (df 4); else TIE (take the simpler codec); overfitting flagged above 3 points excess over the oracle gap" \
  -- ~/miniconda3/envs/gcar/bin/python -u tests/radar_codecs.py 1000 cv 1 > temp/radar/radar_cv.launcher.log 2>&1 &
echo started
