#!/usr/bin/env bash
# start t10_dual_long_s0 (lever 1 of the case, lead's version of card section 3b: the dual chain from scratch at 6000 optimiser steps, accumulation 2, velocity hurdle) under run.py (D23).
# Start it only after (1) the Architecture merge + smoke test, (2) the ar-reviewer pass on the t11 code with no open blocker, (3) the map scoring pitch.maps_lfl has ended and the GPU is free, (4) no veto of Mirko.
# It waits for a GPU slot and for the GPU memory in use to fall below 3000 MB; run it in the foreground (no nohup, the WSL session must stay alive). About 8.4 h on the whole GPU.
cd "$(dirname "$0")/../.." || exit 1
export AR_PROJECT=../bev_fusion
python3 ../autoresearch/tools/run.py --id t10.dual_long_s0 --gpu --gpu-max-mb 3000 --split train \
  --hyp "the dual-feature chain is still under-trained at 3000 steps: at 6000 optimiser steps with accumulation 2 (4x the samples) the lane divider gains at least +0.04 (expectation +0.07, 0.75 x the old features' ladder), and the velocity hurdle learns from scratch" \
  --pred "cold VAL 1000: lane divider@0.5 0.39 (0.355-0.44), walkway 0.37 (0.34-0.41), crossing 0.20 (0.16-0.26), drivable@0 0.775 (0.76-0.795), front mAP2 0.355 (0.33-0.385), vehicle velocity skill on moving cars 0.55 (0.35-0.70), static error 0.25 m/s (0.15-0.35)" \
  --rule "decide.py --band 0.029 on the lane divider against the dual mean 0.3425 (it prints TIE inside the band: read as INCONCLUSIVE for the claim; one seed never KEEPs or CUTs): gain >= +0.04 continue provisionally (twin follows), below that INCONCLUSIVE; no claim dropped on one run; 'SOA reachable on this hardware' is dropped only if the equal-budget arm without object_moving also gains < +0.02; hurdle holds if skill >= 0.50 and static <= 0.30; guards front mAP2 >= 0.307 (with the paired interval), walkway and crossing same sign; kill: mean group.lines over steps 820-1000 > 1.0 = stop and look; GT signature 0.5_4_15efbde2d1 (experiments/t10_dual_features/experiment.md sections 3b and 3c rev 2)" \
  -- bash temp/train/run_ar.sh training/configs/t10_dual_long_s0.json > runs/t10_dual_long_s0.launcher.log 2>&1
echo "launcher exit $?"
